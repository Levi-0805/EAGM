"""CPU MuJoCo execution-memory diagnostic for the local OpenArm v2 asset.

This is a simulation-only, fixed-base/fixed-finger variant, not a validated
hardware controller or the previous IsaacLab environment. Contact is disabled
by default to isolate target accumulation, filtering, and command delay.

Joint/action order: left joint1..7, then right joint1..7. Actions are joint
target increments in rad, clipped before accumulation. Observe is q14, qd14.
Control state layout (57): target14, filter14, queue_oldest14, queue_latest14,
delay_frames1. Both queue slots are always updated, including for delay zero.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def default_model_path() -> Path:
    # Explicit override first, then the original workspace-relative lookup, then
    # the recorded local asset location.  The fallback only locates the same
    # MJCF; it does not alter the simulation or any measured result.
    override = os.environ.get("OPENARM_MODEL_PATH")
    if override and Path(override).is_file():
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "openarm_mujoco" / "v2" / "openarm_bimanual.xml"
        if candidate.is_file():
            return candidate
    local = Path("E:/OPENARM/third_party/openarm_mujoco/v2/openarm_bimanual.xml")
    if local.is_file():
        return local
    raise FileNotFoundError("Cannot locate the local third_party/openarm_mujoco/v2 asset")


class ArmEnv:
    """Headless 14-DoF environment with an explicit, deterministic host stack.

    step(action) -> ndarray(28). No reward is hidden inside this environment.
    reset(seed, delay=None, pose=None, jitter=0.01) -> ndarray(28).
    snapshot()/restore() clone mjSTATE_INTEGRATION plus all Python controller
    and RNG state; restore intentionally does not run mj_forward, since doing
    so could alter integrator warm-start state. The next mj_step recomputes
    derived quantities. Call mj_forward explicitly only for visualization.
    """

    action_dim = 14
    observation_dim = 28
    control_state_dim = 57

    def __init__(
        self,
        delay: int = 0,
        alpha: float = 0.5,
        control_dt: float = 1.0 / 30.0,
        max_increment: float = 0.03,
        model_path: str | Path | None = None,
        enable_contact: bool = False,
    ):
        self._check_delay(delay)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if control_dt <= 0 or max_increment <= 0:
            raise ValueError("control_dt and max_increment must be positive")
        self.delay = int(delay)
        self.alpha = float(alpha)
        self.max_increment = float(max_increment)
        self.model_path = Path(model_path or default_model_path()).resolve()
        self._source_bytes = self.model_path.read_bytes()
        tree = ET.fromstring(self._source_bytes)
        compiler = tree.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(tree, "compiler")
        mesh_base = (self.model_path.parent / compiler.get("meshdir", "")).resolve()
        compiler.set("meshdir", str(mesh_base))

        # Remove the finger DoFs rather than silently omitting their dynamic
        # state from x. The bodies, masses, inertias, and meshes remain, welded
        # in their original zero-angle configurations.
        self.removed_finger_joints = []
        world = tree.find("worldbody")
        if world is None:
            raise ValueError("Expected worldbody in the OpenArm MJCF")
        for parent in world.iter():
            for child in list(parent):
                if child.tag == "joint" and "finger_joint" in child.get("name", ""):
                    self.removed_finger_joints.append(child.get("name"))
                    parent.remove(child)
        equality = tree.find("equality")
        if equality is not None:
            for child in list(equality):
                if any("finger" in child.get(k, "") for k in ("name", "joint1", "joint2")):
                    equality.remove(child)
        actuators = tree.find("actuator")
        if actuators is None:
            raise ValueError("Expected actuator block")
        for child in list(actuators):
            if "finger" in child.get("joint", ""):
                actuators.remove(child)
        self.effective_xml = ET.tostring(tree, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.effective_xml)
        if not enable_contact:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.enable_contact = bool(enable_contact)
        self.data = mujoco.MjData(self.model)
        self.total_control_steps = 0
        self.accounting_phase = 'setup'
        self.control_steps_by_phase = {}
        self.joint_names = [f"openarm_{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
        self.actuator_names = [f"{side}_joint{i}_ctrl" for side in ("left", "right") for i in range(1, 8)]
        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names])
        self.actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.actuator_names])
        if (self.joint_ids < 0).any() or (self.actuator_ids < 0).any():
            raise ValueError("Unexpected joint or actuator naming")
        if (self.model.nq, self.model.nv, self.model.nu) != (14, 14, 14):
            raise ValueError("Fixed-finger variant must contain exactly 14 q, v, and actuators")
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids].copy()
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids].copy()
        if not np.array_equal(self.model.actuator_trnid[self.actuator_ids, 0], self.joint_ids):
            raise ValueError("Actuator-to-joint mapping mismatch")
        ranges = self.model.jnt_range[self.joint_ids]
        ctrl_ranges = self.model.actuator_ctrlrange[self.actuator_ids]
        self.lower = np.maximum(ranges[:, 0], ctrl_ranges[:, 0]) + 0.05
        self.upper = np.minimum(ranges[:, 1], ctrl_ranges[:, 1]) - 0.05
        self.home = np.array([0, -0.25, 0, 0.5, 0, 0, 0, 0, 0.25, 0, 0.5, 0, 0, 0], dtype=np.float64)
        self.physics_dt = float(self.model.opt.timestep)
        self.requested_control_dt = float(control_dt)
        self.substeps = max(1, int(round(control_dt / self.physics_dt)))
        self.control_dt = self.substeps * self.physics_dt
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_size = mujoco.mj_stateSize(self.model, self._spec)
        self.rng = np.random.default_rng(0)
        self.frame = 0
        self.qtarget = self.home.copy()
        self.filtered = self.home.copy()
        self.queue = np.repeat(self.home[None], 2, axis=0)
        self.last_command = self.home.copy()
        self.last_action = np.zeros(14)
        self.last_saturation_count = 0
        self.reset(0)

    @staticmethod
    def _check_delay(delay):
        if isinstance(delay, bool) or int(delay) != delay or int(delay) not in (0, 1, 2):
            raise ValueError("delay must be 0, 1, or 2 control frames")

    def reset(self, seed: int | None = None, delay: int | None = None,
              pose=None, jitter: float = 0.01):
        if delay is not None:
            self._check_delay(delay)
            self.delay = int(delay)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if jitter < 0:
            raise ValueError("jitter must be nonnegative")
        q = self.home.copy() if pose is None else np.asarray(pose, dtype=np.float64).copy()
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("pose must be a finite 14-vector")
        q = np.clip(q + self.rng.uniform(-jitter, jitter, 14), self.lower, self.upper)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = q
        self.data.qvel[self.qvel_ids] = 0.0
        self.data.ctrl[self.actuator_ids] = q
        self.qtarget = q.copy()
        self.filtered = q.copy()
        self.queue = np.repeat(q[None], 2, axis=0)
        self.last_command = q.copy()
        self.last_action = np.zeros(14)
        self.last_saturation_count = 0
        self.frame = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self):
        return np.concatenate((self.data.qpos[self.qpos_ids], self.data.qvel[self.qvel_ids])).copy()

    def control_state(self):
        return np.concatenate((self.qtarget, self.filtered, self.queue.reshape(-1), [float(self.delay)]))

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 14-vector of rad increments")
        applied = np.clip(action, -self.max_increment, self.max_increment)
        unclipped_target = self.qtarget + applied
        new_target = np.clip(unclipped_target, self.lower, self.upper)
        new_filter = self.alpha * new_target + (1.0 - self.alpha) * self.filtered
        # queue[0] = f_(t-2), queue[1] = f_(t-1), new_filter = f_t.
        command = (new_filter if self.delay == 0 else self.queue[2 - self.delay]).copy()
        self.queue = np.stack((self.queue[1], new_filter))
        self.qtarget = new_target
        self.filtered = new_filter
        self.last_action = applied
        self.last_command = command
        self.last_saturation_count = int(np.count_nonzero(action != applied) + np.count_nonzero(unclipped_target != new_target))
        self.data.ctrl[self.actuator_ids] = command
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self.total_control_steps += 1
        self.control_steps_by_phase[self.accounting_phase] = self.control_steps_by_phase.get(self.accounting_phase, 0) + 1
        self.frame += 1
        obs = self.observe()
        if not np.isfinite(obs).all():
            raise FloatingPointError("Nonfinite MuJoCo state")
        return obs

    def snapshot(self):
        state = np.empty(self._state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, self._spec)
        return {"integration": state, "qtarget": self.qtarget.copy(),
                "filtered": self.filtered.copy(), "queue": self.queue.copy(),
                "last_action": self.last_action.copy(), "last_command": self.last_command.copy(),
                "delay": self.delay, "frame": self.frame,
                "saturation_count": self.last_saturation_count,
                "rng": deepcopy(self.rng.bit_generator.state),
                "alpha": self.alpha, "max_increment": self.max_increment,
                "model_sha256": sha256(self.effective_xml.encode()).hexdigest(),
                "contact_enabled": self.enable_contact}

    def restore(self, snapshot):
        if snapshot["model_sha256"] != sha256(self.effective_xml.encode()).hexdigest():
            raise ValueError("Snapshot belongs to a different effective model")
        if snapshot["alpha"] != self.alpha or snapshot["max_increment"] != self.max_increment:
            raise ValueError("Snapshot control parameters differ")
        if snapshot["contact_enabled"] != self.enable_contact:
            raise ValueError("Snapshot contact settings differ")
        mujoco.mj_setState(self.model, self.data, np.array(snapshot["integration"], copy=True), self._spec)
        for name in ("qtarget", "filtered", "queue", "last_action", "last_command"):
            setattr(self, name, np.array(snapshot[name], copy=True))
        self.delay = int(snapshot["delay"])
        self.frame = int(snapshot["frame"])
        self.last_saturation_count = int(snapshot["saturation_count"])
        self.rng.bit_generator.state = deepcopy(snapshot["rng"])
        return self.observe()

    def metadata(self):
        return {"engine": "MuJoCo", "engine_version": mujoco.__version__,
                "model_source": str(self.model_path),
                "model_source_sha256": sha256(self._source_bytes).hexdigest(),
                "effective_xml_sha256": sha256(self.effective_xml.encode()).hexdigest(),
                "nq": self.model.nq, "nv": self.model.nv, "nu": self.model.nu,
                "joint_names": self.joint_names, "actuator_names": self.actuator_names,
                "base": "fixed", "finger_handling": "four finger joints removed; finger masses/geometries welded at zero angle",
                "removed_finger_joints": self.removed_finger_joints,
                "contact_enabled": self.enable_contact,
                "requested_control_dt": self.requested_control_dt,
                "physics_dt": self.physics_dt, "substeps": self.substeps,
                "actual_control_dt": self.control_dt, "actual_control_hz": 1 / self.control_dt,
                "filter_alpha": self.alpha, "delay_frames": self.delay,
                "max_increment_rad": self.max_increment,
                "target_lower_rad": self.lower.tolist(), "target_upper_rad": self.upper.tolist(),
                "home_rad": self.home.tolist(),
                "control_state_layout": "qtarget[0:14],filter[14:28],queue_oldest[28:42],queue_latest[42:56],delay[56]",
                "snapshot_state": "mjSTATE_INTEGRATION and Python control/RNG state"}


def smoke_check():
    """Check queue timing, full-state replay and a small CPU simulation run."""
    reports = []
    for delay in (0, 1, 2):
        env = ArmEnv(delay=delay)
        env.reset(seed=100, jitter=0)
        initial = env.qtarget.copy()
        pulse = np.zeros(14); pulse[0] = 0.01
        commands = []
        for i in range(4):
            env.step(pulse if i == 0 else np.zeros(14))
            commands.append(env.last_command[0] - initial[0])
        expected_all = np.array([0.005, 0.0075, 0.00875, 0.009375])
        expected = np.concatenate((np.zeros(delay), expected_all))[:4]
        np.testing.assert_allclose(commands, expected, rtol=0, atol=1e-12)
        rng = np.random.default_rng(17)
        actions = rng.uniform(-0.002, 0.002, (16, 14))
        snap = env.snapshot()
        path_a = np.stack([env.step(a) for a in actions])
        end_a = env.snapshot()["integration"]
        env.restore(snap)
        path_b = np.stack([env.step(a) for a in actions])
        end_b = env.snapshot()["integration"]
        np.testing.assert_array_equal(path_a, path_b)
        np.testing.assert_array_equal(end_a, end_b)
        before = time.perf_counter()
        for _ in range(300):
            env.step(np.zeros(14))
        elapsed = time.perf_counter() - before
        reports.append({"delay": delay, "pulse_commands": commands,
                        "replay_max_abs_error": float(np.max(np.abs(path_a-path_b))),
                        "300_control_steps_wall_seconds": elapsed,
                        "finite": bool(np.isfinite(env.observe()).all()),
                        "max_joint_speed_rad_s_after_hold": float(np.max(np.abs(env.observe()[14:])))})
    return {"metadata": env.metadata(), "checks": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke_check() if args.smoke else ArmEnv().metadata(), ensure_ascii=False, indent=2))
