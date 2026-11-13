"""Controller B: partially unknown execution mapping on top of CompatibleArmEnv.

Known to the model (c_obs): target, filter, delay scalar (and optionally
declared config summary). Hidden / only for privileged labels: true rate-limit
state, extra lag buffer, episode gain, true executed command after unknown stack.

This is intentionally NOT tuned to make EAGM win. Mechanisms are ordinary
host-side execution effects: gain error, extra lag, rate limits, mild smoothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .compatible_arm_env import CompatibleArmEnv


@dataclass(frozen=True)
class ControllerBConfig:
    gain: float = 1.0
    extra_lag: int = 0  # additional FIFO frames after nominal delay (0..2)
    rate_limit: float = 0.05  # max |Δu| per control step (rad)
    smooth_beta: float = 0.0  # extra 1st-order smooth on command; 0 disables
    # If True, reshuffle a subset of params each episode reset.
    episode_randomize: bool = False
    gain_range: tuple[float, float] = (0.92, 1.08)
    extra_lag_choices: tuple[int, ...] = (0, 1, 2)
    rate_limit_range: tuple[float, float] = (0.03, 0.08)
    smooth_beta_range: tuple[float, float] = (0.0, 0.35)


class PartialUnknownArmEnv(CompatibleArmEnv):
    """Controller A physics + additional unknown execution dynamics (B)."""

    def __init__(self, config: ControllerBConfig | None = None, **kwargs):
        # Must set before super().__init__ because ArmEnv.reset() runs there.
        self.b_config = config or ControllerBConfig()
        self._active = dict(
            gain=float(self.b_config.gain),
            extra_lag=int(self.b_config.extra_lag),
            rate_limit=float(self.b_config.rate_limit),
            smooth_beta=float(self.b_config.smooth_beta),
        )
        self._extra_queue: list[np.ndarray] = []
        super().__init__(**kwargs)
        self._u_smooth = self.home.copy()
        self._u_prev = self.home.copy()
        self.last_true_command = self.home.copy()
        self.last_nominal_command = self.home.copy()

    def observed_control_state(self) -> np.ndarray:
        """Partial interface: target, filter, delay — hide FIFO / B internals."""
        return np.concatenate(
            (self.qtarget, self.filtered, [float(self.delay)])
        ).astype(np.float64)

    def privileged_labels(self) -> dict[str, Any]:
        return {
            "u_true": self.last_true_command.copy(),
            "u_nominal": self.last_nominal_command.copy(),
            "b_active": dict(self._active),
            "full_c": self.control_state().copy(),
            # Train-only execution-state targets for Stage 6 ExecState (never deploy inputs)
            "exec_state": self.privileged_exec_state(),
        }

    def privileged_exec_state(self) -> dict[str, Any]:
        """Hidden Controller-B execution state — TRAIN LABELS ONLY.

        Deployable models must NOT consume these fields as inputs.
        """
        a = self._active
        queue = [q.copy() for q in self._extra_queue]
        # Fixed-width lag buffer pad for supervised heads (max lag considered = 3)
        max_lag = 3
        lag_pad = np.zeros((max_lag, 14), dtype=np.float64)
        for i, q in enumerate(queue[-max_lag:]):
            lag_pad[i] = q
        return {
            "gain": float(a["gain"]),
            "extra_lag": int(a["extra_lag"]),
            "rate_limit": float(a["rate_limit"]),
            "smooth_beta": float(a["smooth_beta"]),
            "u_prev": self._u_prev.copy(),
            "u_smooth": self._u_smooth.copy(),
            "extra_queue": queue,
            "lag_pad": lag_pad,
            "vector": np.concatenate(
                [
                    [a["gain"], a["extra_lag"], a["rate_limit"], a["smooth_beta"]],
                    self._u_prev.ravel(),
                    self._u_smooth.ravel(),
                    lag_pad.ravel(),
                ]
            ).astype(np.float64),
        }

    def known_config_vector(self) -> np.ndarray:
        """ORACLE/debug only: exposes true B parameters.

        Deployable EAGM models must NOT consume this vector — feeding it would
        leak the 'unknown' execution map and invalidate Controller-B claims.
        """
        a = self._active
        return np.asarray(
            [a["gain"], a["extra_lag"], a["rate_limit"], a["smooth_beta"]],
            dtype=np.float64,
        )

    def set_active_config(
        self,
        *,
        gain: float,
        extra_lag: int,
        rate_limit: float,
        smooth_beta: float,
    ) -> None:
        """Force the next/current B parameters (used for frozen OOD test configs)."""
        self._active = dict(
            gain=float(gain),
            extra_lag=int(extra_lag),
            rate_limit=float(rate_limit),
            smooth_beta=float(smooth_beta),
        )
        self._extra_queue = []
        self._u_smooth = self.qtarget.copy() if hasattr(self, "qtarget") else None
        self._u_prev = self.qtarget.copy() if hasattr(self, "qtarget") else None

    def reset(self, seed: int | None = None, delay: int | None = None, pose=None, jitter: float = 0.01):
        obs = super().reset(seed=seed, delay=delay, pose=pose, jitter=jitter)
        if not hasattr(self, "b_config"):
            return obs
        if self.b_config.episode_randomize:
            rng = self.rng
            lo_g, hi_g = self.b_config.gain_range
            lo_r, hi_r = self.b_config.rate_limit_range
            lo_b, hi_b = self.b_config.smooth_beta_range
            self._active = dict(
                gain=float(rng.uniform(lo_g, hi_g)),
                extra_lag=int(rng.choice(self.b_config.extra_lag_choices)),
                rate_limit=float(rng.uniform(lo_r, hi_r)),
                smooth_beta=float(rng.uniform(lo_b, hi_b)),
            )
        else:
            self._active = dict(
                gain=float(self.b_config.gain),
                extra_lag=int(self.b_config.extra_lag),
                rate_limit=float(self.b_config.rate_limit),
                smooth_beta=float(self.b_config.smooth_beta),
            )
        self._extra_queue = []
        self._u_smooth = self.qtarget.copy()
        self._u_prev = self.qtarget.copy()
        self.last_true_command = self.qtarget.copy()
        self.last_nominal_command = self.qtarget.copy()
        return obs

    def step(self, action):
        # Run Controller A stack up to nominal command, then apply unknown map.
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 14-vector of rad increments")
        applied = np.clip(action, -self.max_increment, self.max_increment)
        unclipped_target = self.qtarget + applied
        new_target = np.clip(unclipped_target, self.lower, self.upper)
        new_filter = self.alpha * new_target + (1.0 - self.alpha) * self.filtered
        nominal = (
            new_filter if self.delay == 0 else self.queue[2 - self.delay]
        ).copy()
        self.queue = np.stack((self.queue[1], new_filter))
        self.qtarget = new_target
        self.filtered = new_filter
        self.last_action = applied
        self.last_nominal_command = nominal.copy()

        u = self._apply_unknown(nominal)
        self.last_command = u.copy()
        self.last_true_command = u.copy()
        self.last_saturation_count = int(
            np.count_nonzero(action != applied)
            + np.count_nonzero(unclipped_target != new_target)
        )
        self.data.ctrl[self.actuator_ids] = u
        import mujoco

        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self.total_control_steps += 1
        self.control_steps_by_phase[self.accounting_phase] = (
            self.control_steps_by_phase.get(self.accounting_phase, 0) + 1
        )
        self.frame += 1
        obs = self.observe()
        if not np.isfinite(obs).all():
            raise FloatingPointError("Nonfinite MuJoCo state")
        return obs

    def _apply_unknown(self, nominal: np.ndarray) -> np.ndarray:
        a = self._active
        u = nominal * a["gain"]
        # Extra lag FIFO (independent of declared delay).
        lag = int(a["extra_lag"])
        if lag <= 0:
            delayed = u
            self._extra_queue = []
        else:
            self._extra_queue.append(u.copy())
            if len(self._extra_queue) <= lag:
                delayed = self._u_prev.copy()
            else:
                delayed = self._extra_queue.pop(0)
        # Rate limit.
        rl = float(a["rate_limit"])
        delta = np.clip(delayed - self._u_prev, -rl, rl)
        limited = self._u_prev + delta
        # Extra smoothing.
        beta = float(a["smooth_beta"])
        if beta > 0:
            self._u_smooth = beta * limited + (1.0 - beta) * self._u_smooth
            out = self._u_smooth.copy()
        else:
            out = limited
        out = np.clip(out, self.lower, self.upper)
        self._u_prev = out.copy()
        return out

    def snapshot(self):
        snap = super().snapshot()
        snap = dict(snap)
        snap["controller_b"] = {
            "active": dict(self._active),
            "extra_queue": [q.copy() for q in self._extra_queue],
            "u_smooth": self._u_smooth.copy(),
            "u_prev": self._u_prev.copy(),
            "last_true_command": self.last_true_command.copy(),
            "last_nominal_command": self.last_nominal_command.copy(),
        }
        return snap

    def restore(self, snapshot):
        obs = super().restore(snapshot)
        b = snapshot.get("controller_b")
        if b is None:
            # Legacy Controller-A snapshot: clear B internals.
            self._extra_queue = []
            self._u_smooth = self.qtarget.copy()
            self._u_prev = self.qtarget.copy()
            self.last_true_command = self.qtarget.copy()
            self.last_nominal_command = self.qtarget.copy()
            return obs
        self._active = dict(b["active"])
        self._extra_queue = [np.array(q, copy=True) for q in b["extra_queue"]]
        self._u_smooth = np.array(b["u_smooth"], copy=True)
        self._u_prev = np.array(b["u_prev"], copy=True)
        self.last_true_command = np.array(b["last_true_command"], copy=True)
        self.last_nominal_command = np.array(b["last_nominal_command"], copy=True)
        return obs

    def metadata(self):
        meta = super().metadata()
        meta = dict(meta)
        meta["controller"] = "B_partial_unknown"
        meta["controller_b_config"] = asdict(self.b_config)
        meta["controller_b_active"] = dict(self._active)
        meta["obs_control_state"] = "qtarget[0:14],filter[14:28],delay[28]"
        meta["hidden"] = "FIFO queue, extra lag buffer, rate-limit state, smooth state, true u"
        return meta
