"""ArmEnv that can restore pilot snapshots on this Linux host.

Pilot snapshots hash a Windows effective XML (absolute E:\\ meshdir). The
original Windows OpenArm source XML is not on this machine. We freeze the
archived pilot effective MJCF, rewrite meshdir to local assets, construct the
stock ArmEnv from that file, and accept the pilot model_sha256 on restore
after structural checks.
"""
from __future__ import annotations

import re
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np

EAGM_ROOT = Path(__file__).resolve().parents[1]
LEGACY = (EAGM_ROOT / "vendor" / "legacy_sim").resolve()
sys.path.insert(0, str(LEGACY))

from sim_env import ArmEnv  # noqa: E402

PILOT_EFFECTIVE = LEGACY / "runs" / "pilot_20261101" / "effective_model.xml"
LOCAL_MESHDIR = Path(
    "/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/assets"
).resolve()
PILOT_MODEL_SHA256 = (
    "063f1dae21b51c206d63307cddf72597d6c99a4ac2d0dd70ade42ca39b6ea83e"
)
LOCAL_EFFECTIVE = EAGM_ROOT / "assets" / "pilot_effective_model.linux.xml"


def materialize_pilot_effective_xml(dest: Path = LOCAL_EFFECTIVE) -> Path:
    raw = PILOT_EFFECTIVE.read_text(encoding="utf-8")
    if not LOCAL_MESHDIR.is_dir():
        raise FileNotFoundError(f"Missing local meshdir: {LOCAL_MESHDIR}")
    text = re.sub(r'meshdir="[^"]*"', f'meshdir="{LOCAL_MESHDIR}"', raw, count=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


class CompatibleArmEnv(ArmEnv):
    def __init__(self, **kwargs):
        path = materialize_pilot_effective_xml()
        kwargs = dict(kwargs)
        kwargs["model_path"] = path
        super().__init__(**kwargs)
        # Sanity: same integration width as pilot starts.
        if self._state_size != 197:
            raise RuntimeError(f"Unexpected state size {self._state_size}")

    def restore(self, snapshot):
        snap_sha = snapshot["model_sha256"]
        local_sha = sha256(self.effective_xml.encode()).hexdigest()
        if snap_sha not in (local_sha, PILOT_MODEL_SHA256):
            raise ValueError(
                "Snapshot belongs to a different effective model "
                f"(snap={snap_sha[:12]} local={local_sha[:12]})"
            )
        if snapshot["alpha"] != self.alpha or snapshot["max_increment"] != self.max_increment:
            raise ValueError("Snapshot control parameters differ")
        if snapshot["contact_enabled"] != self.enable_contact:
            raise ValueError("Snapshot contact settings differ")
        integ = np.asarray(snapshot["integration"], dtype=np.float64)
        if integ.shape != (self._state_size,):
            raise ValueError(
                f"Integration state width mismatch: {integ.shape} vs {self._state_size}"
            )
        mujoco.mj_setState(self.model, self.data, np.array(integ, copy=True), self._spec)
        for name in ("qtarget", "filtered", "queue", "last_action", "last_command"):
            setattr(self, name, np.array(snapshot[name], copy=True))
        self.delay = int(snapshot["delay"])
        self.frame = int(snapshot["frame"])
        self.last_saturation_count = int(snapshot["saturation_count"])
        self.rng.bit_generator.state = deepcopy(snapshot["rng"])
        return self.observe()
