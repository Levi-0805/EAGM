"""Smoke tests for CompatibleArmEnv + Controller B."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.compatible_arm_env import CompatibleArmEnv
from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv


def test_compatible_restore_and_step():
    import torch

    env = CompatibleArmEnv()
    starts = torch.load(
        ROOT / "vendor/legacy_sim/runs/pilot_20261101/starts.pt", weights_only=False
    )
    s = [x for x in starts["test"] if x["delay"] == 2][0]
    obs = env.restore(s["snapshot"])
    assert obs.shape == (28,)
    obs2 = env.step(np.zeros(14))
    assert obs2.shape == (28,)


def test_controller_b_changes_command():
    a = CompatibleArmEnv()
    b = PartialUnknownArmEnv(
        ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.04, smooth_beta=0.25)
    )
    a.reset(seed=0, delay=2, jitter=0.0)
    b.reset(seed=0, delay=2, jitter=0.0)
    pulse = np.zeros(14)
    pulse[0] = 0.02
    diffs = []
    for i in range(6):
        act = pulse if i == 0 else np.zeros(14)
        a.step(act)
        b.step(act)
        diffs.append(np.linalg.norm(a.last_command - b.last_command))
    assert max(diffs) > 1e-6
    assert b.observed_control_state().shape == (29,)
    assert "u_true" in b.privileged_labels()


if __name__ == "__main__":
    test_compatible_restore_and_step()
    test_controller_b_changes_command()
    print("ok")
