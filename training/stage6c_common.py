"""Shared helpers for Stage 6C Path-Aware Dir + saturation diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from training.gradients import sim_grad_masked_detailed
from training.policy_opt import STAGE6C_PATH_FRACS, policy_optimize

EAGM_ROOT = Path(__file__).resolve().parents[1]
PARENT_COMMIT = "1f25a971bf42d2dbe600e4b62d9870e8e4a870d7"
STAGE6C_SEEDS = [15, 16, 17, 18, 19]
POLICY_LRS = [0.03, 0.015, 0.0075]
H_IMAG = 8
H_EVAL = 16
POLICY_UPDATES = 50


def config_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def count_fd_control_steps(labels: Sequence[dict]) -> int:
    return int(sum(int(lab.get("control_steps", 0)) for lab in labels))


def match_label_budget(labels: Sequence[dict], target_n: int) -> list[dict]:
    """Trim or cycle labels so len == target_n (same FD query class if steps equal)."""
    labs = list(labels)
    if not labs:
        return []
    if len(labs) >= target_n:
        return labs[:target_n]
    out = list(labs)
    i = 0
    while len(out) < target_n:
        out.append(labs[i % len(labs)])
        i += 1
    return out


def build_labels_at_thetas(
    env,
    starts: list[dict],
    thetas: Sequence[np.ndarray],
    *,
    horizon: int = H_IMAG,
    feasible: bool = True,
    eps: float = 0.05,
    scales=(0.5, 1.0, 2.0),
) -> tuple[list[dict], int]:
    """FD labels; cycles thetas over starts. Records per-label control_steps + FD meta."""
    labels: list[dict] = []
    total = 0
    thetas = [np.asarray(t, dtype=np.float64).reshape(28) for t in thetas]
    for i, s in enumerate(starts):
        th = thetas[i % len(thetas)]
        g, mask, st, meta = sim_grad_masked_detailed(
            env, s, th, horizon=horizon, feasible=feasible, eps=eps, scales=scales
        )
        total += st
        s0 = (
            float(np.median(np.abs(g[mask > 0.5])))
            if (mask > 0.5).any()
            else float(np.median(np.abs(g))) or 1.0
        )
        labels.append(
            dict(
                start=s,
                theta=th.copy(),
                g=g,
                mask=mask,
                s0=s0 or 1.0,
                control_steps=int(st),
                fd_meta={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()},
            )
        )
    return labels, total


def random_probe_thetas(n: int, seed: int, sigma: float = 0.3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.zeros(28)] + [rng.normal(0, sigma, 28).clip(-1, 1) for _ in range(n - 1)]


def path_probe_thetas_from_pilot(
    model,
    train_pol_starts: list[dict],
    *,
    use_history: bool,
    is_mc: bool = False,
    updates: int = POLICY_UPDATES,
    lr: float = 0.03,
    seed: int = 0,
    fracs: Sequence[float] = STAGE6C_PATH_FRACS,
) -> tuple[list[np.ndarray], dict]:
    """Freeze WM; run pilot policy opt on train starts; return θ at path fractions."""
    pol = policy_optimize(
        model,
        train_pol_starts,
        use_history=use_history,
        is_mc=is_mc,
        updates=updates,
        lr=lr,
        seed=seed,
        checkpoint_fracs=fracs,
    )
    thetas = []
    for f in fracs:
        step = max(0, min(updates, int(round(f * updates))))
        key = f"u{step}"
        if key in pol["checkpoints"]:
            thetas.append(np.asarray(pol["checkpoints"][key], dtype=np.float64))
        else:
            # fall back to named
            named = f"theta_{int(round(f * 100))}"
            thetas.append(np.asarray(pol["checkpoints"].get(named, pol["theta"]), dtype=np.float64))
    return thetas, pol


def load_eagm_checkpoint(kind: str, path: Path, stats: dict, hist_len: int = 8):
    from models.eagm import EAGMBase, EAGMHistory, McWorld

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "mc":
        m = McWorld(stats)
    elif kind in ("eagm_base", "eagm_dir_matched", "eagm_dir_random", "eagm_dir_path"):
        m = EAGMBase(stats)
    elif kind in ("eagm_hist", "eagm_hist_dir_matched"):
        m = EAGMHistory(stats, hist_len=hist_len)
    else:
        raise ValueError(kind)
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def paired_seed_stats(diffs: Sequence[float], boot_seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(boot_seed)
    d = np.asarray(diffs, dtype=np.float64)
    if len(d) == 0:
        return dict(mean=float("nan"), median=float("nan"), ci95=[float("nan"), float("nan")],
                    n=0, n_positive=0)
    boots = [d[rng.integers(0, len(d), size=len(d))].mean() for _ in range(20000)]
    return dict(
        mean=float(d.mean()),
        std=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        median=float(np.median(d)),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        n=int(len(d)),
        n_positive=int(np.sum(d > 0)),
        per_seed=d.tolist(),
    )


def summarize_policy_curve(pol: dict) -> dict[str, Any]:
    curve = pol.get("curve") or []
    return dict(
        final_theta_param_sat_frac=pol.get("final_theta_param_sat_frac"),
        mean_rho_proj=pol.get("mean_rho_proj"),
        final_rho_proj=pol.get("final_rho_proj"),
        mean_outward_pressure=pol.get("mean_outward_pressure"),
        mean_theta_param_sat_frac=pol.get("mean_theta_param_sat_frac"),
        first_hit_mean=pol.get("first_hit_mean"),
        sat_at_updates={
            str(c["update"]): c["theta_param_sat_frac"] for c in curve if c["update"] % 10 == 0 or c["update"] == 1
        },
        rho_at_updates={
            str(c["update"]): c["rho_proj"] for c in curve if c["update"] % 10 == 0 or c["update"] == 1
        },
        lr=pol.get("lr"),
        updates=pol.get("updates"),
    )
