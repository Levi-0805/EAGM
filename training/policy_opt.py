"""Policy optimization with θ traces, saturation, and paired physical evaluation."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

from training.eval_metrics import (
    aggregate_per_config_paired,
    fail_rate_vs_theta0,
    theta_param_sat_frac,
)
from training.rollout import imagined_eagm, imagined_mc, physical_return_detailed

DTYPE = torch.float32

# Default path fractions for Stage-6 diagnostics / path-aware Dir
DEFAULT_CHECKPOINT_FRACS = (0.0, 0.25, 0.50, 0.75, 1.0)


def policy_optimize(
    model: nn.Module,
    starts: list[dict],
    *,
    use_history: bool,
    is_mc: bool = False,
    updates: int = 50,
    lr: float = 0.03,
    h_imag: int = 8,
    seed: int = 0,
    checkpoint_fracs: Sequence[float] = DEFAULT_CHECKPOINT_FRACS,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    theta = torch.zeros(28, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    curve = []
    theta0 = theta.detach().numpy().copy()
    want = {max(0, min(updates, int(round(f * updates)))) for f in checkpoint_fracs}
    want.add(0)
    want.add(updates)
    checkpoints: dict[str, list[float]] = {"u0": theta0.tolist()}

    for u in range(updates):
        if is_mc:
            Js = [imagined_mc(model, theta, s, horizon=h_imag) for s in starts]
        else:
            Js = [
                imagined_eagm(model, theta, s, horizon=h_imag, use_history=use_history)
                for s in starts
            ]
        loss = -torch.stack(Js).mean()
        opt.zero_grad()
        loss.backward()
        gnorm = float(theta.grad.detach().norm()) if theta.grad is not None else 0.0
        nn.utils.clip_grad_norm_([theta], 1.0)
        opt.step()
        with torch.no_grad():
            # Record saturation BEFORE clamp (post-step, pre-projection)
            sat_pre = theta_param_sat_frac(theta.detach().numpy())
            n_at_bound_pre = int((theta.detach().abs() >= 1.0 - 1e-8).sum())
            theta.clamp_(-1, 1)
            sat_post = theta_param_sat_frac(theta.detach().numpy())
        step = u + 1
        th_np = theta.detach().numpy().copy()
        curve.append(
            dict(
                update=step,
                imagined_return=float((-loss).detach()),
                grad_norm=gnorm,
                theta_norm=float(np.linalg.norm(th_np)),
                # θ parameter boundary saturation (NOT actor action clip)
                theta_param_sat_frac=sat_post,
                theta_param_sat_frac_pre_clamp=sat_pre,
                n_params_at_bound=n_at_bound_pre,
                # backward-compatible alias
                sat_frac=sat_post,
                theta=th_np.tolist(),
            )
        )
        if step in want:
            checkpoints[f"u{step}"] = th_np.tolist()

    # Named aliases for common path points
    def _frac_key(frac: float) -> str:
        step = max(0, min(updates, int(round(frac * updates))))
        return f"u{step}"

    named = {
        "theta0": checkpoints.get("u0", theta0.tolist()),
        "theta_25": checkpoints.get(_frac_key(0.25), curve[max(0, len(curve) // 4 - 1)]["theta"] if curve else theta0.tolist()),
        "theta_50": checkpoints.get(_frac_key(0.50), curve[max(0, len(curve) // 2 - 1)]["theta"] if curve else theta0.tolist()),
        "theta_75": checkpoints.get(_frac_key(0.75), curve[max(0, (3 * len(curve)) // 4 - 1)]["theta"] if curve else theta0.tolist()),
        "theta_final": theta.detach().numpy().tolist(),
    }
    theta_f = theta.detach().numpy().copy()
    return dict(
        theta0=theta0,
        theta=theta_f,
        curve=curve,
        checkpoints={**checkpoints, **named},
        final_theta_param_sat_frac=theta_param_sat_frac(theta_f),
        lr=lr,
        updates=updates,
    )


def eval_physical(env, theta, starts, horizon=16) -> dict[str, Any]:
    """Physical eval with action-clip and θ-param saturation recorded separately."""
    rets = []
    for i, s in enumerate(starts):
        d = physical_return_detailed(env, theta, s, horizon=horizon)
        rets.append(
            dict(
                start_index=i,
                episode_id=s.get("episode_id"),
                frame=s.get("frame"),
                delay=s.get("delay"),
                config_id=s.get("config_id"),
                return_=d["return_"],
                # action saturation (clipping)
                action_clip_frac=d["action_clip_frac"],
                action_raw_exceed_frac=d["action_raw_exceed_frac"],
                per_joint_clip_frac=d["per_joint_clip_frac"],
                # backward compat: sat_frac == action_clip_frac
                sat_frac=d["action_clip_frac"],
            )
        )
    arr = np.asarray([x["return_"] for x in rets], dtype=np.float64)
    return dict(
        mean=float(arr.mean()) if len(arr) else float("nan"),
        std=float(arr.std()) if len(arr) else float("nan"),
        per_start=rets,
        theta_norm=float(np.linalg.norm(theta)),
        theta_param_sat_frac=theta_param_sat_frac(theta),
        mean_action_clip_frac=float(np.mean([x["action_clip_frac"] for x in rets])) if rets else float("nan"),
        # legacy field: was incorrectly mixing meanings; keep as θ-param sat
        sat_frac=theta_param_sat_frac(theta),
    )


def eval_physical_paired_vs_theta0(
    env,
    theta_final,
    starts,
    horizon=16,
    *,
    theta0: Optional[np.ndarray] = None,
    fail_eps: float = 0.0,
) -> dict[str, Any]:
    """Paired physical eval: same starts for θ_final and θ_0; correct fail rate."""
    th0 = np.zeros(28) if theta0 is None else np.asarray(theta0, dtype=np.float64)
    ev0 = eval_physical(env, th0, starts, horizon=horizon)
    evf = eval_physical(env, theta_final, starts, horizon=horizon)
    r0 = [x["return_"] for x in ev0["per_start"]]
    rf = [x["return_"] for x in evf["per_start"]]
    fr = fail_rate_vs_theta0(rf, r0, eps=fail_eps)
    per_cfg = aggregate_per_config_paired(evf["per_start"], ev0["per_start"], eps=fail_eps)
    # Attach ΔR onto per_start rows
    paired_rows = []
    for a, b, d in zip(evf["per_start"], ev0["per_start"], np.asarray(rf) - np.asarray(r0)):
        paired_rows.append({**a, "return_theta0": b["return_"], "delta_vs_theta0": float(d)})
    return dict(
        theta0_return_mean=ev0["mean"],
        policy_return_mean=evf["mean"],
        delta_vs_theta0_mean=fr["delta_mean"],
        fail_rate_vs_theta0=fr["fail_rate"],
        fail_rate_detail=fr,
        per_config=per_cfg,
        per_start=paired_rows,
        theta_param_sat_frac=evf["theta_param_sat_frac"],
        mean_action_clip_frac=evf["mean_action_clip_frac"],
        theta0_eval=ev0,
        final_eval=evf,
    )
