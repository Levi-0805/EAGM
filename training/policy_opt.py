"""Policy optimization with θ traces, saturation, and paired physical evaluation."""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from training.eval_metrics import (
    aggregate_per_config_paired,
    fail_rate_vs_theta0,
    theta_param_sat_frac,
)
from training.gradients import BOUND_EPS, THETA_HI, THETA_LO
from training.rollout import imagined_eagm, imagined_mc, physical_return_detailed

DTYPE = torch.float32

# Default path fractions for Stage-6 diagnostics / path-aware Dir
DEFAULT_CHECKPOINT_FRACS = (0.0, 0.25, 0.50, 0.75, 1.0)
# Stage 6C Path-Dir fractions (0, 0.2, ..., 1.0)
STAGE6C_PATH_FRACS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def rho_proj(delta_raw: np.ndarray, delta_proj: np.ndarray, eps: float = 1e-12) -> float:
    """Fraction of optimizer step magnitude removed by box projection."""
    raw = np.asarray(delta_raw, dtype=np.float64).ravel()
    proj = np.asarray(delta_proj, dtype=np.float64).ravel()
    return float(np.linalg.norm(raw - proj)) / (float(np.linalg.norm(raw)) + eps)


def outward_pressure_frac(
    theta: np.ndarray,
    grad: np.ndarray,
    lo: float = THETA_LO,
    hi: float = THETA_HI,
) -> float:
    """Of currently saturated coords, fraction whose ascent gradient still pushes out."""
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    g = np.asarray(grad, dtype=np.float64).reshape(-1)
    at_lo = th <= lo + BOUND_EPS
    at_hi = th >= hi - BOUND_EPS
    active = at_lo | at_hi
    n_act = int(active.sum())
    if n_act == 0:
        return 0.0
    # ascent: update ∝ +g; outward at lo means g<0; at hi means g>0
    outward = (at_lo & (g < 0)) | (at_hi & (g > 0))
    return float(outward[active].mean())


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
    physical_eval_fn: Optional[Callable[[np.ndarray, int], dict]] = None,
    bound: float = 1.0,
) -> dict[str, Any]:
    """Adam ascent on imagined return with projection / saturation diagnostics.

    physical_eval_fn(theta, update_idx) optional; called at checkpoint updates
    including 0 and ``updates``.
    """
    torch.manual_seed(seed)
    theta = torch.zeros(28, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    curve = []
    theta0 = theta.detach().numpy().copy()
    want = {max(0, min(updates, int(round(f * updates)))) for f in checkpoint_fracs}
    want.add(0)
    want.add(updates)
    checkpoints: dict[str, list[float]] = {"u0": theta0.tolist()}
    first_hit = np.full(28, -1, dtype=np.int32)
    already = np.abs(theta0) >= bound - BOUND_EPS
    first_hit[already] = 0
    phys_curve: list[dict] = []

    def _maybe_phys(u: int, th_np: np.ndarray):
        if physical_eval_fn is None:
            return
        if u in want:
            info = dict(physical_eval_fn(th_np.copy(), u))
            info["update"] = int(u)
            info["theta_param_sat_frac"] = theta_param_sat_frac(th_np)
            phys_curve.append(info)

    _maybe_phys(0, theta0)

    for u in range(updates):
        th_before = theta.detach().numpy().copy()
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
        g = (
            theta.grad.detach().cpu().numpy().astype(np.float64)
            if theta.grad is not None
            else np.zeros(28)
        )
        # For ascent on J, the stored grad is d(-J)/dθ = -∇J; outward uses ∇J = -g_loss
        g_ascent = -g
        gnorm = float(np.linalg.norm(g))
        nn.utils.clip_grad_norm_([theta], 1.0)
        opt.step()
        with torch.no_grad():
            th_pre = theta.detach().numpy().copy()
            sat_pre = theta_param_sat_frac(th_pre)
            n_at_bound_pre = int((np.abs(th_pre) >= bound - BOUND_EPS).sum())
            delta_raw = th_pre - th_before
            theta.clamp_(-bound, bound)
            th_post = theta.detach().numpy().copy()
            delta_proj = th_post - th_before
            sat_post = theta_param_sat_frac(th_post)
            rho = rho_proj(delta_raw, delta_proj)
            outward = outward_pressure_frac(th_post, g_ascent, -bound, bound)

        step = u + 1
        now_hit = (np.abs(th_post) >= bound - BOUND_EPS) & (first_hit < 0)
        first_hit[now_hit] = step

        curve.append(
            dict(
                update=step,
                imagined_return=float((-loss).detach()),
                grad_norm=gnorm,
                theta_norm=float(np.linalg.norm(th_post)),
                theta_param_sat_frac=sat_post,
                theta_param_sat_frac_pre_clamp=sat_pre,
                n_params_at_bound=n_at_bound_pre,
                rho_proj=rho,
                outward_pressure_frac=outward,
                delta_raw_norm=float(np.linalg.norm(delta_raw)),
                delta_proj_norm=float(np.linalg.norm(delta_proj)),
                # backward-compatible alias
                sat_frac=sat_post,
                theta=th_post.tolist(),
            )
        )
        if step in want:
            checkpoints[f"u{step}"] = th_post.tolist()
        _maybe_phys(step, th_post)

    def _frac_key(frac: float) -> str:
        step = max(0, min(updates, int(round(frac * updates))))
        return f"u{step}"

    named = {
        "theta0": checkpoints.get("u0", theta0.tolist()),
        "theta_25": checkpoints.get(
            _frac_key(0.25),
            curve[max(0, len(curve) // 4 - 1)]["theta"] if curve else theta0.tolist(),
        ),
        "theta_50": checkpoints.get(
            _frac_key(0.50),
            curve[max(0, len(curve) // 2 - 1)]["theta"] if curve else theta0.tolist(),
        ),
        "theta_75": checkpoints.get(
            _frac_key(0.75),
            curve[max(0, (3 * len(curve)) // 4 - 1)]["theta"] if curve else theta0.tolist(),
        ),
        "theta_final": theta.detach().numpy().tolist(),
    }
    # Stage 6C path aliases
    for frac in STAGE6C_PATH_FRACS:
        key = f"theta_{int(round(frac * 100))}"
        named[key] = checkpoints.get(_frac_key(frac), named["theta_final"])

    theta_f = theta.detach().numpy().copy()
    rho_vals = [c["rho_proj"] for c in curve]
    outward_vals = [c["outward_pressure_frac"] for c in curve]
    sat_vals = [c["theta_param_sat_frac"] for c in curve]
    return dict(
        theta0=theta0,
        theta=theta_f,
        curve=curve,
        checkpoints={**checkpoints, **named},
        final_theta_param_sat_frac=theta_param_sat_frac(theta_f),
        mean_rho_proj=float(np.mean(rho_vals)) if rho_vals else 0.0,
        final_rho_proj=float(rho_vals[-1]) if rho_vals else 0.0,
        mean_outward_pressure=float(np.mean(outward_vals)) if outward_vals else 0.0,
        mean_theta_param_sat_frac=float(np.mean(sat_vals)) if sat_vals else 0.0,
        first_hit_update=first_hit.tolist(),
        first_hit_mean=(
            float(np.mean(first_hit[first_hit >= 0])) if np.any(first_hit >= 0) else None
        ),
        physical_curve=phys_curve,
        lr=lr,
        updates=updates,
        seed=seed,
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
                action_clip_frac=d["action_clip_frac"],
                action_raw_exceed_frac=d["action_raw_exceed_frac"],
                per_joint_clip_frac=d["per_joint_clip_frac"],
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
