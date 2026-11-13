"""Finite-difference policy gradients with stability and boundary masks."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from training.eval_metrics import gradient_quality
from training.rollout import physical_return

THETA_LO = -1.0
THETA_HI = 1.0
BOUND_EPS = 1e-8


def _clip_theta(th: np.ndarray, lo: float = THETA_LO, hi: float = THETA_HI) -> np.ndarray:
    return np.clip(th, lo, hi)


def sim_grad_masked(
    env,
    start,
    theta,
    *,
    eps: float = 0.05,
    scales=(0.5, 1.0, 2.0),
    horizon: int = 8,
    feasible: bool = False,
    bound_lo: float = THETA_LO,
    bound_hi: float = THETA_HI,
):
    """∇_θ J_sim at the provided θ.

    Default ``feasible=False`` preserves Stage 3–6B central differences
    (may evaluate outside [-1,1]). Set ``feasible=True`` for Stage 6C
    bound-aware FD: interior central, active bounds one-sided.

    Returns (g, stable_mask, control_steps_used).
    Stability mask = numerical consistency across scales (NOT boundary mask).
    """
    g, stable, steps, _ = sim_grad_masked_detailed(
        env, start, theta, eps=eps, scales=scales, horizon=horizon,
        feasible=feasible, bound_lo=bound_lo, bound_hi=bound_hi,
    )
    return g, stable, steps


def sim_grad_masked_detailed(
    env,
    start,
    theta,
    *,
    eps: float = 0.05,
    scales=(0.5, 1.0, 2.0),
    horizon: int = 8,
    feasible: bool = False,
    bound_lo: float = THETA_LO,
    bound_hi: float = THETA_HI,
):
    """Like sim_grad_masked plus FD metadata (boundary / mode / step counts)."""
    theta = np.asarray(theta, dtype=np.float64).reshape(28).copy()
    if feasible:
        theta = _clip_theta(theta, bound_lo, bound_hi)

    out, stable, modes, boundary = [], [], [], []
    steps = 0

    for i in range(28):
        e = np.zeros(28)
        e[i] = 1.0
        ti = float(theta[i])
        at_lo = ti <= bound_lo + BOUND_EPS
        at_hi = ti >= bound_hi - BOUND_EPS
        boundary.append(1.0 if (at_lo or at_hi) else 0.0)

        vals: list[float] = []
        mode = "central"

        if feasible and at_lo and not at_hi:
            mode = "forward"
            for sc in scales:
                ep = float(eps * sc)
                th_p = theta.copy()
                th_p[i] = min(bound_hi, ti + ep)
                ep_eff = float(th_p[i] - ti)
                if ep_eff <= 1e-15:
                    vals.append(0.0)
                    continue
                plus, _ = physical_return(env, th_p, start, horizon=horizon)
                base, _ = physical_return(env, theta, start, horizon=horizon)
                vals.append((plus - base) / ep_eff)
                steps += 2 * horizon
        elif feasible and at_hi and not at_lo:
            mode = "backward"
            for sc in scales:
                ep = float(eps * sc)
                th_m = theta.copy()
                th_m[i] = max(bound_lo, ti - ep)
                ep_eff = float(ti - th_m[i])
                if ep_eff <= 1e-15:
                    vals.append(0.0)
                    continue
                minus, _ = physical_return(env, th_m, start, horizon=horizon)
                base, _ = physical_return(env, theta, start, horizon=horizon)
                vals.append((base - minus) / ep_eff)
                steps += 2 * horizon
        elif feasible:
            mode = "central"
            for sc in scales:
                ep = float(eps * sc)
                th_p = _clip_theta(theta + ep * e, bound_lo, bound_hi)
                th_m = _clip_theta(theta - ep * e, bound_lo, bound_hi)
                span = float(th_p[i] - th_m[i])
                if abs(span) < 1e-15:
                    # stencil collapsed — one-sided fallback
                    mid = 0.5 * (bound_lo + bound_hi)
                    if ti < mid:
                        mode = "forward"
                        th_p2 = theta.copy()
                        th_p2[i] = min(bound_hi, ti + ep)
                        ep_eff = float(th_p2[i] - ti)
                        plus, _ = physical_return(env, th_p2, start, horizon=horizon)
                        base, _ = physical_return(env, theta, start, horizon=horizon)
                        vals.append((plus - base) / max(ep_eff, 1e-15))
                    else:
                        mode = "backward"
                        th_m2 = theta.copy()
                        th_m2[i] = max(bound_lo, ti - ep)
                        ep_eff = float(ti - th_m2[i])
                        minus, _ = physical_return(env, th_m2, start, horizon=horizon)
                        base, _ = physical_return(env, theta, start, horizon=horizon)
                        vals.append((base - minus) / max(ep_eff, 1e-15))
                    steps += 2 * horizon
                    continue
                plus, _ = physical_return(env, th_p, start, horizon=horizon)
                minus, _ = physical_return(env, th_m, start, horizon=horizon)
                vals.append((plus - minus) / span)
                steps += 2 * horizon
        else:
            # Legacy central FD (may leave [-1,1])
            mode = "central"
            for sc in scales:
                ep = float(eps * sc)
                plus, _ = physical_return(env, theta + ep * e, start, horizon=horizon)
                minus, _ = physical_return(env, theta - ep * e, start, horizon=horizon)
                vals.append((plus - minus) / (2 * ep))
                steps += 2 * horizon

        v = vals[1] if len(vals) >= 2 else (vals[0] if vals else 0.0)
        out.append(float(v))
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            stable.append(spread <= max(1e-9, 0.1 * abs(v)))
        else:
            stable.append(False)
        modes.append(mode)

    meta = dict(
        boundary_mask=np.asarray(boundary, dtype=np.float64),
        fd_mode=modes,
        n_boundary=int(np.sum(boundary)),
        n_central=int(sum(m == "central" for m in modes)),
        n_onesided=int(sum(m != "central" for m in modes)),
        feasible=bool(feasible),
        control_steps=int(steps),
    )
    return (
        np.asarray(out, dtype=np.float64),
        np.asarray(stable, dtype=np.float64),
        int(steps),
        meta,
    )


def diagnose_grad_pair(
    g_hat: np.ndarray,
    g_sim: np.ndarray,
    mask: Optional[np.ndarray] = None,
    boundary_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Full/stable cosine; optionally interior-only (non-boundary) cosine."""
    q = gradient_quality(g_hat, g_sim, mask)
    if boundary_mask is not None:
        interior = (np.asarray(boundary_mask) < 0.5).astype(np.float64)
        if mask is not None:
            interior = interior * (np.asarray(mask) > 0.5).astype(np.float64)
        q["cosine_interior"] = gradient_quality(g_hat, g_sim, interior)["cosine_full"]
        q["n_interior_dims"] = int(interior.sum())
        q["n_boundary_dims"] = int(np.sum(np.asarray(boundary_mask) >= 0.5))
    return q

def sim_grad_true_ee_masked_detailed(
    env,
    start,
    theta,
    *,
    eps: float = 0.05,
    scales=(0.5, 1.0, 2.0),
    horizon: int = 8,
    feasible: bool = True,
    bound_lo: float = THETA_LO,
    bound_hi: float = THETA_HI,
):
    """Like sim_grad_masked plus FD metadata (boundary / mode / step counts)."""
    from training.rollout import physical_return_true_ee
    physical_return = physical_return_true_ee
    theta = np.asarray(theta, dtype=np.float64).reshape(28).copy()
    if feasible:
        theta = _clip_theta(theta, bound_lo, bound_hi)

    out, stable, modes, boundary = [], [], [], []
    steps = 0
    effective_eps = []

    for i in range(28):
        e = np.zeros(28)
        e[i] = 1.0
        ti = float(theta[i])
        at_lo = ti <= bound_lo + BOUND_EPS
        at_hi = ti >= bound_hi - BOUND_EPS
        boundary.append(1.0 if (at_lo or at_hi) else 0.0)

        effective_eps.append([dict(minus=min(float(eps * sc), ti-bound_lo) if feasible else float(eps*sc), plus=min(float(eps*sc), bound_hi-ti) if feasible else float(eps*sc)) for sc in scales])
        vals: list[float] = []
        mode = "central"

        if feasible and at_lo and not at_hi:
            mode = "forward"
            for sc in scales:
                ep = float(eps * sc)
                th_p = theta.copy()
                th_p[i] = min(bound_hi, ti + ep)
                ep_eff = float(th_p[i] - ti)
                if ep_eff <= 1e-15:
                    vals.append(0.0)
                    continue
                plus, _ = physical_return(env, th_p, start, horizon=horizon)
                base, _ = physical_return(env, theta, start, horizon=horizon)
                vals.append((plus - base) / ep_eff)
                steps += 2 * horizon
        elif feasible and at_hi and not at_lo:
            mode = "backward"
            for sc in scales:
                ep = float(eps * sc)
                th_m = theta.copy()
                th_m[i] = max(bound_lo, ti - ep)
                ep_eff = float(ti - th_m[i])
                if ep_eff <= 1e-15:
                    vals.append(0.0)
                    continue
                minus, _ = physical_return(env, th_m, start, horizon=horizon)
                base, _ = physical_return(env, theta, start, horizon=horizon)
                vals.append((base - minus) / ep_eff)
                steps += 2 * horizon
        elif feasible:
            mode = "central"
            for sc in scales:
                ep = float(eps * sc)
                th_p = _clip_theta(theta + ep * e, bound_lo, bound_hi)
                th_m = _clip_theta(theta - ep * e, bound_lo, bound_hi)
                span = float(th_p[i] - th_m[i])
                if abs(span) < 1e-15:
                    # stencil collapsed — one-sided fallback
                    mid = 0.5 * (bound_lo + bound_hi)
                    if ti < mid:
                        mode = "forward"
                        th_p2 = theta.copy()
                        th_p2[i] = min(bound_hi, ti + ep)
                        ep_eff = float(th_p2[i] - ti)
                        plus, _ = physical_return(env, th_p2, start, horizon=horizon)
                        base, _ = physical_return(env, theta, start, horizon=horizon)
                        vals.append((plus - base) / max(ep_eff, 1e-15))
                    else:
                        mode = "backward"
                        th_m2 = theta.copy()
                        th_m2[i] = max(bound_lo, ti - ep)
                        ep_eff = float(ti - th_m2[i])
                        minus, _ = physical_return(env, th_m2, start, horizon=horizon)
                        base, _ = physical_return(env, theta, start, horizon=horizon)
                        vals.append((base - minus) / max(ep_eff, 1e-15))
                    steps += 2 * horizon
                    continue
                plus, _ = physical_return(env, th_p, start, horizon=horizon)
                minus, _ = physical_return(env, th_m, start, horizon=horizon)
                vals.append((plus - minus) / span)
                steps += 2 * horizon
        else:
            # Legacy central FD (may leave [-1,1])
            mode = "central"
            for sc in scales:
                ep = float(eps * sc)
                plus, _ = physical_return(env, theta + ep * e, start, horizon=horizon)
                minus, _ = physical_return(env, theta - ep * e, start, horizon=horizon)
                vals.append((plus - minus) / (2 * ep))
                steps += 2 * horizon

        v = vals[1] if len(vals) >= 2 else (vals[0] if vals else 0.0)
        out.append(float(v))
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            stable.append(spread <= max(1e-9, 0.1 * abs(v)))
        else:
            stable.append(False)
        modes.append(mode)

    meta = dict(
        objective="true_ee",
        effective_epsilon=effective_eps,
        simulator_queries=steps // horizon,
        valid_dimensions=int(np.isfinite(out).sum()),
        stable_dimensions=int(np.sum(stable)),
        boundary_mask=np.asarray(boundary, dtype=np.float64),
        fd_mode=modes,
        n_boundary=int(np.sum(boundary)),
        n_central=int(sum(m == "central" for m in modes)),
        n_onesided=int(sum(m != "central" for m in modes)),
        feasible=bool(feasible),
        control_steps=int(steps),
    )
    return (
        np.asarray(out, dtype=np.float64),
        np.asarray(stable, dtype=np.float64),
        int(steps),
        meta,
    )
