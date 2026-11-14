"""Versioned FD labels and calibration objectives for exploratory EAGM 2.0.

No legacy labels are rewritten. ``objective`` must restore the same physical
snapshot, reference and RNG for every query; query count is not simulator steps.
Multi-scale agreement measures numerical consistency, not confidence/accuracy.
"""
from __future__ import annotations

from typing import Callable, Iterable
import math
import numpy as np
import torch
import torch.nn.functional as F


def finite_difference(
    objective: Callable[[np.ndarray], float],
    theta,
    *,
    epsilon: float = 0.01,
    scales: Iterable[float] = (0.5, 1.0, 2.0),
    bounds=(-1.0, 1.0),
    relative_tolerance: float = 0.10,
    absolute_tolerance: float = 1e-9,
    cache: bool = True,
) -> dict:
    """Estimate the gradient at exactly ``theta`` with feasible stencils.

    Symmetric points use a central difference. Unequal feasible distances use
    the three-point interpolation derivative, including the center query.
    Exact boundaries use a second-order one-sided three-point derivative.
    Invalid/nonfinite labels are masked. A single scale has no stability proof
    and its mask is false. The default caches exact repeated query vectors;
    disable caching when deliberately measuring independent stochastic noise.
    ``bounds=None`` permits symmetric off-domain queries (explicit legacy mode).
    """
    t = np.asarray(theta, dtype=np.float64).copy()
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
        raise ValueError('theta must be a nonempty finite vector')
    scales = np.asarray(tuple(scales), dtype=np.float64)
    if not np.isfinite(epsilon) or epsilon <= 0 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('epsilon and scales must be positive finite values')
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError('tolerances must be nonnegative')
    if bounds is None:
        lo, hi = np.full_like(t, -np.inf), np.full_like(t, np.inf)
    else:
        lo = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), t.shape)
        hi = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), t.shape)
        if np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo >= hi):
            raise ValueError('bounds must be ordered and nondegenerate')
        if np.any(t < lo) or np.any(t > hi):
            raise ValueError('theta outside bounds: refusing silently changed label position')
    query_cache: dict[bytes, float] = {}
    queries = 0
    def evaluate(x):
        nonlocal queries
        key = x.tobytes()
        if cache and key in query_cache:
            return query_cache[key]
        value = float(objective(x.copy()))
        queries += 1
        if cache:
            query_cache[key] = value
        return value

    estimates = np.full((len(t), len(scales)), np.nan)
    details = []
    for j in range(len(t)):
        component = []
        for k, scale in enumerate(scales):
            h = float(epsilon * scale)
            hm, hp = min(h, float(t[j] - lo[j])), min(h, float(hi[j] - t[j]))
            tol = 4 * np.finfo(float).eps * max(1.0, abs(float(t[j])))
            if hm > tol and hp > tol:
                if abs(hp - hm) <= tol:
                    offsets = [-hm, hp]
                    weights = [-1.0 / (hm + hp), 1.0 / (hm + hp)]
                    mode = 'symmetric_central'
                else:
                    offsets = [-hm, 0.0, hp]
                    weights = [-hp / (hm * (hp + hm)), (hp - hm) / (hp * hm), hm / (hp * (hp + hm))]
                    mode = 'nonuniform_three_point'
            elif hp > tol:
                h = min(h, float(hi[j] - t[j]) / 2.0)
                offsets, weights = [0.0, h, 2.0*h], [-1.5/h, 2.0/h, -0.5/h]
                mode = 'forward_three_point'
            elif hm > tol:
                h = min(h, float(t[j] - lo[j]) / 2.0)
                offsets, weights = [-2.0*h, -h, 0.0], [0.5/h, -2.0/h, 1.5/h]
                mode = 'backward_three_point'
            else:
                component.append(dict(scale=float(scale), mode='collapsed', offsets=[], weights=[], estimate=None))
                continue
            values = []
            for offset in offsets:
                x = t.copy()
                x[j] = np.clip(t[j] + offset, lo[j], hi[j])
                values.append(evaluate(x))
            # Centered sum avoids cancellation from a large additive baseline.
            value = sum(w * (v-values[0]) for w, v in zip(weights, values))
            estimates[j, k] = value
            component.append(dict(scale=float(scale), mode=mode, offsets=offsets,
                                  weights=weights, values=values, estimate=float(value)))
        details.append(component)
    chosen = int(np.argmin(np.abs(np.log(scales))))
    gradient = estimates[:, chosen].copy()
    finite = np.isfinite(estimates).all(axis=1)
    spread = np.full(len(t), np.inf)
    spread[finite] = np.ptp(estimates[finite], axis=1)
    stable = finite & (spread <= np.maximum(absolute_tolerance, relative_tolerance*np.abs(gradient)))
    if len(scales) < 2:
        stable[:] = False
    return dict(theta=t, gradient=gradient, mask=stable.astype(np.float64), queries=queries,
                details=details, estimates=estimates, spread=spread, selected_scale=float(scales[chosen]),
                cache_enabled=cache, label_kind='matched_theta_feasible_fd',
                stability_is_confidence_interval=False)


def calibration_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     *, mode: str = 'legacy', scale_floor: float | None = None,
                     epsilon: float = 1e-8, direction_weight: float = 1.0,
                     norm_weight: float = 1.0) -> torch.Tensor:
    """Masked losses with higher-order gradients through ``pred``.

    Targets and masks are fixed label data. Zero target norm has no direction.
    Empty masks return a graph-connected zero, with no model update signal.
    Batched inputs average valid examples for direction/norm/joint modes.
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('pred, target and mask must have identical shapes')
    if mode not in ('none', 'legacy', 'direction', 'norm', 'joint'):
        raise ValueError('unknown calibration mode')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    target = target.detach().to(pred)
    selected = (mask.detach().to(pred.device) > .5) & torch.isfinite(target)
    if not torch.isfinite(pred[selected]).all():
        raise ValueError('nonfinite model gradient on selected coordinates')
    if mode == 'none' or not bool(selected.any()):
        return torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred)).sum() * 0.0
    safe_pred = torch.where(selected, pred, torch.zeros_like(pred))
    safe_target = torch.where(selected, target, torch.zeros_like(target))
    if mode == 'legacy':
        if scale_floor is None:
            scale_floor = max(0.1 * float(safe_target[selected].abs().median()), epsilon)
        if not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError('scale_floor must be finite and positive')
        error = (safe_pred - safe_target) / safe_target.abs().clamp_min(scale_floor)
        return F.huber_loss(error[selected], torch.zeros_like(error[selected]), reduction='mean')
    if pred.ndim == 1:
        safe_pred, safe_target = safe_pred[None], safe_target[None]
    pn = torch.linalg.vector_norm(safe_pred, dim=-1)
    qn = torch.linalg.vector_norm(safe_target, dim=-1)
    valid = qn > epsilon
    if not bool(valid.any()):
        return safe_pred.sum()*0.0
    cosine = (safe_pred*safe_target).sum(-1) / (pn.clamp_min(epsilon)*qn.clamp_min(epsilon))
    direction = (1.0-cosine)[valid].mean()
    log_ratio = torch.log((pn+epsilon)/(qn+epsilon))
    norm = F.huber_loss(log_ratio[valid], torch.zeros_like(log_ratio[valid]), reduction='mean')
    if mode == 'direction':
        return direction
    if mode == 'norm':
        return norm
    return direction_weight*direction + norm_weight*norm


def gradient_metrics(pred, target, mask=None, *, zero_tolerance=1e-12) -> dict:
    """Seed-agnostic descriptive metrics; undefined quantities use JSON null."""
    p, q = np.asarray(pred, dtype=float).ravel(), np.asarray(target, dtype=float).ravel()
    if p.shape != q.shape:
        raise ValueError('gradient shapes differ')
    requested = np.ones(p.shape, bool) if mask is None else np.asarray(mask).ravel() > .5
    if requested.shape != p.shape:
        raise ValueError('mask shape differs')
    finite = np.isfinite(p) & np.isfinite(q)
    valid = requested & finite
    pp, qq = p[valid], q[valid]
    pn, qn = np.linalg.norm(pp), np.linalg.norm(qq)
    nz = np.abs(qq) > zero_tolerance
    positive, negative = qq > zero_tolerance, qq < -zero_tolerance
    correct = np.sign(pp) == np.sign(qq)
    errors = [float(x-y) if np.isfinite(x) and np.isfinite(y) else None for x,y in zip(p,q)]
    return dict(cosine=float(pp@qq/(pn*qn)) if pn>zero_tolerance and qn>zero_tolerance else None,
                norm_ratio=float(pn/qn) if qn>zero_tolerance else None,
                sign_accuracy=float(correct[nz].mean()) if nz.any() else None,
                balanced_sign_accuracy=float(.5*(correct[positive].mean()+correct[negative].mean())) if positive.any() and negative.any() else None,
                rmse=float(np.sqrt(np.mean((pp-qq)**2))) if len(pp) else None,
                relative_l2=float(np.linalg.norm(pp-qq)/qn) if qn>zero_tolerance else None,
                mask_coverage=float(requested.mean()) if len(p) else 0.0,
                finite_selected_coverage=float(valid.mean()) if len(p) else 0.0,
                n_selected=int(valid.sum()), n_nonfinite_selected=int((requested & ~finite).sum()),
                n_nonzero_target=int(nz.sum()), per_coordinate_error=errors)


def spsa_gradient(objective, theta, *, epsilon=.01, directions=1, seed=0, bounds=None):
    """Independent Rademacher-direction estimator with exact query accounting.

    No post-hoc clipping of perturbed points: bounds require an interior theta
    or raise. Projection of optimization updates is the caller's responsibility.
    """
    t = np.asarray(theta, dtype=float).copy()
    if t.ndim != 1 or not np.isfinite(t).all() or epsilon <= 0 or directions < 1:
        raise ValueError('invalid SPSA inputs')
    rng = np.random.default_rng(seed)
    estimates=[]
    for _ in range(directions):
        v=rng.choice([-1.,1.],size=t.shape)
        plus,minus=t+epsilon*v,t-epsilon*v
        if bounds is not None and (np.any(plus<bounds[0]) or np.any(plus>bounds[1]) or np.any(minus<bounds[0]) or np.any(minus>bounds[1])):
            raise ValueError('SPSA stencil exceeds bounds; use interior parameters')
        estimates.append((float(objective(plus))-float(objective(minus)))/(2*epsilon)*v)
    return dict(gradient=np.mean(estimates,axis=0), queries=2*directions, estimates=np.asarray(estimates), seed=seed)
