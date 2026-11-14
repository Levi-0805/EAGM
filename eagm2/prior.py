"""Causal, observable nominal Controller-A prior inside Controller B.

The returned command is nominal: it intentionally excludes the hidden B stack.
State convention is pre-step c=(target,filter,delay), f_prev=previous observed
filter. The cache can be maintained from public observations, including alpha=1.
"""
from __future__ import annotations
import torch


def _check(c, a, f_prev=None):
    if c.shape[-1] != 29 or a.shape[-1] != 14 or c.shape[:-1] != a.shape[:-1]:
        raise ValueError('expected matching [...,29] control and [...,14] action')
    if not torch.isfinite(c).all() or not torch.isfinite(a).all():
        raise ValueError('control/action must be finite')
    d = c[..., 28]
    if not ((d == 0) | (d == 1) | (d == 2)).all():
        raise ValueError('declared delay must be 0, 1, or 2')
    if f_prev is not None and (f_prev.shape != a.shape or not torch.isfinite(f_prev).all()):
        raise ValueError('previous observable filter must match action shape and be finite')


def observable_step(c, action, lower, upper, alpha=0.5, max_increment=0.03):
    """Advance target/filter and public cache; delay remains externally fixed."""
    _check(c, action)
    alpha = torch.as_tensor(alpha, dtype=c.dtype, device=c.device)
    if not ((alpha > 0) & (alpha <= 1)).all():
        raise ValueError('alpha must lie in (0,1]')
    lo = torch.as_tensor(lower, dtype=c.dtype, device=c.device)
    hi = torch.as_tensor(upper, dtype=c.dtype, device=c.device)
    applied = action.clamp(-max_increment, max_increment)
    target = torch.maximum(torch.minimum(c[..., :14] + applied, hi), lo)
    filtered = alpha * target + (1-alpha) * c[..., 14:28]
    return torch.cat((target, filtered, c[..., 28:29]), -1), c[..., 14:28]


def nominal_command(c, action, f_prev, lower, upper, alpha=0.5, max_increment=0.03):
    """Exact nominal command for d=0/1/2 given a valid public filter cache."""
    _check(c, action, f_prev)
    next_c, _ = observable_step(c, action, lower, upper, alpha, max_increment)
    d = c[..., 28:29]
    return torch.where(d == 0, next_c[..., 14:28],
                       torch.where(d == 1, c[..., 14:28], f_prev))


def previous_filter_from_invariant(c, alpha=0.5):
    """Restricted algebraic reconstruction; caller must ensure update invariant.

    Prefer an observed history cache. No claim of validity for arbitrary c pairs.
    """
    if not 0 < float(alpha) < 1:
        raise ValueError('inverse requires known fixed alpha in (0,1)')
    return (c[..., 14:28]-alpha*c[..., :14])/(1-alpha)
