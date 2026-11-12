"""Directional consistency loss with numerical stability masking."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


def huber_dir_loss(
    g_model: torch.Tensor,
    g_sim: torch.Tensor,
    stable_mask: torch.Tensor,
    s0: float,
    tau: float | None = None,
    delta: float = 1.0,
) -> Optional[torch.Tensor]:
    """Return Huber loss over stable components, or None if none are stable.

    Returning None lets callers skip the term instead of inventing a zero loss
    that still participates in autograd graphs incorrectly.
    """
    if tau is None:
        tau = max(float(s0) * 0.1, 1e-8)
    scale = torch.clamp(g_sim.abs(), min=tau)
    err = (g_model - g_sim) / scale
    per = nn.functional.huber_loss(err, torch.zeros_like(err), delta=delta, reduction="none")
    w = stable_mask.to(dtype=per.dtype)
    n = w.sum()
    if float(n.detach()) < 1.0:
        return None
    return (per * w).sum() / n


def stack_labels(
    grads: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.as_tensor(np.stack(list(grads)), dtype=torch.float32)
    m = torch.as_tensor(np.stack(list(masks)), dtype=torch.float32)
    return g, m
