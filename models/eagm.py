"""EAGM world model: known map U0 + learned residual command bottleneck.

Non-visual first. Inference uses c_obs only; true u may supervise training.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, mem: int = 32):
        super().__init__()
        self.rnn = nn.GRU(obs_dim + act_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, mem)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        # obs_hist: [B,T,obs], act_hist: [B,T,act]
        x = torch.cat([obs_hist, act_hist], dim=-1)
        y, _ = self.rnn(x)
        return self.out(y[:, -1])


class CommandBottleneck(nn.Module):
    """u_hat = U0(c_obs, a) + Delta(m, a)."""

    def __init__(self, mem_dim: int, act_dim: int = 14, width: int = 64):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(mem_dim + act_dim, width),
            nn.SiLU(),
            nn.Linear(width, act_dim),
        )

    def forward(
        self,
        u0: torch.Tensor,
        mem: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return u0 + self.delta(torch.cat([mem, action], dim=-1))


class EAGMWorld(nn.Module):
    def __init__(
        self,
        stats: dict,
        width: int = 160,
        mem_dim: int = 32,
        history: bool = True,
    ):
        super().__init__()
        self.use_history = history
        self.encoder = HistoryEncoder(28, 14, hidden=64, mem=mem_dim) if history else None
        self.bottleneck = CommandBottleneck(mem_dim if history else 0, 14, 64)
        in_dim = 28 + 14  # x + u_hat
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 28),
        )
        for name, val in stats.items():
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float32))

    @staticmethod
    def u0_from_cobs(c_obs: torch.Tensor, action: torch.Tensor, lower, upper, alpha: float = 0.5):
        """Known partial map: accumulate+filter, ignore hidden lag/gain."""
        target = torch.maximum(
            torch.minimum(c_obs[..., :14] + action, upper), lower
        )
        filt = alpha * target + (1.0 - alpha) * c_obs[..., 14:28]
        # Without true delay/queue, use filter as nominal command.
        return filt

    def forward(
        self,
        x: torch.Tensor,
        c_obs: torch.Tensor,
        a: torch.Tensor,
        obs_hist: Optional[torch.Tensor] = None,
        act_hist: Optional[torch.Tensor] = None,
        mem: Optional[torch.Tensor] = None,
    ):
        if mem is None:
            if self.use_history:
                if obs_hist is None or act_hist is None:
                    raise ValueError("history tensors required")
                mem = self.encoder(obs_hist, act_hist)
            else:
                mem = x.new_zeros(x.shape[0], 0)
        u0 = self.u0_from_cobs(c_obs, a, self.lower, self.upper, float(self.alpha))
        if self.use_history:
            u_hat = self.bottleneck(u0, mem, a)
        else:
            u_hat = u0
        xx = (x - self.xmean) / self.xstd
        uu = (u_hat - self.umean) / self.ustd
        delta = self.net(torch.cat([xx, uu], dim=-1)) * self.ystd + self.ymean
        return x + delta, u_hat, mem
