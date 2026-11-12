"""EAGM world models: U0 + residual; optional history encoder.

Deployable models must NOT consume Controller B hidden state or known_config_vector
unless explicitly marked privileged.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def u0_from_cobs(c_obs: torch.Tensor, action: torch.Tensor, lower, upper, alpha: float = 0.5):
    target = torch.maximum(torch.minimum(c_obs[..., :14] + action, upper), lower)
    return alpha * target + (1.0 - alpha) * c_obs[..., 14:28]


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim: int = 28, act_dim: int = 14, hidden: int = 64, mem: int = 32):
        super().__init__()
        self.rnn = nn.GRU(obs_dim + act_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, mem)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_hist, act_hist], dim=-1)
        y, _ = self.rnn(x)
        return self.out(y[:, -1])


class EAGMBase(nn.Module):
    """Markov residual on (c_obs, a) only — NOT history-aware."""

    def __init__(self, stats: dict, width: int = 160):
        super().__init__()
        self.kind = "eagm_base"
        self.delta = nn.Sequential(nn.Linear(29 + 14, 64), nn.SiLU(), nn.Linear(64, 14))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.net = nn.Sequential(
            nn.Linear(42, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for name, val in stats.items():
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float32))

    def command(self, c_obs, a):
        u0 = u0_from_cobs(c_obs, a, self.lower, self.upper, float(self.alpha))
        return u0 + self.delta(torch.cat([c_obs, a], -1))

    def forward(self, x, c_obs, a, obs_hist=None, act_hist=None):
        u_hat = self.command(c_obs, a)
        xx = (x - self.xmean) / self.xstd
        uu = (u_hat - self.umean) / self.ustd
        y = x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean
        return y, u_hat


class EAGMHistory(nn.Module):
    """m_t = H(x hist, a hist); u_hat = U0 + Delta(m, a)."""

    def __init__(self, stats: dict, width: int = 160, mem_dim: int = 32, hist_len: int = 8):
        super().__init__()
        self.kind = "eagm_history"
        self.hist_len = hist_len
        self.encoder = HistoryEncoder(28, 14, hidden=64, mem=mem_dim)
        self.delta = nn.Sequential(
            nn.Linear(mem_dim + 14, 64), nn.SiLU(), nn.Linear(64, 14)
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.net = nn.Sequential(
            nn.Linear(42, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for name, val in stats.items():
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float32))

    def encode(self, obs_hist, act_hist):
        return self.encoder(obs_hist, act_hist)

    def command(self, c_obs, a, mem):
        u0 = u0_from_cobs(c_obs, a, self.lower, self.upper, float(self.alpha))
        return u0 + self.delta(torch.cat([mem, a], -1))

    def forward(self, x, c_obs, a, obs_hist, act_hist):
        mem = self.encode(obs_hist, act_hist)
        u_hat = self.command(c_obs, a, mem)
        xx = (x - self.xmean) / self.xstd
        uu = (u_hat - self.umean) / self.ustd
        y = x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean
        return y, u_hat, mem


class McWorld(nn.Module):
    """Full Controller-A 57-D memory. Does NOT include Controller-B hidden state."""

    def __init__(self, stats: dict, width: int = 160):
        super().__init__()
        self.kind = "mc_controller_a_memory"
        self.net = nn.Sequential(
            nn.Linear(28 + 14 + 57, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for name, val in stats.items():
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float32))

    def forward(self, x, c, a, h=None):
        xx = (x - self.xmean) / self.xstd
        return x + self.net(torch.cat([xx, a, c], -1)) * self.ystd + self.ymean


class MuOneStepOracle(nn.Module):
    """Privileged one-step dynamics upper bound: consumes true u_exec for that step.

    Must NOT be used for imagined multi-step policy rollouts with future true u.
    """

    def __init__(self, stats: dict, width: int = 160):
        super().__init__()
        self.kind = "mu_onestep_oracle"
        self.net = nn.Sequential(
            nn.Linear(42, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for name, val in stats.items():
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float32))

    def forward_u(self, x, u):
        xx = (x - self.xmean) / self.xstd
        uu = (u - self.umean) / self.ustd
        return x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean
