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


class HistoryEncoderC(nn.Module):
    """History over (x, a, c_obs). c_obs is OBSERVABLE only — never privileged B state.

    c_obs layout (29-D): target[0:14], filter[14:28], delay[28].
    """

    def __init__(
        self,
        obs_dim: int = 28,
        act_dim: int = 14,
        c_obs_dim: int = 29,
        hidden: int = 64,
        mem: int = 32,
    ):
        super().__init__()
        self.rnn = nn.GRU(obs_dim + act_dim + c_obs_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, mem)

    def forward(
        self,
        obs_hist: torch.Tensor,
        act_hist: torch.Tensor,
        c_obs_hist: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([obs_hist, act_hist, c_obs_hist], dim=-1)
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

    def __init__(
        self,
        stats: dict,
        width: int = 160,
        mem_dim: int = 32,
        hist_len: int = 8,
        encoder_hidden: int = 64,
    ):
        super().__init__()
        self.kind = "eagm_history"
        self.hist_len = hist_len
        self.uses_c_obs_hist = False
        self.encoder = HistoryEncoder(28, 14, hidden=encoder_hidden, mem=mem_dim)
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

    def encode(self, obs_hist, act_hist, c_obs_hist=None):
        return self.encoder(obs_hist, act_hist)

    def command(self, c_obs, a, mem):
        u0 = u0_from_cobs(c_obs, a, self.lower, self.upper, float(self.alpha))
        return u0 + self.delta(torch.cat([mem, a], -1))

    def forward(self, x, c_obs, a, obs_hist, act_hist, c_obs_hist=None):
        mem = self.encode(obs_hist, act_hist, c_obs_hist)
        u_hat = self.command(c_obs, a, mem)
        xx = (x - self.xmean) / self.xstd
        uu = (u_hat - self.umean) / self.ustd
        y = x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean
        return y, u_hat, mem


class EAGMHistoryC(nn.Module):
    """m_t = H(x hist, a hist, c_obs hist); deployable — c_obs only, no privileged B state."""

    def __init__(
        self,
        stats: dict,
        width: int = 160,
        mem_dim: int = 32,
        hist_len: int = 8,
        encoder_hidden: int = 64,
        shuffle_c_obs_hist: bool = False,
    ):
        super().__init__()
        self.kind = "eagm_history_c_shuffled" if shuffle_c_obs_hist else "eagm_history_c"
        self.hist_len = hist_len
        self.uses_c_obs_hist = True
        self.shuffle_c_obs_hist = shuffle_c_obs_hist
        self.encoder = HistoryEncoderC(28, 14, 29, hidden=encoder_hidden, mem=mem_dim)
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

    def encode(self, obs_hist, act_hist, c_obs_hist):
        if self.shuffle_c_obs_hist and c_obs_hist is not None:
            # Break temporal alignment while keeping the same feature volume.
            if c_obs_hist.dim() == 3:
                # [B, T, D] — permute time per batch element with a fixed reverse (deterministic)
                c_obs_hist = torch.flip(c_obs_hist, dims=[1])
            else:
                c_obs_hist = torch.flip(c_obs_hist, dims=[0])
        return self.encoder(obs_hist, act_hist, c_obs_hist)

    def command(self, c_obs, a, mem):
        u0 = u0_from_cobs(c_obs, a, self.lower, self.upper, float(self.alpha))
        return u0 + self.delta(torch.cat([mem, a], -1))

    def forward(self, x, c_obs, a, obs_hist, act_hist, c_obs_hist):
        mem = self.encode(obs_hist, act_hist, c_obs_hist)
        u_hat = self.command(c_obs, a, mem)
        xx = (x - self.xmean) / self.xstd
        uu = (u_hat - self.umean) / self.ustd
        y = x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean
        return y, u_hat, mem


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def match_hist_encoder_hidden_to_histc(
    stats: dict,
    *,
    hist_len: int = 8,
    width: int = 160,
    mem_dim: int = 32,
    histc_hidden: int = 64,
) -> int:
    """Find GRU hidden size so EAGMHistory params ≈ EAGMHistoryC(histc_hidden)."""
    target = count_params(EAGMHistoryC(stats, width=width, mem_dim=mem_dim, hist_len=hist_len, encoder_hidden=histc_hidden))
    best_h, best_diff = 64, abs(count_params(EAGMHistory(stats, width=width, mem_dim=mem_dim, hist_len=hist_len, encoder_hidden=64)) - target)
    for h in range(64, 200):
        n = count_params(EAGMHistory(stats, width=width, mem_dim=mem_dim, hist_len=hist_len, encoder_hidden=h))
        d = abs(n - target)
        if d < best_diff:
            best_h, best_diff = h, d
        if n >= target and d <= best_diff:
            best_h = h
            break
    return best_h


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
