"""Fair non-bottleneck baselines for Stage 5C."""
from __future__ import annotations

import torch
from torch import nn


class ObservableMemoryMLP(nn.Module):
    """Same observables as EAGM (x, a, c_obs) → next state; no command bottleneck.

    Returns (y, u_placeholder) so imagined_eagm unpacking stays valid.
    u_placeholder is zeros and is never used as an execution bottleneck.
    """

    def __init__(self, stats: dict, width: int = 160):
        super().__init__()
        self.kind = "obs_memory_mlp"
        self.net = nn.Sequential(
            nn.Linear(71, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for name, val in stats.items():
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float32))

    def forward(self, x, c_obs, a):
        xx = (x - self.xmean) / self.xstd
        feat = torch.cat([xx, a, c_obs], -1)
        y = x + self.net(feat) * self.ystd + self.ymean
        u_placeholder = torch.zeros(x.shape[0], 14, device=x.device, dtype=x.dtype)
        return y, u_placeholder


class AuxCommandMLP(nn.Module):
    """Predict next state from (x,a,c_obs) + auxiliary û head supervised by u_exec.

    Capacity matched to EAGM-Base. No U0+Δ structured command bottleneck.
    """

    def __init__(self, stats: dict, width: int = 160):
        super().__init__()
        self.kind = "aux_command_mlp"
        self.body = nn.Sequential(
            nn.Linear(71, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.dyn_head = nn.Linear(width, 28)
        self.cmd_head = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 14))
        for name, val in stats.items():
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float32))

    def forward(self, x, c_obs, a):
        xx = (x - self.xmean) / self.xstd
        h = self.body(torch.cat([xx, a, c_obs], -1))
        y = x + self.dyn_head(h) * self.ystd + self.ymean
        u_hat = self.cmd_head(h)
        return y, u_hat


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
