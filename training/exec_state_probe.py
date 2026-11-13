"""Analysis-only linear probes: does encoder memory predict privileged B scalars?

Never used as model inputs. Labels come from privileged_exec_state().
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class LinearExecProbe(nn.Module):
    """Predict [gain, extra_lag, rate_limit, smooth_beta] from memory vector."""

    def __init__(self, mem_dim: int, out_dim: int = 4):
        super().__init__()
        self.lin = nn.Linear(mem_dim, out_dim)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        return self.lin(mem)


@torch.no_grad()
def encode_batch(model: nn.Module, batch: dict, device=torch.device("cpu")) -> torch.Tensor:
    model.eval()
    x = batch["x"].to(device)
    c_obs = batch["c_obs"].to(device)
    a = batch["a"].to(device)
    xh = batch["x_hist"].to(device)
    ah = batch["a_hist"].to(device)
    if getattr(model, "uses_c_obs_hist", False):
        ch = batch["c_obs_hist"].to(device)
        _, _, mem = model(x, c_obs, a, xh, ah, ch)
    else:
        _, _, mem = model(x, c_obs, a, xh, ah)
    return mem.detach()


def fit_exec_state_probe(
    model: nn.Module,
    tr: dict,
    *,
    updates: int = 800,
    batch_size: int = 128,
    lr: float = 1e-2,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Train linear probe on frozen encoder memories → exec_state_scalars."""
    if "exec_state_scalars" not in tr:
        return dict(ok=False, reason="missing exec_state_scalars in train pack")
    torch.manual_seed(seed)
    model.to(device)
    # Infer mem dim
    with torch.no_grad():
        b0 = {k: tr[k][:8] for k in ("x", "c_obs", "a", "x_hist", "a_hist") if k in tr}
        if "c_obs_hist" in tr:
            b0["c_obs_hist"] = tr["c_obs_hist"][:8]
        mem0 = encode_batch(model, b0, device)
        mem_dim = int(mem0.shape[-1])
    probe = LinearExecProbe(mem_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    n = tr["x"].shape[0]
    y_all = tr["exec_state_scalars"]
    # Standardize targets for stable MSE
    y_mean = y_all.mean(0)
    y_std = torch.clamp(y_all.std(0), min=1e-3)
    log = []
    for it in range(updates):
        b = torch.randint(0, n, (batch_size,), generator=gen)
        batch = {k: tr[k][b] for k in ("x", "c_obs", "a", "x_hist", "a_hist")}
        if "c_obs_hist" in tr:
            batch["c_obs_hist"] = tr["c_obs_hist"][b]
        mem = encode_batch(model, batch, device)
        pred = probe(mem)
        tgt = ((y_all[b] - y_mean) / y_std).to(device)
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % 200 == 0:
            log.append(dict(update=it + 1, loss=float(loss.detach())))
    # Holdout: last 20% rows (deterministic slice, not random to keep simple)
    n_te = max(32, n // 5)
    te = {k: tr[k][-n_te:] for k in ("x", "c_obs", "a", "x_hist", "a_hist")}
    if "c_obs_hist" in tr:
        te["c_obs_hist"] = tr["c_obs_hist"][-n_te:]
    mem = encode_batch(model, te, device)
    with torch.no_grad():
        pred = probe(mem).cpu()
        tgt = (y_all[-n_te:] - y_mean) / y_std
        mse = ((pred - tgt) ** 2).mean(0)
        # Correlation per scalar
        corrs = []
        for i in range(4):
            a = pred[:, i].numpy()
            b = tgt[:, i].numpy()
            if a.std() < 1e-8 or b.std() < 1e-8:
                corrs.append(0.0)
            else:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    names = ["gain", "extra_lag", "rate_limit", "smooth_beta"]
    return dict(
        ok=True,
        mem_dim=mem_dim,
        train_log=log,
        mse_z=dict(zip(names, [float(x) for x in mse])),
        corr=dict(zip(names, corrs)),
        mean_abs_corr=float(np.mean(np.abs(corrs))),
        note="Probe is analysis-only; privileged labels never enter the world model.",
    )
