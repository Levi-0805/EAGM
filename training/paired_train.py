"""Paired world-model training: Base/Dir share init + minibatch stream."""
from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from training.dir_loss import huber_dir_loss
from training.rollout import imagined_eagm

DTYPE = torch.float32


def clone_model(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    return m


def train_pair(
    model_base: nn.Module,
    model_dir: Optional[nn.Module],
    tr: dict,
    *,
    updates: int,
    use_history: bool,
    dir_labels: Optional[list[dict]] = None,
    lam_u: float = 1.0,
    lam_d: float = 0.1,
    batch_size: int = 128,
    batch_seed: int = 0,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    h_imag: int = 8,
) -> dict[str, Any]:
    """Train Base and optional Dir twin with identical minibatches.

    If model_dir is None, only Base is trained (still consuming the batch stream).
    dir_labels: list of {start, theta, g, mask, s0_scale} — model gradient at same theta.
    """
    model_base.to(device)
    if model_dir is not None:
        model_dir.to(device)
    opt_b = torch.optim.Adam(model_base.parameters(), lr=lr)
    opt_d = torch.optim.Adam(model_dir.parameters(), lr=lr) if model_dir is not None else None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(batch_seed)
    n = tr["x"].shape[0]
    logs_b, logs_d = [], []
    dir_apps = 0
    dir_skips = 0

    def eagm_step(model, b, add_dir: bool, it: int):
        nonlocal dir_apps, dir_skips
        if use_history:
            y_hat, u_hat, _ = model(
                tr["x"][b], tr["c_obs"][b], tr["a"][b], tr["x_hist"][b], tr["a_hist"][b]
            )
        else:
            y_hat, u_hat = model(tr["x"][b], tr["c_obs"][b], tr["a"][b])
        dyn = (((y_hat - tr["y"][b]) / model.ystd) ** 2).mean()
        cmd = (((u_hat - tr["u_exec"][b]) / model.ustd) ** 2).mean()
        loss = dyn + lam_u * cmd
        if add_dir and dir_labels and (it % 10 == 0):
            lab = dir_labels[(it // 10) % len(dir_labels)]
            th = torch.as_tensor(lab["theta"], dtype=DTYPE, device=device).clone().requires_grad_(True)
            J = imagined_eagm(
                model, th, lab["start"], horizon=h_imag, use_history=use_history
            )
            g = torch.autograd.grad(J, th, create_graph=True)[0]
            dloss = huber_dir_loss(
                g.unsqueeze(0),
                torch.as_tensor(lab["g"], dtype=DTYPE, device=device).unsqueeze(0),
                torch.as_tensor(lab["mask"], dtype=DTYPE, device=device).unsqueeze(0),
                s0=lab["s0"],
            )
            if dloss is None:
                dir_skips += 1
            else:
                loss = loss + lam_d * dloss
                dir_apps += 1
        return loss

    for it in range(updates):
        b = torch.randint(0, n, (batch_size,), generator=gen)
        # Base
        loss_b = eagm_step(model_base, b, add_dir=False, it=it)
        opt_b.zero_grad(set_to_none=True)
        loss_b.backward()
        nn.utils.clip_grad_norm_(model_base.parameters(), 5.0)
        opt_b.step()
        # Dir twin (same batch indices)
        if model_dir is not None:
            loss_d = eagm_step(model_dir, b, add_dir=True, it=it)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            nn.utils.clip_grad_norm_(model_dir.parameters(), 5.0)
            opt_d.step()
        if (it + 1) % 200 == 0:
            logs_b.append(dict(update=it + 1, loss=float(loss_b.detach())))
            if model_dir is not None:
                logs_d.append(dict(update=it + 1, loss=float(loss_d.detach())))
            msg = f"    pair it {it+1} base {float(loss_b.detach()):.5f}"
            if model_dir is not None:
                msg += f" dir {float(loss_d.detach()):.5f}"
            print(msg, flush=True)

    return dict(
        log_base=logs_b,
        log_dir=logs_d,
        dir_applications=dir_apps,
        dir_skips=dir_skips,
        batch_seed=batch_seed,
    )


def train_mc(model, tr, updates, batch_seed=0, batch_size=128, lr=1e-3, device=torch.device("cpu")):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(batch_seed)
    n = tr["x"].shape[0]
    log = []
    for it in range(updates):
        b = torch.randint(0, n, (batch_size,), generator=gen)
        pred = model(tr["x"][b], tr["c"][b], tr["a"][b])
        loss = (((pred - tr["y"][b]) / model.ystd) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (it + 1) % 200 == 0:
            log.append(dict(update=it + 1, loss=float(loss.detach())))
            print(f"    mc it {it+1} loss {float(loss.detach()):.5f}", flush=True)
    return log


def train_obs_mlp(model, tr, updates, batch_seed=0, batch_size=128, lr=1e-3, device=torch.device("cpu")):
    """Train ObservableMemoryMLP: dynamics only on (x, c_obs, a)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(batch_seed)
    n = tr["x"].shape[0]
    log = []
    for it in range(updates):
        b = torch.randint(0, n, (batch_size,), generator=gen)
        y_hat, _ = model(tr["x"][b], tr["c_obs"][b], tr["a"][b])
        loss = (((y_hat - tr["y"][b]) / model.ystd) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (it + 1) % 200 == 0:
            log.append(dict(update=it + 1, loss=float(loss.detach())))
            print(f"    obs_mlp it {it+1} loss {float(loss.detach()):.5f}", flush=True)
    return log
