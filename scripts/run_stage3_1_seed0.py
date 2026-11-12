#!/usr/bin/env python3
"""Stage 3.1 seed-0 fair re-run after protocol repair.

Models:
  - Mc (Controller-A 57-D memory; no B hidden state)
  - Mu one-step privileged oracle (dyn upper bound only; no policy)
  - EAGM-Base / EAGM-Dir (Markov residual)
  - EAGM-History / EAGM-History+Dir (GRU history)

Does not overwrite deprecated stage3_b_* runs.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

EAGM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EAGM_ROOT))
sys.path.insert(0, str(EAGM_ROOT / "vendor" / "legacy_sim"))

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from models.eagm import EAGMBase, EAGMHistory, McWorld, MuOneStepOracle
from training.dataset import (
    collect_trajectories,
    fit_stats,
    flatten_split,
    make_split,
    pick_starts,
)
from training.dir_loss import huber_dir_loss
from training.rollout import imagined_eagm, imagined_mc, physical_return

DTYPE = torch.float32
H_IMAG = 8
H_EVAL = 16
HIST = 8


def detect_device(prefer_cuda: bool = False):
    """Stage-3.1 default is CPU: tiny MLPs + frequent FD labels; CUDA optional.

    Dir/imagined graphs mix NumPy starts with autograd; keeping one device avoids
    silent H2D semantics changes. GPU throughput is reported separately.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sim_grad_masked(env, start, eps=0.05, scales=(0.5, 1.0, 2.0), horizon=H_IMAG):
    out, stable = [], []
    steps = 0
    for i in range(28):
        e = np.zeros(28)
        e[i] = 1.0
        vals = []
        for sc in scales:
            ep = eps * sc
            plus, _ = physical_return(env, ep * e, start, horizon=horizon)
            minus, _ = physical_return(env, -ep * e, start, horizon=horizon)
            vals.append((plus - minus) / (2 * ep))
            steps += 2 * horizon
        v = vals[1]
        out.append(v)
        spread = max(vals) - min(vals)
        stable.append(spread <= max(1e-9, 0.1 * abs(v)))
    return np.asarray(out), np.asarray(stable, dtype=np.float64), steps


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def balanced_dir_acc(g_hat, g_true, mask, eps=1e-12):
    """Sign agreement on stable non-near-zero components, balanced pos/neg."""
    m = (mask > 0.5) & (np.abs(g_true) > eps)
    if m.sum() == 0:
        return float("nan")
    agree = np.sign(g_hat[m]) == np.sign(g_true[m])
    pos = g_true[m] > 0
    neg = ~pos
    scores = []
    if pos.any():
        scores.append(agree[pos].mean())
    if neg.any():
        scores.append(agree[neg].mean())
    return float(np.mean(scores))


def pack_rows(rows, device, with_hist=True):
    keys = ["x", "y", "a", "c", "c_obs", "u_exec"]
    out = {k: torch.as_tensor(np.stack([r[k] for r in rows]), dtype=DTYPE, device=device) for k in keys}
    if with_hist:
        out["x_hist"] = torch.as_tensor(np.stack([r["x_hist"] for r in rows]), dtype=DTYPE, device=device)
        out["a_hist"] = torch.as_tensor(np.stack([r["a_hist"] for r in rows]), dtype=DTYPE, device=device)
    return out


def train_model(
    model,
    kind,
    tr,
    device,
    updates,
    *,
    dir_pack=None,
    lam_u=1.0,
    lam_d=0.1,
    use_history=False,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = []
    n = tr["x"].shape[0]
    dir_steps_used = 0
    for it in range(updates):
        b = torch.randint(0, n, (128,), device=device)
        if kind == "mc":
            pred = model(tr["x"][b], tr["c"][b], tr["a"][b])
            loss = (((pred - tr["y"][b]) / model.ystd) ** 2).mean()
        elif kind == "mu":
            pred = model.forward_u(tr["x"][b], tr["u_exec"][b])
            loss = (((pred - tr["y"][b]) / model.ystd) ** 2).mean()
        else:
            if use_history:
                y_hat, u_hat, _ = model(
                    tr["x"][b], tr["c_obs"][b], tr["a"][b], tr["x_hist"][b], tr["a_hist"][b]
                )
            else:
                y_hat, u_hat = model(tr["x"][b], tr["c_obs"][b], tr["a"][b])
            dyn = (((y_hat - tr["y"][b]) / model.ystd) ** 2).mean()
            cmd = (((u_hat - tr["u_exec"][b]) / model.ustd) ** 2).mean()
            loss = dyn + lam_u * cmd
            if dir_pack is not None and (it % 10 == 0):
                starts, grads, masks, s0, thetas = dir_pack
                j = (it // 10) % len(starts)
                # Alternate θ=0 and a small policy-region θ
                th0 = thetas[j % len(thetas)]
                th = torch.as_tensor(th0, dtype=DTYPE, device=device).clone().requires_grad_(True)
                J = imagined_eagm(model, th, starts[j], horizon=H_IMAG, use_history=use_history)
                g = torch.autograd.grad(J, th, create_graph=True)[0]
                dloss = huber_dir_loss(
                    g.unsqueeze(0),
                    grads[j : j + 1].to(device),
                    masks[j : j + 1].to(device),
                    s0=s0,
                )
                if dloss is not None:
                    loss = loss + lam_d * dloss
                    dir_steps_used += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (it + 1) % 200 == 0:
            log.append(dict(update=it + 1, loss=float(loss.detach().cpu())))
            print(f"  {kind:16s} it {it+1} loss {float(loss.detach()):.5f}", flush=True)
    return log, dir_steps_used


def policy_train(model, starts, *, use_history, updates=50, is_mc=False, device=torch.device("cpu")):
    # Policy Adam stays on CPU tensors inside imagined_*; model may be CUDA —
    # for fairness/simplicity run policy on CPU copy if needed.
    m = model.cpu() if next(model.parameters()).is_cuda else model
    theta = torch.zeros(28, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.03)
    for _ in range(updates):
        if is_mc:
            loss = -torch.stack([imagined_mc(m, theta, s, horizon=H_IMAG) for s in starts]).mean()
        else:
            loss = -torch.stack(
                [imagined_eagm(m, theta, s, horizon=H_IMAG, use_history=use_history) for s in starts]
            ).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([theta], 1.0)
        opt.step()
        with torch.no_grad():
            theta.clamp_(-1, 1)
    return theta.detach().numpy().copy()


def main():
    ap_updates = [1000, 3000]
    seed = 0
    device = detect_device(prefer_cuda=False)
    cfg = ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25)
    env = PartialUnknownArmEnv(cfg)

    out_root = EAGM_ROOT / "runs" / "stage3_1_seed0"
    out_root.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    print("collect 24 trajectories...", flush=True)
    episodes = collect_trajectories(
        env, n_episodes=24, horizon=200, seed=seed, hist_len=HIST, config_id="B_fixed_105"
    )
    split = make_split(24, seed=seed, n_train=12, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    val_rows = flatten_split(episodes, split.val_eps)
    test_rows = flatten_split(episodes, split.test_eps)
    stats = fit_stats(train_rows)

    # Held-out Controller B config episodes (separate generation)
    print("collect held-out B configs...", flush=True)
    env_hold = PartialUnknownArmEnv(
        ControllerBConfig(
            episode_randomize=True,
            gain_range=(0.92, 1.08),
            extra_lag_choices=(0, 1, 2),
            rate_limit_range=(0.03, 0.08),
            smooth_beta_range=(0.0, 0.35),
        )
    )
    hold_eps = collect_trajectories(
        env_hold, n_episodes=6, horizon=200, seed=seed + 77, hist_len=HIST, config_id="B_heldout"
    )
    hold_rows = flatten_split(hold_eps, list(range(6)))

    meta = dict(
        seed=seed,
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        torch=torch.__version__,
        host=platform.uname()._asdict(),
        controller_b=cfg.__dict__,
        split=dict(train=split.train_eps, val=split.val_eps, test=split.test_eps),
        n_train_rows=len(train_rows),
        n_val_rows=len(val_rows),
        n_test_rows=len(test_rows),
        hist_len=HIST,
        notes=dict(
            mc="Controller-A 57-D memory only; no B hidden FIFO/gain/lag state",
            mu="Privileged one-step oracle on u_exec; not used for policy",
            eagm="Deployable: x, a, c_obs=(target,filter,delay), optional x/a history; no known_config_vector",
            deprecated_runs=["runs/stage3_b_min_seed0", "runs/stage3_b_dir_seed0"],
        ),
    )
    (out_root / "environment.json").write_text(json.dumps(meta, indent=2, default=str))

    # Direction labels from TRAIN starts only; mix θ=0 and small random θ region
    # Use all delays so starts span many train/test episodes (delay was random per ep).
    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    test_starts = pick_starts(test_rows, delay=None, max_per_episode=2, max_total=12)
    hold_starts = pick_starts(hold_rows, delay=None, max_per_episode=2, max_total=12)
    print(f"dir starts {len(dir_starts)} test starts {len(test_starts)}", flush=True)

    dir_g, dir_ok, sim_steps = [], [], 0
    thetas = [np.zeros(28)]
    rng = np.random.default_rng(seed + 5)
    for _ in range(3):
        thetas.append(rng.normal(0, 0.3, size=28).clip(-1, 1))
    for s in dir_starts:
        # Label at θ=0 for FD stability (policy-region θ used in training probe)
        g, ok, st = sim_grad_masked(env, s)
        dir_g.append(g)
        dir_ok.append(ok)
        sim_steps += st
    s0 = float(np.median(np.abs(np.stack(dir_g)))) or 1.0
    grads_t = torch.as_tensor(np.stack(dir_g), dtype=DTYPE)
    masks_t = torch.as_tensor(np.stack(dir_ok), dtype=DTYPE)
    dir_pack = (dir_starts, grads_t, masks_t, s0, thetas)

    test_g, test_ok, st2 = [], [], 0
    for s in test_starts:
        g, ok, st = sim_grad_masked(env, s)
        test_g.append(g)
        test_ok.append(ok)
        st2 += st
    sim_steps += st2

    tr = pack_rows(train_rows, device)
    te = pack_rows(test_rows, device)
    hold = pack_rows(hold_rows, device)

    gpu_peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    all_results = {}
    for updates in ap_updates:
        print(f"==== updates {updates} ====", flush=True)
        models = {
            "mc": McWorld(stats),
            "mu_onestep": MuOneStepOracle(stats),
            "eagm_base": EAGMBase(stats),
            "eagm_dir": EAGMBase(stats),
            "eagm_hist": EAGMHistory(stats, hist_len=HIST),
            "eagm_hist_dir": EAGMHistory(stats, hist_len=HIST),
        }
        logs = {}
        dir_counts = {}
        t0 = time.time()
        for name, m in models.items():
            use_h = name.startswith("eagm_hist")
            use_d = name.endswith("dir") or name.endswith("hist_dir")
            # fix naming: eagm_hist_dir ends with dir
            use_d = name in ("eagm_dir", "eagm_hist_dir")
            kind = (
                "mc" if name == "mc" else "mu" if name == "mu_onestep"
                else ("eagm_hist_dir" if use_d and use_h else "eagm_hist" if use_h else "eagm_dir" if use_d else "eagm_base")
            )
            print(f"train {name}...", flush=True)
            log, dused = train_model(
                m,
                "mc" if name == "mc" else "mu" if name == "mu_onestep" else name,
                tr,
                device,
                updates,
                dir_pack=dir_pack if use_d else None,
                use_history=use_h,
            )
            logs[name] = log
            dir_counts[name] = dused
            # move back for eval convenience
            models[name] = m.cpu() if device.type == "cuda" else m

        # Dynamics / cmd on test
        with torch.no_grad():
            te_cpu = {k: v.cpu() for k, v in te.items()}
            hold_cpu = {k: v.cpu() for k, v in hold.items()}
            dyn, cmd = {}, {}
            mc = models["mc"]
            dyn["mc"] = (((mc(te_cpu["x"], te_cpu["c"], te_cpu["a"]) - te_cpu["y"]) / mc.ystd) ** 2).mean().item()
            mu = models["mu_onestep"]
            dyn["mu_onestep"] = (((mu.forward_u(te_cpu["x"], te_cpu["u_exec"]) - te_cpu["y"]) / mu.ystd) ** 2).mean().item()
            for name, use_h in [("eagm_base", False), ("eagm_dir", False), ("eagm_hist", True), ("eagm_hist_dir", True)]:
                m = models[name]
                if use_h:
                    y, u, _ = m(te_cpu["x"], te_cpu["c_obs"], te_cpu["a"], te_cpu["x_hist"], te_cpu["a_hist"])
                else:
                    y, u = m(te_cpu["x"], te_cpu["c_obs"], te_cpu["a"])
                dyn[name] = (((y - te_cpu["y"]) / m.ystd) ** 2).mean().item()
                cmd[name] = (u - te_cpu["u_exec"]).abs().mean().item()
            # held-out dyn for deployable models
            dyn_hold = {}
            for name, use_h in [("eagm_base", False), ("eagm_dir", False), ("eagm_hist", True), ("eagm_hist_dir", True)]:
                m = models[name]
                if use_h:
                    y, u, _ = m(hold_cpu["x"], hold_cpu["c_obs"], hold_cpu["a"], hold_cpu["x_hist"], hold_cpu["a_hist"])
                else:
                    y, u = m(hold_cpu["x"], hold_cpu["c_obs"], hold_cpu["a"])
                dyn_hold[name] = (((y - hold_cpu["y"]) / m.ystd) ** 2).mean().item()

        # Grad metrics on frozen test starts
        cos, bal = {}, {}
        for name, use_h, is_mc in [
            ("mc", False, True),
            ("eagm_base", False, False),
            ("eagm_dir", False, False),
            ("eagm_hist", True, False),
            ("eagm_hist_dir", True, False),
        ]:
            m = models[name]
            vals_c, vals_b = [], []
            for s, g, ok in zip(test_starts, test_g, test_ok):
                th = torch.zeros(28, requires_grad=True)
                if is_mc:
                    J = imagined_mc(m, th, s, horizon=H_IMAG)
                else:
                    J = imagined_eagm(m, th, s, horizon=H_IMAG, use_history=use_h)
                gh = torch.autograd.grad(J, th)[0].detach().numpy()
                vals_c.append(cosine(gh, g))
                vals_b.append(balanced_dir_acc(gh, g, ok))
            cos[name] = float(np.median(vals_c))
            bal[name] = float(np.nanmedian(vals_b))

        # Policy returns (deployable + Mc only; NOT Mu)
        print("policy...", flush=True)
        rets, rets_hold = {}, {}
        train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
        for name, use_h, is_mc in [
            ("mc", False, True),
            ("eagm_base", False, False),
            ("eagm_dir", False, False),
            ("eagm_hist", True, False),
            ("eagm_hist_dir", True, False),
        ]:
            theta = policy_train(
                models[name], train_pol, use_history=use_h, updates=50, is_mc=is_mc
            )
            rets[name] = float(np.mean([physical_return(env, theta, s, horizon=H_EVAL)[0] for s in test_starts]))
            rets_hold[name] = float(
                np.mean([physical_return(env_hold, theta, s, horizon=H_EVAL)[0] for s in hold_starts])
            )
            torch.save(
                {"state": models[name].state_dict(), "stats": stats, "kind": name, "updates": updates},
                out_root / f"{name}_u{updates}_seed0.pt",
            )

        payload = dict(
            updates=updates,
            wall_train_s=time.time() - t0,
            dynamics_nmse_test=dyn,
            dynamics_nmse_heldout=dyn_hold,
            cmd_mae_test=cmd,
            cosine_median_test=cos,
            balanced_dir_acc_test=bal,
            policy_return_test=rets,
            policy_return_heldout=rets_hold,
            dir_loss_applications=dir_counts,
            sim_steps_direction_labels=sim_steps,
            logs=logs,
            n_policy_test_starts=len(test_starts),
            n_policy_hold_starts=len(hold_starts),
        )
        all_results[str(updates)] = payload
        (out_root / f"results_u{updates}.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps({
            "updates": updates,
            "dyn": dyn,
            "cmd": cmd,
            "cos": cos,
            "ret": rets,
            "ret_hold": rets_hold,
        }, indent=2), flush=True)

    if device.type == "cuda":
        gpu_peak = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    summary = dict(
        environment=meta,
        results=all_results,
        gpu_peak_mem_mb=gpu_peak,
        wall_total_s=time.time() - t_all,
        stage4_gate="pending_report",
    )
    (out_root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print("wrote", out_root, "wall", round(summary["wall_total_s"], 1), flush=True)


if __name__ == "__main__":
    main()
