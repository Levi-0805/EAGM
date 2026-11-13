"""Stage-6 evaluation metrics: paired ΔR fail rates, budget accounting, saturation."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(a @ b / (na * nb))


def balanced_dir_acc(g_hat: np.ndarray, g_true: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    g_hat = np.asarray(g_hat, dtype=np.float64).ravel()
    g_true = np.asarray(g_true, dtype=np.float64).ravel()
    if mask is None:
        mask = np.ones_like(g_true)
    m = (mask > 0.5) & (np.abs(g_true) > 1e-12)
    if not m.any():
        return 0.0
    return float((np.sign(g_hat[m]) == np.sign(g_true[m])).mean())


def directional_error(g_hat: np.ndarray, g_true: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    g_hat = np.asarray(g_hat, dtype=np.float64).ravel()
    g_true = np.asarray(g_true, dtype=np.float64).ravel()
    if mask is None:
        mask = np.ones_like(g_true)
    m = mask > 0.5
    if not m.any():
        return float("nan")
    a, b = g_hat[m], g_true[m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.linalg.norm(a / na - b / nb))


def gradient_quality(g_hat: np.ndarray, g_true: np.ndarray, mask: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Full-dimensional and stable-mask gradient diagnostics."""
    g_hat = np.asarray(g_hat, dtype=np.float64).ravel()
    g_true = np.asarray(g_true, dtype=np.float64).ravel()
    if mask is None:
        mask = np.ones_like(g_true, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64).ravel()
    n_stable = int((mask > 0.5).sum())
    return dict(
        cosine_full=cosine(g_hat, g_true),
        cosine_stable=cosine(g_hat[mask > 0.5], g_true[mask > 0.5]) if n_stable > 0 else float("nan"),
        balanced_dir_acc=balanced_dir_acc(g_hat, g_true, mask),
        directional_error=directional_error(g_hat, g_true, mask),
        n_stable_dims=n_stable,
        n_dims=int(g_true.size),
        grad_norm_hat=float(np.linalg.norm(g_hat)),
        grad_norm_true=float(np.linalg.norm(g_true)),
    )


def theta_param_sat_frac(theta: np.ndarray, bound: float = 1.0, eps: float = 1e-8) -> float:
    th = np.asarray(theta, dtype=np.float64).ravel()
    return float((np.abs(th) >= bound - eps).mean())


def paired_delta_returns(
    returns_final: list[float] | np.ndarray,
    returns_theta0: list[float] | np.ndarray,
) -> np.ndarray:
    """ΔR_i = R_i(θ_final) − R_i(θ_0) on identical starts (paired)."""
    a = np.asarray(returns_final, dtype=np.float64).ravel()
    b = np.asarray(returns_theta0, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired ΔR length mismatch: {a.shape} vs {b.shape}")
    return a - b


def fail_rate_vs_theta0(
    returns_final: list[float] | np.ndarray,
    returns_theta0: list[float] | np.ndarray,
    *,
    eps: float = 0.0,
) -> dict[str, Any]:
    """FailRate = mean 1[ΔR_i < −ε]. Higher return is better."""
    deltas = paired_delta_returns(returns_final, returns_theta0)
    fail = deltas < -float(eps)
    return dict(
        fail_rate=float(fail.mean()) if len(deltas) else float("nan"),
        n=int(len(deltas)),
        n_fail=int(fail.sum()),
        delta_mean=float(deltas.mean()) if len(deltas) else float("nan"),
        delta_median=float(np.median(deltas)) if len(deltas) else float("nan"),
        n_improved=int((deltas > 0).sum()),
        eps=float(eps),
        definition="FailRate = mean 1[R(θ_final)-R(θ_0) < -eps] on paired starts",
    )


def aggregate_per_config_paired(
    per_start_final: list[dict],
    per_start_theta0: list[dict],
    *,
    eps: float = 0.0,
) -> dict[str, Any]:
    """Group paired starts by config_id; require same order / start_index."""
    if len(per_start_final) != len(per_start_theta0):
        raise ValueError("final and theta0 per_start lists must be same length")
    by_cfg: dict[str, list[tuple[float, float]]] = {}
    for a, b in zip(per_start_final, per_start_theta0):
        if a.get("start_index") != b.get("start_index"):
            raise ValueError("start_index mismatch between final and theta0 evals")
        cid = str(a.get("config_id") or "unk")
        by_cfg.setdefault(cid, []).append((float(a["return_"]), float(b["return_"])))
    out = {}
    for cid, pairs in by_cfg.items():
        rf = [p[0] for p in pairs]
        r0 = [p[1] for p in pairs]
        fr = fail_rate_vs_theta0(rf, r0, eps=eps)
        out[cid] = dict(
            mean_return_final=float(np.mean(rf)),
            mean_return_theta0=float(np.mean(r0)),
            mean_delta=fr["delta_mean"],
            fail_rate=fr["fail_rate"],
            n=fr["n"],
            n_improved=fr["n_improved"],
        )
    return out


def stage5e_budget_breakdown(
    *,
    n_train_eps: int = 24,
    n_val_eps: int = 6,
    n_iid_hold_eps: int = 6,
    horizon: int = 200,
    dir_label_train_steps: int = 0,
    ood_fd_diagnostic_steps: int = 0,
) -> dict[str, Any]:
    """Strict budget accounting for Stage 5E / Stage 6 OOD protocols."""
    train_dyn = n_train_eps * horizon
    val_dyn = n_val_eps * horizon
    iid_dyn = n_iid_hold_eps * horizon
    collect_total = train_dyn + val_dyn + iid_dyn
    return dict(
        dynamics_train_steps=train_dyn,
        dynamics_val_steps=val_dyn,
        dynamics_iid_holdout_steps=iid_dyn,
        dynamics_collect_total_steps=collect_total,
        dir_label_train_steps=int(dir_label_train_steps),
        # Train interaction budget excludes val/iid/test FD diagnostics
        train_interaction_budget=train_dyn + int(dir_label_train_steps),
        ood_fd_diagnostic_steps=int(ood_fd_diagnostic_steps),
        note=(
            "Train budget = train dynamics + train dir labels only. "
            "Val / IID holdout / OOD FD diagnostics are listed separately."
        ),
    )
