# Stage 3 Protocol Audit (P0/P1)

Date: 2026-11-12  
Branch: `stage3-protocol-repair`  
Old exploratory runs marked deprecated under `runs/stage3_b_*/DEPRECATED.md`.

## Confirmed bugs in Stage 3 exploratory code

### 1. Command label timing (P0) — CONFIRMED

In `run_stage3_b_dir_seed0.py` / `run_stage3_b_min_seed0.py`, `u_true` was read **before** `env.step(a)`.
Controller B computes the executed command inside `step()`, so labels were the **previous** step’s command.

**Fix:** `training/dataset.py::collect_trajectories` records:

```
observe/snapshot → step(a_t) → u_exec = last_true_command (== data.ctrl)
```

Unit tests: `tests/test_protocol_p0.py` (lag 0/1/2, gain, replay).

### 2. Imagined vs physical reward (P0) — CONFIRMED

`imagined_eagm` used only position error and **no** `0.99` discount; `experiment.physical` uses pos+vel+Δa with discount.

**Fix:** `training/reward.py` + `training/rollout.py` shared by EAGM imagined, Mc imagined, and physical eval.

### 3. Transition-level split leakage (P0) — CONFIRMED

Frames from the same episode were shuffled into train/test.

**Fix:** trajectory-level `make_split` (12/6/6 episodes). Starts picked with `max_per_episode` across episodes.

### 4. Mu / Mc fairness (P0) — CONFIRMED

- Mu was not used for policy in the last Stage-3 script (good), but dyn labels used mistimed `u_true`.
- Mc uses Controller-**A** 57-D memory only; it does **not** observe B hidden lag/gain/smooth state. Documented in results meta.

**Fix:** `MuOneStepOracle` is one-step dyn upper bound only; no policy comparison.

### 5. History encoder unused (P1) — CONFIRMED

Stage-3 `EAGMBase` was Markov on `(c_obs,a)`. Must not be called “history-aware EAGM”.

**Fix:** `EAGMHistory` GRU + `EAGM-History` / `EAGM-History+Dir` in Stage 3.1.

### 6. `known_config_vector` leakage risk (P1)

Exposes `(gain, extra_lag, rate_limit, smooth_beta)`. Deployable Stage 3.1 models **do not** consume it.

### 7. CUDA (P1)

Stage-3 scripts trained on CPU by default (no `.cuda()`). Stage 3.1 keeps **CPU FP32** for protocol runs; `scripts/bench_stage3_device.py` records CPU vs CUDA matmul throughput. MuJoCo remains CPU.

### 8. Dir mask (P1)

`huber_dir_loss` now returns `None` when no stable components (skip term). Training samples alternate θ probes including non-zero θ vectors; FD labels still at θ=0 for stability.

## Tests

```bash
python tests/test_protocol_p0.py
```

All P0 checks passed on this host before Stage 3.1 retrain.
