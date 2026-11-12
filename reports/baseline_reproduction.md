# Stage 1: Baseline reproduction

Date: 2026-11-12  
Env: `conda activate eagm_gpu` (torch 2.11.0+cu128, mujoco 3.5.0, numpy 2.1.3)  
Device for Stage-1 training: **CPU FP32** (match legacy path)

## Migration blockers resolved

1. **Missing in-tree OpenArm source** matching Windows sha `3ec76ba7...`. Local HIL-OpenArm v2 is `ac1aff0d...`.
2. **Snapshot `model_sha256` embeds Windows meshdir**. Solved by `controllers/compatible_arm_env.py`: load archived `pilot_20261101/effective_model.xml` with Linux meshdir, accept pilot sha on restore.
3. **NumPy 2 pickles** require dedicated `eagm_gpu` env (not `isaaclab50` NumPy 1.26).

`CVPR2027/` untouched. Legacy `runs/` not overwritten; new outputs under `eagm_gpu/runs/`.

## Seed 0 @ 1,000 updates

Command:

```bash
python scripts/run_stage1_seed0_1k.py
```

Result dir: `runs/stage1_seed0_1k/` (wall **104.2 s**)

| Model | Cosine (new) | Cosine (archive) | Δ cos | Return (new) | Return (archive) | Δ ret |
|---|---:|---:|---:|---:|---:|---:|
| Mx | 0.5653 | 0.5653 | 4.8e-8 | -0.00264866 | -0.00264866 | 3.0e-12 |
| Mc | 0.8917 | 0.8917 | 1.2e-8 | -0.00116300 | -0.00116300 | -3.7e-11 |
| Mu | 0.6527 | 0.6527 | 4.6e-7 | -0.00194577 | -0.00194578 | 8.8e-10 |
| Mu+Dir | 0.6626 | 0.6626 | -1.3e-8 | -0.00191554 | -0.00191554 | -1.0e-10 |

**Verdict:** Seed-0 1k reproduction is within tolerance by a large margin (essentially bit-stable for these metrics). Gate to continue Stage 1 five-seed / 3k: **PASS**.

## Five-seed @ 1,000 updates

Command: `python scripts/run_stage1_1k5.py`  
Result: `runs/stage1_1k5/` (wall **532.4 s**)

| Model | Cosine mean (5 seeds) | Return mean | max ‖Δcos‖ vs archive | max ‖Δret‖ vs archive |
|---|---:|---:|---:|---:|
| Mx | 0.5618 | -0.00269259 | 4.8e-8 | 9.2e-11 |
| Mc | 0.8952 | -0.00110116 | 2.0e-7 | 4.0e-11 |
| Mu | 0.6276 | -0.00215557 | 2.0e-6 | 6.4e-9 |
| Mu+Dir | 0.6404 | -0.00196885 | 1.3e-7 | 3.9e-10 |

Paired \(M_u+\mathrm{Dir}-M_u\) return mean **+1.867×10⁻⁴**, **4/5** seeds positive — matches the archived Stage-2 stop-condition table.

**Verdict:** Five-seed 1k gate **PASS**.

## Five-seed @ 3,000 updates

Command: `python scripts/run_stage1_mudir3k.py`  
Result: `runs/stage1_mudir3k/` (wall **772.4 s**)

| Model | Cosine mean | Return mean | max ‖Δcos‖ | max ‖Δret‖ |
|---|---:|---:|---:|---:|
| Mc 3k | 0.9765 | -0.00098508 | 5.3e-8 | 3.0e-11 |
| Mu 3k | 0.9905 | -0.00096390 | 1.8e-7 | 5.1e-11 |
| Mu+Dir 3k | 0.9905 | -0.00096575 | 6.8e-8 | 5.6e-11 |

Paired \(M_u+\mathrm{Dir}-M_u\) return mean **−1.846×10⁻⁶**, **1/5** seeds positive — matches archived stop condition (−1.85×10⁻⁶, 1/5).

**Verdict:** Five-seed 3k gate **PASS**. Stage 1 complete. Exact Mu+Dir remains a negative result under known control law; do not scale it as a new method.

## Notes

- MuJoCo stepping remains CPU (~2.5k cps). GPU not required for this MLP Stage-1 path.
- Do **not** treat exact Mu+Dir 3k re-runs as a new method contribution; they are regression gates only.
- Next: Controller B / EAGM-Base+Dir under matched budget (`scripts/run_stage3_b_min_seed0.py` smoke already collected).
