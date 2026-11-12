# EAGM GPU Environment Audit (Stage 0)

Date: 2026-11-12  
Host: `aaa-QiTianM610-N000`  
Workspace: `/home/aaa/桌面/汇报/11月论文/eagm_gpu`  
Legacy sim (read/run, not rewritten): `/home/aaa/桌面/汇报/11月论文/本地仿真预实验`

## 1. Hardware and OS

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (Linux 6.8.0-138-generic x86_64) |
| GPU | 1× NVIDIA GeForce RTX 4060, 8188 MiB |
| Driver | 580.178.04 |
| CUDA (driver) | 13.0 |
| Free GPU memory at audit | ~2.3 GiB free / ~5.5 GiB used by desktop + other apps |

**Note:** GPU memory is partly occupied by display compositor, browsers, FreeCAD, Steam++, and an `isaaclab50` Python process. Small MLP world models fit easily; large visual encoders or big batches may need reclaiming memory first.

## 2. Software stack used for EAGM GPU work

Dedicated conda env: **`eagm_gpu`** (do not upgrade `isaaclab50` in place).

| Package | Version |
|---|---|
| Python | 3.11.18 (`/home/aaa/miniconda3/envs/eagm_gpu`) |
| PyTorch | 2.11.0+cu128 |
| `torch.cuda.is_available()` | **True** |
| CUDA (torch) | 12.8 |
| MuJoCo | 4.0.1 (legacy CPU experiments used 3.5.0; ArmEnv smoke OK) |
| NumPy | 2.1.3 (**required** — `starts.pt` was pickled under NumPy 2.x) |
| Matplotlib | 3.10.8 |

Fallback note: `isaaclab50` has torch 2.7.0+cu128 but NumPy 1.26.0; loading `starts.pt` fails (`numpy._core`). Base conda Python 3.14 has no torch. Windows `.venv` is unusable on Linux.

Legacy lockfile `requirements-observed.txt` recorded `torch==2.10.0+cpu` / `mujoco==3.5.0`. Exact bit-reproducibility vs the old CPU box is **not** expected; Stage 1 must document numerical tolerances.

## 3. Repository / migration status

| Path | Status |
|---|---|
| `CVPR2027/` paper sources + PDFs | Present — **do not modify** |
| `本地仿真预实验/` code + runs | Present |
| `本地仿真预实验/EAGM_archive_20261112/` | Present (scripts, report, runs, MANIFEST) |
| `sim_env.py`, `experiment.py`, EAGM scripts | Present |
| `runs/pilot_20261101/transitions.npz`, `starts.pt` | Present |
| `runs/extended_20261102/alpha05` Mx/Mc 1k ckpts | Present |
| `runs/revision4_20261111/long3000` Mc 3k | Present |
| `runs/revision4_20261111/mu3000` Mu 3k | Present |
| Project-root git | **Absent** |
| `eagm_gpu/` git branch `eagm-gpu` | Created (isolated) |
| OpenArm MJCF in-repo `third_party/...` | **Missing from paper tree** |
| OpenArm asset on machine | Found at `/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml` (88 mesh files) |
| Symlink in `eagm_gpu/third_party/openarm_mujoco/v2` | Created |
| Env override | `OPENARM_MODEL_PATH` required when running from legacy scripts (they also search parents for `third_party/...`) |

### Missing / not migrated

1. No in-tree copy of `third_party/openarm_mujoco` inside the paper folder (resolved via HIL-OpenArm + symlink/env).
2. **Original Windows OpenArm source XML** sha `3ec76ba7...` is not on this host (local v2 is `ac1aff0d...`). Stage 1 uses archived `pilot_20261101/effective_model.xml` via `CompatibleArmEnv` instead — five-seed 1k metrics match archive to ~1e-6.
3. No Linux clone of the old Windows CPU venv; dedicated conda env `eagm_gpu` created.
4. No project-level git history of the CVPR tree (only new `eagm_gpu` repo). Git commits in `eagm_gpu` need a local `user.name`/`user.email` (not set here per policy).
5. Public manipulation stacks for Stage 6 (`robosuite`, ManiSkill, etc.) not audited as installed for EAGM yet.

## 4. Throughput benchmarks (raw JSON: `reports/gpu_bench_raw.json`)

Tiny world-model MLP (~same size class as \(M_u\)): Linear(42→160→160→28).

| Test | Result |
|---|---|
| MuJoCo OpenArm control stepping | **2520 control steps/s** (legacy report ~1890–1903) |
| Train CPU, batch 256, 1000 updates | 289.5 updates/s |
| Train GPU FP32, batch 256, 1000 updates | 429.4 updates/s, peak ~19 MB |
| Train GPU AMP | 357.1 updates/s (slower than FP32 for this tiny net) |
| GPU batch 4096 | 434 updates/s, peak ~32 MB |
| CPU batch 4096 | 48.5 updates/s |
| H2D 10M floats ×20 | ~3.4 GB/s |
| Train with per-step H2D vs resident GPU | 323 vs 420 updates/s |

### Bottleneck interpretation

1. **MuJoCo physics remains CPU-bound.** An NVIDIA GPU does not accelerate serial `mj_step`. This machine is already faster than the old CPU box (~2.5k vs ~1.9k cps).
2. **Neural training of 37k-param MLPs is only moderately GPU-accelerated** at batch 256; AMP does not help. GPU wins more at larger batches / visual encoders.
3. **Direction labels and physical returns are dominated by MuJoCo stepping**, not by GPU matmul. Parallel envs / async collection matter more than AMP for Stage 1–5 OpenArm work.
4. Keep **FP32** for gradient diagnostics unless AMP is proven numerically safe.

## 5. Old EAGM Stage 1–2 stop condition (summary)

From `EAGM_archive_20261112/report/EAGM_阶段1_2报告.md`:

- 1k updates, 5 seeds: \(M_u+\mathrm{Dir}-M_u\) return \(+1.87\times10^{-4}\) (~+8.7%), 4/5 seeds positive.
- 3k updates, 5 seeds: \(M_u+\mathrm{Dir}-M_u\) return \(-1.85\times10^{-6}\), 95% CI crosses 0, 1/5 seeds positive.
- At 3k, \(M_u\) cosine ≈ 0.991; direction loss adds no reliable policy gain; does not beat \(M_c\) reliably.
- Stop condition **triggered**: do not promote EAGM into the CVPR diagnostic paper.
- Audits: exact \(u_t\) reconstructible; FIFO partly redundant; stability mask unused in training; direction labels at \(\theta=0\) not policy-region; learned bottleneck / partial interface **not implemented**.

**Implication for this machine:** GPU does not salvage exact \(M_u+\mathrm{Dir}\) at 3k. New value is Controller B/C (partially unknown execution) + proper masked/policy-region Dir + CF losses + real policy return.

## 6. Code readiness vs task book

| Module | Status in legacy tree |
|---|---|
| ArmEnv + exact controller A | Implemented (`sim_env.py`) |
| \(M_x,M_c,M_h,M_u\), diagnostics | Implemented (`experiment.py`) |
| Exact \(M_u+\mathcal L_{dir}\) pilot | Implemented (`eagm_pilot.py`) — mask **not** applied in loss |
| Stage 2 / 3k scripts | Implemented |
| Learned command bottleneck | **Missing** |
| Execution memory encoder | **Missing** |
| Controllers B/C partial/history interfaces | **Missing** |
| Stability-masked Dir + policy-θ sampling | **Missing** |
| Counterfactual consistency loss | Diagnostics exist; **train loss missing** |
| Visual EAGM | Paper visual aliasing exists; EAGM-visual **not** |

## 7. Minimal runnable GPU experiment config (next)

1. Env: `conda activate eagm_gpu`
2. `export OPENARM_MODEL_PATH=/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml`
3. Work under `eagm_gpu/`; write runs only to `eagm_gpu/runs/`
4. Stage 1 first: reproduce seed-0 \(M_u\) / \(M_u+\mathrm{Dir}\) at 1k (and load archived \(M_c\)) on **CPU FP32** for comparison, then optional CUDA parity check
5. Do **not** start five-seed 3k exact-Mu+Dir as a “new method”
6. After Stage 1 tolerances OK → Controller B + EAGM-Base/Dir single-seed smoke

## 8. Protection rules in force

- Do not edit `CVPR2027/` sources or overwrite paper hashes.
- Do not overwrite legacy `runs/` or `EAGM_archive_20261112/`.
- Do not write unverified EAGM claims into the diagnostic paper.
- New git history lives only under `eagm_gpu/`.
