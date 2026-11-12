# EAGM — Execution-Aware Gradient-consistent World Models

Independent follow-up to the CVPR 2027 diagnostic paper *Execution Memory and Policy Gradients*.  
GitHub: https://github.com/Levi-0805/EAGM

This tree does **not** contain the CVPR paper sources. Legacy OpenArm diagnostic code/data needed for Stage-1 reproduction live under `vendor/`.

## Environment

Use the dedicated conda env `eagm_gpu` (NumPy 2.x is required to load legacy `starts.pt` pickles; `isaaclab50` has NumPy 1.26 and cannot).

```bash
conda activate eagm_gpu
export OPENARM_MODEL_PATH=/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml
cd /home/aaa/桌面/汇报/11月论文/eagm_gpu
```

## Stage order

0. Environment audit — `reports/environment_audit.md`
1. Baseline reproduction — Mc / Mu / Mu+Dir
2. Partial-unknown controllers B/C
3. EAGM modules + single-seed min experiment
4. Five-seed matrix (only after Stage 3 gate)
5. Closed-loop policy gains
6. Visual / public tasks (only after Stage 4–5 gains)

## Reproduce Stage-1 seed 0 (1k)

```bash
python scripts/run_stage1_seed0_1k.py
```
