## Current research archive (Stages 7–8.3)

The historical Stage0–6 documentation below is preserved. The current research branch adds corrected Reach evaluation and frozen-model mechanism audits; it does **not** establish successful Reach transfer or a deployable oracle method.

- [Stage7C.2 true-EE correction](reports/stage7c2_research_decision.md)
- [Stage8 historical evidence audit](reports/stage8_historical_audit.md) and [evidence ledger](reports/stage8_evidence_ledger.md)
- [Stage8.1 geometry decision](reports/stage8_1_research_decision.md)
- [Stage8.2 dynamics/gradient decision](reports/stage8_2_research_decision.md) and [reproduction guide](runs/stage8_2_summary/reproduce.md)
- [Stage8.3 preregistration](reports/stage8_3_preregistration.md), [optimizer–data audit](reports/stage8_3_optimizer_data_alignment.md), and [decision](reports/stage8_3_research_decision.md)
- [Independent EAGM manuscript draft](paper_eagm/README.md); the submitted CVPR2027 manuscript is unchanged.

Stage8.3's validation/fresh-test joint gates did not pass; no Stage8.4 or multi-seed method confirmation was launched. To reconstruct archived large trajectory datasets, run `python scripts/restore_research_data.py`; checksums and ordered parts are under `artifacts/research_data_parts/`. New runs refuse to overwrite existing output directories. Use a separate workspace for reproduction.

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
