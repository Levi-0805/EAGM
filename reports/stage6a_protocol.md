# Stage 6A — Protocol & Evaluation Repair

Branch: `stage6-exec-generalization` · base Stage 5: `6f66823`  
Stage 5 artifacts not overwritten.

## Delivered

| Item | Location |
|------|----------|
| Paired ΔR / fail rate API | `training/eval_metrics.py`, `policy_opt.eval_physical_paired_vs_theta0` |
| θ-param vs action-clip saturation | `training/rollout.py`, `training/policy_opt.py` |
| Full + stable-mask gradient metrics | `training/gradients.py` + `eval_metrics.gradient_quality` |
| Stage 5E script fixes | `scripts/run_stage5e_ood.py` |
| Corrected Stage 5E re-eval | `runs/stage6a_stage5e_corrected/`, `reports/stage5e_corrected_stats.md` |
| Saturation figures | `reports/figures/stage6a_theta_saturation.png`, `stage6a_sat_vs_cosine.png` |
| OOD config list (+ compound) | `configs/stage6_ood_configs.json` |
| Frozen Stage 6 config | `configs/stage6_frozen.json` |
| Unit tests | `tests/test_stage6a_metrics.py` (7/7 pass) |
| Train-only exec-state labels | `PartialUnknownArmEnv.privileged_exec_state()` |
| Issue board | `reports/stage6_issues.md` |

## Pass criteria

- Same starts → paired ΔR / fail rate consistent across re-runs.
- Budget unit test: train 4800 + dir 16128; val/IID/FD excluded.
- Fail-rate unit test: rejects `mean(R>0)` as failure definition.

## Next (Issue 2)

Reproduce Stage 5E protocol on seeds **10–14** with Stage 6A metrics; do not change model structure yet.
