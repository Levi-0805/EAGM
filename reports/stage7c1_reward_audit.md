# Stage 7C.1 — Reward Parity Audit

**Artifact:** `runs/stage7c1_diagnosis/reward_audit.json`  
**Status:** **PASS** (`parity_pass=true`)

## Shared definition (Reach)

\[
R=-\mathrm{mean}(e_{\mathrm{ee}}^2)-5\cdot10^{-4}\,\mathrm{mean}(\dot q^2)-0.05\,\mathrm{mean}((a-a_{\mathrm{prev}})^2)
\]

| Item | Value |
|------|-------|
| Left/right EE weights | Equal (stacked 6-D, mean over dims) |
| Discount | 0.99 |
| H_imag / H_eval | 8 / 16 |
| Imagined ↔ physical primary path | Same `_step_reward(task_id=reach)` via FK surrogate |
| Secondary metric | `true_mujoco_ee_return` (logged only) |

## Numerical parity (identical q, a=0, shared goals)

| Metric | Value |
|--------|-------|
| mean \|R_surr − R_true\| | **5.89e−4** |
| max \|R_surr − R_true\| | 9.48e−4 |
| mean FK EE L2 (m) | 0.057 |
| max FK EE L2 (m) | 0.059 |
| FK fit RMSE | 0.0285 |

Gate used in code: mean\|ΔR\| &lt; 0.01 and mean FK L2 &lt; 0.08 → **PASS**.

## Conclusion

Reward implementation is shared for Dir FD / imagined / primary physical return. FK surrogate introduces a **small but nonzero** EE error (~5–6 cm L2); this is documented and within the audit gate. It does **not** explain the 400-update Dir&lt;Base smoke by itself — proceed to gradient + full-budget analysis.
