# Single-step dynamics Jacobian audit

Exploratory seed 20, frozen Stage 7C.2 Dir-Random checkpoint. Six held-out trajectory starts are not six independent training seeds. No model training, no test-driven epsilon/LR selection. Reproducer: `analysis/stage8_2_reports.py`.

The learned transition derivative is substantially wrong on these six starts. Mean state-Jacobian cosine: **0.073335**; relative Frobenius error **1.017813**. This is a local finding, not a cross-seed statement.

## State and physical interface

Physical augmented state is 154 dimensional: q/qdot (28), integrator warm-start (14), target (14), filter (14), nominal FIFO (28), prior executed command (14), smoother (14), extra FIFO (28). Inputs are 14 issued incremental actions. Fixed context includes controller parameters, nominal delay, extra lag, FIFO occupancy schedule, external forces and time. Solver warm-start is numerical state, not deployable controller observation. No actuator activation state (`na=0`); nq=nv=14. The model sees x28 + observed controller state29 + action14, not hidden Controller B parameters. Hidden-state access is oracle-only.

Independent scratch environment/MjData restores the entire source snapshot for every perturbation. No live evaluation data are mutated. Execution memory is perturbed explicitly; discrete parameters are held fixed. Eight-step replay matched original environment bit for bit on six starts. A/B matrices therefore include issued-action execution, not just bare actuator physics. Four starts have exactly zero immediate physical action Jacobian because of execution delay; undefined cosine/relative error is reported null, not interpreted as success.

| start | delta-normalized MSE | A relative | A cosine | issued-action B cosine |
|---|---|---|---|---|
| 0 | 1.3043 | 1.0175 | 0.0766 | None |
| 1 | 12.0164 | 1.0117 | 0.0713 | -0.02363497320844635 |
| 2 | 2.0878 | 1.0134 | 0.0776 | -0.022597510709131408 |
| 3 | 1.0832 | 1.0259 | 0.0714 | None |
| 4 | 1.4178 | 1.0246 | 0.0757 | None |
| 5 | 8.3682 | 1.0138 | 0.0675 | None |

## Separate physical F(x,u) check

Bypass the software controller with the actual executed actuator target u; hold warm-start and context fixed. Direct MuJoCo replay equals the composed next state to 1e-10. Compare the identical learned F subnetwork at that u: mean A cosine 0.073152, mean B_u cosine 0.078918. This localizes an error inside the learned physical dynamics map as well as the missing hidden execution sensitivities. Because F was trained jointly on predicted commands, evaluating at true u is a conditional privileged probe, not an independent deployment benchmark. The full oracle replaces both physical and execution derivatives; their separate causal shares remain unresolved.

## Numerical checks and units

Central local eps: q/qdot=1e-4, warm-start=1e-3, execution/action=1e-5. Independently compare scales .5 and 2 on all six starts. Maximum A-block change is 0.8567%, versus roughly 100% learned error. Full augmented-J change reaches 7.2904%; do not claim exact smoothness of every internal solver/controller coordinate. Base epsilon fixed before outcomes, no return-based selection. Clamps remain in the simulator, so derivatives are local piecewise derivatives, not global smoothness claims. Policy theta0 is interior; no boundary mask is conflated with numerical stability.

Native model directional AD/FD relative error 9.35e-05. Singular values and leading input singular-direction errors, q/qdot output blocks, full raw matrices and six starting configurations are saved. Frobenius/SVD metrics mix physical units and are scale-sensitive; component blocks accompany them. Forward error here is mean(((x_hat-x_true)/ystd)^2), normalized by training DELTA standard deviations, not the historical aggregated dynamics NMSE. Pearson correlation with A-relative error is -0.6942 (n=6); descriptive only and cannot establish prediction/derivative independence.

Sources: `runs/stage8_2a_single_step/single_step.json`, `runs/stage8_2_localization_v2/direct_actuator_dynamics.json`, `runs/stage8_2_numerical/local_epsilon.json`; derived `runs/stage8_2_summary/single_step.csv`.
