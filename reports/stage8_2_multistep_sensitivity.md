# Multi-step sensitivity propagation

Exploratory seed 20, frozen Stage 7C.2 Dir-Random checkpoint. Six held-out trajectory starts are not six independent training seeds. No model training, no test-driven epsilon/LR selection. Reproducer: `analysis/stage8_2_reports.py`.

| horizon | A | B | C | D |
|---|---|---|---|---|
| 1 | 0.044423 | 0.026689 | 0.253030 | 0.330487 |
| 2 | 0.000267 | 0.005976 | 0.911072 | 0.997603 |
| 4 | -0.095266 | -0.040319 | 0.781624 | 0.990450 |
| 8 | -0.132108 | -0.115517 | 0.794742 | 0.987378 |

A/B use learned dynamics; C/D use the augmented physical Jacobian. A/C use FK reward derivatives; B/D use true geometry derivatives. All conditions share bit-identical forward x and forward reward values. These are two derivative factors, not four deployment algorithms.

Sensitivity is the complete dx_t/dtheta (28x28), with policy feedback through x and target included automatically. Thus the recurrence includes the actor's state dependence, not only a direct du/dtheta term. Physical sensitivity comes from whole-trajectory theta +/- 0.01 rollouts. At horizons 1/2/4/8 the CSV includes state error, sensitivity error, gradient norm ratio, relative gradient error, full and stable cosine, direction agreement and true norm. A zero true gradient at horizon 1 can arise from delay; the legacy cosine helper returns zero there, which is not an observed opposite direction. Inspect norm alongside cosine.

Additional whole-policy FD at .005 and .02 produces independent stability masks (spread <= max(1e-9,.1*abs(g_.01))). All 28 coordinates are stable on all six starts at H8. Boundary masks are separately zero at theta0. `validated_metrics.json` is authoritative: the initial `multistep.json` used all-coordinate masks as placeholders, so its unvalidated stable-coordinate fields must not be used for other horizons. No dimensions were removed to improve the primary full cosine.

The native model passes a local AD/FD check (relative error 9.35e-5). On a true forward physical trajectory, full local-J chain propagation agrees with direct whole-policy FD at cosine 0.9999992 and relative error 0.0012603 (predefined start 0). This verifies actor feedback, execution memory and chain assembly at that start. It does not prove every optimizer behavior is correct or identify a deployable derivative estimator.

Single-step derivatives are already erroneous; this is not evidence for a purely long-horizon accumulation mechanism. For example, D is about 0.998 at H2 and 0.987 at H8, whereas learned-dynamics gradients deteriorate to negative alignment. No horizon 16 gradients were requested by the frozen training protocol (H8); physical return at H16 is separately analyzed.

Sources: `stage8_2a_single_step/{multistep,references,numerical_checks}.json`, `stage8_2_numerical/{policy_epsilon,validated_metrics}.json`; derived `stage8_2_summary/multistep.csv`.
