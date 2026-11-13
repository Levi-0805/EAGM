# Controlled gradient-path intervention

Exploratory seed 20, frozen Stage 7C.2 Dir-Random checkpoint. Six held-out trajectory starts are not six independent training seeds. No model training, no test-driven epsilon/LR selection. Reproducer: `analysis/stage8_2_reports.py`.

Forward transition x always comes from the SAME frozen learned model; hidden-state forward values follow a common scratch physical transition. Custom VJP replaces only derivatives in C/D. Common hidden state does not enter the native model. All conditions share the same issued actor actions, states and reward values at theta0; this is asserted for every start. Oracle derivatives are evaluated at these common imagined augmented states, not silently at a different physical trajectory.

Reward forward value is always FK-based. B/D replace the complete position reward derivative using true position AND true J. Joint-velocity and delta-action penalties are unchanged. This is not the Stage 8.1 “true J only” operation. A/C keep FK derivatives. `training/stage8_2_dynamics.py::ReplaceJacobian` supplies the local physical 154x168 VJP; actor autograd stays identical in all conditions.

| condition | H8 full / stable cosine at theta0 | H16 true return after ONE update | paired delta vs theta0 | FailRate (delta<0) |
|---|---:|---:|---:|---:|
| A | -0.132108 / -0.132108 | -0.015446619 | +0.000009566 | 0.500 |
| B | -0.115517 / -0.115517 | -0.015446820 | +0.000009364 | 0.500 |
| C | 0.794742 / 0.794742 | -0.015420930 | +0.000035255 | 0.167 |
| D | 0.987378 / 0.987378 | -0.015463239 | -0.000007054 | 0.333 |

The return column belongs to the separate one-update check, not a gradient intervention at fixed theta. H8 D-B cosine differences: [0.7842121891553253, 1.2954194230058942, 1.207974740461435, 1.2280038825362039, 1.223764739300434, 0.8779956222569857]. All six improve. C-A also improves all six. Thus the complete transition sensitivity is a causal contributor to the fixed-checkpoint policy-gradient error under this diagnostic, even with identical forward prediction error. This does not establish that a particular A block, B block, hidden memory or parameterization is the sole cause. Bare F(x,u) derivative errors show execution omission is not the entire explanation.

Whole-policy FD does NOT validate a custom substituted derivative by perturbing its frozen forward function: that would intentionally return the learned derivative. Instead we verify the custom VJP algebra, independently finite-difference the physical local transition, and compare the entire physical-forward oracle chain with direct policy FD. All 28-dimensional vectors are preserved. Stable coordinate quality comes from three epsilon scales; all H8 coordinates pass.

Gate before D: local A/B discrepancy, physical-chain cosine > .99, D-B improvement >0 on >=4/6 starts and mean >.1. Observed conditions pass. No test-return-selected settings. No deployable model or independent confirmation is claimed.

Sources: raw `stage8_2a_single_step`, validated `stage8_2_numerical`, code `training/stage8_2_dynamics.py`, tests `tests/test_stage8_2_dynamics.py`.
