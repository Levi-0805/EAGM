# Stage 8.3 optimizer–data alignment audit

Preregistered commit `0c5a2fe`, before new-test collection. Frozen seed20 Dir checkpoint; no training, no independent model-seed confirmation. Complete protocol: configs/stage8_3_preregistered.json. Raw data and runner: runs/stage8_3_alignment/ and analysis/stage8_3_run.py; regeneration: analysis/stage8_3_reports.py.

K2 training episodes0,2; K4 adds3,5, original training split, frame0. Validation episodes22,30. Fresh collection seed83017, episodes0,1: distinct from collection seed20 despite repeated numeric episode IDs. These two fresh trajectories are a one-shot exploratory check, not a new training seed or broad controller coverage (nominal delays0,1 only). Historical six test starts are explicitly previously inspected. Training summaries below restrict evaluation to the K active training starts, not all four.

Two gradient sources share TRUE reward derivatives: B learned transition vs D full privileged transition oracle. At each K, source gradients identical between optimizers. One fresh Adam step vs raw direction, both rescaled to L2=0.03 sqrt(28); no projection or saturation. This differs from unnormalized Stage8.2 Adam, so numbers must not be pooled. H8 gradient, H16 primary true EE return; H8 returns also retained. FD eps .01 at training points; .005/.01 at validation/fresh points. Reference gate passed. Numerical thresholds were fixed in the runner before results; the preregistration commit states the stability rule, while the runner additionally uses cosine>.999 and at least26 stable coordinates. Full coordinates remain primary; no weight/epsilon/start tuning followed outcomes.

| policy | validation delta | new-test delta | new-test positive | historical delta |
|---|---:|---:|---:|---:|
| K2_learned_raw | +1.8866372e-05 | -1.6228361e-06 | 1/2 | -6.1026535e-06 |
| K2_learned_adam_first | -3.5503835e-05 | +1.263568e-05 | 1/2 | +1.2952095e-05 |
| K2_oracle_raw | -5.4095568e-05 | -2.482002e-05 | 0/2 | +1.7324952e-05 |
| K2_oracle_adam_first | +9.5439434e-06 | -1.2553845e-05 | 0/2 | -9.5031439e-06 |
| K4_learned_raw | +2.9095544e-05 | +1.1620059e-05 | 1/2 | -9.2216211e-06 |
| K4_learned_adam_first | +6.0525386e-05 | +1.6141626e-05 | 1/2 | -2.3889134e-05 |
| K4_oracle_raw | -0.00014261356 | +7.676737e-05 | 2/2 | +5.7840841e-05 |
| K4_oracle_adam_first | -7.0958073e-05 | +4.2639354e-05 | 2/2 | +3.0844657e-05 |

## True training-gradient representation

| K | group | aggregate cosine |
|---|---|---:|
| 2 | validation | -0.289708 |
| 2 | fresh_test | -0.176948 |
| 2 | historical_test | 0.439796 |
| 4 | validation | -0.829528 |
| 4 | fresh_test | 0.381154 |
| 4 | historical_test | 0.751195 |

## Decision

Neither H1 nor H2 passes the preregistered joint validation/fresh-test screen. Oracle K4 improves both fresh starts but worsens both validation starts. Raw has better fresh-test return at K4, but worse validation return at both K values and worse fresh-test return at K2. Increasing training coverage rotates the gradient toward one distribution and away from another; two versus four fixed starts do not establish a monotone coverage law. This does not show that Adam alone caused Stage8.2 failure, nor that an inaccurate learned gradient is preferable in general.

No Stage8.4 method promotion, no five/ten seeds, no additional optimizer/start/step sweeps. Preserve the mixed outcome. The next research question, if separately pursued, concerns objective and start-distribution heterogeneity rather than tuning a method on these now-inspected trajectories. A larger audit would need a separately fixed sampling plan and cost; no large confirmation is justified by these data.

## Accounting and reproducibility

Actual control steps: **18240/20000**, phases {'collection': 400, 'training_gradients': 10784, 'reference_fd': 5048, 'evaluation': 2008}. Worst-case preregistered plan18576; exact-query caching reduces actual counts. CPU wall time 16.72s, no GPU training. Parameter counts/checkpoint identical in all cells. Collection shared; privileged oracle query cost explicitly included, not claimed as a deployable budget-equivalent algorithm. Tests compare first Adam to torch and equal update norms without physical steps. No task success threshold is invented; FailRate is delta<0. No multi-seed CI is computed from two fresh starts.

## Execution-distribution caveat

Post-result descriptive inspection (not a selection rule): training nominal delays are [0,2] for K2 and [0,2,0,2] for K4; validation delays are [1,0], fresh-test delays [0,1]. K4 increases starts but does not add the missing nominal-delay1 stratum. Goals and hidden-controller draws also differ. Thus this is not a balanced coverage experiment, and its mixed outcome cannot isolate sample count from composition. No start was replaced after inspecting this fact. Exact fixed contexts: execution_context.json.
