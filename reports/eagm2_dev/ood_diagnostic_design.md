# Sparse exploratory OOD postprocessing

Implementation: `reports/eagm2_dev/evaluate_ood.py`. No frozen `eagm2/`, `scripts/`, `tests/`, or config files changed for this task. Only explicit `development_v1` checkpoints, authorized development roots, and tracking/contact tasks are accepted. The fixed comparison is ff_base, combined_dir, joint_calibration; checkpoint presence is mandatory and methods cannot be selected after seeing results.

Four single-factor configurations are specified before evaluation: gain1.15, smoothing beta0.5, rate limit0.025, extra lag2; other factors retain the development controller settings. Each configuration creates a real new controller and collects three validation-role episodes at delays0/1/2, from independent named substreams of the existing exploratory root. This does not create three new independent research seeds. Each saved start contains a physically evolved snapshot and its32-frame observable history. No hidden state is edited or transplanted across controllers. The snapshot's hidden fields are solely for simulator restoration; the deployable world-model calls receive x/c/action/public history only.

The original saved final policies are frozen before new observations. Paired theta0/final returns use horizon16 for tracking and the existing ten-second contact deadline. Default execution has no FD:1152 collection steps plus48 physical rollouts. Optional `--fd` uses only the gain configuration's first start, at theta0 and final theta for each fixed method, imagined/physical horizons8 or32, epsilon0.01 at scales0.5/1.0, with feasible bounded stencils. Two scales and a single start give limited numerical-stability information. They are not a confidence interval or broad calibration validation. A hard50000-issued-control-step cap protects against unexpectedly large cost; partial failures save raw results and measured budget.

Output is a fresh directory beneath reports/eagm2_dev and includes frozen configuration, checkpoint/source hashes, postprocessing implementation hashes, full newly collected histories/snapshots, raw paired returns, optional raw FD coordinates/masks/predicted gradients, failures and measured query budget. Original results and checkpoints remain unchanged. Vision is explicitly rejected because loading a state-based checkpoint is not a valid visual-policy reconstruction.

Validation: three isolated tests in `test_evaluate_ood.py` passed (`tests_ood_postprocessing.log`). These cover hard cap, checkpoint reconstruction for all fixed architectures, orchestration across four configurations with causal history collection, no training/test splits, and wrong-attempt rejection. The orchestration test uses a synthetic environment and mocked physical returns; it creates no scientific performance evidence. No actual checkpoint OOD evaluation or training was launched in this task.

This is sparse exploratory diagnosis, not sufficient OOD evidence. Three starts per configuration cannot establish statistical robustness; one root cannot establish seed variability; optional one-start FD cannot establish calibration across tasks or policies. Do not merge these results with confirmation or use a successful stratum to claim broad generalization.

Example (after all three checkpoints exist):

```sh
PYTHONPATH=. /home/aaa/miniconda3/envs/eagm_gpu/bin/python reports/eagm2_dev/evaluate_ood.py --source runs/eagm2_development/tracking/seed200001/development_v1 --output reports/eagm2_dev/ood_tracking_200001_v1
```
