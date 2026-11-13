# Stage 8.3 decision

**Stop this screening route; neither preregistered joint gate passed.**



Neither H1 nor H2 passes the preregistered joint validation/fresh-test screen. Oracle K4 improves both fresh starts but worsens both validation starts. Raw has better fresh-test return at K4, but worse validation return at both K values and worse fresh-test return at K2. Increasing training coverage rotates the gradient toward one distribution and away from another; two versus four fixed starts do not establish a monotone coverage law. This does not show that Adam alone caused Stage8.2 failure, nor that an inaccurate learned gradient is preferable in general.

No Stage8.4 method promotion, no five/ten seeds, no additional optimizer/start/step sweeps. Preserve the mixed outcome. The next research question, if separately pursued, concerns objective and start-distribution heterogeneity rather than tuning a method on these now-inspected trajectories. A larger audit would need a separately fixed sampling plan and cost; no large confirmation is justified by these data.

## Accounting and reproducibility

Actual control steps: **18240/20000**, phases {'collection': 400, 'training_gradients': 10784, 'reference_fd': 5048, 'evaluation': 2008}. Worst-case preregistered plan18576; exact-query caching reduces actual counts. CPU wall time 16.72s, no GPU training. Parameter counts/checkpoint identical in all cells. Collection shared; privileged oracle query cost explicitly included, not claimed as a deployable budget-equivalent algorithm. Tests compare first Adam to torch and equal update norms without physical steps. No task success threshold is invented; FailRate is delta<0. No multi-seed CI is computed from two fresh starts.
