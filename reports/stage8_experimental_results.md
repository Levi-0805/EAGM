# Stage 8 engineering results

All numbers are validation-only exploratory results, not confirmatory or OOD results. No success threshold was preregistered; success rate is therefore not invented. Test trajectories were collected but not used for model/weight selection.

| Seed | Model | True tracking return | full cosine | FailRate |
|---|---|---:|---:|---:|
| 40 | mc | -0.00436236 | 0.393988 | 0.333 |
| 40 | base | -0.00600683 | -0.003224 | 1.000 |
| 40 | dir | -0.00570197 | 0.075291 | 0.667 |
| 40 | reliability | -0.00371143 | 0.607398 | 0.167 |
| 41 | mc | -0.00348978 | 0.483960 | 0.000 |
| 41 | base | -0.00517080 | 0.226176 | 0.500 |
| 41 | dir | -0.00651183 | 0.220541 | 1.000 |
| 41 | reliability | -0.00541153 | 0.172123 | 0.833 |

Same EAGM architecture/parameters/init, same minibatch stream and FD queries. Candidate uses fractional mask weights; evaluation remains full-coordinate plus original binary stable mask. Training budget per Dir: 4800 dynamics steps +16128 FD control steps (2016 queries). Validation FD=8064 steps, collection total=7200; four-model paired physical evaluation=768 steps. 3000 WM / 50 policy updates. CPU. Disposable Base twin costs an additional 3000 training updates, explicitly recorded; its exact equality to Base is asserted. No extra environment queries buy candidate advantage.

Wall times: seed40 108.1s; seed41 110.5s. Raw model checkpoints, labels, splits and trajectories saved. The initial engineering implementation stores final policies in result JSON but not full optimizer traces; further method trials must add such traces. No five-seed expansion because second seed failed joint return+gradient gate.
