# Stage 7C.2 research decision

Seed20 is exploratory and was used to discover this protocol issue. No multiseed experiment is authorized by this report. Old Stage 7C.1 artifacts are unchanged.

```json
{
  "dir_true_return_beats_base": false,
  "dir_true_gradient_beats_base_at_common_theta0": false,
  "dir_beats_mc": false,
  "fk_accepted": false,
  "rankings_equal": false,
  "propose_pilot": false,
  "launch_pilot": false,
  "launch_stage7d": false
}
```

This is a protocol correction. True-FD labels were regenerated and Dir retrained for 3000 updates; existing surrogate labels were not reused. Whether supervision helps relative to Base is shown above; isolating the causal effect of switching label objectives requires a matched surrogate-label Dir control, which this four-model run does not provide.

FK inadequacy or a material rank reversal blocks a clean transfer claim even if Dir beats Base. No claim of monotonic training-budget dependence is supported. Stop for human review. Any future engineering pilot uses fresh seeds 21–23; confirmation 30–39 requires separate approval.

Primary artifacts: `runs/stage7c2_true_ee_current_sites_seed20/`. True reward refreshes site positions from current qpos in separate MjData, preserving live integration/warm-start state. Earlier cached-site preliminary diagnostics remain under `runs/stage7c2_true_ee_seed20/` and are superseded. Each run used 111264 control steps; both runs together used 222528, excluding separately recorded replay tests.
