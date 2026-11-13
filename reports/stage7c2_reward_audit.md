# Stage 7C.2 Reward audit

Seed20 is exploratory and was used to discover this protocol issue. No multiseed experiment is authorized by this report. Old Stage 7C.1 artifacts are unchanged.

FK training RMSE: 0.02849009 m. Independent uniform-joint validation RMSE (1000 samples): 0.03245129 m. Acceptance remains **0.02 m**; passed: **False**. No threshold was loosened. Training continued solely as the authorized objective-correction diagnostic, not as acceptance of FK quality.

{
  "ee_l2": {
    "mean": 0.05682889324485536,
    "max": 0.05851982887554537
  },
  "abs_reward_error": {
    "mean": 0.0012176692116250924,
    "max": 0.0019881075968924983
  },
  "relative_reward_error": {
    "mean": 1.2513084076561727,
    "max": 2.512718621087242
  },
  "return_abs_diff_h4": {
    "mean": 0.004398575092025802,
    "max": 0.00736572115428526
  },
  "return_abs_diff_h16": {
    "mean": 0.01593657792760507,
    "max": 0.02759487074708085
  }
}

Rank correlations across held-out starts (final policies):

- mc: 0.371429
- base: 0.314286
- dir_random: 0.428571
- aux: 0.314286

Static errors use held-out start states; return errors cover full physical trajectories. They do not bound FK Jacobian error.

Primary artifacts: `runs/stage7c2_true_ee_current_sites_seed20/`. True reward refreshes site positions from current qpos in separate MjData, preserving live integration/warm-start state. Earlier cached-site preliminary diagnostics remain under `runs/stage7c2_true_ee_seed20/` and are superseded. Each run used 111264 control steps; both runs together used 222528, excluding separately recorded replay tests.
