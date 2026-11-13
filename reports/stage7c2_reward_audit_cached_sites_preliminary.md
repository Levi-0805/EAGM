SUPERSEDED: preliminary cached-site diagnostic; use current-site final reports.

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
    "mean": 0.00439972081059049,
    "max": 0.0073471195465432335
  },
  "return_abs_diff_h16": {
    "mean": 0.01604114488527333,
    "max": 0.027580982618906576
  }
}

Rank correlations across held-out starts (final policies):

- mc: 0.371429
- base: 0.314286
- dir_random: 0.314286
- aux: 0.314286

Static errors use held-out start states; return errors cover full physical trajectories. They do not bound FK Jacobian error.
