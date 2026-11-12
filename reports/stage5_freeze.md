# Stage 5 Freeze Record

- **Branch:** `stage5-confirm-budget`
- **Frozen Stage-4 commit:** `c386ed3193bea0ea676b4d014be33a3bfe9fe24b`
- **Config:** `configs/stage5_frozen.json`
- **Controller B:** gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25
- **Budget:** 3000 world-model updates, 50 policy updates, H_imag=8, H_eval=16
- **Loss weights:** λ_u=1.0, λ_d=0.1
- **Exploratory seeds (Stage 4):** 0–4 (do not retune on these)
- **Confirmatory seeds (Stage 5A):** 5–9
- **Rule:** Do not retune after seeing Stage 5A test outcomes. Do not overwrite Stage 4 artifacts. Do not edit CVPR2027/.
