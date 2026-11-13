SUPERSEDED: preliminary cached-site diagnostic; use current-site final reports.

# Stage 7C.2 True gradient audit

Seed20 is exploratory and was used to discover this protocol issue. No multiseed experiment is authorized by this report. Old Stage 7C.1 artifacts are unchanged.

| Model | theta | full cosine | stable cosine | stable sign agreement | relative gradient error |
|---|---|---:|---:|---:|---:|
| mc | theta0 | -0.02727 | -0.02692 | 0.36486 | 1.52160 |
| mc | final | -0.04376 | -0.09430 | 0.41480 | 1.56899 |
| base | theta0 | 0.06893 | 0.06873 | 0.56989 | 2.90862 |
| base | final | 0.05583 | 0.06388 | 0.48718 | 2.84817 |
| dir_random | theta0 | -0.07790 | -0.07819 | 0.45569 | 1.00915 |
| dir_random | final | -0.09207 | -0.11475 | 0.43199 | 1.00931 |
| aux | theta0 | -0.11253 | -0.11381 | 0.36530 | 1.35016 |
| aux | final | -0.13261 | -0.13270 | 0.35682 | 1.30739 |

Means across all held-out starts. θ0 is a common comparison; final θ is model-specific. H=8 for both model and FD gradients; physical policy evaluation H=16. Full coordinate gradients, stability/boundary masks, effective perturbations, surrogate-reference diagnostics and query counts are stored in gradients.json. Training labels are separate from these held-out probes. Magnitude error is ||g_model−g_true|| / ||g_true||, not only norm-ratio error.
