# Stage 7C.2 True gradient audit

Seed20 is exploratory and was used to discover this protocol issue. No multiseed experiment is authorized by this report. Old Stage 7C.1 artifacts are unchanged.

| Model | theta | full cosine | stable cosine | stable sign agreement | relative gradient error |
|---|---|---:|---:|---:|---:|
| mc | theta0 | -0.02719 | -0.02688 | 0.36486 | 1.50184 |
| mc | final | -0.04322 | -0.09381 | 0.41480 | 1.54759 |
| base | theta0 | 0.06879 | 0.06860 | 0.56989 | 2.84533 |
| base | final | 0.05600 | 0.06407 | 0.49027 | 2.78974 |
| dir_random | theta0 | -0.13210 | -0.12680 | 0.50948 | 1.05805 |
| dir_random | final | -0.18487 | -0.17032 | 0.44702 | 1.06449 |
| aux | theta0 | -0.11240 | -0.11373 | 0.36530 | 1.33622 |
| aux | final | -0.13222 | -0.13229 | 0.35682 | 1.29515 |

Means across all held-out starts. θ0 is a common comparison; final θ is model-specific. H=8 for both model and FD gradients; physical policy evaluation H=16. Full coordinate gradients, stability/boundary masks, effective perturbations, surrogate-reference diagnostics and query counts are stored in gradients.json. Training labels are separate from these held-out probes. Magnitude error is ||g_model−g_true|| / ||g_true||, not only norm-ratio error.

Primary artifacts: `runs/stage7c2_true_ee_current_sites_seed20/`. True reward refreshes site positions from current qpos in separate MjData, preserving live integration/warm-start state. Earlier cached-site preliminary diagnostics remain under `runs/stage7c2_true_ee_seed20/` and are superseded. Each run used 111264 control steps; both runs together used 222528, excluding separately recorded replay tests.
