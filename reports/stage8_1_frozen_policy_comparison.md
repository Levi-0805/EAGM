# Frozen-policy physical comparison

Exploratory seed20 frozen-model causal diagnostic. No new world-model training, no independent confirmation. Sources: runs/stage8_1_geometry/; script analysis/stage8_1_reports.py.

| condition | true return | paired Δ vs theta0 | FailRate | theta0 cosine | own-final cosine | stable final cosine | theta saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -0.01571363 | -0.00025745 | 0.667 | -0.13211 | -0.18454 | -0.18513 | 0.250 |
| B | -0.01548414 | -0.00002795 | 0.500 | -0.11552 | -0.09302 | -0.09302 | 0.000 |
| C | -0.01548414 | -0.00002795 | 0.500 | -0.11552 | -0.09302 | -0.09302 | 0.000 |
| D | -0.01571363 | -0.00025745 | 0.667 | -0.13211 | -0.18454 | -0.18513 | 0.250 |

B−A per start: [0.0014196697526748445, 0.0026140723059728532, 0.00013627425774184396, -0.0016082036668798594, 0.0003096853310386779, -0.0014945376819478773]; positive 4/6; mean 0.00022949338. These are six episodes from ONE exploratory seed, not six independent model-training seeds.

Frozen Dir only; same initial theta0, 12 original train starts, 50 Adam steps at 0.03, gradient norm clipping 1 and projection [-1,1]. Model and FK weights frozen. No metric-based checkpoint selection. Full optimization traces include actual update/gradient cosine and rho_proj. A/D final theta and B/C final theta asserted bit-identical; hence final FD queries are shared within each pair. Value-only correction cannot change deterministic Adam updates when gradients and stopping schedule are unchanged.

Reference FD: theta0 three scales, final two scales, eps=.01 feasible perturbations. Different mask strengths are explicitly reported; primary full-coordinate cosine avoids masking selection. H gradient=8; H physical return=16. Final gradient comparisons occur at different theta and should not be interpreted as the common-theta intervention effect.

Budget: {"theta0_fd": 8064, "physical_evaluation": 480, "final_fd": 10752}

Total added physical control steps including actual-state coverage: **19488/20000**. Main and supplemental raw budgets are preserved separately; combined accounting is total_budget.json. Static metrics CSV, component/goal/configuration summaries and SHA-256 inventory are in the run directory. No prior file changed relative to Stage8.1 preservation manifest.
