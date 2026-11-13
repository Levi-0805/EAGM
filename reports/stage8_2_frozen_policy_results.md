# Frozen-policy true-return check

Exploratory seed 20, frozen Stage 7C.2 Dir-Random checkpoint. Six held-out trajectory starts are not six independent training seeds. No model training, no test-driven epsilon/LR selection. Reproducer: `analysis/stage8_2_reports.py`.

Budgeted local experiment: **one** Adam update, lr .03, norm clip1, theta clip[-1,1], theta0=0, first TWO pre-existing training starts, H8. All four conditions use the same frozen Dir checkpoint, initialization, starts and update schedule. This is not the 50-update Stage7 protocol and results must not be pooled with it. No new world model training. Six original held-out starts, H16 actual current MuJoCo EE reward for evaluation.

| condition | H8 full / stable cosine at theta0 | H16 true return after ONE update | paired delta vs theta0 | FailRate (delta<0) |
|---|---:|---:|---:|---:|
| A | -0.132108 / -0.132108 | -0.015446619 | +0.000009566 | 0.500 |
| B | -0.115517 / -0.115517 | -0.015446820 | +0.000009364 | 0.500 |
| C | 0.794742 / 0.794742 | -0.015420930 | +0.000035255 | 0.167 |
| D | 0.987378 / 0.987378 | -0.015463239 | -0.000007054 | 0.333 |

Theta saturation is zero in all conditions; projection retains the full update (rho_proj=1). FailRate above uses historical strict delta<0. The raw runner also saved an eps=1e-6 failure indicator; both are retained in derived CSV and this reporting distinction is explicit. Task success rate is unavailable: no frozen reach success threshold exists, and none was chosen from these results. Initial gradient cosine is at COMMON theta0; final true FD vectors are also saved, but no final oracle cosine is claimed because recomputing all final local-J paths exceeds the planned budget. Final FD is single-scale .01; final stable mask is explicitly unavailable.

| group | H | condition | paired delta |
|---|---|---|---|
| train | 8 | A | -1.91772688e-06 |
| train | 8 | B | -1.89454614e-06 |
| train | 8 | C | +1.77919814e-06 |
| train | 8 | D | +2.01258607e-06 |
| train | 16 | A | -1.8986658e-05 |
| train | 16 | B | -1.87997517e-05 |
| train | 16 | C | +1.56249002e-05 |
| train | 16 | D | +2.10246019e-05 |
| test | 8 | A | +1.67651233e-06 |
| test | 8 | B | +1.57394608e-06 |
| test | 8 | C | +1.14195326e-05 |
| test | 8 | D | -4.26590964e-07 |
| test | 16 | A | +9.56596645e-06 |
| test | 16 | B | +9.3644797e-06 |
| test | 16 | C | +3.52548559e-05 |
| test | 16 | D | -7.05373666e-06 |

The full oracle improves the TWO training starts' mean true return at both horizons but not held-out mean return. Hence horizon mismatch alone cannot explain the held-out failure. C's small improvement is exploratory; it cannot justify preferring inaccurate reward gradients or selecting a new method from the six test starts.

| condition | train g cosine | update vs predicted g | update vs train true g | update vs test true g |
|---|---|---|---|---|
| A | -0.386281 | 0.729513 | -0.332753 | 0.084711 |
| B | -0.257069 | 0.836130 | -0.350117 | 0.084738 |
| C | 0.715004 | 0.488553 | 0.293284 | 0.534341 |
| D | 0.996318 | 0.396606 | 0.360583 | -0.017911 |

True aggregate training-vs-test gradient cosine is 0.439796. D's predicted training gradient is accurate, but its Adam update is not parallel to that gradient; coordinatewise preconditioning shifts alignment with the test gradient below zero. The linear prediction for D test H8 is -3.53352095e-07, versus observed -4.26590964e-07. This supports a remaining interaction between selected training starts and optimizer geometry; it does not prove Adam is generally faulty. No optimizer was tuned, and no alternative optimizer was selected on test performance.

D physical control steps including supplemental train/test/horizon checks: **17432/20000**. Shared identical oracle queries are cached rather than counted repeatedly; all conditions have identical available stencils and no extra learned parameters. Direct-FD final vectors are diagnostic evaluation cost, not supervision. Sources: `stage8_2d_frozen_policy/`, `stage8_2_localization_v2/optimizer_horizon.json`, `train_true_gradients.json`; derived per-start CSV and optimizer CSV.
