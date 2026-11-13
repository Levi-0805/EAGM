# Stage 8 method selection

C1 reliability weighting: retain as an exploratory lead, do not promote. Seed40 passes both return/cosine gate and 6/6 validation starts. Seed41 improves return on 6/6 starts but lowers full cosine, remains worse than Mc and has high FailRate. Failed replication of the joint mechanism gate; no five-seed run. Mean weights near one can still alter nonconvex training; this is not proof of calibrated uncertainty.

C2 lag structure: same 616-parameter linear feature screen lowers validation command MAE only slightly (0.035284 to 0.035042). All six validation episodes improve, but this is not a capacity-matched neural bottleneck comparison or policy result. Keep only as a second low-priority hypothesis, no full network expansion.

C3 calibration: train-label LOEO scalar screening performed without new simulation. World model already consumed those labels, so regressor LOEO is not independent model calibration. Twelve labels cannot justify a flexible reliability head or validated update rejection. Defer deployment/gating until genuinely held-out train/calibration episodes are budgeted.

Priority recommendation: resolve known geometry/Jacobian objective fidelity before any new Reach architecture. This is a correctness intervention, not a new learned-method contribution. C1/C2 remain unconfirmed research leads; none merits a paper method claim today.

C3 scalar LOEO feasibility MSE=0.05830 versus constant predictor 0.06032. This small training-label-only difference does not establish independent calibration; raw predictions in analysis/stage8_calibration_screen.json.
