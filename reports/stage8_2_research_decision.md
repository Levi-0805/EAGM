# Stage 8.2 research decision

Exploratory seed 20, frozen Stage 7C.2 Dir-Random checkpoint. Six held-out trajectory starts are not six independent training seeds. No model training, no test-driven epsilon/LR selection. Reproducer: `analysis/stage8_2_reports.py`.

**Mechanism: Case A. Performance: Case D. Do not start five seeds or promote an oracle into a new method.**

1. Confirmed locally: state-transition and executed-command derivatives of the frozen Dir model disagree with physical derivatives. Single-step A cosine averages 0.073; conditional actuator B_u cosine averages 0.079.
2. Controlled causal evidence: keeping forward trajectories/rewards identical, changing the complete dynamics derivative raises H8 mean true-gradient cosine from -.132 to .795 with original reward derivatives, and from -.116 to .987 with true reward derivatives. All six starts improve in D-B. Geometry-only correction was insufficient; complete transition derivatives matter.
3. Implementation checks: independent scratch replay exact; native model AD/FD passes; actual-physical-forward local-J chain agrees with direct policy FD at .9999992. No evidence for an actor chain-rule bug at this checked start. Hidden execution state and solver warm-start must not be omitted from this conclusion.
4. Performance stop: full oracle one-update held-out delta is -7.05373666e-06; only C achieves positive mean improvement among derivative corrections. D improves the two training starts at H8 and H16, yet not the six held-out starts at either horizon. Gradient accuracy alone is insufficient for generalization.
5. Remaining explanation: training/test direction mismatch and Adam's coordinatewise update geometry are supported by measured first-order alignments, not established as universal causes. No saturation occurs. The full oracle's update/test alignment is -0.017911, despite training gradient cosine 0.996318. Only two training starts and one update were budgeted; do not generalize this failure to a 12-start/50-update optimizer.

## Recommended next step and limits

Keep Geometry-aware EAGM stopped. Do not retrain FK, C1/C2/C3, add vision, or launch Stage9/five-seed confirmation. A justified next *exploratory* diagnostic would preregister independent training/validation trajectories and compare raw-gradient vs Adam update alignment with matched step norms, without choosing settings on these six test starts. Use additional original training starts to test whether the two-start aggregate is representative. This is an optimizer/distribution-control question before a Dynamics-Jacobian-aware method claim.

Cost estimate for a future one-update H8 full-state oracle: at most 8*(2*(154+14)+1)=2696 physical steps per start, before cache savings. Twelve training starts would cost about 32352 before evaluation, exceeding this phase's 20k D cap. Therefore it was not silently expanded. A 2–4-start independent diagnostic could fit roughly 6k–12k plus evaluation/FD, but its exact protocol and held-out data should be fixed before use. No large confirmation is proposed from current evidence.

## Integrity, cost and scope

Branch `stage8-2-dynamics-jacobian-audit`; base HEAD `bcdce42f7f93cda2dc3b9b090c241b0809660187`; previous uncommitted research retained. No main merge, CVPR2027 edit, push or commit. Source preservation: `analysis/stage8_2_preservation/`.

A-C and numerical/localization diagnostics used 31372 physical control steps (declared cap40000, separate from D). D used 17432/20000. Replay test96. Total **48900** including one failed-step attempt. Failed localization attempt used a wrong attribute name; its traceback and cost are preserved, then a separate v2 run completed. Static geometry calls and CPU wall time are separately recorded in each raw budget. No GPU training or model-parameter changes occurred. The full project total is not being misreported as under20k; that cap applies to Stage8.2D in the task and was announced before execution.

Full local-J oracle queries expose hidden parameters/state only for diagnosis. Oracle not deployable; single exploratory checkpoint; no independent seeds; unscaled Jacobian norms depend on units; full hidden-state FD is not perfectly smooth. Exact numerical tolerances, all vectors, per-start outcomes, scripts, logs and SHA-256 manifest accompany the reports. No change to the paper's core claim is justified yet.
