from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
def write(n,s):(R/('stage8_'+n+'.md')).write_text(s)
write('evidence_ledger','''# Stage 8 evidence ledger

A — Repeated multiseed support within tested tracking protocols: matched-θ Dir improves Base physical return in Stage 4, independent Stage 5A, Stage 5E and independent Stage 6B.1. This does not imply all tasks or all controllers. Source: master CSV and independent bootstrap JSON.

B — Distribution-limited: fixed-B Dir vs Mc; control-step budget advantage; Base vs ObsMLP/Aux under the selected architecture sizes. Stage 6B.1 OOD vs Mc remains uncertain. Five seeds cannot establish broad superiority. Stage 6C Base was not batch-paired with reported Dir, so its Base contrast is weaker than Path−Random.

C — Exploratory: Reach seed20 failure and surrogate/true ranking reversal; no transfer success. Stage8 candidate seed40 is validation-only engineering evidence.

D — Unsupported under tested protocols: known-controller Mu+Dir at 3k; HistC capacity-independent advantage; Path superiority; reduced saturation as performance cure. These are failed hypotheses in these implementations, not universal impossibility theorems. ExecState was deferred, not empirically disproved.

E — Unverified: reliability weighting improves policy return; calibrated gradient risk generalizes; execution temporal residual resolves OOD; geometry repair restores Reach; hardware/vision/public-benchmark superiority. No SOTA claim.

Selection caveat: sequential stage exploration, reused seeds, multiple contrasts and reused test pools reduce independence. Never pool stages into a larger confirmatory n. A fresh final confirmation must lock the complete protocol and selection rule before data generation.
''')
write('method_candidates',r'''# Stage 8 candidate mechanisms and gates

C1 — Continuous numerical reliability weighting. For existing FD scale estimates g_i^(s), let d_i=max_s g_i^(s)-min_s g_i^(s), t_i=max(1e-9,0.1|g_i|), w_i=clip((1+(d_i/t_i)^2)^(-1),0.25,1). Replace binary Huber weighting m_i by m_i w_i and normalize by sum m_i w_i. Same architecture, stencils, label count and training updates. No new simulator budget. Existing eval masks remain binary; report full-coordinate metrics. This is an incremental candidate, not established novelty. Three-scale disagreement is truncation/smoothness evidence, not a calibrated stochastic variance estimate. Failure: labels already reliable, objective derivative bias, or collapsed predicted magnitudes.

C2 — Structured lag mixture residual: uhat=sum_l softmax(z)_l U0(c_{t-l},a_{t-l})+Delta, l in {0,1,2}. All history must be observed; never use hidden controller parameters. Reduce residual width to match total parameters within 1%; compare unordered/reversed histories and plain residual with same feature count. Use same trajectories and no extra FD. Before neural implementation, equal-dimensional ridge current-vs-lagged-action validation screens whether history alone has a useful command signal. A gain in command MAE is insufficient to authorize expansion; need downstream full-gradient and return gains. Do not claim this elementary mixture is novel without further work.

C3 — Reliability calibration: estimate q(z)=P(g_true^T d>0 | z), using features such as model gradient norm, dynamics residual and ensemble disagreement; calibrate only on disjoint train episodes. Accept an update only if a lower confidence bound is positive; retain rejected-update fraction and full-set return. This is not Path-Dir label redistribution. Must match FD budgets, account for ensemble parameters/compute, compare unconditional smaller updates, and test coverage/calibration on untouched episodes. Twelve training episode labels are insufficient for a flexible calibration model; initial screen defers it rather than fitting on test probes.

Theory boundary: for a differentiable finite-horizon composition, g=Dx_theta^T r_x plus action terms. Thus ||ghat-g|| <= ||Dhat-D|| ||r_x|| + ||Dhat|| ||rhat_x-r_x|| (with rewards/states evaluated on a common trajectory; rollout mismatch adds another term). Small state MSE alone bounds neither Jacobian error. Finite-horizon smoothness and bounded Jacobians are assumptions; clipping/contact violate differentiability at boundaries. An L-smooth true objective gives J(theta+eta*d)>=J(theta)+eta*g^T*d-L*eta^2||d||^2/2. Positive cosine alone is insufficient without norm/step control; projection and Adam change d. No empirical correlation is a theorem.

Predeclared single-seed gate is in configs/stage8_protocol.json. Candidate decisions use validation only; test is not used to choose weights. No 5+ seed run until replicated engineering evidence. No visual expansion while objective fidelity fails.
''')
write('paper_outline','''# Stage 8 paper outline

1. Problem and scope: partial execution dynamics in observable software state; distinguish commanded action from executed target. Figure: controller/state/command dependencies. Specify simulator-reset access and command-label availability.
2. Observation: state prediction accuracy does not ensure policy gradient quality. Show matched models, fixed objectives and full/stable metrics; use historical gradient/return figure and Reach failure as a boundary case.
3. Method: U0 plus learned command residual, matched-θ three-scale masked normalized Huber gradient supervision. Clearly acknowledge magnitude supervision. Add a reliability extension only if experiments support it.
4. Analysis: chain-rule decomposition of dynamics Jacobian error, reward Jacobian error and trajectory mismatch. State smoothness assumptions; separate clipping boundaries. No universal monotonicity claim.
5. Evaluation protocol: paired initialization/minibatches, episode splits, seed-level inference, control-step plus actual compute accounting, all-start physical returns. Include frozen protocol table and artifact provenance.
6. Results: Stage4 exploratory; Stage5A independent confirmation; Stage5B budget; Stage5C architecture; Stage5E/6B1 interpolation/extrapolation separately. Show every seed and LOO sensitivity.
7. Ablations and limitations: HistC, Path, saturation; new method validation only if it passes; negative Reach True-EE audit. Required future geometry-Jacobian intervention before any transfer claim. Do not hide failed extensions.
8. Scope: low-dimensional fixed controller family, short horizon, simulator resets, no hardware/vision validation. External baselines have not been run; no SOTA statement.

Submission readiness: core method can be drafted now as a narrow study, but a high-level method claim requires stronger novelty against PAML/GAMPS/AWM and a robust true-objective task. New-stage labels do not themselves make a contribution.
''')
write('venue_assessment','''# Stage 8 venue assessment

This is a scope assessment, not an acceptance forecast or deadline recommendation.

CoRL: strongest conceptual fit for execution-aware robot learning if there is a clear algorithmic mechanism, credible independent tasks and hardware or transferable simulation evidence.
ICRA: plausible fit for an execution-interface method with reproducible robotics validation and measured practical cost. A meaningful system demonstration would strengthen it.
RSS: would require a particularly crisp insight/mechanism and convincing robotics evidence; current narrow simulator results alone do not justify confidence.
CVPR: current state-only control experiments offer limited computer-vision contribution. Adding images without isolating perception/execution/model errors would not resolve this. Prioritize a scientifically necessary visual task only after low-dimensional objective validity and method gains pass.

No external public-benchmark reproduction was attempted this cycle. Local CPU runs are feasible; the historical host has an 8GB RTX4060 but this process reports CUDA unavailable. This does not establish that any baseline is impossible. A fair external comparison needs a shared action interface, reset privilege, observation specification, simulator/compute budget and official baseline configuration; no scores should be invented. Before proposing a full baseline run, benchmark memory/time on one task and separate observation/action advantages.
''')
if __name__=='__main__':print('reports written')
