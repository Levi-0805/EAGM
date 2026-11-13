from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
js=[json.loads((ROOT/f'runs/stage8_c1_seed{s}/results.json').read_text()) for s in [40,41]]
rows=[]
for j in js:
 for m,v in j['results'].items():rows.append(dict(seed=j['seed'],model=m,return_=v['policy_return_mean'],cosine=v['cosine_full_mean'],nmse=v['dynamics_nmse'],command_mae=v['cmd_mae'],fail_rate=v['fail_rate_vs_theta0'],source=f'runs/stage8_c1_seed{j["seed"]}/results.json'))
with (ROOT/'analysis/stage8_candidate_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Stage 8 engineering results','','All numbers are validation-only exploratory results, not confirmatory or OOD results. No success threshold was preregistered; success rate is therefore not invented. Test trajectories were collected but not used for model/weight selection.','','| Seed | Model | True tracking return | full cosine | FailRate |','|---|---|---:|---:|---:|']
for r in rows:lines.append(f'| {r["seed"]} | {r["model"]} | {r["return_"]:.8f} | {r["cosine"]:.6f} | {r["fail_rate"]:.3f} |')
lines+=['','Same EAGM architecture/parameters/init, same minibatch stream and FD queries. Candidate uses fractional mask weights; evaluation remains full-coordinate plus original binary stable mask. Training budget per Dir: 4800 dynamics steps +16128 FD control steps (2016 queries). Validation FD=8064 steps, collection total=7200; four-model paired physical evaluation=768 steps. 3000 WM / 50 policy updates. CPU. Disposable Base twin costs an additional 3000 training updates, explicitly recorded; its exact equality to Base is asserted. No extra environment queries buy candidate advantage.', '',f'Wall times: seed40 {js[0]["wall_seconds"]:.1f}s; seed41 {js[1]["wall_seconds"]:.1f}s. Raw model checkpoints, labels, splits and trajectories saved. The initial engineering implementation stores final policies in result JSON but not full optimizer traces; further method trials must add such traces. No five-seed expansion because second seed failed joint return+gradient gate.']
(R/'stage8_experimental_results.md').write_text('\n'.join(lines)+'\n')
(R/'stage8_method_selection.md').write_text('''# Stage 8 method selection

C1 reliability weighting: retain as an exploratory lead, do not promote. Seed40 passes both return/cosine gate and 6/6 validation starts. Seed41 improves return on 6/6 starts but lowers full cosine, remains worse than Mc and has high FailRate. Failed replication of the joint mechanism gate; no five-seed run. Mean weights near one can still alter nonconvex training; this is not proof of calibrated uncertainty.

C2 lag structure: same 616-parameter linear feature screen lowers validation command MAE only slightly (0.035284 to 0.035042). All six validation episodes improve, but this is not a capacity-matched neural bottleneck comparison or policy result. Keep only as a second low-priority hypothesis, no full network expansion.

C3 calibration: train-label LOEO scalar screening performed without new simulation. World model already consumed those labels, so regressor LOEO is not independent model calibration. Twelve labels cannot justify a flexible reliability head or validated update rejection. Defer deployment/gating until genuinely held-out train/calibration episodes are budgeted.

Priority recommendation: resolve known geometry/Jacobian objective fidelity before any new Reach architecture. This is a correctness intervention, not a new learned-method contribution. C1/C2 remain unconfirmed research leads; none merits a paper method claim today.
''')
(R/'stage8_research_decision.md').write_text('''# Stage 8 research decision

P0 completed as a bounded audit: preserved prior workspace, independently aggregated raw CSVs with provenance, seed/bootstrap/config sensitivity, evidence ledger and targeted Reach numerical/geometry/dynamics/optimization diagnosis. Full independent historical retraining was not performed; past overwrites before the initial snapshot cannot be excluded.

P1 executed: three mechanisms designed; C1 implemented and trained on two fresh engineering seeds; C2 equal-capacity linear feature screen; C3 train-label scalar calibration feasibility. C1 seed41 fails joint gradient+return gate. No five-seed, ten-seed, visual or contact expansion launched. No core claim changed.

Strongest supported finding remains Stage4–6 tracking Dir versus Base, with limited task/controller coverage and n=5 inference. Stage6C Base batch mismatch weakens that specific contrast. Reach failure is not explained by tested FD epsilon instability. Reward geometry and Jacobian mismatch remain important unresolved mechanisms, not a uniquely established causal explanation.

Recommended next controlled intervention: exact differentiable robot FK or a verified MuJoCo Jacobian-backed reward derivative, tested against current-state FK and directional finite differences on train/validation poses; then fixed-data frozen-model gradient comparison before retraining. This can distinguish reward-Jacobian error from dynamics-Jacobian error. Estimate: 0 control steps for FK/Jacobian validation; <=20k control steps for frozen-policy comparisons, CPU minutes, no large GPU. Any retraining should retain Base/Dir seed pairing and fresh validation episodes. This cycle has not implemented that interface or claimed a successful rescue.

Budget proposal only if a future method passes two-seed gate: five seeds approximately 5×110s CPU for this small four-model validation setup, ~160,800 collected/label/evaluation control steps; full OOD and diagnostic grids add cost. Independent 10-seed confirmation requires a frozen candidate and human approval. Do not extrapolate tiny-network runtime to visual agents.

Literature screening highlights GAMPS/PAML and action-conditioned world models as direct novelty challenges; recent execution-interface work also narrows broad claims. Equation-level comparisons and external baseline adaptation remain open. Venue assessment favors robot-learning venues for current scope; CVPR needs a real perception contribution, not an appended image encoder.
''')
print('decisions written')
