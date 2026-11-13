"""Rebuild derived CSV, figures and five research reports from preserved raw arrays."""
import json,csv,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.eval_metrics import cosine
R=ROOT/'runs';O=R/'stage8_2_summary';O.mkdir(exist_ok=True);FIG=ROOT/'reports/figures/stage8_2';FIG.mkdir(parents=True,exist_ok=True)
def read(d,n):return json.loads((R/d/n).read_text())
def save(n,v):(O/n).write_text(json.dumps(v,indent=2))
def csvout(n,rows):
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'---|'*len(head)]+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])
def report(n,s):(ROOT/'reports'/('stage8_2_'+n+'.md')).write_text(s)
def mean(a):return float(np.mean(a))
s=read('stage8_2a_single_step','single_step.json');m=read('stage8_2a_single_step','multistep.json');q=read('stage8_2_numerical','validated_metrics.json');local=read('stage8_2_numerical','local_epsilon.json');direct=read('stage8_2_localization_v2','direct_actuator_dynamics.json');num=read('stage8_2a_single_step','numerical_checks.json');refs=read('stage8_2a_single_step','references.json');pol=read('stage8_2d_frozen_policy','policies.json');rows=read('stage8_2d_frozen_policy','per_start.json');ev=read('stage8_2_localization_v2','optimizer_horizon.json');train=read('stage8_2_localization_v2','train_true_gradients.json')
common='Exploratory seed 20, frozen Stage 7C.2 Dir-Random checkpoint. Six held-out trajectory starts are not six independent training seeds. No model training, no test-driven epsilon/LR selection. Reproducer: `analysis/stage8_2_reports.py`.\n\n'
jrows=[]
for r,d in zip(s,direct):
 row=dict(start=r['start'],episode=r['episode'],forward_delta_normalized_mse=r['forward_nmse'],A_relative=r['A']['relative'],A_cosine=r['A']['cosine'],B_action_relative=r['B_action']['relative'],B_action_cosine=r['B_action']['cosine'],B_action_true_norm=r['B_action']['true_norm'],F_A_cosine=d['A']['cosine'],F_B_u_cosine=d['B_executed_command']['cosine'])
 for name,cols in [('A',slice(0,28)),('B',slice(154,168))]:
  a=np.array(r['J_model'])[:28,cols];b=np.array(r['J_physical'])[:28,cols]
  for region,sl in [('q',slice(0,14)),('qdot',slice(14,28))]:row[f'{name}_{region}_cosine']=None if np.linalg.norm(b[sl])<1e-12 else cosine(a[sl],b[sl])
  u,v,vt=np.linalg.svd(b,full_matrices=False);row[name+'_singular_value_relative']=None if np.linalg.norm(v)<1e-12 else float(np.linalg.norm(np.linalg.svd(a,compute_uv=False)-v)/np.linalg.norm(v))
  row[name+'_leading_input_direction_error']=None if np.linalg.norm(b@vt[0])<1e-12 else float(np.linalg.norm((a-b)@vt[0])/np.linalg.norm(b@vt[0]))
 jrows.append(row)
csvout('single_step.csv',jrows)
quality=[]
for r,v in zip(m,q):
 assert (r['start'],r['condition'],r['horizon'])==(v['start'],v['condition'],v['horizon'])
 z=v['metrics'];quality.append(dict(start=r['start'],condition=r['condition'],horizon=r['horizon'],full_cosine=z['cosine_full'],stable_cosine=z['cosine_stable'],stable_dimensions=z['n_stable_dims'],direction_agreement=z['balanced_dir_acc'],gradient_relative_error=r['metrics']['relative_error'],gradient_norm_ratio=z['grad_norm_hat']/max(z['grad_norm_true'],1e-30),true_gradient_norm=z['grad_norm_true'],state_error=r['state_error'],sensitivity_relative=r['sensitivity_relative']))
csvout('multistep.csv',quality)
# Primary historical failure definition is strict delta < 0; preserve runner's 1e-6 variant too.
for r in rows:r['failure_eps0']=int(r['delta']<0);r['failure_eps1e6']=r['failed']
csvout('policy_per_start.csv',rows)
agg={c:{'return':mean([r['true_return'] for r in rows if r['condition']==c]),'delta':mean([r['delta'] for r in rows if r['condition']==c]),'failrate':mean([r['failure_eps0'] for r in rows if r['condition']==c]),'cosine':mean([r['full_cosine'] for r in quality if r['condition']==c and r['horizon']==8]),'stable_cosine':mean([r['stable_cosine'] for r in quality if r['condition']==c and r['horizon']==8])} for c in 'ABCD'}
gtrain=np.mean([r['g_true_h8_theta0'] for r in train],0);gtest=np.mean([r['gradients'][7] for r in refs],0);opt=[]
for c in 'ABCD':
 g=np.array(pol[c]['training_gradient']);step=np.array(pol[c]['theta']);opt.append(dict(condition=c,train_gradient_cosine=cosine(g,gtrain),update_vs_predicted_gradient=cosine(step,g),update_vs_train_true_gradient=cosine(step,gtrain),update_vs_test_true_gradient=cosine(step,gtest),linear_train_gain=float(step@gtrain),linear_test_gain=float(step@gtest),theta_saturation=pol[c]['theta_saturation']))
csvout('optimizer.csv',opt)
hor=[]
for group in ['train','test']:
 for h in [8,16]:
  r0=mean([r[f'return_h{h}'] for r in ev if r['group']==group and r['condition']=='theta0'])
  for c in 'ABCD':hor.append(dict(group=group,horizon=h,condition=c,delta=mean([r[f'return_h{h}'] for r in ev if r['group']==group and r['condition']==c])-r0))
csvout('horizon_distribution.csv',hor)
budgets={d:read(d,'budget.json') for d in ['stage8_2a_single_step','stage8_2_numerical','stage8_2d_frozen_policy','stage8_2_localization_v2']};abc=budgets['stage8_2a_single_step']['control_steps']+budgets['stage8_2_numerical']['control_steps']+budgets['stage8_2_localization_v2']['ABC_extra_steps']+1;dc=budgets['stage8_2_localization_v2']['D_total_steps'];budget=dict(ABC_control_steps=abc,ABC_limit=40000,D_control_steps=dc,D_limit=20000,replay_test_control_steps=96,failed_attempt_steps=1,failed_attempt_included_in_ABC=True,total_control_steps=abc+dc+96,device='cpu',measured_successful_run_seconds=sum(v['wall_seconds'] for v in budgets.values()),raw_budgets=budgets);save('total_budget.json',budget);save('aggregates.json',agg)
assert abc<=40000 and dc<=20000
head='| condition | H8 full / stable cosine at theta0 | H16 true return after ONE update | paired delta vs theta0 | FailRate (delta<0) |\n|---|---:|---:|---:|---:|\n'
tab=head+'\n'.join(f"| {c} | {agg[c]['cosine']:.6f} / {agg[c]['stable_cosine']:.6f} | {agg[c]['return']:.9f} | {agg[c]['delta']:+.9f} | {agg[c]['failrate']:.3f} |" for c in 'ABCD')
cor=float(np.corrcoef([r['forward_nmse'] for r in s],[r['A']['relative'] for r in s])[0,1])
report('dynamics_jacobian', '# Single-step dynamics Jacobian audit\n\n'+common+f'''The learned transition derivative is substantially wrong on these six starts. Mean state-Jacobian cosine: **{mean([r['A']['cosine'] for r in s]):.6f}**; relative Frobenius error **{mean([r['A']['relative'] for r in s]):.6f}**. This is a local finding, not a cross-seed statement.

## State and physical interface

Physical augmented state is 154 dimensional: q/qdot (28), integrator warm-start (14), target (14), filter (14), nominal FIFO (28), prior executed command (14), smoother (14), extra FIFO (28). Inputs are 14 issued incremental actions. Fixed context includes controller parameters, nominal delay, extra lag, FIFO occupancy schedule, external forces and time. Solver warm-start is numerical state, not deployable controller observation. No actuator activation state (`na=0`); nq=nv=14. The model sees x28 + observed controller state29 + action14, not hidden Controller B parameters. Hidden-state access is oracle-only.

Independent scratch environment/MjData restores the entire source snapshot for every perturbation. No live evaluation data are mutated. Execution memory is perturbed explicitly; discrete parameters are held fixed. Eight-step replay matched original environment bit for bit on six starts. A/B matrices therefore include issued-action execution, not just bare actuator physics. Four starts have exactly zero immediate physical action Jacobian because of execution delay; undefined cosine/relative error is reported null, not interpreted as success.

{table(['start','delta-normalized MSE','A relative','A cosine','issued-action B cosine'],[[r['start'],f"{r['forward_nmse']:.4f}",f"{r['A']['relative']:.4f}",f"{r['A']['cosine']:.4f}",r['B_action']['cosine']] for r in s])}

## Separate physical F(x,u) check

Bypass the software controller with the actual executed actuator target u; hold warm-start and context fixed. Direct MuJoCo replay equals the composed next state to 1e-10. Compare the identical learned F subnetwork at that u: mean A cosine {mean([r['A']['cosine'] for r in direct]):.6f}, mean B_u cosine {mean([r['B_executed_command']['cosine'] for r in direct]):.6f}. This localizes an error inside the learned physical dynamics map as well as the missing hidden execution sensitivities. Because F was trained jointly on predicted commands, evaluating at true u is a conditional privileged probe, not an independent deployment benchmark. The full oracle replaces both physical and execution derivatives; their separate causal shares remain unresolved.

## Numerical checks and units

Central local eps: q/qdot=1e-4, warm-start=1e-3, execution/action=1e-5. Independently compare scales .5 and 2 on all six starts. Maximum A-block change is {max(r['A']['relative'] for r in local):.4%}, versus roughly 100% learned error. Full augmented-J change reaches {max(r['full']['relative'] for r in local):.4%}; do not claim exact smoothness of every internal solver/controller coordinate. Base epsilon fixed before outcomes, no return-based selection. Clamps remain in the simulator, so derivatives are local piecewise derivatives, not global smoothness claims. Policy theta0 is interior; no boundary mask is conflated with numerical stability.

Native model directional AD/FD relative error {next(r['relative'] for r in num if r['type']=='model_local_direction'):.3g}. Singular values and leading input singular-direction errors, q/qdot output blocks, full raw matrices and six starting configurations are saved. Frobenius/SVD metrics mix physical units and are scale-sensitive; component blocks accompany them. Forward error here is mean(((x_hat-x_true)/ystd)^2), normalized by training DELTA standard deviations, not the historical aggregated dynamics NMSE. Pearson correlation with A-relative error is {cor:.4f} (n=6); descriptive only and cannot establish prediction/derivative independence.

Sources: `runs/stage8_2a_single_step/single_step.json`, `runs/stage8_2_localization_v2/direct_actuator_dynamics.json`, `runs/stage8_2_numerical/local_epsilon.json`; derived `runs/stage8_2_summary/single_step.csv`.
''')
report('multistep_sensitivity','# Multi-step sensitivity propagation\n\n'+common+table(['horizon']+list('ABCD'),[[h]+[f"{mean([r['full_cosine'] for r in quality if r['horizon']==h and r['condition']==c]):.6f}" for c in 'ABCD'] for h in [1,2,4,8]])+'''

A/B use learned dynamics; C/D use the augmented physical Jacobian. A/C use FK reward derivatives; B/D use true geometry derivatives. All conditions share bit-identical forward x and forward reward values. These are two derivative factors, not four deployment algorithms.

Sensitivity is the complete dx_t/dtheta (28x28), with policy feedback through x and target included automatically. Thus the recurrence includes the actor's state dependence, not only a direct du/dtheta term. Physical sensitivity comes from whole-trajectory theta +/- 0.01 rollouts. At horizons 1/2/4/8 the CSV includes state error, sensitivity error, gradient norm ratio, relative gradient error, full and stable cosine, direction agreement and true norm. A zero true gradient at horizon 1 can arise from delay; the legacy cosine helper returns zero there, which is not an observed opposite direction. Inspect norm alongside cosine.

Additional whole-policy FD at .005 and .02 produces independent stability masks (spread <= max(1e-9,.1*abs(g_.01))). All 28 coordinates are stable on all six starts at H8. Boundary masks are separately zero at theta0. `validated_metrics.json` is authoritative: the initial `multistep.json` used all-coordinate masks as placeholders, so its unvalidated stable-coordinate fields must not be used for other horizons. No dimensions were removed to improve the primary full cosine.

The native model passes a local AD/FD check (relative error 9.35e-5). On a true forward physical trajectory, full local-J chain propagation agrees with direct whole-policy FD at cosine 0.9999992 and relative error 0.0012603 (predefined start 0). This verifies actor feedback, execution memory and chain assembly at that start. It does not prove every optimizer behavior is correct or identify a deployable derivative estimator.

Single-step derivatives are already erroneous; this is not evidence for a purely long-horizon accumulation mechanism. For example, D is about 0.998 at H2 and 0.987 at H8, whereas learned-dynamics gradients deteriorate to negative alignment. No horizon 16 gradients were requested by the frozen training protocol (H8); physical return at H16 is separately analyzed.

Sources: `stage8_2a_single_step/{multistep,references,numerical_checks}.json`, `stage8_2_numerical/{policy_epsilon,validated_metrics}.json`; derived `stage8_2_summary/multistep.csv`.
''')
report('gradient_path_intervention','# Controlled gradient-path intervention\n\n'+common+'''Forward transition x always comes from the SAME frozen learned model; hidden-state forward values follow a common scratch physical transition. Custom VJP replaces only derivatives in C/D. Common hidden state does not enter the native model. All conditions share the same issued actor actions, states and reward values at theta0; this is asserted for every start. Oracle derivatives are evaluated at these common imagined augmented states, not silently at a different physical trajectory.

Reward forward value is always FK-based. B/D replace the complete position reward derivative using true position AND true J. Joint-velocity and delta-action penalties are unchanged. This is not the Stage 8.1 “true J only” operation. A/C keep FK derivatives. `training/stage8_2_dynamics.py::ReplaceJacobian` supplies the local physical 154x168 VJP; actor autograd stays identical in all conditions.

'''+tab+f'''

The return column belongs to the separate one-update check, not a gradient intervention at fixed theta. H8 D-B cosine differences: {read('stage8_2a_single_step','gate.json')['D_minus_B']}. All six improve. C-A also improves all six. Thus the complete transition sensitivity is a causal contributor to the fixed-checkpoint policy-gradient error under this diagnostic, even with identical forward prediction error. This does not establish that a particular A block, B block, hidden memory or parameterization is the sole cause. Bare F(x,u) derivative errors show execution omission is not the entire explanation.

Whole-policy FD does NOT validate a custom substituted derivative by perturbing its frozen forward function: that would intentionally return the learned derivative. Instead we verify the custom VJP algebra, independently finite-difference the physical local transition, and compare the entire physical-forward oracle chain with direct policy FD. All 28-dimensional vectors are preserved. Stable coordinate quality comes from three epsilon scales; all H8 coordinates pass.

Gate before D: local A/B discrepancy, physical-chain cosine > .99, D-B improvement >0 on >=4/6 starts and mean >.1. Observed conditions pass. No test-return-selected settings. No deployable model or independent confirmation is claimed.

Sources: raw `stage8_2a_single_step`, validated `stage8_2_numerical`, code `training/stage8_2_dynamics.py`, tests `tests/test_stage8_2_dynamics.py`.
''')
report('frozen_policy_results','# Frozen-policy true-return check\n\n'+common+'''Budgeted local experiment: **one** Adam update, lr .03, norm clip1, theta clip[-1,1], theta0=0, first TWO pre-existing training starts, H8. All four conditions use the same frozen Dir checkpoint, initialization, starts and update schedule. This is not the 50-update Stage7 protocol and results must not be pooled with it. No new world model training. Six original held-out starts, H16 actual current MuJoCo EE reward for evaluation.

'''+tab+'''

Theta saturation is zero in all conditions; projection retains the full update (rho_proj=1). FailRate above uses historical strict delta<0. The raw runner also saved an eps=1e-6 failure indicator; both are retained in derived CSV and this reporting distinction is explicit. Task success rate is unavailable: no frozen reach success threshold exists, and none was chosen from these results. Initial gradient cosine is at COMMON theta0; final true FD vectors are also saved, but no final oracle cosine is claimed because recomputing all final local-J paths exceeds the planned budget. Final FD is single-scale .01; final stable mask is explicitly unavailable.

'''+table(['group','H','condition','paired delta'],[[r['group'],r['horizon'],r['condition'],f"{r['delta']:+.9g}"] for r in hor])+'''

The full oracle improves the TWO training starts' mean true return at both horizons but not held-out mean return. Hence horizon mismatch alone cannot explain the held-out failure. C's small improvement is exploratory; it cannot justify preferring inaccurate reward gradients or selecting a new method from the six test starts.

'''+table(['condition','train g cosine','update vs predicted g','update vs train true g','update vs test true g'],[[r['condition']]+[f"{r[k]:.6f}" for k in ['train_gradient_cosine','update_vs_predicted_gradient','update_vs_train_true_gradient','update_vs_test_true_gradient']] for r in opt])+f'''

True aggregate training-vs-test gradient cosine is {cosine(gtrain,gtest):.6f}. D's predicted training gradient is accurate, but its Adam update is not parallel to that gradient; coordinatewise preconditioning shifts alignment with the test gradient below zero. The linear prediction for D test H8 is {opt[3]['linear_test_gain']:+.9g}, versus observed {next(r['delta'] for r in hor if r['condition']=='D' and r['group']=='test' and r['horizon']==8):+.9g}. This supports a remaining interaction between selected training starts and optimizer geometry; it does not prove Adam is generally faulty. No optimizer was tuned, and no alternative optimizer was selected on test performance.

D physical control steps including supplemental train/test/horizon checks: **{dc}/20000**. Shared identical oracle queries are cached rather than counted repeatedly; all conditions have identical available stencils and no extra learned parameters. Direct-FD final vectors are diagnostic evaluation cost, not supervision. Sources: `stage8_2d_frozen_policy/`, `stage8_2_localization_v2/optimizer_horizon.json`, `train_true_gradients.json`; derived per-start CSV and optimizer CSV.
''')
decision=dict(mechanism_case='A: learned transition Jacobian erroneous; oracle restores gradient',performance_case='D: full oracle does not improve held-out mean return',new_method_success=False,five_seed_launch=False,geometry_method_reopened=False,train_test_gradient_cosine=cosine(gtrain,gtest),budget=budget);save('decision.json',decision)
report('research_decision','# Stage 8.2 research decision\n\n'+common+f'''**Mechanism: Case A. Performance: Case D. Do not start five seeds or promote an oracle into a new method.**

1. Confirmed locally: state-transition and executed-command derivatives of the frozen Dir model disagree with physical derivatives. Single-step A cosine averages {mean([r['A']['cosine'] for r in s]):.3f}; conditional actuator B_u cosine averages {mean([r['B_executed_command']['cosine'] for r in direct]):.3f}.
2. Controlled causal evidence: keeping forward trajectories/rewards identical, changing the complete dynamics derivative raises H8 mean true-gradient cosine from -.132 to .795 with original reward derivatives, and from -.116 to .987 with true reward derivatives. All six starts improve in D-B. Geometry-only correction was insufficient; complete transition derivatives matter.
3. Implementation checks: independent scratch replay exact; native model AD/FD passes; actual-physical-forward local-J chain agrees with direct policy FD at .9999992. No evidence for an actor chain-rule bug at this checked start. Hidden execution state and solver warm-start must not be omitted from this conclusion.
4. Performance stop: full oracle one-update held-out delta is {agg['D']['delta']:+.9g}; only C achieves positive mean improvement among derivative corrections. D improves the two training starts at H8 and H16, yet not the six held-out starts at either horizon. Gradient accuracy alone is insufficient for generalization.
5. Remaining explanation: training/test direction mismatch and Adam's coordinatewise update geometry are supported by measured first-order alignments, not established as universal causes. No saturation occurs. The full oracle's update/test alignment is {opt[3]['update_vs_test_true_gradient']:.6f}, despite training gradient cosine {opt[3]['train_gradient_cosine']:.6f}. Only two training starts and one update were budgeted; do not generalize this failure to a 12-start/50-update optimizer.

## Recommended next step and limits

Keep Geometry-aware EAGM stopped. Do not retrain FK, C1/C2/C3, add vision, or launch Stage9/five-seed confirmation. A justified next *exploratory* diagnostic would preregister independent training/validation trajectories and compare raw-gradient vs Adam update alignment with matched step norms, without choosing settings on these six test starts. Use additional original training starts to test whether the two-start aggregate is representative. This is an optimizer/distribution-control question before a Dynamics-Jacobian-aware method claim.

Cost estimate for a future one-update H8 full-state oracle: at most 8*(2*(154+14)+1)=2696 physical steps per start, before cache savings. Twelve training starts would cost about 32352 before evaluation, exceeding this phase's 20k D cap. Therefore it was not silently expanded. A 2–4-start independent diagnostic could fit roughly 6k–12k plus evaluation/FD, but its exact protocol and held-out data should be fixed before use. No large confirmation is proposed from current evidence.

## Integrity, cost and scope

Branch `stage8-2-dynamics-jacobian-audit`; base HEAD `bcdce42f7f93cda2dc3b9b090c241b0809660187`; previous uncommitted research retained. No main merge, CVPR2027 edit, push or commit. Source preservation: `analysis/stage8_2_preservation/`.

A-C and numerical/localization diagnostics used {abc} physical control steps (declared cap40000, separate from D). D used {dc}/20000. Replay test96. Total **{abc+dc+96}** including one failed-step attempt. Failed localization attempt used a wrong attribute name; its traceback and cost are preserved, then a separate v2 run completed. Static geometry calls and CPU wall time are separately recorded in each raw budget. No GPU training or model-parameter changes occurred. The full project total is not being misreported as under20k; that cap applies to Stage8.2D in the task and was announced before execution.

Full local-J oracle queries expose hidden parameters/state only for diagnosis. Oracle not deployable; single exploratory checkpoint; no independent seeds; unscaled Jacobian norms depend on units; full hidden-state FD is not perfectly smooth. Exact numerical tolerances, all vectors, per-start outcomes, scripts, logs and SHA-256 manifest accompany the reports. No change to the paper's core claim is justified yet.
''')
# Plotting uses only above traceable raw/derived sources.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for c in 'ABCD':ax[0].plot([1,2,4,8],[mean([r['full_cosine'] for r in quality if r['condition']==c and r['horizon']==h]) for h in [1,2,4,8]],'-o',label=c)
ax[0].set(xlabel='Horizon',ylabel='True policy-gradient cosine');ax[0].legend();ax[0].axhline(0,color='gray',lw=.5)
ax[1].bar(list('ABCD'),[agg[c]['delta'] for c in 'ABCD']);ax[1].set(ylabel='Held-out true return delta (one Adam step)');ax[1].axhline(0,color='gray',lw=.5)
ax[2].scatter([r['forward_nmse'] for r in s],[r['A']['relative'] for r in s]);ax[2].set(xlabel='Delta-normalized forward MSE',ylabel='A Jacobian relative error');fig.tight_layout();fig.savefig(FIG/'mechanism_and_return.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4))
for c in 'ABCD':ax.plot([0,1],[next(r['delta'] for r in hor if r['condition']==c and r['group']==g and r['horizon']==16) for g in ['train','test']],'-o',label=c)
ax.set_xticks([0,1],['Two training starts','Six test starts']);ax.set_ylabel('H16 true return delta');ax.axhline(0,color='gray',lw=.5);ax.legend();fig.tight_layout();fig.savefig(FIG/'train_test_update.png',dpi=180);plt.close(fig)
print('reports generated',budget['total_control_steps'],agg)
