"""Report and plot generation from immutable audit arrays; no simulation."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'runs/stage8_1_geometry';R=ROOT/'reports';FIG=R/'figures/stage8_1';FIG.mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((P/n).read_text())
def cosine(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def write(name,title,body):
 p=R/f'stage8_1_{name}.md'
 if p.exists():raise FileExistsError(p)
 p.write_text('# '+title+'\n\nExploratory seed20 frozen-model causal diagnostic. No new world-model training, no independent confirmation. Sources: runs/stage8_1_geometry/; script analysis/stage8_1_reports.py.\n\n'+body+'\n')
def main():
 ss=read('static.json')+read('physical_state_geometry.json');meta=read('geometry_metadata.json');checks=read('geometry_fd_checks.json');groups=sorted(set(x['group'] for x in ss));summary={}
 for group in groups:
  rr=[x for x in ss if x['group']==group];summary[group]={k:float(np.mean([x[k] for x in rr])) for k in ['position_rmse','position_mae','jacobian_rmse','jacobian_relative','jacobian_cosine','reward_gradient_cosine','reward_gradient_relative']}
  for arm,sl in [('left',slice(0,3)),('right',slice(3,6))]:
   summary[group][arm+'_position_rmse']=float(np.sqrt(np.mean([(np.asarray(x['p_fk'])[sl]-np.asarray(x['p_true'])[sl])**2 for x in rr])))
   summary[group][arm+'_jacobian_rmse']=float(np.sqrt(np.mean([(np.asarray(x['J_fk'])[sl]-np.asarray(x['J_true'])[sl])**2 for x in rr])))
  summary[group]['joint_jacobian_rmse']=np.sqrt(np.mean([(np.array(x['J_fk'])-np.array(x['J_true']))**2 for x in rr],axis=(0,1))).tolist()
  summary[group]['position_pooled_rmse']=float(np.sqrt(np.mean([(np.array(x['p_fk'])-np.array(x['p_true']))**2 for x in rr])))
  summary[group]['trueJ_fkPosition_cosine']=float(np.mean([cosine(x['g_trueJ_fkPosition'],x['g_true']) for x in rr]));summary[group]['fkJ_truePosition_cosine']=float(np.mean([cosine(x['g_fkJ_truePosition'],x['g_true']) for x in rr]))
 (P/'static_summary.json').write_text(json.dumps(summary,indent=2))
 lines=['| Distribution | Position pooled RMSE m | J RMSE | relative J error | J cosine | reward gradient cosine |','|---|---:|---:|---:|---:|---:|']
 for k,v in summary.items():lines.append(f'| {k} | {v["position_pooled_rmse"]:.6f} | {v["jacobian_rmse"]:.6f} | {v["jacobian_relative"]:.4f} | {v["jacobian_cosine"]:.4f} | {v["reward_gradient_cosine"]:.4f} |')
 write('geometry_audit','Geometry and Jacobian audit','\n'.join(lines)+f'\n\nGeometry FD max error across three eps and five random configurations: {max(x["max_abs"] for x in checks):.3g}. Static phase control steps=0. Mapping:\n\n```json\n'+json.dumps(meta,indent=2)+'\n```\n\nAll joints asserted hinge, nq=nv=14; qpos/dof addresses identical. Jacobians are world translation metres/radian, ordered left xyz then right xyz. Robot body tree controls which arm columns contribute; no manual column permutation. Left/right and per-joint errors in static_summary.json. Independent MjData is used for mj_forward/mj_jacSite. Train/test samples retain original episode IDs; there is no new adjacent-frame split. Random states respect limits. Imagined theta0/old-final states are explicitly model predictions, not physical trajectories. Supplemental actual theta-state observations, if present, are separately labeled and charged to the control budget.\n')
 lines=['Actual reward: −mean(EE_error²) −0.0005 mean(qdot²) −0.05 mean((a−a_prev)²), discount 0.99. Position gradient is −Jᵀ(p−goal)/3 with J∈R^(6×14), W=I/6. There is **no EE velocity reward**, so no omitted derivative of J(q) qdot. qdot derivative=−0.001 qdot/14; action derivative=−0.1(a−a_prev)/14; previous action has opposite derivative and remains in the BPTT graph. Fixed horizon, unchanged control_dt and no new termination rule.','', '| Distribution | native reward grad cos | true J / FK residual cos | FK J / true residual cos |','|---|---:|---:|---:|']
 for k,v in summary.items():lines.append(f'| {k} | {v["reward_gradient_cosine"]:.5f} | {v["trueJ_fkPosition_cosine"]:.5f} | {v["fkJ_truePosition_cosine"]:.5f} |')
 lines+=['','Reward-value errors and goals are stored per sample in static.json. This factorial decomposition changes J and positional residual independently at the state level. Different goals come from original episodes; random configurations cycle held-out goals and do not constitute independent task seeds. state derivative identities checked in reward_derivative_checks.json. No causal claim follows solely from these correlations.']
 write('reward_jacobian','Reward Jacobian decomposition','\n'.join(lines))
 iv=read('intervention.json');lines=['| Model | condition | full cosine | stable cosine | relative gradient error |','|---|---|---:|---:|---:|']
 for model in ['mc','base','dir_random']:
  for c in 'ABCD':
   rr=[x['metrics'] for x in iv if x['model']==model and x['condition']==c];lines.append(f'| {model} | {c} | {np.mean([x["cosine_full"] for x in rr]):.5f} | {np.mean([x["cosine_stable"] for x in rr]):.5f} | {np.mean([x["relative_error"] for x in rr]):.5f} |')
 write('gradient_intervention','Controlled gradient interventions','\n'.join(lines)+'\n\nAll rows share theta0, held-out starts, H=8, fixed dynamics and controller recursion. A=surrogate value/gradient, B=true value/gradient, C=surrogate value/true complete reward gradient, D=true value/surrogate complete reward gradient. C/D use scalar stop-gradient corrections. A vs C isolates complete reward derivative at identical forward reward values; it does not isolate geometric J from position residual. Static factorial analysis supplies that distinction. Native-rollout value/gradient parity and A/C, B/D forward and A/D, B/C backward equality tested. Full 28-coordinate gradient arrays and norm/sign diagnostics stored in intervention.json. Analytical TrueFK backward is first-order only; it must not be used for second-order Dir model training without extending/testing it.\n')
 s=read('summary.json');rows=list(csv.DictReader((P/'per_start.csv').open()));lines=['| condition | true return | paired Δ vs theta0 | FailRate | theta0 cosine | own-final cosine | stable final cosine | theta saturation |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for c,v in s.items():lines.append(f'| {c} | {v["true_return"]:.8f} | {v["delta"]:.8f} | {v["fail_rate"]:.3f} | {v["theta0_metrics"]["cosine_full"]:.5f} | {v["final_metrics"]["cosine_full"]:.5f} | {v["final_metrics"]["cosine_stable"]:.5f} | {v["theta_sat"]:.3f} |')
 aa=[float(x['true_return']) for x in rows if x['condition']=='A'];bb=[float(x['true_return']) for x in rows if x['condition']=='B'];diff=np.array(bb)-aa
 lines += ['',f'B−A per start: {diff.tolist()}; positive {int((diff>0).sum())}/6; mean {diff.mean():.8g}. These are six episodes from ONE exploratory seed, not six independent model-training seeds.','', 'Frozen Dir only; same initial theta0, 12 original train starts, 50 Adam steps at 0.03, gradient norm clipping 1 and projection [-1,1]. Model and FK weights frozen. No metric-based checkpoint selection. Full optimization traces include actual update/gradient cosine and rho_proj. A/D final theta and B/C final theta asserted bit-identical; hence final FD queries are shared within each pair. Value-only correction cannot change deterministic Adam updates when gradients and stopping schedule are unchanged.', '', 'Reference FD: theta0 three scales, final two scales, eps=.01 feasible perturbations. Different mask strengths are explicitly reported; primary full-coordinate cosine avoids masking selection. H gradient=8; H physical return=16. Final gradient comparisons occur at different theta and should not be interpreted as the common-theta intervention effect.','', 'Budget: '+json.dumps(read('budget.json')['by_phase'])]
 write('frozen_policy_comparison','Frozen-policy physical comparison','\n'.join(lines))
 improve_grad=s['B']['theta0_metrics']['cosine_full']>s['A']['theta0_metrics']['cosine_full'];improve_return=diff.mean()>0
 case='B' if s['B']['theta0_metrics']['cosine_full']<=0 or s['B']['delta']<=0 else ('A' if improve_grad and improve_return else ('D' if improve_grad else 'B'))
 decision=dict(case=case,common_theta_gradient_improved=improve_grad,mean_return_improved=bool(improve_return),positive_starts=int((diff>0).sum()),five_seed_launch=False,independent_confirmation=False)
 (P/'decision.json').write_text(json.dumps(decision,indent=2))
 write('research_decision','Stage 8.1 research decision','```json\n'+json.dumps(decision,indent=2)+'\n```\n\nCase B here means partial mitigation without restored reliability: common-theta mean cosine remains negative and true return remains below theta0. Small numerical improvements do not meet the requested significant recovery criterion. No significance claim is made. Geometry derivative replacement is an intervention on the complete reward derivative with dynamics frozen; it is not a proof that Jacobian error alone caused all Stage7 failure. Position residual and J both contribute. The selected Dir model was originally trained with surrogate geometry and true FD labels, so geometry replacement may expose training/model derivative compensation. No retraining or cross-seed validation was performed.\n\nDo not launch five seeds from one exploratory seed. If both gradient and return improve, next is one fresh, preregistered engineering replication with verified geometry and unchanged budgets; otherwise investigate dynamics Jacobians/horizon/update direction before architecture changes. C1/C2 remain exploratory, C3 requires independent calibration episodes. No new losses, vision, hardware or paper-core claim changes. Exact geometry here is an oracle diagnostic, not yet a deployable learned-method contribution.\n')
 # Traceable figures
 sources={}
 def save(name,source):plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=160);plt.close();sources[name]=source
 plt.figure(figsize=(8,4));plt.bar(groups,[summary[k]['jacobian_relative'] for k in groups]);plt.xticks(rotation=20,fontsize=8);plt.ylabel('Relative geometric Jacobian error');save('jacobian_error','static.json')
 plt.figure();plt.scatter([x['position_rmse'] for x in ss],[x['reward_gradient_cosine'] for x in ss],s=5,alpha=.4);plt.xlabel('Position RMSE (m)');plt.ylabel('Reward-gradient cosine');save('position_reward_gradient','static.json')
 plt.figure();plt.bar(list(s),[v['theta0_metrics']['cosine_full'] for v in s.values()]);plt.ylabel('Common-theta true policy-gradient cosine');save('intervention_cosine','intervention.json')
 plt.figure();plt.bar(list(s),[v['true_return'] for v in s.values()]);plt.ylabel('True physical return');save('physical_return','per_start.csv')
 plt.figure();plt.bar(range(6),diff);plt.axhline(0,color='k',linewidth=.5);plt.xlabel('Held-out episode index');plt.ylabel('True return B - A');save('paired_return','per_start.csv')
 (FIG/'sources.json').write_text(json.dumps(sources,indent=2));print(decision)
if __name__=='__main__':main()
