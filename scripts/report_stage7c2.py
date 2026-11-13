"""Recompute summaries from stored artifacts; no training or simulation."""
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/stage7c2_true_ee_current_sites_seed20'
def main():
 r=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'reward_audit.json').read_text());g=json.loads((OUT/'gradients.json').read_text());b=json.loads((OUT/'budget.json').read_text());rows=list(csv.DictReader((OUT/'per_start.csv').open()))
 b['train_dynamics_steps']=4800
 b['validation_collection_steps']=1200
 b['test_collection_steps']=1200
 b['dir_total_training_steps']=4800+b['train_label_steps']
 b['gradient_audit_queries']=sum(x['fd']['simulator_queries']+56 for x in g)
 b['policy_eval_queries']=len(rows)*8
 b['all_fd_and_eval_queries']=b['train_label_queries']+b['gradient_audit_queries']+b['policy_eval_queries']
 assert b['gradient_audit_steps']==b['gradient_audit_queries']*8
 assert b['evaluation_control_steps']==b['gradient_audit_steps']+len(rows)*4*(4+16)
 (OUT/'budget.json').write_text(json.dumps(b,indent=2))
 common='Seed20 is exploratory and was used to discover this protocol issue. No multiseed experiment is authorized by this report. Old Stage 7C.1 artifacts are unchanged.\n\n'
 def write(name,title,body):
  p=ROOT/'reports'/name
  if p.exists():raise FileExistsError(p)
  p.write_text('# '+title+'\n\n'+common+body)
 errors={}
 for key in ['ee_l2','abs_reward_error','relative_reward_error']:
  v=[x[key] for x in a['starts']];errors[key]={'mean':float(np.mean(v)),'max':float(np.max(v))}
 for h in [4,16]:
  v=[abs(float(x[f'true_final_h{h}'])-float(x[f'surrogate_final_h{h}'])) for x in rows];errors[f'return_abs_diff_h{h}']={'mean':float(np.mean(v)),'max':float(np.max(v))}
 (OUT/'reward_summary.json').write_text(json.dumps(errors,indent=2))
 write('stage7c2_reward_audit.md','Stage 7C.2 Reward audit',f'FK training RMSE: {a["fit_rmse"]:.8f} m. Independent uniform-joint validation RMSE (1000 samples): {a["independent_rmse"]:.8f} m. Acceptance remains **0.02 m**; passed: **{a["fk_accept"]}**. No threshold was loosened. Training continued solely as the authorized objective-correction diagnostic, not as acceptance of FK quality.\n\n'+json.dumps(errors,indent=2)+'\n\nRank correlations across held-out starts (final policies):\n\n'+'\n'.join(f'- {k}: {v["rank_correlation"]:.6f}' for k,v in r.items())+'\n\nStatic errors use held-out start states; return errors cover full physical trajectories. They do not bound FK Jacobian error.\n')
 lines=['| Model | True R0 | True Rfinal | Δ true | FailRate | Surrogate Rfinal | NMSE | Command MAE |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for k,v in r.items():lines.append(f'| {k} | {v["true_theta0"]:.8f} | {v["true_return"]:.8f} | {v["delta"]:.8f} | {v["fail_rate"]:.3f} | {v["surrogate_return"]:.8f} | {v["nmse"]:.5f} | {v["command_mae"]} |')
 true_rank=sorted(r,key=lambda k:r[k]['true_return'],reverse=True);surr_rank=sorted(r,key=lambda k:r[k]['surrogate_return'],reverse=True)
 ds=[]
 for i in sorted(set(x['start'] for x in rows)):
  rr={x['model']:float(x['true_final_h16']) for x in rows if x['start']==i};ds.append(rr['dir_random']-rr['base'])
 write('stage7c2_model_comparison.md','Stage 7C.2 model comparison','\n'.join(lines)+f'\n\nTrue ranking: {true_rank}. Surrogate ranking: {surr_rank}.\n\nDir−Base per-start differences: {ds}; positive {sum(x>0 for x in ds)}/{len(ds)}. Starts are not independent training seeds.\n\nAll models use identical trajectories/splits and batch seed 50. Base/Dir share exact initialization; different architectures cannot share identical parameter tensors. Old 7C.1 Mc used batch seed 40, so this is a corrected protocol, not exact replication.\n\nBudget:\n\n```json\n'+json.dumps(b,indent=2)+'\n```\n')
 coordinate=[]
 for name in r:
  gg=[x for x in g if x['model']==name and x['theta']=='theta0']
  for group,idx in [('left_bias',list(range(7))),('right_bias',list(range(7,14))),('left_feedback',list(range(14,21))),('right_feedback',list(range(21,28)))]:
   err=np.array([np.array(x['g_model'])[idx]-np.array(x['g_true'])[idx] for x in gg])
   coordinate.append(dict(model=name,group=group,gradient_rmse=float(np.sqrt(np.mean(err**2)))))
 (OUT/'coordinate_diagnostics.json').write_text(json.dumps(coordinate,indent=2))
 lines=['| Model | theta | full cosine | stable cosine | stable sign agreement | relative gradient error |','|---|---|---:|---:|---:|---:|']
 for k,v in r.items():
  for tag in ['theta0','final']:
   z=v[tag+'_gradient'];lines.append(f'| {k} | {tag} | {z["cosine_full"]:.5f} | {z["cosine_stable"]:.5f} | {z["balanced_dir_acc"]:.5f} | {z["magnitude_error"]:.5f} |')
 write('stage7c2_true_gradient_audit.md','Stage 7C.2 True gradient audit','\n'.join(lines)+'\n\nMeans across all held-out starts. θ0 is a common comparison; final θ is model-specific. H=8 for both model and FD gradients; physical policy evaluation H=16. Full coordinate gradients, stability/boundary masks, effective perturbations, surrogate-reference diagnostics and query counts are stored in gradients.json. Training labels are separate from these held-out probes. Magnitude error is ||g_model−g_true|| / ||g_true||, not only norm-ratio error.\n')
 gain=r['dir_random']['true_return']>r['base']['true_return'];grad=r['dir_random']['theta0_gradient']['cosine_full']>r['base']['theta0_gradient']['cosine_full'];mc=r['dir_random']['true_return']>r['mc']['true_return']
 decision=dict(dir_true_return_beats_base=gain,dir_true_gradient_beats_base_at_common_theta0=grad,dir_beats_mc=mc,fk_accepted=a['fk_accept'],rankings_equal=true_rank==surr_rank,propose_pilot=gain and grad and mc and a['fk_accept'] and true_rank==surr_rank,launch_pilot=False,launch_stage7d=False)
 (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
 write('stage7c2_research_decision.md','Stage 7C.2 research decision','```json\n'+json.dumps(decision,indent=2)+'\n```\n\nThis is a protocol correction. True-FD labels were regenerated and Dir retrained for 3000 updates; existing surrogate labels were not reused. Whether supervision helps relative to Base is shown above; isolating the causal effect of switching label objectives requires a matched surrogate-label Dir control, which this four-model run does not provide.\n\nFK inadequacy or a material rank reversal blocks a clean transfer claim even if Dir beats Base. No claim of monotonic training-budget dependence is supported. Stop for human review. Any future engineering pilot uses fresh seeds 21–23; confirmation 30–39 requires separate approval.\n')
 print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
