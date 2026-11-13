"""Traceable static figures. Sources declared in figure_sources.json."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/figures/stage8';OUT.mkdir(parents=True,exist_ok=True);sources={}
def save(name,src):
 plt.tight_layout();plt.savefig(OUT/(name+'.png'),dpi=160);plt.close();sources[name]=src
s=json.loads((ROOT/'analysis/stage8_statistics.json').read_text());fd=ROOT/'runs/stage8_failure';p=ROOT/'runs/stage7c2_true_ee_current_sites_seed20'
chosen=[x for x in s if x['a']=='eagm_dir_matched' and x['b']=='eagm_base' and x['stage'] in ['stage4_fixedB_five_seed','stage5a_confirm','stage6b1_confirm'] and x['pool']!='iid_holdout']
plt.figure(figsize=(9,4));plt.bar(range(len(chosen)),[x['mean'] for x in chosen]);plt.xticks(range(len(chosen)),[x['stage']+'\n'+x['pool'] for x in chosen],fontsize=7);plt.ylabel('Paired return improvement');save('historical_core',['analysis/stage8_master_results.csv'])
r=json.loads((p/'results.json').read_text());names=list(r)
plt.figure();plt.scatter([r[k]['theta0_gradient']['cosine_full'] for k in names],[r[k]['delta'] for k in names]);
for k in names:plt.annotate(k,(r[k]['theta0_gradient']['cosine_full'],r[k]['delta']))
plt.xlabel('True gradient cosine at theta0');plt.ylabel('True policy delta');save('gradient_return',[str(p/'results.json')])
plt.figure();x=np.arange(len(names));plt.bar(x-.18,[r[k]['true_return'] for k in names],.36,label='True');plt.bar(x+.18,[r[k]['surrogate_return'] for k in names],.36,label='Surrogate');plt.xticks(x,names);plt.legend();save('objective_ranking',[str(p/'results.json')])
o=[x for x in s if x['stage']=='stage6b1_confirm' and x['a']=='eagm_dir_matched' and x['b']=='eagm_base' and x['pool']!='iid_holdout'];plt.figure();plt.bar([x['pool'] for x in o],[x['mean'] for x in o]);plt.ylabel('Dir - Base physical return');save('ood',['analysis/stage8_master_results.csv'])
a=[x for x in s if x['stage']=='stage6b2_histc' and x['a']=='eagm_histc_dir' and x['pool']=='interpolation' and x['hist_len']=='8'];plt.figure(figsize=(9,4));plt.bar([x['b'] for x in a],[x['mean'] for x in a]);plt.xticks(rotation=20,fontsize=8);plt.ylabel('HistC minus control');save('ablation',['analysis/stage8_master_results.csv'])
g=json.loads((fd/'geometry.json').read_text());true=np.array([x['true'] for x in g]);pred=np.array([x['pred'] for x in g]);plt.figure();plt.scatter(true.ravel(),pred.ravel(),s=2,alpha=.3);plt.xlabel('MuJoCo EE coordinate (m)');plt.ylabel('FK predicted (m)');save('fk_scatter',[str(fd/'geometry.json')])
q=np.array([x['q'] for x in g]);err=np.array([x['l2'] for x in g]);corr=[np.corrcoef(q[:,i],err)[0,1] for i in range(14)];plt.figure();plt.bar(range(14),corr);plt.xlabel('Joint index');plt.ylabel('Correlation with EE error (not causal effect)');save('fk_joint',[str(fd/'geometry.json')])
# Start-matched FK vs gradient and objective discrepancy; small n explicitly.
data=__import__('torch').load(p/'data.pt',weights_only=False);gr=json.loads((p/'gradients.json').read_text());ra=json.loads((p/'reward_audit.json').read_text());e=[x['ee_l2'] for x in ra['starts']];gv=[x['metrics']['cosine_full'] for x in gr if x['model']=='dir_random' and x['theta']=='theta0'];plt.figure();plt.scatter(e,gv);plt.xlabel('FK EE error at start (m)');plt.ylabel('Dir true cosine (n=6)');save('fk_gradient',[str(p/'reward_audit.json'),str(p/'gradients.json')])
rr=list(csv.DictReader((p/'per_start.csv').open()));rr=[x for x in rr if x['model']=='dir_random'];plt.figure();plt.scatter(e,[float(x['surrogate_final_h16'])-float(x['true_final_h16']) for x in rr]);plt.xlabel('FK error at start');plt.ylabel('Surrogate - true return (n=6)');save('fk_return',[str(p/'per_start.csv'),str(p/'reward_audit.json')])
c=ROOT/'runs/stage8_c1_seed40/results.json'
if c.exists():
 j=json.loads(c.read_text());v=j['results'];j41=json.loads((ROOT/'runs/stage8_c1_seed41/results.json').read_text());plt.figure();xx=np.arange(len(v));plt.bar(xx-.18,[x['policy_return_mean'] for x in v.values()],.36,label='seed40');plt.bar(xx+.18,[j41['results'][k]['policy_return_mean'] for k in v],.36,label='seed41');plt.xticks(xx,list(v));plt.legend();plt.ylabel('Validation physical return; exploratory');save('candidate_comparison',[str(c),'runs/stage8_c1_seed41/results.json'])
 b=j['budget'];plt.figure();plt.bar(['shared train dynamics','train FD','validation FD','policy eval'],[b['train_dynamics_steps'],b['train_fd_steps'],b['validation_fd_steps'],b['policy_eval_steps']]);plt.xticks(rotation=20);plt.ylabel('Control steps');save('budget',[str(c)])
(OUT/'figure_sources.json').write_text(json.dumps(sources,indent=2))
