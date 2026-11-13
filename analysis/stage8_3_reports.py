import json,csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.eval_metrics import cosine
P=ROOT/'runs/stage8_3_alignment'
def load(n):return json.loads((P/n).read_text())
def dump(n,v):(P/n).write_text(json.dumps(v,indent=2))
def csvwrite(n,rows):
 with (P/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pol=load('policies.json');rr=load('returns.json');refs=load('references.json');grad=load('training_gradients.json');rows=[];align=[];agg=[]
for key,p in pol.items():
 for group in ['train','validation','fresh_test','historical_test']:
  for r in [r for r in rr if r['group']==group and r['policy']==key]:
   zero=next(z for z in rr if z['group']==group and z['start']==r['start'] and z['policy']=='theta0');rows.append(dict(policy=key,K=p['K'],source=p['source'],optimizer=p['optimizer'],group=group,start=r['start'],active_training_start=group=='train' and r['start']<p['K'],return_h16=r['return_h16'],delta_h16=r['return_h16']-zero['return_h16'],delta_h8=r['return_h8']-zero['return_h8']))
  gs=[r['g'] for r in refs if r['group']==group and (group!='train' or r['start']<p['K'])];gt=np.mean(gs,axis=0);g=np.array(p['g']);u=np.array(p['theta']);al=dict(policy=key,group=group,gradient_vs_true=cosine(g,gt),update_vs_true=cosine(u,gt),update_vs_estimate=cosine(u,g),update_norm=float(np.linalg.norm(u)),true_gradient_norm=float(np.linalg.norm(gt)),linear_gain=float(u@gt),coordinate_agreement=float(np.mean(np.sign(u[np.abs(gt)>1e-12])==np.sign(gt[np.abs(gt)>1e-12]))));align.append(al)
  data=[r for r in rows if r['policy']==key and r['group']==group and (group!='train' or r['active_training_start'])];agg.append(dict(policy=key,group=group,n=len(data),mean_delta_h16=float(np.mean([r['delta_h16'] for r in data])),mean_delta_h8=float(np.mean([r['delta_h8'] for r in data])),failrate=float(np.mean([r['delta_h16']<0 for r in data])),positive=sum(r['delta_h16']>0 for r in data),update_true_cosine=al['update_vs_true']))
contrast=[]
for hyp in ['H1','H2']:
 for level in (['raw','adam_first'] if hyp=='H1' else [2,4]):
  a=f'K4_oracle_{level}' if hyp=='H1' else f'K{level}_oracle_raw';b=f'K2_oracle_{level}' if hyp=='H1' else f'K{level}_oracle_adam_first'
  for group in ['validation','fresh_test','historical_test']:
   aa=next(r for r in agg if r['policy']==a and r['group']==group);bb=next(r for r in agg if r['policy']==b and r['group']==group);ds=[r['delta_h16']-next(s['delta_h16'] for s in rows if s['policy']==b and s['group']==group and s['start']==r['start']) for r in rows if r['policy']==a and r['group']==group];contrast.append(dict(hypothesis=hyp,level=str(level),group=group,a=a,b=b,return_difference=aa['mean_delta_h16']-bb['mean_delta_h16'],alignment_difference=aa['update_true_cosine']-bb['update_true_cosine'],positive_starts=sum(v>0 for v in ds),n=len(ds)))
gates={h:all(r['return_difference']>0 and r['alignment_difference']>0 and (r['group']!='fresh_test' or r['positive_starts']==2) for r in contrast if r['hypothesis']==h and r['group'] in ['validation','fresh_test']) for h in ['H1','H2']};dump('decision.json',dict(gates=gates,stage8_4_authorized=False,five_seeds=False,interpretation='fresh-test improvements conflict with validation; neither mechanism screen passes',preregistered_commit='0c5a2fe'))
csvwrite('per_start.csv',rows);csvwrite('alignment.csv',align);csvwrite('aggregate.csv',agg);csvwrite('contrasts.csv',contrast)
# Training representation itself, independent of optimizer.
representation=[]
for K in [2,4]:
 gt=np.mean([r['g'] for r in refs if r['group']=='train' and r['start']<K],axis=0)
 for group in ['validation','fresh_test','historical_test']:
  gr=np.mean([r['g'] for r in refs if r['group']==group],axis=0);representation.append(dict(K=K,group=group,cosine=cosine(gt,gr),train_norm=float(np.linalg.norm(gt)),reference_norm=float(np.linalg.norm(gr))))
csvwrite('true_gradient_representation.csv',representation)
body='''# Stage 8.3 optimizer–data alignment audit

Preregistered commit `0c5a2fe`, before new-test collection. Frozen seed20 Dir checkpoint; no training, no independent model-seed confirmation. Complete protocol: configs/stage8_3_preregistered.json. Raw data and runner: runs/stage8_3_alignment/ and analysis/stage8_3_run.py; regeneration: analysis/stage8_3_reports.py.

K2 training episodes0,2; K4 adds3,5, original training split, frame0. Validation episodes22,30. Fresh collection seed83017, episodes0,1: distinct from collection seed20 despite repeated numeric episode IDs. These two fresh trajectories are a one-shot exploratory check, not a new training seed or broad controller coverage (nominal delays0,1 only). Historical six test starts are explicitly previously inspected. Training summaries below restrict evaluation to the K active training starts, not all four.

Two gradient sources share TRUE reward derivatives: B learned transition vs D full privileged transition oracle. At each K, source gradients identical between optimizers. One fresh Adam step vs raw direction, both rescaled to L2=0.03 sqrt(28); no projection or saturation. This differs from unnormalized Stage8.2 Adam, so numbers must not be pooled. H8 gradient, H16 primary true EE return; H8 returns also retained. FD eps .01 at training points; .005/.01 at validation/fresh points. Reference gate passed. Numerical thresholds were fixed in the runner before results; the preregistration commit states the stability rule, while the runner additionally uses cosine>.999 and at least26 stable coordinates. Full coordinates remain primary; no weight/epsilon/start tuning followed outcomes.

| policy | validation delta | new-test delta | new-test positive | historical delta |
|---|---:|---:|---:|---:|
'''
for key in pol:
 values={g:next(r for r in agg if r['policy']==key and r['group']==g) for g in ['validation','fresh_test','historical_test']};body+=f"| {key} | {values['validation']['mean_delta_h16']:+.8g} | {values['fresh_test']['mean_delta_h16']:+.8g} | {values['fresh_test']['positive']}/2 | {values['historical_test']['mean_delta_h16']:+.8g} |\n"
body+='\n## True training-gradient representation\n\n| K | group | aggregate cosine |\n|---|---|---:|\n'
for r in representation:body+=f"| {r['K']} | {r['group']} | {r['cosine']:.6f} |\n"
body+='''
## Decision

Neither H1 nor H2 passes the preregistered joint validation/fresh-test screen. Oracle K4 improves both fresh starts but worsens both validation starts. Raw has better fresh-test return at K4, but worse validation return at both K values and worse fresh-test return at K2. Increasing training coverage rotates the gradient toward one distribution and away from another; two versus four fixed starts do not establish a monotone coverage law. This does not show that Adam alone caused Stage8.2 failure, nor that an inaccurate learned gradient is preferable in general.

No Stage8.4 method promotion, no five/ten seeds, no additional optimizer/start/step sweeps. Preserve the mixed outcome. The next research question, if separately pursued, concerns objective and start-distribution heterogeneity rather than tuning a method on these now-inspected trajectories. A larger audit would need a separately fixed sampling plan and cost; no large confirmation is justified by these data.

## Accounting and reproducibility

'''+f"Actual control steps: **{load('budget.json')['control_steps']}/20000**, phases {load('budget.json')['phases']}. Worst-case preregistered plan18576; exact-query caching reduces actual counts. CPU wall time {load('budget.json')['wall_seconds']:.2f}s, no GPU training. Parameter counts/checkpoint identical in all cells. Collection shared; privileged oracle query cost explicitly included, not claimed as a deployable budget-equivalent algorithm. Tests compare first Adam to torch and equal update norms without physical steps. No task success threshold is invented; FailRate is delta<0. No multi-seed CI is computed from two fresh starts.\n"
(ROOT/'reports/stage8_3_optimizer_data_alignment.md').write_text(body)
(ROOT/'reports/stage8_3_research_decision.md').write_text('# Stage 8.3 decision\n\n**Stop this screening route; neither preregistered joint gate passed.**\n\n'+body[body.index('## Decision')+len('## Decision'):])
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for j,group in enumerate(['validation','fresh_test']):
 for opt,offset in [('raw',-.15),('adam_first',.15)]:
  vals=[next(r['mean_delta_h16'] for r in agg if r['group']==group and r['policy']==f'K{k}_oracle_{opt}') for k in [2,4]];ax[j].bar(np.array([0,1])+offset,vals,width=.3,label=opt)
 ax[j].set_xticks([0,1],['K2','K4']);ax[j].set_title(group);ax[j].axhline(0,color='black',lw=.6);ax[j].set_ylabel('True EE return delta');ax[j].legend()
fig.tight_layout();fig.savefig(P/'optimizer_data.pdf');fig.savefig(P/'optimizer_data.png',dpi=180);plt.close(fig);print(gates)
