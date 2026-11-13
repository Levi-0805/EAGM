"""Rebuild manuscript tables/figure from archived raw-data-linked statistics."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;(P/'data').mkdir(exist_ok=True)
stats=json.loads((ROOT/'analysis/stage8_statistics.json').read_text());master=list(csv.DictReader((ROOT/'analysis/stage8_master_results.csv').open()));choices=[('stage4_fixedB_five_seed','eagm_dir_matched','mc','4: Dir--Mc'),('stage4_fixedB_five_seed','eagm_dir_matched','eagm_base','4: Dir--Base'),('stage5a_confirm','eagm_dir_matched','mc','5A: Dir--Mc'),('stage5a_confirm','eagm_dir_matched','eagm_base','5A: Dir--Base'),('stage5b_budget','C_dir_matched','B_base_budget_matched','5B: budget')];selected=[];seedrows=[];sources={};cache={}
for stage,a,b,label in choices:
 stat=next(r for r in stats if r['stage']==stage and r['a']==a and r['b']==b and r['pool']=='fixed_B');selected.append((label,stat))
 rs=[r for r in master if r['stage']==stage and r['model'] in [a,b] and r['pool']=='fixed_B'];paired={}
 for r in rs:
  path=r['source'];p=ROOT/path
  if path not in cache:cache[path]=list(csv.DictReader(p.open()));sources[path]=hashlib.sha256(p.read_bytes()).hexdigest()
  raw=cache[path][int(r['source_line'])-2];assert abs(float(raw['return_'])-float(r['return_']))<1e-14
  key=(r['seed'],r['episode_id'],r['frame'],r['config_id']);paired.setdefault(key,{})[r['model']]=float(r['return_'])
 for seed in sorted({k[0] for k in paired},key=int):
  ds=[v[a]-v[b] for k,v in paired.items() if k[0]==seed];seedrows.append(dict(stage=stage,contrast=label,seed=seed,paired_delta=float(np.mean(ds)),n_paired_starts=len(ds)))
 assert abs(np.mean([r['paired_delta'] for r in seedrows if r['contrast']==label])-stat['mean'])<1e-12
with (P/'data/core_seed_differences.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(seedrows[0]));w.writeheader();w.writerows(seedrows)
tex=r'''\begin{center}\small
\begin{tabular}{lrrr}\toprule
Comparison & Mean & 95\% CI & Wins\\\midrule
'''
for label,s in selected:tex+=f"{label} & {s['mean']*1e4:.2f} & [{s['ci95'][0]*1e4:.2f}, {s['ci95'][1]*1e4:.2f}] & {s['positive']}/5 \\\\\n"
tex+=r'''\bottomrule\end{tabular}
\refstepcounter{table}\label{tab:main}
\par\smallskip Table \thetable: Paired return differences in units of $10^{-4}$. CIs are within-stage seed bootstrap intervals; no cross-stage pooling.
\end{center}
''';(P/'tables/main_results.tex').write_text(tex)
a=list(csv.DictReader((ROOT/'runs/stage8_3_alignment/aggregate.csv').open()));tex=r'''\begin{center}\small
\begin{tabular}{lrr}\toprule
Oracle update & Validation & Fresh test\\\midrule
'''
for k in [2,4]:
 for opt in ['raw','adam_first']:
  vals=[float(next(r['mean_delta_h16'] for r in a if r['policy']==f'K{k}_oracle_{opt}' and r['group']==g))*1e5 for g in ['validation','fresh_test']];tex+=f"K{k} {'Raw' if opt=='raw' else 'Adam'} & {vals[0]:+.3f} & {vals[1]:+.3f} \\\\\n"
tex+=r'''\bottomrule\end{tabular}
\par\smallskip Mean true-return changes in units of $10^{-5}$. Two starts per column, one frozen checkpoint; no significance claim.
\end{center}
''';(P/'tables/stage8_3.tex').write_text(tex)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(5,3));mu=np.array([s['mean']*1e4 for _,s in selected]);lo=np.array([s['ci95'][0]*1e4 for _,s in selected]);hi=np.array([s['ci95'][1]*1e4 for _,s in selected]);ax.errorbar(mu,np.arange(5),xerr=np.stack([mu-lo,hi-mu]),fmt='o',capsize=3);ax.set_yticks(np.arange(5),[l for l,_ in selected]);ax.axvline(0,color='gray',lw=.7);ax.set_xlabel('Paired return difference (1e-4)');ax.invert_yaxis();fig.tight_layout();fig.savefig(P/'figures/historical.pdf');plt.close(fig)
for f in ['analysis/stage8_statistics.json','analysis/stage8_master_results.csv','runs/stage8_3_alignment/aggregate.csv']:sources[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
for r in master:
 if r['stage'].startswith(('stage4','stage5','stage6')):
  path=r['source']
  if path not in sources:sources[path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
(P/'data/source_manifest.json').write_text(json.dumps(sources,indent=2));print('Raw-linked table verification passed:',len(seedrows),'seed contrasts;',len(sources),'source files')
