"""Independent paired raw-CSV audit. Seed remains the inferential unit."""
from pathlib import Path
import csv,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STAGES=['stage4_fixedB_five_seed','stage5a_confirm','stage5b_budget','stage5c_baselines','stage5e_ood','stage6b1_confirm','stage6b2_histc','stage6c2_pathdir']
def main():
 records=[];duplicates=[]
 for stage in STAGES:
  seen=set()
  for p in sorted((ROOT/'runs'/stage).glob('seed*/per_start.csv')):
   for line,r in enumerate(csv.DictReader(p.open()),2):
    x=dict(stage=stage,seed=int(r['seed']),pool=r.get('pool','fixed_B'),model=r.get('model',r.get('method')),hist_len=r.get('hist_len',''),config_id=r.get('config_id',''),episode_id=r['episode_id'],frame=r.get('frame',''),return_=float(r['return_']),source=str(p.relative_to(ROOT)),source_line=line)
    key=tuple(x[k] for k in ['stage','seed','pool','model','hist_len','config_id','episode_id','frame'])
    if key in seen:duplicates.append(key)
    seen.add(key);records.append(x)
 assert not duplicates,duplicates[:3]
 with (ROOT/'analysis/stage8_master_results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 pairs=[('eagm_dir_matched','mc'),('eagm_dir_matched','eagm_base'),('C_dir_matched','B_base_budget_matched'),('C_dir_matched','A_base_original'),('eagm_base','aux_command_mlp'),('eagm_base','obs_memory_mlp'),('eagm_hist_dir_matched','mc'),('eagm_histc_dir','eagm_hist_dir'),('eagm_histc_dir','eagm_hist_wide_dir'),('eagm_histc_dir','eagm_histc_shuf_dir'),('eagm_dir_path','eagm_dir_random'),('eagm_dir_random','eagm_base'),('eagm_dir_random','mc')]
 rng=np.random.default_rng(808);summaries=[]
 for stage,pool,hist in sorted(set((x['stage'],x['pool'],x['hist_len']) for x in records)):
  rr=[x for x in records if (x['stage'],x['pool'],x['hist_len'])==(stage,pool,hist)]
  for a,b in pairs:
   keyed=lambda model:{(x['seed'],x['config_id'],x['episode_id'],x['frame']):x for x in rr if x['model']==model}
   aa,bb=keyed(a),keyed(b)
   if not aa or not bb:continue
   assert aa.keys()==bb.keys(),(stage,pool,a,b)
   seeds=sorted(set(k[0] for k in aa));values=[];seedrows=[];groups={}
   for seed in seeds:
    keys=[k for k in aa if k[0]==seed];diff=np.array([aa[k]['return_']-bb[k]['return_'] for k in keys]);values.append(diff)
    seedrows.append(dict(seed=seed,a_mean=float(np.mean([aa[k]['return_'] for k in keys])),b_mean=float(np.mean([bb[k]['return_'] for k in keys])),delta=float(diff.mean()),n_starts=len(keys)))
    for k,d in zip(keys,diff):groups.setdefault(k[1] or ('episode_'+k[2]),[]).append(float(d))
   d=np.array([v.mean() for v in values]);n=len(d);boot=d[rng.integers(n,size=(20000,n))].mean(1)
   hier=[]
   for _ in range(3000):
    chosen=rng.integers(n,size=n);hier.append(np.mean([rng.choice(values[i],len(values[i]),replace=True).mean() for i in chosen]))
   # exact paired sign-flip test; explicitly exploratory family.
   null=np.array([np.mean(d*np.array(signs)) for signs in itertools.product([-1,1],repeat=n)])
   summaries.append(dict(stage=stage,pool=pool,hist_len=hist,a=a,b=b,n_seeds=n,mean=float(d.mean()),median=float(np.median(d)),std=float(d.std(ddof=1)),positive=int((d>0).sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),hierarchical_ci95=np.quantile(hier,[.025,.975]).tolist(),leave_one_seed_out=[float(np.delete(d,i).mean()) for i in range(n)],group_means={k:float(np.mean(v)) for k,v in groups.items()},sign_flip_p=float(np.mean(abs(null)>=abs(d.mean())-1e-15)),per_seed=seedrows))
 m=len(summaries)
 for s in summaries:s['bonferroni_p']=min(1.,s['sign_flip_p']*m)
 (ROOT/'analysis/stage8_statistics.json').write_text(json.dumps(summaries,indent=2))
 lines=['# Stage 8 historical audit','','Raw source: `analysis/stage8_master_results.csv`; every observation carries file and CSV line. Script: `analysis/stage8_statistics.py`. No cross-stage pooling. SD uses ddof=1; 20,000 seed bootstrap draws; hierarchical sensitivity uses 3,000 paired seed/start draws. Starts are resampled jointly across methods. These CIs assume sampled seeds are exchangeable and are not guarantees with n=5.','','| Stage | pool/history | contrast | Δ mean | median | SD | positive | seed CI | hierarchical CI | LOO range |','|---|---|---|---:|---:|---:|---:|---|---|---|']
 for s in summaries:
  lines.append(f'| {s["stage"]} | {s["pool"]}/{s["hist_len"]} | {s["a"]} − {s["b"]} | {s["mean"]:.7g} | {s["median"]:.7g} | {s["std"]:.6g} | {s["positive"]}/{s["n_seeds"]} | {s["ci95"]} | {s["hierarchical_ci95"]} | {min(s["leave_one_seed_out"]):.6g}, {max(s["leave_one_seed_out"]):.6g} |')
 lines+=['','Per-seed raw means/deltas and configuration/trajectory sensitivity are in stage8_statistics.json. Pooling starts as independent seeds is prohibited. Episode IDs across different seeds do not identify the same trajectory; episode-group summaries are descriptive only. OOD config groups have meaningful shared config IDs. Hierarchical resampling here treats saved starts as trajectory clusters only when one start per episode; inspect counts before using this sensitivity as inference.','',f'{m} exploratory contrasts were inspected. Exact two-sided sign-flip p with five seeds cannot be below 0.0625 even with five positive signs; a percentile bootstrap CI excluding zero is not equivalent to strong hypothesis-test evidence. Bonferroni values for the complete inspected family are stored. This family is broader than historical preregistered primary contrasts; it illustrates selection risk, not a retroactive replacement of preregistration.','', 'Stage 5B/C CSVs must be compared within their own evaluation subsets. Repeated seeds across stages and repeated configurations are not additional independent evidence. HistC reverse-order control and probe leakage limit mechanistic conclusions. Existing result provenance cannot establish that no files were overwritten before this audit; preservation hashes only establish the current snapshot.']
 (ROOT/'reports/stage8_historical_audit.md').write_text('\n'.join(lines)+'\n')
 print(len(records),'observations',len(summaries),'contrasts')
if __name__=='__main__':main()
