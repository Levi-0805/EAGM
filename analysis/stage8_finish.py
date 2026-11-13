from pathlib import Path
import json,csv,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';A=ROOT/'analysis';F=ROOT/'runs/stage8_failure'
def write(n,s):(R/('stage8_'+n+'.md')).write_text(s)
j=json.loads((F/'fd.json').read_text());fdrows=[]
for tag in ['zero','dir_final']:
 for e in [.001,.003,.01,.03]:
  rr=[r for r in j if r['theta']==tag and r['eps']==e];cs=[]
  for r in rr:
   v=np.array(r['g']);w=np.array(next(x['g'] for x in j if x['theta']==tag and x['start']==r['start'] and x['eps']==.01));cs.append(float(v@w/(np.linalg.norm(v)*np.linalg.norm(w))))
  fdrows.append(dict(theta=tag,epsilon=e,mean_cosine_vs_001=float(np.mean(cs)),stable_fraction=float(np.mean([np.mean(r['mask']) for r in rr]))))
(A/'stage8_fd_summary.json').write_text(json.dumps(fdrows,indent=2))
write('failure_decomposition','''# Stage 8 failure decomposition

B1 GEOMETRY — Confirmed: original FK fit and independent validation exceed 2cm; surrogate/true rankings differ. New geometry.json samples 200 states each from train/validation/test trajectories; plots show coordinate predictions and joint-error correlations. Joint correlation is observational and cannot identify causal coupling. Start FK error vs cosine/return plots have only six points. Uniform-joint validation and visited-state errors are different distributions. A controlled exact-geometry/Jacobian intervention is still needed to attribute Dir failure causally. Return-gap growth does not by itself prove FK position-error growth.

B2 FD — Actual MuJoCo current-state sites, independent scratch data, restored snapshots. Zero and final Dir θ, all six starts, epsilon {0.001,0.003,0.01,0.03}, three scales each. Repeated eps=.01 gradient is exactly reproducible on this CPU. Directions vs eps=.01 have mean cosine >0.99999; stability fraction 0.982–1.0. This does not support numerical FD noise as the dominant cause. Metadata preserves boundary masks, one-sided modes, and effective perturbations. Larger/smaller eps do not establish robustness to stochastic hardware noise. Raw: runs/stage8_failure/fd.json, stage8_fd_summary.json.

B3 DYNAMICS — Saved held-out observed actions replayed through each model for 16 steps; per-step NMSE and command MAE in multistep.json. This isolates model rollout under fixed actions; it is not a closed-loop policy rollout. Low value error cannot bound Jacobian error; high reward-Jacobian error can remain even for exact dynamics. This explains possibility, not the unique cause of observed failure.

B4 OPTIMIZATION — Saved policies contain θ saturation, rho_proj and parameter trajectories. path.json evaluates checkpoints 1/10/25/50 on all held-out starts with both objectives. Imagined return was logged on training starts before the corresponding update, while physical checks use held-out starts after update: these are not a numerical parity test. Saved traces lack full step gradients, so Adam update/gradient directional alignment cannot be reconstructed exactly; no claim is made that it follows raw gradients. H_imag=8 vs H_eval=16 is a deliberate surrogate horizon difference, requiring a separate controlled intervention. Parameter saturation is not action clipping.

B5 OBJECTIVE — Confirmed: same reward coefficients/discount but different EE maps create a real objective mismatch; corrected true path refreshes current qpos sites without disturbing warm-start state. Historical label reading is after step; no newly demonstrated command time-alignment bug. Rollouts have fixed horizon with no newly introduced early termination. Geometry error, dynamics Jacobian error and horizon effects remain confounded; numerical FD noise has been weakened as an explanation.

No additional model trained on these test diagnostics. No Reach multiseed expansion. Failure-analysis budget and wall time are in runs/stage8_failure/budget.json; 68,928 control steps, CPU.
''')
# integrity at end relative to pre-stage snapshot
old=json.loads((A/'stage8_preservation/initial_manifest.json').read_text());changed=[];missing=[]
for f,meta in old.items():
 p=ROOT/f
 if not p.exists():missing.append(f)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']:changed.append(f)
inventory={d.name:dict(files=sum(p.is_file() for p in d.rglob('*')),checkpoints=len(list(d.rglob('*.pt')))) for d in (ROOT/'runs').iterdir() if d.is_dir()}
(A/'stage8_integrity.json').write_text(json.dumps(dict(changed_prior_files=changed,missing_prior_files=missing,runs=inventory,branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),remote='TLS CA verification failed; actual remote HEAD not verified'),indent=2))
with (R/'stage8_historical_audit.md').open('a') as f:f.write('\n\nStage8 integrity: prior changed files '+str(changed)+'; missing '+str(missing)+'. See stage8_integrity.json for directory/checkpoint inventory. Stage7C2 uncommitted source backed up before branching. Remote ls-remote failed certificate verification; no insecure TLS bypass used. Configs unchanged. Stage6C code audit: reported Base uses seed+30 batches; Random/Path use seed+40. Downgrade the Base contrast pairing claim, retain Random/Path pairing.\n')
# paired config cluster sensitivity with same clusters drawn across all seeds
raw=list(csv.DictReader((A/'stage8_master_results.csv').open()));summ=json.loads((A/'stage8_statistics.json').read_text());rng=np.random.default_rng(88);sens=[]
for s in summ:
 if s['pool'] not in ['interpolation','extrapolation']:continue
 rr=[r for r in raw if r['stage']==s['stage'] and r['pool']==s['pool'] and r['hist_len']==s['hist_len']];cfg=sorted(set(r['config_id'] for r in rr if r['config_id']))
 if not cfg:continue
 seeds=sorted(set(r['seed'] for r in rr));mat=[]
 for seed in seeds:
  mat.append([np.mean([float(r['return_']) for r in rr if r['seed']==seed and r['config_id']==c and r['model']==s['a']])-np.mean([float(r['return_']) for r in rr if r['seed']==seed and r['config_id']==c and r['model']==s['b']]) for c in cfg])
 mat=np.array(mat);boot=[mat[np.ix_(rng.integers(len(seeds),size=len(seeds)),rng.integers(len(cfg),size=len(cfg)))].mean() for _ in range(3000)]
 sens.append(dict(stage=s['stage'],pool=s['pool'],hist_len=s['hist_len'],a=s['a'],b=s['b'],crossed_seed_config_ci=np.quantile(boot,[.025,.975]).tolist(),leave_one_config_out=[float(np.delete(mat,i,axis=1).mean()) for i in range(len(cfg))]))
(A/'stage8_cluster_sensitivity.json').write_text(json.dumps(sens,indent=2))
print('integrity',changed,missing)
