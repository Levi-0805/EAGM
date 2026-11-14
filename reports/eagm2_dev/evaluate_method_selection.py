"""Validation-only fixed exploratory selector; permission is not a scientific gate.

Reads only config identity and validation_physical / validation_gradient fields.
Never uses development-test outcomes, and never launches training.
"""
import argparse
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SEEDS={'tracking':(200001,200002,200003),'contact':(201001,201002,201003),'vision':(202001,202002,202003)}
CANDIDATES=('delay_dir','recurrent_dir','combined_dir')
BASELINES=('ff_dir','legacy_dir')


def number(x):
 return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)


def read_validation(root,task,seed,attempt,method):
 run=root/task/f'seed{seed}'/attempt
 try:
  cfg=json.loads((run/'config.json').read_text())
  if cfg.get('smoke') or 'smoke' in attempt.lower() or 'invalid' in attempt.lower():return None
  if cfg.get('task')!=task or cfg.get('seed')!=seed or cfg.get('attempt')!=attempt:return None
  manifest=json.loads((run/'run_manifest.json').read_text())
  if manifest.get('status')!='completed' or manifest.get('invalid'):return None
  result=json.loads((run/method/'results.json').read_text())
  if result.get('status')!='completed' or result.get('method')!=method or result.get('seed')!=seed:return None
  # No physical/gradient (development_test) field access here.
  physical=result.get('validation_physical') or {};gradient=result.get('validation_gradient') or {}
  values={'return':physical.get('mean_return'),'cosine':gradient.get('mean_cosine')}
  if task!='tracking':values['success']=physical.get('success_rate')
  return {k:v if number(v) else None for k,v in values.items()}
 except (OSError,ValueError,TypeError,AttributeError):return None


def gate_status(gates,task,baseline,candidate):
 # External acceptance must be explicit for this comparison, with evidence links.
 entry=gates.get(task,{}).get(baseline,{}).get(candidate,{})
 needed=('capacity','budget','tests')
 records={name:entry.get(name) for name in needed}
 def valid(v):return isinstance(v,dict) and v.get('status') in ('pass','fail') and isinstance(v.get('evidence'),str) and bool(v['evidence'].strip())
 if any(not valid(v) for v in records.values()):return 'pending',records
 return ('fail' if any(v['status']=='fail' for v in records.values()) else 'pass'),records


def evaluate(root,attempt,gates):
 report={'phase':'exploratory_validation_selection','permission_is_scientific_pass':False,'attempt':attempt,'candidate_order':list(CANDIDATES),'baseline_resolution':'parallel fixed reports; no automatic choice between ff_dir and legacy_dir','tasks':{}}
 for task,seeds in SEEDS.items():
  task_report={}
  for baseline in BASELINES:
   if task=='vision' and baseline=='legacy_dir':
    task_report[baseline]={'status':'not_applicable','reason':'legacy physical-state model unavailable for vision','selected':None};continue
   candidates={}
   for candidate in CANDIDATES:
    rows=[];missing=[];metrics=('return','cosine') if task=='tracking' else ('return','cosine','success')
    for seed in seeds:
     left=read_validation(root,task,seed,attempt,candidate);right=read_validation(root,task,seed,attempt,baseline)
     differences={}
     for metric in metrics:
      x=(left or {}).get(metric);y=(right or {}).get(metric)
      differences[metric]=x-y if number(x) and number(y) else None
      if differences[metric] is None:missing.append({'seed':seed,'metric':metric})
     rows.append({'seed':seed,'candidate_validation':left,'baseline_validation':right,'difference':differences})
    external,checks=gate_status(gates,task,baseline,candidate)
    scientific='pending';summary={};criteria={}
    if not missing:
     for metric in metrics:
      vals=[r['difference'][metric] for r in rows]
      summary[metric]={'mean_difference':sum(vals)/3,'positive':sum(v>0 for v in vals),'nonnegative':sum(v>=0 for v in vals),'n_roots':3}
     criteria={'return_mean_positive':summary['return']['mean_difference']>0,'return_at_least_two_positive':summary['return']['positive']>=2,'cosine_mean_positive':summary['cosine']['mean_difference']>0}
     if task!='tracking':criteria.update(success_mean_positive=summary['success']['mean_difference']>0,success_at_least_two_nonnegative=summary['success']['nonnegative']>=2)
     scientific='pass' if all(criteria.values()) else 'fail'
    status='pending' if missing or external=='pending' else 'pass' if scientific=='pass' and external=='pass' else 'fail'
    candidates[candidate]={'status':status,'scientific_status':scientific,'external_gate_status':external,'external_gates':checks,'missing':missing,'seed_rows':rows,'summary':summary,'criteria':criteria}
   # Any unknown candidate could outrank a currently passing one: no premature recommendation.
   pending=any(c['status']=='pending' for c in candidates.values())
   passing=[c for c in CANDIDATES if candidates[c]['status']=='pass']
   task_report[baseline]={'status':'pending' if pending else 'pass' if passing else 'no_go','selected':None if pending or not passing else passing[0],'candidates':candidates}
  report['tasks'][task]=task_report
 report['limitations']=['Three roots are exploratory, not a significance test.','Both baselines are reported without resolving the protocol ambiguity by observed outcomes.','Gate evidence is an explicit external attestation; this script does not certify the evidence contents.','No confirmation runner or authorization file is modified.']
 return report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--runs-root',type=Path,default=ROOT/'runs/eagm2_development');p.add_argument('--gates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not a.attempt or Path(a.attempt).name!=a.attempt or a.attempt in ('.','..'):p.error('exact attempt directory name required')
 gates=json.loads(a.gates.read_text())
 if not isinstance(gates,dict):p.error('gates must be an object')
 report=evaluate(a.runs_root,a.attempt,gates)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
 print(str(a.output))

if __name__=='__main__':main()
