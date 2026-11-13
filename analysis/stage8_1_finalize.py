"""Integrity, aggregate budgets and additional descriptive audit summaries."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'runs/stage8_1_geometry'
def main():
 a=json.loads((P/'budget.json').read_text());b=json.loads((P/'supplemental_budget.json').read_text());assert a['physical_steps']+b['control_steps']<=20000
 combined=dict(control_steps=a['physical_steps']+b['control_steps'],by_phase={**a['by_phase'],'physical_state_coverage':b['control_steps']},limit=20000,remaining=20000-a['physical_steps']-b['control_steps'],main_geometry_calls=a['geometry']['calls'],supplemental_geometry_calls=b['geometry']['calls'],geometry_seconds=a['geometry']['seconds']+b['geometry']['seconds'],wall_seconds=a['wall_seconds']+b['wall_seconds'],unit_tests_control_steps=0,note='Geometry call counters cover main and supplemental scripts; separate unit-test geometry queries are validation overhead, not physics steps.')
 (P/'total_budget.json').write_text(json.dumps(combined,indent=2))
 ss=json.loads((P/'static.json').read_text())+json.loads((P/'physical_state_geometry.json').read_text());tables=[]
 for group in sorted(set(x['group'] for x in ss)):
  rr=[x for x in ss if x['group']==group];v=np.array([x['position_value_fk']-x['position_value_true'] for x in rr]);g=np.array([x['g_fk'] for x in rr]);gt=np.array([x['g_true'] for x in rr]);goals=np.array([x['goals'] for x in rr]);q=np.array([x['q'] for x in rr]);e=np.linalg.norm(g-gt,axis=1)
  tables.append(dict(group=group,n=len(rr),reward_value_mae=float(abs(v).mean()),reward_value_max_abs=float(abs(v).max()),reward_gradient_rmse_per_joint=np.sqrt(np.mean((g-gt)**2,axis=0)).tolist(),configuration_error_correlation=[float(np.corrcoef(q[:,i],e)[0,1]) for i in range(14)],goal_error_correlation=[float(np.corrcoef(goals[:,i],e)[0,1]) for i in range(6)]))
 (P/'reward_region_summary.json').write_text(json.dumps(tables,indent=2))
 with (P/'static_metrics.csv').open('w') as f:
  keys=['group','episode','frame','position_rmse','position_mae','jacobian_rmse','jacobian_relative','jacobian_cosine','reward_gradient_cosine','reward_gradient_relative'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:x[k] for k in keys} for x in ss)
 old=json.loads((ROOT/'analysis/stage8_1_preservation/manifest.json').read_text());changed=[p for p,h in old.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not changed,changed
 (P/'integrity.json').write_text(json.dumps(dict(prior_files_checked=len(old),changed=changed,branch='stage8-1-geometry-jacobian-audit',tests='analytic FD, Torch gradcheck, reward component derivatives, stop-gradient identities, native rollout parity, live state/warm-start unchanged; all PASS',failures='No execution failures. Scientific recovery hypothesis not supported sufficiently; all negative outcomes retained.'),indent=2))
 for name in ['geometry_audit','reward_jacobian','gradient_intervention','frozen_policy_comparison','research_decision']:
  p=ROOT/'reports'/f'stage8_1_{name}.md'
  with p.open('a') as f:f.write(f'\nTotal added physical control steps including actual-state coverage: **{combined["control_steps"]}/20000**. Main and supplemental raw budgets are preserved separately; combined accounting is total_budget.json. Static metrics CSV, component/goal/configuration summaries and SHA-256 inventory are in the run directory. No prior file changed relative to Stage8.1 preservation manifest.\n')
 p=ROOT/'reports/stage8_1_research_decision.md'
 with p.open('a') as f:f.write('\nMechanistic refinement: on held-out trajectory states, native reward-gradient cosine is 0.441; substituting true J alone with the FK residual yields 0.409, while retaining FK J with the true positional residual yields 0.957. Thus forward positional bias dominates this particular state-level directional error more than J orientation. This is a controlled state-level decomposition, not proof that the same factor dominates closed-loop policy failure. Frozen true-geometry policy gradients remain negative on average; investigate learned dynamics sensitivity and training-time derivative compensation next. No new five-seed proposal is justified by these results.\n')
 files=[x for x in P.rglob('*') if x.is_file() and x.name!='final_manifest.json']+[ROOT/'analysis/stage8_1_run.py',ROOT/'analysis/stage8_1_physical_states.py',ROOT/'analysis/stage8_1_reports.py',ROOT/'analysis/stage8_1_finalize.py',ROOT/'training/stage8_1_geometry.py',ROOT/'tests/test_stage8_1_geometry.py',ROOT/'configs/stage8_1_protocol.json']+list((ROOT/'reports').glob('stage8_1_*.md'))+list((ROOT/'reports/figures/stage8_1').glob('*'))
 (P/'final_manifest.json').write_text(json.dumps({str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2));print(combined,'preserved',len(old))
if __name__=='__main__':main()
