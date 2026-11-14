"""Inspect tracked historical text; reserve proposed seeds without running experiments."""
from pathlib import Path
import subprocess,json,re,csv,collections,hashlib
root=Path(__file__).resolve().parents[2];out=root/'reports/eagm2_plan'
files=subprocess.check_output(['git','ls-tree','-r','--name-only','412cf8c'],cwd=root,text=True).splitlines()
uses=collections.defaultdict(set); errors=[]; examined=0

def put(v,p,key):
 if isinstance(v,int) and not isinstance(v,bool) and v>=0: uses[v].add(p+':'+key)
 elif isinstance(v,list):
  for e in v: put(e,p,key)
def walk(obj,p,key=''):
 if isinstance(obj,dict):
  for k,v in obj.items():
   if 'seed' in k.lower():put(v,p,key+'/'+k)
   walk(v,p,key+'/'+k)
 elif isinstance(obj,list):
  for i,v in enumerate(obj):walk(v,p,key+'/'+str(i))
for p in files:
 f=root/p
 for s in re.findall(r'(?:seed|seeds)[_-]?(\d+)',p):uses[int(s)].add(p+':filename')
 if f.suffix=='.json':
  try:walk(json.loads(f.read_text()),p);examined+=1
  except (ValueError,OSError) as e:errors.append({'path':p,'error':str(e)})
 elif f.suffix in ('.py','.yaml','.md') and not p.startswith('reports/eagm2_audit/'):
  for m in re.finditer(r'\bseed\s*[=:]\s*(\d+)\b',f.read_text(errors='replace')):uses[int(m[1])].add(p+':literal')
new=[]
for task,base in [('tracking',200000),('contact',201000),('vision',202000)]:
 for n in range(1,4):new.append((base+n,task,'development','proposed_reserved_not_run'))
 for n in range(101,116):new.append((base+n,task,'confirmation','proposed_reserved_not_run'))
assert not set(uses).intersection(s for s,*_ in new),'proposed seed collides'
rows=[]
for s in sorted(uses):
 role='historical_other_not_new'
 if 0<=s<=4:role='historical_exploration_or_legacy_reproduction'
 elif 5<=s<=9:role='historical_stage5_confirmation_reused_by_stage5_controls'
 elif 10<=s<=14:role='historical_stage6b_confirmation_and_histc_exploration'
 elif 15<=s<=19:role='historical_stage6c_path_exploration_pairing_caveat'
 elif s==20:role='historical_reach_exploration_not_transfer_confirmation'
 rows.append(dict(seed=s,task='historical_mixed',role=role,status='historical_do_not_relabel',source=';'.join(sorted(uses[s]))))
for s,t,r,status in new:rows.append(dict(seed=s,task=t,role=r,status=status,source='04_experiment_protocol.md; draft pending author approval'))
with (out/'04_seed_registry.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
info={'base_commit':'412cf8c9533cde37e12e1e0b8d55fef7e19a5a80','tracked_files':len(files),'json_parsed':examined,'parse_errors':errors,'historical_values_found':len(uses),'proposed_root_seeds':len(new),'collision':False,'scope_limit':'Static JSON seed keys, filename patterns and literal seed assignments only; not complete dynamic RNG provenance. Binary checkpoint RNG and untracked/original-Windows data not inferred. Recheck generated stream manifests before launch.'}
(out/'04_seed_scan.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps(info,ensure_ascii=False))
