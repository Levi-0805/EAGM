"""Watch the authorized batch, preserve failures, then produce review artifacts.

No training, configuration selection, paper editing or confirmation is performed.
The original frozen batch's analysis failure is preserved, not rewritten.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,sys,tarfile,time

ROOT=Path(__file__).resolve().parents[2]
BATCH=ROOT/'runs/eagm2_batches/development_v1'
REPORT=ROOT/'reports/eagm2_dev'
EXPECTED={(task,seed) for task,ss in {'tracking':[200001,200002,200003],
 'contact':[201001,201002,201003],'vision':[202001,202002,202003]}.items() for seed in ss}

def read(p):return json.loads(p.read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(command,log):
 with log.open('x') as f:
  result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 return result.returncode
def validate_rows(rows):
 if len(rows)!=9 or {(r['task'],r['seed']) for r in rows}!=EXPECTED:
  raise RuntimeError('Batch root identities differ from nine authorized exploratory roots')
 if any(r['attempt']!='development_v1' for r in rows):raise RuntimeError('Unexpected attempt')
 return all(r['status']=='completed' for r in rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
 statepath=REPORT/'development_completion_worker.json'
 if statepath.exists():raise FileExistsError(statepath)
 state=dict(status='waiting_for_batch',pid=os.getpid(),confirmation_authorized=False,
            started_unix=time.time(),implementation_sha256=sha(Path(__file__)))
 save(statepath,state)
 try:
  while True:
   batch=read(BATCH/'batch_manifest.json');rows=read(BATCH/'experiment_status.json')
   all_complete=validate_rows(rows)
   if batch['status']!='running':break
   state.update(completed_roots=sum(r['status']=='completed' for r in rows),checked_unix=time.time())
   save(statepath,state)
   if time.time()-state['started_unix']>48*3600:
    raise TimeoutError('Completion monitor expired; no training terminated or restarted')
   time.sleep(30)
  state.update(status='postprocessing',training_complete=all_complete,original_batch_status=batch['status'])
  save(statepath,state)
  # The immutable old automatic analyzer has a known methods=None parsing bug.
  # Its traceback and failed batch manifest remain original evidence.
  output=REPORT/'development_v1_postrun_analysis'
  code=run([sys.executable,str(REPORT/'analyze_development_postrun.py'),
      '--attempt','development_v1','--output-dir',str(output),'--report-tag','development_v1_postrun'],
      REPORT/'development_v1_postrun_analysis.console.log')
  state['analysis_exit_code']=code;save(statepath,state)
  if code:raise RuntimeError('Postrun analysis failed; see retained console log')
  if not all_complete:
   raise RuntimeError('At least one training root incomplete; partial audit saved, no retries or confirmation started')
  # Fixed, sparse supplemental probes: never select methods based on outcomes.
  diagnostics=[]
  for task,seed in sorted(EXPECTED):
   if task=='vision':continue
   dest=REPORT/f'ood_{task}_{seed}_v1'
   code=run([sys.executable,str(REPORT/'evaluate_ood.py'),'--source',
      str(ROOT/f'runs/eagm2_development/{task}/seed{seed}/development_v1'),
      '--output',str(dest),'--fd'],REPORT/f'ood_{task}_{seed}_v1.console.log')
   diagnostics.append(dict(task=task,seed=seed,exit_code=code,path=str(dest.relative_to(ROOT))))
   state['ood_diagnostics']=diagnostics;save(statepath,state)
  records=[];artifacts=[];resources=[]
  artifact_dir=ROOT/'runs/eagm2_artifacts';artifact_dir.mkdir(exist_ok=True)
  for task,seed in sorted(EXPECTED):
   directory=ROOT/f'runs/eagm2_development/{task}/seed{seed}/development_v1'
   result=read(directory/'results.json')
   resources.append(dict(task=task,seed=seed,wall_seconds=result['wall_seconds'],
       control_steps=result['budget']['total_control_steps'],
       simulator_wall_seconds=sum(result['budget']['simulator_wall_seconds'].values())))
   archive=artifact_dir/f'{task}_{seed}_development_v1_checkpoints.tar.gz'
   with tarfile.open(archive,'x:gz') as tar:
    for f in sorted(directory.rglob('*.pt')):
     if 'checkpoints' in f.parts or f.name=='checkpoint.pt':
      tar.add(f,arcname=str(f.relative_to(directory)))
   if archive.stat().st_size>=95*1024*1024:
    raise RuntimeError('Checkpoint archive too large for planned Git publication; retained locally')
   artifacts.append(dict(path=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive)))
   for f in sorted(directory.rglob('*')):
    if f.is_file():records.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)))
  save(REPORT/'05_reproducibility_manifest.json',dict(phase='exploratory',confirmation_authorized=False,
       artifacts=artifacts,files=records,raw_tensor_data='Retained locally; hashes listed. Checkpoints additionally archived for Git publication.'))
  save(REPORT/'06_actual_root_resources.json',resources)
  lines=['# Measured development resources','',
   'All rows are actual exploratory roots on RTX 4060. Wall elapsed is not GPU-active time.',
   'Simulator time is included in root wall time; do not add these columns.',
   'Per-method CUDA-event time and peak memory remain in the analysis resource CSV.',
   'Shared encoder/critic and failed attempts require separate attribution; this table is successful main roots only.','',
   '| Task | Root | Wall s | Simulator wall s | Control steps |','|---|---:|---:|---:|---:|']
  lines += [f"| {r['task']} | {r['seed']} | {r['wall_seconds']:.1f} | {r['simulator_wall_seconds']:.1f} | {r['control_steps']} |" for r in resources]
  lines += ['', 'No 45-root budget is authorized or frozen by multiplying these runtimes. Final candidate selection, failed-attempt costs, task changes and the confirmation method roster still require review.']
  (REPORT/'06_compute_resource_report.md').write_text('\n'.join(lines)+'\n')
  (REPORT/'06_development_handoff.md').write_text(
   '# Development handoff — author review required\n\n'
   'All nine exploratory root runs completed. This is run completeness, not scientific acceptance. '
   'The original automatic-analysis error is preserved; the separately versioned postrun analyzer produced descriptive tables.\n\n'
   'Read development_v1_postrun_analysis for per-seed results, paired comparisons, factorial ablations, '
   'capacity, gradient and resource tables, plus the not_authorized confirmation draft. '
   'Read the frozen-pipeline review for protocol boundaries and the sparse OOD directories for retained diagnostic failures.\n\n'
   'No method has been automatically selected. Positive return does not prove a gradient mechanism. '
   'No paper or historical data was edited. No confirmation seed was started. '
   'Visual OOD, full oracle, refresh-frequency and RGB information ablations remain incomplete; '
   'these gaps prevent treating this handoff as unconditional authorization for 45 roots.\n')
  original=read(REPORT/'05_baseline_preservation.json')['tracked_source_hashes']
  changed=[name for name,digest in original.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
  save(REPORT/'05_final_historical_preservation.json',dict(changed=changed,checked_files=len(original)))
  if changed:raise RuntimeError('Historical tracked files changed; do not publish unrelated edits')
  state.update(status='artifacts_ready_for_review',finished_unix=time.time());save(statepath,state)
  if args.publish:
   branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
   if branch!='EAGM-2.0-full-validation':raise RuntimeError('Branch changed; publication refused')
   allowed=('reports/eagm2_dev/','runs/eagm2_development/','runs/eagm2_batches/','runs/eagm2_artifacts/')
   staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
   if any(not x.startswith(allowed) for x in staged):raise RuntimeError('Unrelated staged edits; publication refused')
   subprocess.run(['git','add',*(x.rstrip('/') for x in allowed)],cwd=ROOT,check=True)
   subprocess.run(['git','-c','user.name=levi','-c','user.email=ma2b2c2@gmail.com','commit','-m',
       'Record nine exploratory roots and independent development audit artifacts'],cwd=ROOT,check=True)
   code=run(['/home/aaa/.local/bin/github-direct','git','-c','credential.helper=!gh auth git-credential',
       'push','origin','EAGM-2.0-full-validation'],REPORT/'development_final_push.log')
   state.update(publication_exit_code=code,published_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() if code==0 else None)
   if code:raise RuntimeError('Push failed; committed local artifacts preserved')
   state['status']='published_for_author_review';save(statepath,state)
 except BaseException as exc:
  state.update(status='needs_review',error=repr(exc),finished_unix=time.time());save(statepath,state);raise

if __name__=='__main__':main()
