"""Read-only inventory of current evidence; never changes running experiments."""
import csv,hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
 def save(name,data):(out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 frozen=json.loads((ROOT/'runs/eagm2_batches/development_v1/source_hashes.json').read_text())
 sources={name:sha(ROOT/name) for name in frozen}
 assets={}
 for folder in (ROOT/'assets',ROOT/'eagm2/assets',Path('/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/assets')):
  for f in sorted(folder.rglob('*')):
   if f.is_file():assets[str(f.resolve())]=dict(bytes=f.stat().st_size,sha256=sha(f))
 save('source_hash_manifest.json',dict(training_sources=sources,frozen_batch_match=sources==frozen,
     assets=assets,external_asset_scope='Entire existing OpenArm v2 assets tree, plus repository assets and contact MJCFs'))
 env=json.loads((ROOT/'runs/eagm2_development/tracking/seed200001/development_v1/environment.json').read_text())
 packages=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True).splitlines()
 save('environment_lock.json',dict(observed_environment=env,python_executable=sys.executable,pip_freeze=packages,
     scope='Observed package/version snapshot, not a clean-environment reproduction certificate'))
 rows=[];artifacts=[];failures=[]
 for config in sorted((ROOT/'runs/eagm2_development').glob('*/*/*/config.json')):
  run=config.parent;cfg=json.loads(config.read_text());manifest=run/'run_manifest.json'
  status=json.loads(manifest.read_text()).get('status') if manifest.exists() else 'manifest_missing'
  if (run/'failure.json').exists():failures.append(str(run.relative_to(ROOT)))
  methods=cfg.get('resolved_methods') or cfg.get('methods') or []
  if isinstance(methods,str):methods=methods.split(',')
  if not methods:methods=[f.parent.name for f in run.glob('*/results.json')] or ['unspecified_legacy_or_failed_attempt']
  for method in methods:
   result=run/method/'results.json'
   rows.append(dict(task=run.parents[1].name,seed=cfg.get('seed',run.parent.name),method=method,
      config_hash=sha(config),attempt_id=run.name,attempt_status=status,
      result_exists=result.exists(),failure_retained=(run/'failure.json').exists(),path=str(run.relative_to(ROOT))))
  for f in sorted(run.rglob('*.pt')):
   artifacts.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)))
 with (out/'experiment_registry.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save('artifact_inventory.json',dict(local_tensor_artifacts=artifacts,failed_attempts_retained=failures,
     mutable_running_artifacts='Hashes describe this snapshot only; final completion inventory must be captured again.'))
 (out/'complete_git_diff.patch').write_text(git('diff','--binary','f3a998e'))
 (out/'implementation_since_plan.patch').write_text(git('diff','--binary','81ecd5f','HEAD'))
 with tarfile.open(out/'development_source_archive.tar.gz','x:gz') as tar:
  for name in sorted(sources):tar.add(ROOT/name,arcname=name)
 save('project_baseline_manifest.json',dict(captured_unix=time.time(),current_commit=git('rev-parse','HEAD').strip(),
     formal_batch_commit='5037c4904dcd3ee99d15ab95e31c15cac413de94',user_reference_commit='f3a998e',
     git_status=git('status','--short'),confirmation_authorized=False,
     running_batch=json.loads((ROOT/'runs/eagm2_batches/development_v1/experiment_status.json').read_text()),
     missing_or_unstarted_configs='Queued roots have no actual config hash yet; plan.json is their prospective registration.',
     snapshot_files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
 print(str(out))
if __name__=='__main__':main()
