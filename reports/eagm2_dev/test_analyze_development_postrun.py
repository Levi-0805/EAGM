"""Synthetic-only postrun integration checks; no actual run directories read."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).with_name('analyze_development_postrun.py')
SPEC=importlib.util.spec_from_file_location('postrun',SCRIPT)
MOD=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MOD)

class PostrunIntegration(unittest.TestCase):
 def test_default_resolved_full_matrix_failure_and_resources(self):
  frozen=SCRIPT.parents[2]/'scripts/eagm2/analyze_development.py'
  before=hashlib.sha256(frozen.read_bytes()).hexdigest()
  methods=list(MOD.FACTORIAL.values())+['capacity_mlp','capacity_gru','base_extra_data','paml_adaptation','vagram_adaptation','direct_fd','spsa']
  with tempfile.TemporaryDirectory(prefix='postrun_analysis_fixture_') as temp:
   root=Path(temp);runs=root/'runs';out=root/'out'
   def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
   for task,seeds in MOD.AUTHORIZED.items():
    for seed in seeds:
     run=runs/task/f'seed{seed}'/'development_v1'
     write(run/'config.json',dict(task=task,seed=seed,attempt='development_v1',methods=None,resolved_methods=methods))
     write(run/'run_manifest.json',dict(status='completed'))
     write(run/'direct_controls_resource.json',dict(wall_seconds=7,cuda_elapsed_seconds=6,peak_allocated_bytes=123))
     write(run/'budget.json',dict(control_steps=dict(collection_train=20,labels_train=40)))
     for method in methods:
      result=dict(status='completed',method=method,seed=seed,physical=dict(mean_return=1,success_rate=.1),gradient=dict(mean_cosine=.2),model_metrics=dict(one_step_nmse=2),validation_model_metrics=dict(one_step_nmse=3),parameter_counts=dict(trainable=100),training_simulator_budget=dict(collection=20,gradient_labels=40,extra_collection=0,labels_generated_but_unused=False))
      if method in ('direct_fd','spsa'):
       result=dict(method=method,stop_reason='max_updates_reached',physical=dict(mean_return=1),training_control_steps=60)
       (run/method).mkdir(parents=True,exist_ok=True);(run/method/'checkpoint.pt').touch()
      if task=='tracking' and seed==200003 and method=='spsa':
       write(run/method/'optimization.json',dict(stop_reason='failed',wall_seconds=3,training_control_steps=12,error='synthetic failure'))
       write(run/method/'actual_simulator_accounting.json',dict(training_control_steps=10))
       write(run/'failure.json',dict(wall_seconds=9,budget=dict(control_steps=dict(partial=10)),traceback='synthetic'))
       write(run/'run_manifest.json',dict(status='failed'));continue
      if task=='contact' and seed==201003 and method=='capacity_gru':continue
      write(run/method/'results.json',result)
   command=[sys.executable,str(SCRIPT),'--attempt','development_v1','--runs-root',str(runs),'--output-dir',str(out)]
   subprocess.run(command,check=True,capture_output=True,text=True)
   audit=json.loads((out/'06_development_audit.json').read_text());tables=audit['tables']
   self.assertEqual(len(tables['seed_raw']),9*len(methods))
   failed=next(r for r in tables['seed_raw'] if r['seed']==200003 and r['method']=='spsa')
   self.assertEqual(failed['status'],'failed');self.assertFalse(failed['included']);self.assertIsNone(failed['return'])
   missing=next(r for r in tables['seed_raw'] if r['seed']==201003 and r['method']=='capacity_gru')
   self.assertEqual(missing['status'],'missing');self.assertFalse(missing['included'])
   direct=next(r for r in tables['seed_raw'] if r['method']=='direct_fd')
   self.assertTrue(direct['included']);self.assertTrue(direct['checkpoint_exists']);self.assertIsNone(direct['cosine'])
   self.assertTrue(any(r['scope']=='direct_controls_joint' and r['measure']=='cuda_elapsed_seconds' for r in tables['resources']))
   self.assertTrue(any(r['scope']=='per_method_attribution_not_additional_actual_cost' and r['category']=='gradient_labels' and r['value']==40 for r in tables['budgets']))
   self.assertTrue(any(r['scope']=='actual_simulator_accounting.json' and r['value']==10 and r['excluded'] for r in tables['budgets']))
   mismatched=[r for r in tables['paired_differences'] if r['metric'] in ('nmse','validation_nmse') and 'base_extra_data' in (r['left'],r['right'])]
   self.assertTrue(mismatched);self.assertTrue(all(not r['comparable'] and r['difference'] is None for r in mismatched))
   self.assertFalse(audit['confirmation_authorized'])
   self.assertEqual(json.loads((out/'06_development_confirmation_draft.json').read_text())['status'],'not_authorized')
   self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)  # no overwrite
  self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(),before)

if __name__=='__main__':unittest.main(verbosity=2)
