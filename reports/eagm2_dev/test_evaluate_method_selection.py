"""Temporary synthetic fixtures only; actual development data are never read."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('selection',Path(__file__).with_name('evaluate_method_selection.py'))
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

class SelectionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='selection_fixture_');self.root=Path(self.tmp.name);self.gates={}
  for task,seeds in M.SEEDS.items():
   self.gates[task]={b:{c:{g:{'status':'pass','evidence':'synthetic-test-evidence'} for g in ('capacity','budget','tests')} for c in M.CANDIDATES} for b in M.BASELINES}
   for seed in seeds:
    run=self.root/task/f'seed{seed}'/'development_v1';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(dict(task=task,seed=seed,attempt='development_v1',smoke=False)))
    (run/'run_manifest.json').write_text('{"status":"completed"}')
    for method in M.BASELINES+M.CANDIDATES:
     d=run/method;d.mkdir();is_candidate=method in M.CANDIDATES
     result=dict(status='completed',method=method,seed=seed,validation_physical=dict(mean_return=int(is_candidate),success_rate=.1*is_candidate),validation_gradient=dict(mean_cosine=.1*is_candidate),physical={'mean_return':-1e99 if is_candidate else 1e99},gradient={'mean_cosine':-1 if is_candidate else 1})
     (d/'results.json').write_text(json.dumps(result))
 def tearDown(self):self.tmp.cleanup()
 def run_selection(self):return M.evaluate(self.root,'development_v1',self.gates)
 def path(self,task='tracking',seed=200001,method='delay_dir'):return self.root/task/f'seed{seed}'/'development_v1'/method/'results.json'
 def edit(self,path,fn):
  value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))
 def test_fixed_order_and_heldout_invariance(self):
  report=self.run_selection()
  for task in M.SEEDS:
   self.assertEqual(report['tasks'][task]['ff_dir']['selected'],'delay_dir')
  self.assertEqual(report['tasks']['vision']['legacy_dir']['status'],'not_applicable')
  self.edit(self.path(),lambda r:r.update(physical=None,gradient='irrelevant'))
  self.assertEqual(self.run_selection(),report)
 def test_missing_root_pending(self):
  self.path().unlink();report=self.run_selection()['tasks']['tracking']['ff_dir']
  self.assertEqual(report['status'],'pending');self.assertIsNone(report['selected'])
 def test_missing_field_pending_and_gate_evidence_required(self):
  self.edit(self.path(),lambda r:r['validation_gradient'].update(mean_cosine=None))
  self.assertEqual(self.run_selection()['tasks']['tracking']['ff_dir']['status'],'pending')
  self.gates['contact']['ff_dir']['delay_dir']['budget']={'status':'pass'}
  self.assertEqual(self.run_selection()['tasks']['contact']['ff_dir']['status'],'pending')
 def test_negative_return_counts_and_ties(self):
  self.edit(self.path(seed=200001),lambda r:r['validation_physical'].update(mean_return=0))
  self.edit(self.path(seed=200002),lambda r:r['validation_physical'].update(mean_return=0))
  report=self.run_selection()['tasks']['tracking']['ff_dir']
  self.assertEqual(report['candidates']['delay_dir']['status'],'fail');self.assertEqual(report['selected'],'recurrent_dir')
 def test_contact_success_rule_and_gate_failure(self):
  for seed,v in zip(M.SEEDS['contact'],[.5,-.1,-.1]):
   self.edit(self.path('contact',seed),lambda r,v=v:r['validation_physical'].update(success_rate=v+.1))
   self.edit(self.path('contact',seed,'ff_dir'),lambda r:r['validation_physical'].update(success_rate=.1))
  report=self.run_selection()['tasks']['contact']['ff_dir']
  self.assertEqual(report['candidates']['delay_dir']['status'],'fail')
  for candidate in M.CANDIDATES:self.gates['tracking']['ff_dir'][candidate]['tests']['status']='fail'
  report=self.run_selection()['tasks']['tracking']['ff_dir'];self.assertEqual(report['status'],'no_go');self.assertIsNone(report['selected'])
 def test_baseline_ambiguity_kept_and_nonfinite_pending(self):
  for seed in M.SEEDS['tracking']:
   self.edit(self.path(seed=seed,method='legacy_dir'),lambda r:r['validation_physical'].update(mean_return=10))
  report=self.run_selection()['tasks']['tracking']
  self.assertEqual(report['ff_dir']['status'],'pass');self.assertEqual(report['legacy_dir']['status'],'no_go')
  self.edit(self.path(),lambda r:r['validation_gradient'].update(mean_cosine=float('nan')))
  self.assertEqual(self.run_selection()['tasks']['tracking']['ff_dir']['status'],'pending')

if __name__=='__main__':unittest.main(verbosity=2)
