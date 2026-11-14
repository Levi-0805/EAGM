import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.eagm2.run_authorized_batch import AUTHORIZED, build_plan, source_hashes, validate_gates, run_batch


class AuthorizedBatchTests(unittest.TestCase):
    def test_plan_exactly_nine_exploratory_roots_and_defaults(self):
        rows = build_plan(Path('/repo'), '/python', 'development_v1')
        self.assertEqual([(r['task'], r['seed']) for r in rows], [(t,s) for t,seeds in AUTHORIZED.items() for s in seeds])
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertNotIn('--methods', row['command'])
            self.assertNotIn('--smoke', row['command'])
            self.assertEqual(row['command'][-4:], ['--updates', '1000', '--policy-updates', '50'])

    def test_unsafe_attempt_and_excess_updates_rejected(self):
        for attempt in ('../other', '/absolute', '..', ''):
            with self.assertRaises(ValueError): build_plan(Path('/repo'), '/python', attempt)
        with self.assertRaises(ValueError): build_plan(Path('/repo'), '/python', 'good', policy_updates=51)

    def test_source_hash_detects_change_and_new_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'eagm2').mkdir(); p=root/'eagm2/model.py'; p.write_text('a=1')
            a=source_hashes(root); p.write_text('a=2'); b=source_hashes(root)
            self.assertNotEqual(a,b)
            (root/'eagm2/new.py').write_text('a=3'); self.assertNotEqual(b,source_hashes(root))

    def test_dry_run_starts_no_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('subprocess.Popen') as worker, patch('subprocess.run') as analysis:
                run_batch(root, root, build_plan(root, '/python', 'development_v1'), {}, python='/python', dry_run=True)
            worker.assert_not_called(); analysis.assert_not_called()
            self.assertTrue((root/'experiment_status.csv').exists())

    def test_drift_stops_before_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('subprocess.Popen') as worker, self.assertRaisesRegex(RuntimeError,'changed'):
                run_batch(root,root,build_plan(root,'/python','development_v1'),{'missing.py':'hash'},python='/python')
            worker.assert_not_called()

    def test_first_failure_stops_remaining_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('subprocess.Popen') as worker:
                worker.return_value.pid=123; worker.return_value.wait.return_value=7
                with self.assertRaisesRegex(RuntimeError,'failed'):
                    run_batch(root,root,build_plan(root,'/python','development_v1'),{},python='/python')
                self.assertEqual(worker.call_count,1)
            rows=json.loads((root/'experiment_status.json').read_text())
            self.assertEqual(rows[0]['status'],'failed')
            self.assertTrue(all(r['status']=='planned' for r in rows[1:]))

    def test_smoke_and_full_suite_log_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); smokes={}
            (root/'tests/eagm2').mkdir(parents=True)
            (root/'tests/eagm2/test_one.py').write_text('def test_example(): pass\n')
            log=root/'tests.log'; log.write_text('test_example (test_one.Suite.test_example) ... ok\nRan 1 test in 1s\nOK\n')
            for task,seeds in AUTHORIZED.items():
                path=root/task; path.mkdir(); smokes[task]=path
                (path/'config.json').write_text(json.dumps(dict(task=task,seed=seeds[0],smoke=True)))
                for name in ('run_manifest.json','results.json'):
                    (path/name).write_text('{"status":"completed"}')
            self.assertIn('tests_log_sha256',validate_gates(root,smokes,log))
            (smokes['vision']/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):validate_gates(root,smokes,log)
            (smokes['vision']/'failure.json').unlink()
            log.write_text('Ran 1 test in 1s\nOK\n')
            with self.assertRaisesRegex(ValueError,'full current suite'):validate_gates(root,smokes,log)
            with self.assertRaises(ValueError):validate_gates(root,{'tracking':smokes['tracking']},log)


if __name__=='__main__': unittest.main()
