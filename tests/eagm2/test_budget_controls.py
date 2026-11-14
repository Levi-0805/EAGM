import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from eagm2.budget_controls import optimize_query_budget, run_direct_controls


class BudgetControlTests(unittest.TestCase):
    def test_complete_iteration_cost_and_unused(self):
        for method, queries in [('direct_fd', 6), ('spsa', 16)]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def objective(theta):
                    calls.append(theta.copy())
                    return -np.square(theta-.4).sum()
                root = Path(tmp)/method
                result = optimize_query_budget(objective, 3, queries*7+6, 7,
                                               method=method, seed=2, output=root)
                self.assertEqual(len(calls), queries)
                self.assertEqual(result['training_control_steps'], queries*7)
                self.assertEqual(result['unused_control_steps'], 6)
                self.assertEqual(result['updates'], 1)
                if method == 'direct_fd':
                    self.assertTrue(np.all(result['theta'] > 0))
                checkpoint = torch.load(root/'checkpoint.pt', weights_only=False)
                self.assertEqual(checkpoint['completed_updates'], 1)
                self.assertTrue(checkpoint['optimizer']['state'])
                self.assertIn('numpy_rng', checkpoint)

    def test_insufficient_full_fd_iteration_uses_no_simulator(self):
        calls = []
        result = optimize_query_budget(lambda t: calls.append(t) or 1., 28, 55, 1,
                                       method='direct_fd', seed=0)
        self.assertEqual(calls, [])
        self.assertEqual(result['updates'], 0)
        self.assertEqual(result['unused_control_steps'], 55)

    def test_boundary_fd_cost_and_no_partial_coordinate(self):
        calls = []
        def objective(t):
            self.assertLessEqual(np.max(np.abs(t)), 1.)
            calls.append(t.copy())
            return t.sum()
        result = optimize_query_budget(objective, 2, 200, 1, method='direct_fd', seed=3)
        self.assertLessEqual(result['updates'], 50)
        self.assertEqual(len(calls), result['training_control_steps'])
        self.assertLessEqual(len(calls), 200)
        self.assertTrue(any(row['objective_queries'] == 5 for row in result['ledger']))
        self.assertTrue(np.all(np.abs(result['theta']) <= 1.))

    def test_spsa_deterministic(self):
        args = dict(dimension=4, query_cap=96, steps_per_query=1, method='spsa', seed=12)
        a = optimize_query_budget(lambda t: -np.square(t-.2).sum(), **args)
        b = optimize_query_budget(lambda t: -np.square(t-.2).sum(), **args)
        np.testing.assert_array_equal(a['theta'], b['theta'])
        self.assertEqual(a['updates'], 6)

    def test_failure_keeps_reserved_cost_and_checkpoint(self):
        def broken(_):
            raise RuntimeError('simulator failure')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'failed'
            with self.assertRaisesRegex(RuntimeError, 'simulator failure'):
                optimize_query_budget(broken, 2, 20, 3, method='direct_fd', seed=0, output=root)
            record = json.loads((root/'optimization.json').read_text())
            self.assertEqual(record['training_control_steps'], 3)
            self.assertEqual(record['updates'], 0)
            self.assertEqual(record['stop_reason'], 'failed')
            self.assertTrue((root/'checkpoint.pt').exists())

    def test_confirmation_rejected_before_any_environment_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PermissionError, 'confirmation locked'):
                run_direct_controls(None, [], [], {}, 300001, 'tracking', 5, 5, None, tmp, 100)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                optimize_query_budget(lambda t: t.sum(), 2, 10, 1, method='direct_fd', seed=0, output=tmp)


if __name__ == '__main__':
    unittest.main()
