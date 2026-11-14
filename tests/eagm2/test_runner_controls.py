"""Method routing tests; no simulator/training is started by importing runner."""
import unittest
from scripts.eagm2.run_development import selected_methods


class RunnerControlTests(unittest.TestCase):
    def test_physical_defaults_include_budget_controls(self):
        for task in ('tracking', 'contact'):
            methods = selected_methods(task)
            self.assertTrue({'legacy_base', 'legacy_dir', 'base_extra_data', 'direct_fd', 'spsa'} <= set(methods))
            self.assertEqual(len(methods), len(set(methods)))

    def test_vision_default_excludes_incompatible_models(self):
        methods = selected_methods('vision')
        self.assertNotIn('base_extra_data', methods)
        self.assertNotIn('legacy_base', methods)
        self.assertNotIn('legacy_dir', methods)
        self.assertIn('direct_fd', methods)
        self.assertIn('spsa', methods)

    def test_explicit_vision_legacy_rejected(self):
        for name in ('legacy_base', 'legacy_dir', 'base_extra_data'):
            with self.assertRaises(ValueError):
                selected_methods('vision', name)

    def test_explicit_subset_preserved(self):
        self.assertEqual(selected_methods('tracking', 'ff_base,delay_dir'), ['ff_base', 'delay_dir'])
        self.assertEqual(selected_methods('contact', 'direct_fd,spsa'), ['direct_fd', 'spsa'])

    def test_invalid_requests_rejected_before_run(self):
        for request in ('bogus', '', 'ff_base,ff_base', 'direct_fd', 'spsa'):
            with self.assertRaises(ValueError):
                selected_methods('tracking', request)


if __name__ == '__main__':
    unittest.main()
