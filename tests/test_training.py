import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from secrets_pipeline.metrics import average_precision, binary_metrics, calibrate_thresholds, wilson_lower
from secrets_pipeline.train import config_fingerprint, load_training_data, normalize_config, redact_prediction, train_model, model_input
from secrets_pipeline.candidates import preprocessing_config


class CalibrationTests(unittest.TestCase):
    def test_tied_scores_are_a_single_threshold(self):
        self.assertAlmostEqual(average_precision([1, 0], [.5, .5]), .5)
        self.assertAlmostEqual(average_precision([1, 0, 1], [.9, .8, .7]), (1 + 2 / 3) / 2)

    def test_no_false_confident_blocking(self):
        result = calibrate_thresholds([1, 1, 0, 0], [.99, .9, .2, .1])
        self.assertIsNone(result['block'])
        self.assertFalse(result['block_supported'])
        self.assertEqual(result['warn'], .9)

    def test_supported_block_uses_lower_bound(self):
        result = calibrate_thresholds([1] * 250 + [0] * 250, [.9] * 250 + [.1] * 250)
        self.assertTrue(result['block_supported'])
        self.assertGreaterEqual(result['block_metrics']['precision_lower'], .98)
        self.assertGreaterEqual(result['block'], result['warn'])

    def test_unknown_and_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics([None], [.2])
        with self.assertRaises(ValueError):
            average_precision([1], [float('nan')])
        with self.assertRaises(ValueError):
            calibrate_thresholds([0, 0], [.2, .3])

    def test_disabled_threshold_does_not_flag(self):
        self.assertEqual(binary_metrics([1, 0], [.9, .5], None)['predicted_positive'], 0)
        self.assertEqual(wilson_lower(0, 0), 0)


class DataAuditTests(unittest.TestCase):
    def fixture(self, **changes):
        rows = []
        for split in ['train', 'val', 'test']:
            for y in [0, 1]:
                rows.append(dict(id=f'{split}-{y}', document_id=f'doc-{split}', repo_id=f'repo-{split}',
                                 group_id=f'group-{split}', split=split, source='creddata', path='x.py',
                                 language='python', start=0, end=5, input_text='TARGET FIRST [fictional]',
                                 label=y, category='password', metadata={'preprocessing': preprocessing_config(legacy_scanner='none')}))
        for index, values in changes.items():
            rows[int(index)].update(values)
        return rows

    def with_file(self, rows, fn):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'candidates.jsonl'
            p.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            return fn(p)

    def test_unknowns_excluded_and_tests_not_returned(self):
        rows = self.fixture()
        rows.append({**rows[0], 'id': 'unknown', 'label': None})
        train, val, report = self.with_file(rows, lambda p: load_training_data(p, {}))
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        self.assertEqual(report['ignored_unknown_counts']['train'], 1)
        self.assertTrue(all(r['split'] not in ('test', 'test_ood') for r in train + val))

    def test_repo_and_group_leakage_rejected(self):
        for key in ('repo_id', 'group_id'):
            rows = self.fixture()
            rows[4][key] = rows[0][key]
            with self.assertRaisesRegex(ValueError, 'leakage'):
                self.with_file(rows, lambda p: load_training_data(p, {}))

    def test_synthetic_only_validation_requires_opt_in(self):
        rows = self.fixture()
        for r in rows:
            r['source'] = 'generated'
        with self.assertRaisesRegex(ValueError, 'Real validation'):
            self.with_file(rows, lambda p: load_training_data(p, {}))
        report = self.with_file(rows, lambda p: load_training_data(p, {'allow_synthetic_validation': True})[2])
        self.assertTrue(report['synthetic_validation_fallback'])

    def test_dry_run_needs_no_ml_dependencies(self):
        previously_loaded = set(sys.modules)
        result = self.with_file(self.fixture(), lambda p: train_model({}, p, p.parent / 'out', dry_run=True))
        self.assertFalse(result['model_trained'])
        self.assertFalse(result['model_downloaded'])
        self.assertNotIn('torch', set(sys.modules) - previously_loaded)
        self.assertNotIn('transformers', set(sys.modules) - previously_loaded)

    def test_resume_fingerprint_binds_schedule(self):
        self.assertEqual(config_fingerprint({}), config_fingerprint({'resume': 'a/path'}))
        self.assertNotEqual(config_fingerprint({}), config_fingerprint({'epochs': 7}))
        with self.assertRaises(ValueError):
            normalize_config({'learing_rate': .01})
        with self.assertRaises(ValueError):
            normalize_config({'epochs': 5.0})
        with self.assertRaises(ValueError):
            normalize_config({'learning_rate': float('nan')})

    def test_predictions_do_not_export_context(self):
        row = self.fixture()[0]
        row['metadata'] = {'raw_secret': 'never-export-this'}
        out = redact_prediction(row, .3)
        self.assertNotIn('input_text', out)
        self.assertNotIn('metadata', out)
        self.assertEqual(out['score'], .3)

    def test_missing_or_mixed_preprocessing_rejected(self):
        rows = self.fixture()
        rows[0]['metadata'] = {}
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.with_file(rows, lambda p: load_training_data(p, {}))
        rows = self.fixture()
        rows[0]['metadata']['preprocessing']['context_lines'] = 9
        with self.assertRaisesRegex(ValueError, 'Mixed candidate preprocessing'):
            self.with_file(rows, lambda p: load_training_data(p, {}))

    def test_mock_fixtures_rejected_outside_explicit_smoke(self):
        rows = self.fixture()
        rows[0]['metadata']['mock_data'] = True
        with self.assertRaisesRegex(ValueError, 'Mock generator'):
            self.with_file(rows, lambda p: load_training_data(p, {}))
        train, _, _ = self.with_file(rows, lambda p: load_training_data(p, {'allow_synthetic_validation': True}))
        self.assertEqual(len(train), 2)

    def test_key_ablation_preserves_target(self):
        original = '<candidate>password=fictional-value</candidate>\n<file>config.yml</file>\n>> password: fictional-value\n  username = example'
        masked = model_input(original, True)
        self.assertTrue(masked.startswith('<candidate>password=fictional-value</candidate>'))
        self.assertIn('>> KEY: fictional-value', masked)
        self.assertIn('  KEY = example', masked)
        with self.assertRaises(ValueError):
            model_input('noncanonical input', True)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Optional Torch/Transformers training environment is not installed')
class FrameworkSmokeTests(unittest.TestCase):
    def test_real_training_export_and_completed_resume_without_downloads(self):
        from secrets_pipeline.smoke_train import run_training_smoke
        with tempfile.TemporaryDirectory() as td:
            report = run_training_smoke(td)
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['downloaded_models'])
        self.assertLessEqual(report['checkpoint_export_max_score_error'], 1e-7)
        self.assertLessEqual(report['completed_resume_max_score_error'], 1e-7)


if __name__ == '__main__':
    unittest.main()
