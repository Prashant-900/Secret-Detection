import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from secrets_pipeline.experiments import (average_predictions, filter_dataset, normalize_suite,
                                          plan_experiments, run_suite, file_sha256)
from secrets_pipeline.train import atomic_json, normalize_config, redact_prediction
from secrets_pipeline.metrics import calibrate_thresholds
from secrets_pipeline.candidates import preprocessing_config


def fixture():
    docs, candidates = [], []
    for split, source in [('train', 'creddata'), ('train', 'generated'), ('val', 'creddata'), ('test', 'creddata')]:
        for label in (0, 1):
            ident = f'{split}-{source}-{label}'
            doc = dict(id=ident, repo_id=f'repo-{split}-{source}', group_id=f'group-{split}-{source}',
                       split=split, source=source, path='config.py', language='python', text='value unknown # ' + ident,
                       spans=[dict(id='span', start=0, end=5, label=label, category='password')], metadata={})
            docs.append(doc)
            candidates.append(dict(id=ident, document_id=ident, repo_id=doc['repo_id'], group_id=doc['group_id'],
                                   split=split, source=source, path=doc['path'], language=doc['language'], start=0, end=5,
                                   input_text='<candidate>value</candidate>\npassword=value', label=label, category='password',
                                   metadata={'preprocessing': preprocessing_config(legacy_scanner='none')}))
    val_doc = next(d for d in docs if d['split'] == 'val' and d['spans'][0]['label'] == 1)
    candidates.append(dict(id='val-unknown', document_id=val_doc['id'], repo_id=val_doc['repo_id'], group_id=val_doc['group_id'],
                           split='val', source='creddata', path='config.py', language='python', start=6, end=13,
                           input_text='<candidate>unknown</candidate>', label=None, category='unknown',
                           metadata={'preprocessing': preprocessing_config(legacy_scanner='none')}))
    return docs, candidates


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


class ExperimentTests(unittest.TestCase):
    def test_plan_contains_all_requested_ablations(self):
        jobs = plan_experiments({'include_mask_ablation': True})
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len({j['id'] for j in jobs}), 16)
        with self.assertRaises(ValueError):
            normalize_suite({'models': [{'name': '../bad', 'model_name': 'model'}]})

    def test_real_only_changes_train_not_evaluation(self):
        _, rows = fixture()
        filtered = filter_dataset(rows, 'real_only', ['creddata'])
        self.assertFalse(any(r['split'] == 'train' and r['source'] == 'generated' for r in filtered))
        self.assertEqual([r for r in filtered if r['split'] != 'train'], [r for r in rows if r['split'] != 'train'])

    def test_ensemble_matches_identity_not_order(self):
        first = [dict(id='a', document_id='d', start=0, end=1, label=1, score=.8),
                 dict(id='b', document_id='d', start=2, end=3, label=0, score=.4)]
        second = [{**first[1], 'score': .2}, {**first[0], 'score': 1.}]
        result = average_predictions([first, second])
        self.assertAlmostEqual(result[0]['score'], .9)
        self.assertAlmostEqual(result[1]['score'], .3)
        with self.assertRaises(ValueError):
            average_predictions([first, second[:1]])

    def test_suite_dry_run_does_not_train(self):
        docs, rows = fixture()
        with tempfile.TemporaryDirectory() as td:
            data, corpus = Path(td) / 'candidates.jsonl', Path(td) / 'documents.jsonl'
            write_jsonl(data, rows); write_jsonl(corpus, docs)
            result = run_suite({}, data, corpus, Path(td) / 'out', dry_run=True)
            self.assertEqual(result['jobs_count'], 8)
            self.assertFalse(result['model_trained'])
            self.assertFalse((Path(td) / 'out').exists())
            counts = {j['dataset']: j['training_examples'] for j in result['jobs']}
            self.assertEqual(counts, {'real_only': 2, 'augmented': 4})

    def test_coarse_positive_supervision_stays_ambiguous(self):
        docs, rows = fixture()
        coarse = {**docs[0], 'id': 'coarse', 'text': 'first second coarse-example',
                  'spans': [{'id': 'coarse-span', 'start': 0, 'end': 12, 'label': 1, 'category': 'password',
                             'metadata': {'annotation_precision': 'line_range'}}]}
        docs.append(coarse)
        for i, (start, end) in enumerate([(0, 5), (6, 12)]):
            rows.append({**rows[0], 'id': f'coarse-{i}', 'document_id': 'coarse', 'start': start, 'end': end,
                         'label': None, 'category': 'unknown'})
        with tempfile.TemporaryDirectory() as td:
            data, corpus = Path(td) / 'candidates.jsonl', Path(td) / 'documents.jsonl'
            write_jsonl(data, rows); write_jsonl(corpus, docs)
            self.assertEqual(run_suite({}, data, corpus, Path(td) / 'out', dry_run=True)['status'], 'dry_run')
            rows[-1]['label'] = 1
            write_jsonl(data, rows)
            with self.assertRaisesRegex(ValueError, 'labels do not match'):
                run_suite({}, data, corpus, Path(td) / 'out', dry_run=True)

    def test_suite_selects_on_real_val_and_resume_rejects_stale_inputs(self):
        docs, rows = fixture()
        predicted_splits = []
        suite_config = dict(models=[{'name': 'one', 'model_name': 'fake-one'}, {'name': 'two', 'model_name': 'fake-two'}],
                            datasets=['augmented'], seeds=[13], try_ensemble=True)
        def fake_train(config, data_path, out_dir, dry_run=False):
            normalized = normalize_config(config)
            out = Path(out_dir); (out / 'best').mkdir(parents=True)
            atomic_json(out / 'best' / 'config.json', {'fake_model': config['model_name']})
            atomic_json(out / 'best' / 'inference_config.json', {'max_length': 512, 'mask_keys': False,
                        'preprocessing': preprocessing_config(legacy_scanner='none'), 'inference_precision': 'fp32'})
            thresholds = calibrate_thresholds([0, 1], [.1, .9])
            report = dict(status='trained', config=normalized, data_audit={'data_sha256': file_sha256(data_path)}, thresholds=thresholds)
            atomic_json(out / 'run.json', report)
            return report
        def fake_predict(model_dir, population, **kwargs):
            predicted_splits.extend(r['split'] for r in population)
            model_two = 'two-' in str(model_dir)
            return [redact_prediction(r, (.2 if model_two else .95) if r['label'] is None else (.9 if r['label'] else .1)) for r in population]
        with tempfile.TemporaryDirectory() as td:
            data, corpus, out = Path(td) / 'candidates.jsonl', Path(td) / 'documents.jsonl', Path(td) / 'suite'
            write_jsonl(data, rows); write_jsonl(corpus, docs)
            with patch('secrets_pipeline.experiments.train_model', side_effect=fake_train), patch('secrets_pipeline.experiments.predict_records', side_effect=fake_predict):
                result = run_suite(suite_config, data, corpus, out)
                self.assertEqual(set(predicted_splits), {'val'})
                self.assertIn('two-', result['selected_name'])
                selected = json.loads((out / 'selected.json').read_text())
                self.assertFalse(selected['test_used_for_selection'])
                self.assertFalse(Path(selected['models'][0]['path']).is_absolute())
                self.assertEqual(selected['validation_metrics']['overall']['candidate_spans'], 3)
                ensemble = json.loads((out / 'ensemble_validation.json').read_text())
                self.assertEqual(len(ensemble['models']), 2)
                self.assertAlmostEqual(sum(m['weight'] for m in ensemble['models']), 1.)
                self.assertEqual(ensemble['thresholds']['calibration_precision'], 'fp32')
                self.assertIsNone(ensemble['thresholds']['block'])
                # Exercise the actual deployment bundle reader and weighted
                # scoring path, while replacing only the unavailable encoder.
                from secrets_pipeline.scan import read_detector, score_records
                detector = read_detector(bundle=out / 'ensemble_validation.json')
                val_rows = [r for r in rows if r['split'] == 'val']
                with patch('secrets_pipeline.predict.predict_records', side_effect=fake_predict):
                    deployed = score_records(val_rows, detector)
                self.assertAlmostEqual(next(p['score'] for p in deployed if p['id'] == 'val-unknown'), .575)
                # Unchanged resume reuses all saved models and validated prediction caches.
                with patch('secrets_pipeline.experiments.train_model', side_effect=AssertionError('Unexpected retrain')), patch('secrets_pipeline.experiments.predict_records', side_effect=AssertionError('Unexpected inference')):
                    self.assertEqual(run_suite(suite_config, data, corpus, out, resume=True)['status'], 'complete')
                rows[-1]['input_text'] += ' changed'
                write_jsonl(data, rows)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    run_suite(suite_config, data, corpus, out, resume=True)


if __name__ == '__main__':
    unittest.main()
