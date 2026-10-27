"""Compare encoder/data ablations and an optional ensemble on real validation.

This command intentionally never evaluates test/test_ood. It runs one experiment
at a time on the caller's visible GPU, which is suitable for a shared server.
Use the separate train command under torchrun for an individual distributed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .evaluate import evaluate_documents
from .candidates import supervise_candidates
from .metrics import average_precision, calibrate_thresholds
from .predict import predict_records
from .schema import read_jsonl, validate_corpus
from .train import atomic_json, config_fingerprint, normalize_config, train_model


SUITE_DEFAULTS = dict(base_config={}, models=[{'name': 'codebert', 'model_name': 'microsoft/codebert-base'},
                                             {'name': 'unixcoder', 'model_name': 'microsoft/unixcoder-base'}],
                      datasets=['real_only', 'augmented'], seeds=[13, 37],
                      include_mask_ablation=False, try_ensemble=True)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_sha256(directory):
    """Bind prediction caches to actual exported weights and inference settings."""
    h = hashlib.sha256()
    root = Path(directory)
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(bytes.fromhex(file_sha256(path)))
    return h.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalize_suite(config):
    unknown = set(config) - set(SUITE_DEFAULTS)
    if unknown:
        raise ValueError('Unknown experiment suite keys: ' + ', '.join(sorted(unknown)))
    suite = {**SUITE_DEFAULTS, **config}
    suite['base_config'] = normalize_config(suite['base_config'])
    if suite['base_config']['resume']:
        raise ValueError('Use experiments --resume; base_config.resume must be null')
    if suite['base_config']['allow_synthetic_validation']:
        raise ValueError('Model selection requires real validation; synthetic validation is only for training smoke tests')
    if not suite['models'] or not suite['datasets'] or not suite['seeds']:
        raise ValueError('A suite needs models, datasets and seeds')
    names = []
    for model in suite['models']:
        if set(model) - {'name', 'model_name', 'revision'}:
            raise ValueError('Model entries support only name, model_name and revision')
        name = model.get('name', '')
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name) or not model.get('model_name'):
            raise ValueError('Model names must be safe unique directory labels and have model_name')
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate model names in suite')
    if set(suite['datasets']) - {'real_only', 'augmented'} or len(set(suite['datasets'])) != len(suite['datasets']):
        raise ValueError('datasets must contain unique real_only/augmented variants')
    if any(type(seed) is not int or not 0 <= seed < 2 ** 31 for seed in suite['seeds']) or len(set(suite['seeds'])) != len(suite['seeds']):
        raise ValueError('seeds must contain unique nonnegative integers')
    return suite


def plan_experiments(config):
    suite = normalize_suite(config)
    masks = [False, True] if suite['include_mask_ablation'] else [suite['base_config']['mask_keys']]
    jobs = []
    for model in suite['models']:
        for dataset in suite['datasets']:
            for mask in masks:
                for seed in suite['seeds']:
                    cfg = {**suite['base_config'], 'model_name': model['model_name'], 'seed': seed, 'mask_keys': mask}
                    if 'revision' in model:
                        cfg['revision'] = model['revision']
                    name = f"{model['name']}-{dataset}-{'masked' if mask else 'plain'}-seed{seed}"
                    jobs.append(dict(id=name, model_name=model['name'], dataset=dataset, seed=seed,
                                     mask_keys=mask, config=normalize_config(cfg)))
    return jobs


def filter_dataset(rows, variant, real_sources):
    """Only training composition changes; every validation/test identity survives."""
    if variant not in ('real_only', 'augmented'):
        raise ValueError('Unknown training data variant')
    return [r for r in rows if variant == 'augmented' or r['split'] != 'train' or r['source'] in real_sources]


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def _validation_population(candidates, documents_path, real_sources):
    """Keep only real validation documents for all model comparisons."""
    val_docs, identity = [], {}
    for doc in read_jsonl(documents_path):
        if doc['id'] in identity:
            raise ValueError('Duplicate document identity in experiment input')
        identity[doc['id']] = dict(split=doc['split'], source=doc['source'], repo_id=doc['repo_id'],
                                  group_id=doc['group_id'], path=doc['path'], language=doc['language'],
                                  length=len(doc['text']), spans=doc['spans'],
                                  text_sha256=hashlib.sha256(doc['text'].encode()).hexdigest())
        if doc['split'] == 'val' and doc['source'] in real_sources:
            val_docs.append(doc)
    grouped = {}
    for c in candidates:
        grouped.setdefault(c['document_id'], []).append(c)
    expected_labels = {}
    for document_id, proposals in grouped.items():
        if document_id not in identity:
            raise ValueError('Candidate references an unknown document')
        supervision = supervise_candidates({'spans': identity[document_id]['spans']}, proposals)
        for c, label in zip(proposals, supervision):
            if c['id'] in expected_labels:
                raise ValueError('Duplicate candidate identity in experiment input')
            expected_labels[c['id']] = label['label']
    for c in candidates:
        expected = identity.get(c['document_id'])
        if expected is None or any(expected[k] != c[k] for k in ('split', 'source', 'repo_id', 'group_id', 'path', 'language')):
            raise ValueError('Candidate provenance does not match the document corpus')
        if type(c['start']) is not int or type(c['end']) is not int or not 0 <= c['start'] < c['end'] <= expected['length']:
            raise ValueError('Candidate offsets do not match document text')
        if c.get('label') != expected_labels[c['id']]:
            raise ValueError('Candidate labels do not match the document annotations')
        recorded_sha = c.get('metadata', {}).get('document_text_sha256')
        if recorded_sha is not None and recorded_sha != expected['text_sha256']:
            raise ValueError('Candidate text provenance does not match the document corpus')
    val_ids = {d['id'] for d in val_docs}
    val_candidates = [c for c in candidates if c['document_id'] in val_ids]
    if not val_docs or {c.get('label') for c in val_candidates if c.get('label') is not None} != {0, 1}:
        raise ValueError('Experiment selection requires real validation documents and explicit positive/negative candidates')
    return val_docs, val_candidates


def average_predictions(prediction_sets):
    if not prediction_sets:
        raise ValueError('Cannot ensemble zero prediction sets')
    maps = []
    for rows in prediction_sets:
        mapping = {r['id']: r for r in rows}
        if len(mapping) != len(rows):
            raise ValueError('Duplicate candidate ids in ensemble predictions')
        maps.append(mapping)
    expected = set(maps[0])
    if any(set(m) != expected for m in maps[1:]):
        raise ValueError('Ensemble members must predict the same candidate population')
    out = []
    for row in prediction_sets[0]:
        for mapping in maps[1:]:
            other = mapping[row['id']]
            if any(row.get(k) != other.get(k) for k in ('document_id', 'start', 'end', 'label')):
                raise ValueError('Ensemble prediction identity/label mismatch')
        out.append({**row, 'score': sum(m[row['id']]['score'] for m in maps) / len(maps)})
    return out


def selection_key(result):
    overall = result['validation_metrics']['overall']
    return (overall['f1'], result['candidate_average_precision'],
            -overall['unknown_alerts_per_kloc'], -len(result['models']))


def _score(val_docs, predictions, thresholds, models, name):
    known = [r for r in predictions if r.get('label') in (0, 1)]
    metrics = evaluate_documents(val_docs, predictions, threshold=thresholds['warn'])
    return dict(name=name, models=models, thresholds=thresholds, validation_metrics=metrics,
                candidate_average_precision=average_precision([r['label'] for r in known], [r['score'] for r in known]),
                test_used_for_selection=False)


def run_suite(config, data_path, documents_path, out_dir, *, dry_run=False, resume=False):
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('Run the experiment suite once, outside torchrun. Train individual jobs with torchrun when needed.')
    suite = normalize_suite(config)
    jobs = plan_experiments(suite)
    corpus_manifest = validate_corpus(read_jsonl(documents_path))
    rows = list(read_jsonl(data_path))
    val_docs, val_candidates = _validation_population(rows, documents_path, suite['base_config']['real_validation_sources'])
    manifest = dict(config=suite, input_sha256=file_sha256(data_path), documents_sha256=file_sha256(documents_path),
                    corpus_sha256=corpus_manifest['corpus_sha256'],
                    validation_document_count=len(val_docs), validation_candidate_count=len(val_candidates),
                    test_used_for_selection=False)
    manifest['fingerprint'] = _fingerprint(manifest)
    if dry_run:
        audited_jobs = []
        with tempfile.TemporaryDirectory(prefix='secrets-suite-plan-') as td:
            paths = {}
            for variant in suite['datasets']:
                p = Path(td) / f'{variant}.jsonl'
                _write_jsonl(p, filter_dataset(rows, variant, suite['base_config']['real_validation_sources']))
                paths[variant] = p
            for job in jobs:
                audit = train_model(job['config'], paths[job['dataset']], Path(td) / job['id'], dry_run=True)
                audited_jobs.append({**job, 'training_examples': audit['training_examples'],
                                     'validation_examples': audit['validation_examples']})
        return dict(status='dry_run', manifest=manifest, jobs=audited_jobs, jobs_count=len(jobs),
                    model_downloaded=False, model_trained=False, test_used_for_selection=False)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'suite_manifest.json'
    if manifest_path.exists():
        if not resume:
            raise ValueError('Suite output already exists; choose a new output or pass --resume')
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('fingerprint') != manifest['fingerprint']:
            raise ValueError('Resume suite configuration, candidates or documents changed')
    elif resume and any(out.iterdir()):
        raise ValueError('Cannot resume a nonempty output without a matching suite manifest')
    atomic_json(manifest_path, manifest)
    dataset_paths = {}
    for variant in suite['datasets']:
        p = out / 'inputs' / f'{variant}.jsonl'
        _write_jsonl(p, filter_dataset(rows, variant, suite['base_config']['real_validation_sources']))
        dataset_paths[variant] = p
    results, prediction_cache, completed_jobs = [], {}, []
    for job in jobs:
        job_out = out / 'runs' / job['id']
        candidate_file = dataset_paths[job['dataset']]
        data_hash = file_sha256(candidate_file)
        cfg_hash = config_fingerprint(job['config'])
        report_path = job_out / 'run.json'
        if resume and report_path.exists():
            trained = json.loads(report_path.read_text(encoding='utf-8'))
            if config_fingerprint(trained['config']) != cfg_hash or trained['data_audit']['data_sha256'] != data_hash:
                raise ValueError('A completed experiment has stale config/data; use a new output')
        else:
            run_cfg = dict(job['config'])
            if resume and (job_out / 'last_checkpoint.json').exists():
                run_cfg['resume'] = str(job_out)
            trained = train_model(run_cfg, candidate_file, job_out)
        best_model = job_out / 'best'
        if not (best_model / 'config.json').is_file():
            raise ValueError('Completed experiment is missing its exported model')
        prediction_path = job_out / 'real_validation_all_predictions.jsonl'
        cache_path = job_out / 'real_validation_all_predictions.meta.json'
        cache_key = dict(config_fingerprint=cfg_hash, input_sha256=data_hash,
                         documents_sha256=manifest['documents_sha256'], model_sha256=model_sha256(best_model))
        cached = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else None
        if resume and cached and prediction_path.exists() and all(cached.get(k) == v for k, v in cache_key.items()) and cached.get('predictions_sha256') == file_sha256(prediction_path):
            predictions = list(read_jsonl(prediction_path))
        else:
            predictions = predict_records(best_model, val_candidates, batch_size=job['config']['eval_batch_size'])
            _write_jsonl(prediction_path, predictions)
            atomic_json(cache_path, {**cache_key, 'predictions_sha256': file_sha256(prediction_path)})
        if {p['id'] for p in predictions} != {c['id'] for c in val_candidates} or len(predictions) != len(val_candidates):
            raise ValueError('Validation predictions do not cover the fixed candidate population')
        model_ref = {'path': str(best_model.relative_to(out)).replace('\\', '/'), 'weight': 1.0,
                     'name': job['model_name'], 'run': job['id'], 'sha256': cache_key['model_sha256']}
        # Calibrate from the actual fp32 predictions used for full-span scoring,
        # rather than reusing an older/mixed-precision threshold artifact.
        known = [r for r in predictions if r.get('label') in (0, 1)]
        thresholds = calibrate_thresholds([r['label'] for r in known], [r['score'] for r in known], **{
            k: job['config'][k] for k in ('warn_beta', 'block_target_precision', 'block_min_predictions', 'block_wilson_z')})
        thresholds.update(validation_sources=sorted({d['source'] for d in val_docs}),
                          test_used_for_selection=False, calibration_precision='fp32')
        scored = _score(val_docs, predictions, thresholds, [model_ref], job['id'])
        atomic_json(job_out / 'real_validation_span_metrics.json', scored)
        results.append(scored)
        prediction_cache[job['id']] = predictions
        completed_jobs.append(job['id'])
        atomic_json(out / 'progress.json', {'completed_jobs': completed_jobs, 'jobs_count': len(jobs), 'test_used_for_selection': False})
    if suite['try_ensemble'] and len(suite['models']) > 1:
        winners = []
        for model in suite['models']:
            options = [r for r in results if r['models'][0]['name'] == model['name']]
            winners.append(max(options, key=selection_key))
        predictions = average_predictions([prediction_cache[r['name']] for r in winners])
        known = [r for r in predictions if r.get('label') in (0, 1)]
        thresholds = calibrate_thresholds([r['label'] for r in known], [r['score'] for r in known], **{
            k: suite['base_config'][k] for k in ('warn_beta', 'block_target_precision', 'block_min_predictions', 'block_wilson_z')})
        thresholds.update(validation_sources=sorted({d['source'] for d in val_docs}), test_used_for_selection=False,
                          calibration_precision='fp32')
        members = [{**r['models'][0], 'weight': 1 / len(winners)} for r in winners]
        ensemble = _score(val_docs, predictions, thresholds, members, 'mean-of-family-winners')
        _write_jsonl(out / 'ensemble_validation_predictions.jsonl', predictions)
        atomic_json(out / 'ensemble_validation.json', ensemble)
        results.append(ensemble)
    selected = max(results, key=selection_key)
    selected = {**selected, 'selection_rule': 'real_validation_span_f1_then_candidate_ap_then_lower_unknown_burden_then_fewer_models',
                'suite_fingerprint': manifest['fingerprint'], 'input_sha256': manifest['input_sha256'],
                'documents_sha256': manifest['documents_sha256'], 'corpus_sha256': manifest['corpus_sha256'], 'validation_only': True}
    atomic_json(out / 'selected.json', selected)
    # Keep this compact; detailed per-category errors remain in each run's report.
    leaderboard = [dict(name=r['name'], models=r['models'], span_f1=r['validation_metrics']['overall']['f1'],
                        span_precision=r['validation_metrics']['overall']['precision'],
                        span_recall=r['validation_metrics']['overall']['recall'],
                        unknown_alerts_per_kloc=r['validation_metrics']['overall']['unknown_alerts_per_kloc'],
                        candidate_average_precision=r['candidate_average_precision'])
                   for r in sorted(results, key=selection_key, reverse=True)]
    atomic_json(out / 'leaderboard.json', {'results': leaderboard, 'test_used_for_selection': False})
    return dict(status='complete', selected=str(out / 'selected.json'), selected_name=selected['name'],
                jobs_count=len(jobs), leaderboard=leaderboard, test_used_for_selection=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--documents', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(argv)
    result = run_suite(json.loads(Path(args.config).read_text(encoding='utf-8')), args.data,
                       args.documents, args.out, dry_run=args.dry_run, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
