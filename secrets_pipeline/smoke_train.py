"""Exercise the real Torch/Transformers loop with a tiny local random BERT.

No model downloads and no performance claims: the synthetic fixture verifies
training, checkpoint export, inference reload, and exact completed-run resume.
Run after installing training dependencies, before launching costly experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import atomic_json, train_model
from .predict import predict_records


def run_training_smoke(out_dir, *, check_resume=True):
    try:
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    except ImportError as exc:
        raise RuntimeError('The framework smoke test requires the training environment (torch and transformers). It never downloads dependencies or models.') from exc
    out = Path(out_dir).resolve()
    if (out / 'training' / 'run.json').exists():
        raise ValueError('This directory already contains a training smoke run; choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    seed_model = out / 'tiny-random-bert'
    seed_model.mkdir(parents=True, exist_ok=True)
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<', '>', '/', 'candidate', 'file',
             'context', 'api', '_', 'key', 'password', '=', ':', 'dummy', 'value', 'example',
             'replace', 'me', 'safe', 'config', '.', 'py', 'json', 'secret', 'reference',
             'alpha', 'beta', 'gamma', 'delta', 'zero', 'one', 'two', 'three', 'four', 'five']
    (seed_model / 'vocab.txt').write_text('\n'.join(vocab) + '\n', encoding='utf-8')
    tokenizer = BertTokenizerFast(vocab_file=str(seed_model / 'vocab.txt'), do_lower_case=True,
                                  model_max_length=128)
    tokenizer.save_pretrained(seed_model)
    torch.manual_seed(37)
    torch.set_num_threads(2)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=4, intermediate_size=64, max_position_embeddings=128,
                        num_labels=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
    BertForSequenceClassification(config).save_pretrained(seed_model, safe_serialization=True)
    rows = []
    for split, count in [('train', 24), ('val', 12), ('test', 6)]:
        for i in range(count):
            label = i % 2
            value = 'dummy alpha beta' if label else 'replace me'
            rows.append(dict(id=f'{split}-{i}', document_id=f'document-{split}-{i}',
                             repo_id=f'repo-{split}', group_id=f'group-{split}', split=split,
                             source='generated', path='config.py', language='python',
                             start=0, end=len(value), input_text=f'<candidate>\n{value}\n</candidate>\n<file>config.py</file>\napi_key = {value}',
                             label=label, category='synthetic_smoke_fixture', metadata={'smoke_only': True}))
    candidate_path = out / 'candidates.jsonl'
    candidate_path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    train_config = dict(model_name=str(seed_model), epochs=1, batch_size=4, eval_batch_size=8,
                        grad_accumulation=2, learning_rate=1e-3, max_length=64, device='cpu',
                        precision='fp32', gradient_checkpointing=False, local_files_only=True,
                        allow_synthetic_validation=True, class_weight='none', seed=37,
                        source_weights={'generated': 1.0}, block_min_predictions=2,
                        early_stopping_patience=1, deterministic=True)
    atomic_json(out / 'training_config.json', train_config)
    result = train_model(train_config, candidate_path, out / 'training')
    test_rows = [r for r in rows if r['split'] == 'test']
    saved = predict_records(out / 'training', test_rows, device='cpu')
    checkpoint = out / 'training' / result['best_checkpoint'] / 'model'
    reloaded = predict_records(checkpoint, test_rows, max_length=64, device='cpu')
    max_error = max(abs(a['score'] - b['score']) for a, b in zip(saved, reloaded))
    if len(saved) != len(test_rows) or len(saved) != len(reloaded) or max_error > 1e-7:
        raise AssertionError('Best export and saved checkpoint predictions disagree')
    if result['thresholds']['block'] is not None:
        raise AssertionError('Synthetic-only calibration must not enable automatic blocking')
    resumed_error = None
    if check_resume:
        resume_cfg = {**train_config, 'resume': str(out / 'training')}
        train_model(resume_cfg, candidate_path, out / 'resumed-training')
        if not (out / 'resumed-training' / 'last_checkpoint.json').is_file():
            raise AssertionError('Resumed output is missing its own continuation checkpoint')
        resumed = predict_records(out / 'resumed-training', test_rows, device='cpu')
        resumed_error = max(abs(a['score'] - b['score']) for a, b in zip(saved, resumed))
        if len(resumed) != len(saved) or resumed_error > 1e-7:
            raise AssertionError('Completed-run resume changed predictions')
    report = dict(status='passed', trained_random_local_model=True, downloaded_models=False,
                  model='tiny random BERT, not a benchmark model', device='cpu',
                  training_examples=24, validation_examples=12, inference_examples=6,
                  checkpoint_export_max_score_error=max_error, completed_resume_checked=check_resume,
                  completed_resume_max_score_error=resumed_error, synthetic_block_disabled=True,
                  torch_version=torch.__version__)
    atomic_json(out / 'smoke_report.json', report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, help='Fresh scratch output directory')
    ap.add_argument('--no-resume-check', action='store_true')
    args = ap.parse_args(argv)
    print(json.dumps(run_training_smoke(args.out, check_resume=not args.no_resume_check), indent=2))


if __name__ == '__main__':
    main()
