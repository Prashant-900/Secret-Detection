"""Explicit inference stage for a saved encoder; no training or threshold tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .train import atomic_json, redact_prediction, model_input


def resolve_model_dir(model_dir):
    model_dir = Path(model_dir)
    if (model_dir / 'best' / 'config.json').is_file():
        model_dir = model_dir / 'best'
    if not (model_dir / 'config.json').is_file():
        raise ValueError('Expected a saved local model directory or completed run directory')
    return model_dir


def predict_records(model_dir, records, *, batch_size=32, max_length=None, device=None, precision='fp32'):
    """Return location-only predictions. Always loads local saved weights/tokenizer.

    ``records`` must be label-independent candidates, generated identically at
    training and deployment. Labels are carried for evaluation but never fed in.
    """
    model_dir = resolve_model_dir(model_dir)
    config_path = model_dir / 'inference_config.json'
    inference_config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    max_length = max_length if max_length is not None else inference_config.get('max_length', 512)
    if batch_size < 1 or max_length < 1 or precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('Invalid inference configuration')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    selected_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if precision != 'fp32' and selected_device.type != 'cuda':
        raise ValueError('Mixed-precision inference requires CUDA')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.truncation_side = 'right'
    if max_length > tokenizer.model_max_length:
        raise ValueError(f"max_length exceeds this tokenizer's supported budget ({tokenizer.model_max_length})")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
            attn_implementation=inference_config.get('attention_implementation', 'eager')).to(selected_device)
    model.eval()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    result, batch = [], []

    def run(part):
        expected_preprocessing = inference_config.get('preprocessing')
        if expected_preprocessing is not None and any(r.get('metadata', {}).get('preprocessing') != expected_preprocessing for r in part):
            raise ValueError('Candidate preprocessing differs from the saved model; use its frozen context/extractor settings')
        texts = [model_input(r['input_text'], inference_config.get('mask_keys', False)) for r in part]
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Every candidate needs target-first input_text')
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        encoded = {k: v.to(selected_device) for k, v in encoded.items()}
        with torch.inference_mode(), torch.autocast(device_type=selected_device.type, dtype=dtype, enabled=precision != 'fp32'):
            logits = model(**encoded).logits
        if logits.shape[-1] != 2:
            raise ValueError('Saved model must have two classification labels')
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
        return [redact_prediction(r, p) for r, p in zip(part, probabilities)]

    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            result.extend(run(batch)); batch = []
    if batch:
        result.extend(run(batch))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', required=True, choices=['val', 'test', 'test_ood', 'train', 'all'])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-length', type=int, help='Default: saved training token budget')
    ap.add_argument('--device')
    ap.add_argument('--precision', default='fp32', choices=['fp32', 'fp16', 'bf16'])
    args = ap.parse_args(argv)
    def rows():
        with Path(args.data).open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    r = json.loads(line)
                    if args.split == 'all' or r['split'] == args.split:
                        yield r
    predictions = predict_records(args.model_dir, rows(), batch_size=args.batch_size,
                                  max_length=args.max_length, device=args.device, precision=args.precision)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as out:
        for r in predictions:
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
    digest = hashlib.sha256()
    with Path(args.data).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    atomic_json(destination.with_suffix('.meta.json'), dict(model_dir=str(resolve_model_dir(args.model_dir).resolve()),
                data_sha256=digest.hexdigest(), split=args.split, predictions=len(predictions),
                threshold_tuned=False, max_length=args.max_length))
    print(json.dumps({'predictions': len(predictions), 'out': str(destination), 'split': args.split}))


if __name__ == '__main__':
    main()
