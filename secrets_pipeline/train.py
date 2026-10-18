"""Train and export a contextual code encoder without consulting held-out tests.

Run with ``python -m secrets_pipeline.train`` or ``torchrun --nproc_per_node=N
-m secrets_pipeline.train``. ML libraries are imported only after the CPU audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
from collections import Counter
from contextlib import nullcontext

from .metrics import average_precision, binary_metrics, calibrate_thresholds


DEFAULTS = dict(
    model_name='microsoft/codebert-base', revision=None, epochs=5, batch_size=16,
    eval_batch_size=32, grad_accumulation=2, learning_rate=2e-5, weight_decay=0.01,
    warmup_ratio=0.06, max_length=512, precision='bf16', gradient_checkpointing=True,
    seed=13, early_stopping_patience=2, selection_metric='average_precision',
    real_validation_sources=['creddata', 'reviewed_real'], allow_synthetic_validation=False,
    source_weights={'creddata': 1.0, 'tier_b': 0.5, 'generated': 0.5},
    class_weight='balanced', max_class_weight=10.0, warn_beta=1.0,
    block_target_precision=0.98, block_min_predictions=30, block_wilson_z=1.96,
    resume=None, local_files_only=False, num_workers=0, max_grad_norm=1.0,
    deterministic=True, min_delta=1e-5, mask_keys=False, device='auto', attention_implementation='eager',
)


def normalize_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown training config keys: ' + ', '.join(sorted(unknown)))
    cfg = {**DEFAULTS, **config}
    for k in ('epochs', 'batch_size', 'eval_batch_size', 'grad_accumulation', 'max_length', 'early_stopping_patience'):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    for k in ('learning_rate', 'weight_decay', 'warmup_ratio', 'max_class_weight', 'warn_beta',
              'block_target_precision', 'block_wilson_z', 'max_grad_norm', 'min_delta'):
        if not isinstance(cfg[k], (int, float)) or not math.isfinite(cfg[k]):
            raise ValueError(f'{k} must be a finite number')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2 ** 31:
        raise ValueError('seed must be an integer in [0, 2**31)')
    if type(cfg['num_workers']) is not int or type(cfg['block_min_predictions']) is not int:
        raise ValueError('num_workers and block_min_predictions must be integers')
    if cfg['precision'] not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('precision must be bf16, fp16 or fp32')
    if cfg['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if cfg['attention_implementation'] not in ('eager', 'sdpa'):
        raise ValueError('attention_implementation must be eager or sdpa')
    if cfg['selection_metric'] not in ('average_precision', 'f1'):
        raise ValueError('selection_metric must be average_precision or f1')
    if cfg['class_weight'] not in ('balanced', 'none'):
        raise ValueError('class_weight must be balanced or none')
    if not 0 < cfg['learning_rate'] or not 0 <= cfg['warmup_ratio'] < 1:
        raise ValueError('Invalid learning rate / warmup ratio')
    if cfg['num_workers'] < 0 or cfg['max_class_weight'] <= 0 or cfg['max_grad_norm'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('Invalid loader or loss parameters')
    if not cfg['real_validation_sources'] or any(not isinstance(s, str) for s in cfg['real_validation_sources']):
        raise ValueError('real_validation_sources must be a non-empty list of source names')
    if cfg['warn_beta'] <= 0 or not 0 < cfg['block_target_precision'] <= 1 or cfg['block_min_predictions'] < 1 or cfg['block_wilson_z'] < 0:
        raise ValueError('Invalid threshold calibration parameters')
    if any(not math.isfinite(float(w)) or float(w) <= 0 for w in cfg['source_weights'].values()):
        raise ValueError('Source weights must be finite and positive')
    return cfg


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def config_fingerprint(config):
    # A continuation must use the same optimization schedule and input data.
    stable = {k: v for k, v in normalize_config(config).items() if k not in ('resume', 'local_files_only', 'num_workers')}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


_ASSIGNMENT_KEY = re.compile(r'''(?m)^(\s*(?:>>\s*)?(?:[-{]\s*)?)(["']?)([A-Za-z_][A-Za-z0-9_.-]*)\2(\s*[:=])''')


def model_input(text, mask_keys=False):
    """Optional ablation: hide assignment keys in context, preserving the target.

    This intentionally removes context assignment names, not the secret value or
    the <candidate> section. It tests reliance on key names; it is not a semantic
    parser and does not promise to remove every identifier in every language.
    """
    if not mask_keys:
        return text
    marker = '</candidate>'
    if marker not in text:
        raise ValueError('mask_keys requires the canonical <candidate>...</candidate> input format')
    target, context = text.split(marker, 1)
    return target + marker + _ASSIGNMENT_KEY.sub(lambda m: m[1] + m[2] + 'KEY' + m[2] + m[4], context)


def load_training_data(data_path, config):
    """Stream the input; retain only train and validation records, never test text.

    Read identity metadata for every split so train/test source overlap fails
    before training. No held-out labels contribute to class weights or tuning.
    """
    cfg = normalize_config(config)
    parts = {'train': [], 'val': []}
    group_split, repo_split, counts, unknown_counts = {}, {}, Counter(), Counter()
    sha = hashlib.sha256()
    ids = set()
    source_counts = Counter()
    preprocessing = None
    preprocessing_seen = False
    with Path(data_path).open('rb') as stream:
        for n, raw in enumerate(stream, 1):
            sha.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if (row.get('metadata', {}).get('mock_data') is True or row.get('metadata', {}).get('generation_backend') == 'mock') and not cfg['allow_synthetic_validation']:
                raise ValueError('Mock generator fixtures cannot enter production training/evaluation; use explicit smoke training or regenerate with the real local LLM')
            settings = row.get('metadata', {}).get('preprocessing')
            if settings is None:
                if not cfg['allow_synthetic_validation']:
                    raise ValueError('Candidate preprocessing provenance is missing; regenerate candidates with the current pipeline (unversioned inputs are allowed only for explicit smoke training)')
            else:
                if not isinstance(settings, dict) or settings.get('version') != 1:
                    raise ValueError('Unsupported candidate preprocessing provenance')
                if type(settings.get('context_lines')) is not int or settings['context_lines'] < 0 or type(settings.get('max_chars')) is not int or settings['max_chars'] < 256:
                    raise ValueError('Invalid recorded candidate context settings')
                hashes = settings.get('code_sha256')
                if (not isinstance(hashes, dict) or not isinstance(hashes.get('candidates'), str)
                    or any((h is None and (k not in ('legacy_scan', 'rules', 'model') or settings.get('legacy_scanner') != 'none'))
                           or (h is not None and (not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h))) for k, h in hashes.items())):
                    raise ValueError('Missing or invalid candidate implementation hashes')
                if not isinstance(settings.get('legacy_scanner'), str):
                    raise ValueError('Missing legacy candidate extractor setting')
                if settings['legacy_scanner'] != 'none' and any(not hashes.get(k) for k in ('legacy_scan', 'rules', 'model')):
                    raise ValueError('Legacy candidate extraction requires its scanner/rules/model hashes')
            if preprocessing_seen and settings != preprocessing:
                raise ValueError('Mixed candidate preprocessing settings or implementation versions; regenerate one consistent candidate corpus')
            preprocessing = settings
            preprocessing_seen = True
            split = row.get('split')
            if split not in ('train', 'val', 'test', 'test_ood'):
                raise ValueError(f'Invalid split at candidate row {n}')
            rid = row.get('id')
            if not rid or rid in ids:
                raise ValueError(f'Missing or duplicate candidate id at row {n}')
            ids.add(rid)
            for key, mapping in (('group_id', group_split), ('repo_id', repo_split)):
                identity = row.get(key)
                if not identity:
                    raise ValueError(f'Missing {key} at row {n}')
                previous = mapping.setdefault(identity, split)
                if previous != split:
                    raise ValueError(f'{key} leakage across {previous} and {split}: {identity}')
            counts[split] += 1
            if split not in parts:
                continue
            label = row.get('label')
            if label is None:
                unknown_counts[split] += 1
                continue
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f'Non-binary supervised label at row {n}')
            if not isinstance(row.get('input_text'), str) or not row['input_text'].strip():
                raise ValueError(f'Missing target-first input_text at row {n}')
            parts[split].append(row)
            source_counts[(split, row.get('source', 'unknown'), label)] += 1
    train = parts['train']
    real_val = [r for r in parts['val'] if r.get('source') in cfg['real_validation_sources']]
    synthetic_fallback = False
    if {r['label'] for r in real_val} != {0, 1}:
        if not cfg['allow_synthetic_validation']:
            raise ValueError('Real validation data needs both explicit positives and negatives. Import held-out CredData/reviewed-real labels; allow_synthetic_validation is only for a smoke experiment.')
        real_val = parts['val']
        synthetic_fallback = True
    if {r['label'] for r in train} != {0, 1}:
        raise ValueError('Training requires explicit positive and negative examples')
    if {r['label'] for r in real_val} != {0, 1}:
        raise ValueError('Validation requires explicit positive and negative examples')
    summary = dict(data_sha256=sha.hexdigest(), all_split_candidate_counts=dict(counts),
                   training_examples=len(train), validation_examples=len(real_val),
                   ignored_unknown_counts=dict(unknown_counts), synthetic_validation_fallback=synthetic_fallback,
                   validation_sources=sorted({r['source'] for r in real_val}),
                   supervised_source_counts=[dict(split=s, source=src, label=y, count=c)
                                             for (s, src, y), c in sorted(source_counts.items())],
                   train_positive_count=sum(r['label'] for r in train),
                   validation_positive_count=sum(r['label'] for r in real_val),
                   test_used_for_selection=False, preprocessing=preprocessing)
    return train, real_val, summary


def redact_prediction(row, score):
    """Keep location and supervision metadata; never copy literals/context."""
    safe = {k: row[k] for k in ('id', 'document_id', 'repo_id', 'group_id', 'split', 'source',
                               'path', 'language', 'start', 'end', 'label', 'category') if k in row}
    safe['score'] = float(score)
    return safe


def _resolve_resume(path):
    p = Path(path)
    if p.is_file():
        pointer = json.loads(p.read_text(encoding='utf-8'))
        p = p.parent / pointer['checkpoint']
    elif (p / 'last_checkpoint.json').is_file():
        pointer = json.loads((p / 'last_checkpoint.json').read_text(encoding='utf-8'))
        p = p / pointer['checkpoint']
    if not (p / 'complete.json').is_file() or not (p / 'training_state.pt').is_file():
        raise ValueError('Resume must point to a complete checkpoint or a run with last_checkpoint.json')
    return p.resolve()


def train_model(config, data_path, out_dir, dry_run=False):
    cfg = normalize_config(config)
    train_rows, val_rows, audit = load_training_data(data_path, cfg)
    out = Path(out_dir)
    if dry_run:
        return {'status': 'dry_run', **audit, 'config': cfg,
                'effective_batch_size_per_gpu': cfg['batch_size'] * cfg['grad_accumulation'],
                'model_downloaded': False, 'model_trained': False}

    # Optional dependencies are intentionally below the entire data audit.
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world > 1
    cuda = torch.cuda.is_available() and cfg['device'] != 'cpu'
    if cfg['device'] == 'cuda' and not cuda:
        raise ValueError('CUDA was requested but is unavailable')
    if distributed:
        if not cuda:
            raise ValueError('Distributed encoder training currently requires CUDA')
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    device = torch.device(f'cuda:{local_rank}' if cuda else 'cpu')
    if cfg['precision'] != 'fp32' and not cuda:
        raise ValueError('Mixed precision requires CUDA. Set precision=fp32 for CPU smoke training.')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; choose fp16 or fp32 explicitly')
    random.seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    torch.manual_seed(cfg['seed'] + rank)
    if cuda:
        torch.cuda.manual_seed_all(cfg['seed'] + rank)
    if cfg['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(cfg['deterministic'])
    resume_dir = _resolve_resume(cfg['resume']) if cfg['resume'] else None
    if not resume_dir and any((out / artifact).exists() for artifact in ('run.json', 'last_checkpoint.json', 'best/config.json')):
        raise ValueError('Output already contains a run/checkpoint. Choose a new output or explicitly resume it.')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'data_audit.json', audit)
        atomic_json(out / 'config.json', cfg)
    if distributed:
        dist.barrier()
    origin = str(resume_dir / 'model') if resume_dir else cfg['model_name']
    load_args = {'local_files_only': True if resume_dir else cfg['local_files_only'], 'trust_remote_code': False}
    if not resume_dir and cfg['revision']:
        load_args['revision'] = cfg['revision']
    tokenizer = AutoTokenizer.from_pretrained(origin, **load_args)
    tokenizer.truncation_side = 'right'  # candidates.build_input puts the target first.
    if cfg['max_length'] > tokenizer.model_max_length:
        raise ValueError(f"max_length exceeds this tokenizer's supported budget ({tokenizer.model_max_length})")
    base = AutoModelForSequenceClassification.from_pretrained(origin, num_labels=2,
            attn_implementation=cfg['attention_implementation'], **load_args).to(device)
    effective_model_revision = getattr(base.config, '_commit_hash', None)
    if cfg['gradient_checkpointing']:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.config.use_cache = False
    model = DistributedDataParallel(base, device_ids=[local_rank]) if distributed else base

    def collate(rows):
        enc = tokenizer([model_input(r['input_text'], cfg['mask_keys']) for r in rows], truncation=True,
                        max_length=cfg['max_length'], padding=True, return_tensors='pt')
        enc['labels'] = torch.tensor([r['label'] for r in rows], dtype=torch.long)
        enc['source_weights'] = torch.tensor([cfg['source_weights'].get(r['source'], 1.0) for r in rows], dtype=torch.float32)
        return enc

    sampler = DistributedSampler(train_rows, shuffle=True, seed=cfg['seed']) if distributed else None
    loader = DataLoader(train_rows, batch_size=cfg['batch_size'], shuffle=sampler is None,
                        sampler=sampler, collate_fn=collate, num_workers=cfg['num_workers'], pin_memory=cuda)
    n0, n1 = len(train_rows) - audit['train_positive_count'], audit['train_positive_count']
    weights = [min(cfg['max_class_weight'], len(train_rows) / (2 * n)) for n in (n0, n1)] if cfg['class_weight'] == 'balanced' else [1., 1.]
    class_weights = torch.tensor(weights, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    steps_per_epoch = math.ceil(len(loader) / cfg['grad_accumulation'])
    total_steps = steps_per_epoch * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg['warmup_ratio']), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'fp16')
    amp_dtype = torch.bfloat16 if cfg['precision'] == 'bf16' else torch.float16
    start_epoch, global_step, best_value, bad_epochs, best_checkpoint = 0, 0, -1., 0, None
    history = []

    def capture_rng():
        return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if cuda else None}

    if resume_dir:
        state = torch.load(resume_dir / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['config_fingerprint'] != config_fingerprint(cfg) or state['data_sha256'] != audit['data_sha256']:
            raise ValueError('Resume configuration/data differs from checkpoint; use a new run for a changed experiment')
        if state['world_size'] != world:
            raise ValueError('Exact resume requires the original number of GPU workers')
        optimizer.load_state_dict(state['optimizer'])
        for opt_state in optimizer.state.values():
            for k, value in opt_state.items():
                if torch.is_tensor(value):
                    opt_state[k] = value.to(device)
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, global_step = state['epoch'] + 1, state['global_step']
        best_value, bad_epochs = state['best_value'], state['bad_epochs']
        best_checkpoint, history = state['best_checkpoint'], state['history']
        effective_model_revision = state.get('effective_model_revision', effective_model_revision)
        # Checkpoint references are relative to the run, so moving a complete run
        # archive is safe. Continuing into another output imports its best model.
        previous_root = resume_dir.parent.parent
        previous_best = previous_root / best_checkpoint
        if not (previous_best / 'model' / 'config.json').is_file():
            raise ValueError('Resume needs the complete run, including its best checkpoint model')
        if previous_root.resolve() != out.resolve():
            imported_best = out / 'checkpoints' / ('resumed-best-' + previous_best.name)
            if rank == 0:
                imported_best.mkdir(parents=True, exist_ok=True)
                shutil.copytree(previous_best / 'model', imported_best / 'model', dirs_exist_ok=True)
                atomic_json(imported_best / 'complete.json', {'imported_best_model': True, 'data_sha256': audit['data_sha256']})
            best_checkpoint = str(imported_best.relative_to(out))
            if rank == 0:
                # Preserve a resumable last checkpoint even if the resumed run
                # has already completed all epochs or is interrupted before its
                # next optimizer epoch. The new output is self-contained.
                imported_last = out / 'checkpoints' / 'resumed-last'
                imported_last.mkdir(parents=True, exist_ok=True)
                shutil.copytree(resume_dir / 'model', imported_last / 'model', dirs_exist_ok=True)
                imported_state = {**state, 'best_checkpoint': best_checkpoint}
                torch.save(imported_state, imported_last / 'training_state.pt.tmp')
                os.replace(imported_last / 'training_state.pt.tmp', imported_last / 'training_state.pt')
                atomic_json(imported_last / 'complete.json', {'epoch': state['epoch'] + 1, 'data_sha256': audit['data_sha256'], 'imported': True})
                atomic_json(out / 'last_checkpoint.json', {'checkpoint': str(imported_last.relative_to(out))})
            if distributed:
                dist.barrier()
        rs = state['rng_states'][rank]
        random.setstate(rs['python']); np.random.set_state(rs['numpy']); torch.set_rng_state(rs['torch'])
        if cuda:
            torch.cuda.set_rng_state_all(rs['cuda'])

    def infer_validation(net):
        net.eval()
        scores = []
        with torch.inference_mode():
            for b in DataLoader(val_rows, batch_size=cfg['eval_batch_size'], collate_fn=collate, num_workers=0):
                inputs = {k: v.to(device) for k, v in b.items() if k not in ('labels', 'source_weights')}
                # Calibrate in the same fp32 mode used by the default deployed
                # predictor. Training precision must not move threshold edges.
                logits = net(**inputs).logits
                scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        return scores

    try:
        for epoch in range(start_epoch, cfg['epochs']):
            if sampler:
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            began = time.monotonic()
            for step, batch in enumerate(loader):
                labels = batch.pop('labels').to(device)
                source_weights = batch.pop('source_weights').to(device)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                # Correct the last partial accumulation window instead of shrinking its update.
                window_start = (step // cfg['grad_accumulation']) * cfg['grad_accumulation']
                window_size = min(cfg['grad_accumulation'], len(loader) - window_start)
                update_now = (step + 1) % cfg['grad_accumulation'] == 0 or step + 1 == len(loader)
                sync = model.no_sync() if distributed and not update_now else nullcontext()
                with sync:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg['precision'] != 'fp32'):
                        logits = model(**batch).logits
                        example_weights = source_weights * class_weights[labels]
                        per_example = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='none')
                        loss = (per_example * example_weights).sum() / example_weights.sum()
                    total_loss += float(loss.detach())
                    scaler.scale(loss / window_size).backward()
                if update_now:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
                    before_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # Overflowed fp16 updates must not advance the learning-rate schedule.
                    if scaler.get_scale() >= before_scale:
                        scheduler.step()
                        global_step += 1
                    optimizer.zero_grad(set_to_none=True)
            if distributed:
                dist.barrier()
            result = None
            if rank == 0:
                scores = infer_validation(base)
                ys = [r['label'] for r in val_rows]
                operating = calibrate_thresholds(ys, scores, **{k: cfg[k] for k in
                    ('warn_beta', 'block_target_precision', 'block_min_predictions', 'block_wilson_z')})
                value = operating['average_precision'] if cfg['selection_metric'] == 'average_precision' else operating['warn_metrics']['f1']
                improved = value > best_value + cfg['min_delta']
                best_value = value if improved else best_value
                bad_epochs = 0 if improved else bad_epochs + 1
                checkpoint = out / 'checkpoints' / f'epoch-{epoch + 1:04d}'
                if improved:
                    best_checkpoint = str(checkpoint.relative_to(out))
                result = dict(epoch=epoch + 1, training_loss_rank0=total_loss / len(loader),
                              validation_average_precision=operating['average_precision'],
                              validation_warn_f1=operating['warn_metrics']['f1'],
                              best_value=best_value, bad_epochs=bad_epochs, best_checkpoint=best_checkpoint,
                              improved=improved, seconds=time.monotonic() - began)
            if distributed:
                shared = [result]
                dist.broadcast_object_list(shared, src=0)
                result = shared[0]
            best_value, bad_epochs, best_checkpoint = result['best_value'], result['bad_epochs'], result['best_checkpoint']
            history.append(result)
            rng_states = [capture_rng()]
            if distributed:
                rng_states = [None] * world
                dist.all_gather_object(rng_states, capture_rng())
            if rank == 0:
                checkpoint = out / 'checkpoints' / f'epoch-{epoch + 1:04d}'
                checkpoint.mkdir(parents=True, exist_ok=True)
                base.save_pretrained(checkpoint / 'model', safe_serialization=True)
                tokenizer.save_pretrained(checkpoint / 'model')
                state = dict(epoch=epoch, global_step=global_step, optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng_states=rng_states,
                             best_value=best_value, bad_epochs=bad_epochs, best_checkpoint=best_checkpoint,
                             history=history, config_fingerprint=config_fingerprint(cfg), data_sha256=audit['data_sha256'],
                             world_size=world, effective_model_revision=effective_model_revision)
                tmp_state = checkpoint / 'training_state.pt.tmp'
                torch.save(state, tmp_state)
                os.replace(tmp_state, checkpoint / 'training_state.pt')
                atomic_json(checkpoint / 'complete.json', {'epoch': epoch + 1, 'data_sha256': audit['data_sha256']})
                atomic_json(out / 'last_checkpoint.json', {'checkpoint': str(checkpoint.relative_to(out))})
                atomic_json(out / 'history.json', history)
                print(json.dumps(result, sort_keys=True), flush=True)
            if distributed:
                dist.barrier()
            if bad_epochs >= cfg['early_stopping_patience']:
                break
        if rank == 0:
            if best_checkpoint is None:
                raise ValueError('No complete best checkpoint is available')
            # This final export is deployable: full weights, tokenizer and operating points.
            del model
            del base
            del optimizer
            del scheduler
            if cuda:
                torch.cuda.empty_cache()
            best = AutoModelForSequenceClassification.from_pretrained(out / best_checkpoint / 'model',
                    local_files_only=True, attn_implementation=cfg['attention_implementation']).to(device)
            scores = infer_validation(best)
            ys = [r['label'] for r in val_rows]
            thresholds = calibrate_thresholds(ys, scores, **{k: cfg[k] for k in
                ('warn_beta', 'block_target_precision', 'block_min_predictions', 'block_wilson_z')})
            thresholds.update(validation_sources=audit['validation_sources'],
                              synthetic_validation_fallback=audit['synthetic_validation_fallback'],
                              data_sha256=audit['data_sha256'], test_used_for_selection=False,
                              calibration_precision='fp32')
            if audit['synthetic_validation_fallback']:
                thresholds.update(block=None, block_supported=False, block_metrics=None,
                                  block_reason='synthetic_validation_is_not_eligible_for_automatic_blocking')
            best.save_pretrained(out / 'best', safe_serialization=True)
            tokenizer.save_pretrained(out / 'best')
            atomic_json(out / 'best' / 'inference_config.json', {'max_length': cfg['max_length'], 'mask_keys': cfg['mask_keys'],
                        'input_format': 'candidate-first-v1', 'model_name': cfg['model_name'], 'model_revision': effective_model_revision,
                        'attention_implementation': cfg['attention_implementation'], 'inference_precision': 'fp32',
                        'preprocessing': audit['preprocessing']})
            atomic_json(out / 'thresholds.json', thresholds)
            atomic_json(out / 'best' / 'thresholds.json', thresholds)
            with (out / 'validation_predictions.jsonl').open('w', encoding='utf-8') as fh:
                for r, p in zip(val_rows, scores):
                    fh.write(json.dumps(redact_prediction(r, p), ensure_ascii=False) + '\n')
            import transformers
            report = dict(status='trained', config=cfg, data_audit=audit, thresholds=thresholds,
                          best_checkpoint=best_checkpoint, exported_model=str((out / 'best').resolve()),
                          effective_batch_size=cfg['batch_size'] * cfg['grad_accumulation'] * world,
                          world_size=world, history=history, model_revision=effective_model_revision,
                          versions={'torch': torch.__version__, 'transformers': transformers.__version__},
                          validation_metric_scope='explicit labeled candidates; full-document span evaluation is separate')
            atomic_json(out / 'run.json', report)
        else:
            report = {'status': 'worker_complete', 'rank': rank}
        if distributed:
            dist.barrier()
        return report
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--data', required=True, help='Candidate JSONL from secrets_pipeline.candidates')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--resume', help='Complete checkpoint directory or run directory')
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.resume:
        cfg['resume'] = args.resume
    result = train_model(cfg, args.data, args.out, args.dry_run)
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
