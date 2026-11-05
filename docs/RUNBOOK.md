# DS404 secrets dataset and model pipeline: runbook

This is the complete operational guide linked from the [project README](../README.md). **Run every command below from the project root**, containing `Scanner/`, `Dataset/`, `scripts/`, and `secrets_pipeline/`, even though this document lives in `docs/`.

The new `secrets_pipeline/` package extends the existing project with a complete data-to-model workflow. Original `Dataset/` and `Scanner/` files are retained. The existing scanner's provider rules are reused as **candidate proposals**; CodeBERT/UniXcoder learn which candidates are credentials from value and context.

The deliverable is runnable code, configurations, tests and this runbook. The project owner reported completing the experiment suite and the `test` / `test_ood` assessments on the remote server. The remote selected-model weights and scores are not included in this local source checkout; no real LLM generation or GPU training was performed in this local copy. Read measured outcomes from the matching remote artifacts. A strong model must earn that description on the frozen real test set; synthetic validation scores are not evidence of real-world quality.

## 1. Design: extend diversity without inventing ground truth

1. Import local CredData and the current Tier B data with their original annotations and provenance. Keep CredData UUID labels unchanged. Keep `X` and unannotated regions unknown; never silently turn them into negatives. Coarse positive line annotations with multiple candidates are excluded from candidate supervision instead of marking every literal positive; the original line-range annotation remains in evaluation.
2. Freeze repository-owner splits before generation. Existing Tier B splits are retained; newly assigned owners use a deterministic 70/15/15 train/validation/test allocation. Unresolved CredData owners are reported. Jenkins/Pulumi in Tier B remain the separate OOD test.
3. Audit **training data only** to set generation quotas. Coverage estimates use format/category mappings and are conservative; an unmapped category does not prove absence from CredData.
4. A local LLM writes fictional, realistic source/configuration scaffolds with typed slots. It receives a coverage specification, not original credential values or held-out code. Deterministic code inserts the values, computes exact character spans, and generates positive/reference/placeholder counterfactuals in the same split.
5. Check slot integrity, syntax with real parsers, unexpected credential-like literals, exact duplicates and value-masked near duplicates. Reject invalid output. Labels apply only to deliberately inserted spans, not all remaining text. Syntax validity does not prove operational semantics.
6. Use a shared, label-independent candidate extractor for training, evaluation and deployment. It combines the existing scanner with broad literals, assignments, auth/URL and whole-PEM proposals. Candidate coverage is measured explicitly; missed secrets still count as false negatives.
7. Compare real-only and augmented training for CodeBERT and UniXcoder across three seeds. Choose on real validation span F1, with candidate AP, unknown-alert burden and simplicity as tie-breakers. A mean-probability ensemble also competes; it is selected only if validation supports it.
8. Evaluate the frozen winner on real test once, and report Tier B/OOD separately. Use repository bootstrap intervals and review unknown alerts on genuine code.

Coverage targets span 18 families (16 used in default training generation), 19 credential types, low-entropy passwords, custom tokens, encoded and multiline values, provider tokens, neutral/misleading key names, several naming locales, nested/legacy/commented layouts, and hard negatives such as hashes, IDs, references and public keys. **Actual gaps must be measured after importing your full CredData checkout.**

## 2. Clone or extract the source and create environments on the remote PC

Run commands from the project root containing `README.md`, `Scanner/`, `Dataset/`, and `secrets_pipeline/`. The commands target the Linux remote environment; use WSL for the shell workflow on Windows.

Git tracks the source recipes, repository manifests, frozen splits, scanner model and aggregate historical results. Downloaded repositories, scanner executables, generated datasets, virtual environments and experiment artifacts are local outputs. In particular, `Dataset/pipeline_bundle/tier_a/out/` is ignored and is absent from a fresh clone. Restore the existing Tier B release or regenerate it using [Dataset/README.md](../Dataset/README.md), starting from `Dataset/pipeline_bundle/tier_a/` and preserving the committed split assignments. Fetch CredData separately as shown below. A fresh clone alone does not contain either complete input corpus or the remotely trained transformer weights.

```bash
cd ~/OurWork/ISP                 # adjust only this path to your extraction location
bash scripts/setup_remote.sh all
.venv-train/bin/python -m secrets_pipeline doctor
.venv-train/bin/python -m unittest discover -s tests -v
.venv-train/bin/python -m secrets_pipeline smoke --train --out artifacts/smoke-1
```

The server has system Python **3.14**. The setup script obtains a private **Python 3.12** using your existing Conda installation (or installed Python 3.12/uv), then creates actual venvs. It does not replace system Python or install into Conda base. The reference environments are:

| Environment | Purpose | Requirements |
|---|---|---|
| `.venv-train` | import, parsing, training, evaluation, deployment | `requirements/data.txt`, `requirements/train.txt`, PyTorch 2.9.1 CUDA 12.8 wheel |
| `.venv-generator` | local vLLM server | `requirements/generator.txt`; vLLM resolves its own Torch version |
| `.venv-collector` | optional CredData downloader | Created by `scripts/fetch_creddata.sh` |

Separating vLLM and training avoids conflicting Torch requirements. The pinned reference stack uses Python 3.12; this is not a claim that all current ML packages lack Python 3.14 support. Pure-core commands and the no-ML smoke can run on Python 3.14. Setup records installed package locks and environment details in `artifacts/setup/`. First setup/model loading needs internet; scanning saved models and the tiny BERT smoke do not.

`smoke --train` builds a tiny random local BERT, trains for one CPU epoch, verifies model/tokenizer export, score equality after reload, and checkpoint resume. It downloads no model. Passing it validates framework integration, not detection accuracy or multi-GPU performance. Use a fresh smoke output path each time.

If you only need data preparation first:

```bash
bash scripts/setup_remote.sh data
```

## 3. Obtain/import the original data and audit it

If the full CredData checkout already exists on the server, use that location. Otherwise:

```bash
bash scripts/fetch_creddata.sh
```

The default download location is `Scanner/datasets/CredData`. The downloader's commit is recorded. Do not use a small quick subset as the headline benchmark.

Before importing, ensure `Dataset/pipeline_bundle/tier_a/out/tierB` has been restored or regenerated as described in [Dataset/README.md](../Dataset/README.md). Its indexed files and label tables are required by the following command. Preserve the frozen split file when regenerating; the `--skip-invalid` flag is not a substitute for obtaining the dataset.

```bash
.venv-train/bin/python -m secrets_pipeline ingest \
  --creddata Scanner/datasets/CredData \
  --tier-b Dataset/pipeline_bundle/tier_a/out/tierB \
  --out artifacts/import/documents.jsonl \
  --registry artifacts/import/frozen_splits.json \
  --skip-invalid

.venv-train/bin/python -m secrets_pipeline audit \
  --documents artifacts/import/documents.jsonl \
  --seed-catalog creddata_seed_repos.json \
  --out artifacts/import/coverage.json \
  --generation-config artifacts/import/generation-allocation.json
```

`--skip-invalid` is explicit because this local Tier B copy has one indexed file missing. It quarantines bad/missing/conflicting or leaking documents and lists the reasons in the import manifest. Without the flag, the importer refuses to publish an incomplete corpus. Read the manifest before training; do not assume every quarantined file is just the known missing file.

CredData `snapshot.json` resolves repository owners when available. Use `ingest --repo-map mapping.json` for a verified mapping if needed. Inspect unresolved-owner counts. Preserve `frozen_splits.json` with all results. The seed-repository JSON is an unverified expansion catalogue; this pipeline does not claim those repositories were fetched or licensed by reading that catalogue.

## 4. Start the local generator

The supplied server has eight RTX A6000 48 GB GPUs and about 1 TB RAM. A two-GPU starting configuration uses Qwen3-Coder-30B-A3B-Instruct in BF16; actual speed and topology must be checked on the server. Check `nvidia-smi` and use GPUs allocated to your job. The snapshot showed other processes, so no script claims all eight devices or stops other work.

In a dedicated terminal:

```bash
GENERATOR_GPUS=4,5 bash scripts/serve_generator.sh
```

This serves only on `127.0.0.1:8000`. Change GPU IDs to your allocation. Environment options include `GENERATOR_MODEL`, `GENERATOR_REVISION`, `GENERATOR_PORT`, `GENERATOR_CONTEXT`, and `GENERATOR_MEMORY_FRACTION`. If you change the model/port, update the corresponding generation JSON too. Pin the model revision for a long experiment; the local HTTP API may not expose the server's resolved weight revision.

An already-running local Ollama or other compatible server may be used by editing `backend.endpoint` and `backend.model`. The client uses the `/v1/chat/completions` interface; start with a small pilot to verify the server accepts its request parameters.

## 5. Pilot, inspect, then scale generation

In another terminal, run the full data pipeline on a small plan:

```bash
bash scripts/run_dataset.sh \
  --documents artifacts/import/documents.jsonl \
  --generation-config configs/generation.pilot.json \
  --out artifacts/pilot
```

Inspect `artifacts/pilot/generation/report.json`, accepted examples, rejection reasons, `coverage.before.json`, `coverage.after.json`, and `candidate_coverage.json`. Review a stratified sample for realistic semantics and negative-label correctness; parsers cannot certify these. If rejection exceeds the configured rate, the pipeline stops before calling that run prepared. Improve prompts/configuration and use a new output directory.

After the pilot, generate a larger corpus targeted to training coverage deficits:

```bash
bash scripts/run_dataset.sh \
  --documents artifacts/import/documents.jsonl \
  --generation-config configs/generation.full.json \
  --audit-quotas --target-per-cell 100 --max-cells 304 --max-per-cell 50 \
  --out artifacts/full
```

The audit allocation replaces the JSON's default family counts. Without `--audit-quotas`, the full profile requests 512 groups per family. Accepted counts depend on filters; quotas are requests, not guaranteed coverage. Generated data is train-only by default. Original validation/test splits remain frozen, and source templates are used locally for exclusion checking without being sent to the generator.

Outputs include:

| File | Meaning |
|---|---|
| `documents.jsonl` | canonical text, annotations, splits and provenance |
| `candidates.jsonl` | model input derived without consulting labels for proposal discovery |
| `real.candidates.jsonl` | original real-data view |
| `generation/plan.json` | immutable generation settings, exclusions and source hashes |
| `generation/shards/` | resumable per-plan results with integrity hashes |
| `candidate_coverage.json` | recall ceiling and missed ground-truth spans |
| `dataset_status.json` | `prepared`, `building`, or failure status |

Rerun an identical dataset command to reuse completed generation shards. Config/source/input changes require a fresh run directory; do not delete manifests to force reuse. Mock generation (`--mock`) is for tests and is refused by normal training even when real validation is present.

## 6. Train and select the model

First audit the complete 12-run experiment plan without loading any models:

```bash
.venv-train/bin/python -m secrets_pipeline experiments \
  --config configs/experiments.json \
  --data artifacts/full/candidates.jsonl \
  --documents artifacts/full/documents.jsonl \
  --out artifacts/experiments --dry-run
```

Run the suite on one allocated GPU; the modest encoders fit the A6000 comfortably at the supplied starting settings, but adjust batch size if the actual run reports OOM:

```bash
TRAIN_GPUS=6 bash scripts/run_experiments.sh \
  --config configs/experiments.json \
  --data artifacts/full/candidates.jsonl \
  --documents artifacts/full/documents.jsonl \
  --out artifacts/experiments
```

The suite is sequential: CodeBERT/UniXcoder × real-only/augmented × three seeds, plus a validation-tested ensemble. `include_mask_ablation=true` doubles the runs and tests reliance on assignment key names. Source weights reduce synthetic dominance; class weights address imbalance. Training uses BF16, gradient accumulation, checkpointing, gradient clipping, early stopping and validation-only threshold selection. Calibration/inference use FP32 for consistent operating points.

To resume, rerun the same suite command with `--resume`. It checks configuration, input, model and prediction hashes. Full epoch checkpoints save optimizer/scheduler/scaler/RNG state; interrupted epochs restart from the last completed epoch. Keep the whole run directory when moving it.

For a single model across multiple allocated GPUs:

```bash
TRAIN_GPUS=4,5 bash scripts/train_model.sh \
  --config configs/train.codebert.json \
  --data artifacts/full/candidates.jsonl \
  --out artifacts/codebert-ddp
```

Stop your generator first if reusing its GPUs. `train_model.sh` uses `torchrun` for multiple IDs. For a single-run resume append `--resume artifacts/codebert-ddp`. Exact resume requires the same worker count, data and optimization schedule.

Deliverables: `best/` contains real model weights and tokenizer; `thresholds.json` stores operating points; `history.json` stores training progress; `selected.json` identifies the winning model(s); `leaderboard.json` compares validation results. Auto-BLOCK is disabled (`null`) if validation does not support the requested precision lower bound. WARN remains available. This is preferable to inventing a high-confidence threshold.

## 7. Frozen test and baseline comparison

Only after model selection:

```bash
.venv-train/bin/python -m secrets_pipeline assess \
  --bundle artifacts/experiments/selected.json \
  --documents artifacts/full/documents.jsonl \
  --data artifacts/full/candidates.jsonl \
  --split test --out artifacts/test --bootstrap-samples 500

.venv-train/bin/python -m secrets_pipeline assess \
  --bundle artifacts/experiments/selected.json \
  --documents artifacts/full/documents.jsonl \
  --data artifacts/full/candidates.jsonl \
  --split test_ood --out artifacts/test-ood
```

Gitleaks and TruffleHog executables are downloaded locally and are not tracked in Git. Install the pinned evaluation tools on Linux/macOS, or WSL on Windows, with the training environment active so its Python dependencies are installed into that environment:

```bash
source .venv-train/bin/activate
bash Scanner/evaluation/setup_tools.sh

.venv-train/bin/python -m secrets_pipeline baselines \
  --documents artifacts/full/documents.jsonl --split test \
  --tools legacy,gitleaks,trufflehog --out artifacts/baselines-test
```

The setup script places the executables under `Scanner/tools/bin/`. If you restored existing binaries from an archive and executable bits were lost, use `chmod u+x Scanner/tools/bin/gitleaks Scanner/tools/bin/trufflehog`. The baseline CLI also accepts `--gitleaks` and `--trufflehog` paths for an existing installation.

All comparisons use the new span evaluator. Unknown-region alerts are reported as review burden, not certified false positives. Whole PEM blocks count once. Ground-truth positives without candidates count as misses. The old gradient-boosted model was trained using different CredData splits, so its legacy result is a historical baseline with possible training overlap, not a clean new held-out learned-model comparison. External detectors do not verify credentials over the network.

The headline is `warn.headline_real_test`; synthetic/modified/OOD strata stay separate. Test scores are never used by the suite to select configurations. For a genuinely new evaluation corpus, `assess --external-corpus` explicitly keeps model and thresholds frozen. The old PDF results used another protocol and should not be spliced into the new table.

## 8. Use the selected scanner and improve training data

```bash
.venv-train/bin/python -m secrets_pipeline scan /path/to/project \
  --bundle artifacts/experiments/selected.json --out artifacts/project-scan.json
```

Exit codes: **0** = PASS/WARN, **1** = BLOCK, **2** = incomplete/error. Reports redact all values, including short passwords. File read/size errors prevent a false clean result. Scanning uses the same candidate settings, source hashes and calibrated score combination as evaluation. Preserve scanner/rule files with exported models. Runtime mixed-precision overrides are refused for calibrated gating.

For the existing CPU scanner through the same proposal layer:

```bash
.venv-train/bin/python -m secrets_pipeline scan /path/to/project \
  --legacy --out artifacts/legacy-scan.json
```

Hard-example iteration is training-only:

```bash
.venv-train/bin/python -m secrets_pipeline predict \
  --model-dir artifacts/experiments/runs/codebert-augmented-plain-seed13 \
  --data artifacts/full/candidates.jsonl --split train \
  --out artifacts/train-predictions.jsonl
.venv-train/bin/python -m secrets_pipeline mine \
  --data artifacts/full/candidates.jsonl --predictions artifacts/train-predictions.jsonl \
  --threshold 0.5 --out artifacts/training-review.jsonl
```

The example threshold is only for a review queue. Mining never assigns new labels; unknown findings require human review. Use recurring error categories and reviewed real examples to design the next generation recipe. Do not feed held-out errors into the same experiment's generation or prompt tuning.

## 9. Package and provenance

To build the report from completed remote artifacts, install its lightweight dependencies and run the reporting launcher from the project root:

```bash
.venv-train/bin/python -m pip install -r reporting_bundle/requirements.txt
bash reporting_bundle/run_report.sh
```

See [reporting instructions](../reporting_bundle/README.md) for required artifacts and custom paths. Reporting reads saved predictions and metrics; it does not retrain or download a model. The [project explanation](../reporting_bundle/PROJECT_EXPLAINED.md) documents the architecture, and [historical results](../reporting_bundle/HISTORICAL_RESULTS.md) preserve earlier scanner measurements with their protocol limitations.

You can ZIP the entire source directory yourself. To omit machine-specific venvs, caches and experiment outputs while retaining the current input data and scanner assets:

```bash
python scripts/package_release.py --out artifacts/ISP-pipeline.zip
```

The archive opens under `ISP/`. Build environments after extraction; copied Windows/Linux virtual environments are not portable. Preserve licences/provenance for existing rules and derivative source. Model cards and upstream install references used for these profiles:

- [Qwen3-Coder model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [vLLM 0.11.2 GPU installation](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/)
- [PyTorch 2.9.1 wheel commands](https://pytorch.org/get-started/previous-versions/)
- [Tree-sitter language pack 0.10.0 API](https://pypi.org/project/tree-sitter-language-pack/0.10.0/)

The dependency pins are a reference stack, not a claim that they are the newest releases. The complete installed transitive versions are captured after setup. Actual GPU training, vLLM execution, parser-wheel availability and multi-GPU performance must pass the remote smoke/pilot before a large run.
