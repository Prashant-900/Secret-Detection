# Generate the complete model and scanner report

Copy this entire `reporting_bundle/` folder into the **root of the remote ISP project**, beside `secrets_pipeline/`, `Scanner/`, `.venv-train/`, and `artifacts/`. Then run these two commands from that project root:

```bash
.venv-train/bin/python -m pip install -r reporting_bundle/requirements.txt
bash reporting_bundle/run_report.sh
```

The script reads the experiment and assessment artifacts you already produced. It needs **no GPU, retraining, model download, or fresh inference**. Your experiment-suite command and the two completed `assess` commands are sufficient when their complete output directories are present. You do not need to rerun them to generate this report.

It prints the output location when finished. The default is a new directory named:

```text
reporting_bundle/output/report-<UTC timestamp>/
```

Open its `report.html` in a browser. Plot PNGs are embedded directly in the HTML, so the report opens offline. Copy the whole output directory to your laptop to keep the Markdown walkthroughs, CSV tables, and SVG figures together with the HTML.

## What is included

The generated report combines the selected deep-learning model's saved results with the previous scanner benchmarks. It contains:

- The actual validation winner, its selected member model(s), ensemble weights, frozen WARN/BLOCK thresholds, and saved architecture/training settings.
- Real-test headline precision, recall, F1, TP/FP/FN, candidate-extraction ceiling, negative flag rate, and unknown-alert burden; all-test and OOD results remain separately labeled.
- Per-source, language, and credential-category breakdowns and identifier-only missed-span tables.
- Candidate confusion matrices, PR and ROC curves, AP/ROC-AUC, score distributions, threshold sensitivity, reliability diagrams, Brier score, log loss, ECE, accuracy, balanced accuracy, MCC, specificity, NPV, FPR and FNR where defined.
- Saved repository-bootstrap intervals, their support, and an explicit explanation when an interval is unavailable.
- The complete validation leaderboard, each available run's training-loss and validation curves, sample variation across seeds, and matched real-only versus augmented validation comparisons.
- Dataset composition and available generation acceptance, rejection, and before/after coverage statistics.
- Current same-corpus scanner baselines if their optional artifacts exist.
- **All historical tables and explanations from [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md), embedded in the generated report's `README.md` itself**, with separate historical plots and CSV/JSON data.
- The complete project walkthrough in [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md): original scanner construction, dataset/label design, local-LLM generation, model architecture and loss, training, selection, evaluation, and CI use. A copy accompanies every generated report.

The number of plots depends on the available artifacts and defined metrics. Large category plots may show the 25 largest strata; CSV tables retain the full available rows. Missing results are reported as unavailable, never replaced with zeroes.

## Expected remote directory

```text
ISP/
├── .venv-train/bin/python
├── secrets_pipeline/
├── Scanner/
├── artifacts/
│   ├── full/
│   │   ├── documents.jsonl
│   │   └── candidates.jsonl
│   ├── experiments/
│   │   ├── selected.json
│   │   ├── leaderboard.json
│   │   ├── suite_manifest.json
│   │   └── runs/<run-name>/
│   │       ├── run.json
│   │       ├── history.json
│   │       ├── real_validation_all_predictions.jsonl
│   │       └── best/config.json
│   ├── test/
│   │   ├── metrics.json
│   │   └── predictions.jsonl
│   └── test-ood/
│       ├── metrics.json
│       └── predictions.jsonl
└── reporting_bundle/
    ├── run_report.sh
    ├── build_report.py
    ├── report_stats.py
    ├── plotting.py
    ├── requirements.txt
    ├── README.md
    ├── PROJECT_EXPLAINED.md
    ├── HISTORICAL_RESULTS.md
    ├── data/historical_results.json
    └── tests/
```

Keep the whole `experiments/` directory: the report needs every leaderboard run's `run.json` and the selected member(s)' `real_validation_all_predictions.jsonl`. `history.json` can fall back to the history embedded in `run.json`. Exported `best/config.json` supplies architecture details; absent optional metadata is listed as unavailable. Reporting does not load model-weight files.

Optional inputs include `full/generation/report.json`, `full/coverage.before.json`, `full/coverage.after.json`, and `artifacts/baselines-test/` or `artifacts/baselines-test-ood/`. Their absence does not require retraining and does not block an otherwise complete core report.

The report calls the project's existing span evaluator on saved predictions. Keep the original `secrets_pipeline/` package in the remote project; copying only model outputs into an unrelated directory is not the supported layout.

## The commands you already completed

These commands created the required experiment/test outputs. They are recorded here for provenance; **running the report does not run them again**.

```bash
TRAIN_GPUS=6 bash scripts/run_experiments.sh \
  --config configs/experiments.json \
  --data artifacts/full/candidates.jsonl \
  --documents artifacts/full/documents.jsonl \
  --out artifacts/experiments

.venv-train/bin/python -m secrets_pipeline assess \
  --bundle artifacts/experiments/selected.json \
  --documents artifacts/full/documents.jsonl \
  --data artifacts/full/candidates.jsonl \
  --split test \
  --out artifacts/test \
  --bootstrap-samples 500

.venv-train/bin/python -m secrets_pipeline assess \
  --bundle artifacts/experiments/selected.json \
  --documents artifacts/full/documents.jsonl \
  --data artifacts/full/candidates.jsonl \
  --split test_ood \
  --out artifacts/test-ood \
  --bootstrap-samples 500
```

`TRAIN_GPUS=6` means GPU device ID 6, not six GPUs. The report itself uses CPU processing and plotting.

## Output files

| File or directory | Purpose |
|---|---|
| `README.md` | Complete generated report with current results, full historical tables, figures, limitations and missing-artifact inventory |
| `report.html` | Offline browser report with inline PNG figures |
| `report.json` | Structured metrics, selection, training, dataset, history and baseline summaries |
| `input_manifest.json` | Consumed input paths, SHA-256 fingerprints, reporting dependency versions and unavailable inputs |
| `tables/*.csv` | Full numeric tables, curves, seed comparisons, category metrics and identifier-only misses |
| `plots/*.png` | Raster figures for slides and documents |
| `plots/*.svg` | Vector figures for publication/editing |
| `PROJECT_EXPLAINED.md` | Beginning-to-end explanation of the implementation |
| `HISTORICAL_RESULTS.md` | Standalone historical benchmark appendix |

A fresh output directory avoids mixing plots from different runs. The script refuses a nonempty `--out` directory and refuses output locations that would contain existing source/input artifacts.

## Paths and optional settings

`run_report.sh` resolves its own folder and the parent project root to absolute paths. It therefore works when invoked by its full path from another working directory. Relative argument paths are interpreted relative to the **project root**, not your shell's current directory.

Choose a named, new output directory:

```bash
bash reporting_bundle/run_report.sh --out artifacts/final-report-01
```

Point to a different completed experiment/dataset assessment layout:

```bash
bash reporting_bundle/run_report.sh \
  --experiments artifacts/experiments-v2 \
  --dataset artifacts/full-v2 \
  --test artifacts/test-v2 \
  --test-ood artifacts/test-ood-v2 \
  --baselines artifacts/baselines-test-v2 artifacts/baselines-test-ood-v2 \
  --out artifacts/final-report-v2
```

The inputs must belong to the same frozen experiment. Absolute paths are also accepted. A different directory name does not relax fingerprint or threshold checks.

Use another compatible Python interpreter when needed:

```bash
REPORT_PYTHON=/absolute/path/to/python bash reporting_bundle/run_report.sh
```

Install the reporting requirements into that interpreter first. The code requires Python 3.10+; the project's existing `.venv-train` normally uses Python 3.12. The lightweight requirements are `matplotlib>=3.8,<4` and `Markdown>=3.6,<4`; the report does not require additional Torch/model packages beyond reading the existing project artifacts.

For direct Python execution or an explicitly different project root:

```bash
.venv-train/bin/python reporting_bundle/build_report.py \
  --project-root /absolute/path/to/ISP
```

| Option | Default |
|---|---|
| `--project-root` | Parent of `reporting_bundle/` |
| `--experiments` | `artifacts/experiments` |
| `--dataset` | `artifacts/full` |
| `--test` | `artifacts/test` |
| `--test-ood` | `artifacts/test-ood` |
| `--baselines [DIR ...]` | `artifacts/baselines-test artifacts/baselines-test-ood` |
| `--out` | Fresh `reporting_bundle/output/report-<UTC timestamp>` |
| `--allow-partial` | Off |

To skip searching for optional current baselines, supply `--baselines` with no directory arguments. The bundled historical results are still included.

```bash
bash reporting_bundle/run_report.sh --baselines
```

## Optional: run current scanner baselines

Your three completed commands evaluated the selected deep-learning model. They did not create new Gitleaks, TruffleHog, or legacy-scanner results under the current span protocol. Historical scanner scores are already included in this folder; no extra command is needed to include those.

For new baseline measurements on the same frozen test and OOD documents, run the following from the project root. These tools run on CPU. The default paths use the bundled Linux Gitleaks/TruffleHog executables, so restore executable permission if it was lost during Windows ZIP transfer:

```bash
chmod u+x Scanner/tools/bin/gitleaks Scanner/tools/bin/trufflehog

.venv-train/bin/python -m secrets_pipeline baselines \
  --documents artifacts/full/documents.jsonl \
  --split test \
  --tools legacy,gitleaks,trufflehog \
  --out artifacts/baselines-test

.venv-train/bin/python -m secrets_pipeline baselines \
  --documents artifacts/full/documents.jsonl \
  --split test_ood \
  --tools legacy,gitleaks,trufflehog \
  --out artifacts/baselines-test-ood

bash reporting_bundle/run_report.sh
```

If the executables are stored elsewhere, the baseline CLI accepts `--gitleaks /path/to/gitleaks` and `--trufflehog /path/to/trufflehog`. Its compatible scanner interfaces are Gitleaks `dir` with JSON reporting and TruffleHog `filesystem --json --no-verification --no-update`. The legacy baseline uses the existing shipped CPU ranker.

The new report discovers `*.evaluation.json` in the baseline directories. It does not execute the detectors itself. The legacy ranker may have trained on some repositories in the new test split, so the report preserves that overlap caveat. Detector localization also differs; current shared scoring alone does not eliminate every comparison limitation.

## Previous headline results included

These are selected historical values for orientation. The full tables, counts, category breakdowns, thresholds, timings and limitations are in [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md) and are inserted into the generated report.

| Historical experiment / operating point | Scanner | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| All-repository CredData, WARN+BLOCK ≥ 0.40 | `ours` | 0.829 | 0.762 | 0.794 |
| Same all-repository experiment | Gitleaks | 0.861 | 0.458 | 0.598 |
| Historical held-out v2 CredData, BLOCK ≥ 0.45 | Ours v2 | 0.748 | 0.883 | 0.810 |
| Same historical held-out experiment | Gitleaks | 0.843 | 0.775 | 0.808 |
| Small historical synthetic family test, BLOCK ≥ 0.45 | Ours v2 | 0.862 | 0.926 | 0.893 |
| Same small synthetic family test | Gitleaks | 1.000 | 0.667 | 0.800 |

The all-repository CredData result is not clean held-out evidence and does not identify its scanner/model revision. The v2 held-out protocol excluded UUID-only positives; under its strict protocol v2 F1 was 0.771 versus Gitleaks 0.805. The small synthetic test had only 27 true lines and was inspected during scanner development. These historical line-based scores must not be directly subtracted from the new model's span scores as a controlled improvement claim. `ours` is not a proven regex-only baseline; the shipped legacy scanner contains a learned gradient-boosted ranker.

## Metric scope and unavailable statistics

**End-to-end span results** include annotated positives that the candidate extractor missed. Matching is one-to-one character-span overlap. An excess alert on a previously matched positive counts as a duplicate false positive. An alert on explicitly benign content counts as an explicit-negative false positive. An alert in unknown content is reported separately as review burden.

**Candidate classification results** use only explicitly labeled extracted candidates. Their confusion matrices, accuracy, AP, ROC-AUC and calibration plots exclude unknowns and secrets with no proposal. They must not be described as whole-scanner accuracy. Undefined ratios and absent-class AUCs are N/A. The real-test headline excludes modified/synthetic sources; OOD stays separate.

`assess --bootstrap-samples 500` creates real-repository WARN intervals only when the assessed split contains real-source documents. A Tier B-only OOD split has no `real_repository_bootstrap`, even with that flag. The report explains the absence rather than constructing an unrelated interval. Seed standard deviations describe validation-run variation and are not test confidence intervals.

Training histories contain rank-0 weighted training loss and candidate validation metrics, plus saved epoch timers. They do not normally contain train-set accuracy, validation loss, inference latency/throughput, or peak GPU memory. The report does not invent these statistics. Summed epoch timers exclude some checkpoint/export work and are not full-job wall time. Disabled BLOCK thresholds remain disabled; descriptive test threshold curves do not retune them.

The reporter does not export raw source code or credential values. Per-span tables contain identifiers/categories and proposal-availability information. It reads saved inputs, calculates aggregate diagnostics, and preserves artifact provenance.

## Validation, partial reports and troubleshooting

By default the reporter fails if required inputs are absent. It also checks frozen data fingerprints, suite/selection agreement, assessment split flags, thresholds, candidate identities/labels, and recomputed span results. Existing `assess` outputs do not record the model/bundle hash that produced them; these consistency checks do not cryptographically prove the prediction model's identity. That limitation is recorded in the report.

To inspect genuinely incomplete artifacts, opt in explicitly:

```bash
bash reporting_bundle/run_report.sh --allow-partial
```

Such an output is marked **INCOMPLETE REPORT**, lists missing required artifacts, and cannot support a complete model-performance claim. `--allow-partial` permits missing files; it does **not** bypass corrupt JSON, inconsistent metrics, mismatched corpora, or changed thresholds.

| Problem | Resolution |
|---|---|
| `Python not found` | Run from the correct project copy or set `REPORT_PYTHON` to the existing compatible interpreter. |
| Missing `matplotlib` or `markdown` | Install `reporting_bundle/requirements.txt` with the same interpreter that runs the report. |
| Required artifact missing | Check the listed path. Copy the complete experiment/assessment directory, or pass the matching path option. |
| Missing selected validation predictions | Restore `runs/<selected-member>/real_validation_all_predictions.jsonl` from the completed suite. Copying only `selected.json` and `best/` is insufficient. |
| Frozen hash/corpus/threshold mismatch | Use the exact documents, candidates, selection and assessments from the same experiment. The reporter will not combine different runs. |
| Saved metrics disagree with recomputation | Restore consistent original artifacts and the matching pipeline version; do not edit scores to force the report through. |
| Output directory is nonempty | Omit `--out` to get a fresh timestamped directory, or choose a new empty directory. |
| No current-baseline plots | Baseline outputs are optional. Run the two CPU baseline commands above if this comparison is needed. |
| No OOD confidence interval | Check whether the OOD split contains real-source repositories; `assess` does not bootstrap Tier B-only OOD. |
| No architecture or generation detail | Restore optional exported configs or generation reports. The reporting code cannot reconstruct missing run metadata. |
| Remote shell reports `bash\r` or `$'\r'` | Preserve LF line endings when transferring `run_report.sh`, or invoke `build_report.py` directly with Python. |

Check the reporting tests after installing the lightweight requirements:

```bash
.venv-train/bin/python -m unittest discover -s reporting_bundle/tests -v
```

For all flags:

```bash
bash reporting_bundle/run_report.sh --help
```
