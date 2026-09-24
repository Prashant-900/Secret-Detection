# Testing & evaluation guide

Everything below runs from the **repository root** (the folder that contains `scanner/`,
`dataset/` and `evaluation/`). Copy this `evaluation/` folder into your repo root; it
adds files next to your existing `evaluation/evaluate.py` and does not change it.

## 1. Links

| What | Link |
|---|---|
| **CredData** (labeled test data, Samsung) | https://github.com/Samsung/CredData |
| Gitleaks | https://github.com/gitleaks/gitleaks  (we pin [v8.24.3](https://github.com/gitleaks/gitleaks/releases/tag/v8.24.3)) |
| TruffleHog v3 | https://github.com/trufflesecurity/trufflehog  (we pin [v3.88.0](https://github.com/trufflesecurity/trufflehog/releases/tag/v3.88.0)) |
| detect-secrets | https://github.com/Yelp/detect-secrets  ·  https://pypi.org/project/detect-secrets/  (we pin 1.5.0) |
| git-secrets | https://github.com/awslabs/git-secrets  (pre-commit hook, needs a git repo; **not** in the automated comparison) |
| CredSweeper (Samsung's own scanner, good extra baseline) | https://github.com/Samsung/CredSweeper |
| SecretBench (another secrets benchmark; check its README for how to obtain the data) | https://github.com/setu1421/SecretBench |

The pinned versions are the ones your README table was produced with.

CredData notes: repo licence is Apache-2.0 and every file keeps its original project's
licence, so **do not redistribute the downloaded data**. The credential values are
obfuscated by the dataset authors (random strings of the same length/charset). Labels:
`T` = real credential, `F` = not a credential, `X` = unknown (test values, placeholders).

## 2. One command

```bash
bash evaluation/run_all.sh quick     # ~2-3 min: tools + synthetic + 15-repo CredData sample
bash evaluation/run_all.sh full      # everything, all 337 CredData repos (long; see below)
```

Requirements: Linux or macOS (Windows: use WSL), `git`, `curl`, internet, and Python
**3.10, 3.11 or 3.12** (3.14+ also works). **Not 3.13** - CredData's own downloader crashes on it
(`ModuleNotFoundError: annotationlib`). `setup_tools.sh` picks a suitable Python itself,
or set `PYTHON=/path/to/python3.12`.

Outputs go to `results/`: `<dataset>_results.md`, `<dataset>_results.json`, `<dataset>_chart.png`.

## 3. Step by step (what run_all.sh does)

```bash
# 1) install gitleaks + trufflehog binaries into tools/bin and a .venv with detect-secrets
bash evaluation/setup_tools.sh

# 2) get CredData  (it is not stored in git; download_data.py clones the original repos)
bash evaluation/get_creddata.sh quick 15      # random 15 repos  -> datasets/CredData_quick
bash evaluation/get_creddata.sh full          # all 337 repos    -> datasets/CredData
#    JOBS=16 bash evaluation/get_creddata.sh full     # more parallel clones

# 3) evaluate
.venv/bin/python evaluation/evaluate_all.py --dataset synthetic
.venv/bin/python evaluation/evaluate_all.py --dataset creddata --creddata-dir datasets/CredData
.venv/bin/python evaluation/evaluate_all.py --help
```

Useful options:

| Option | Meaning |
|---|---|
| `--tools ours,gitleaks` | run only some tools (`ours,gitleaks,trufflehog,detect-secrets`) |
| `--tolerance 2` | a flag within +-2 lines of a true row counts as detecting it |
| `--x-policy ignore` | do not count flags on CredData `X` (unknown) rows as false positives |
| `--block-at 0.7 --warn-at 0.4` | verdict thresholds of our scanner |
| `--no-fixed` | skip the `ours+fix` variant |
| `--reuse-raw` | re-score cached tool output in `results/raw/` without re-running tools |

The `full` CredData download clones 337 repositories at pinned commits. It needs several GB
of disk and a long time (`--jobs` helps). The `quick` sample is a smoke test: repositories
differ hugely in size and style, so **quick-sample numbers do not predict full-dataset numbers**
(in our test, Gitleaks recall on the 15-repo sample was very different from the full-data
figure you measured). Report the `full` run in the paper.

## 4. How scoring works (so you can defend it)

All tools are scored by the same code, on unique `(file, line)` pairs:

* **labeled protocol** (primary): only lines that have a ground-truth label are judged.
  Flag on a `T` line = TP; flag on a labeled `F`/`X` line = FP; flag on any other line is
  ignored. CredData only labels candidate lines, so an unlabeled line is *unknown*, not negative.
* **strict protocol**: additionally counts flags on unlabeled lines (inside labeled files) as FP.
  Harsh to broad scanners; it is how the synthetic data was originally scored (there, every
  unlabeled line really is benign).
* **recall**: detected `T` rows / all `T` rows (same in both protocols).
* **ours** is reported at two operating points (BLOCK only, WARN+BLOCK) plus a threshold sweep
  (the precision-recall curve in the chart). `ours+fix` is an *evaluation-only* variant that
  patches the scanner in memory so rules with exactly one capture group use that group as the
  secret instead of the whole match (see the progress report). It does not edit `scanner/`.
* `results/*_results.md` also lists recall per secret category, the false-positive rate per
  non-secret category, and the top rules causing false positives for our scanner.
* The scanner is run on **every** file regardless of extension (`scan.py`'s own directory walk
  only visits IaC extensions, which would silently skip CredData's `.py/.js/.java/...` files).
* TruffleHog runs with `--no-verification` (no network checks; verification is meaningless for
  fake or obfuscated secrets).

## 5. Troubleshooting

* `gitleaks: no report written` - run `tools/bin/gitleaks version`; on macOS you may need to
  allow the binary (System Settings -> Privacy & Security).
* `download_data.py` fails on a repo - a GitHub repo in the snapshot may have disappeared; re-run
  (it resumes), or use `quick`.
* `creddata skipped: data not found` - run step 2 first.

## 6. Added later: scanner v2, splits, extra options
* `scanner_v2/` = improved scanner (see `V2_REPORT.md`), `training/train_v2.py` retrains its model.
  Evaluate it with `--extra-scanner v2=scanner_v2` (it uses its own thresholds from `model_v2.json`).
* `--repo-split dev|test` scores a repo-level 60/40 split of CredData (train/tune on dev only, report on test).
* `--synthetic-split train|test` does the same for the synthetic families.
* `--drop-categories UUID` drops True rows of a CredData category (UUIDs are identifiers, not IaC secrets).
* CredSweeper is now one of the tools (`pip install credsweeper`, installed by `setup_tools.sh`'s venv step if added to it).
* detect-secrets is run with `--no-verify`: by default it sends found secrets to AWS/Telegram/... to verify them.
  Your original `evaluation/evaluate.py` does not pass that flag.
* Raw tool outputs are cached per dataset directory and split in `results/raw/`; use `--reuse-raw` to re-score.
