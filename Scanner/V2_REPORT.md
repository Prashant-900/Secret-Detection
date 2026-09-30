# Scanner comparison on CredData and the v2 scanner — findings

## 1. Question
Is our scanner better than the scanners benchmarked by Samsung's CredData
(https://github.com/Samsung/CredData/tree/main/benchmark/scanner: CredSweeper, credential-digger,
detect-secrets, gitleaks, shhgit, truffleHog, truffleHog3, wraith), and can it be improved?

## 2. Method (so the numbers can be trusted)
* Full CredData (337 repos, 67,896 labeled lines) downloaded with its own `download_data.py`.
* **Repo-level split**: 60 % dev / 40 % test (`crc32(repo) % 5 in {0,1}` -> test). All fitting and
  threshold choice used dev repos only. **Reported numbers are on the 119 test repos** (26,489 labeled rows).
* **Same scoring code for every tool** (`evaluation/evaluate_all.py`): unique (file, line) hits;
  "labeled" protocol = judge only labeled lines (T = TP, F/X = FP, unlabeled ignored); a stricter
  protocol also counts flags on unlabeled lines. `UUID`-only True rows (2,316 of them, ~25 % of all
  positives) are dropped: CredData calls a bare UUID a credential; for IaC a UUID is an identifier
  (Azure tenant IDs, resource IDs) and every scanner here, ours included, deliberately ignores them.
* Tools: our v1 scanner as in your zip, v1 with an in-memory capture-group fix, new v2, Gitleaks 8.24.3,
  TruffleHog 3.88.0 (no verification), detect-secrets 1.5.0, CredSweeper 1.19.2.
  The benchmark's other scanners (shhgit, wraith, credential-digger, truffleHog3) were not run.

## 3. Results

### CredData, held-out test repos (labeled protocol; strict F1 in brackets)
| Tool | Precision | Recall | F1 |
|---|---|---|---|
| **CredSweeper** | 0.994 | 0.993 | 0.993 (0.969)  *see caveat 1* |
| Gitleaks | 0.843 | 0.775 | 0.808 (0.805) |
| **Ours v2 @ BLOCK (0.45)** | 0.748 | 0.883 | 0.810 (0.771) |
| Ours v2 @ WARN (0.24) | 0.673 | 0.921 | 0.778 (0.681) |
| Ours v1 @ WARN+BLOCK (0.40) | 0.474 | 0.781 | 0.590 (0.476) |
| Ours v1 @ BLOCK (0.70) | 0.537 | 0.225 | 0.317 (0.279) |
| Ours v1 + capture-group fix @ WARN | 0.409 | 0.427 | 0.418 (0.351) |
| detect-secrets | 0.387 | 0.155 | 0.222 (0.157) |
| TruffleHog | 0.441 | 0.009 | 0.018 (0.017) |

For reference, CredData's own README table (April 2022, all data, different scoring) lists
CredSweeper F1 0.859, gitleaks 0.334, detect-secrets 0.206, truffleHog3 0.235, truffleHog 0.018.

### Synthetic IaC benchmark, held-out families (cfn, k8s; only 27 true lines — directional only)
| Tool | Precision | Recall | F1 |
|---|---|---|---|
| Ours v1 @ WARN+BLOCK | 0.958 | 0.852 | 0.902 |
| Ours v1 + fix @ WARN | 1.000 | 0.852 | 0.920 |
| Ours v2 @ BLOCK (0.45) | 0.862 | 0.926 | 0.893 |
| Ours v2 @ WARN (0.24) | 0.730 | 1.000 | 0.844 |
| CredSweeper | 0.913 | 0.778 | 0.840 |
| Gitleaks | 1.000 | 0.667 | 0.800 |
| detect-secrets | 0.581 | 0.926 | 0.714 |
| TruffleHog | 1.000 | 0.370 | 0.540 |

## 4. Answers
**Is v1 better than the benchmark scanners?** Better than detect-secrets and TruffleHog (and, by the
CredData README's old numbers, than truffleHog3/shhgit/wraith/credential-digger), but **clearly worse than
Gitleaks on CredData** (F1 0.59 vs 0.81) and nowhere near CredSweeper. On our IaC-style synthetic data
it is competitive or best, which is the setting it was designed for.

**Could it be improved? Yes, and v2 does so on CredData:** F1 0.59 -> 0.81, i.e. it **matches Gitleaks**
(0.810 vs 0.808) with higher recall (0.88 vs 0.78) but lower precision (0.75 vs 0.84), and it is slightly
worse than Gitleaks in the strict protocol (0.771 vs 0.805). So v2 is *on par with*, not better than,
Gitleaks. On the synthetic IaC families v2 is about equal to v1 (0.89 vs 0.90) and above Gitleaks/CredSweeper.

## 5. Why v1 lost (diagnosis on dev repos, v1 at WARN: precision 0.42, recall 0.49)
* ~40 % of true credentials never became candidates: PEM key headers (rules are line-based), `Authorization:
  Basic/Bearer` headers, `scheme://user:pass@host` credentials, PASETO / NATS NKey tokens, **short passwords**
  (gitleaks-derived rules need values >= 10 chars) and camelCase keys such as `domainPassword = 'vatbjza0'`
  (the keyword rule's look-behind blocks them).
* Most false positives came from `generic-api-key` firing on `"api_version": "2008-08-01"`, `"key": "flex_chat_channels"`,
  `config.script.db.password`, `YOUR_API_KEY`, type annotations (`token: string`), XML attributes.
* Capture-group bug: for ~150 rules the whole match (key name included) was used as the secret. Fixing it
  alone cuts false positives but also recall (table: "+ capture-group fix"): the bug was accidentally
  helping recall. v2 fixes it *and* replaces the hand-set context penalties with a learned score.

## 6. What v2 is (`scanner_v2/`, same CLI and exit codes as v1)
Stage 1 candidates (v1 rules with correct capture groups + PEM/auth-header/URL-credential/PASETO/NKey rules +
keyword-assignment extractor, camelCase aware, any value length >= 3 + entropy dragnet) ->
Stage 2 filter (placeholders, reference values) -> Stage 2b re-ranker: gradient-boosted trees over 197 features
(value shape, key words, line/previous-line context, file type), exported to `model_v2.json` and evaluated in
pure Python (no ML dependency at scan time). Provider-format rules keep a score floor. Thresholds come from
out-of-fold dev predictions: BLOCK = lowest threshold with precision >= 0.90 (0.45), WARN = max-F1 (0.24).
Retrain: `python3 training/train_v2.py --creddata-dir datasets/CredData --synthetic-dir dataset/generated`.
v2 was built in two iterations; the second added benign-key words (fingerprint, digest, ...) and previous-line
context (k8s `- name: X` / `value: Y`) after the first one mis-handled one synthetic trap type. On CredData the
second iteration was not better (F1 0.810 vs 0.819 at BLOCK); we ship the second because it generalises better on IaC.
**Recommended gate: use the BLOCK threshold (0.45); the WARN tier (0.24) trades a lot of precision for recall.**

## 7. Caveats (please read before quoting numbers)
1. **CredSweeper's ~0.99 is not an out-of-sample result.** CredData's labeled lines are CredSweeper's own rule
   candidates (README: labels follow "CredSweeper rules"), and its benchmark mixes train and test data. Treat it as a
   reference upper bound, not a competitor we can fairly beat.
2. **Obfuscation artifacts.** CredData replaces credential values by random strings of the same style; non-credential
   values stay original. Our out-of-fold AUC (0.98) is therefore probably optimistic: the model can partly learn
   "looks randomly generated". Real-world precision/recall on genuine repositories is unknown and likely lower.
3. **Unlabeled-line assumption.** To keep false alarms low on ordinary code the model also saw candidates from unlabeled
   lines of dev files as negatives (assumes lines CredSweeper did not select are benign).
4. **UUID policy** (section 2) changes the benchmark; with UUID rows included recall of every tool falls.
5. The synthetic test set is tiny (27 true lines) and v2's 2nd iteration was designed after inspecting its errors,
   so it is no longer a clean held-out test. The CredData test repos were never inspected or used for any choice.
6. Single run, no confidence intervals; v2 training used a fixed seed.
7. v2 is slower than v1 (~87-107 s vs ~52 s on the same 40 % of data) but still scans ~1 GB in a few minutes.

## 8. Other things found along the way
* **detect-secrets verifies secrets over the network by default** (`scan` calls AWS STS, Telegram, IBM IAM, ... for
  tokens it finds). The first version of my evaluator did not pass `--no-verify`; the sandbox blocked all of those
  requests, but anyone running it elsewhere would have sent obfuscated CredData / synthetic secrets to those services.
  **Fixed in the kit** (`--no-verify`). Your original `evaluation/evaluate.py` has the same issue (line ~88) — add `--no-verify`.
* CredData's downloader crashes on Python 3.13 (`annotationlib`); use 3.10-3.12.
* My evaluator's first cache layout mixed quick and full CredData runs; fixed (cache is keyed by dataset directory and split).
* Your README's results table was produced by an older scanner version (see earlier report).

## 9. Reproduce
```bash
bash evaluation/setup_tools.sh && bash evaluation/get_creddata.sh full
.venv/bin/python training/train_v2.py --creddata-dir datasets/CredData --synthetic-dir dataset/generated   # optional, model is shipped
.venv/bin/python evaluation/evaluate_all.py --dataset creddata --creddata-dir datasets/CredData \
    --repo-split test --drop-categories UUID --extra-scanner v2=scanner_v2 --jobs 2
.venv/bin/python evaluation/evaluate_all.py --dataset synthetic --synthetic-split test --extra-scanner v2=scanner_v2
```
(Full CredData evaluation including CredSweeper and detect-secrets takes roughly 20-30 minutes on 2 cores.)
