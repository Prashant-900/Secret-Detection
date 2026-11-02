# Historical results and how to interpret them

This file preserves the earlier scanner benchmarks found in this project. The machine-readable copy is [data/historical_results.json](data/historical_results.json). It travels with the reporting folder, so the report generator can include previous results even if the remote copy does not contain the original benchmark files.

These are **historical reported results**, not outputs from your new CodeBERT/UniXcoder experiment. Your new model's results must come from its saved `selected.json` and the `assess` outputs for `test` and `test_ood`. No remote model score has been filled in here.

## What can be compared

Compare tools only within one of the experiment groups below. Dataset versions, splits, category policy, thresholds, and scoring units differ across groups. In particular:

- The saved all-repository CredData comparison is not a clean held-out test. The shipped v2 model was trained on a subset of those repositories. The saved result does not record which model/code revision produced the label `ours`, so it cannot establish a held-out result for that implementation.
- The separate historical v2 report uses test repositories and removes UUID-only positive rows. Its F1 scores cannot be directly subtracted from those of the all-repository run.
- The old large synthetic results count individual labeled lines. Multiline private keys occupy many lines. The newer pipeline evaluates labeled spans, counting one key as one value; this changes the evaluation unit.
- The old `cfn`/`k8s` synthetic test has only 27 true lines, and the report says a later v2 revision followed inspection of its errors. It is directional evidence, not an untouched final test.
- `ours` means the scanner implementation saved under that historical label. It is not evidence for a regex-only baseline: the shipped scanner already includes a learned gradient-boosted reranker. The new transformer models are separate.

All precision, recall, F1 and AUC values below are fractions from 0 to 1. `not recorded` means unavailable, never zero. Historical timings belong to their original hardware and run settings.

## Legacy metric definitions

The old evaluator deduplicates predictions to unique `(file, line)` hits. Its labeled precision is `TP_hits / (TP_hits + FP_labeled_hits)`. Its recall is `detected_positive_label_rows / positive_label_rows`. One hit can intersect more than one label row; one positive row can also span multiple lines. **The precision TP count and recall detected-row count therefore need not match.** F1 is the harmonic mean of that precision and recall.

Strict precision includes `FP_unlabeled_hits` in the denominator. For CredData these unlabeled lines are unknown, so this is a stricter assumption rather than verified ground truth. The saved CredData run restricts strict scoring to files that contain labels. The saved synthetic run includes all files. Both saved JSON files use zero line tolerance and treat `X` labels as negative.

Do not turn these mixed-unit counts into an ordinary 2x2 confusion matrix, specificity, accuracy, or a true-negative count. The portable JSON intentionally sets normalized `tp/fp/fn/tn` to null and preserves exact original counts under `raw_metrics`; `missed_positive_rows` is `T_rows - T_detected`.

Sources: [legacy evaluator](../Scanner/evaluation/evaluate_all.py), [evaluation guide](../Scanner/evaluation/README_TESTING.md), and the later [span evaluator](../Dataset/pipeline_bundle/tier_a/pipeline/evaluate.py). Source links are project-relative; the JSON also records SHA-256 hashes of the source snapshot.

## 1. Saved original-scanner comparison on all CredData repositories

**Dataset:** CredData  
**Split:** all repositories (not a held-out result)  
**Scoring:** Legacy labeled-line precision and positive-label-row recall; strict variant adds unlabeled-line false positives

Source: [Scanner/results/creddata_results.json](../Scanner/results/creddata_results.json).

| Tool / model | Operating point | Threshold | Precision | Recall | F1 | Strict precision | Strict F1 | Seconds |
|---|---|---|---|---|---|---|---|---|
| ours [BLOCK>=0.7] | BLOCK | 0.7000 | 0.8852 | 0.6860 | 0.7730 | 0.8462 | 0.7577 | 106.1193 |
| ours [WARN+BLOCK>=0.4] | WARN + BLOCK | 0.4000 | 0.8292 | 0.7623 | 0.7943 | 0.7631 | 0.7627 | 106.1193 |
| gitleaks | Tool default | not recorded | 0.8613 | 0.4577 | 0.5978 | 0.8363 | 0.5916 | 68.3063 |
| trufflehog | Tool default | not recorded | 0.5191 | 0.0132 | 0.0258 | 0.2070 | 0.0248 | 35.6451 |
| detect-secrets | Tool default | not recorded | 0.5280 | 0.2579 | 0.3466 | 0.2858 | 0.2711 | 285.0208 |

- Saved result has repo_split=all. The shipped v2 model was trained on the dev subset of these repositories, so all-repository numbers are not a clean test of generalization.
- The result labels the implementation only as ours and records no scanner code hash or model hash. Preserve that label; do not assume it is v1, a regex-only scanner, or the newly trained deep-learning model.
- Uses BLOCK >= 0.70 and WARN + BLOCK >= 0.40, which differ from shipped v2 defaults of 0.45 and 0.24.
- Tolerance is zero; X labels are negative; strict false positives are counted only inside labeled files. UUID-positive categories are present in this result.
- Recorded times are historical execution times on unspecified hardware, not a controlled latency comparison with the remote deep-learning run.
- Category lists contain the top 15 categories only. Threshold sweeps are descriptive; do not use this all-repository sweep to retune a final test threshold.

### Saved counts (keep hit counts separate from label rows)

| Tool / point | Hits | TP hits | FP labeled hits | FP unlabeled hits | Positive label rows | Detected positive rows | Missed positive rows |
|---|---|---|---|---|---|---|---|
| ours [BLOCK>=0.7] | 12887 | 10905 | 1414 | 568 | 15753 | 10807 | 4946 |
| ours [WARN+BLOCK>=0.4] | 16005 | 12213 | 2516 | 1276 | 15753 | 12008 | 3745 |
| gitleaks | 8356 | 6988 | 1125 | 243 | 15753 | 7210 | 8543 |
| trufflehog | 855 | 177 | 164 | 514 | 15753 | 208 | 15545 |
| detect-secrets | 15409 | 4404 | 3937 | 7068 | 15753 | 4063 | 11690 |

### Saved recall by secret category

Only the categories saved by the original evaluator are available; its category output was capped at 15. Counts are category label support.

| Category | Support | ours [BLOCK>=0.7] | ours [WARN+BLOCK>=0.4] | gitleaks | trufflehog | detect-secrets |
|---|---|---|---|---|---|---|
| Key | 3822 | 0.7040 | 0.9200 | 0.8810 | 0.0010 | 0.0330 |
| Password | 2573 | 0.8360 | 0.8950 | 0.0580 | 0.0030 | 0.6600 |
| UUID | 2316 | 0.0030 | 0.0060 | 0.0040 | 0.0000 | 0.0160 |
| Secret | 1263 | 0.9590 | 0.9780 | 0.8350 | 0.0130 | 0.2250 |
| PEM Private Key | 1161 | 1.0000 | 1.0000 | 1.0000 | 0.0120 | 0.2740 |
| Auth:Basic Authorization | 601 | 0.9950 | 0.9950 | 0.0000 | 0.0000 | 0.0170 |
| Token | 583 | 0.8730 | 0.9140 | 0.5180 | 0.0310 | 0.3340 |
| URL Credentials | 213 | 0.9200 | 0.9720 | 0.0000 | 0.2020 | 0.5590 |
| Auth | 174 | 0.5920 | 0.6840 | 0.2070 | 0.0060 | 0.1440 |
| Token:UUID | 169 | 0.9940 | 0.9940 | 0.9230 | 0.0060 | 0.0000 |
| AWS Client ID | 162 | 0.3700 | 0.3700 | 0.6980 | 0.0490 | 0.6980 |
| Nonce | 148 | 0.8850 | 0.9730 | 0.0000 | 0.0000 | 0.4930 |
| Key:Secret | 124 | 0.9520 | 0.9600 | 0.6290 | 0.0000 | 0.6450 |
| API:Token | 112 | 0.0540 | 0.0980 | 0.0450 | 0.0000 | 0.0540 |
| NTLM Token:Bearer Authorization:Auth | 105 | 0.9900 | 0.9900 | 0.0000 | 0.0000 | 0.0000 |

### Saved flag rate by benign category

Only the categories saved by the original evaluator are available; its category output was capped at 15. Counts are category label support.

| Category | Support | ours [BLOCK>=0.7] | ours [WARN+BLOCK>=0.4] | gitleaks | trufflehog | detect-secrets |
|---|---|---|---|---|---|---|
| Key | 19929 | 0.0630 | 0.1090 | 0.0490 | 0.0010 | 0.0140 |
| Password | 10941 | 0.0060 | 0.0180 | 0.0020 | 0.0020 | 0.2240 |
| Token | 4594 | 0.0000 | 0.0020 | 0.0050 | 0.0000 | 0.0030 |
| UUID | 3652 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| API | 3598 | 0.0010 | 0.0060 | 0.0040 | 0.0000 | 0.0200 |
| Auth | 3263 | 0.0020 | 0.0030 | 0.0050 | 0.0010 | 0.0140 |
| Secret | 2175 | 0.0010 | 0.0040 | 0.0050 | 0.0000 | 0.2320 |
| Credential | 547 | 0.0040 | 0.0050 | 0.0070 | 0.0000 | 0.0040 |
| Basic Authorization | 538 | 0.0000 | 0.0000 | 0.0000 | 0.0020 | 0.0060 |
| URL Credentials | 288 | 0.0070 | 0.0420 | 0.0000 | 0.2920 | 0.4440 |
| Auth:Token | 268 | 0.0000 | 0.0040 | 0.0000 | 0.0000 | 0.0000 |
| Key:Token | 263 | 0.0040 | 0.0110 | 0.0000 | 0.0000 | 0.0340 |
| API:Key | 201 | 0.0000 | 0.0100 | 0.0100 | 0.0000 | 0.4880 |
| Key:Secret | 145 | 0.0070 | 0.0140 | 0.0140 | 0.0000 | 0.5100 |
| Password:URL Credentials | 117 | 0.0000 | 0.0170 | 0.0000 | 0.2560 | 0.4440 |

### Largest recorded false-positive sources

Counts below combine labeled and unlabeled false-positive hits as recorded by the legacy evaluator. The original output lists only leading sources, not a complete distribution.

| Tool / point | Source counts |
|---|---|
| ours [BLOCK>=0.7] | rule:generic-api-key: 1031; entropy: 698; rule:pem-private-key-header: 82; assign: 67; rule:vault-service-token: 29; rule:generic-password-assignment: 25; rule:curl-auth-user: 16; call: 11; rule:gcp-service-account-json: 7; rule:auth-basic: 4 |
| ours [WARN+BLOCK>=0.4] | entropy: 1894; rule:generic-api-key: 1413; assign: 233; rule:pem-private-key-header: 82; rule:generic-password-assignment: 51; rule:vault-service-token: 29; call: 22; rule:curl-auth-user: 16; rule:connection-string-password: 12; rule:helm-secret-value: 8 |
| gitleaks | rule:generic-api-key: 1266; rule:private-key: 59; rule:kubernetes-secret-yaml: 9; rule:curl-auth-user: 8; rule:vault-service-token: 8; rule:curl-auth-header: 7; rule:jwt: 6; rule:twitch-api-token: 2; rule:hashicorp-tf-password: 2; rule:linkedin-client-id: 1 |
| trufflehog | detector: 678 |
| detect-secrets | plugin: 11005 |

### Historical threshold sweeps

These describe the already evaluated historical dataset. They must not be used to choose a new final-test operating threshold. Your new model uses thresholds frozen on validation data.

**ours**

| Threshold | Labeled precision | Recall | Labeled F1 | Strict precision | Strict F1 |
|---|---|---|---|---|---|
| 0.0500 | 0.6980 | 0.8033 | 0.7469 | 0.3736 | 0.5100 |
| 0.1000 | 0.7396 | 0.7975 | 0.7674 | 0.4636 | 0.5864 |
| 0.1500 | 0.7596 | 0.7907 | 0.7748 | 0.5482 | 0.6475 |
| 0.2000 | 0.7700 | 0.7861 | 0.7780 | 0.6149 | 0.6901 |
| 0.2500 | 0.7870 | 0.7778 | 0.7824 | 0.6686 | 0.7191 |
| 0.3000 | 0.7974 | 0.7689 | 0.7829 | 0.7017 | 0.7338 |
| 0.3500 | 0.8165 | 0.7646 | 0.7897 | 0.7395 | 0.7518 |
| 0.4000 | 0.8292 | 0.7623 | 0.7943 | 0.7631 | 0.7627 |
| 0.4500 | 0.8430 | 0.7592 | 0.7989 | 0.7815 | 0.7702 |
| 0.5000 | 0.8595 | 0.7553 | 0.8040 | 0.8018 | 0.7779 |
| 0.5500 | 0.8674 | 0.7524 | 0.8058 | 0.8134 | 0.7817 |
| 0.6000 | 0.8745 | 0.7472 | 0.8058 | 0.8247 | 0.7840 |
| 0.6500 | 0.8862 | 0.7298 | 0.8004 | 0.8399 | 0.7810 |
| 0.7000 | 0.8852 | 0.6860 | 0.7730 | 0.8462 | 0.7577 |
| 0.7500 | 0.9288 | 0.6300 | 0.7508 | 0.8909 | 0.7381 |
| 0.8000 | 0.9383 | 0.5783 | 0.7156 | 0.9067 | 0.7062 |
| 0.8500 | 0.9524 | 0.5219 | 0.6743 | 0.9280 | 0.6681 |
| 0.9000 | 0.9658 | 0.4859 | 0.6465 | 0.9496 | 0.6428 |
| 0.9500 | 0.9778 | 0.4378 | 0.6048 | 0.9645 | 0.6022 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 2. Historical v1/v2 comparison on held-out CredData repositories

**Dataset:** CredData  
**Split:** 119 test repositories, 26,489 labeled rows as stated in report; crc32(repo) % 5 in {0,1}  
**Scoring:** Legacy labeled protocol; UUID-only positive rows dropped; strict F1 separately reported

Source: [Scanner/V2_REPORT.md](../Scanner/V2_REPORT.md).

| Tool / model | Operating point | Threshold | Precision | Recall | F1 | Strict precision | Strict F1 | Seconds |
|---|---|---|---|---|---|---|---|---|
| CredSweeper 1.19.2 | Tool default | not recorded | 0.9940 | 0.9930 | 0.9930 | not recorded | 0.9690 | not recorded |
| Gitleaks 8.24.3 | Tool default | not recorded | 0.8430 | 0.7750 | 0.8080 | not recorded | 0.8050 | not recorded |
| Ours v2 | BLOCK | 0.4500 | 0.7480 | 0.8830 | 0.8100 | not recorded | 0.7710 | not recorded |
| Ours v2 | WARN + BLOCK | 0.2400 | 0.6730 | 0.9210 | 0.7780 | not recorded | 0.6810 | not recorded |
| Ours v1 | WARN + BLOCK | 0.4000 | 0.4740 | 0.7810 | 0.5900 | not recorded | 0.4760 | not recorded |
| Ours v1 | BLOCK | 0.7000 | 0.5370 | 0.2250 | 0.3170 | not recorded | 0.2790 | not recorded |
| Ours v1 + capture-group fix | WARN + BLOCK | 0.4000 | 0.4090 | 0.4270 | 0.4180 | not recorded | 0.3510 | not recorded |
| detect-secrets 1.5.0 | Tool default | not recorded | 0.3870 | 0.1550 | 0.2220 | not recorded | 0.1570 | not recorded |
| TruffleHog 3.88.0 | No verification | not recorded | 0.4410 | 0.0090 | 0.0180 | not recorded | 0.0170 | not recorded |

- Manually transcribed from the committed historical report; raw held-out outputs and counts are not present in this local snapshot. Values are rounded to three decimals in that report.
- Full source dataset is reported as 337 repositories and 67,896 labeled lines. The test subset is described as 119 repositories / 26,489 labeled rows; no post-UUID-exclusion row count is supplied.
- The report states fitting and threshold selection used dev repositories only. UUID-only positive rows are dropped at evaluation and treated as negatives in v2 training.
- At BLOCK, v2 has labeled F1 0.810 vs Gitleaks 0.808, with higher recall but lower precision. Strict F1 is 0.771 vs 0.805; do not claim consistent superiority.
- The report cautions that CredSweeper helped define CredData candidate labels and its benchmark mixes train/test data. Its near-0.99 result is not established as an independent out-of-sample competitor result.
- CredData obfuscates secret values, and v2 assumes unlabeled dev candidates are benign. Both affect interpretation of generalization.
- Single seed/run; no confidence intervals. shhgit, wraith, credential-digger and truffleHog3 were not run in this local comparison.
- The historical report mentions v2 iteration-one BLOCK F1 0.819, versus 0.810 for the shipped second iteration; other first-iteration metrics are not supplied.

## 3. Historical small IaC family test (directional only)

**Dataset:** Earlier synthetic IaC benchmark  
**Split:** cfn and k8s families; 27 true lines  
**Scoring:** Legacy synthetic line-based protocol as reported

Source: [Scanner/V2_REPORT.md](../Scanner/V2_REPORT.md).

| Tool / model | Operating point | Threshold | Precision | Recall | F1 | Strict precision | Strict F1 | Seconds |
|---|---|---|---|---|---|---|---|---|
| Ours v1 | WARN + BLOCK | 0.4000 | 0.9580 | 0.8520 | 0.9020 | not recorded | not recorded | not recorded |
| Ours v1 + capture-group fix | WARN + BLOCK | 0.4000 | 1.0000 | 0.8520 | 0.9200 | not recorded | not recorded | not recorded |
| Ours v2 | BLOCK | 0.4500 | 0.8620 | 0.9260 | 0.8930 | not recorded | not recorded | not recorded |
| Ours v2 | WARN + BLOCK | 0.2400 | 0.7300 | 1.0000 | 0.8440 | not recorded | not recorded | not recorded |
| CredSweeper | Tool default | not recorded | 0.9130 | 0.7780 | 0.8400 | not recorded | not recorded | not recorded |
| Gitleaks | Tool default | not recorded | 1.0000 | 0.6670 | 0.8000 | not recorded | not recorded | not recorded |
| detect-secrets | Tool default | not recorded | 0.5810 | 0.9260 | 0.7140 | not recorded | not recorded | not recorded |
| TruffleHog | Tool default | not recorded | 1.0000 | 0.3700 | 0.5400 | not recorded | not recorded | not recorded |

- Transcribed from V2_REPORT.md; rounded metrics only, no raw counts or exact timings in this table.
- Only 27 true lines. v2 was revised after examining this synthetic test, so it is no longer a clean held-out test.
- This is a different dataset from the larger Tier-B saved synthetic results below; neither is the new remote test_ood assessment.

## 4. Saved larger synthetic IaC line-level comparison

**Dataset:** Tier-B generated IaC snapshot  
**Split:** all files represented by saved legacy result (no held-out subset recorded)  
**Scoring:** Legacy labeled-line precision and positive-label-row recall; strict variant adds unlabeled-line false positives

Source: [Dataset/pipeline_bundle/tier_a/out/eval_graph/synthetic_results.json](../Dataset/pipeline_bundle/tier_a/out/eval_graph/synthetic_results.json).

| Tool / model | Operating point | Threshold | Precision | Recall | F1 | Strict precision | Strict F1 | Seconds |
|---|---|---|---|---|---|---|---|---|
| ours [BLOCK>=0.7] | BLOCK | 0.7000 | 0.9987 | 0.3600 | 0.5292 | 0.9742 | 0.5257 | 7.3643 |
| ours [WARN+BLOCK>=0.4] | WARN + BLOCK | 0.4000 | 0.9974 | 0.3789 | 0.5492 | 0.9548 | 0.5425 | 7.3643 |
| ours+fix [BLOCK>=0.7] | BLOCK | 0.7000 | 0.9987 | 0.3600 | 0.5292 | 0.9742 | 0.5257 | 7.2921 |
| ours+fix [WARN+BLOCK>=0.4] | WARN + BLOCK | 0.4000 | 0.9974 | 0.3789 | 0.5492 | 0.9548 | 0.5425 | 7.2921 |
| gitleaks | Tool default | not recorded | 0.9951 | 0.2641 | 0.4175 | 0.9692 | 0.4151 | 3.5330 |
| trufflehog | Tool default | not recorded | 0.9986 | 0.1138 | 0.2043 | 0.9986 | 0.2043 | 3.1876 |

- T_rows=6186; 3980 are private-key PEM line labels. This is a line-level evaluation: detecting one line of a multiline PEM key does not detect all its other labeled lines, which depresses reported recall.
- The later Dataset pipeline evaluator groups each injected value by span_id, counting a multiline key as one span. Its span-level metrics are not interchangeable with these saved line-level metrics.
- Tolerance is zero; X labels are negative; strict false positives include unlabeled lines in all scanned files.
- Saved names ours and ours+fix are preserved; scanner/model version hashes are absent. The saved ours+fix metrics are identical to ours at each threshold, but timings differ.
- The stored legacy result does not include detect-secrets. Missing tool rows must not be filled with zeros.
- Historical dataset counts can differ from the current files/labels snapshot; do not infer current remote dataset size from this result.

### Saved counts (keep hit counts separate from label rows)

| Tool / point | Hits | TP hits | FP labeled hits | FP unlabeled hits | Positive label rows | Detected positive rows | Missed positive rows |
|---|---|---|---|---|---|---|---|
| ours [BLOCK>=0.7] | 2286 | 2227 | 3 | 56 | 6186 | 2227 | 3959 |
| ours [WARN+BLOCK>=0.4] | 2455 | 2344 | 6 | 105 | 6186 | 2344 | 3842 |
| ours+fix [BLOCK>=0.7] | 2286 | 2227 | 3 | 56 | 6186 | 2227 | 3959 |
| ours+fix [WARN+BLOCK>=0.4] | 2455 | 2344 | 6 | 105 | 6186 | 2344 | 3842 |
| gitleaks | 1686 | 1634 | 8 | 44 | 6186 | 1634 | 4552 |
| trufflehog | 705 | 704 | 1 | 0 | 6186 | 704 | 5482 |

### Saved recall by secret category

Only the categories saved by the original evaluator are available; its category output was capped at 15. Counts are category label support.

| Category | Support | ours [BLOCK>=0.7] | ours [WARN+BLOCK>=0.4] | ours+fix [BLOCK>=0.7] | ours+fix [WARN+BLOCK>=0.4] | gitleaks | trufflehog |
|---|---|---|---|---|---|---|---|
| secret_known_format:private_key_pem | 3980 | 0.0670 | 0.0820 | 0.0670 | 0.0820 | 0.0540 | 0.0000 |
| unknown_format_secret:connection_string | 371 | 0.9950 | 1.0000 | 0.9950 | 1.0000 | 0.0000 | 0.5440 |
| unknown_format_secret:custom_token | 181 | 0.8010 | 0.8730 | 0.8010 | 0.8730 | 0.7680 | 0.0000 |
| unknown_format_secret:generic_token_urlsafe | 171 | 0.8770 | 0.9360 | 0.8770 | 0.9360 | 0.8070 | 0.0000 |
| unknown_format_secret:db_password_random | 151 | 0.7880 | 0.8340 | 0.7880 | 0.8340 | 0.2520 | 0.0000 |
| unknown_format_secret:generic_api_key_hex | 150 | 0.7600 | 0.8130 | 0.7600 | 0.8130 | 0.7600 | 0.0330 |
| secret_known_format:github_pat | 150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9930 |
| secret_known_format:aws_secret_access_key | 144 | 0.8890 | 0.9310 | 0.8890 | 0.9310 | 0.8190 | 0.0000 |
| low_entropy_secret:db_password_human | 141 | 0.6380 | 0.6880 | 0.6380 | 0.6880 | 0.2060 | 0.0000 |
| unknown_format_secret:webhook_secret | 134 | 0.6490 | 0.6490 | 0.6490 | 0.6490 | 0.6190 | 0.0000 |
| secret_known_format:azure_client_secret | 111 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9820 |
| secret_known_format:slack_bot_token | 106 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9910 |
| secret_known_format:aws_access_key_id | 101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0400 |
| secret_known_format:jwt | 100 | 0.9700 | 1.0000 | 0.9700 | 1.0000 | 0.9900 | 0.3100 |
| secret_known_format:stripe_secret | 99 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 1.0000 |

### Saved flag rate by benign category

Only the categories saved by the original evaluator are available; its category output was capped at 15. Counts are category label support.

| Category | Support | ours [BLOCK>=0.7] | ours [WARN+BLOCK>=0.4] | ours+fix [BLOCK>=0.7] | ours+fix [WARN+BLOCK>=0.4] | gitleaks | trufflehog |
|---|---|---|---|---|---|---|---|
| placeholder | 664 | 0.0050 | 0.0080 | 0.0050 | 0.0080 | 0.0120 | 0.0000 |
| reference | 638 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| trap:digest | 79 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| trap:sha1 | 79 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0130 |
| trap:sha256 | 73 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| trap:md5 | 69 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| trap:account | 67 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| trap:ami | 63 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| trap:subnet | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| trap:uuid | 57 | 0.0000 | 0.0180 | 0.0000 | 0.0180 | 0.0000 | 0.0000 |

### Largest recorded false-positive sources

Counts below combine labeled and unlabeled false-positive hits as recorded by the legacy evaluator. The original output lists only leading sources, not a complete distribution.

| Tool / point | Source counts |
|---|---|
| ours [BLOCK>=0.7] | assign: 14; rule:generic-password-assignment: 11; rule:generic-api-key: 11; rule:helm-secret-value: 9; entropy: 7; rule:hashicorp-tf-password: 3; rule:generic-secret-assignment: 3; rule:connection-string-password: 1 |
| ours [WARN+BLOCK>=0.4] | entropy: 34; rule:generic-api-key: 26; assign: 19; rule:generic-password-assignment: 12; rule:helm-secret-value: 12; rule:hashicorp-tf-password: 4; rule:generic-secret-assignment: 3; rule:connection-string-password: 1 |
| ours+fix [BLOCK>=0.7] | assign: 14; rule:generic-password-assignment: 11; rule:generic-api-key: 11; rule:helm-secret-value: 9; entropy: 7; rule:hashicorp-tf-password: 3; rule:generic-secret-assignment: 3; rule:connection-string-password: 1 |
| ours+fix [WARN+BLOCK>=0.4] | entropy: 34; rule:generic-api-key: 26; assign: 19; rule:generic-password-assignment: 12; rule:helm-secret-value: 12; rule:hashicorp-tf-password: 4; rule:generic-secret-assignment: 3; rule:connection-string-password: 1 |
| gitleaks | rule:kubernetes-secret-yaml: 27; rule:hashicorp-tf-password: 13; rule:generic-api-key: 12 |
| trufflehog | detector: 1 |

### Historical threshold sweeps

These describe the already evaluated historical dataset. They must not be used to choose a new final-test operating threshold. Your new model uses thresholds frozen on validation data.

**ours**

| Threshold | Labeled precision | Recall | Labeled F1 | Strict precision | Strict F1 |
|---|---|---|---|---|---|
| 0.0500 | 0.9593 | 0.3964 | 0.5610 | 0.7199 | 0.5113 |
| 0.1000 | 0.9842 | 0.3931 | 0.5619 | 0.8269 | 0.5329 |
| 0.1500 | 0.9922 | 0.3901 | 0.5600 | 0.8810 | 0.5407 |
| 0.2000 | 0.9958 | 0.3875 | 0.5579 | 0.9135 | 0.5442 |
| 0.2500 | 0.9971 | 0.3857 | 0.5562 | 0.9357 | 0.5462 |
| 0.3000 | 0.9975 | 0.3834 | 0.5539 | 0.9462 | 0.5457 |
| 0.3500 | 0.9975 | 0.3815 | 0.5519 | 0.9543 | 0.5451 |
| 0.4000 | 0.9974 | 0.3789 | 0.5492 | 0.9548 | 0.5425 |
| 0.4500 | 0.9974 | 0.3765 | 0.5466 | 0.9545 | 0.5400 |
| 0.5000 | 0.9978 | 0.3731 | 0.5431 | 0.9573 | 0.5369 |
| 0.5500 | 0.9983 | 0.3702 | 0.5401 | 0.9614 | 0.5345 |
| 0.6000 | 0.9987 | 0.3663 | 0.5360 | 0.9713 | 0.5320 |
| 0.6500 | 0.9987 | 0.3639 | 0.5334 | 0.9728 | 0.5296 |
| 0.7000 | 0.9987 | 0.3600 | 0.5292 | 0.9742 | 0.5257 |
| 0.7500 | 0.9986 | 0.3552 | 0.5240 | 0.9804 | 0.5214 |
| 0.8000 | 0.9986 | 0.3485 | 0.5167 | 0.9818 | 0.5144 |
| 0.8500 | 0.9991 | 0.3413 | 0.5087 | 0.9828 | 0.5066 |
| 0.9000 | 0.9995 | 0.3290 | 0.4950 | 0.9845 | 0.4932 |
| 0.9500 | 1.0000 | 0.3091 | 0.4722 | 0.9846 | 0.4705 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**ours+fix**

| Threshold | Labeled precision | Recall | Labeled F1 | Strict precision | Strict F1 |
|---|---|---|---|---|---|
| 0.0500 | 0.9593 | 0.3964 | 0.5610 | 0.7199 | 0.5113 |
| 0.1000 | 0.9842 | 0.3931 | 0.5619 | 0.8269 | 0.5329 |
| 0.1500 | 0.9922 | 0.3901 | 0.5600 | 0.8810 | 0.5407 |
| 0.2000 | 0.9958 | 0.3875 | 0.5579 | 0.9135 | 0.5442 |
| 0.2500 | 0.9971 | 0.3857 | 0.5562 | 0.9357 | 0.5462 |
| 0.3000 | 0.9975 | 0.3834 | 0.5539 | 0.9462 | 0.5457 |
| 0.3500 | 0.9975 | 0.3815 | 0.5519 | 0.9543 | 0.5451 |
| 0.4000 | 0.9974 | 0.3789 | 0.5492 | 0.9548 | 0.5425 |
| 0.4500 | 0.9974 | 0.3765 | 0.5466 | 0.9545 | 0.5400 |
| 0.5000 | 0.9978 | 0.3731 | 0.5431 | 0.9573 | 0.5369 |
| 0.5500 | 0.9983 | 0.3702 | 0.5401 | 0.9614 | 0.5345 |
| 0.6000 | 0.9987 | 0.3663 | 0.5360 | 0.9713 | 0.5320 |
| 0.6500 | 0.9987 | 0.3639 | 0.5334 | 0.9728 | 0.5296 |
| 0.7000 | 0.9987 | 0.3600 | 0.5292 | 0.9742 | 0.5257 |
| 0.7500 | 0.9986 | 0.3552 | 0.5240 | 0.9804 | 0.5214 |
| 0.8000 | 0.9986 | 0.3485 | 0.5167 | 0.9818 | 0.5144 |
| 0.8500 | 0.9991 | 0.3413 | 0.5087 | 0.9828 | 0.5066 |
| 0.9000 | 0.9995 | 0.3290 | 0.4950 | 0.9845 | 0.4932 |
| 0.9500 | 1.0000 | 0.3091 | 0.4722 | 0.9846 | 0.4705 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 5. Shipped gradient-boosted v2 model: development out-of-fold results

**Dataset:** CredData dev + synthetic train families  
**Split:** dev (crc32(repo)%5 not in {0,1})  
**Scoring:** Development group out-of-fold threshold-selection metrics; not test metrics

Source: [Scanner/scanner/model_v2.json](../Scanner/scanner/model_v2.json).

| Tool / model | Operating point | Threshold | Precision | Recall | F1 | Strict precision | Strict F1 | Seconds |
|---|---|---|---|---|---|---|---|---|
| Ours v2 | WARN | 0.2400 | 0.8544 | 0.7965 | 0.8244 | not recorded | not recorded | not recorded |
| Ours v2 | BLOCK | 0.4500 | 0.9007 | 0.7511 | 0.8191 | not recorded | not recorded | not recorded |

- The shipped file contains 197 features and 300 boosted trees; it is not CodeBERT or UniXcoder.
- These out-of-fold estimates were used to choose WARN (maximum F1) and BLOCK (lowest threshold meeting 0.90 development precision). They are not performance guarantees for the test set.
- Five GroupKFold splits are used by training/train_v2.py. Training metadata reports 131095 candidates, 8530 positives, synthetic weight 5.0, and unlabeled negatives enabled.
- The metadata reports candidate recall 0.8975 and AUC 0.9765. Both belong to development evaluation, not the remote DL test.

### Other saved development statistics

| Statistic | Value |
|---|---|
| Candidate recall | 0.8975 |
| Out-of-fold AUC | 0.9765 |
| Engineered features | 197 |
| Boosted trees | 300 |
| Training candidates | 131095 |
| Positive candidates | 8530 |
| Synthetic sample weight | 5.0 |
| Use unlabeled dev candidates as negatives | True |
| Treat UUID-only positives as negatives during training | True |

## 6. Historical external CredData table quoted by the v2 report

**Dataset:** CredData README table (April 2022, per V2_REPORT.md)  
**Split:** all data; external historical reference  
**Scoring:** Different historical scoring; only F1 quoted locally

Source: [Scanner/V2_REPORT.md](../Scanner/V2_REPORT.md).

| Tool / model | Operating point | Threshold | Precision | Recall | F1 | Strict precision | Strict F1 | Seconds |
|---|---|---|---|---|---|---|---|---|
| CredSweeper | Historical external table | not recorded | not recorded | not recorded | 0.8590 | not recorded | not recorded | not recorded |
| Gitleaks | Historical external table | not recorded | not recorded | not recorded | 0.3340 | not recorded | not recorded | not recorded |
| detect-secrets | Historical external table | not recorded | not recorded | not recorded | 0.2060 | not recorded | not recorded | not recorded |
| truffleHog3 | Historical external table | not recorded | not recorded | not recorded | 0.2350 | not recorded | not recorded | not recorded |
| truffleHog | Historical external table | not recorded | not recorded | not recorded | 0.0180 | not recorded | not recorded | not recorded |

- Included to preserve the previous report, not as a newly verified benchmark. These F1 values were quoted in V2_REPORT.md from an April 2022 CredData README.
- Different scoring, scanner versions and dataset protocol. Do not rank these together with the local or remote experiments.

## What the earlier results support

In the historical held-out CredData report, v2 raises WARN-tier labeled F1 from v1's 0.590 to 0.778; at its recommended BLOCK point, v2 reaches 0.810. At that BLOCK point it roughly matches Gitleaks' labeled F1 of 0.808 by trading higher recall for lower precision, while Gitleaks has higher strict F1 (0.805 versus 0.771). There are no confidence intervals, so the 0.002 labeled-F1 difference is not evidence of a statistically reliable win.

The all-repository table shows the scanner finding more labeled positives than the saved baseline runs, but its split policy cannot support an unseen-repository superiority claim. The large synthetic table shows that line-based scoring heavily affects multiline keys; it should motivate inspecting span-level metrics, not be cited as a directly comparable measure of the new model.

To measure how much the new DL model improves over the old scanner, run both through the same current evaluation on the same frozen `test`/`test_ood` documents, with the same label policy, and report paired repository-level uncertainty if available. The bundled historic tables remain useful background and an audit trail; they do not replace that matched comparison.

## Provenance and availability

This snapshot was assembled on 2026-10-31 from the files named above. Original saved JSON metrics retain their four-decimal precision; values from `V2_REPORT.md` retain its three-decimal precision even when displayed with an extra trailing zero. Counts, category breakdowns and sweeps are copied only where an original JSON contains them. No missing raw held-out counts, runtime values, confidence intervals, model version hashes or remote DL results were reconstructed.

The quoted April 2022 external table is explicitly secondary evidence from the local historical report. Its source was not independently retrieved for this bundle. The 73 passing tests reported in the earlier project chat checked software behavior; they were not a model-accuracy result and are not included as a benchmark score.
