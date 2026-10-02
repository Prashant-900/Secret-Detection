# IaC secret-detection dataset: Tier A / B / C pipeline

```
tier_a/
  manifest.csv         100 permissive repos (pinned SHA + licence)         <- original list
  manifest_v2.csv      the same + 4 extra Jenkins/Pulumi repos (104)       <- use this one
  fetch_repos.sh       shallow-clone every repo at its pinned commit
  splits.csv           FROZEN repo-level split (made by make_split.py)
  pipeline/
    secrets_lib.py     typed secret slots, key-name compatibility, placeholders, references, traps
    make_split.py      owner-grouped, family-balanced split; jenkins+pulumi -> test_ood
    audit_preexisting.py  finds secret-looking strings ALREADY in the real repos (human review)
    mutate_real.py     Tier B: 10 operators + validation gate -> files/ + labels.csv
    verify_labels.py   independent consistency gate (parse, spans, pairs)
    evaluate.py        span-level scoring of ANY detector (gitleaks, trufflehog, model predictions)
    gpu/
      llm_backend.py       vLLM backend (+ mock for CPU dry runs)
      gen_skeletons.py     LLM writes skeletons with @@SLOT:type@@ markers; code fills them (Tier A' / C)
      semantic_edits.py    LLM edits labeled files; labels re-derived by string search
      llm_baseline.py      zero-shot LLM baseline, P(YES) per candidate line
      train_codebert.py    fine-tune CodeBERT (or UniXcoder); --mask-key ablation
```

Install: `pip install pyyaml python-hcl2 scikit-learn numpy`  (GPU steps also: `torch transformers vllm`).
Run the scripts normally (`python script.py`), not with `python -I`: they import their siblings.

## Run order: no GPU needed

```bash
cd tier_a
./fetch_repos.sh manifest_v2.csv repos                      # ~104 repos, a few GB

cd pipeline
python make_split.py --manifest ../manifest_v2.csv --out ../splits.csv     # COMMIT splits.csv, never regenerate
python audit_preexisting.py --repos ../repos --manifest ../manifest_v2.csv --out ../preexisting_hits.csv
#   -> two people review preexisting_hits.csv (fill human_verdict). Files with hits are skipped below.
python mutate_real.py --repos ../repos --manifest ../manifest_v2.csv --splits ../splits.csv \
       --out ../out/tierB --exclude-hits ../preexisting_hits.csv --max-files-per-repo 20 \
       --cap-override jenkins=40,pulumi=40
python verify_labels.py --data ../out/tierB                 # must print "0 problems"

# baselines: install the pinned scanners once, then put them on PATH
# from /OurWork/Scanner: bash evaluation/setup_tools.sh
# when running from tier_a/: export PATH="$(pwd)/../../../Scanner/tools/bin:$PATH"
# when running from pipeline/: export PATH="$(pwd)/../../../../Scanner/tools/bin:$PATH"
export PATH="$(pwd)/../../../../Scanner/tools/bin:$PATH"
gitleaks dir ../out/tierB/files --report-format json --report-path ../out/gl.json --exit-code 0 --no-banner
trufflehog filesystem ../out/tierB/files --json --no-verification > ../out/th.jsonl
python evaluate.py --data ../out/tierB --gitleaks gitleaks=../out/gl.json --trufflehog trufflehog=../out/th.jsonl
```

False positives on REAL files (the number reviewers care about): run the same two scanners over `../repos`
(unmutated), then human-review every hit; report confirmed false positives per KLOC.

## Run order: GPU (vLLM; 100 GB VRAM: 32B bf16 on one big GPU, or `--tp N`; 70B with `--quantization fp8`)

```bash
cd tier_a/pipeline/gpu
python gen_skeletons.py --out ../../out/gen --n-per-family 300 --model Qwen/Qwen2.5-Coder-32B-Instruct
#   watch gen_stats.json: if reject_rate > 0.4, fix the prompt / add few-shot examples before scaling.
python semantic_edits.py --data ../../out/tierB --out ../../out/tierB_edits --n-files 600 --model Qwen/Qwen2.5-Coder-32B-Instruct
python train_codebert.py --data ../../out/tierB,../../out/gen,../../out/tierB_edits --out ../../out/runs/codebert
python train_codebert.py --data ../../out/tierB,../../out/gen,../../out/tierB_edits --out ../../out/runs/codebert_masked --mask-key
# judge = a DIFFERENT model family than the generator, so it is not grading its own family's text
python llm_baseline.py --data ../../out/tierB,../../out/gen --splits val,test,test_ood --out ../../out/preds_llm.csv --model <other-family-model>

cd .. && python evaluate.py --data ../out/tierB,../out/gen --out ../out/results.json \
   --gitleaks gitleaks=../out/gl.json --trufflehog trufflehog=../out/th.jsonl \
   --pred codebert=../out/runs/codebert/predictions.csv --pred codebert_masked=../out/runs/codebert_masked/predictions.csv \
   --pred llm=../out/preds_llm.csv
```

Dry-run any GPU script on CPU first with `--backend mock` (train_codebert: `--dry-run`).

## Split policy (what to train / tune / report on)

| Split | Repos (manifest_v2) | Use |
|---|---|---|
| train | 65 | train CodeBERT; the only split LLM/few-shot examples may be drawn from |
| val | 10 | choose the decision threshold, early stopping, prompt tuning. Nothing else |
| test | 22 | in-distribution held-out repos: the headline number, touched once |
| test_ood | 7 | Jenkins (4) + Pulumi (3): families never seen in any training data. Report separately, never pooled |

* The split is by **repo owner**, so `terraform-aws-modules/*` or `geerlingguy/*` can never straddle train and test.
* Edited files (`semantic_edits.py`) inherit their parent's split. Generated skeletons are split by (family, topic):
  all samples of one topic share a split. Generated test files are a secondary, synthetic test set, never the headline.
* Jenkins and Pulumi are not in `gen_skeletons.py`'s families. If you add them there, put them in test_ood only.
* Never tune on test or test_ood. Freeze `splits.csv` (record its sha256) before the first training run.
* Report at span level, with the per-category breakdown (known-format / unknown-format / low-entropy), the key-name
  ablation (`recall_by_key_hint`, `--mask-key`) and FPR on placeholder / reference / trap look-alikes.

## Known limitations (put these in the datasheet)

* Labels exist only for strings WE injected. Unlabeled lines are assumed benign; `audit_preexisting.py` reduces but does not remove that risk.
* HCL validation needs `python-hcl2`; Groovy / TypeScript files are not parse-validated (`validated=0` in files_index.csv).
* Insert sites are weighted toward natural parents (env, data, locals ...) but placement is still generic, not semantic.
* test_ood is small (7 repos): report confidence intervals, treat it as a stress test, not a benchmark.
* Manifest diversity: Terraform is dominated by 3 owners and Ansible by `geerlingguy`; Ansible has no val repo. Add owners if you can.
* The model ids above are examples; the GPU scripts were dry-run with the mock backend only, not on a GPU.
* GitHub push protection will reject pushes of realistic-format fake secrets (AKIA..., ghp_..., sk_live_...). Release as an archive / HF dataset.
* Mutated real files are derivative works of permissive repos: keep licence notices, or release only the recipe (manifest + seed + code) plus the generated files.
* If you find a live credential in a real repo: do not copy it into the dataset or publish it; redact it and report it to the owner.
