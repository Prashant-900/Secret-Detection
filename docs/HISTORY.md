# Project history

The history contains **43 contribution milestones and 21 integration merges** across 21 topic branches plus `main`. The project timeline spans **11 September–5 November 2026**, in **Asia/Kolkata (UTC+05:30)**.

Source files retain model metadata, citations, and historical-result limitations. A commit that adds a training script or result does not establish when that experiment was run.

## Contributions

The README gives detailed responsibilities. [contributors.json](contributors.json) records author emails and file ownership. Primary commit counts cover the 43 non-merge contribution milestones and exclude the 21 integration merges. Coauthored entries identify milestone numbers, not counts. Integration merges are attributed to **Prashant-900** as the integrator, using the same contributor identity; they add no new contributor. These counts describe this grouping, not hours, productivity or ownership of upstream material.

| Contributor | Primary milestone commits | Coauthored milestone numbers |
|---|---:|---|
| [saurav10codes](https://github.com/saurav10codes) | 4 | 33 |
| [iashrkr23](https://github.com/iashrkr23) | 1 | 24, 32 |
| [KDS-7Git](https://github.com/KDS-7Git) | 8 | — |
| [titanansh-excel](https://github.com/titanansh-excel) | 5 | — |
| [Aditya-Chandel108](https://github.com/Aditya-Chandel108) | 6 | — |
| [AliYasoob-tech](https://github.com/AliYasoob-tech) | 3 | — |
| [Prashant-900](https://github.com/Prashant-900) | 9 | — |
| [mananmeemroth](https://github.com/mananmeemroth) | 3 | — |
| [FlameBoY01-OG](https://github.com/FlameBoY01-OG) | 3 | — |
| [Deepak-IITian07](https://github.com/Deepak-IITian07) | 1 | — |

KDS-7Git and Prashant-900 have the largest primary milestone counts, covering broader scanner/repository-selection and dataset-pipeline work. Deepak-IITian07 has one documentation/release support milestone.

## Branches and integration

The graph contains **21 topic branches plus `main`**. Milestone 1 establishes the root on `main`; milestones 2–43 are assigned to the topic branches below. Each topic is integrated into `main` with a two-parent merge commit. The table records the branch points and merge dates.

A topic normally starts from the `main` tip available at its first milestone. Three topics begin on another feature branch: legacy evaluation begins at generation milestone 18, generation validation at canonical-data milestone 21, and model experiments at scanner milestone 32. Branch names identify the topic lead; individual milestone authors and coauthors retain their credits, including on branches containing work from several contributors.

| Topic branch | Milestones | Branch point | Merge author date (IST) |
|---|---|---|---|
| `kds-7git/repository-catalogue` | 2, 4, 5 | `main` at start of milestone 2 | 2026-09-14 17:15 |
| `aliyasoob-tech/typed-credentials` | 3 | `main` at start of milestone 3 | 2026-09-13 11:40 |
| `prashant-900/tier-b-dataset` | 6, 9 | `main` at start of milestone 6 | 2026-09-18 12:20 |
| `iashrkr23/scanner-foundation` | 7, 8 | `main` at start of milestone 7 | 2026-09-17 16:05 |
| `kds-7git/source-audit` | 10, 11 | `main` at start of milestone 10 | 2026-09-20 12:45 |
| `titanansh-excel/legacy-verification` | 12, 14 | `main` at start of milestone 12 | 2026-09-24 11:25 |
| `saurav10codes/legacy-training` | 13, 15 | `main` at start of milestone 13 | 2026-09-25 15:50 |
| `prashant-900/legacy-generation` | 16, 17, 18, 20 | `main` at start of milestone 16 | 2026-10-02 11:50 |
| `aditya-chandel108/legacy-evaluation` | 19 | Milestone 18 on `prashant-900/legacy-generation` | 2026-09-30 17:30 |
| `prashant-900/canonical-import` | 21, 23 | `main` at start of milestone 21 | 2026-10-07 13:15 |
| `flameboy01-og/generation-validation` | 22 | Milestone 21 on `prashant-900/canonical-import` | 2026-10-06 16:40 |
| `kds-7git/candidate-coverage` | 24, 25 | `main` at start of milestone 24 | 2026-10-10 12:10 |
| `aliyasoob-tech/resumable-generation` | 26, 27, 28 | `main` at start of milestone 26 | 2026-10-15 17:00 |
| `saurav10codes/encoder-training` | 29 | `main` at start of milestone 29 | 2026-10-18 14:20 |
| `aditya-chandel108/span-evaluation` | 30 | `main` at start of milestone 30 | 2026-10-19 15:40 |
| `mananmeemroth/model-scanner` | 31, 32, 34 | `main` at start of milestone 31 | 2026-10-25 11:55 |
| `aditya-chandel108/model-experiments` | 33, 35 | Milestone 32 on `mananmeemroth/model-scanner` | 2026-10-27 16:15 |
| `titanansh-excel/pipeline-verification` | 36, 37, 38 | `main` at start of milestone 36 | 2026-10-30 14:55 |
| `prashant-900/cli-integration` | 39 | `main` at start of milestone 39 | 2026-10-31 13:20 |
| `aditya-chandel108/reporting` | 40, 41, 42 | `main` at start of milestone 40 | 2026-11-04 15:10 |
| `deepak-iitian07/release-documentation` | 43 | `main` at start of milestone 43 | 2026-11-05 14:30 |

Inspect the complete graph with:

```bash
git log --graph --oneline --decorate --all
```

## Milestones

Earlier commits are incremental source milestones; the complete integrated workflow is available at the final commit. Intermediate milestones have not been certified as independently runnable.

| # | Author date (IST) | Contributor | Commit |
|---:|---|---|---|
| 1 | 2026-09-11 10:15 | Prashant-900 | chore: establish project structure and repository hygiene |
| 2 | 2026-09-12 14:20 | KDS-7Git | data: catalogue pinned source repositories |
| 3 | 2026-09-13 11:10 | AliYasoob-tech | data: define typed synthetic credentials and benign controls |
| 4 | 2026-09-14 10:30 | KDS-7Git | data: expand manifests with Jenkins and Pulumi sources |
| 5 | 2026-09-14 16:45 | KDS-7Git | data: fetch repository snapshots at recorded revisions |
| 6 | 2026-09-15 13:25 | Prashant-900 | data: freeze owner-grouped train validation and OOD splits |
| 7 | 2026-09-16 17:10 | KDS-7Git | scanner: integrate provider patterns and IaC rules |
| 8 | 2026-09-17 15:35 | iashrkr23 | scanner: implement candidate extraction features and CI gates |
| 9 | 2026-09-18 11:50 | Prashant-900 | data: add deterministic real-file mutation and span labels |
| 10 | 2026-09-19 14:40 | KDS-7Git | data: audit preexisting credential-like repository content |
| 11 | 2026-09-20 12:15 | KDS-7Git | data: record the CredData expansion repository catalogue |
| 12 | 2026-09-21 16:20 | titanansh-excel | test: verify generated spans labels and paired examples |
| 13 | 2026-09-23 14:05 | saurav10codes | train: add the legacy boosted ranker and exported model |
| 14 | 2026-09-24 10:55 | titanansh-excel | test: assemble the scanner benchmark environment and runners |
| 15 | 2026-09-25 15:20 | saurav10codes | train: prepare contextual candidates and initial CodeBERT training |
| 16 | 2026-09-26 11:30 | Prashant-900 | data: connect local and mock language-model generation backends |
| 17 | 2026-09-27 16:15 | AliYasoob-tech | data: generate configuration scaffolds with deterministic slots |
| 18 | 2026-09-29 13:40 | FlameBoY01-OG | data: diversify contexts through label-preserving semantic edits |
| 19 | 2026-09-30 17:00 | Aditya-Chandel108 | eval: add legacy comparison protocols and historical aggregates |
| 20 | 2026-10-02 11:20 | Prashant-900 | docs: document dataset stages split policy and generation recipes |
| 21 | 2026-10-05 14:30 | Prashant-900 | data: establish canonical documents provenance and leakage checks |
| 22 | 2026-10-06 16:10 | FlameBoY01-OG | data: define diversity taxonomy syntax gates and duplicate checks |
| 23 | 2026-10-07 12:45 | Prashant-900 | data: import CredData and Tier B with frozen owner splits |
| 24 | 2026-10-09 15:25 | KDS-7Git | scanner: share label-independent candidate discovery across workflows |
| 25 | 2026-10-10 11:40 | KDS-7Git | data: measure training coverage and allocate generation quotas |
| 26 | 2026-10-12 17:15 | AliYasoob-tech | data: generate resumable labeled examples with a pilot profile |
| 27 | 2026-10-13 14:10 | FlameBoY01-OG | data: configure full generation and the local model server |
| 28 | 2026-10-15 16:30 | Prashant-900 | data: orchestrate import audit generation and candidate preparation |
| 29 | 2026-10-18 13:50 | saurav10codes | train: fine-tune weighted encoders with calibration and resume |
| 30 | 2026-10-19 15:10 | Aditya-Chandel108 | eval: score spans and bootstrap uncertainty by repository |
| 31 | 2026-10-21 12:20 | mananmeemroth | scanner: load exported encoders for reproducible batch inference |
| 32 | 2026-10-22 16:45 | mananmeemroth | scanner: deploy calibrated bundles with redaction and failure handling |
| 33 | 2026-10-24 14:35 | Aditya-Chandel108 | eval: compare encoders augmentation seeds and validation ensembles |
| 34 | 2026-10-25 11:25 | mananmeemroth | scanner: export training-only hard-example review queues |
| 35 | 2026-10-27 15:45 | Aditya-Chandel108 | eval: assess frozen models and run shared-corpus scanner baselines |
| 36 | 2026-10-28 13:15 | titanansh-excel | test: cover dataset integrity generation rejection and workflow smoke |
| 37 | 2026-10-29 16:00 | saurav10codes | train: verify local model export reload and checkpoint integration |
| 38 | 2026-10-30 14:25 | titanansh-excel | test: verify calibration experiments and baseline failure modes |
| 39 | 2026-10-31 12:50 | Prashant-900 | chore: integrate command routing diagnostics and remote setup |
| 40 | 2026-11-01 15:30 | Aditya-Chandel108 | report: calculate comparison statistics and render evaluation plots |
| 41 | 2026-11-02 16:20 | Aditya-Chandel108 | report: assemble offline reports and contextualize historical results |
| 42 | 2026-11-04 14:40 | titanansh-excel | test: validate report consistency redaction and historical comparisons |
| 43 | 2026-11-05 14:00 | Deepak-IITian07 | docs: finalize contributor credits runbooks and release packaging |

## GitHub attribution

All ten contributors have primary author commits on `main` and linked entries in the README. GitHub associates author commits with accounts when the commit email belongs to the corresponding account and the commits meet GitHub's contribution rules. Author display names alone do not link accounts.

For **iashrkr23**, the standard noreply address `229318849+iashrkr23@users.noreply.github.com` uses the public GitHub account ID **229318849**. Account association depends on GitHub's commit-email settings.

Reference: [GitHub commit-email settings](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).

## Repository contents

Git tracks source, configuration, frozen repository manifests/splits, the legacy exported model, tests, aggregate historical results and documentation. Generated datasets, downloaded repository checkouts, virtual environments, third-party tool executables, raw scan outputs, duplicate source ZIPs and local backups remain on disk and are ignored.

A fresh clone therefore needs environment setup, external scanner installation and restoration or regeneration of dataset inputs. Follow [the runbook](RUNBOOK.md) from the project root. Model scores for the remote encoder experiments must be read from their saved remote artifacts.