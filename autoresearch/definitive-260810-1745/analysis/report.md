# Hybrid Memory Experiment Analysis

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10T16:55:41.680793+00:00
- Verification Status: ANALYZED
- Version Label: validation_v1

## Analysis scope

- Context rows: 61600
- Model rows: 0
- Primary budget: 8000
- Primary family: semantic_paraphrase
- Candidate: fused_hybrid
- Evidence status: frozen_definitive_context
- Independent-unit field: independent_unit_id
- Bootstrap iterations: 10000
- All cases are synthetic. Results do not establish clinical effectiveness.

## Configured primary comparisons

| Source | Baseline | N pairs | Candidate | Baseline | Risk difference [95% CI] | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| context | latest_only | 100 | 1.000 | 0.000 | 1.000 [1.000, 1.000] | 0.0000 |
| context | raw_recent | 100 | 1.000 | 0.000 | 1.000 [1.000, 1.000] | 0.0000 |
| context | short_term | 100 | 1.000 | 0.000 | 1.000 [1.000, 1.000] | 0.0000 |
| context | long_term | 100 | 1.000 | 0.030 | 0.970 [0.930, 1.000] | 0.0000 |
| context | hybrid | 100 | 1.000 | 0.030 | 0.970 [0.930, 1.000] | 0.0000 |
| context | bm25_long_term | 100 | 1.000 | 0.010 | 0.990 [0.970, 1.000] | 0.0000 |
| context | bm25_hybrid | 100 | 1.000 | 0.010 | 0.990 [0.970, 1.000] | 0.0000 |
| context | dense_long_term | 100 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 1.0000 |
| context | dense_hybrid | 100 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 1.0000 |
| context | matched_sparse_hybrid | 100 | 1.000 | 0.010 | 0.990 [0.970, 1.000] | 0.0000 |
| context | union_hybrid | 100 | 1.000 | 0.030 | 0.970 [0.930, 1.000] | 0.0000 |
| context | rrf_hybrid | 100 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 1.0000 |
| context | cascade_hybrid | 100 | 1.000 | 0.010 | 0.990 [0.970, 1.000] | 0.0000 |

## Safety endpoint

- Observed context forbidden-token hits at the primary 8,000-character budget: 0/4,200 rows from 300 synthetic case probes across 14 methods. Across all four configured budgets, validation found 0/16,800 forbidden-token hits.
- Model-answer leakage was not evaluated.
- These are deterministic synthetic probe counts, not independent deployment samples; no population leak-rate confidence bound is reported.

## Missingness and run integrity

- Model benchmark was not supplied; no end-to-end claims are available.
- Pairwise tests use only complete matched case-condition pairs.
- Exact McNemar tests are used only when each row is one independent unit;
  otherwise concept or case clusters are descriptive and receive no p-value.
- Holm family size is derived from the actual inferential comparison set.
- Primary-budget CIs resample the configured independent unit.
- Cross-budget pooled comparisons use cluster bootstrap by case_id to respect
  within-case repeated-measure dependence; they are descriptive robustness checks
  and receive no McNemar or Holm-adjusted p-value.

## Statistical fallacy scan

Coverage: 11/11 checked.

1. Simpson's paradox: aggregate and per-family tables are emitted for direction checks.
2. Ecological fallacy: inference is restricted to synthetic benchmark cases.
3. Berkson's paradox: selected synthetic cases do not estimate clinical prevalence.
4. Collider bias: no post-treatment covariate adjustment is performed.
5. Base-rate neglect: abstention and isolation family sizes are reported explicitly.
6. Regression to the mean: no extreme-score enrollment or pre/post outcome is used.
7. Survivorship bias: failed model calls are counted and reported, not silently dropped.
8. Look-elsewhere effect: comparisons follow the frozen config contract and are Holm-corrected
   with family size derived from the actual comparison set.
9. Garden of forking paths: configuration, dataset, result hashes, and code revision are recorded.
10. Correlation versus causation: paired condition assignment supports claims only about this benchmark harness.
11. Reverse causality: not applicable to randomized memory-condition evaluation.

## Interpretation boundary

This report describes frozen_definitive_context benchmark behavior. A broader external claim additionally requires an independently reviewed evidence profile, successful end-to-end model execution on a pinned harness, independent artifact validation, and robustness analyses across context budgets. Multi-budget aggregates are descriptive; inferential claims require one paired observation per configured independent unit. Repeated variants are clustered and reported descriptively.
