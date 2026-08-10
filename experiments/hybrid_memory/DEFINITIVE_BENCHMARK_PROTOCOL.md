# Fused Memory Definitive Held-Out Benchmark Protocol

## Freeze status

- Status: frozen before result generation
- Profile: `definitive_heldout`
- Configuration: `configs/definitive_heldout.json`
- Primary endpoint: top-one context retrieval success
- Downstream language model: not part of the retrieval endpoint

No method weights, thresholds, cases, distractors, conditions, budgets, or
analysis rules may be changed after the result file is opened. A failed method
or family remains a reported result. Any later modification requires a new
versioned benchmark namespace and is exploratory until independently repeated.

## Research question

Does Fused Memory provide broader top-one retrieval coverage than current-only,
recent-window, summarized, legacy long-term, BM25, dense, matched sparse,
union, reciprocal-rank, cascade, and prior hybrid memory architectures when all
methods receive the same fully unseen memories and queries?

## Cases and independent units

- 100 independent cases in each of eight efficacy or integrity families.
- 300 independent cross-user isolation probes.
- Total: 1,100 cases.
- Semantic recall uses 100 distinct concept-value facts with one query per fact.
  There are no repeated wording variants counted as independent evidence.
- Case identifiers use the `definitive-v1` namespace.
- The seed, semantic catalog, queries, distractors, and identifiers are disjoint
  from the pilot, prior 1,180-case benchmark, development pool, and 23-case
  matched-control set.

## Conditions

Every case is evaluated under all 14 frozen conditions in the configuration.
All conditions receive identical case contents and budgets. Each condition is
executed independently. Per-case stores are isolated, and the matched ranking
controls share only the storage, tokenization, scope, and candidate contract
needed to isolate the ranking rule.

## Budgets and comparison count

The benchmark uses 4,000, 8,000, 16,000, and 24,000 character budgets. The full
matrix contains 1,100 cases x 14 conditions x 4 budgets = 61,600 context rows.
The primary budget is 8,000 characters and the primary ranking constraint is
top one.

## Analysis

- Report overall and per-family success for every method.
- Use the synthetic case as the independent unit.
- Compare methods within case; never assign different test cases to different
  methods.
- Report paired risk differences and exact paired tests where their assumptions
  hold, with Holm correction across the frozen comparison family.
- Report all failures, latency, context size, and budget compliance.
- Report the 300 isolation probes as deterministic probe counts, not as a
  deployment leak-rate estimate.

## Reproduction gate

Generate a second result artifact from the frozen dataset and configuration.
The deterministic payload hash must match after excluding measured latency.
The primary result is not suitable for external claims until structural
validation and the independent deterministic reproduction both pass.

## Model-agnostic scope

Fused Memory is downstream-generator agnostic: retrieval uses BM25, a fixed BGE
sentence encoder, recency, source weighting, and structural scope filtering,
not the agent's generative LLM. The benchmark therefore measures retrieval
without invoking a language model. Claims about whether a retrieved fact is
used correctly in an answer require a separate, model-dependent evaluation.
