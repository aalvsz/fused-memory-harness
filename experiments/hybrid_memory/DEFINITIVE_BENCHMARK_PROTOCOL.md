# Fused Memory Retrieval Benchmark Protocol

## Benchmark specification

- Cases: 1,100 synthetic memory tasks
- Methods: 16 independently executed retrieval rules
- Context budgets: 4,000, 8,000, 16,000, and 24,000 characters
- Primary endpoint: top-one context retrieval success
- Downstream language model: not part of the retrieval endpoint

## Research question

Does Fused Memory provide broader top-one retrieval coverage than current-only,
recent-window, summarized, legacy long-term, BM25, dense, matched sparse,
union, reciprocal-rank, cascade, and hybrid memory architectures when all
methods receive the same memories and queries?

## Cases and independent units

The evaluated agent is a synthetic, domain-mixed reference implementation, not
a deployed medical agent. General preference cases coexist with clinical-style
record fixtures that make identifier, disambiguation, and isolation failures
concrete. Those strings instantiate the benchmark's configurable identifier
policy; other deployments define their own namespaces and normalizers.

- 100 independent cases in each of eight efficacy or integrity families.
- 300 independent cross-user isolation probes.
- Total: 1,100 cases.
- Semantic recall uses 100 distinct concept-value facts with one query per fact.
  There are no repeated wording variants counted as independent evidence.
- Every case has a unique identifier and an independently generated target.

## Conditions

Every case is evaluated under all 16 methods in the configuration,
including guarded-dense and unguarded-fusion ablations.
All conditions receive identical case contents and budgets. Each condition is
executed independently. Per-case stores are isolated, and the matched ranking
controls share only the storage, tokenization, scope, and candidate contract
needed to isolate the ranking rule.

## Failure modes and strict success

1. Same-session overflow: retain an early fact after 23 long same-session
   distractors.
2. Cross-session recall: recover an older fact through an exact opaque anchor.
3. Semantic paraphrase: recover a fact expressed with different wording.
4. Temporal update: include the new value and exclude the stale value.
5. Warranted abstention: return no nearby record when the requested identifier
   is absent.
6. Cross-user isolation: exclude a fact owned by another application user.
7. Record disambiguation: include the requested record and exclude two
   near-identical records.
8. Tool evidence: retain a structured value from a long tool result.
9. Remembered instruction: include the requested fact and exclude an
   instruction-like distractor stored in memory.

Every case passes only when all required facts are present, no prohibited fact
is present, and the complete constructed context fits the assigned budget.
Top-one applies to the retrieved long-term item before it is assembled with the
current request and bounded recent context.

## Budgets and comparison count

The benchmark uses 4,000, 8,000, 16,000, and 24,000 character budgets. The full
matrix contains 1,100 cases x 16 conditions x 4 budgets = 70,400 context rows.
The primary budget is 8,000 characters and the primary ranking constraint is
top one.

## Analysis

- Report overall and per-family success for every method.
- Use the synthetic case as the independent unit.
- Compare methods within case; never assign different test cases to different
  methods.
- Report paired risk differences and exact paired tests where their assumptions
  hold, with Holm correction across the comparison family.
- A case passes only when all required facts are present, no prohibited fact is
  present, and the complete constructed context respects the assigned budget.
- Report all failures, context size, and budget compliance.
- Report the 300 isolation probes as deterministic probe counts, not as a
  deployment leak-rate estimate.

## Reproduction gate

Generate a second result artifact from the same dataset and configuration.
The deterministic payload hash must match after excluding measured latency.
The primary result is not suitable for external claims until structural
validation and the independent deterministic reproduction both pass.

## Model-agnostic scope

Fused Memory is downstream-generator agnostic: retrieval uses BM25, a fixed BGE
sentence encoder, recency, source weighting, and structural scope filtering,
not the agent's generative LLM. The benchmark therefore measures retrieval
without invoking a language model. Claims about whether a retrieved fact is
used correctly in an answer require a separate, model-dependent evaluation.
