# Fused Memory

Fused Memory is a downstream-model-agnostic retrieval method for long-term
agents. It first enforces application and user scope, then combines calibrated
dense and BM25 evidence with recency tie-breaking and source-aware ranking.

The central problem is coverage. Exact identifiers favor lexical retrieval;
paraphrases favor dense retrieval; updates require temporal ordering; and
cross-user safety requires a structural boundary before ranking. Fused Memory
puts those signals into one deterministic retrieval policy.

## Definitive benchmark

The frozen benchmark evaluates 14 independently executed memory methods on
1,100 unseen synthetic cases across nine failure-mode families and four context
budgets. Each method receives the same memories, queries, and top-one
constraint, producing 61,600 paired evaluations.

At the primary 8,000-character budget:

| Method | Retrieval success |
| --- | ---: |
| **Fused Memory** | **1,100/1,100 (100.0%)** |
| Gated reciprocal-rank fusion | 1,008/1,100 (91.6%) |
| Dense + recent | 1,003/1,100 (91.2%) |
| Union + recent | 1,003/1,100 (91.2%) |
| Legacy hybrid | 998/1,100 (90.7%) |

Fused Memory was the only tested method to retrieve every target in every
family. A second execution reproduced every deterministic retrieval decision.
The isolation suite observed zero forbidden-context hits in 16,800 deterministic
probes; that is a benchmark count, not a deployment leak-rate estimate.

The benchmark evaluates retrieval before answer generation. Fused Memory can
therefore feed any downstream language model, while final answer quality still
depends on that model.

## Repository contents

- `experiments/hybrid_memory/`: generator, retrievers, benchmark, validation,
  analysis, frozen configuration, and protocol.
- `fused_memory_harness/runtime/`: standalone context-compaction and legacy
  retrieval controls required by the comparison.
- `autoresearch/definitive-260810-1745/`: synthetic cases and compact frozen
  evidence. The two 134 MiB raw context dumps are omitted from Git and can be
  regenerated from the dataset and configuration.
- `tests/`: focused behavioral, provenance, and reproduction tests.

No production application, patient data, deployment configuration, credentials,
or private repository history is included.

## Setup

The embedding model is downloaded and cached by FastEmbed on first use.

```bash
git clone https://github.com/aalvsz/fused-memory-harness.git
cd fused-memory-harness
uv sync --extra dev
uv run pytest -q
```

## Quick smoke run

```bash
run_dir=autoresearch/smoke-local
mkdir -p "$run_dir"

uv run python -m experiments.hybrid_memory.generate_cases \
  --config experiments/hybrid_memory/configs/smoke.json \
  --output "$run_dir/cases.jsonl"

uv run python -m experiments.hybrid_memory.context_benchmark \
  --config experiments/hybrid_memory/configs/smoke.json \
  --dataset "$run_dir/cases.jsonl" \
  --output "$run_dir/context-results.jsonl"

uv run python -m experiments.hybrid_memory.validate_artifacts \
  --config experiments/hybrid_memory/configs/smoke.json \
  --dataset "$run_dir/cases.jsonl" \
  --contexts "$run_dir/context-results.jsonl" \
  --output "$run_dir/validation.json"

uv run python -m experiments.hybrid_memory.analyze \
  --config experiments/hybrid_memory/configs/smoke.json \
  --contexts "$run_dir/context-results.jsonl" \
  --output-dir "$run_dir/analysis"
```

## Reproduce the definitive retrieval benchmark

The primary and reproduction runs each take approximately 15--20 minutes on a
modern laptop CPU. GPU acceleration is not required for this workload.

```bash
run_dir=autoresearch/definitive-local
mkdir -p "$run_dir"

uv run python -m experiments.hybrid_memory.generate_cases \
  --config experiments/hybrid_memory/configs/definitive_heldout.json \
  --output "$run_dir/cases.jsonl"

uv run python -m experiments.hybrid_memory.context_benchmark \
  --config experiments/hybrid_memory/configs/definitive_heldout.json \
  --dataset "$run_dir/cases.jsonl" \
  --output "$run_dir/context-results.jsonl"

uv run python -m experiments.hybrid_memory.context_benchmark \
  --config experiments/hybrid_memory/configs/definitive_heldout.json \
  --dataset "$run_dir/cases.jsonl" \
  --output "$run_dir/context-results-reproduction.jsonl"

uv run python -m experiments.hybrid_memory.validate_artifacts \
  --config experiments/hybrid_memory/configs/definitive_heldout.json \
  --dataset "$run_dir/cases.jsonl" \
  --contexts "$run_dir/context-results.jsonl" \
  --context-reproduction-reference "$run_dir/context-results-reproduction.jsonl" \
  --require-context-reproduction \
  --output "$run_dir/validation.json"
```

Read the [frozen protocol](experiments/hybrid_memory/DEFINITIVE_BENCHMARK_PROTOCOL.md)
before interpreting or extending the benchmark. The current evidence establishes
superiority on this synthetic retrieval benchmark; it does not establish
universal end-to-end agent superiority or production security.

## License

Apache-2.0. See [LICENSE](LICENSE).
