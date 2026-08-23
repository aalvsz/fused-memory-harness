# Fused Memory

![Fused Memory — Scoped retrieval. Reproducible evidence.](assets/social-preview.jpg)

[![Tests](https://github.com/aalvsz/fused-memory-harness/actions/workflows/tests.yml/badge.svg)](https://github.com/aalvsz/fused-memory-harness/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2ea44f.svg)](LICENSE)

Fused Memory is a downstream-model-agnostic retrieval method for long-term
agents. It first enforces application, user, and structured-identifier
consistency, then combines calibrated dense and BM25 evidence with
query-adaptive weights, recency tie-breaking, and source-aware ranking.

The central problem is coverage. Exact identifiers favor lexical retrieval;
paraphrases favor dense retrieval; updates require temporal ordering; and
cross-user safety requires a structural boundary before ranking. Fused Memory
puts those signals into one deterministic retrieval policy.

The harness uses a synthetic, domain-mixed reference agent. General preference
memories are combined with clinical-style record fixtures so identifier,
disambiguation, and isolation behavior is easy to inspect. These fixtures are
test data, not a medical-agent dependency: deployments configure their own
identifier namespaces, such as account IDs, issue keys, or document IDs.

## Benchmark harness

The benchmark code evaluates 16 independently executed retrieval methods on
1,100 synthetic cases across nine failure-mode families and four context
budgets. Every method receives the same memories, queries, and top-one
constraint. The primary endpoint is retrieval success before answer generation;
the harness also records forbidden-fact checks, context size, and budget
compliance. Run outputs are generated locally and are intentionally excluded
from this repository.

The benchmark evaluates retrieval before answer generation. Fused Memory can
therefore feed any downstream language model, while final answer quality still
depends on that model.

## Repository contents

- `experiments/hybrid_memory/`: generator, retrievers, benchmark, validation,
  analysis, configuration, and protocol.
- `fused_memory_harness/runtime/`: standalone context-compaction and legacy
  retrieval controls required by the comparison.
- `autoresearch/` is reserved for local benchmark outputs and is ignored by
  Git; no experimental result artifacts are published here.
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

## Run the retrieval benchmark locally

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

Read the [benchmark protocol](experiments/hybrid_memory/DEFINITIVE_BENCHMARK_PROTOCOL.md)
before interpreting or extending the benchmark. The repository contains the
method and evaluation code only; it makes no leaderboard or universal
end-to-end agent claim.

## License

Apache-2.0. See [LICENSE](LICENSE).
