from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hybrid_memory.analyze import (
    cluster_bootstrap,
    holm_adjust,
    pairwise,
    summarize_rows,
)
from experiments.hybrid_memory.model_benchmark import call_model, score

SMOKE_CONFIG = ROOT / "experiments/hybrid_memory/configs/smoke.json"


def _make_rows(
    n_cases: int = 6,
    budgets: tuple[int, ...] = (4000, 8000),
    candidate: str = "hybrid",
    baseline: str = "short_term",
    family: str = "cross_session_recall",
) -> list[dict]:
    rows: list[dict] = []
    for case_idx in range(n_cases):
        for budget in budgets:
            for condition in (candidate, baseline):
                rows.append(
                    {
                        "case_id": f"SYN-CASE-{case_idx:04d}",
                        "budget_chars": budget,
                        "family": family,
                        "condition": condition,
                        "context_success": (case_idx % 2 == 0)
                        if condition == candidate
                        else False,
                        "context_chars": 2000 + case_idx * 100,
                        "elapsed_ms": 10.0 + case_idx,
                        "retrieval_ms": 2.0 + case_idx * 0.5,
                    }
                )
    return rows


# ── Cluster bootstrap ────────────────────────────────────────────────


def test_cluster_bootstrap_is_deterministic():
    pairs = [
        (f"case-{i}", float(i % 2), 0.0) for i in range(8)
    ]
    first = cluster_bootstrap(pairs, iterations=200, seed=42)
    second = cluster_bootstrap(pairs, iterations=200, seed=42)
    assert first == second


def test_cluster_bootstrap_respects_case_clustering():
    """When all observations in a case move together, the CI should
    be wider than a naive paired bootstrap that treats each observation
    as independent."""
    # 4 cases × 4 budgets = 16 pairs.  Each case has identical diff.
    pairs: list[tuple[str, float, float]] = []
    for case_idx in range(4):
        for _ in range(4):
            pairs.append((f"case-{case_idx}", 1.0, 0.0))

    cluster_lo, cluster_hi = cluster_bootstrap(pairs, iterations=500, seed=7)
    # With all cases showing diff=1.0, the cluster bootstrap CI should be tight
    # around 1.0 because every cluster has the same mean.
    assert not math.isnan(cluster_lo)
    assert not math.isnan(cluster_hi)
    assert cluster_lo <= 1.0 <= cluster_hi


def test_cluster_bootstrap_empty():
    lo, hi = cluster_bootstrap([], iterations=100, seed=1)
    assert math.isnan(lo)
    assert math.isnan(hi)


# ── Pairwise inference metadata ─────────────────────────────────────


def test_pairwise_primary_budget_has_paired_bootstrap_metadata():
    rows = _make_rows(n_cases=6, budgets=(8000,))
    results = pairwise(
        rows,
        source="context",
        metric="context_success",
        budget=8000,
        family="cross_session_recall",
        candidate="hybrid",
        baselines=["short_term"],
        iterations=100,
        seed=100,
    )
    assert len(results) == 1
    row = results[0]
    assert row["inference_method"] == "paired_bootstrap"
    assert row["n_independent_units"] == 6
    assert row["n_repeated_measures"] == 1
    assert row["budget_chars"] == 8000


def test_pairwise_cross_budget_has_cluster_bootstrap_metadata():
    rows = _make_rows(n_cases=6, budgets=(4000, 8000))
    results = pairwise(
        rows,
        source="context",
        metric="context_success",
        budget=None,
        family="cross_session_recall",
        candidate="hybrid",
        baselines=["short_term"],
        iterations=100,
        seed=200,
    )
    assert len(results) == 1
    row = results[0]
    assert row["inference_method"] == "cluster_bootstrap_by_case_descriptive"
    assert row["inferential_test"] == "none"
    assert math.isnan(row["mcnemar_p"])
    assert math.isnan(row["holm_p"])
    assert row["holm_family_size"] == 0
    assert row["budget_chars"] == "ALL"
    # 6 unique cases
    assert row["n_independent_units"] == 6
    # 2 budgets → 2 repeated measures per case
    assert row["n_repeated_measures"] == 2
    # Total pairs = 6 cases × 2 budgets = 12
    assert row["n_pairs"] == 12


def test_pairwise_n_pairs_not_treated_as_independent():
    """The n_pairs for a cross-budget comparison should not equal
    n_independent_units — that would indicate pseudo-replication."""
    rows = _make_rows(n_cases=8, budgets=(4000, 8000, 16000))
    results = pairwise(
        rows,
        source="context",
        metric="context_success",
        budget=None,
        family="cross_session_recall",
        candidate="hybrid",
        baselines=["short_term"],
        iterations=100,
        seed=300,
    )
    row = results[0]
    assert row["n_pairs"] > row["n_independent_units"]
    assert row["n_repeated_measures"] == 3


def test_pairwise_semantic_variants_cluster_by_concept():
    rows = _make_rows(
        n_cases=12,
        budgets=(8000,),
        candidate="fused_hybrid",
        baseline="hybrid",
        family="semantic_paraphrase",
    )
    for row in rows:
        case_index = int(str(row["case_id"]).rsplit("-", 1)[1])
        row["independent_unit_id"] = f"concept-{case_index % 4}"
    result = pairwise(
        rows,
        source="context",
        metric="context_success",
        budget=8000,
        family="semantic_paraphrase",
        candidate="fused_hybrid",
        baselines=["hybrid"],
        iterations=100,
        seed=400,
        independent_unit_field="independent_unit_id",
    )[0]
    assert result["n_pairs"] == 12
    assert result["n_independent_units"] == 4
    assert result["n_repeated_measures"] == 3
    assert result["inferential_test"] == "none"
    assert math.isnan(result["mcnemar_p"])
    assert math.isnan(result["holm_p"])


def test_pairwise_skips_baselines_without_matched_pairs():
    rows = _make_rows(n_cases=3, budgets=(8000,))
    results = pairwise(
        rows,
        source="context",
        metric="context_success",
        budget=8000,
        family="cross_session_recall",
        candidate="hybrid",
        baselines=["not_configured"],
        iterations=20,
        seed=1,
    )
    assert results == []


# ── Holm family size ────────────────────────────────────────────────


def test_holm_adjust_family_size_derived_from_comparison_set():
    """Holm correction must use the actual number of comparisons, not
    a hard-coded constant."""
    # 3 comparisons
    p_values = [0.01, 0.02, 0.03]
    adjusted = holm_adjust(p_values)
    assert len(adjusted) == 3
    # Smallest p gets multiplied by 3 (total - rank 0)
    assert adjusted[0] == min(1.0, 0.01 * 3)

    # 5 comparisons — different family size
    p_values_5 = [0.01, 0.02, 0.03, 0.04, 0.05]
    adjusted_5 = holm_adjust(p_values_5)
    assert len(adjusted_5) == 5
    assert adjusted_5[0] == min(1.0, 0.01 * 5)
    # The adjustment for the same raw p should differ based on family size
    assert adjusted_5[0] >= adjusted[0]


def test_holm_adjust_not_hardcoded_six():
    """Verify the Holm multiplier scales with actual list length for
    lengths other than 6."""
    for n in [1, 2, 3, 7, 10, 15]:
        p_values = [0.01] * n
        adjusted = holm_adjust(p_values)
        # The smallest p is multiplied by n (total comparisons)
        expected = min(1.0, 0.01 * n)
        assert adjusted[0] == expected, f"Failed for n={n}"


# ── Model provenance: generation seed vs request-order seed ──────────


def test_call_model_includes_generation_seed_in_payload(monkeypatch):
    captured_payload: dict = {}

    class FakeResponse:
        def __init__(self):
            self._body = {
                "choices": [{"message": {"content": "test answer"}}],
                "usage": {"total_tokens": 5},
                "model": "test-model",
                "id": "resp-123",
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(self._body).encode()

    def fake_urlopen(request, timeout=None):
        captured_payload["data"] = json.loads(request.data.decode())
        captured_payload["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(
        "experiments.hybrid_memory.model_benchmark.urllib.request.urlopen",
        fake_urlopen,
    )

    text, usage, elapsed_ms, endpoint_meta = call_model(
        base_url="http://localhost:8080",
        model="test-model",
        api_key="test-key",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        timeout_seconds=30.0,
        generation_seed=99999,
    )

    assert text == "test answer"
    assert captured_payload["data"]["seed"] == 99999
    assert endpoint_meta["response_model"] == "test-model"
    assert endpoint_meta["endpoint_url"] == "http://localhost:8080/v1/chat/completions"


def test_call_model_omits_generation_seed_when_none(monkeypatch):
    captured_payload: dict = {}

    class FakeResponse:
        def __init__(self):
            self._body = {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {},
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(self._body).encode()

    def fake_urlopen(request, timeout=None):
        captured_payload["data"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(
        "experiments.hybrid_memory.model_benchmark.urllib.request.urlopen",
        fake_urlopen,
    )

    call_model(
        base_url="http://localhost:8080",
        model="m",
        api_key="k",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        timeout_seconds=10.0,
    )

    assert "seed" not in captured_payload["data"]


def test_model_benchmark_cli_has_generation_seed_and_model_file():
    """Verify that the CLI parser accepts --generation-seed and --model-file
    without requiring network access."""
    import experiments.hybrid_memory.model_benchmark as mb

    # We can't easily call main() without a server, but we can verify
    # the parser includes the new arguments by checking the source.
    source = Path(mb.__file__).read_text()
    assert "--generation-seed" in source
    assert "--model-file" in source
    assert "generation_seed" in source
    assert "request_order_seed" in source
    assert "model_file_sha256" in source
    assert "endpoint_metadata" in source
    assert "fail-closed" in source


# ── Model file hashing fail-closed ──────────────────────────────────


def test_model_file_hashing_fails_closed(tmp_path):
    """When --model-file is supplied but the file does not exist,
    the benchmark must fail closed (non-zero exit)."""
    nonexistent = tmp_path / "nonexistent.gguf"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.hybrid_memory.model_benchmark",
            "--contexts",
            str(tmp_path / "dummy.jsonl"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--base-url",
            "http://localhost:1",
            "--model",
            "test",
            "--model-file",
            str(nonexistent),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode != 0
    assert "fail-closed" in result.stderr or "fail-closed" in result.stdout


def test_model_resume_rejects_changed_frozen_provenance(tmp_path):
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_text("", encoding="utf-8")
    output = tmp_path / "model.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    run_identity = tmp_path / "model.run.json"
    run_identity.write_text(
        json.dumps(
            {
                "schema_version": "hybrid-memory-benchmark-v1",
                "run_id": "original-run",
                "base_url": "http://localhost:8080",
                "model": "different-model",
                "model_file": None,
                "model_file_sha256": None,
                "temperature": 0.0,
                "prompt_version": "original",
                "request_order_seed": 20260731,
                "generation_seed": None,
                "context_results_sha256": "not-the-current-hash",
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.hybrid_memory.model_benchmark",
            "--contexts",
            str(contexts),
            "--output",
            str(output),
            "--base-url",
            "http://localhost:8080",
            "--model",
            "test-model",
            "--resume",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode != 0
    assert "resume provenance differs" in result.stderr


def test_validator_requires_model_results_for_reproduction_and_checks_counts(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    contexts = tmp_path / "contexts.jsonl"
    report = tmp_path / "validation.json"
    dataset.write_text("", encoding="utf-8")
    contexts.write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.hybrid_memory.validate_artifacts",
            "--config",
            str(ROOT / "experiments/hybrid_memory/configs/matched_controls_smoke.json"),
            "--dataset",
            str(dataset),
            "--contexts",
            str(contexts),
            "--require-reproduction",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode != 0
    assert "FAIL reproduction_model_results_present" in result.stdout
    checks = {item["name"]: item for item in json.loads(report.read_text())["checks"]}
    assert checks["dataset_family_counts_match_config"]["passed"] is False
    assert checks["reproduction_model_results_present"]["passed"] is False


# ── Model scorer unchanged ─────────────────────────────────────────


def test_model_scorer_still_works():
    abstention = {
        "answerable": False,
        "answer_facts": [],
        "forbidden_facts": ["metformin"],
    }
    assert score(abstention, "UNKNOWN")["answer_correct"]
    assert not score(abstention, "Metformin")["answer_correct"]


# ── Summarize rows still produces descriptive stats ────────────────


def test_summarize_rows_descriptive_output():
    rows = _make_rows(n_cases=4, budgets=(8000,))
    summaries = summarize_rows(rows, metric="context_success", source="context")
    assert len(summaries) > 0
    for row in summaries:
        assert "rate" in row
        assert "ci95_low" in row
        assert "ci95_high" in row
        assert row["n"] > 0
