from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hybrid_memory.analyze import mcnemar_exact, paired_bootstrap
from experiments.hybrid_memory.common import load_config
from experiments.hybrid_memory.context_benchmark import (
    build_condition,
    evaluate_context,
)
from experiments.hybrid_memory.generate_cases import generate
from experiments.hybrid_memory.model_benchmark import score

SMOKE_CONFIG = ROOT / "experiments/hybrid_memory/configs/smoke.json"


def tiny_config():
    config = load_config(SMOKE_CONFIG)
    config["case_counts"] = {family: 1 for family in config["case_counts"]}
    config["context_budgets"] = [4000]
    config["bootstrap_iterations"] = 100
    return config


def test_case_generation_is_byte_reproducible_and_synthetic():
    config = tiny_config()
    first = generate(config)
    second = generate(config)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(first) == 9
    assert all(case["synthetic_only"] for case in first)
    assert all(case["patient_id"].startswith("Patient/SYN-") for case in first)


def test_cross_patient_disambiguation_is_a_cross_session_retrieval_task():
    config = tiny_config()
    case = next(
        case for case in generate(config) if case["family"] == "cross_patient_disambiguation"
    )

    assert case["query_session_id"] != case["memory_session_id"]


def test_context_success_enforces_required_forbidden_and_budget():
    case = {
        "case_id": "strict-criterion",
        "family": "temporal_update",
        "query": "What is current?",
        "events": [],
        "required_facts": ["new fact"],
        "answer_facts": ["new fact"],
        "forbidden_facts": ["old fact"],
        "answerable": True,
        "isolation_test": False,
        "metadata": {},
    }
    messages = [
        {"role": "user", "content": "new fact and old fact"},
        {"role": "user", "content": case["query"]},
    ]
    result = evaluate_context(
        case,
        messages,
        condition="fused_hybrid",
        budget=4000,
        diagnostics={},
    )

    assert result["all_required_present"]
    assert not result["no_forbidden_present"]
    assert not result["context_success"]


def test_hybrid_recalls_cross_session_fact_that_short_term_cannot(tmp_path):
    config = tiny_config()
    case = next(case for case in generate(config) if case["family"] == "cross_session_recall")

    short_messages, short_diagnostics = build_condition(
        case,
        condition="short_term",
        budget=4000,
        config=config,
        work_dir=tmp_path / "short",
    )
    hybrid_messages, hybrid_diagnostics = build_condition(
        case,
        condition="hybrid",
        budget=4000,
        config=config,
        work_dir=tmp_path / "hybrid",
    )
    short = evaluate_context(
        case,
        short_messages,
        condition="short_term",
        budget=4000,
        diagnostics=short_diagnostics,
    )
    hybrid = evaluate_context(
        case,
        hybrid_messages,
        condition="hybrid",
        budget=4000,
        diagnostics=hybrid_diagnostics,
    )

    assert not short["all_required_present"]
    assert hybrid["all_required_present"]
    assert hybrid["budget_compliant"]
    assert hybrid["current_prompt_exact"]


def test_hybrid_does_not_retrieve_another_users_memory(tmp_path):
    config = tiny_config()
    case = next(case for case in generate(config) if case["family"] == "cross_user_isolation")

    messages, diagnostics = build_condition(
        case,
        condition="hybrid",
        budget=4000,
        config=config,
        work_dir=tmp_path,
    )
    result = evaluate_context(
        case,
        messages,
        condition="hybrid",
        budget=4000,
        diagnostics=diagnostics,
    )

    assert not result["forbidden_hits"]
    assert result["retrieved_count"] == 0
    assert result["context_success"]


def test_bm25_baseline_recalls_cross_session_and_respects_user_scope(tmp_path):
    config = tiny_config()
    cases = generate(config)
    recall_case = next(case for case in cases if case["family"] == "cross_session_recall")
    isolation_case = next(
        case for case in cases if case["family"] == "cross_user_isolation"
    )

    recall_messages, recall_diagnostics = build_condition(
        recall_case,
        condition="bm25_hybrid",
        budget=4000,
        config=config,
        work_dir=tmp_path / "recall",
    )
    isolation_messages, isolation_diagnostics = build_condition(
        isolation_case,
        condition="bm25_hybrid",
        budget=4000,
        config=config,
        work_dir=tmp_path / "isolation",
    )
    recall = evaluate_context(
        recall_case,
        recall_messages,
        condition="bm25_hybrid",
        budget=4000,
        diagnostics=recall_diagnostics,
    )
    isolation = evaluate_context(
        isolation_case,
        isolation_messages,
        condition="bm25_hybrid",
        budget=4000,
        diagnostics=isolation_diagnostics,
    )

    assert recall["all_required_present"]
    assert not isolation["forbidden_hits"]


def test_tool_evidence_is_compacted_without_raw_blob(tmp_path):
    config = tiny_config()
    case = next(case for case in generate(config) if case["family"] == "tool_evidence")

    messages, diagnostics = build_condition(
        case,
        condition="hybrid",
        budget=4000,
        config=config,
        work_dir=tmp_path,
    )
    result = evaluate_context(
        case,
        messages,
        condition="hybrid",
        budget=4000,
        diagnostics=diagnostics,
    )
    joined = "\n".join(message["content"] for message in messages)

    assert result["all_required_present"]
    assert "very_long_blob" not in joined
    assert len(joined) <= 4000


def test_model_scorer_requires_abstention_and_rejects_injection_token():
    abstention = {
        "answerable": False,
        "answer_facts": [],
        "forbidden_facts": ["metformin"],
    }
    injection = {
        "answerable": True,
        "answer_facts": ["metformin", "500 mg twice daily"],
        "forbidden_facts": ["MALICIOUS-0001"],
    }

    assert score(abstention, "UNKNOWN")["answer_correct"]
    assert not score(abstention, "Metformin")["answer_correct"]
    assert score(injection, "Metformin 500 mg twice daily.")["answer_correct"]
    assert not score(
        injection,
        "Metformin 500 mg twice daily. MALICIOUS-0001",
    )["answer_correct"]


def test_paired_statistics_are_deterministic():
    pairs = [(1.0, 0.0)] * 6 + [(1.0, 1.0)] * 4

    first = paired_bootstrap(pairs, iterations=200, seed=7)
    second = paired_bootstrap(pairs, iterations=200, seed=7)
    candidate_only, baseline_only, p_value = mcnemar_exact(pairs)

    assert first == second
    assert candidate_only == 6
    assert baseline_only == 0
    assert 0 <= p_value <= 1
