from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.hybrid_memory import SCHEMA_VERSION
from experiments.hybrid_memory.common import (
    canonical_json,
    deterministic_rows_sha256,
    load_config,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
)


def duplicate_keys(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[tuple[Any, ...]]:
    counts = Counter(tuple(row.get(field) for field in fields) for row in rows)
    return [key for key, count in counts.items() if count > 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument(
        "--context-reproduction-reference",
        type=Path,
        default=None,
        help="Independent context-results JSONL whose deterministic payload must match.",
    )
    parser.add_argument(
        "--require-context-reproduction",
        action="store_true",
        help="Fail unless a distinct, manifest-bound context reproduction is supplied.",
    )
    parser.add_argument("--model-results", type=Path)
    parser.add_argument(
        "--reproduction-reference",
        type=Path,
        default=None,
        help="Reference model-results JSONL for reproduction validation. "
        "Must be a distinct file from --model-results.",
    )
    parser.add_argument(
        "--require-reproduction",
        action="store_true",
        help="Fail validation unless a distinct, complete reproduction reference is supplied.",
    )
    parser.add_argument(
        "--reproduction-mode",
        choices=("exact", "tolerance"),
        default="tolerance",
        help="Exact requires identical answers; tolerance requires the paired accuracy rates "
        "to differ by no more than --reproduction-tolerance.",
    )
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=0.05,
        help="Maximum absolute accuracy-rate difference in tolerance mode.",
    )
    parser.add_argument(
        "--require-model-budget",
        action="append",
        type=int,
        dest="model_budgets",
    )
    parser.add_argument(
        "--require-model-condition",
        action="append",
        dest="model_conditions",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.reproduction_tolerance <= 1.0:
        parser.error("--reproduction-tolerance must be between 0 and 1")
    config = load_config(args.config)
    cases = read_jsonl(args.dataset)
    contexts = read_jsonl(args.contexts)
    model_rows = read_jsonl(args.model_results) if args.model_results else []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    case_ids = {row.get("case_id") for row in cases}
    actual_family_counts = Counter(str(row.get("family")) for row in cases)
    expected_family_counts = {
        str(family): int(count) for family, count in config["case_counts"].items()
    }
    check(
        "dataset_schema",
        all(row.get("schema_version") == SCHEMA_VERSION for row in cases),
        f"cases={len(cases)}",
    )
    check(
        "dataset_unique_case_ids",
        len(case_ids) == len(cases),
        f"unique={len(case_ids)} total={len(cases)}",
    )
    check(
        "dataset_family_counts_match_config",
        dict(actual_family_counts) == expected_family_counts,
        f"actual={dict(actual_family_counts)} expected={expected_family_counts}",
    )
    expected_semantic_profile = str(config.get("semantic_profile", "legacy"))
    semantic_cases = [row for row in cases if row.get("family") == "semantic_paraphrase"]
    check(
        "dataset_semantic_profile_matches_config",
        bool(semantic_cases)
        and all(
            str((row.get("metadata") or {}).get("semantic_profile", "legacy"))
            == expected_semantic_profile
            for row in semantic_cases
        ),
        f"expected={expected_semantic_profile} semantic_cases={len(semantic_cases)}",
    )
    check(
        "dataset_synthetic_only",
        all(row.get("synthetic_only") is True for row in cases)
        and all("Patient/SYN-" in row.get("patient_id", "") for row in cases),
        "all cases must carry synthetic_only=true and Patient/SYN-* identifiers",
    )
    expected_contexts = (
        len(cases) * len(config["conditions"]) * len(config["context_budgets"])
    )
    check(
        "context_result_count",
        len(contexts) == expected_contexts,
        f"actual={len(contexts)} expected={expected_contexts}",
    )
    context_duplicates = duplicate_keys(
        contexts,
        ("case_id", "condition", "budget_chars"),
    )
    check(
        "context_unique_keys",
        not context_duplicates,
        f"duplicates={context_duplicates[:5]}",
    )
    expected_keys = {
        (case_id, condition, int(budget))
        for case_id in case_ids
        for condition in config["conditions"]
        for budget in config["context_budgets"]
    }
    actual_keys = {
        (row.get("case_id"), row.get("condition"), int(row.get("budget_chars", -1)))
        for row in contexts
    }
    check(
        "context_complete_matrix",
        actual_keys == expected_keys,
        f"missing={len(expected_keys - actual_keys)} extra={len(actual_keys - expected_keys)}",
    )
    check(
        "current_prompt_exact",
        all(row.get("current_prompt_exact") is True for row in contexts),
        f"failures={sum(row.get('current_prompt_exact') is not True for row in contexts)}",
    )
    gated = [row for row in contexts if row.get("condition") != "raw_recent"]
    check(
        "gated_conditions_respect_budget",
        all(row.get("budget_compliant") is True for row in gated),
        f"failures={sum(row.get('budget_compliant') is not True for row in gated)}",
    )
    isolation_rows = [
        row
        for row in contexts
        if row.get("family") == "cross_user_isolation"
    ]
    # Determine all conditions present in the isolation family, not a
    # hardcoded subset.  Post-pilot or future conditions are automatically
    # included so scope-isolation is validated exhaustively.
    isolation_conditions_present = sorted(
        {str(row.get("condition")) for row in isolation_rows}
    )
    isolation_empirical_leaks = sum(
        bool(row.get("forbidden_hits")) for row in isolation_rows
    )
    # Structural scope check: every isolation row must carry a non-empty
    # forbidden_facts list so that the deterministic scope gate was actually
    # exercised.  This is independent of whether leaks were empirically observed.
    case_lookup = {row.get("case_id"): row for row in cases}

    def has_foreign_user_probe(context_row: dict[str, Any]) -> bool:
        case = case_lookup.get(context_row.get("case_id")) or {}
        query_user = case.get("query_user_id")
        forbidden = list(context_row.get("forbidden_facts") or [])
        foreign_events = [
            event
            for event in case.get("events", [])
            if event.get("user_id") != query_user
        ]
        return (
            bool(query_user)
            and query_user != case.get("owner_user_id")
            and bool(forbidden)
            and all(
                any(fact in canonical_json(event) for event in foreign_events)
                for fact in forbidden
            )
        )

    isolation_rows_with_probe = [row for row in isolation_rows if has_foreign_user_probe(row)]
    check(
        "cross_user_isolation_structural_scope",
        bool(isolation_rows)
        and len(isolation_rows) == len(isolation_rows_with_probe)
        and set(isolation_conditions_present) == set(config["conditions"]),
        f"rows={len(isolation_rows)} with_probe={len(isolation_rows_with_probe)} "
        f"conditions={isolation_conditions_present}",
    )
    # Empirical leak count: observed forbidden-token hits in the constructed
    # context.  Deterministic clones are counted as observations, not as a
    # population leak-rate estimate.
    check(
        "cross_user_isolation_empirical_zero_leaks",
        bool(isolation_rows) and isolation_empirical_leaks == 0,
        f"leaks={isolation_empirical_leaks}/{len(isolation_rows)} "
        f"(descriptive count, not population rate)",
    )
    dataset_manifest_path = args.dataset.with_suffix(".manifest.json")
    context_manifest_path = args.contexts.with_suffix(".manifest.json")
    manifests_present = dataset_manifest_path.exists() and context_manifest_path.exists()
    check("manifests_present", manifests_present, "dataset and context manifests")
    if dataset_manifest_path.exists():
        dataset_manifest = read_json(dataset_manifest_path)
        check(
            "dataset_manifest_binds_artifacts",
            dataset_manifest.get("stage") == "generate"
            and dataset_manifest.get("dataset_sha256") == sha256_file(args.dataset)
            and dataset_manifest.get("family_counts") == expected_family_counts
            and (dataset_manifest.get("inputs") or {}).get("config", {}).get("sha256")
            == sha256_file(args.config),
            str(dataset_manifest_path),
        )
    if context_manifest_path.exists():
        context_manifest = read_json(context_manifest_path)
        deterministic_hash = deterministic_rows_sha256(
            contexts,
            exclude_fields={"index_ms", "retrieval_ms"},
        )
        check(
            "deterministic_context_hash",
            context_manifest.get("deterministic_result_sha256") == deterministic_hash,
            deterministic_hash,
        )
        context_inputs = context_manifest.get("inputs") or {}
        check(
            "context_manifest_binds_artifacts",
            context_manifest.get("stage") == "context_benchmark"
            and context_manifest.get("dataset_sha256") == sha256_file(args.dataset)
            and context_manifest.get("result_sha256") == sha256_file(args.contexts)
            and context_manifest.get("conditions") == config["conditions"]
            and context_manifest.get("context_budgets") == config["context_budgets"]
            and context_inputs.get("config", {}).get("sha256") == sha256_file(args.config)
            and context_inputs.get("dataset", {}).get("sha256") == sha256_file(args.dataset),
            str(context_manifest_path),
        )

    context_reproduction_status = "not_requested"
    if args.context_reproduction_reference:
        reference_path = args.context_reproduction_reference
        reference_manifest_path = reference_path.with_suffix(".manifest.json")
        distinct = reference_path.resolve() != args.contexts.resolve()
        present = reference_path.exists() and reference_manifest_path.exists()
        check(
            "context_reproduction_reference_present_and_distinct",
            present and distinct,
            f"reference={reference_path} distinct={distinct}",
        )
        if present and distinct:
            reference_rows = read_jsonl(reference_path)
            reference_manifest = read_json(reference_manifest_path)
            reference_keys = {
                (row.get("case_id"), row.get("condition"), row.get("budget_chars"))
                for row in reference_rows
            }
            current_keys = {
                (row.get("case_id"), row.get("condition"), row.get("budget_chars"))
                for row in contexts
            }
            reference_hash = deterministic_rows_sha256(
                reference_rows,
                exclude_fields={"index_ms", "retrieval_ms"},
            )
            current_hash = deterministic_rows_sha256(
                contexts,
                exclude_fields={"index_ms", "retrieval_ms"},
            )
            reference_inputs = reference_manifest.get("inputs") or {}
            bound = (
                reference_manifest.get("stage") == "context_benchmark"
                and reference_manifest.get("dataset_sha256") == sha256_file(args.dataset)
                and reference_manifest.get("result_sha256") == sha256_file(reference_path)
                and reference_manifest.get("deterministic_result_sha256") == reference_hash
                and reference_inputs.get("config", {}).get("sha256")
                == sha256_file(args.config)
                and reference_inputs.get("dataset", {}).get("sha256")
                == sha256_file(args.dataset)
            )
            complete = (
                not duplicate_keys(reference_rows, ("case_id", "condition", "budget_chars"))
                and reference_keys == current_keys
            )
            matched = reference_hash == current_hash
            check(
                "context_reproduction_manifest_bound",
                bound,
                str(reference_manifest_path),
            )
            check(
                "context_reproduction_complete_matrix",
                complete,
                f"reference_rows={len(reference_rows)} expected={len(contexts)}",
            )
            check(
                "context_reproduction_deterministic_hash",
                matched,
                f"current={current_hash} reference={reference_hash}",
            )
            context_reproduction_status = "passed" if bound and complete and matched else "failed"
        else:
            context_reproduction_status = "missing_or_invalid_reference"
    elif args.require_context_reproduction:
        context_reproduction_status = "missing_reference"
        check(
            "context_reproduction_reference_present_and_distinct",
            False,
            "no --context-reproduction-reference supplied",
        )

    reproduction_status = "not_requested"
    if args.require_reproduction and not args.model_results:
        reproduction_status = "missing_model_results"
        check(
            "reproduction_model_results_present",
            False,
            "--require-reproduction also requires --model-results",
        )
    if args.reproduction_reference and not args.model_results:
        reproduction_status = "missing_model_results"
        check(
            "reproduction_model_results_present",
            False,
            "--reproduction-reference cannot be validated without --model-results",
        )
    if args.model_results and not model_rows:
        check(
            "model_results_nonempty",
            False,
            f"model results are empty: {args.model_results}",
        )
    if model_rows:
        model_duplicates = duplicate_keys(
            model_rows,
            ("case_id", "condition", "budget_chars"),
        )
        check(
            "model_unique_keys",
            not model_duplicates,
            f"duplicates={model_duplicates[:5]}",
        )
        check(
            "model_rows_reference_contexts",
            all(
                (row.get("case_id"), row.get("condition"), int(row.get("budget_chars", -1)))
                in actual_keys
                for row in model_rows
            ),
            f"rows={len(model_rows)}",
        )
        required_budgets = args.model_budgets or config["context_budgets"]
        required_conditions = args.model_conditions or config["conditions"]
        required_model_keys = {
            (case_id, condition, int(budget))
            for case_id in case_ids
            for condition in required_conditions
            for budget in required_budgets
        }
        actual_model_keys = {
            (row.get("case_id"), row.get("condition"), int(row.get("budget_chars", -1)))
            for row in model_rows
        }
        check(
            "model_complete_required_matrix",
            actual_model_keys == required_model_keys,
            f"missing={len(required_model_keys - actual_model_keys)} "
            f"extra={len(actual_model_keys - required_model_keys)}",
        )
        check(
            "model_no_silent_failures",
            all(row.get("status") == "ok" for row in model_rows),
            f"errors={sum(row.get('status') != 'ok' for row in model_rows)}",
        )
        check(
            "model_manifest_present",
            args.model_results.with_suffix(".manifest.json").exists(),
            str(args.model_results.with_suffix(".manifest.json")),
        )
        model_manifest_path = args.model_results.with_suffix(".manifest.json")
        model_run_path = args.model_results.with_suffix(".run.json")
        if model_manifest_path.exists():
            model_manifest = read_json(model_manifest_path)
            model_inputs = model_manifest.get("inputs") or {}
            check(
                "model_manifest_binds_artifacts",
                model_manifest.get("stage") == "model_benchmark"
                and model_manifest.get("result_sha256") == sha256_file(args.model_results)
                and model_manifest.get("context_results_sha256") == sha256_file(args.contexts)
                and model_inputs.get("contexts", {}).get("sha256") == sha256_file(args.contexts)
                and model_inputs.get("model_results", {}).get("sha256")
                == sha256_file(args.model_results),
                str(model_manifest_path),
            )
        check(
            "model_run_identity_present",
            model_run_path.exists(),
            str(model_run_path),
        )
        if model_manifest_path.exists() and model_run_path.exists():
            model_manifest = read_json(model_manifest_path)
            model_run = read_json(model_run_path)
            model_run_id = str(model_run.get("run_id") or "")
            check(
                "model_run_identity_binds_rows_and_manifest",
                bool(model_run_id)
                and all(row.get("run_id") == model_run_id for row in model_rows)
                and model_manifest.get("run_id") == model_run_id
                and model_manifest.get("run_identity_sha256") == sha256_file(model_run_path),
                f"run_id={model_run_id or 'missing'}",
            )
        # Model cross-user isolation: check every condition present in model
        # results, not just the hybrid condition.
        model_isolation_rows = [
            row
            for row in model_rows
            if row.get("family") == "cross_user_isolation"
            and row.get("status") == "ok"
        ]
        model_isolation_conditions = sorted(
            {str(row.get("condition")) for row in model_isolation_rows}
        )
        model_isolation_leaks = sum(
            bool(row.get("forbidden_answer_hits")) for row in model_isolation_rows
        )
        check(
            "model_cross_user_isolation_all_conditions",
            bool(model_isolation_rows)
            and model_isolation_leaks == 0
            and set(model_isolation_conditions) == set(required_conditions),
            f"leaks={model_isolation_leaks}/{len(model_isolation_rows)} "
            f"conditions={model_isolation_conditions}",
        )
        # Reproduction validation compares two independently identified runs.
        # A renamed or copied result cannot pass because its run_id is unchanged.
        if args.reproduction_reference:
            ref_path = args.reproduction_reference
            if not ref_path.exists():
                reproduction_status = "missing_reference"
                check(
                    "reproduction_reference_present",
                    False,
                    f"reference file not found: {ref_path}",
                )
            else:
                ref_rows = read_jsonl(ref_path)
                if ref_path.resolve() == args.model_results.resolve():
                    reproduction_status = "invalid_self_comparison"
                    check(
                        "reproduction_not_self_comparison",
                        False,
                        "reference path resolves to the same file as model results",
                    )
                else:
                    primary_manifest_path = args.model_results.with_suffix(".manifest.json")
                    primary_run_path = args.model_results.with_suffix(".run.json")
                    ref_manifest_path = ref_path.with_suffix(".manifest.json")
                    ref_run_path = ref_path.with_suffix(".run.json")
                    provenance_files_present = all(
                        path.exists()
                        for path in (
                            primary_manifest_path,
                            primary_run_path,
                            ref_manifest_path,
                            ref_run_path,
                        )
                    )
                    check(
                        "reproduction_provenance_files_present",
                        provenance_files_present,
                        f"primary_manifest={primary_manifest_path.exists()} "
                        f"primary_run={primary_run_path.exists()} "
                        f"reference_manifest={ref_manifest_path.exists()} "
                        f"reference_run={ref_run_path.exists()}",
                    )
                    ref_lookup = {
                        (r.get("case_id"), r.get("condition"), int(r.get("budget_chars", -1))): r
                        for r in ref_rows
                    }
                    model_lookup = {
                        (r.get("case_id"), r.get("condition"), int(r.get("budget_chars", -1))): r
                        for r in model_rows
                    }
                    ref_duplicates = duplicate_keys(
                        ref_rows, ("case_id", "condition", "budget_chars")
                    )
                    reference_all_ok = bool(ref_rows) and all(
                        row.get("status") == "ok" for row in ref_rows
                    )
                    candidate_all_ok = bool(model_rows) and all(
                        row.get("status") == "ok" for row in model_rows
                    )
                    check(
                        "reproduction_reference_unique_and_complete",
                        not ref_duplicates and reference_all_ok,
                        f"duplicates={ref_duplicates[:5]} "
                        f"errors={sum(row.get('status') != 'ok' for row in ref_rows)}",
                    )
                    matched_keys = set(ref_lookup) & set(model_lookup)
                    reproduced = 0
                    exact_answers = 0
                    context_matches = 0
                    total = 0
                    for key in sorted(matched_keys):
                        ref_row = ref_lookup[key]
                        model_row = model_lookup[key]
                        if ref_row.get("status") == "ok" and model_row.get("status") == "ok":
                            total += 1
                            if bool(ref_row.get("answer_correct")) == bool(
                                model_row.get("answer_correct")
                            ):
                                reproduced += 1
                            if ref_row.get("answer") == model_row.get("answer"):
                                exact_answers += 1
                            if ref_row.get("context_sha256") == model_row.get("context_sha256"):
                                context_matches += 1
                    reproduction_rate = reproduced / total if total else 0.0
                    complete_matrix = set(ref_lookup) == set(model_lookup)
                    candidate_accuracy = (
                        sum(bool(row.get("answer_correct")) for row in model_rows)
                        / len(model_rows)
                        if candidate_all_ok
                        else 0.0
                    )
                    reference_accuracy = (
                        sum(bool(row.get("answer_correct")) for row in ref_rows)
                        / len(ref_rows)
                        if reference_all_ok
                        else 0.0
                    )
                    accuracy_delta = abs(candidate_accuracy - reference_accuracy)
                    provenance_ok = False
                    distinct_runs = False
                    immutable_model = False
                    if provenance_files_present:
                        primary_manifest = read_json(primary_manifest_path)
                        ref_manifest = read_json(ref_manifest_path)
                        primary_run = read_json(primary_run_path)
                        ref_run = read_json(ref_run_path)
                        primary_run_id = str(primary_run.get("run_id") or "")
                        ref_run_id = str(ref_run.get("run_id") or "")
                        distinct_runs = bool(primary_run_id and ref_run_id and primary_run_id != ref_run_id)
                        immutable_model = bool(
                            primary_run.get("model_file_sha256")
                            and primary_run.get("model_file_sha256")
                            == ref_run.get("model_file_sha256")
                        )
                        frozen_keys = (
                            "base_url",
                            "model",
                            "model_file_sha256",
                            "temperature",
                            "prompt_version",
                            "request_order_seed",
                            "generation_seed",
                            "context_results_sha256",
                        )
                        rows_match_run = all(
                            row.get("run_id") == primary_run_id for row in model_rows
                        ) and all(row.get("run_id") == ref_run_id for row in ref_rows)
                        manifests_match_run = (
                            primary_manifest.get("run_id") == primary_run_id
                            and ref_manifest.get("run_id") == ref_run_id
                            and primary_manifest.get("run_identity_sha256")
                            == sha256_file(primary_run_path)
                            and ref_manifest.get("run_identity_sha256")
                            == sha256_file(ref_run_path)
                        )
                        ref_inputs = ref_manifest.get("inputs") or {}
                        reference_manifest_binds_artifacts = (
                            ref_manifest.get("stage") == "model_benchmark"
                            and ref_manifest.get("result_sha256") == sha256_file(ref_path)
                            and ref_manifest.get("context_results_sha256")
                            == sha256_file(args.contexts)
                            and ref_inputs.get("contexts", {}).get("sha256")
                            == sha256_file(args.contexts)
                            and ref_inputs.get("model_results", {}).get("sha256")
                            == sha256_file(ref_path)
                        )
                        provenance_ok = (
                            distinct_runs
                            and immutable_model
                            and rows_match_run
                            and manifests_match_run
                            and reference_manifest_binds_artifacts
                            and all(primary_run.get(key) == ref_run.get(key) for key in frozen_keys)
                        )
                    check(
                        "reproduction_independent_run_provenance",
                        provenance_ok,
                        f"distinct_runs={distinct_runs} immutable_model={immutable_model}",
                    )
                    if args.reproduction_mode == "exact":
                        outcome_ok = reproduced == total and exact_answers == total
                    else:
                        outcome_ok = accuracy_delta <= args.reproduction_tolerance
                    reproduction_status = (
                        "passed"
                        if total > 0
                        and complete_matrix
                        and not ref_duplicates
                        and reference_all_ok
                        and candidate_all_ok
                        and total == len(model_rows) == len(ref_rows)
                        and context_matches == total
                        and provenance_ok
                        and outcome_ok
                        else "failed"
                    )
                    check(
                        "reproduction_not_self_comparison",
                        True,
                        "reference is distinct from model results",
                    )
                    check(
                        "reproduction_answer_correctness",
                        reproduction_status == "passed",
                        f"mode={args.reproduction_mode} complete_matrix={complete_matrix} "
                        f"matched={total} correctness_agreement={reproduced} "
                        f"exact_answers={exact_answers} context_hashes={context_matches} "
                        f"agreement_rate={reproduction_rate:.4f} "
                        f"accuracy_delta={accuracy_delta:.4f} "
                        f"tolerance={args.reproduction_tolerance:.4f}",
                    )
        elif args.require_reproduction:
            reproduction_status = "missing_reference"
            check(
                "reproduction_reference_present",
                False,
                "no --reproduction-reference supplied; reproduction status is missing",
            )

    passed = all(item["passed"] for item in checks)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "passed" if passed else "failed",
        "context_reproduction_status": context_reproduction_status,
        "reproduction_status": reproduction_status,
        "checks": checks,
        "artifacts": {
            "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
            "dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset)},
            "contexts": {"path": str(args.contexts), "sha256": sha256_file(args.contexts)},
        },
    }
    if args.model_results:
        report["artifacts"]["model_results"] = {
            "path": str(args.model_results),
            "sha256": sha256_file(args.model_results),
        }
    if args.context_reproduction_reference:
        report["artifacts"]["context_reproduction_reference"] = {
            "path": str(args.context_reproduction_reference),
            "sha256": sha256_file(args.context_reproduction_reference)
            if args.context_reproduction_reference.exists()
            else None,
        }
    write_json(args.output, report)
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
