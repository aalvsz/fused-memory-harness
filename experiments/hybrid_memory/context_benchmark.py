from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from google.genai import types

from experiments.hybrid_memory import SCHEMA_VERSION
from experiments.hybrid_memory.bm25_memory import (
    as_content as bm25_content,
    retrieve as bm25_retrieve,
    store as bm25_store,
)
from experiments.hybrid_memory.common import (
    canonical_json,
    deterministic_rows_sha256,
    environment_manifest,
    fact_hits,
    load_config,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from fused_memory_harness.runtime.legacy_memory import (
    LongTermMemorySettings,
    retrieve_memories,
    retrieved_memory_content,
    store_event_memory,
)
from fused_memory_harness.runtime.context_compaction import (
    RunnerContextSettings,
    compact_llm_contents,
    compact_runner_events,
)


MATCHED_MODE_BY_CONDITION = {
    "dense_long_term": "dense",
    "dense_hybrid": "dense",
    "fused_hybrid": "fused",
    "matched_sparse_hybrid": "bm25",
    "union_hybrid": "union",
    "rrf_hybrid": "rrf",
    "cascade_hybrid": "cascade",
}


def event_object(row: dict[str, Any]) -> Any:
    if row["kind"] == "tool_result":
        part = types.Part.from_function_response(
            name=row["tool_name"],
            response=row["tool_payload"],
        )
    else:
        part = types.Part.from_text(text=row["text"])
    role = row["role"]
    content_role = "model" if role in {"model", "assistant"} else "user"
    return SimpleNamespace(
        id=row["event_id"],
        author=role,
        invocation_id=f"invocation-{row['event_id']}",
        timestamp=float(row["timestamp"]),
        partial=False,
        content=types.Content(role=content_role, parts=[part]),
    )


def raw_event_text(row: dict[str, Any]) -> str:
    if row["kind"] == "tool_result":
        return f"Tool result: {row['tool_name']} -> {canonical_json(row['tool_payload'])}"
    return str(row["text"])


def raw_contents(rows: list[dict[str, Any]], *, recent_events: int) -> list[types.Content]:
    selected = rows[-recent_events:] if recent_events else []
    contents: list[types.Content] = []
    for row in selected:
        role = "model" if row["role"] in {"model", "assistant"} else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=raw_event_text(row))])
        )
    return contents


def content_text(content: Any) -> str:
    return "\n".join(
        str(getattr(part, "text", "") or "")
        for part in (getattr(content, "parts", None) or [])
        if not getattr(part, "thought", False)
    ).strip()


def serialized_messages(contents: list[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for content in contents:
        text = content_text(content)
        if not text:
            continue
        role = str(getattr(content, "role", "") or "user")
        messages.append({"role": "assistant" if role == "model" else "user", "content": text})
    return messages


def settings_for_budget(config: dict[str, Any], budget: int) -> RunnerContextSettings:
    return RunnerContextSettings(
        recent_events=int(config["recent_events"]),
        max_context_chars=budget,
        max_event_chars=int(config["max_event_chars"]),
        max_text_chars=int(config["max_text_chars"]),
        max_tool_result_chars=int(config["max_tool_result_chars"]),
        memory_event_chars=min(int(config["memory_event_chars"]), max(500, budget // 4)),
    )


def build_condition(
    case: dict[str, Any],
    *,
    condition: str,
    budget: int,
    config: dict[str, Any],
    work_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    settings = settings_for_budget(config, budget)
    current = types.Content(
        role="user",
        parts=[types.Part.from_text(text=case["query"])],
    )
    same_session_rows = [
        row
        for row in case["events"]
        if row["user_id"] == case["query_user_id"]
        and row["session_id"] == case["query_session_id"]
    ]
    compacted_session_events = compact_runner_events(
        [event_object(row) for row in same_session_rows],
        settings=settings,
    )
    session_contents = [
        event.content
        for event in compacted_session_events
        if getattr(event, "content", None) is not None
    ]

    index_ms = 0.0
    retrieval_ms = 0.0
    retrieved_count = 0
    memory_content = None
    if condition in {"long_term", "hybrid"}:
        memory_settings = LongTermMemorySettings(
            enabled=True,
            database_path=work_dir / f"{condition}-{budget}.sqlite3",
            max_entries=max(100, len(case["events"]) + 10),
            max_entry_chars=int(config["long_term_entry_max_chars"]),
            retrieve_top_k=int(config["long_term_top_k"]),
            retrieve_min_score=float(config["long_term_min_score"]),
            retrieve_max_chars=int(config["long_term_max_chars"]),
            vector_dimensions=int(config["long_term_vector_dims"]),
            include_cross_session=True,
        )
        started = time.perf_counter()
        for row in case["events"]:
            store_event_memory(
                event_object(row),
                app_name=case["app_name"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                settings=memory_settings,
            )
        index_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        entries = retrieve_memories(
            case["query"],
            app_name=case["app_name"],
            user_id=case["query_user_id"],
            session_id=case["query_session_id"],
            settings=memory_settings,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        retrieved_count = len(entries)
        memory_content = retrieved_memory_content(
            entries,
            max_chars=memory_settings.retrieve_max_chars,
        )
    elif condition in {"bm25_long_term", "bm25_hybrid"}:
        database_path = work_dir / f"{condition}-{budget}.sqlite3"
        started = time.perf_counter()
        for row in case["events"]:
            bm25_store(
                database_path,
                event_object(row),
                app_name=case["app_name"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                max_chars=int(config["long_term_entry_max_chars"]),
            )
        index_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        entries = bm25_retrieve(
            database_path,
            case["query"],
            app_name=case["app_name"],
            user_id=case["query_user_id"],
            top_k=int(config["long_term_top_k"]),
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        retrieved_count = len(entries)
        memory_content = bm25_content(
            entries,
            max_chars=int(config["long_term_max_chars"]),
        )
    elif condition in {
        "dense_long_term",
        "dense_hybrid",
        "fused_hybrid",
        "matched_sparse_hybrid",
        "union_hybrid",
        "rrf_hybrid",
        "cascade_hybrid",
    }:
        # Improved semantic/fused retriever. One FusedMemoryIndex per
        # (condition, budget) so cases do not contaminate each other; the
        # registry is rebuilt per case via reset_registry() in the store loop.
        from experiments.hybrid_memory.fused_memory import (
            as_content as fused_content,
            reset_registry,
            retrieve as fused_retrieve,
            store as fused_store,
        )
        reset_registry()
        mode = MATCHED_MODE_BY_CONDITION[condition]
        started = time.perf_counter()
        for row in case["events"]:
            fused_store(
                event_object(row),
                app_name=case["app_name"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                max_chars=int(config["long_term_entry_max_chars"]),
                condition=condition,
                budget=budget,
                config=config,
            )
        index_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        entries = fused_retrieve(
            case["query"],
            app_name=case["app_name"],
            user_id=case["query_user_id"],
            session_id=case["query_session_id"],
            top_k=int(config["long_term_top_k"]),
            mode=mode,
            condition=condition,
            budget=budget,
            config=config,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        retrieved_count = len(entries)
        memory_content = fused_content(
            entries,
            max_chars=int(config["long_term_max_chars"]),
        )

    if condition == "latest_only":
        final_contents = [current]
    elif condition == "raw_recent":
        final_contents = [
            *raw_contents(same_session_rows, recent_events=settings.recent_events),
            current,
        ]
    elif condition == "short_term":
        final_contents = compact_llm_contents(
            [*session_contents, current],
            current_user_text=case["query"],
            settings=settings,
        )
    elif condition in {"long_term", "bm25_long_term", "dense_long_term"}:
        long_term_settings = RunnerContextSettings(
            recent_events=1,
            max_context_chars=settings.max_context_chars,
            max_event_chars=settings.max_event_chars,
            max_text_chars=settings.max_text_chars,
            max_tool_result_chars=settings.max_tool_result_chars,
            memory_event_chars=settings.memory_event_chars,
        )
        final_contents = compact_llm_contents(
            [*([memory_content] if memory_content is not None else []), current],
            current_user_text=case["query"],
            settings=long_term_settings,
        )
    elif condition in {
        "hybrid",
        "bm25_hybrid",
        "dense_hybrid",
        "fused_hybrid",
        "matched_sparse_hybrid",
        "union_hybrid",
        "rrf_hybrid",
        "cascade_hybrid",
    }:
        final_contents = compact_llm_contents(
            [
                *([memory_content] if memory_content is not None else []),
                *session_contents,
                current,
            ],
            current_user_text=case["query"],
            settings=settings,
        )
    else:
        raise ValueError(f"unknown condition: {condition}")

    messages = serialized_messages(final_contents)
    return messages, {
        "index_ms": index_ms,
        "retrieval_ms": retrieval_ms,
        "retrieved_count": retrieved_count,
        "same_session_event_count": len(same_session_rows),
    }


def evaluate_context(
    case: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    condition: str,
    budget: int,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    context_text = "\n".join(message["content"] for message in messages)
    history_text = "\n".join(raw_event_text(row) for row in case["events"])
    required = list(case["required_facts"])
    forbidden = list(case["forbidden_facts"])
    required_hits = fact_hits(context_text, required)
    forbidden_hits = fact_hits(context_text, forbidden)
    context_success = (
        len(required_hits) == len(required)
        if case["answerable"]
        else (not forbidden_hits if case["isolation_test"] else True)
    )
    metadata = case.get("metadata") or {}
    semantic_concept_id = metadata.get("semantic_concept_id")
    independent_unit_id = (
        f"semantic_concept:{semantic_concept_id}"
        if semantic_concept_id
        else str(case["case_id"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case["case_id"],
        "independent_unit_id": independent_unit_id,
        "independent_unit_kind": (
            "semantic_concept" if semantic_concept_id else "case"
        ),
        "family": case["family"],
        "condition": condition,
        "budget_chars": budget,
        "query": case["query"],
        "answerable": bool(case["answerable"]),
        "isolation_test": bool(case["isolation_test"]),
        "required_facts": required,
        "answer_facts": list(case["answer_facts"]),
        "forbidden_facts": forbidden,
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "all_required_present": len(required_hits) == len(required),
        "no_forbidden_present": not forbidden_hits,
        "context_success": context_success,
        "current_prompt_exact": bool(messages and messages[-1]["content"] == case["query"]),
        "context_chars": len(context_text),
        "history_chars": len(history_text),
        "compression_ratio": len(context_text) / max(1, len(history_text)),
        "budget_compliant": len(context_text) <= budget,
        "messages": messages,
        "context_sha256": sha256_bytes(context_text.encode("utf-8")),
        **diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    cases = read_jsonl(args.dataset)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hybrid-memory-context-") as tmp:
        root = Path(tmp)
        for case in cases:
            for budget in config["context_budgets"]:
                for condition in config["conditions"]:
                    case_dir = root / case["case_id"]
                    case_dir.mkdir(parents=True, exist_ok=True)
                    messages, diagnostics = build_condition(
                        case,
                        condition=condition,
                        budget=int(budget),
                        config=config,
                        work_dir=case_dir,
                    )
                    results.append(
                        evaluate_context(
                            case,
                            messages,
                            condition=condition,
                            budget=int(budget),
                            diagnostics=diagnostics,
                        )
                    )
    write_jsonl(args.output, results)
    manifest = environment_manifest(
        command=sys.argv,
        inputs={
            "config": args.config,
            "dataset": args.dataset,
            "context_results": args.output,
        },
    )
    manifest.update(
        {
            "stage": "context_benchmark",
            "dataset_sha256": sha256_file(args.dataset),
            "result_sha256": sha256_file(args.output),
            "deterministic_result_sha256": deterministic_rows_sha256(
                results,
                exclude_fields={"index_ms", "retrieval_ms"},
            ),
            "nondeterministic_telemetry_fields": ["index_ms", "retrieval_ms"],
            "case_count": len(cases),
            "result_count": len(results),
            "conditions": config["conditions"],
            "context_budgets": config["context_budgets"],
            "synthetic_only": True,
        }
    )
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    passed = sum(bool(row["context_success"]) for row in results)
    print(f"results={len(results)} context_success={passed}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
