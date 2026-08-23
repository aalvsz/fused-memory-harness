from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.hybrid_memory import SCHEMA_VERSION


CONDITIONS = (
    "latest_only",
    "raw_recent",
    "short_term",
    "long_term",
    "hybrid",
    "bm25_long_term",
    "bm25_hybrid",
    "dense_long_term",
    "dense_hybrid",
    "dense_guarded_hybrid",
    "fused_hybrid",
    "fused_unguarded_hybrid",
    "matched_sparse_hybrid",
    "union_hybrid",
    "rrf_hybrid",
    "cascade_hybrid",
)

FAMILIES = (
    "same_session_overflow",
    "cross_session_recall",
    "semantic_paraphrase",
    "temporal_update",
    "abstention",
    "cross_user_isolation",
    "cross_patient_disambiguation",
    "tool_evidence",
    "memory_prompt_injection",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deterministic_rows_sha256(
    rows: Iterable[dict[str, Any]],
    *,
    exclude_fields: set[str] | None = None,
) -> str:
    excluded = exclude_fields or set()
    payload = "\n".join(
        canonical_json({key: value for key, value in row.items() if key not in excluded})
        for row in rows
    )
    return sha256_bytes((payload + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def environment_manifest(*, command: list[str], inputs: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "command": command,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(inputs.items())
        },
    }


def normalize_for_match(text: str) -> str:
    return " ".join(
        text.casefold()
        .replace(",", "")
        .replace(";", " ")
        .replace(":", " ")
        .replace("−", "-")
        .split()
    )


def fact_hits(text: str, facts: list[str]) -> list[str]:
    normalized = normalize_for_match(text)
    hits: list[str] = []
    for fact in facts:
        target = normalize_for_match(fact)
        if not target:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])"
        if re.search(pattern, normalized):
            hits.append(fact)
    return hits


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {SCHEMA_VERSION!r}, "
            f"got {config.get('schema_version')!r}"
        )
    counts = config.get("case_counts")
    if not isinstance(counts, dict) or set(counts) != set(FAMILIES):
        raise ValueError(f"{path}: case_counts must define exactly {FAMILIES}")
    if any(not isinstance(value, int) or value < 1 for value in counts.values()):
        raise ValueError(f"{path}: every case count must be a positive integer")
    budgets = config.get("context_budgets")
    if not isinstance(budgets, list) or not budgets or any(
        not isinstance(value, int) or value < 2_000 for value in budgets
    ):
        raise ValueError(f"{path}: context_budgets must contain integers >= 2000")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"{path}: conditions must be a non-empty list")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"{path}: unknown conditions {unknown}; must be subset of {CONDITIONS}")
    return config
