"""Optional long-term memory retrieval for model prompts.

This module is intentionally dependency-free.  It stores compact, text-only
memory chunks in SQLite with a deterministic hashed sparse vector, then retrieves
nearest chunks with cosine similarity plus simple metadata filters.  It is not a
replacement for the bounded prompt compactor; retrieved memories are always fed
back through that final budget gate before they reach the model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai import types

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{1,}")
_IDENTIFIER_RE = re.compile(
    r"\b(?:Patient/[A-Za-z0-9_.-]+|PMID[:\s]*\d+|topic\s+[A-Z0-9]+|FHIR|CKD)\b",
    re.I,
)


@dataclass(frozen=True)
class LongTermMemorySettings:
    enabled: bool = False
    database_path: Path = PACKAGE_ROOT / ".cache/fused_memory_legacy.sqlite3"
    max_entries: int = 8000
    max_entry_chars: int = 900
    retrieve_top_k: int = 5
    retrieve_min_score: float = 0.08
    retrieve_max_chars: int = 1800
    vector_dimensions: int = 512
    include_cross_session: bool = True


@dataclass(frozen=True)
class MemoryEntry:
    text: str
    score: float
    source_kind: str
    session_id: str
    created_at: float


def _bool_env(name: str, default: bool) -> bool:
    raw = _env_value(name).strip().lower()
    if not raw:
        return default
    return raw in TRUE_ENV_VALUES


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    raw = _env_value(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(low, min(value, high))


def _float_env(name: str, default: float, *, low: float, high: float) -> float:
    raw = _env_value(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(low, min(value, high))


def _env_value(name: str, default: str = "") -> str:
    """Read a direct harness environment variable."""
    value = os.environ.get(name)
    return str(value) if value is not None and str(value).strip() else default


def long_term_memory_settings_from_env() -> LongTermMemorySettings:
    path = _env_value("FUSED_MEMORY_DATABASE_PATH").strip()
    return LongTermMemorySettings(
        enabled=_bool_env("FUSED_MEMORY_ENABLED", False),
        database_path=(
            Path(path).expanduser()
            if path
            else PACKAGE_ROOT / ".cache/fused_memory_legacy.sqlite3"
        ),
        max_entries=_int_env("FUSED_MEMORY_MAX_ENTRIES", 8000, low=100, high=200_000),
        max_entry_chars=_int_env("FUSED_MEMORY_ENTRY_MAX_CHARS", 900, low=160, high=10_000),
        retrieve_top_k=_int_env("FUSED_MEMORY_TOP_K", 5, low=1, high=30),
        retrieve_min_score=_float_env("FUSED_MEMORY_MIN_SCORE", 0.08, low=0.0, high=1.0),
        retrieve_max_chars=_int_env("FUSED_MEMORY_MAX_CHARS", 1800, low=200, high=20_000),
        vector_dimensions=_int_env("FUSED_MEMORY_VECTOR_DIMS", 512, low=64, high=4096),
        include_cross_session=_bool_env("FUSED_MEMORY_CROSS_SESSION", True),
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for token in _TOKEN_RE.findall(text.casefold()):
        if len(token) < 2:
            continue
        out.append(token)
        if "/" in token:
            out.extend(part for part in token.split("/") if len(part) >= 2)
        if "-" in token:
            out.extend(part for part in token.split("-") if len(part) >= 2)
    return out


def _hashed_vector(text: str, dimensions: int) -> dict[int, float]:
    vector: dict[int, float] = {}
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] = vector.get(idx, 0.0) + sign
    norm = math.sqrt(sum(v * v for v in vector.values()))
    if norm:
        vector = {idx: value / norm for idx, value in vector.items()}
    return vector


def _vector_json(text: str, dimensions: int) -> str:
    vector = _hashed_vector(text, dimensions)
    return json.dumps(vector, separators=(",", ":"), sort_keys=True)


def _load_vector(value: str) -> dict[int, float]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for key, item in raw.items():
        try:
            out[int(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return out


def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(idx, 0.0) for idx, value in a.items())


def _safe_json(value: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return _truncate(_normalize_text(text), max_chars)


def _part_text(part: Any, *, max_chars: int) -> str:
    if getattr(part, "thought", False):
        return ""
    text = getattr(part, "text", None)
    if text:
        return _truncate(_normalize_text(text), max_chars)
    function_call = getattr(part, "function_call", None)
    if function_call is not None:
        name = getattr(function_call, "name", "tool")
        args = getattr(function_call, "args", None)
        return f"Tool call: {name} args={_safe_json(args or {}, max_chars=360)}"
    function_response = getattr(part, "function_response", None)
    if function_response is not None:
        from fused_memory_harness.runtime.context_compaction import compact_tool_result

        name = getattr(function_response, "name", "tool")
        response = getattr(function_response, "response", None)
        return f"Tool result: {name} summary={compact_tool_result(response, max_chars=500)}"
    return ""


def compact_event_text_for_memory(event: Any, *, max_chars: int) -> str:
    if getattr(event, "partial", False):
        return ""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    lines = [_part_text(part, max_chars=max_chars) for part in parts]
    text = "\n".join(line for line in lines if line).strip()
    return _truncate(text, max_chars)


def _source_kind(event: Any, text: str) -> str:
    author = str(getattr(event, "author", "") or "").lower()
    if "tool result:" in text.lower() or author == "tool":
        return "tool"
    role = str(getattr(getattr(event, "content", None), "role", "") or "").lower()
    if role == "user" or author == "user":
        return "user"
    return "assistant"


def _importance(text: str, source_kind: str) -> float:
    score = 1.0
    if source_kind == "user":
        score += 1.2
    elif source_kind == "tool":
        score += 1.0
    else:
        score += 0.2
    score += min(1.0, len(_IDENTIFIER_RE.findall(text)) * 0.25)
    if len(text) > 600 and source_kind == "assistant":
        score -= 0.3
    return max(0.1, score)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entries (
            id TEXT PRIMARY KEY,
            app_name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            invocation_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            text TEXT NOT NULL,
            vector TEXT NOT NULL,
            importance REAL NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_entries(app_name, user_id, session_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_entries(app_name, user_id, created_at)"
    )


def _connect(settings: LongTermMemorySettings) -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.database_path))
    _ensure_schema(conn)
    return conn


def store_event_memory(
    event: Any,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    settings: LongTermMemorySettings | None = None,
) -> None:
    settings = settings or long_term_memory_settings_from_env()
    if not settings.enabled:
        return

    text = compact_event_text_for_memory(event, max_chars=settings.max_entry_chars)
    if len(text) < 12:
        return

    invocation_id = str(getattr(event, "invocation_id", "") or "")
    event_id = str(getattr(event, "id", "") or "")
    source_kind = _source_kind(event, text)
    digest_source = f"{app_name}\0{user_id}\0{session_id}\0{invocation_id}\0{event_id}\0{text}"
    memory_id = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
    created_at = float(getattr(event, "timestamp", 0.0) or time.time())

    with _connect(settings) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_entries
            (id, app_name, user_id, session_id, invocation_id, source_kind, text, vector, importance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                app_name,
                user_id,
                session_id,
                invocation_id,
                source_kind,
                text,
                _vector_json(text, settings.vector_dimensions),
                _importance(text, source_kind),
                created_at,
            ),
        )
        if settings.max_entries > 0:
            conn.execute(
                """
                DELETE FROM memory_entries
                WHERE id IN (
                    SELECT id FROM memory_entries
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (settings.max_entries,),
            )
        conn.commit()


def retrieve_memories(
    query: str,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    settings: LongTermMemorySettings | None = None,
) -> list[MemoryEntry]:
    settings = settings or long_term_memory_settings_from_env()
    if not settings.enabled or not query.strip() or not settings.database_path.exists():
        return []

    query_vector = _hashed_vector(query, settings.vector_dimensions)
    if not query_vector:
        return []

    if settings.include_cross_session:
        where = "app_name = ? AND user_id = ?"
        params: tuple[Any, ...] = (app_name, user_id)
    else:
        where = "app_name = ? AND user_id = ? AND session_id = ?"
        params = (app_name, user_id, session_id)

    with _connect(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT text, vector, source_kind, session_id, importance, created_at
            FROM memory_entries
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT 600
            """,
            params,
        ).fetchall()

    scored: list[MemoryEntry] = []
    seen: set[str] = set()
    for text, vector_json, source_kind, row_session_id, importance, created_at in rows:
        normalized = _normalize_text(text).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        similarity = _cosine(query_vector, _load_vector(vector_json))
        score = similarity * float(importance or 1.0)
        if score < settings.retrieve_min_score:
            continue
        scored.append(
            MemoryEntry(
                text=text,
                score=score,
                source_kind=str(source_kind or "memory"),
                session_id=str(row_session_id or ""),
                created_at=float(created_at or 0.0),
            )
        )

    return sorted(scored, key=lambda item: (item.score, item.created_at), reverse=True)[
        : settings.retrieve_top_k
    ]


def retrieved_memory_content(entries: list[MemoryEntry], *, max_chars: int) -> types.Content | None:
    lines: list[str] = []
    used = 0
    for entry in entries:
        prefix = f"- {entry.source_kind}"
        if entry.session_id:
            prefix += f" session={entry.session_id}"
        line = f"{prefix}: {_truncate(_normalize_text(entry.text), 420)}"
        if used + len(line) > max_chars:
            continue
        lines.append(line)
        used += len(line)
    if not lines:
        return None
    text = (
        "Retrieved long-term memory. Use only when relevant to the latest user request; "
        "do not override newer session context.\n"
        + "\n".join(lines)
    )
    return types.Content(role="user", parts=[types.Part.from_text(text=_truncate(text, max_chars))])
