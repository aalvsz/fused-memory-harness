from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from google.genai import types

from fused_memory_harness.runtime.legacy_memory import compact_event_text_for_memory


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{1,}")
STOPWORDS = {
    "about",
    "across",
    "after",
    "answer",
    "current",
    "earlier",
    "exactly",
    "give",
    "only",
    "our",
    "please",
    "recorded",
    "returned",
    "the",
    "this",
    "use",
    "what",
    "when",
}


def source_kind(event: Any) -> str:
    author = str(getattr(event, "author", "") or "").casefold()
    role = str(getattr(getattr(event, "content", None), "role", "") or "").casefold()
    if author == "tool":
        return "tool"
    if author == "user" or role == "user":
        return "user"
    return "assistant"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            app_name UNINDEXED,
            user_id UNINDEXED,
            session_id UNINDEXED,
            source_kind UNINDEXED,
            created_at UNINDEXED,
            text,
            tokenize='unicode61'
        )
        """
    )
    return connection


def store(
    path: Path,
    event: Any,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    max_chars: int,
) -> None:
    text = compact_event_text_for_memory(event, max_chars=max_chars)
    if len(text) < 12:
        return
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO memory_fts
            (app_name, user_id, session_id, source_kind, created_at, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                app_name,
                user_id,
                session_id,
                source_kind(event),
                float(getattr(event, "timestamp", 0.0) or 0.0),
                text,
            ),
        )
        connection.commit()


def query_expression(query: str) -> str:
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(query.casefold()):
        token = raw.strip("./:-_")
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append('"' + token.replace('"', '""') + '"')
    return " OR ".join(tokens[:32])


def retrieve(
    path: Path,
    query: str,
    *,
    app_name: str,
    user_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    expression = query_expression(query)
    if not path.exists() or not expression:
        return []
    with connect(path) as connection:
        rows = connection.execute(
            """
            SELECT text, source_kind, session_id, created_at, bm25(memory_fts) AS score
            FROM memory_fts
            WHERE memory_fts MATCH ? AND app_name = ? AND user_id = ?
            ORDER BY score ASC, created_at DESC
            LIMIT ?
            """,
            (expression, app_name, user_id, top_k),
        ).fetchall()
    return [
        {
            "text": str(text),
            "source_kind": str(kind),
            "session_id": str(session_id),
            "created_at": float(created_at),
            "score": float(score),
        }
        for text, kind, session_id, created_at, score in rows
    ]


def as_content(entries: list[dict[str, Any]], *, max_chars: int) -> types.Content | None:
    lines: list[str] = []
    used = 0
    for entry in entries:
        line = (
            f"- {entry['source_kind']} session={entry['session_id']}: "
            f"{' '.join(entry['text'].split())[:420]}"
        )
        if used + len(line) > max_chars:
            continue
        lines.append(line)
        used += len(line)
    if not lines:
        return None
    text = (
        "Retrieved BM25 baseline memory. Use only when relevant to the latest user request; "
        "do not override newer session context.\n"
        + "\n".join(lines)
    )
    return types.Content(
        role="user",
        parts=[types.Part.from_text(text=text[:max_chars])],
    )
