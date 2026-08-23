"""Bound conversation history before it is sent to a downstream model.

These dependency-light helpers keep recent history, compacted context, and
retrieved long-term memory within the same character budget.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any

from google.genai import types


@dataclass(frozen=True)
class RunnerContextSettings:
    recent_events: int = 24
    max_context_chars: int = 24_000
    max_event_chars: int = 2_000
    max_text_chars: int = 1_400
    max_tool_result_chars: int = 1_000
    memory_event_chars: int = 4_000


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(low, min(value, high))


def runner_context_settings_from_env() -> RunnerContextSettings:
    return RunnerContextSettings(
        recent_events=_int_env("FUSED_MEMORY_RECENT_EVENTS", 24, low=0, high=120),
        max_context_chars=_int_env(
            "FUSED_MEMORY_CONTEXT_MAX_CHARS",
            24_000,
            low=4_000,
            high=200_000,
        ),
        max_event_chars=_int_env(
            "FUSED_MEMORY_CONTEXT_EVENT_MAX_CHARS",
            2_000,
            low=300,
            high=20_000,
        ),
        max_text_chars=_int_env(
            "FUSED_MEMORY_CONTEXT_TEXT_MAX_CHARS",
            1_400,
            low=200,
            high=20_000,
        ),
        max_tool_result_chars=_int_env(
            "FUSED_MEMORY_CONTEXT_TOOL_RESULT_MAX_CHARS",
            1_000,
            low=200,
            high=20_000,
        ),
        memory_event_chars=_int_env(
            "FUSED_MEMORY_CONTEXT_MEMORY_CHARS",
            4_000,
            low=500,
            high=40_000,
        ),
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _safe_json(value: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return _truncate(_normalize_text(text), max_chars)


def _compact_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate(_normalize_text(value), 220)
    return value


def _id_from_patient(patient: dict[str, Any]) -> str:
    for key in (
        "reference",
        "fhir_reference",
        "synthea_reference",
        "patient_id",
        "fhir_patient_id",
        "id",
    ):
        raw = str(patient.get(key) or "").strip()
        if raw:
            return raw.split("/", 1)[1] if raw.startswith("Patient/") else raw
    return ""


def _compact_patients(patients: list[Any]) -> dict[str, Any]:
    ids: list[str] = []
    names: list[str] = []
    for item in patients[:5]:
        if not isinstance(item, dict):
            continue
        pid = _id_from_patient(item)
        if pid:
            ids.append(pid)
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    out: dict[str, Any] = {"count": len(patients)}
    if ids:
        out["first_ids"] = ids
    if names:
        out["first_names"] = names
    return out


def _compact_mapping(value: dict[str, Any], *, depth: int = 0) -> dict[str, Any]:
    if depth > 2:
        return {"summary": _safe_json(value, max_chars=600)}

    out: dict[str, Any] = {}
    priority = (
        "ok",
        "run_id",
        "status",
        "summary",
        "message",
        "patient_count",
        "patients_with_conditions",
        "population_size_requested",
        "imported_bundles",
        "failed_bundles",
        "resource_counts",
        "patient",
        "patients",
        "timeline",
        "condition_counts",
        "selected_condition_counts",
        "results",
        "items",
        "content",
    )

    def _keep_key(key: str) -> bool:
        lower = key.lower()
        if any(marker in lower for marker in ("blob", "raw", "base64", "bytes")):
            return False
        return lower not in {"html", "xml", "pdf", "document", "bundle"}

    keys = [key for key in priority if key in value and _keep_key(key)]
    keys.extend(key for key in value.keys() if key not in keys and _keep_key(key))

    for key in keys[:16]:
        item = value.get(key)
        if item in (None, "", [], {}):
            continue
        if key == "patients" and isinstance(item, list):
            out[key] = _compact_patients(item)
        elif key == "timeline" and isinstance(item, list):
            out[key] = {
                "count": len(item),
                "first_items": [
                    _compact_mapping(x, depth=depth + 1)
                    for x in item[:3]
                    if isinstance(x, dict)
                ],
            }
        elif isinstance(item, list):
            out[key] = {
                "count": len(item),
                "first_items": [
                    _compact_mapping(x, depth=depth + 1) if isinstance(x, dict) else _compact_scalar(x)
                    for x in item[:3]
                ],
            }
        elif isinstance(item, dict):
            out[key] = _compact_mapping(item, depth=depth + 1)
        else:
            out[key] = _compact_scalar(item)
    return out


def _unwrap_tool_payload(value: Any) -> Any:
    cur = value
    seen = 0
    while isinstance(cur, dict) and seen < 4:
        seen += 1
        for key in ("result", "data", "output"):
            nested = cur.get(key)
            if isinstance(nested, (dict, list, str)):
                cur = nested
                break
        else:
            break

    if isinstance(cur, dict):
        content = cur.get("content")
        if isinstance(content, list):
            parsed_items: list[Any] = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else None
                if not isinstance(text, str):
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed_items.append(text)
                else:
                    parsed_items.append(parsed)
            if len(parsed_items) == 1:
                return parsed_items[0]
            if parsed_items:
                return parsed_items
    return cur


def compact_tool_result(value: Any, *, max_chars: int) -> str:
    payload = _unwrap_tool_payload(value)
    if isinstance(payload, dict):
        compacted: Any = _compact_mapping(payload)
    elif isinstance(payload, list):
        compacted = {
            "count": len(payload),
            "first_items": [
                _compact_mapping(x) if isinstance(x, dict) else _compact_scalar(x)
                for x in payload[:5]
            ],
        }
    else:
        compacted = _compact_scalar(payload)
    return _safe_json(compacted, max_chars=max_chars)


def _part_lines(part: Any, settings: RunnerContextSettings) -> list[str]:
    if getattr(part, "thought", False):
        return []
    lines: list[str] = []
    text = getattr(part, "text", None)
    if text:
        lines.append(_truncate(_normalize_text(text), settings.max_text_chars))

    function_call = getattr(part, "function_call", None)
    if function_call is not None:
        name = getattr(function_call, "name", "tool")
        args = getattr(function_call, "args", None)
        arg_text = _safe_json(args, max_chars=420) if args else "{}"
        lines.append(f"Tool call: {name} args={arg_text}")

    function_response = getattr(part, "function_response", None)
    if function_response is not None:
        name = getattr(function_response, "name", "tool")
        response = getattr(function_response, "response", None)
        lines.append(
            f"Tool result summary: {name} -> "
            f"{compact_tool_result(response, max_chars=settings.max_tool_result_chars)}"
        )
    return lines


def _event_role(event: Any) -> str:
    content = getattr(event, "content", None)
    role = str(getattr(content, "role", "") or "").strip()
    if role:
        return role
    author = str(getattr(event, "author", "") or "").lower()
    return "user" if author == "user" else "model"


def _event_text(event: Any, settings: RunnerContextSettings) -> str:
    if getattr(event, "partial", False):
        return ""
    content = getattr(event, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    lines: list[str] = []
    for part in parts:
        lines.extend(_part_lines(part, settings))
    text = "\n".join(line for line in lines if line).strip()
    return _truncate(text, settings.max_event_chars) if text else ""


def _copy_event_with_text(event: Any, text: str) -> Any:
    role = _event_role(event)
    content = types.Content(role=role, parts=[types.Part.from_text(text=text)])
    if hasattr(event, "model_copy"):
        copied = event.model_copy(deep=True)
    else:
        copied = copy.copy(event)
    copied.content = content
    copied.partial = False
    return copied


def _make_memory_event(reference_event: Any, text: str) -> Any:
    try:
        from google.adk.events import Event

        return Event(
            author="user",
            invocation_id=str(getattr(reference_event, "invocation_id", "") or ""),
            content=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
            id="context-memory",
            timestamp=float(getattr(reference_event, "timestamp", 0.0) or 0.0),
        )
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(
            author="user",
            invocation_id=str(getattr(reference_event, "invocation_id", "") or ""),
            content=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
            partial=False,
            id="context-memory",
        )


def _summary_line(event: Any, text: str) -> str:
    role = _event_role(event)
    author = str(getattr(event, "author", "") or role)
    label = "assistant" if role == "model" else role
    if author and author not in {"user", "model"} and label == "assistant":
        label = author
    return f"- {label}: {_truncate(_normalize_text(text), 500)}"


def _content_summary_line(content: Any, text: str) -> str:
    role = "assistant" if getattr(content, "role", None) == "model" else str(
        getattr(content, "role", None) or "user"
    )
    return f"- {role}: {_truncate(_normalize_text(text), 500)}"


def _memory_line_score(line: str, *, index: int, total: int) -> int:
    """Rank overflow context for a tiny deterministic memory block.

    This is deliberately local and dependency-free: newer user asks and compact
    tool outcomes usually carry the facts needed for follow-ups, while old
    assistant prose is often verbose restatement.
    """
    lower = line.lower()
    score = index
    if lower.startswith("- user:"):
        score += total + 40
    if "tool result summary:" in lower:
        score += total + 35
    if "tool call:" in lower:
        score += total + 20
    if "patient/" in lower or "pmid" in lower or "fhir" in lower:
        score += 18
    if lower.startswith("- assistant:"):
        score -= 20
    return score


def _select_memory_lines(lines: list[str], *, budget: int) -> list[str]:
    if not lines or budget <= 0:
        return []

    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    total = len(lines)
    for index, line in enumerate(lines):
        normalized = _normalize_text(line).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            (_memory_line_score(line, index=index, total=total), index, line)
        )

    selected: list[tuple[int, str]] = []
    used = 0
    for _, index, line in sorted(candidates, key=lambda item: item[0], reverse=True):
        if used + len(line) > budget:
            continue
        selected.append((index, line))
        used += len(line)
        if used >= budget:
            break

    return [line for _, line in sorted(selected, key=lambda item: item[0])]


def compact_runner_events(
    events: list[Any],
    *,
    settings: RunnerContextSettings | None = None,
) -> list[Any]:
    settings = settings or runner_context_settings_from_env()
    if settings.recent_events == 0 or not events:
        return []

    source = list(events)[-settings.recent_events :]
    compacted: list[tuple[Any, str]] = []
    for event in source:
        text = _event_text(event, settings)
        if text:
            compacted.append((_copy_event_with_text(event, text), text))

    if not compacted:
        return []

    total = sum(len(text) for _, text in compacted)
    if total <= settings.max_context_chars:
        return [event for event, _ in compacted]

    memory_budget = min(
        settings.memory_event_chars,
        max(500, settings.max_context_chars // 4),
    )
    kept_budget = max(500, settings.max_context_chars - memory_budget)
    kept_reversed: list[tuple[Any, str]] = []
    kept_chars = 0
    overflow: list[tuple[Any, str]] = []

    for item in reversed(compacted):
        event, text = item
        if kept_chars + len(text) <= kept_budget or not kept_reversed:
            kept_reversed.append(item)
            kept_chars += len(text)
        else:
            overflow.append(item)

    kept = list(reversed(kept_reversed))
    overflow = list(reversed(overflow))
    if not overflow:
        return [event for event, _ in kept]

    memory_lines = _select_memory_lines(
        [_summary_line(event, text) for event, text in overflow],
        budget=memory_budget,
    )

    if not memory_lines:
        memory_lines.append(
            "- Older conversation context was omitted because it exceeded the context budget."
        )

    memory_text = (
        "Compressed prior conversation memory. Use it only as background; the latest user request "
        "and the following recent messages are authoritative.\n"
        + "\n".join(memory_lines)
    )
    return [
        _make_memory_event(overflow[0][0], _truncate(memory_text, memory_budget)),
        *[event for event, _ in kept],
    ]


def _content_text_for_compaction(content: Any, settings: RunnerContextSettings) -> str:
    lines: list[str] = []
    for part in getattr(content, "parts", None) or []:
        lines.extend(_part_lines(part, settings))
    return _truncate(
        "\n".join(line for line in lines if line).strip(),
        settings.max_event_chars,
    )


def _content_exact_text(content: Any) -> str:
    chunks: list[str] = []
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def compact_llm_contents(
    contents: list[Any],
    *,
    current_user_text: str = "",
    settings: RunnerContextSettings | None = None,
) -> list[Any]:
    """Compact ADK ``LlmRequest.contents`` in-place-compatible form.

    This reduces an already assembled model request before downstream
    serialization.
    """

    settings = settings or runner_context_settings_from_env()
    if settings.recent_events == 0 or not contents:
        current = list(contents)[-1:] if contents else []
        return current

    current_norm = _normalize_text(current_user_text)
    source = list(contents)
    current_index = -1
    for idx in range(len(source) - 1, -1, -1):
        content = source[idx]
        if getattr(content, "role", None) != "user":
            continue
        text = _normalize_text(_content_exact_text(content))
        if current_norm and text == current_norm:
            current_index = idx
            break
        if not current_norm:
            current_index = idx
            break

    current_content = source[current_index] if current_index >= 0 else None
    prior = source[:current_index] if current_index >= 0 else source
    prior = prior[-settings.recent_events :]

    compacted: list[tuple[types.Content, str]] = []
    for content in prior:
        text = _content_text_for_compaction(content, settings)
        if not text:
            continue
        role = str(getattr(content, "role", "") or "")
        if role not in {"user", "model", "assistant"}:
            role = "user"
        if role == "assistant":
            role = "model"
        compacted.append(
            (types.Content(role=role, parts=[types.Part.from_text(text=text)]), text)
        )

    tail: list[Any] = [current_content] if current_content is not None else []
    current_chars = len(_content_exact_text(current_content)) if current_content is not None else 0
    budget = max(500, settings.max_context_chars - current_chars)

    total = sum(len(text) for _, text in compacted)
    if total <= budget:
        return [content for content, _ in compacted] + tail

    memory_budget = min(settings.memory_event_chars, max(500, budget // 4))
    kept_budget = max(500, budget - memory_budget)
    kept_reversed: list[tuple[types.Content, str]] = []
    kept_chars = 0
    overflow: list[tuple[types.Content, str]] = []

    for item in reversed(compacted):
        content, text = item
        if kept_chars + len(text) <= kept_budget or not kept_reversed:
            kept_reversed.append(item)
            kept_chars += len(text)
        else:
            overflow.append(item)

    kept = list(reversed(kept_reversed))
    overflow = list(reversed(overflow))
    if not overflow:
        return [content for content, _ in kept] + tail

    memory_lines = _select_memory_lines(
        [_content_summary_line(content, text) for content, text in overflow],
        budget=memory_budget,
    )
    if not memory_lines:
        memory_lines.append(
            "- Older model-request context was omitted because it exceeded the context budget."
        )

    memory = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=_truncate(
                    "Compressed prior conversation memory. Use it only as background; "
                    "the latest user request is authoritative.\n" + "\n".join(memory_lines),
                    memory_budget,
                )
            )
        ],
    )
    return [memory, *[content for content, _ in kept], *tail]
