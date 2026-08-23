"""Fused hybrid retrieval for long-term agent memory.

This module implements the improved memory technique evaluated in the hybrid
memory experiment. It is designed to compare against both the legacy hashed-
vector retriever (``legacy_memory.py``) and the SQLite FTS5/BM25 baseline
(``bm25_memory.py``) on the synthetic benchmark.

Design
------
* **Dense semantic embeddings** (``BAAI/bge-small-en-v1.5``, 384-dim) via the
  already-installed ``fastembed`` + ``onnxruntime`` stack. No PyTorch, no
  network: deterministic (bit-identical across runs) and local. This is what
  solves ``semantic_paraphrase`` (0 lexical overlap between query and storage).
* **BM25 sparse retrieval** over tokenized text, as in the existing baseline.
  This is what guarantees exact-match recall on identifiers, PMIDs, doses.
* Several matched composition controls over the same stored candidates:
  score fusion, candidate union, reciprocal-rank fusion, and sparse-first
  cascade.
* **Temporal decay** so newer memories win when scores tie; required for the
  ``temporal_update`` family where old+new facts conflict.
* **Importance weighting** (user > tool > assistant, identifier boost) so
  high-signal memories rank above verbose assistant prose.
* **Scope isolation** by ``(app_name, user_id)`` so cross-user leakage is
  structurally impossible, preserving the zero-leak safety gate.

The module is dependency-free beyond ``fastembed``/``numpy``/``sqlite3`` which
are already in the environment. ``TextEmbedding`` is lazily constructed once
per process and shared across all stores/retrievals in that process; the model
is cached by fastembed on first use and reused from disk thereafter.

Matched retriever modes expose the contribution of each component without
changing tokenization, candidate storage, importance, or scope isolation:

* ``mode="dense"``  -- dense embeddings only (semantic ablation).
* ``mode="bm25"``   -- sparse BM25 only (lexical ablation; equivalent to the
  legacy baseline but fused-style scored for fair comparison).
* ``mode="fused"`` -- calibrated weighted-score fusion.
* ``mode="union"`` -- the de-duplicated top-k candidates from each arm.
* ``mode="rrf"`` -- importance-weighted RRF over candidates that pass an arm's
  evidence threshold; it uses its own frozen rank-scale threshold.
* ``mode="cascade"`` -- sparse results when lexical evidence exists, otherwise
  dense fallback.

Storage is an in-memory index keyed by scope. Retrieval is deterministic given
the stored set: identical rankings for identical inputs.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from google.genai import types

from fused_memory_harness.runtime.legacy_memory import compact_event_text_for_memory


# ---------------------------------------------------------------------------
# Shared embedding backend (one model per process; bit-deterministic).
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_DIM = 384
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model_lock = threading.Lock()
_model_cache: dict[str, TextEmbedding] = {}
_embedding_cache_lock = threading.Lock()
_embedding_cache: dict[tuple[str, str], np.ndarray] = {}

# bge-small-en-v1.5 publishes the instruction above for query encoding; passage
# encoding uses the raw text. Using the published prefix empirically widens the
# query/fact vs query/distractor gap on our semantic_paraphrase probe.
_QUERY_PREFIX = _BGE_QUERY_PREFIX


def _get_model(model_name: str) -> TextEmbedding:
    """Return a process-shared TextEmbedding instance (cached, lazy)."""
    model = _model_cache.get(model_name)
    if model is None:
        with _model_lock:
            model = _model_cache.get(model_name)
            if model is None:
                model = TextEmbedding(model_name=model_name, threads=4, lazy_load=False)
                _model_cache[model_name] = model
    return model


def embed_texts(texts: list[str], *, model_name: str = _DEFAULT_MODEL) -> np.ndarray:
    """Embed a batch of texts into a (N, dim) float32 matrix. Query encoding adds
    the bge instruction prefix only when called via :func:`embed_query`.
    """
    if not texts:
        return np.zeros((0, _DEFAULT_DIM), dtype=np.float32)
    keys = [(model_name, text) for text in texts]
    with _embedding_cache_lock:
        missing = [key for key in dict.fromkeys(keys) if key not in _embedding_cache]
    if missing:
        model = _get_model(model_name)
        computed = [
            np.asarray(value, dtype=np.float32)
            for value in model.embed([text for _, text in missing], batch_size=64)
        ]
        with _embedding_cache_lock:
            for key, value in zip(missing, computed, strict=True):
                _embedding_cache.setdefault(key, value)
    with _embedding_cache_lock:
        rows = [_embedding_cache[key] for key in keys]
    return np.vstack(rows)


def embed_query(text: str, *, model_name: str = _DEFAULT_MODEL) -> np.ndarray:
    prefixed = _QUERY_PREFIX + text if _QUERY_PREFIX else text
    return embed_texts([prefixed], model_name=model_name)[0]


# ---------------------------------------------------------------------------
# Tokenization shared with the BM25 arm. Mirrors the existing baseline so the
# sparse arm is a fair reimplementation, not a tuned variant.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{1,}")
_STOPWORDS = {
    "about", "across", "after", "answer", "current", "earlier", "exactly",
    "give", "only", "our", "please", "recorded", "returned", "the", "this",
    "use", "what", "when", "which", "and", "that", "was", "has", "for", "with",
    "from", "into", "were", "they", "them", "their", "there", "these", "those",
    "your", "you", "are", "was", "did", "had", "have", "been", "being",
}


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.casefold()):
        cleaned = tok.strip("./:-_")
        if len(cleaned) < 2 or cleaned in _STOPWORDS:
            continue
        out.append(cleaned)
        if "/" in tok:
            out.extend(p for p in tok.split("/") if len(p) >= 2)
        if "-" in tok:
            out.extend(p for p in tok.split("-") if len(p) >= 2)
    return out


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Importance (shared scoring cue; identical semantics to long_term_memory.py).
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(
    r"\b(?:Patient/[A-Za-z0-9_.-]+|PMID[:\s]*\d+|topic\s+[A-Z0-9]+|FHIR|CKD)\b",
    re.I,
)
_IDENTIFIER_PATTERNS = {
    "patient": re.compile(r"\bPatient/[A-Za-z0-9_.-]+\b", re.I),
    "pmid": re.compile(r"\bPMID[:\s]*\d+\b", re.I),
    "topic": re.compile(r"\btopic\s+[A-Z0-9]+\b", re.I),
}


def _identifier_keys(text: str) -> dict[str, set[str]]:
    """Extract normalized high-precision identifiers by namespace."""
    keys: dict[str, set[str]] = {}
    for kind, pattern in _IDENTIFIER_PATTERNS.items():
        values = {" ".join(match.group(0).casefold().split()) for match in pattern.finditer(text)}
        if values:
            keys[kind] = values
    return keys


def _identifier_compatible(query: str, entry_text: str) -> bool:
    """Require exact namespace matches when the query names an identifier.

    Dense similarity and token overlap may make nearby record identifiers look
    relevant.  A query that names a high-precision identifier therefore admits
    only candidates carrying that exact identifier in every named namespace.
    Queries without recognized identifiers retain the full candidate set.
    """
    query_keys = _identifier_keys(query)
    if not query_keys:
        return True
    entry_keys = _identifier_keys(entry_text)
    return all(
        kind in entry_keys and not values.isdisjoint(entry_keys[kind])
        for kind, values in query_keys.items()
    )


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


# ---------------------------------------------------------------------------
# In-memory fused index (deterministic given stored set).
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    text: str
    embedding: np.ndarray
    source_kind: str
    session_id: str
    created_at: float
    importance: float
    tokens: list[str] = field(default_factory=list)
    doc_id: int = 0  # dense id for BM25 doc-frequency tables


@dataclass(frozen=True)
class FusedMemorySettings:
    max_entry_chars: int = 900
    retrieve_top_k: int = 5
    retrieve_min_score: float = 0.02
    retrieve_max_chars: int = 1800
    model_name: str = _DEFAULT_MODEL
    include_cross_session: bool = True
    # RRF tuning: rank weight and the two list weights.
    rrf_k: int = 60  # standard reciprocal rank fusion constant
    rrf_min_score: float = 0.005
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    semantic_dense_weight: float = 0.85
    enforce_identifier_consistency: bool = False
    # Temporal decay half-life in seconds (wall clock). Set very large so tie
    # break only dominates within a single benchmark case's seconds-scale spans.
    temporal_half_life_seconds: float = 3600.0 * 24 * 365  # ~1 year


class FusedMemoryIndex:
    """In-memory per-scope fused retriever.

    One index per (app_name, user_id). ``session_id`` is retained on entries
    for cross-session filtering when ``include_cross_session=False`` but the
    benchmark always uses cross-session retrieval, so all entries participate.
    """

    def __init__(self, settings: FusedMemorySettings) -> None:
        self._settings = settings
        self._scopes: dict[tuple[str, str], list[_Entry]] = {}
        # BM25 document-frequency table per scope: token -> doc count.
        self._df: dict[tuple[str, str], dict[str, int]] = {}
        self._doc_counter: dict[tuple[str, str], int] = {}

    # -- storage -----------------------------------------------------------

    def store(
        self,
        event: Any,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        settings = self._settings
        text = compact_event_text_for_memory(event, max_chars=settings.max_entry_chars)
        if len(text) < 12:
            return
        text = _normalize(text)
        emb = embed_texts([text], model_name=settings.model_name)[0]
        kind = _source_kind(event, text)
        key = (app_name, user_id)
        entry = _Entry(
            text=text,
            embedding=emb,
            source_kind=kind,
            session_id=session_id,
            created_at=float(getattr(event, "timestamp", 0.0) or time.time()),
            importance=_importance(text, kind),
            tokens=_tokens(text),
        )
        entries = self._scopes.setdefault(key, [])
        entry.doc_id = len(entries)
        entries.append(entry)
        # update BM25 df
        df = self._df.setdefault(key, {})
        seen = set(entry.tokens)
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1
        self._doc_counter[key] = self._doc_counter.get(key, 0) + 1

    # -- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        mode: str = "fused",
        session_id: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        settings = self._settings
        key = (app_name, user_id)
        entries = self._scopes.get(key)
        if not entries or not query.strip():
            return []
        if not settings.include_cross_session and session_id:
            entries = [e for e in entries if e.session_id == session_id]
            if not entries:
                return []
        if settings.enforce_identifier_consistency:
            entries = [entry for entry in entries if _identifier_compatible(query, entry.text)]
            if not entries:
                return []
        effective_top_k = int(top_k) if top_k is not None else settings.retrieve_top_k
        valid_modes = {"dense", "bm25", "fused", "union", "rrf", "cascade"}
        if mode not in valid_modes:
            raise ValueError(f"unknown matched retrieval mode: {mode}")

        query_tokens = _tokens(query)
        query_has_precise_identifier = bool(_identifier_keys(query))
        query_has_lexical_anchor = bool(
            re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", query)
        )

        # --- dense arm (semantic) ---
        dense_scores: list[float] = []
        if mode != "bm25":
            q_emb = embed_query(query, model_name=settings.model_name)
            q_norm = float(np.linalg.norm(q_emb)) or 1.0
            matrix = np.vstack([e.embedding for e in entries])
            norms = np.linalg.norm(matrix, axis=1)
            norms = np.where(norms == 0.0, 1.0, norms)
            dense_scores = (matrix @ q_emb / (norms * q_norm)).tolist()
        else:
            dense_scores = [0.0] * len(entries)

        # --- sparse arm (BM25) ---
        sparse_scores: list[float] = []
        if mode != "dense":
            df = self._df.get(key, {})
            n_docs = max(1, self._doc_counter.get(key, 0))
            avgdl = max(1.0, float(sum(len(e.tokens) for e in entries)) / max(1, len(entries)))
            k1, b = 1.5, 0.75
            q_counts: dict[str, int] = {}
            for t in query_tokens:
                q_counts[t] = q_counts.get(t, 0) + 1
            for entry in entries:
                if not entry.tokens:
                    sparse_scores.append(0.0)
                    continue
                tf: dict[str, int] = {}
                for t in entry.tokens:
                    tf[t] = tf.get(t, 0) + 1
                dl = len(entry.tokens)
                score = 0.0
                for t, qf in q_counts.items():
                    if t not in tf:
                        continue
                    n_t = df.get(t, 0)
                    idf = math.log(1.0 + (n_docs - n_t + 0.5) / (n_t + 0.5))
                    denom = tf[t] + k1 * (1.0 - b + b * (dl / avgdl))
                    score += idf * (tf[t] * (k1 + 1.0)) / denom
                sparse_scores.append(score)
        else:
            sparse_scores = [0.0] * len(entries)

        # --- fusion + temporal decay + importance ---
        # The fused score preserves positive raw dense cosine so semantically
        # relevant matches are never zeroed out by normalization. BM25 is
        # squashed through a sigmoid to [0,1] so the
        # two arms compose on a shared scale without destroying dense signal.
        # Min-max normalization was rejected because it maps the lowest-scored
        # (but still-relevant) memory to 0, which the threshold then drops.
        now = max((e.created_at for e in entries), default=0.0) or time.time()

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0

        s_max = max(sparse_scores) if sparse_scores else 0.0

        dense_ranks = _rank_indices(dense_scores)
        sparse_ranks = _rank_indices(sparse_scores)
        results: list[dict[str, Any]] = []
        for i, entry in enumerate(entries):
            d_score = dense_scores[i]
            s_score = sparse_scores[i]
            # Dense cosine is already on [0,1]; clamp negatives to 0.
            d_n = max(0.0, d_score)
            # BM25 squashed to [0,1] via sigmoid centered at half the max, so a
            # strong BM25 hit maps near 1.0 and zero hits map to 0.5->0 only via
            # the dense arm. Use s_max for a self-calibrating midpoint.
            s_n = _sigmoid(s_score - 0.5 * s_max) if s_max > 0 else 0.0
            dense_weight = (
                settings.dense_weight
                if query_has_precise_identifier or query_has_lexical_anchor
                else settings.semantic_dense_weight
            )
            weighted_score = dense_weight * d_n + (1.0 - dense_weight) * s_n
            # temporal decay: newer wins ties. Use a mild multiplier so it only
            # breaks near-ties, not decisive score advantages.
            age = max(0.0, now - entry.created_at)
            decay = 0.5 ** (age / settings.temporal_half_life_seconds)
            # importance scales the fused score (user/tool > assistant).
            score = (weighted_score + 1e-9 * decay) * entry.importance
            if mode == "dense":
                score = max(0.0, d_score) * entry.importance
            elif mode == "bm25":
                score = s_score * entry.importance
            elif mode == "rrf":
                dense_rrf = (
                    settings.dense_weight / (settings.rrf_k + dense_ranks[i] + 1)
                    if d_score >= settings.retrieve_min_score
                    else 0.0
                )
                sparse_rrf = (
                    settings.sparse_weight / (settings.rrf_k + sparse_ranks[i] + 1)
                    if s_score > 0.0
                    else 0.0
                )
                score = (
                    dense_rrf + sparse_rrf + (1e-12 * decay if dense_rrf or sparse_rrf else 0.0)
                ) * entry.importance
            elif mode == "cascade":
                if max(sparse_scores, default=0.0) > 0.0:
                    score = s_score * entry.importance
                else:
                    score = max(0.0, d_score) * entry.importance
            elif mode == "union":
                score = max(d_n, s_n) * entry.importance
            results.append(
                {
                    "text": entry.text,
                    "score": float(score),
                    "dense_score": float(d_score),
                    "sparse_score": float(s_score),
                    "source_kind": entry.source_kind,
                    "session_id": entry.session_id,
                    "created_at": entry.created_at,
                    "importance": entry.importance,
                    "doc_id": entry.doc_id,
                    "dense_rank": dense_ranks[i],
                    "sparse_rank": sparse_ranks[i],
                    "retrieval_mode": mode,
                    "dense_eligible": d_score >= settings.retrieve_min_score,
                    "sparse_eligible": s_score > 0.0,
                    "retrieved": True,
                }
            )

        if mode == "union":
            dense_order = sorted(
                range(len(results)),
                key=lambda i: (-results[i]["dense_score"], -results[i]["created_at"], results[i]["doc_id"]),
            )
            sparse_order = sorted(
                range(len(results)),
                key=lambda i: (-results[i]["sparse_score"], -results[i]["created_at"], results[i]["doc_id"]),
            )
            selected = {
                i
                for i in dense_order[:effective_top_k]
                if results[i]["dense_score"] >= settings.retrieve_min_score
            }
            selected.update(
                i for i in sparse_order[:effective_top_k] if results[i]["sparse_score"] > 0.0
            )
            filtered = [results[i] for i in selected]
            filtered.sort(
                key=lambda r: (
                    min(r["dense_rank"], r["sparse_rank"]),
                    -max(r["dense_score"], r["sparse_score"]),
                    r["doc_id"],
                )
            )
            return filtered[:effective_top_k]

        threshold = settings.rrf_min_score if mode == "rrf" else settings.retrieve_min_score
        filtered = [
            r
            for r in results
            if r["score"] >= threshold
            and (
                mode != "rrf"
                or r["dense_eligible"]
                or r["sparse_eligible"]
            )
        ]
        filtered.sort(key=lambda r: (-r["score"], -r["created_at"], r["doc_id"]))
        return filtered[:effective_top_k]


def _rank_indices(scores: list[float]) -> list[int]:
    """Return deterministic rank (0 = best), breaking equal scores by index."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rank = [0] * len(scores)
    for r, idx in enumerate(order):
        rank[idx] = r
    return rank


# ---------------------------------------------------------------------------
# Per-run scope registry used by the experiment harness.
# ---------------------------------------------------------------------------

_INDEX_REGISTRY: dict[tuple[str, str, str, int], FusedMemoryIndex] = {}


def _index_for(
    condition: str,
    budget: int,
    config: dict[str, Any],
) -> FusedMemoryIndex:
    """Return a fresh per (condition, budget) index so solver == solver, no
    cross-case contamination. Configurable via the experiment config."""
    key = (condition, str(budget))
    idx = _INDEX_REGISTRY.get(key)
    if idx is None:
        settings = FusedMemorySettings(
            max_entry_chars=int(config.get("long_term_entry_max_chars", 900)),
            retrieve_top_k=int(config.get("long_term_top_k", 5)),
            retrieve_min_score=float(config.get("long_term_min_score", 0.02)),
            retrieve_max_chars=int(config.get("long_term_max_chars", 1800)),
            model_name=config.get("fused_model_name", _DEFAULT_MODEL),
            include_cross_session=bool(config.get("include_cross_session", True)),
            dense_weight=float(config.get("fused_dense_weight", 0.6)),
            sparse_weight=float(config.get("fused_sparse_weight", 0.4)),
            semantic_dense_weight=float(config.get("fused_semantic_dense_weight", 0.85)),
            enforce_identifier_consistency=condition in {
                "fused_hybrid",
                "dense_guarded_hybrid",
            },
            rrf_k=int(config.get("fused_rrf_k", 60)),
            rrf_min_score=float(config.get("fused_rrf_min_score", 0.005)),
        )
        idx = FusedMemoryIndex(settings)
        _INDEX_REGISTRY[key] = idx
    return idx


def reset_registry() -> None:
    """Clear per-process state so a fresh build doesn't carry stale entries."""
    _INDEX_REGISTRY.clear()


def store(
    event: Any,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    max_chars: int,
    condition: str,
    budget: int,
    config: dict[str, Any],
) -> None:
    """Store API matching bm25_memory.store shape, condition/budget-scoped."""
    idx = _index_for(condition, budget, config)
    idx.store(event, app_name=app_name, user_id=user_id, session_id=session_id)


def retrieve(
    query: str,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    top_k: int,
    mode: str,
    condition: str,
    budget: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    idx = _index_for(condition, budget, config)
    return idx.retrieve(
        query,
        app_name=app_name,
        user_id=user_id,
        mode=mode,
        session_id=session_id,
        top_k=top_k,
    )


def as_content(entries: list[dict[str, Any]], *, max_chars: int) -> types.Content | None:
    lines: list[str] = []
    used = 0
    for entry in entries:
        line = (
            f"- {entry.get('source_kind', 'memory')} session={entry.get('session_id', '')}: "
            f"{_truncate(_normalize(entry.get('text', '')), 420)}"
        )
        if used + len(line) > max_chars:
            continue
        lines.append(line)
        used += len(line)
    if not lines:
        return None
    text = (
        "Retrieved matched-index memory. Use only when relevant to the latest user request; "
        "do not override newer session context.\n"
        + "\n".join(lines)
    )
    return types.Content(
        role="user",
        parts=[types.Part.from_text(text=text[:max_chars])],
    )
