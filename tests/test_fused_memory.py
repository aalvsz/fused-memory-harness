from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import fused_memory_harness.fused_memory as fused
from fused_memory_harness import FusedMemoryIndex, FusedMemorySettings


def _event(text: str, *, timestamp: float = 1.0) -> SimpleNamespace:
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part], role="user")
    return SimpleNamespace(
        author="user",
        content=content,
        partial=False,
        timestamp=timestamp,
    )


def test_identifier_guard_rejects_nearby_records(monkeypatch):
    monkeypatch.setattr(
        fused,
        "embed_texts",
        lambda texts, *, model_name: np.ones((len(texts), 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        fused,
        "embed_query",
        lambda text, *, model_name: np.ones(2, dtype=np.float32),
    )
    index = FusedMemoryIndex(
        FusedMemorySettings(retrieve_top_k=1, enforce_identifier_consistency=True)
    )
    index.store(
        _event("Patient/SYN-KNOWN-001 takes metformin."),
        app_name="app",
        user_id="owner",
        session_id="s1",
    )

    assert index.retrieve(
        "Medication for Patient/SYN-MISSING-001?",
        app_name="app",
        user_id="owner",
        mode="fused",
    ) == []
    exact = index.retrieve(
        "Medication for Patient/SYN-KNOWN-001?",
        app_name="app",
        user_id="owner",
        mode="fused",
    )
    assert exact[0]["text"] == "Patient/SYN-KNOWN-001 takes metformin."


def test_scope_isolation_applies_to_every_retrieval_mode(monkeypatch):
    monkeypatch.setattr(
        fused,
        "embed_texts",
        lambda texts, *, model_name: np.ones((len(texts), 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        fused,
        "embed_query",
        lambda text, *, model_name: np.ones(2, dtype=np.float32),
    )
    index = FusedMemoryIndex(FusedMemorySettings(retrieve_top_k=2))
    index.store(
        _event("The user's preferred color is blue."),
        app_name="app",
        user_id="owner",
        session_id="s1",
    )

    for mode in ("dense", "bm25", "fused", "union", "rrf", "cascade"):
        assert index.retrieve(
            "preferred color",
            app_name="app",
            user_id="other",
            mode=mode,
        ) == []


def test_retrieval_modes_have_deterministic_ties(monkeypatch):
    def fake_embed_texts(texts, *, model_name):
        rows = []
        for text in texts:
            rows.append(
                np.asarray(
                    [0.0, 1.0] if "alpha identifier" in text else [1.0, 0.0],
                    dtype=np.float32,
                )
            )
        return np.vstack(rows)

    monkeypatch.setattr(fused, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        fused,
        "embed_query",
        lambda text, *, model_name: np.asarray([1.0, 0.0], dtype=np.float32),
    )
    index = FusedMemoryIndex(FusedMemorySettings(retrieve_top_k=1))
    index.store(
        _event("alpha identifier exact", timestamp=1.0),
        app_name="app",
        user_id="owner",
        session_id="s1",
    )
    index.store(
        _event("meaning concept without lexical overlap", timestamp=2.0),
        app_name="app",
        user_id="owner",
        session_id="s2",
    )

    first = index.retrieve(
        "alpha", app_name="app", user_id="owner", mode="rrf"
    )
    second = index.retrieve(
        "alpha", app_name="app", user_id="owner", mode="rrf"
    )
    assert [row["doc_id"] for row in first] == [row["doc_id"] for row in second]


def test_identifier_namespaces_are_exact():
    assert fused._identifier_compatible(
        "Medication for Patient/SYN-EXACT-001?",
        "Patient/SYN-EXACT-001 takes metformin.",
    )
    assert not fused._identifier_compatible(
        "Medication for Patient/SYN-EXACT-001?",
        "Patient/SYN-NEAR-001 takes metformin.",
    )
    assert fused._identifier_compatible(
        "Which plan did we choose?", "The selected plan was blue."
    )
