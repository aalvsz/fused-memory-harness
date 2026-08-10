from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hybrid_memory.common import load_config
from experiments.hybrid_memory.context_benchmark import MATCHED_MODE_BY_CONDITION
from experiments.hybrid_memory.fused_memory import (
    FusedMemoryIndex,
    FusedMemorySettings,
    _rank_indices,
)
from experiments.hybrid_memory.generate_cases import (
    SEMANTIC_DEFINITIVE_POOL,
    SEMANTIC_DEVELOPMENT_POOL,
    SEMANTIC_HELDOUT_POOL,
    generate,
)
from experiments.hybrid_memory.model_benchmark import SYSTEM_PROMPT_V3


HELDOUT_CONFIG = (
    ROOT / "experiments/hybrid_memory/configs/heldout_semantic_evaluation.json"
)
DEFINITIVE_CONFIG = (
    ROOT / "experiments/hybrid_memory/configs/definitive_heldout.json"
)


def _pool_text(pool: tuple[dict, ...]) -> set[str]:
    text: set[str] = set()
    for item in pool:
        text.add(item["stored"].casefold())
        text.update(query.casefold() for query in item["queries"])
        text.update(fact.casefold() for fact in item["answer_facts"])
    return text


def _event(text: str, *, timestamp: float) -> SimpleNamespace:
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part], role="user")
    return SimpleNamespace(
        author="user",
        content=content,
        partial=False,
        timestamp=timestamp,
    )


def test_heldout_semantic_pool_is_disjoint_and_not_prompt_example():
    development = _pool_text(SEMANTIC_DEVELOPMENT_POOL)
    heldout = _pool_text(SEMANTIC_HELDOUT_POOL)

    assert development.isdisjoint(heldout)
    assert all("nephropathy" not in text for text in heldout)
    assert all("renal diagnosis" not in text for text in heldout)
    assert len({item["concept_id"] for item in SEMANTIC_HELDOUT_POOL}) >= 4
    assert all(len(item["queries"]) >= 3 for item in SEMANTIC_HELDOUT_POOL)
    prompt = SYSTEM_PROMPT_V3.casefold()
    assert all(text not in prompt for text in heldout)


def test_heldout_generation_is_deterministic_and_template_diverse():
    config = load_config(HELDOUT_CONFIG)
    config["case_counts"] = {family: 1 for family in config["case_counts"]}
    config["case_counts"]["semantic_paraphrase"] = 12

    first = generate(config)
    second = generate(config)
    semantic = [row for row in first if row["family"] == "semantic_paraphrase"]

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len({row["query"] for row in semantic}) == 12
    assert {row["metadata"]["semantic_concept_id"] for row in semantic} == {
        item["concept_id"] for item in SEMANTIC_HELDOUT_POOL
    }


def test_definitive_semantic_pool_has_100_independent_disjoint_concepts():
    development = _pool_text(SEMANTIC_DEVELOPMENT_POOL)
    heldout = _pool_text(SEMANTIC_HELDOUT_POOL)
    definitive = _pool_text(SEMANTIC_DEFINITIVE_POOL)

    assert len(SEMANTIC_DEFINITIVE_POOL) == 100
    assert len({item["concept_id"] for item in SEMANTIC_DEFINITIVE_POOL}) == 100
    assert all(len(item["queries"]) == 1 for item in SEMANTIC_DEFINITIVE_POOL)
    assert development.isdisjoint(definitive)
    assert heldout.isdisjoint(definitive)


def test_definitive_generation_is_frozen_complete_and_namespaced():
    config = load_config(DEFINITIVE_CONFIG)
    first = generate(config)
    second = generate(config)
    semantic = [row for row in first if row["family"] == "semantic_paraphrase"]

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(first) == 1100
    assert len(semantic) == 100
    assert len({row["metadata"]["semantic_concept_id"] for row in semantic}) == 100
    assert all(row["case_id"].startswith("definitive-v1-") for row in first)
    assert len(config["conditions"]) == 14
    assert config["context_budgets"] == [4000, 8000, 16000, 24000]


def test_matched_condition_wiring_is_explicit():
    assert MATCHED_MODE_BY_CONDITION == {
        "dense_long_term": "dense",
        "dense_hybrid": "dense",
        "fused_hybrid": "fused",
        "matched_sparse_hybrid": "bm25",
        "union_hybrid": "union",
        "rrf_hybrid": "rrf",
        "cascade_hybrid": "cascade",
    }


def test_rank_ties_are_deterministic():
    assert _rank_indices([0.5, 0.5, 0.7]) == [1, 2, 0]


def test_matched_modes_share_scope_and_union_complements_arms(monkeypatch):
    import experiments.hybrid_memory.fused_memory as fused

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
        lambda text, *, model_name: np.asarray(
            [0.0, 0.0] if text == "unrelated" else [1.0, 0.0],
            dtype=np.float32,
        ),
    )
    index = FusedMemoryIndex(
        FusedMemorySettings(retrieve_top_k=1, retrieve_min_score=0.01)
    )
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

    sparse = index.retrieve("alpha", app_name="app", user_id="owner", mode="bm25")
    dense = index.retrieve("alpha", app_name="app", user_id="owner", mode="dense")
    union = index.retrieve("alpha", app_name="app", user_id="owner", mode="union")
    cascade = index.retrieve("alpha", app_name="app", user_id="owner", mode="cascade")
    first_rrf = index.retrieve("alpha", app_name="app", user_id="owner", mode="rrf")
    second_rrf = index.retrieve("alpha", app_name="app", user_id="owner", mode="rrf")

    assert sparse[0]["text"] == "alpha identifier exact"
    assert dense[0]["text"] == "meaning concept without lexical overlap"
    assert len(union) == 1
    assert union[0]["text"] in {sparse[0]["text"], dense[0]["text"]}
    assert cascade[0]["text"] == sparse[0]["text"]
    assert [row["doc_id"] for row in first_rrf] == [row["doc_id"] for row in second_rrf]
    assert len(first_rrf) <= 1
    assert index.retrieve(
        "unrelated", app_name="app", user_id="owner", mode="rrf"
    ) == []
    for mode in ("bm25", "dense", "union", "rrf", "cascade", "fused"):
        assert index.retrieve("alpha", app_name="app", user_id="other", mode=mode) == []
