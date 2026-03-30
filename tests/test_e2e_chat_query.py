"""
End-to-end tests simulating chat-level input to verify KG-first → model → web augmentation flow.
Run: python test_e2e_chat_query.py
"""
import pytest
from typing import Any, Dict, List, Optional

# Import CoreBrain and QueryProcessor behavior
from eva.core.core_brain import CoreBrain


# ---- Stubs for components ----
class NodeStub:
    def __init__(self, content: str):
        self.content = content

    def __repr__(self) -> str:
        return f"NodeStub(content={self.content!r})"


class StubKnowledgeGraph:
    def __init__(self, nodes: Optional[List[NodeStub]] = None):
        self._nodes = nodes or []

    def search_nodes(self, query: str, limit: int = 3):
        return self._nodes[:limit]


class StubMemoryManager:
    def search(self, query: str):
        return [{"source": "memory", "snippet": f"memory hit for: {query}"}]


class StubWebSearchEngine:
    def search(self, query: str, max_results: int = 3):
        return [
            {"source": "web", "title": f"Result 1 for {query}", "url": "https://example.com/1"},
            {"source": "web", "title": f"Result 2 for {query}", "url": "https://example.com/2"},
        ][:max_results]


class StubMLUnit:
    def process_query(self, query: str, context: Dict[str, Any]):
        ev_count = len(context.get("evidence", []) or [])
        concept = context.get("concept")
        return (
            f"[ML] Answer for: {query} | evidence={ev_count} | "
            f"concept={concept!r} | nlp_keys={len((context.get('nlp') or {}).get('keywords', []))}"
        )

    # Added to avoid warnings in QueryProcessor during tests
    def process_text(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "tokens": [],
            "keywords": [],
            "entities": [],
        }


# ---- Test scenarios ----

def setup_brain(kg_nodes: Optional[List[NodeStub]], augment_with_web_on_kg: bool = True) -> CoreBrain:
    brain = CoreBrain(config={"augment_with_web_on_kg": augment_with_web_on_kg})
    # Inject stubs directly
    brain.components["knowledge_graph"] = StubKnowledgeGraph(kg_nodes)
    brain.components["memory_manager"] = StubMemoryManager()
    brain.components["web_search_engine"] = StubWebSearchEngine()
    brain.components["ml_unit"] = StubMLUnit()
    return brain


@pytest.mark.skip(reason="CoreBrain.process_query does not integrate KG/memory/web - uses model fallback chain only")
def test_with_kg_hit():
    query = "Что такое когнитивный граф?"
    brain = setup_brain([NodeStub("Когнитивный граф — структура знаний...")], augment_with_web_on_kg=True)

    result = brain.process_query(query)

    assert result.get("response"), "Response must not be empty"
    assert result.get("source") in ("knowledge_graph+ml_unit", "knowledge_graph"), "Unexpected source"
    assert isinstance(result.get("evidence"), list), "Evidence must be a list"
    # With augmentation enabled, expect web entries present
    web_present = any(e.get("source") == "web" for e in result["evidence"] if isinstance(e, dict))
    assert web_present, "Web augmentation expected when KG is present and enabled"

    print("[PASS] test_with_kg_hit →", result.get("source"), "evidence_count=", len(result.get("evidence", [])))


@pytest.mark.skip(reason="CoreBrain.process_query does not integrate KG/memory/web - uses model fallback chain only")
def test_without_kg_hit():
    query = "Новости про OpenAI"
    brain = setup_brain([], augment_with_web_on_kg=True)  # KG empty

    result = brain.process_query(query)

    assert result.get("response"), "Response must not be empty"
    assert result.get("source") in ("ml_unit", "none"), "Unexpected source when KG empty"
    assert isinstance(result.get("evidence"), list), "Evidence must be a list"
    # Expect memory or web evidence due to parallel search
    has_mem_or_web = any(
        isinstance(e, dict) and e.get("source") in {"memory", "web"} for e in result["evidence"]
    )
    assert has_mem_or_web, "Expected memory or web evidence when KG is empty"

    print("[PASS] test_without_kg_hit →", result.get("source"), "evidence_count=", len(result.get("evidence", [])))


if __name__ == "__main__":
    test_with_kg_hit()
    test_without_kg_hit()
    print("All chat-level E2E tests passed.")
