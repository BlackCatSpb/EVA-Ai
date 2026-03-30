import os
from typing import List, Dict, Any

from eva.core.core_brain import CoreBrain
from eva.core.response_generator import ResponseGenerator
from eva.adapters.kg_adapter import KGAdapter


def _project_root() -> str:
    here = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(here, ".."))


def _kg_demo_dir() -> str:
    return os.path.join(_project_root(), "cogniflex_models", "kg_demo")


def test_kg_metrics_emission_basic():
    brain = CoreBrain(config={})
    rg = ResponseGenerator(brain=brain)

    kg = KGAdapter(base_dir=_kg_demo_dir())

    # Подменим доступ к KG, чтобы не трогать ModelManager
    rg._get_kg_adapter = lambda: kg  # type: ignore[attr-defined]

    # Выполняем подготовку промпта (вызовет retrieve/expand + эмиссию метрик)
    prompt = "Что такое ruGPT3 XL и как он используется в RAG?"
    _ = rg._prepare_prompt(prompt=prompt, task="chat", context="")

    # Проверяем, что метрики были эмитированы
    buf: List[Dict[str, Any]] = brain.metrics_manager.flush()
    names = {m.get("name") for m in buf if isinstance(m, dict)}

    assert "kg.retrieve.latency_ms" in names, f"no kg.retrieve.latency_ms in {names}"
    assert "kg.expand.latency_ms" in names, f"no kg.expand.latency_ms in {names}"
    assert "kg.retrieve.hit" in names, f"no kg.retrieve.hit in {names}"
