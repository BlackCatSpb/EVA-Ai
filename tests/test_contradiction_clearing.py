import os
import shutil
import tempfile
import time
import json
import pytest

from typing import Dict, Any

from cogniflex.contradiction.contradiction_core import OptimizedContradictionDetector
from cogniflex.core.core_brain import CoreBrain


@pytest.fixture()
def temp_cache_dir():
    d = tempfile.mkdtemp(prefix="cogniflex_test_contradictions_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_detector_clear_all_contradictions(temp_cache_dir):
    # 1) Создаем детектор с временным хранилищем
    det = OptimizedContradictionDetector(cache_dir=temp_cache_dir)

    # 2) Создаем противоречие
    c = det.detect_contradiction(
        concept="unit_test_concept",
        facts=[
            {"value": 10, "text": "A: 10"},
            {"value": 20, "text": "B: 20"},
        ],
        metadata={"domain": "tests"},
    )
    assert c is not None, "Contradiction should be detected for numeric conflict"

    # 3) Убеждаемся, что оно сохранено в БД (вызов сохранения в detect)
    assert len(det.get_all_contradictions()) >= 1

    # 4) Очищаем
    report = det.clear_all_contradictions()
    assert isinstance(report, dict)
    assert report.get("ok") in (True, False)
    assert report.get("cleared", 0) >= 1

    # 5) Пересоздаем детектор на том же пути и убеждаемся, что всё чисто
    det2 = OptimizedContradictionDetector(cache_dir=temp_cache_dir)
    assert det2.get_all_contradictions() == []


def test_corebrain_clear_all_contradictions_wrapper():
    # 1) Поднимаем CoreBrain (минимально)
    brain = CoreBrain(config={})

    # 2) Подменяем резолвер на простой стаб, чтобы избежать зависимости
    class StubResolver:
        def __init__(self):
            self.called = False
        def clear_all_contradictions(self) -> Dict[str, Any]:
            self.called = True
            return {"ok": True, "cleared": 0, "db_path": None, "error": None}

    stub = StubResolver()
    try:
        brain.components["contradiction_resolver"] = stub
    except Exception:
        # В крайне редком случае, используем атрибут напрямую
        setattr(brain, "contradiction_resolver", stub)

    # 3) Вызываем обертку
    report = brain.clear_all_contradictions()
    assert isinstance(report, dict)
    assert report.get("ok") is True
    assert report.get("cleared") == 0
    assert report.get("error") is None


def test_corebrain_clear_all_contradictions_no_resolver():
    brain = CoreBrain(config={})
    # Убедимся, что резолвер отсутствует
    brain.components.pop("contradiction_resolver", None)
    if hasattr(brain, "contradiction_resolver"):
        delattr(brain, "contradiction_resolver")

    report = brain.clear_all_contradictions()
    assert report.get("ok") is False
    assert report.get("cleared") == 0
    assert "unavailable" in (report.get("error") or "")
