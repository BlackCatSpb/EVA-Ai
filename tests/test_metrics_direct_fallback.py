import time
from typing import Any, Dict, List

from eva_ai.core.core_brain import CoreBrain


def test_direct_metrics_emission_without_events():
    # Arrange: init brain and disable events system
    brain = CoreBrain(config={})
    brain.events = None  # simulate events system disabled

    received: List[Dict[str, Any]] = []

    # Monkeypatch SystemMetricsManager.emit/emit_many to capture direct calls
    orig_emit_many = getattr(brain.metrics_manager, "emit_many", None)
    orig_emit = getattr(brain.metrics_manager, "emit", None)

    def fake_emit_many(metrics: List[Dict[str, Any]]):
        received.extend(metrics)
        return len(metrics)

    brain.metrics_manager.emit_many = fake_emit_many  # type: ignore[attr-defined]
    
    def fake_emit(metric: Dict[str, Any]):
        received.append(metric)
        return 1
    brain.metrics_manager.emit = fake_emit  # type: ignore[attr-defined]

    # Act: use direct emit via brain API
    m1 = {"name": "fallback.single", "value": 1, "component": "test", "type": "counter"}
    m2 = {"name": "fallback.list.a", "value": 2, "component": "test", "type": "gauge"}
    m3 = {"name": "fallback.list.b", "value": 3, "component": "test", "type": "counter"}

    brain.emit_metric(m1)
    brain.emit_metrics([m2, m3])

    time.sleep(0.01)

    # Assert: all three reached metrics manager directly
    assert len(received) == 3, f"Expected 3 metrics delivered, got {len(received)}: {received}"

    names = [m.get("name") for m in received]
    assert "fallback.single" in names
    assert "fallback.list.a" in names
    assert "fallback.list.b" in names

    # Cleanup
    if orig_emit_many is not None:
        brain.metrics_manager.emit_many = orig_emit_many  # type: ignore[attr-defined]
    if orig_emit is not None:
        brain.metrics_manager.emit = orig_emit  # type: ignore[attr-defined]
