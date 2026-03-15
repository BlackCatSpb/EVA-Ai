import time
from typing import Any, Dict, List

from cogniflex.core.core_brain import CoreBrain


def test_metrics_event_routed_to_manager():
    # Arrange
    brain = CoreBrain(config={})

    # Ensure events system exists
    assert getattr(brain, "events", None) is not None, "Event system should be initialized"

    received: List[Dict[str, Any]] = []

    # Monkeypatch metrics manager methods to capture emissions
    orig_emit_many = getattr(brain.metrics_manager, "emit_many", None)
    orig_emit = getattr(brain.metrics_manager, "emit", None)

    def fake_emit_many(metrics: List[Dict[str, Any]]):
        # Record and return count as original contract suggests
        received.extend(metrics)
        return len(metrics)

    brain.metrics_manager.emit_many = fake_emit_many  # type: ignore[attr-defined]
    
    # Also patch single-metric path used by CoreBrain.emit_metric
    def fake_emit(metric: Dict[str, Any]):
        received.append(metric)
        return 1
    brain.metrics_manager.emit = fake_emit  # type: ignore[attr-defined]

    # Act: trigger with a single metric (dict)
    m1 = {"name": "unit.single", "value": 1, "component": "test", "type": "counter"}
    brain.events.trigger("metrics", m1)

    # Act: trigger with a list of metrics
    m2 = {"name": "unit.list.a", "value": 2, "component": "test", "type": "counter"}
    m3 = {"name": "unit.list.b", "value": 3, "component": "test", "type": "gauge"}
    brain.events.trigger("metrics", [m2, m3])

    # Give async hooks (if any) a moment (EventSystem in this codebase is sync, but be safe)
    time.sleep(0.01)

    # Assert: all three metrics reached the manager via CoreBrain._on_metrics_event -> emit_metrics/emit_metric
    assert len(received) == 3, f"Expected 3 metrics delivered, got {len(received)}: {received}"

    names = [m.get("name") for m in received]
    assert "unit.single" in names
    assert "unit.list.a" in names
    assert "unit.list.b" in names

    # Cleanup: restore original
    if orig_emit_many is not None:
        brain.metrics_manager.emit_many = orig_emit_many  # type: ignore[attr-defined]
    if orig_emit is not None:
        brain.metrics_manager.emit = orig_emit  # type: ignore[attr-defined]
