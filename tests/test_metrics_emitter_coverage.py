import time
from typing import Any, Dict, List

import pytest

from cogniflex.core.core_brain import CoreBrain
from cogniflex.core.system_metrics import SystemMetricsManager
from cogniflex.core.response_generator import ResponseGenerator
from cogniflex.mlearning.ml_unit import MLUnit
from cogniflex.mlearning.training_orchestrator import TrainingOrchestrator


def _valid_metrics() -> List[Dict[str, Any]]:
    return [
        {"name": "test.ok.counter", "component": "unit", "type": "counter", "value": 1},
        {"name": "test.ok.summary", "component": "unit", "type": "summary", "value": 0.12},
    ]


def _invalid_metric_missing_component() -> Dict[str, Any]:
    return {"name": "test.bad", "type": "counter", "value": 1}


def _assert_counts(brain: CoreBrain, expected_ok: int, expected_bad: int):
    # Give transport a moment (events are sync but be safe)
    time.sleep(0.01)
    buf = brain.metrics_manager.flush()
    quar = brain.metrics_manager.get_quarantine()
    assert len([m for m in buf if isinstance(m, dict)]) == expected_ok, (
        f"expected {expected_ok} accepted, got {len(buf)}; buf={buf}"
    )
    assert len(quar) == expected_bad, (
        f"expected {expected_bad} quarantined, got {len(quar)}; quar={quar}"
    )


def test_response_generator_emits_and_schema_validates_events_path():
    brain = CoreBrain(config={})
    rg = ResponseGenerator(brain=brain)

    metrics = _valid_metrics() + [_invalid_metric_missing_component()]

    # Pre-check validation API on originals
    sm: SystemMetricsManager = brain.metrics_manager
    assert sm.validate_many(metrics) == 2

    # Act: use emitter
    rg._emit_metrics(metrics)  # type: ignore[attr-defined]

    # Assert counts via real manager
    _assert_counts(brain, expected_ok=2, expected_bad=1)


def test_mlunit_emits_and_schema_validates_events_path():
    brain = CoreBrain(config={})
    mlu = MLUnit(brain=brain)

    metrics = _valid_metrics() + [_invalid_metric_missing_component()]

    sm: SystemMetricsManager = brain.metrics_manager
    assert sm.validate_many(metrics) == 2

    mlu._emit_metrics(metrics)  # type: ignore[attr-defined]

    _assert_counts(brain, expected_ok=2, expected_bad=1)


def test_training_orchestrator_emits_and_schema_validates_events_path():
    brain = CoreBrain(config={})
    to = TrainingOrchestrator(brain=brain)

    metrics = _valid_metrics() + [_invalid_metric_missing_component()]

    sm: SystemMetricsManager = brain.metrics_manager
    assert sm.validate_many(metrics) == 2

    to._emit_metrics(metrics)  # type: ignore[attr-defined]

    _assert_counts(brain, expected_ok=2, expected_bad=1)


@pytest.mark.parametrize("component_factory", [
    lambda b: ResponseGenerator(brain=b),
    lambda b: MLUnit(brain=b),
    lambda b: TrainingOrchestrator(brain=b),
])
def test_emitters_work_with_direct_fallback_when_events_disabled(component_factory):
    brain = CoreBrain(config={})
    # Disable events to force direct fallback path
    brain.events = None

    comp = component_factory(brain)

    metrics = _valid_metrics() + [_invalid_metric_missing_component()]

    # Validate originals
    assert brain.metrics_manager.validate_many(metrics) == 2

    # Act
    comp._emit_metrics(metrics)  # type: ignore[attr-defined]

    # Assert via real manager
    _assert_counts(brain, expected_ok=2, expected_bad=1)


def test_unified_text_processor_emitter_schema_validation_if_available():
    """
    Import UnifiedTextProcessor lazily. If heavy deps are missing, skip.
    Only exercise its _emit_metrics path to avoid heavy model init.
    """
    try:
        from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor  # noqa: WPS433
    except Exception:
        pytest.skip("UnifiedTextProcessor not importable in this environment")

    brain = CoreBrain(config={})
    utp = UnifiedTextProcessor(brain=brain)

    metrics = _valid_metrics() + [_invalid_metric_missing_component()]

    assert brain.metrics_manager.validate_many(metrics) == 2

    utp._emit_metrics(metrics)  # type: ignore[attr-defined]

    _assert_counts(brain, expected_ok=2, expected_bad=1)
