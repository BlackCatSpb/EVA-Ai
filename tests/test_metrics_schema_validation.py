import time
from cogniflex.core.system_metrics import SystemMetricsManager


def test_validate_metric_schema_valid():
    mgr = SystemMetricsManager()
    metric = {
        "name": "unit.test.counter",
        "component": "unit",
        "type": "counter",
        "value": 1,
        "timestamp": time.time(),
        "labels": {"a": "b"},
        "unit": "items",
        "subsystem": "tests",
    }
    assert mgr.validate_metric_schema(metric) is True


def test_validate_metric_schema_invalid_missing_fields():
    mgr = SystemMetricsManager()
    # Missing name
    bad1 = {"component": "unit", "type": "counter", "value": 1}
    # Wrong type
    bad2 = {"name": "x", "component": "unit", "type": "unknown", "value": 1}
    # Non-numeric value
    bad3 = {"name": "x", "component": "unit", "type": "gauge", "value": "nan"}

    assert mgr.validate_metric_schema(bad1) is False
    assert mgr.validate_metric_schema(bad2) is False
    assert mgr.validate_metric_schema(bad3) is False


def test_validate_many_mixed_and_emit_quarantine():
    mgr = SystemMetricsManager()
    good = {"name": "ok", "component": "unit", "type": "gauge", "value": 0.5}
    bad = {"name": "bad", "component": 123, "type": "gauge", "value": 0.5}

    assert mgr.validate_many([good, bad]) == 1

    # emit should accept only the valid one and quarantine the bad one
    accepted = mgr.emit_many([good, bad])
    assert accepted == 1

    flushed = mgr.flush()
    assert len(flushed) == 1
    assert flushed[0]["name"] == "ok"

    quarantined = mgr.get_quarantine()
    # bad should be quarantined
    assert len(quarantined) == 1
    assert quarantined[0]["name"] == "bad"
