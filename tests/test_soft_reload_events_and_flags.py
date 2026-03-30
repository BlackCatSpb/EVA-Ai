import os
import pytest

from eva.core.core_brain import CoreBrain


class _DummyEvents:
    def __init__(self):
        self.triggered = []

    def trigger(self, name, *args, **kwargs):
        self.triggered.append((name, args, kwargs))


def _enable_training_mode(monkeypatch):
    # Ensure tests don't attempt to download/load transformers
    monkeypatch.setenv("COGNIFLEX_TRAINING", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.mark.timeout(20)
def test_soft_reload_triggers_gui_event_and_resets_flag(monkeypatch):
    _enable_training_mode(monkeypatch)

    core = CoreBrain()
    assert core.initialize(), "Core should initialize"
    assert core.start(), "Core should start"

    # Replace event bus with dummy collector
    dummy = _DummyEvents()
    setattr(core, "events", dummy)

    # Run soft reload with GUI trigger
    ok = core.soft_reload(reload_gui=True)
    assert ok, "soft_reload should return True"

    # Check that GUI reload was requested
    names = [n for (n, _, __) in dummy.triggered]
    assert "request_gui_reload" in names, "soft_reload should trigger request_gui_reload event"

    # Preserve flag must be reset after reload
    assert not getattr(core, "preserve_ml_state", False), "preserve_ml_state flag must be cleared after soft_reload"

    # Core remains running
    assert core.initialized is True
    assert core.running is True

    core.stop(preserve_ml=True)


@pytest.mark.timeout(20)
def test_stop_preserve_ml_keeps_models(monkeypatch):
    _enable_training_mode(monkeypatch)

    core = CoreBrain()
    assert core.initialize(), "Core should initialize"
    assert core.start(), "Core should start"

    ml_before = core.components.get("ml_unit")
    mm_before = getattr(core, "model_manager", None)

    # Stop with preserve flag
    core.stop(preserve_ml=True)

    # ML references remain
    assert core.components.get("ml_unit") is ml_before, "ml_unit must remain when stopped with preserve_ml"
    assert getattr(core, "model_manager", None) is mm_before, "model_manager must remain when stopped with preserve_ml"

    # Start again should not reinitialize ML
    assert core.start(), "Core should start after preserved stop"
    assert core.components.get("ml_unit") is ml_before, "ml_unit must be same after restart"
    assert getattr(core, "model_manager", None) is mm_before, "model_manager must be same after restart"

    core.stop(preserve_ml=True)
