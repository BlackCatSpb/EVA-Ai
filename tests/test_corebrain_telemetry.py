import sys
import json
import time
import importlib
import types

import pytest


def test_corebrain_init_and_telemetry_methods_do_not_raise():
    # Lazy import CoreBrain
    mod = importlib.import_module('cogniflex.core.core_brain')
    CoreBrain = getattr(mod, 'CoreBrain')

    # Ensure known heavy provider modules are not imported as a side effect
    heavy_modules_before = set(m for m in sys.modules.keys())

    core = CoreBrain()

    # Call safe wrappers; they must not raise and must return dicts
    snap = core.get_resource_snapshot()
    stats = core.get_cache_stats()

    assert isinstance(snap, dict)
    assert isinstance(stats, dict)

    # Basic sanity: keys may vary, but dicts should be JSON serializable
    json.dumps(snap)
    json.dumps(stats)

    # Ensure commonly heavy model names didn't appear in sys.modules after calls
    heavy_suspects = [
        'transformers', 'torch', 'qwen', 'qwen2', 'accelerate', 'bitsandbytes',
    ]
    heavy_modules_after = set(m for m in sys.modules.keys())
    newly_loaded = heavy_modules_after - heavy_modules_before
    assert all(name not in newly_loaded for name in heavy_suspects)


def test_corebrain_multiple_calls_idempotent_and_fast():
    mod = importlib.import_module('cogniflex.core.core_brain')
    CoreBrain = getattr(mod, 'CoreBrain')
    core = CoreBrain()

    t0 = time.time()
    for _ in range(5):
        snap = core.get_resource_snapshot()
        stats = core.get_cache_stats()
        assert isinstance(snap, dict)
        assert isinstance(stats, dict)
    elapsed = time.time() - t0

    # Should be very fast (<0.5s on typical dev machines); use generous bound to avoid flakes
    assert elapsed < 2.0
