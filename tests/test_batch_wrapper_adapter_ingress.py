import torch

from eva.core.batch_wrapper import (
    WrapperMetadata,
    wrap_for_transfer,
)
from eva.core.event_system import EventSystem
from eva.adapters.torch_adapter import TorchBatchAdapter, Meta


def test_adapter_unwrap_and_clean_and_telemetry():
    # Prepare clean payload with an extra unknown key to ensure whitelist drops it
    item = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "foo2": torch.tensor([9]),  # unknown key should be dropped by whitelist/sanitizer
        "_meta": Meta(idx=0, length=3),
    }
    env = wrap_for_transfer(payload=item, meta=WrapperMetadata(source="unit", target="adapter"))

    # Capture metrics via EventSystem
    events = EventSystem()
    captured = []
    events.subscribe("metrics", lambda data: captured.append(data))

    # Adapter with whitelist and events
    adapter = TorchBatchAdapter(max_items=1, timeout_ms=1, events=events)

    # Push envelope; adapter must unwrap, emit telemetry, and sanitize
    adapter.push(env)
    batch = adapter.try_pop_batch()
    assert batch is not None

    # No wrapper artefacts in tensors
    forbidden = {"_wrapper", "envelope_meta", "wrapper_meta"}
    for k in forbidden:
        assert k not in batch.tensors

    # input_ids and derived lengths must be present
    assert "input_ids" in batch.tensors
    assert "input_ids_lengths" in batch.tensors

    # Telemetry must be emitted with expected shape
    assert any(isinstance(ev, dict) and ev.get("kind") == "batch_wrapper" and ev.get("event") == "wrapper_removed" for ev in captured)


def test_whitelist_drops_unknown_keys():
    events = EventSystem()
    adapter = TorchBatchAdapter(max_items=1, timeout_ms=1, events=events)

    item = {
        "input_ids": torch.tensor([5, 5], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "foo": torch.tensor([7, 7], dtype=torch.long),  # not in whitelist -> must be dropped
        "_meta": Meta(idx=0, length=2),
    }

    adapter.push(item)
    batch = adapter.try_pop_batch()
    assert batch is not None
    assert "foo" not in batch.tensors
    assert "input_ids" in batch.tensors
    assert "attention_mask" in batch.tensors
