from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

# Minimal, model-agnostic metadata carried only across module boundaries.
# Must never be injected into tensors or ML-consumed payloads.
@dataclass(frozen=True)
class WrapperMetadata:
    # Correlation and routing
    aggregation_id: Optional[str] = None
    shard_index: Optional[int] = None
    shard_count: Optional[int] = None
    source: Optional[str] = None
    target: Optional[str] = None

    # Timing
    t_created_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    t_wrapped_ms: Optional[int] = None
    t_unwrapped_ms: Optional[int] = None

    # Hints and validation
    routing_hints: Optional[Dict[str, Any]] = None
    checksum: Optional[str] = None

    # Free-form extension space (non-breaking)
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BatchEnvelope:
    payload: Any  # The clean batch payload for ML consumption
    meta: WrapperMetadata


def wrap_for_transfer(payload: Any, meta: WrapperMetadata | Dict[str, Any]) -> BatchEnvelope:
    """
    Create a BatchEnvelope around a clean payload.
    The payload must be free of wrapper artefacts and safe for ML adapters.
    """
    assert_clean_batch(payload)
    if isinstance(meta, dict):
        meta_obj = WrapperMetadata(**meta)
    else:
        meta_obj = meta
    # Stamp wrap time without mutating original meta (dataclass is frozen)
    meta_wrapped = WrapperMetadata(
        aggregation_id=meta_obj.aggregation_id,
        shard_index=meta_obj.shard_index,
        shard_count=meta_obj.shard_count,
        source=meta_obj.source,
        target=meta_obj.target,
        t_created_ms=meta_obj.t_created_ms,
        t_wrapped_ms=int(time.time() * 1000),
        t_unwrapped_ms=meta_obj.t_unwrapped_ms,
        routing_hints=meta_obj.routing_hints,
        checksum=meta_obj.checksum,
        extra=meta_obj.extra,
    )
    return BatchEnvelope(payload=payload, meta=meta_wrapped)


def unwrap_for_adapter(envelope: BatchEnvelope) -> Tuple[Any, WrapperMetadata]:
    """Remove the wrapper at adapter boundary and return the clean payload and metadata."""
    if not isinstance(envelope, BatchEnvelope):
        # Already a clean payload
        assert_clean_batch(envelope)
        # Create a minimal metadata stub
        return envelope, WrapperMetadata(t_unwrapped_ms=int(time.time() * 1000))
    meta = envelope.meta
    # Stamp unwrapped time
    stamped = WrapperMetadata(
        aggregation_id=meta.aggregation_id,
        shard_index=meta.shard_index,
        shard_count=meta.shard_count,
        source=meta.source,
        target=meta.target,
        t_created_ms=meta.t_created_ms,
        t_wrapped_ms=meta.t_wrapped_ms,
        t_unwrapped_ms=int(time.time() * 1000),
        routing_hints=meta.routing_hints,
        checksum=meta.checksum,
        extra=meta.extra,
    )
    # Ensure we do not pass metadata into the ML path
    assert_clean_batch(envelope.payload)
    return envelope.payload, stamped


# Conservative checks to ensure no wrapper artefacts contaminate the payload.
# Extend this as we learn concrete payload shapes across adapters.
_FORBIDDEN_KEYS = {"_wrapper", "envelope_meta", "wrapper_meta"}


def assert_clean_batch(payload: Any) -> None:
    # Dictionaries: must not contain wrapper keys
    if isinstance(payload, dict):
        forbidden = _FORBIDDEN_KEYS.intersection(payload.keys())
        assert not forbidden, f"Payload contains wrapper artefacts: {forbidden}"
        return
    # TorchBatchAdapter.Batch-like: expose tensors and metas but no wrapper keys
    try:
        # Avoid importing torch or adapter types; duck-typing only
        if hasattr(payload, "tensors") and hasattr(payload, "metas"):
            # Expect plain mapping for tensors
            tensors = getattr(payload, "tensors")
            if isinstance(tensors, dict):
                forbidden = _FORBIDDEN_KEYS.intersection(tensors.keys())
                assert not forbidden, f"Batch.tensors contains wrapper artefacts: {forbidden}"
        return
    except Exception:
        return


def emit_wrapper_event(
    event_type: str,
    meta: WrapperMetadata,
    *,
    brain: Optional[Any] = None,
    events: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Safe telemetry hook. Emits a 'metrics' event if the central EventSystem is available.
    Event payload is stable and JSON-serializable.
    """
    try:
        ev = events or getattr(brain, "events", None)
        if ev is None:
            return
        payload = {
            "kind": "batch_wrapper",
            "event": str(event_type),
            "meta": asdict(meta),
        }
        if extra:
            payload["extra"] = extra
        # Align with CoreBrain's subscription to 'metrics'
        ev.trigger("metrics", payload)
    except Exception:
        # Never propagate telemetry issues
        return
