#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal training smoke test for CogniFlex TrainingOrchestrator integration via MLUnit.
Usage:
  python test_training_smoke.py
"""
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("test_training_smoke")


def main():
    # Enable training mode to avoid heavy model autoloading during smoke test
    import os as _os
    _os.environ["COGNIFLEX_TRAINING"] = "1"
    
    # Import brain
    from importlib import import_module
    core = import_module("cogniflex.core.core_brain")
    CoreBrain = getattr(core, "CoreBrain")
    brain = CoreBrain()
    # Ensure full system initialization (creates ml_unit, knowledge_graph, etc.)
    ok = False
    try:
        ok = bool(brain.initialize())
    except Exception as e:
        logger.exception(f"CoreBrain.initialize() failed: {e}")
        ok = False
    if not ok:
        raise RuntimeError("CoreBrain.initialize() failed; cannot proceed with training smoke test")

    # Prepare progress capture from MLUnit orchestrator
    events = []
    def on_evt(evt: dict):
        events.append(evt)
        # Keep log concise
        if evt.get("event") in {"start", "batch_start", "batch_end", "completed", "failed"}:
            logger.info(f"training event: {json.dumps(evt, ensure_ascii=False)[:300]}")
    # Prefer explicit event emitter if available
    if hasattr(brain, "on_training_progress"):
        try:
            # Normalize to list container
            cb_list = getattr(brain, "on_training_progress")
            if cb_list is None:
                setattr(brain, "on_training_progress", [on_evt])
            elif isinstance(cb_list, list):
                cb_list.append(on_evt)
            else:
                setattr(brain, "on_training_progress", [cb_list, on_evt])
        except Exception:
            pass
    elif hasattr(brain, "emit_training_event"):
        # Wrap emitter to collect
        orig = getattr(brain, "emit_training_event")
        def _wrap(payload):
            try:
                events.append(payload)
            finally:
                return orig(payload)
        setattr(brain, "emit_training_event", _wrap)

    # Minimal imported_doc stub
    class ImportedDoc:
        def __init__(self, doc_id: str, segments):
            self.id = doc_id
            self._segments = list(segments)
            self.metadata = {"source": "training_smoke"}
        def iter_segments(self):
            for s in self._segments:
                yield s

    doc = ImportedDoc(
        doc_id="smoke_doc_001",
        segments=[
            "CogniFlex is a modular cognitive system.",
            "It integrates MLUnit, KnowledgeGraph, and a hybrid cache.",
            "This is a small chunk to validate training orchestrator.",
        ],
    )

    # Ensure MLUnit exists
    if not hasattr(brain, "ml_unit") or brain.ml_unit is None:
        raise RuntimeError("brain.ml_unit is not available")

    # Ensure KnowledgeGraph exists
    if not hasattr(brain, "knowledge_graph") or brain.knowledge_graph is None:
        raise RuntimeError("brain.knowledge_graph is not available")

    logger.info("Starting training from document via MLUnit...")
    t0 = time.time()
    result = brain.ml_unit.train_from_document(doc)
    dt = time.time() - t0
    logger.info(f"Training result: {json.dumps(result, ensure_ascii=False)} (took {dt:.2f}s)")

    # Basic assertions for smoke test
    assert isinstance(result, dict), "Result should be a dict"
    assert result.get("status") in {"completed", "failed", "error"}, "Unexpected status"

    # Print a short event summary
    summary = {
        "total_events": len(events),
        "last_event": events[-1]["event"] if events else None,
    }
    print(json.dumps({"ok": True, "result": result, "event_summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
