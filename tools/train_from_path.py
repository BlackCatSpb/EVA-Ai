"""
CLI runner: Import a document (TXT/PDF/EPUB) and train KnowledgeGraph using TrainingOrchestrator.
Usage:
  python -m cogniflex.tools.train_from_path --path <FILE> [--model-id <HF_ID>] [--batch-size 16] [--chunk 512] [--overlap 64]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Ensure package root on path when running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cogniflex.core.core_brain import CoreBrain
from cogniflex.tools.import_pipeline import ImportPipeline
from cogniflex.mlearning.training_orchestrator import TrainingOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cogniflex.tools.train_from_path")


def main():
    parser = argparse.ArgumentParser(description="CogniFlex training runner")
    parser.add_argument("--path", required=True, help="Path to TXT/PDF/EPUB file")
    parser.add_argument("--model-id", default=None, help="Preferred model id (e.g., HuggingFace repo)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for orchestrator")
    parser.add_argument("--chunk", type=int, default=512, help="Target chunk tokens for importer")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap tokens between chunks")
    parser.add_argument("--mode", type=str, default=None, choices=[None, "context_first"], help="Runtime mode (e.g., context_first)")
    # Auto-adaptation flags
    parser.add_argument("--auto-adapt", action="store_true", help="Enable auto adaptation to reduce batch/context under memory pressure")
    parser.add_argument("--mem-high-pct", type=float, default=85.0, help="Memory percent to start reducing batch/context")
    parser.add_argument("--mem-critical-pct", type=float, default=95.0, help="Memory percent to force minimum batch/context")
    parser.add_argument("--min-batch-size", type=int, default=1, help="Minimum batch size when adapting")
    parser.add_argument("--adapt-cooldown-sec", type=float, default=10.0, help="Cooldown seconds between adaptations")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        logger.error(f"File not found: {args.path}")
        return 2

    # Pass mode into CoreBrain to enable policies like ContextFirst
    brain_cfg = {"mode": args.mode} if args.mode else None
    brain = CoreBrain(config=brain_cfg)
    if not brain.initialize():
        logger.error("Failed to initialize CoreBrain")
        return 3

    # Optional psutil import for adaptation
    psutil = None
    if args.auto_adapt:
        try:
            import psutil as _psutil  # type: ignore
            psutil = _psutil
        except Exception:
            psutil = None

    # Adapt chunk/overlap before import if needed
    chunk_tokens = args.chunk
    overlap_tokens = args.overlap
    if args.auto_adapt and psutil is not None:
        try:
            mem_pct = float(psutil.virtual_memory().percent)
            if mem_pct >= args.mem_critical_pct:
                new_chunk = min(chunk_tokens, 256)
                new_overlap = min(overlap_tokens, 32)
                if new_chunk != chunk_tokens or new_overlap != overlap_tokens:
                    logger.warning(f"Auto-adapt (import): chunk {chunk_tokens}->{new_chunk}, overlap {overlap_tokens}->{new_overlap} (critical_mem={mem_pct:.1f}%)")
                chunk_tokens, overlap_tokens = new_chunk, new_overlap
            elif mem_pct >= args.mem_high_pct:
                new_chunk = min(chunk_tokens, 384)
                new_overlap = min(overlap_tokens, 64)
                if new_chunk != chunk_tokens or new_overlap != overlap_tokens:
                    logger.warning(f"Auto-adapt (import): chunk {chunk_tokens}->{new_chunk}, overlap {overlap_tokens}->{new_overlap} (high_mem={mem_pct:.1f}%)")
                chunk_tokens, overlap_tokens = new_chunk, new_overlap
        except Exception:
            pass

    importer = ImportPipeline(brain, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    doc = importer.import_path(args.path)
    logger.info(f"Imported document {doc.id} with {sum(1 for _ in doc.iter_segments())} segments")

    def on_progress(evt: dict):
        ev = evt.get("event")
        if ev == "start":
            logger.info(f"Start training doc={evt.get('document_id')} total={evt.get('total_chunks')} resume_from={evt.get('resume_from')}")
        elif ev == "batch_start":
            logger.info(f"Batch start {evt.get('start_idx')}..{evt.get('end_idx')} (attempt {evt.get('attempt')}/{evt.get('max_attempts')})")
        elif ev == "batch_end":
            logger.info(f"Batch end up to {evt.get('end_idx')}/{evt.get('total_chunks')}")
        elif ev == "batch_retry":
            logger.warning(f"Retry batch {evt.get('start_idx')}..{evt.get('end_idx')}: {evt.get('error')}")
        elif ev == "failed":
            logger.error(f"Training failed: {evt}")
        elif ev == "completed":
            logger.info(f"Completed: {evt.get('processed_chunks')}/{evt.get('total_chunks')}")

    orch = TrainingOrchestrator(
        brain,
        batch_size=args.batch_size,
        overlap_tokens=overlap_tokens,
        progress_cb=on_progress,
        auto_adapt=args.auto_adapt,
        mem_high_pct=args.mem_high_pct,
        mem_critical_pct=args.mem_critical_pct,
        min_batch_size=args.min_batch_size,
        adapt_cooldown_sec=args.adapt_cooldown_sec,
    )
    result = orch.train_from_document(doc, model_id=args.model_id)
    logger.info(f"Training result: {result}")
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
