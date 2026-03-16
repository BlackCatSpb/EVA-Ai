import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from cogniflex.mlearning.storage.fractal_store import FractalWeightStore

logger = logging.getLogger("cogniflex.scripts.batch_load_to_graph")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                logger.exception("Bad JSON line skipped")
    return data


def to_min_graph(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a minimal graph from JSONL records.
    Supported input formats per record:
      - {"id": str, "text": str}
      - {"nodes": [...], "edges": [...]}  (passes through)
    """
    # Pass-through if already a graph-like structure in a single record
    if len(records) == 1 and isinstance(records[0], dict) and "nodes" in records[0] and "edges" in records[0]:
        g = records[0]
        g.setdefault("meta", {})
        g["meta"].setdefault("source", "batch_loader")
        return g

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for i, rec in enumerate(records):
        if "nodes" in rec and "edges" in rec:
            # Merge nodes/edges from provided graph fragments
            for n in rec.get("nodes", []):
                if isinstance(n, dict):
                    nodes.append(n)
            for e in rec.get("edges", []):
                if isinstance(e, dict):
                    edges.append(e)
            continue

        rid = str(rec.get("id", f"doc_{i}"))
        text = rec.get("text")
        if not text:
            # skip empty
            continue
        nodes.append({"id": rid, "node_type": "doc", "content": text})
        # lightweight sentence split by periods to form pseudo concepts
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for j, s in enumerate(sentences[:5]):
            nid = f"{rid}#s{j}"
            nodes.append({"id": nid, "node_type": "sentence", "content": s})
            edges.append({"source": rid, "target": nid, "relation_type": "contains"})

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {"source": "batch_loader", "counts": {"nodes": len(nodes), "edges": len(edges)}},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-load JSONL into fractal knowledge graph store")
    ap.add_argument("--input", required=True, help="Path to JSONL with records or a single graph object")
    ap.add_argument("--output", required=True, help="Output directory for the fractal store")
    ap.add_argument("--model-id", default="external_ingest", help="Model id tag for the store")
    ap.add_argument("--levels", type=int, default=4, help="Fractal levels")
    ap.add_argument("--block-size", type=int, default=64, help="Block size for base containers")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output)

    if not inp.exists():
        logger.error("Input file does not exist: %s", inp)
        return 2

    logger.info("Reading input JSONL: %s", inp)
    records = read_jsonl(inp)
    if not records:
        logger.error("No valid records found")
        return 2

    logger.info("Building minimal knowledge graph...")
    graph = to_min_graph(records)

    store = FractalWeightStore(block_size=args.block_size, fractal_levels=args.levels)
    store.model_id = args.model_id

    logger.info("Packing graph into fractal store...")
    store.pack_knowledge_graph(graph)

    logger.info("Validating packing...")
    v = store.validate_knowledge_graph_packing()
    if not v.get("ok"):
        logger.error("Validation failed: %s", v)
        return 3
    logger.info("Validation OK: %s blocks, total length %s", v.get("total_blocks"), v.get("total_length"))

    logger.info("Saving store atomically to: %s", out_dir)
    res = store.save_to_disk_atomic(str(out_dir))
    if not res.get("ok"):
        logger.error("Save failed: %s", res)
        return 4

    logger.info("Saved. checksum=%s", res.get("checksum"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
