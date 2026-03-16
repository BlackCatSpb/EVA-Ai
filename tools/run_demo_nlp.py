from __future__ import annotations

import json
import sys
from pathlib import Path
import logging

# Ensure project root on sys.path when running as a file from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cogniflex.core.core_brain import CoreBrain
import torch


def main():
    # Configure logging for demo
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    # Provide config at construction so NLP init sees overrides
    use_cuda = torch.cuda.is_available()
    cfg = {
        "nlp_demo_integration": True,
        # Macroblocks/Hotset integration test overrides
        "nlp_macroblocks_enabled": True,
        # Use absolute path so MacroArchive finds data.bin regardless of CWD
        "nlp_macroblocks_root": str(ROOT / "cogniflex_cache" / "hindex_demo"),
        "nlp_macroblocks_device": "cuda:0" if use_cuda else "cpu",
        "nlp_hotset_enabled": True,
        "nlp_hotset_target_vram_frac": 0.65,
        "nlp_lazy_io_budget_bps": 100663296,  # ~96 MB/s
        "nlp_lazy_max_pending": 6,
    }
    brain = CoreBrain(config=cfg)
    resp = brain.process_query("Demo query for NLP pool")
    metrics = resp.get("metrics", {})
    # Print NLP info and any macroblocks-related metrics if present
    out = {
        "nlp_info": metrics.get("nlp_info", {}),
        "macroblocks": metrics.get("macroblocks", {}),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
