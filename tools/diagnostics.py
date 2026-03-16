from __future__ import annotations

import json
import logging

from cogniflex.core.core_brain import CoreBrain


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Minimal config; can be overridden by environment or later wiring
    cfg = {
        "use_gpu_if_available": True,
        "prefer_precision": "fp16",
        "allow_bfloat16": True,
    }
    brain = CoreBrain(config=cfg)
    diag = brain.get_runtime_diagnostics()
    print(json.dumps(diag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
