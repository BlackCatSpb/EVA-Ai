#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified smoke test for CogniFlex system with optional GUI entry.

Usage:
  python system_smoke_test.py           # runs checks then launches GUI
  python system_smoke_test.py --no-gui  # runs checks only
"""
import argparse
import importlib
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict
from pathlib import Path

# Ensure project root on sys.path when running as a file from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("system_smoke_test")


def run_step(name: str, func: Callable[[], Any]) -> Dict[str, Any]:
    start = time.time()
    try:
        result = func()
        duration = time.time() - start
        logger.info(f"✔ {name} OK (%.2fs)" % duration)
        return {"step": name, "status": "ok", "duration_sec": duration, "result": result}
    except Exception as e:
        duration = time.time() - start
        logger.exception(f"✖ {name} FAILED (%.2fs): %s" % (duration, e))
        return {"step": name, "status": "failed", "duration_sec": duration, "error": str(e)}


def import_optional(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        logger.debug(f"Optional import failed: {module_name}: {e}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="CogniFlex unified smoke test")
    parser.add_argument("--no-gui", action="store_true", help="Do not launch the GUI after checks")
    args = parser.parse_args(argv)

    results = []

    # 1) Initialize CoreBrain
    def _init_brain():
        core = importlib.import_module("cogniflex.core.core_brain")
        CoreBrain = getattr(core, "CoreBrain")
        brain = CoreBrain()
        return brain

    brain_holder = {"brain": None}

    def _step_init():
        brain_holder["brain"] = _init_brain()
        return {
            "components": [attr for attr in dir(brain_holder["brain"]) if not attr.startswith("_")][:20]
        }

    results.append(run_step("Initialize CoreBrain", _step_init))

    # 2) System health
    def _step_health():
        brain = brain_holder["brain"]
        if not brain:
            raise RuntimeError("Brain not initialized")
        health = None
        if hasattr(brain, "get_system_health") and callable(brain.get_system_health):
            health = brain.get_system_health()
        else:
            # Fallback: try system_state or metrics if exposed differently
            health = {"warning": "get_system_health not found; limited report"}
        # Keep it compact in logs
        logger.info("System health summary: " + json.dumps(health if isinstance(health, dict) else {"health": str(health)})[:500])
        return health

    results.append(run_step("Check system health", _step_health))

    # 3) Process a simple query
    def _step_query():
        brain = brain_holder["brain"]
        if not brain:
            raise RuntimeError("Brain not initialized")
        query_text = "Hello CogniFlex, run smoke test."
        if hasattr(brain, "process_query") and callable(brain.process_query):
            resp = brain.process_query(query_text)
        else:
            # Fallback to a known pipeline if exposed (best-effort)
            raise RuntimeError("process_query not available on CoreBrain")
        # Try to summarize response
        summary = str(resp)
        if isinstance(resp, dict):
            summary = json.dumps({k: resp[k] for k in list(resp)[:5]})
        logger.info("Query response (truncated): " + summary[:500])
        return {"response_preview": summary[:500]}

    results.append(run_step("Process query", _step_query))

    # 4) Ethics assessment (optional, robust)
    def _step_ethics():
        brain = brain_holder["brain"]
        ethics_mod = import_optional("cogniflex.ethics.ethics_core")
        if not ethics_mod or not hasattr(ethics_mod, "EthicsFramework"):
            return {"skipped": True, "reason": "EthicsFramework not available"}
        EthicsFramework = getattr(ethics_mod, "EthicsFramework")
        ef = EthicsFramework(brain=brain)
        ctx = {"query": "Is it ethical to proceed with this sample action?", "metadata": {"source": "smoke_test"}}
        decision = ef.assess_ethics(ctx)
        return {
            "decision": getattr(decision, "decision", None) or getattr(decision, "approved", None),
            "requires_human_review": getattr(decision, "requires_human_review", None),
        }

    results.append(run_step("Ethics assessment", _step_ethics))

    # 5) Adaptation report (optional)
    def _step_adaptation():
        brain = brain_holder["brain"]
        adapt_mod = import_optional("cogniflex.adaptation.adaptation_integration")
        if not adapt_mod:
            return {"skipped": True, "reason": "adaptation_integration not available"}
        report = None
        # Try common function names used previously
        for fname in ("get_system_adaptation_report", "get_adaptation_report"):
            fn = getattr(adapt_mod, fname, None)
            if callable(fn):
                try:
                    # Prefer passing brain, but also support legacy signature
                    try:
                        report = fn(brain=brain)
                    except TypeError:
                        report = fn()
                    break
                except Exception as inner:
                    logger.debug(f"Adaptation call {fname} failed: {inner}")
        if report is None:
            return {"skipped": True, "reason": "no suitable report function"}
        # Truncate for logs
        logger.info("Adaptation report preview: " + json.dumps(report)[:500])
        return {"has_report": True}

    results.append(run_step("Adaptation report", _step_adaptation))

    # Console summary
    ok = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") == "failed")
    logger.info("=" * 60)
    logger.info(f"Smoke test summary: OK={ok}, FAILED={failed}, TOTAL={len(results)}")

    # 6) GUI entry (unless disabled)
    if not args.no_gui:
        def _start_gui():
            gui_mod = import_optional("run_gui")
            if gui_mod:
                for attr in ("main", "run", "start_app", "start"):
                    fn = getattr(gui_mod, attr, None)
                    if callable(fn):
                        logger.info(f"Launching GUI via run_gui.{attr}() ...")
                        return fn()
                # Fallback: if no callable entry found, try executing as a script
            # Final fallback: spawn a new Python process
            import subprocess
            exe = sys.executable or "python"
            script = os.path.join(os.path.dirname(__file__), "run_gui.py")
            if not os.path.isfile(script):
                raise FileNotFoundError("run_gui.py not found at project root")
            logger.info("Launching GUI via subprocess: python run_gui.py")
            return subprocess.Popen([exe, "-X", "utf8", script])

        results.append(run_step("Launch GUI", _start_gui))

    # Exit code: non-zero if any failure
    if any(r.get("status") == "failed" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
