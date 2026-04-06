"""
AutopilotCache: простой персистентный кэш/журнал для Автопилота
- JSONL журнал событий (детекции, планирования, выполнений)
- Ключ-значение с namespace
"""
from __future__ import annotations
import os
import json
import time
import threading
from typing import Any, Optional, Dict


class AutopilotCache:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.join(base_dir, "autopilot")
        os.makedirs(self.base_dir, exist_ok=True)
        self.kv_dir = os.path.join(self.base_dir, "kv")
        os.makedirs(self.kv_dir, exist_ok=True)
        self.log_path = os.path.join(self.base_dir, "events.jsonl")
        self._lock = threading.RLock()

    def put(self, ns: str, key: str, value: Any) -> None:
        with self._lock:
            ns_dir = os.path.join(self.kv_dir, ns)
            os.makedirs(ns_dir, exist_ok=True)
            path = os.path.join(ns_dir, f"{key}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)

    def get(self, ns: str, key: str) -> Optional[Any]:
        path = os.path.join(self.kv_dir, ns, f"{key}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def append_event(self, kind: str, payload: Dict[str, Any]) -> None:
        evt = {
            "ts": time.time(),
            "kind": kind,
            "data": payload,
        }
        line = json.dumps(evt, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
