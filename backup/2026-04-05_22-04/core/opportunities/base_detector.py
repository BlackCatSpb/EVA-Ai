"""
BaseDetector: базовый детектор возможностей для Автопилота
"""
from __future__ import annotations
import time
from typing import List, Dict, Any


class BaseDetector:
    name: str = "base"
    cooldown_sec: float = 30.0

    def __init__(self) -> None:
        self._last_probe_ts: float = 0.0

    def probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        now = time.time()
        if now - self._last_probe_ts < float(self.cooldown_sec):
            return []
        self._last_probe_ts = now
        return self._do_probe(context)

    def _do_probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:  # override
        return []
