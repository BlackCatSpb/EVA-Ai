"""
WebDiscoveryDetector: запускает веб-индексацию при простое
"""
from __future__ import annotations
from typing import List, Dict, Any
from .base_detector import BaseDetector


class WebDiscoveryDetector(BaseDetector):
    name = "web_discovery"
    cooldown_sec = 120.0

    def _do_probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        brain = context.get('brain')
        if not brain:
            return []
        web = getattr(brain, 'web_search_engine', None)
        if not web:
            return []
        # Если есть метод статуса — пробуем не запускать при выполняющейся индексации
        try:
            busy = False
            if hasattr(web, 'is_indexing'):
                busy = bool(web.is_indexing() if callable(web.is_indexing) else web.is_indexing)
            elif hasattr(web, 'is_busy'):
                busy = bool(web.is_busy() if callable(web.is_busy) else web.is_busy)
            if busy:
                return []
        except Exception:
            pass
        # Можно расширить params будущими настройками (queries, max_results)
        return [{"job_type": "WebIndexJob", "params": {}}]


class WebDiscoveryDetector(BaseDetector):
    name = "web_discovery"
    cooldown_sec = 120.0

    def _do_probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        brain = context.get('brain')
        if not brain:
            return []
        web = getattr(brain, 'web_search_engine', None)
        if not web:
            return []
        # Если есть метод статуса — пробуем не запускать при выполняющейся индексации
        try:
            busy = False
            if hasattr(web, 'is_indexing'):
                busy = bool(web.is_indexing() if callable(web.is_indexing) else web.is_indexing)
            elif hasattr(web, 'is_busy'):
                busy = bool(web.is_busy() if callable(web.is_busy) else web.is_busy)
            if busy:
                return []
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Error checking web search engine status: {e}")
        # Можно расширить params будущими настройками (queries, max_results)
        return [{"job_type": "WebIndexJob", "params": {}}]
