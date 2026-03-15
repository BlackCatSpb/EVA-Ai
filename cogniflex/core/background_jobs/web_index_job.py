"""
WebIndexJob: индексирование/загрузка материалов из веба для обучения
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from .base_job import BaseJob, CommandPriority

logger = logging.getLogger("cogniflex.core.autopilot.web")


class WebIndexJob(BaseJob):
    job_type = "WebIndexJob"
    resource_class = "IO"
    default_priority = CommandPriority.LOW

    def run(self, context: Dict) -> None:
        brain = self.brain
        web = getattr(brain, 'web_search_engine', None)
        if web is None:
            logger.info("WebIndexJob: web_search_engine недоступен, пропуск")
            return
        try:
            # получаем список запросов из params или формируем дефолтный набор
            params = context.get('job_params') or {}
            queries: List[str] = params.get('queries') or []
            if not queries:
                # дефолтные лёгкие запросы для прогрева кэша/проверки
                queries = [
                    "AI news today",
                    "machine learning recent papers",
                    "knowledge graph basics",
                ]
            max_results = int(params.get('max_results', 5))
            for q in queries:
                try:
                    web.search(q, max_results=max_results)
                except Exception as ex:
                    logger.debug(f"WebIndexJob: ошибка поиска '{q}': {ex}")
        except Exception as e:
            logger.error(f"WebIndexJob: ошибка веб-индексации: {e}", exc_info=True)
