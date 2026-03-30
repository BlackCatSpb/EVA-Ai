"""
ContextFirstPolicy: адаптивная политика, отдающая приоритет ширине контекста
при ограничениях памяти/CPU. Выполняет безопасные (неинвазивные) настройки
через brain.config и мягко взаимодействует с доступными компонентами.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # psutil опционален
    psutil = None  # type: ignore

logger = logging.getLogger("eva.core.context_first_policy")


class ContextFirstPolicy:
    def __init__(self, brain: Any, config_overrides: Optional[Dict[str, Any]] = None) -> None:
        self.brain = brain
        self.overrides = config_overrides or {}

    def _safe_set(self, key: str, value: Any) -> None:
        try:
            self.brain.config[key] = value
        except Exception:
            pass

    def _resource_hint(self) -> Dict[str, Any]:
        if not psutil:
            return {"mem_total_mb": None, "mem_used_pct": None, "cpu_pct": None}
        try:
            vm = psutil.virtual_memory()
            cpu_pct = psutil.cpu_percent(interval=0.0)
            return {
                "mem_total_mb": int(vm.total / (1024 * 1024)),
                "mem_used_pct": int(vm.percent),
                "cpu_pct": int(cpu_pct),
            }
        except Exception:
            return {"mem_total_mb": None, "mem_used_pct": None, "cpu_pct": None}

    def apply(self) -> None:
        """Применяет настройки приоритета ширины контекста. Безопасно.
        Только выставляет параметры в brain.config и логирует подсказки.
        """
        hints = self._resource_hint()
        logger.info(f"ContextFirstPolicy активирована, ресурсы: {hints}")

        # Базовые настройки на ширину контекста:
        # - увеличиваем top_k
        # - увеличиваем размер фрагмента и overlap для индексации
        # - уменьшаем batch_size для устойчивости
        # - включаем дисковую часть гибридного кэша
        defaults = {
            "retrieval_top_k": 32,
            "graph_top_k": 64,
            "chunk_size": 1536,
            "chunk_overlap": 256,
            "batch_size": 8,
            "hybrid_cache_disk_enabled": True,
            "mode": "context_first",
        }
        # Пользовательские оверрайды имеют приоритет
        defaults.update(self.overrides)
        for k, v in defaults.items():
            self._safe_set(k, v)

        # Мягкие подсказки компонентам, если у них есть совместимые свойства/методы
        # (не полагаемся на их наличие; все операции условные)
        tp = getattr(self.brain, "text_processor", None)
        if tp:
            try:
                # Если у гибридного кэша есть режим диска — активируем
                hc = getattr(tp, "hybrid_cache", None)
                if hc and hasattr(hc, "enable_disk"):
                    try:
                        hc.enable_disk(True)
                        logger.info("Гибридный кэш: диск активирован через enable_disk(True)")
                    except Exception:
                        pass
            except Exception:
                pass

        # Подсказки для компонентов, если они их поддерживают (всё условно):
        # knowledge_graph / retriever могут уважать top_k из brain.config
        # обучающие модули могут уважать batch_size, chunk_size, chunk_overlap
        logger.info("ContextFirstPolicy применена: высокое top_k, крупные chunk/overlap, малый batch, диск-кэш")


