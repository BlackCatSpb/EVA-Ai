"""
Модуль ядра этической рамки для ЕВА - основной класс, инициализация, жизненный цикл, главные проверки
"""
import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any

from .framework_principles import EthicalPrinciple, EthicsPrinciplesMixin
from .framework_checks import EthicalDecision, EthicalAssessment, EthicsAnalysisResult, EthicsChecksMixin
from .framework_violations import EthicsViolationsMixin

logger = logging.getLogger("eva.ethics")

class EthicsFramework(EthicsPrinciplesMixin, EthicsChecksMixin, EthicsViolationsMixin):
    """
    Этическая рамка для ЕВА - управление этическими решениями и проверками.

    Основные функции:
    - Оценка запросов на соответствие этическим принципам
    - Выявление потенциальных этических проблем
    - Генерация рекомендаций по разрешению этических дилемм
    - Отслеживание и анализ этических решений
    """

    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует этическую рамку.

        Args:
            brain: Ссылка на ядро ЕВА (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "ethics_cache")
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()

        os.makedirs(self.cache_dir, exist_ok=True)

        self.principles_file = os.path.join(self.cache_dir, "principles.json")
        self.violations_file = os.path.join(self.cache_dir, "violations.json")
        self.stats_file = os.path.join(self.cache_dir, "stats.json")

        self.principles: Dict[str, EthicalPrinciple] = {}
        self.violations: List[EthicalDecision] = []
        self.stats = {
            "total_assessments": 0,
            "violations_detected": 0,
            "high_severity_violations": 0,
            "resolved_violations": 0,
            "pending_reviews": 0,
            "last_assessment": 0
        }

        self.lock = threading.Lock()

        self._load_configuration()

        self._init_background_services()

        logger.info("Этическая рамка ЕВА инициализирована")
        self.initialized = True

    def is_ready(self) -> bool:
        """Проверяет готовность этической рамки к работе."""
        return self.initialized and len(self.principles) > 0

    def _init_background_services(self):
        """Инициализирует фоновые службы для мониторинга этики."""
        try:
            import threading

            self._violation_monitor_thread = threading.Thread(
                target=self._monitor_violations,
                daemon=True,
                name="EthicsViolationMonitor"
            )
            self._violation_monitor_thread.start()

            self._principle_check_thread = threading.Thread(
                target=self._periodic_principle_check,
                daemon=True,
                name="EthicsPrincipleCheck"
            )
            self._principle_check_thread.start()

            logger.debug("Фоновые службы этической рамки инициализированы")
        except Exception as e:
            logger.error(f"Ошибка инициализации фоновых служб: {e}")

    def _monitor_violations(self):
        """Мониторит нарушения в фоне."""
        while self.running:
            try:
                time.sleep(60)
                if not self.running:
                    break

                self._check_resolved_violations()
            except Exception as e:
                logger.error(f"Ошибка мониторинга нарушений: {e}")

    def _check_resolved_violations(self):
        """Проверяет разрешенные нарушения."""
        try:
            current_time = time.time()
            for violation in list(self.violations):
                if not isinstance(violation, EthicalDecision):
                    continue
                if not violation.resolved and (current_time - violation.timestamp) > 7 * 24 * 3600:
                    violation.resolved = True
                    violation.resolution_timestamp = current_time
                    logger.info(f"Автоматически разрешено старое нарушение: {violation.violation_id}")
        except Exception as e:
            logger.error(f"Ошибка проверки разрешенных нарушений: {e}")

    def _periodic_principle_check(self):
        """Периодически проверяет актуальность принципов."""
        while self.running:
            try:
                time.sleep(3600)
                if not self.running:
                    break

                self._update_principle_stats()
            except Exception as e:
                logger.error(f"Ошибка периодической проверки принципов: {e}")

    def _update_principle_stats(self):
        """Обновляет статистику использования принципов."""
        try:
            for principle_name, principle in self.principles.items():
                principle.last_updated = time.time()
        except Exception as e:
            logger.error(f"Ошибка обновления статистики принципов: {e}")

    def start(self):
        """Запускает фоновые процессы этической рамки."""
        if self.running:
            return

        self.running = True
        logger.info("Этическая рамка запущена")

    def stop(self):
        """Останавливает фоновые процессы этической рамки."""
        if not self.running:
            return

        self.stop_event.set()
        self.running = False
        logger.info("Этическая рамка остановлена")

    def get_system_health(self) -> Dict[str, Any]:
        try:
            score = 1.0 if self.is_ready() else 0.3
        except Exception:
            score = 0.0
        return {
            "health_score": score,
            "status": "healthy" if score > 0.7 else "warning" if score > 0.3 else "critical",
            "initialized": bool(getattr(self, 'initialized', False)),
            "running": bool(getattr(self, 'running', False)),
            "principles_count": len(getattr(self, 'principles', {}) or {}),
            "timestamp": time.time()
        }

    def get_system_status(self) -> Dict[str, Any]:
        return self.get_system_health()
