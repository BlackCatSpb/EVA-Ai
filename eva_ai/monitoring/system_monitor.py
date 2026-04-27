#!/usr/bin/env python3
"""
ЕВА Monitoring System
Система мониторинга и метрик для отслеживания состояния компонентов.
"""

import os
import time
import threading
import logging
import platform
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger("eva_ai.monitoring")

@dataclass
class Metric:
    """Метрика системы."""
    name: str
    value: Any
    timestamp: datetime
    tags: Dict[str, str]

@dataclass
class Alert:
    """Предупреждение системы."""
    alert_id: str
    level: str  # 'info', 'warning', 'error', 'critical'
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MetricsCollector:
    """Сборщик метрик системы."""

    def __init__(self):
        self.metrics: List[Metric] = []
        self.custom_metrics: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.max_metrics = 10000  # Максимальное количество хранимых метрик

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None):
        """Записывает метрику."""
        if tags is None:
            tags = {}

        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags
        )

        with self.lock:
            self.metrics.append(metric)
            # Ограничение количества метрик
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]

    def get_metrics(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None,
                   since: Optional[datetime] = None) -> List[Metric]:
        """Возвращает метрики по фильтрам."""
        with self.lock:
            filtered_metrics = self.metrics

            if name:
                filtered_metrics = [m for m in filtered_metrics if m.name == name]

            if tags:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if all(m.tags.get(k) == v for k, v in tags.items())
                ]

            if since:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]

            return filtered_metrics

    def get_latest_metric(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """Возвращает последнюю метрику по имени."""
        metrics = self.get_metrics(name, tags)
        return metrics[-1] if metrics else None

    def get_metric_stats(self, name: str, hours: int = 1) -> Dict[str, Any]:
        """Возвращает статистику по метрике."""
        since = datetime.now() - timedelta(hours=hours)
        metrics = self.get_metrics(name, since=since)

        if not metrics:
            return {"count": 0}

        values = [m.value for m in metrics if isinstance(m.value, (int, float))]

        if not values:
            return {"count": len(metrics)}

        result = {
            "count": len(metrics),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": statistics.mean(values) if values else 0,
            "median": statistics.median(values) if values else 0,
        }
        
        if len(values) > 1:
            result["std_dev"] = statistics.stdev(values)
        else:
            result["std_dev"] = 0
        
        return result

class HealthChecker:
    """Проверяет здоровье компонентов системы."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.component_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.lock = threading.Lock()

    def register_check(self, component_name: str, check_function: Callable[[], Dict[str, Any]]):
        """Регистрирует функцию проверки здоровья компонента."""
        with self.lock:
            self.component_checks[component_name] = check_function

    def check_all_components(self) -> Dict[str, Dict[str, Any]]:
        """Проверяет здоровье всех компонентов."""
        results = {}

        with self.lock:
            for component_name, check_func in self.component_checks.items():
                try:
                    health_status = check_func()
                    results[component_name] = health_status

                    # Записываем метрики
                    for key, value in health_status.items():
                        if isinstance(value, (int, float)):
                            self.metrics_collector.record_metric(
                                f"health.{component_name}.{key}",
                                value,
                                {"component": component_name}
                            )

                except Exception as e:
                    logger.error(f"Ошибка проверки здоровья {component_name}: {e}")
                    results[component_name] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }

        return results

    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает общее состояние здоровья системы."""
        component_health = self.check_all_components()

        # Определяем общее состояние
        error_count = sum(1 for h in component_health.values() if h.get("status") == "error")
        warning_count = sum(1 for h in component_health.values() if h.get("status") == "warning")

        if error_count > 0:
            overall_status = "error"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "components": component_health,
            "error_count": error_count,
            "warning_count": warning_count,
            "timestamp": datetime.now().isoformat()
        }

class AlertManager:
    """Менеджер предупреждений системы."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Callable[[], Optional[Alert]]] = {}
        self.lock = threading.Lock()

    def add_alert_rule(self, rule_name: str, rule_function: Callable[[], Optional[Alert]]):
        """Добавляет правило генерации предупреждений."""
        with self.lock:
            self.alert_rules[rule_name] = rule_function

    def check_alerts(self) -> List[Alert]:
        """Проверяет все правила и возвращает новые предупреждения."""
        new_alerts = []

        with self.lock:
            for rule_name, rule_func in self.alert_rules.items():
                try:
                    alert = rule_func()
                    if alert:
                        self.alerts.append(alert)
                        new_alerts.append(alert)
                        logger.warning(f"New alert: {alert.message}")
                except Exception as e:
                    logger.error(f"Ошибка проверки правила {rule_name}: {e}")

        return new_alerts

    def resolve_alert(self, alert_id: str) -> bool:
        """Разрешает предупреждение."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Возвращает активные предупреждения."""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]

class SystemMonitor:
    """Основной монитор системы."""

    def __init__(self, event_bus=None):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        self.collection_interval = 30  # секунды
        self.event_bus = event_bus

        self._setup_default_checks()
        self._setup_default_alerts()

    def _setup_default_checks(self):
        """Настраивает проверки здоровья по умолчанию."""

        # Проверка системных ресурсов
        def system_resources_check():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Определяем корневой диск для Windows и Unix
                if platform.system() == 'Windows':
                    disk_path = os.environ.get('SystemDrive', 'C:') + '\\'
                else:
                    disk_path = '/'
                
                disk = psutil.disk_usage(disk_path)

                return {
                    "status": "healthy" if cpu_percent < 80 and memory.percent < 90 else "warning",
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        self.health_checker.register_check("system_resources", system_resources_check)

        # Проверка Python процесса
        def python_process_check():
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_times = process.cpu_times()

                return {
                    "status": "healthy",
                    "memory_rss_mb": memory_info.rss / (1024**2),
                    "memory_vms_mb": memory_info.vms / (1024**2),
                    "cpu_user_seconds": cpu_times.user,
                    "cpu_system_seconds": cpu_times.system,
                    "threads_count": process.num_threads(),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        self.health_checker.register_check("python_process", python_process_check)

    def _setup_default_alerts(self):
        """Настраивает предупреждения по умолчанию."""

        # Предупреждение о высокой загрузке CPU
        def high_cpu_alert():
            cpu_metric = self.metrics_collector.get_latest_metric("health.system_resources.cpu_percent")
            if cpu_metric and cpu_metric.value > 90:
                return Alert(
                    alert_id=f"high_cpu_{int(time.time())}",
                    level="warning",
                    message=f"High CPU usage: {cpu_metric.value:.1f}%",
                    component="system_resources",
                    timestamp=datetime.now()
                )
            return None

        self.alert_manager.add_alert_rule("high_cpu", high_cpu_alert)

        # Предупреждение о нехватке памяти
        def low_memory_alert():
            mem_metric = self.metrics_collector.get_latest_metric("health.system_resources.memory_percent")
            if mem_metric and mem_metric.value > 95:
                return Alert(
                    alert_id=f"low_memory_{int(time.time())}",
                    level="critical",
                    message=f"Low memory: {mem_metric.value:.1f}% used",
                    component="system_resources",
                    timestamp=datetime.now()
                )
            return None

        self.alert_manager.add_alert_rule("low_memory", low_memory_alert)

    def start_monitoring(self):
        """Запускает мониторинг системы."""
        if self.running:
            return

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Останавливает мониторинг."""
        if not self.running:
            return
        
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            # Даем потоку время завершиться
            self.monitoring_thread.join(timeout=2)
            if self.monitoring_thread.is_alive():
                logger.warning("Поток мониторинга не завершился за 2 секунды")
        logger.info("System monitoring stopped")

    def _monitoring_loop(self):
        """Основной цикл мониторинга."""
        while self.running:
            try:
                # Проверяем здоровье компонентов
                health_status = self.health_checker.check_all_components()

                # Проверяем предупреждения
                new_alerts = self.alert_manager.check_alerts()

                # Записываем системные метрики
                self._record_system_metrics()

                # Логируем проблемы
                self._log_issues(health_status, new_alerts)

            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")

            # Используем более короткий интервал и проверяем флаг running
            for _ in range(self.collection_interval):
                if not self.running:
                    break
                time.sleep(1)

    def _record_system_metrics(self):
        """Записывает системные метрики."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent()
            self.metrics_collector.record_metric("system.cpu_percent", cpu_percent)

            # Memory
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric("system.memory_percent", memory.percent)
            self.metrics_collector.record_metric("system.memory_used_gb", memory.used / (1024**3))

            # Disk
            disk = psutil.disk_usage('/') if platform.system() != 'Windows' else psutil.disk_usage(os.environ.get('SystemDrive', 'C:') + '\\')
            self.metrics_collector.record_metric("system.disk_percent", disk.percent)

            # Network (если доступно)
            try:
                net = psutil.net_io_counters()
                self.metrics_collector.record_metric("system.net_bytes_sent", net.bytes_sent)
                self.metrics_collector.record_metric("system.net_bytes_recv", net.bytes_recv)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Ошибка записи системных метрик: {e}")

    def _log_issues(self, health_status: Dict[str, Dict[str, Any]], alerts: List[Alert]):
        """Логирует проблемы системы и публикует события в EventBus."""
        unhealthy_components = [
            name for name, status in health_status.items()
            if status.get("status") in ["error", "warning"]
        ]

        if unhealthy_components:
            logger.warning(f"Unhealthy components: {', '.join(unhealthy_components)}")

            # H4 FIX: Публикуем в EventBus
            if self.event_bus:
                try:
                    from eva_ai.core.event_bus import EventPriority
                    self.event_bus.publish(
                        "monitor.warning",
                        {
                            "components": unhealthy_components,
                            "health_status": health_status
                        },
                        priority=EventPriority.HIGH
                    )
                except Exception as e:
                    logger.debug(f"Failed to publish monitor warning: {e}")

        for alert in alerts:
            logger.warning(f"Alert [{alert.level}]: {alert.message}")

            # H4 FIX: Публикуем событие алерта
            if self.event_bus:
                try:
                    from eva_ai.core.event_bus import EventPriority
                    self.event_bus.publish(
                        "monitor.alert",
                        {
                            "alert_id": alert.alert_id,
                            "level": alert.level,
                            "message": alert.message,
                            "component": alert.component
                        },
                        priority=EventPriority.HIGH if alert.level == "critical" else EventPriority.NORMAL
                    )
                except Exception as e:
                    logger.debug(f"Failed to publish alert event: {e}")

    def register_component_check(self, component_name: str, check_function: Callable[[], Dict[str, Any]]):
        """Регистрирует проверку здоровья компонента."""
        self.health_checker.register_check(component_name, check_function)

    def get_system_status(self) -> Dict[str, Any]:
        """Возвращает текущий статус системы."""
        health = self.health_checker.get_system_health()
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "health": health,
            "active_alerts": [
                {"alert_id": a.alert_id, "level": a.level, "message": a.message, 
                 "component": a.component, "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                 "resolved": a.resolved, "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None}
                for a in active_alerts
            ],
            "metrics_summary": {
                "total_metrics": len(self.metrics_collector.metrics),
                "latest_collection": datetime.now().isoformat()
            }
        }

    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """Возвращает отчет о производительности."""
        cpu_stats = self.metrics_collector.get_metric_stats("system.cpu_percent", hours)
        memory_stats = self.metrics_collector.get_metric_stats("system.memory_percent", hours)

        return {
            "period_hours": hours,
            "cpu_usage": cpu_stats,
            "memory_usage": memory_stats,
            "generated_at": datetime.now().isoformat()
        }

# Глобальный экземпляр монитора системы
system_monitor = SystemMonitor()

def get_system_monitor() -> SystemMonitor:
    """Возвращает глобальный монитор системы."""
    return system_monitor

# Вспомогательные функции для интеграции
def record_metric(name: str, value: Any, tags: Optional[Dict[str, str]] = None):
    """Записывает метрику."""
    system_monitor.metrics_collector.record_metric(name, value, tags)

def get_system_health() -> Dict[str, Any]:
    """Возвращает состояние здоровья системы."""
    return system_monitor.health_checker.get_system_health()

def create_performance_report(hours: int = 1) -> Dict[str, Any]:
    """Создает отчет о производительности."""
    return system_monitor.get_performance_report(hours)

# Автоматический запуск мониторинга при импорте
def _auto_start_monitoring():
    """Автоматически запускает мониторинг."""
    try:
        system_monitor.start_monitoring()
        logger.info("System monitoring auto-started")
    except Exception as e:
        logger.error(f"Failed to auto-start monitoring: {e}")

# Запускаем мониторинг при импорте модуля
_auto_start_monitoring()
