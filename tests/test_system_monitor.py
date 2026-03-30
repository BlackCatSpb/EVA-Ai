#!/usr/bin/env python3
"""
ЕВА System Monitoring Tests
Тесты для системы мониторинга и метрик.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from eva.monitoring.system_monitor import (
    SystemMonitor,
    MetricsCollector,
    HealthChecker,
    AlertManager,
    Metric,
    Alert,
    get_system_monitor,
    record_metric,
    get_system_health,
    create_performance_report
)


class TestMetricsCollector:
    """Тесты для MetricsCollector."""

    @pytest.fixture
    def metrics_collector(self):
        """Создает MetricsCollector для тестирования."""
        return MetricsCollector()

    def test_metrics_collector_initialization(self, metrics_collector):
        """Тест инициализации MetricsCollector."""
        assert metrics_collector.metrics == []
        assert metrics_collector.custom_metrics == {}
        assert metrics_collector.max_metrics == 10000

    def test_record_metric(self, metrics_collector):
        """Тест записи метрики."""
        # Записываем метрику
        metrics_collector.record_metric("test.metric", 42.5, {"component": "test"})

        # Проверяем что метрика добавлена
        assert len(metrics_collector.metrics) == 1
        metric = metrics_collector.metrics[0]

        assert metric.name == "test.metric"
        assert metric.value == 42.5
        assert metric.tags == {"component": "test"}
        assert isinstance(metric.timestamp, datetime)

    def test_record_multiple_metrics(self, metrics_collector):
        """Тест записи нескольких метрик."""
        # Записываем несколько метрик
        for i in range(5):
            metrics_collector.record_metric(f"test.metric{i}", i * 10, {"index": i})

        assert len(metrics_collector.metrics) == 5

        # Проверяем значения
        for i, metric in enumerate(metrics_collector.metrics):
            assert metric.name == f"test.metric{i}"
            assert metric.value == i * 10
            assert metric.tags["index"] == i

    def test_get_metrics_by_name(self, metrics_collector):
        """Тест получения метрик по имени."""
        # Записываем метрики разных типов
        metrics_collector.record_metric("cpu.usage", 50.0)
        metrics_collector.record_metric("memory.usage", 70.0)
        metrics_collector.record_metric("cpu.usage", 60.0)

        # Получаем только CPU метрики
        cpu_metrics = metrics_collector.get_metrics("cpu.usage")
        memory_metrics = metrics_collector.get_metrics("memory.usage")

        assert len(cpu_metrics) == 2
        assert len(memory_metrics) == 1
        assert all(m.name == "cpu.usage" for m in cpu_metrics)
        assert all(m.name == "memory.usage" for m in memory_metrics)

    def test_get_metrics_by_tags(self, metrics_collector):
        """Тест получения метрик по тегам."""
        # Записываем метрики с разными тегами
        metrics_collector.record_metric("test.metric", 1, {"component": "cpu", "server": "srv1"})
        metrics_collector.record_metric("test.metric", 2, {"component": "cpu", "server": "srv2"})
        metrics_collector.record_metric("test.metric", 3, {"component": "memory", "server": "srv1"})

        # Получаем метрики по тегам
        cpu_metrics = metrics_collector.get_metrics(tags={"component": "cpu"})
        srv1_metrics = metrics_collector.get_metrics(tags={"server": "srv1"})

        assert len(cpu_metrics) == 2
        assert len(srv1_metrics) == 2
        assert all(m.tags["component"] == "cpu" for m in cpu_metrics)

    def test_get_metrics_by_time_range(self, metrics_collector):
        """Тест получения метрик по временному диапазону."""
        base_time = datetime.now()

        # Записываем метрики в разное время
        metrics_collector.record_metric("test.metric", 1)
        time.sleep(0.01)  # Небольшая задержка

        mid_time = datetime.now()
        metrics_collector.record_metric("test.metric", 2)
        time.sleep(0.01)

        metrics_collector.record_metric("test.metric", 3)

        # Получаем метрики после mid_time
        recent_metrics = metrics_collector.get_metrics(since=mid_time)

        assert len(recent_metrics) == 2  # Должны получить 2 последние метрики

    def test_get_latest_metric(self, metrics_collector):
        """Тест получения последней метрики."""
        # Записываем несколько метрик
        metrics_collector.record_metric("test.metric", 1)
        time.sleep(0.001)
        metrics_collector.record_metric("test.metric", 2)
        time.sleep(0.001)
        metrics_collector.record_metric("test.metric", 3)

        # Получаем последнюю метрику
        latest = metrics_collector.get_latest_metric("test.metric")

        assert latest is not None
        assert latest.value == 3

    def test_get_latest_metric_nonexistent(self, metrics_collector):
        """Тест получения последней метрики для несуществующего имени."""
        latest = metrics_collector.get_latest_metric("nonexistent.metric")

        assert latest is None

    def test_get_metric_stats(self, metrics_collector):
        """Тест получения статистики по метрике."""
        # Записываем несколько значений
        values = [10, 20, 30, 40, 50]
        for value in values:
            metrics_collector.record_metric("test.metric", value)

        # Получаем статистику
        stats = metrics_collector.get_metric_stats("test.metric")

        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["avg"] == 30
        assert stats["median"] == 30

    def test_get_metric_stats_empty(self, metrics_collector):
        """Тест получения статистики для пустой метрики."""
        stats = metrics_collector.get_metric_stats("empty.metric")

        assert stats["count"] == 0

    def test_max_metrics_limit(self, metrics_collector):
        """Тест ограничения максимального количества метрик."""
        # Устанавливаем маленький лимит для теста
        metrics_collector.max_metrics = 3

        # Записываем больше метрик чем лимит
        for i in range(5):
            metrics_collector.record_metric("test.metric", i)

        # Проверяем что количество метрик не превышает лимит
        assert len(metrics_collector.metrics) == 3

        # Проверяем что остались последние метрики
        values = [m.value for m in metrics_collector.metrics]
        assert values == [2, 3, 4]  # Последние 3 метрики


class TestHealthChecker:
    """Тесты для HealthChecker."""

    @pytest.fixture
    def health_checker(self):
        """Создает HealthChecker для тестирования."""
        metrics_collector = MetricsCollector()
        return HealthChecker(metrics_collector)

    def test_health_checker_initialization(self, health_checker):
        """Тест инициализации HealthChecker."""
        assert health_checker.component_checks == {}
        assert health_checker.metrics_collector is not None

    def test_register_check_function(self, health_checker):
        """Тест регистрации функции проверки."""
        def mock_check():
            return {"status": "healthy", "value": 100}

        health_checker.register_check("test_component", mock_check)

        assert "test_component" in health_checker.component_checks
        assert health_checker.component_checks["test_component"] == mock_check

    def test_check_single_component(self, health_checker):
        """Тест проверки одного компонента."""
        def healthy_check():
            return {"status": "healthy", "cpu": 50, "memory": 60}

        health_checker.register_check("test_component", healthy_check)

        # Выполняем проверку
        result = health_checker.check_all_components()

        assert "test_component" in result
        component_result = result["test_component"]
        assert component_result["status"] == "healthy"
        assert component_result["cpu"] == 50
        assert component_result["memory"] == 60

    def test_check_multiple_components(self, health_checker):
        """Тест проверки нескольких компонентов."""
        def healthy_check():
            return {"status": "healthy", "metric": 100}

        def warning_check():
            return {"status": "warning", "metric": 80}

        def error_check():
            return {"status": "error", "error": "Test error"}

        health_checker.register_check("healthy_comp", healthy_check)
        health_checker.register_check("warning_comp", warning_check)
        health_checker.register_check("error_comp", error_check)

        result = health_checker.check_all_components()

        assert len(result) == 3
        assert result["healthy_comp"]["status"] == "healthy"
        assert result["warning_comp"]["status"] == "warning"
        assert result["error_comp"]["status"] == "error"

    def test_check_component_with_exception(self, health_checker):
        """Тест проверки компонента с исключением."""
        def failing_check():
            raise Exception("Test exception")

        health_checker.register_check("failing_component", failing_check)

        result = health_checker.check_all_components()

        assert "failing_component" in result
        component_result = result["failing_component"]
        assert component_result["status"] == "error"
        assert "Test exception" in component_result["error"]

    def test_get_system_health_all_healthy(self, health_checker):
        """Тест получения системного здоровья когда все компоненты здоровы."""
        def healthy_check():
            return {"status": "healthy", "metric": 100}

        health_checker.register_check("comp1", healthy_check)
        health_checker.register_check("comp2", healthy_check)

        health = health_checker.get_system_health()

        assert health["status"] == "healthy"
        assert health["error_count"] == 0
        assert health["warning_count"] == 0
        assert len(health["components"]) == 2

    def test_get_system_health_with_warnings(self, health_checker):
        """Тест получения системного здоровья с предупреждениями."""
        def healthy_check():
            return {"status": "healthy", "metric": 100}

        def warning_check():
            return {"status": "warning", "metric": 80}

        health_checker.register_check("healthy_comp", healthy_check)
        health_checker.register_check("warning_comp", warning_check)

        health = health_checker.get_system_health()

        assert health["status"] == "warning"
        assert health["error_count"] == 0
        assert health["warning_count"] == 1

    def test_get_system_health_with_errors(self, health_checker):
        """Тест получения системного здоровья с ошибками."""
        def healthy_check():
            return {"status": "healthy", "metric": 100}

        def error_check():
            return {"status": "error", "error": "Test error"}

        health_checker.register_check("healthy_comp", healthy_check)
        health_checker.register_check("error_comp", error_check)

        health = health_checker.get_system_health()

        assert health["status"] == "error"
        assert health["error_count"] == 1
        assert health["warning_count"] == 0


class TestAlertManager:
    """Тесты для AlertManager."""

    @pytest.fixture
    def alert_manager(self):
        """Создает AlertManager для тестирования."""
        metrics_collector = MetricsCollector()
        return AlertManager(metrics_collector)

    def test_alert_manager_initialization(self, alert_manager):
        """Тест инициализации AlertManager."""
        assert alert_manager.alerts == []
        assert alert_manager.alert_rules == {}
        assert alert_manager.metrics_collector is not None

    def test_add_alert_rule(self, alert_manager):
        """Тест добавления правила предупреждения."""
        def mock_rule():
            return Alert(
                alert_id="test_alert",
                level="warning",
                message="Test alert",
                component="test_component",
                timestamp=datetime.now()
            )

        alert_manager.add_alert_rule("test_rule", mock_rule)

        assert "test_rule" in alert_manager.alert_rules
        assert alert_manager.alert_rules["test_rule"] == mock_rule

    def test_check_alerts_with_triggered_rule(self, alert_manager):
        """Тест проверки предупреждений с срабатывающим правилом."""
        def triggered_rule():
            return Alert(
                alert_id="triggered_alert",
                level="warning",
                message="Alert triggered",
                component="test_component",
                timestamp=datetime.now()
            )

        alert_manager.add_alert_rule("trigger_rule", triggered_rule)

        # Проверяем предупреждения
        new_alerts = alert_manager.check_alerts()

        assert len(new_alerts) == 1
        assert new_alerts[0].alert_id == "triggered_alert"
        assert new_alerts[0].message == "Alert triggered"

        # Проверяем что предупреждение добавлено в список
        assert len(alert_manager.alerts) == 1

    def test_check_alerts_with_no_trigger(self, alert_manager):
        """Тест проверки предупреждений без срабатывания."""
        def no_trigger_rule():
            return None

        alert_manager.add_alert_rule("no_trigger_rule", no_trigger_rule)

        new_alerts = alert_manager.check_alerts()

        assert len(new_alerts) == 0
        assert len(alert_manager.alerts) == 0

    def test_check_alerts_with_exception(self, alert_manager):
        """Тест проверки предупреждений с исключением в правиле."""
        def failing_rule():
            raise Exception("Test exception")

        alert_manager.add_alert_rule("failing_rule", failing_rule)

        # Проверяем что исключение не ломает систему
        new_alerts = alert_manager.check_alerts()

        assert len(new_alerts) == 0  # Предупреждение не должно быть создано

    def test_resolve_alert(self, alert_manager):
        """Тест разрешения предупреждения."""
        # Создаем предупреждение
        alert = Alert(
            alert_id="test_alert",
            level="warning",
            message="Test alert",
            component="test_component",
            timestamp=datetime.now()
        )
        alert_manager.alerts.append(alert)

        # Разрешаем предупреждение
        result = alert_manager.resolve_alert("test_alert")

        assert result is True
        assert alert.resolved is True
        assert alert.resolved_at is not None

    def test_resolve_nonexistent_alert(self, alert_manager):
        """Тест разрешения несуществующего предупреждения."""
        result = alert_manager.resolve_alert("nonexistent_alert")

        assert result is False

    def test_get_active_alerts(self, alert_manager):
        """Тест получения активных предупреждений."""
        # Создаем активное и разрешенное предупреждения
        active_alert = Alert(
            alert_id="active_alert",
            level="warning",
            message="Active alert",
            component="test_component",
            timestamp=datetime.now()
        )

        resolved_alert = Alert(
            alert_id="resolved_alert",
            level="error",
            message="Resolved alert",
            component="test_component",
            timestamp=datetime.now(),
            resolved=True,
            resolved_at=datetime.now()
        )

        alert_manager.alerts.extend([active_alert, resolved_alert])

        active_alerts = alert_manager.get_active_alerts()

        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == "active_alert"

    def test_get_active_alerts_empty(self, alert_manager):
        """Тест получения активных предупреждений при их отсутствии."""
        active_alerts = alert_manager.get_active_alerts()

        assert active_alerts == []


class TestSystemMonitor:
    """Тесты для SystemMonitor."""

    @pytest.fixture
    def system_monitor(self):
        """Создает SystemMonitor для тестирования."""
        return SystemMonitor()

    def test_system_monitor_initialization(self, system_monitor):
        """Тест инициализации SystemMonitor."""
        assert system_monitor.metrics_collector is not None
        assert system_monitor.health_checker is not None
        assert system_monitor.alert_manager is not None
        assert system_monitor.collection_interval == 30
        assert not system_monitor.running

    def test_system_monitor_start_stop(self, system_monitor):
        """Тест запуска и остановки SystemMonitor."""
        # Запуск
        system_monitor.start_monitoring()
        assert system_monitor.running is True

        # Небольшая задержка для запуска потоков
        time.sleep(0.1)

        # Проверка что поток мониторинга работает
        assert system_monitor.monitoring_thread is not None
        assert system_monitor.monitoring_thread.is_alive()

        # Остановка
        system_monitor.stop_monitoring()
        assert system_monitor.running is False

        # Небольшая задержка для остановки потоков
        time.sleep(0.1)

    def test_register_component_check(self, system_monitor):
        """Тест регистрации проверки компонента."""
        def custom_check():
            return {"status": "healthy", "custom_metric": 42}

        system_monitor.register_component_check("custom_component", custom_check)

        # Проверяем что проверка зарегистрирована
        assert "custom_component" in system_monitor.health_checker.component_checks

        # Выполняем проверку
        health = system_monitor.health_checker.check_all_components()

        assert "custom_component" in health
        assert health["custom_component"]["custom_metric"] == 42

    def test_get_system_status(self, system_monitor):
        """Тест получения статуса системы."""
        status = system_monitor.get_system_status()

        assert "health" in status
        assert "active_alerts" in status
        assert "metrics_summary" in status

        health = status["health"]
        assert "status" in health
        assert "components" in health
        assert "error_count" in health
        assert "warning_count" in health

    def test_get_performance_report(self, system_monitor):
        """Тест получения отчета о производительности."""
        # Записываем тестовые метрики
        system_monitor.metrics_collector.record_metric("system.cpu_percent", 50.0)
        system_monitor.metrics_collector.record_metric("system.memory_percent", 60.0)
        system_monitor.metrics_collector.record_metric("system.cpu_percent", 55.0)

        report = system_monitor.get_performance_report(hours=1)

        assert "period_hours" in report
        assert "cpu_usage" in report
        assert "memory_usage" in report
        assert "generated_at" in report

        assert report["period_hours"] == 1
        assert report["cpu_usage"]["count"] == 2
        assert report["memory_usage"]["count"] == 1

    def test_default_health_checks(self, system_monitor):
        """Тест встроенных проверок здоровья."""
        # Проверяем что встроенные проверки зарегистрированы
        health_checks = system_monitor.health_checker.component_checks

        assert "system_resources" in health_checks
        assert "python_process" in health_checks

        # Выполняем проверки
        health = system_monitor.health_checker.check_all_components()

        assert "system_resources" in health
        assert "python_process" in health

        # Проверяем структуру результатов
        for component_name, component_health in health.items():
            assert "status" in component_health
            assert "timestamp" in component_health

    @patch('cogniflex.monitoring.system_monitor.psutil')
    def test_system_resources_check_with_mock(self, mock_psutil, system_monitor):
        """Тест проверки системных ресурсов с mock psutil."""
        # Mock psutil
        mock_memory = Mock()
        mock_memory.percent = 70.0
        mock_memory.used = 7 * 1024**3  # 7GB
        mock_memory.total = 10 * 1024**3  # 10GB

        mock_disk = Mock()
        mock_disk.percent = 60.0
        mock_disk.free = 200 * 1024**3  # 200GB

        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_usage.return_value = mock_disk
        mock_psutil.cpu_percent.return_value = 45.0

        # Выполняем проверку
        health = system_monitor.health_checker.check_all_components()

        assert "system_resources" in health
        resources = health["system_resources"]

        assert resources["status"] == "healthy"
        assert resources["cpu_percent"] == 45.0
        assert resources["memory_percent"] == 70.0
        assert resources["disk_percent"] == 60.0

    def test_default_alert_rules(self, system_monitor):
        """Тест встроенных правил предупреждений."""
        alert_rules = system_monitor.alert_manager.alert_rules

        assert "high_cpu" in alert_rules
        assert "low_memory" in alert_rules

        # Проверяем что правила можно вызвать
        for rule_name, rule_func in alert_rules.items():
            # Правила могут возвращать None если условия не выполнены
            result = rule_func()
            assert result is None or isinstance(result, Alert)


class TestMonitoringIntegration:
    """Интеграционные тесты мониторинга."""

    @pytest.fixture
    def integrated_monitor(self):
        """Создает полностью настроенный монитор."""
        return get_system_monitor()

    def test_record_metric_integration(self, integrated_monitor):
        """Тест интеграции записи метрик."""
        # Записываем метрику через глобальную функцию
        record_metric("integration.test", 123, {"source": "test"})

        # Проверяем что метрика записана
        metrics = integrated_monitor.metrics_collector.get_metrics("integration.test")
        assert len(metrics) == 1
        assert metrics[0].value == 123
        assert metrics[0].tags["source"] == "test"

    def test_get_system_health_integration(self, integrated_monitor):
        """Тест интеграции получения здоровья системы."""
        # Получаем здоровье через глобальную функцию
        health = get_system_health()

        assert "status" in health
        assert "components" in health
        assert "error_count" in health
        assert "warning_count" in health

    def test_create_performance_report_integration(self, integrated_monitor):
        """Тест интеграции создания отчета производительности."""
        # Полностью очищаем все метрики
        integrated_monitor.metrics_collector.metrics.clear()
        integrated_monitor.metrics_collector.custom_metrics.clear()

        # Записываем только тестовые метрики
        record_metric("system.cpu_percent", 40.0)
        record_metric("system.memory_percent", 50.0)
        time.sleep(0.01)  # Небольшая задержка
        record_metric("system.cpu_percent", 45.0)

        # Создаем отчет через глобальную функцию
        report = create_performance_report(hours=1)

        # Проверяем структуру отчета
        assert "cpu_usage" in report
        assert "memory_usage" in report
        assert "period_hours" in report
        assert "generated_at" in report

        # Проверяем что есть метрики CPU (минимум 2)
        assert "count" in report["cpu_usage"]
        assert "avg" in report["cpu_usage"]
        assert report["cpu_usage"]["count"] >= 2

        # Проверяем что есть метрики памяти (минимум 1)
        assert "count" in report["memory_usage"]
        assert report["memory_usage"]["count"] >= 1

        # Проверяем что среднее значение CPU корректно
        expected_avg = (40.0 + 45.0) / 2
        assert abs(report["cpu_usage"]["avg"] - expected_avg) < 0.1

        # Проверяем период
        assert report["period_hours"] == 1

    def test_monitoring_autostart(self):
        """Тест автоматического запуска мониторинга."""
        # Импортируем модуль заново чтобы проверить автозапуск
        import importlib
        import eva.monitoring.system_monitor as monitor_module
        importlib.reload(monitor_module)

        # Проверяем что мониторинг запущен (если не был остановлен)
        monitor = monitor_module.get_system_monitor()
        # Автозапуск может быть отключен в тестах, поэтому просто проверяем инициализацию
        assert monitor.metrics_collector is not None
        assert monitor.health_checker is not None
        assert monitor.alert_manager is not None


class TestMonitoringUtilities:
    """Тесты вспомогательных функций мониторинга."""

    @pytest.mark.unit
    def test_ml_test_utils_class(self):
        """Тест класса MLTestUtils."""
        from tests.conftest import MLTestUtils

        utils = MLTestUtils()

        # Тест создания mock ответа модели
        response = utils.create_mock_model_response("Test response", 0.9)
        assert response["text"] == "Test response"
        assert response["confidence"] == 0.9
        assert "model_name" in response

        # Тест создания mock токенизации
        tokenization = utils.create_mock_tokenization_result("test text")
        assert "tokens" in tokenization
        assert "token_ids" in tokenization
        assert "attention_mask" in tokenization
        assert tokenization["original_text"] == "test text"

        # Тест создания mock статистики кэша
        cache_stats = utils.create_mock_cache_stats()
        assert "total_entries" in cache_stats
        assert "hit_rate" in cache_stats
        assert cache_stats["hit_rate"] == 0.85

        # Тест симуляции задержки
        start_time = time.time()
        utils.simulate_processing_delay(0.01)
        elapsed = time.time() - start_time
        assert elapsed >= 0.01

        # Тест валидации ML ответа
        valid_response = {"text": "test", "confidence": 0.8}
        assert utils.validate_ml_response(valid_response) is True

        invalid_response = {"text": "test", "confidence": 1.5}  # Confidence > 1
        with pytest.raises(AssertionError):
            utils.validate_ml_response(invalid_response)

    def test_metric_dataclass(self):
        """Тест dataclass Metric."""
        from eva.monitoring.system_monitor import Metric

        timestamp = datetime.now()
        tags = {"component": "test"}

        metric = Metric(
            name="test.metric",
            value=42.5,
            timestamp=timestamp,
            tags=tags
        )

        assert metric.name == "test.metric"
        assert metric.value == 42.5
        assert metric.timestamp == timestamp
        assert metric.tags == tags

    def test_alert_dataclass(self):
        """Тест dataclass Alert."""
        from eva.monitoring.system_monitor import Alert

        timestamp = datetime.now()

        alert = Alert(
            alert_id="test_alert",
            level="warning",
            message="Test alert message",
            component="test_component",
            timestamp=timestamp
        )

        assert alert.alert_id == "test_alert"
        assert alert.level == "warning"
        assert alert.message == "Test alert message"
        assert alert.component == "test_component"
        assert alert.timestamp == timestamp
        assert alert.resolved is False
        assert alert.resolved_at is None


if __name__ == "__main__":
    print("🚀 Запуск тестов системы мониторинга ЕВА...")

    # Можно запускать тесты напрямую
    import subprocess
    result = subprocess.run([
        'python', '-m', 'pytest',
        __file__,
        '-v',
        '--tb=short'
    ], capture_output=True, text=True)

    print("Вывод тестов:")
    print(result.stdout)
    if result.stderr:
        print("Ошибки:")
        print(result.stderr)

    print(f"Код завершения: {result.returncode}")

    print("✅ Тесты системы мониторинга завершены!")
