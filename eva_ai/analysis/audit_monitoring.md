# Отчёт: Monitoring

## 1. Структура

### Директории и файлы

`
eva_ai/
├── monitoring/
│   ├── system_monitor.py          # Основной мониторинг системы
│   └── __pycache__/
├── system/
│   ├── health_monitor.py          # Мониторинг здоровья компонентов
│   ├── system_types.py            # Типы (HealthStatus, AlertLevel)
│   ├── fault_tolerance.py         # Отказоустойчивость
│   └── __pycache__/
└── analytics/
    ├── analytics_manager.py       # Менеджер аналитики
    ├── analytics_integrated.py    # Интегрированный аналитик
    ├── learning_integration.py    # Интеграция с обучением
    ├── contradiction_analyzer.py  # Анализ противоречий
    └── __init__.py
`

### Архитектура компонентов

| Компонент | Назначение | Ключевые классы |
|-----------|------------|-----------------|
| SystemMonitor | Мониторинг системных ресурсов | MetricsCollector, HealthChecker, AlertManager |
| HealthMonitor | Здоровье компонентов EVA | Анализ ml_core, knowledge_graph, memory_manager |
| FaultTolerance | Обработка отказов | Регистрация обработчиков, история ошибок |
| AnalyticsManager | Аналитика производительности | Сбор метрик, тренды, рекомендации |
| IntegratedAnalyticsManager | Интеграция с BaseComponent | EventBus, отслеживание запросов |

---

## 2. Метрики

### 2.1 SystemMonitor (eva_ai/monitoring/system_monitor.py)

**Сборщик метрик (MetricsCollector):**
- ecord_metric(name, value, tags) - запись метрик с тегами
- get_metrics(name, tags, since) - фильтрация метрик
- get_metric_stats(name, hours) - статистика (min, max, avg, median, std_dev)
- Лимит хранения: 10,000 метрик

**Типы метрик:**
`
system.cpu_percent          - загрузка CPU
system.memory_percent      - использование памяти
system.memory_used_gb      - память в GB
system.disk_percent        - использование диска
system.net_bytes_sent      - отправлено байт
system.net_bytes_recv      - получено байт
health.{component}.{key}  - метрики здоровья компонентов
`

**HealthChecker:**
- Регистрация проверок через egister_check(component_name, check_func)
- check_all_components() - проверка всех компонентов
- get_system_health() - агрегированный статус

**AlertManager:**
- Правила генерации алертов
- dd_alert_rule(rule_name, rule_function)
- check_alerts() - проверка правил
- esolve_alert(alert_id) - разрешение алерта
- Уровни: info, warning, error, critical

**Встроенные проверки здоровья:**
- system_resources - CPU < 80%, Memory < 90% = healthy
- python_process - RSS, VMS, threads

**Встроенные алерты:**
- high_cpu - CPU > 90%
- low_memory - Memory > 95% (critical)

### 2.2 HealthMonitor (eva_ai/system/health_monitor.py)

**Взвешенный анализ здоровья компонентов:**

`python
DEFAULT_WEIGHTS = {
    'ml': 0.3,        # Machine Learning Core
    'kg': 0.2,        # Knowledge Graph
    'mm': 0.15,       # Memory Manager
    'am': 0.1,        # Adaptation Manager
    'ef': 0.1,        # Ethical Framework
    'ns': 0.1,        # Neuromorphic Simulator
    'wse': 0.05       # Web Search Engine
}
`

**Статусы:**
- healthy - health_score > 0.7
- warning - health_score > 0.4
- critical - health_score <= 0.4

**Методы анализа:**
- nalyze_system_health() - общий анализ
- nalyze_evolution() - тренды за 30 дней
- _analyze_trends() - определение тренда
- _identify_critical_events() - критические события

### 2.3 FaultTolerance (eva_ai/system/fault_tolerance.py)

**Структура:**
- ault_handlers - словарь обработчиков по типу
- ecovery_strategies - стратегии восстановления
- ault_history - история ошибок

**Методы:**
- egister_fault_handler(fault_type, handler)
- handle_fault(fault_type, error, context)
- get_system_health() - здоровье на основе recent_faults

**Health Score Calculation:**
`python
health_score = 100.0
if recent_faults > 10: health_score -= 30
elif recent_faults > 5: health_score -= 15
# > 80: healthy, > 50: warning, else: critical
`

### 2.4 AnalyticsManager (eva_ai/analytics/analytics_manager.py)

**Сбор метрик:**
- performance_metrics - bottlenecks, optimization_opportunities
- learning_metrics - opportunities_count, pattern_analysis
- system_metrics - cpu_usage, memory_usage, response_time, error_rate

**Интервалы:**
- Сбор метрик: каждые 10 секунд
- Анализ трендов: последние 10 точек
- Хранение: до 1000 точек на метрику

**Рекомендации:**
`python
if avg_bottlenecks > 2: recommend 'optimize_performance'
if avg_opportunities > 3: recommend 'review_learning_opportunities'
`

### 2.5 ContradictionAnalyzer (eva_ai/analytics/contradiction_analyzer.py)

**Детекция противоречий:**
- detect_contradictions(model_facts, web_results)
- calculate_divergence(fact_statement, web_statement)

**Типы противоречий:**
- 
egation - отрицание (negation_matches)
- 
umerical - числовые расхождения (>2x разница)
- 	emporal - временные несогласованности
- keyword - противоречивые ключевые слова

**Пороги:**
- min_divergence_threshold = 0.3
- significant_divergence_threshold = 0.6

**Метрики:**
- 	otal_checked - проверено фактов
- significant_found - найдено значимых
- esolved - разрешено
- divergence_scores - история расхождений

### 2.6 IntegratedAnalyticsManager (eva_ai/analytics/analytics_integrated.py)

**Интеграция с BaseComponent и EventBus:**
- Наследует BaseComponent
- Публикует события: nalytics_manager.initialized, nalytics_manager.started, nalytics_manager.stopped

**Отслеживание:**
- 	rack_query(query, response_time, success)
- get_performance_metrics() - total_queries, success_rate, avg_response_time

---

## 3. Интеграция

### 3.1 Инициализация в CoreBrain

**Файл:** va_ai/core/init_factories.py

`python
def create_system_monitor(initializer):
    from eva_ai.monitoring.system_monitor import SystemMonitor
    system_monitor = SystemMonitor()
    initializer.core_brain.system_monitor = system_monitor
    return system_monitor

# Регистрация компонентов:
'system_monitor': lambda: create_system_monitor(initializer)
`

**Файл:** va_ai/core/init_core.py
`python
' system_monitor',
`

### 3.2 Подключения (init_connections.py)

`python
analytics_manager': ['system_monitor']
'system_monitor': ['resource_manager']
'metrics_collector': ['system_monitor']
`

### 3.3 Глобальные функции

**В system_monitor.py:**
`python
system_monitor = SystemMonitor()  # Глобальный экземпляр
get_system_monitor()              # Получить глобальный монитор
record_metric(name, value, tags) # Записать метрику
get_system_health()              # Статус здоровья
create_performance_report(hours) # Отчёт о производительности

# Автозапуск при импорте
_auto_start_monitoring()
`

### 3.4 События EventBus

**IntegratedAnalyticsManager публикует:**
- nalytics_manager.initialized
- nalytics_manager.started
- nalytics_manager.stopped
- nalytics_manager.query_tracked
- nalytics_manager.report_generated

---

## 4. Оценка

### Сильные стороны

1. **Многоуровневый мониторинг:**
   - Системные ресурсы (CPU, RAM, Disk, Network)
   - Компоненты EVA (ml_core, knowledge_graph и др.)
   - Производительность запросов
   - Анализ противоречий

2. **Гибкая система алертов:**
   - Настраиваемые правила
   - Несколько уровней (info, warning, error, critical)
   - Автоматическое разрешение

3. **Интеграция с EventBus:**
   - Событийная модель
   - Интеграция с BaseComponent

4. **Аналитика и тренды:**
   - Временные ряды метрик
   - Тренды производительности
   - Рекомендации по оптимизации

5. **Автономность:**
   - Автозапуск при импорте
   - Фоновый поток мониторинга

### Проблемы и риски

1. **HealthMonitor частично изолирован:**
   - Требует rain для полного функционирования
   - Не подключен к CoreBrain через init_factories
   - Не использует EventBus

2. **FaultTolerance минимален:**
   - Нет автоматической регистрации обработчиков
   - Не интегрирован в CoreBrain
   - Простая логика health_score

3. **Дублирование функциональности:**
   - SystemMonitor и HealthMonitor частично дублируют мониторинг
   - AnalyticsManager и IntegratedAnalyticsManager - разные реализации

4. **Нет API для внешнего доступа:**
   - Нет REST endpoints для метрик
   - Нет WebSocket для real-time мониторинга

5. **Устаревший ContradictionAnalyzer:**
   - Требует sentence-transformers (опционально)
   - Фоллбэк на Jaccard similarity

### Рекомендации

1. **Интегрировать HealthMonitor в CoreBrain** через init_factories
2. **Унифицировать AnalyticsManager** - оставить один класс
3. **Добавить API endpoints** для мониторинга
4. **Расширить FaultTolerance** - автообработка ошибок
5. **Документировать метрики** - создать схему метрик

### Итоговая оценка: 7/10

Система мониторинга реализована хорошо, но имеет проблемы с интеграцией компонентов между собой и отсутствием внешнего API.
