# H3+H4 Deep Dive

## H3: summary_parts в dialog_core.py

### Проблема (код)

**Файл:** `eva_ai/learning/dialog_core.py:1049`

Функция `_run_graph_curator_after_cycle()` возвращает строку с `summary_parts`, но эта переменная не определена в области видимости метода.

```python
def _run_graph_curator_after_cycle(self) -> None:
    # ... code ...
    return " | ".join(summary_parts)  # <-- summary_parts не определён!
```

**Проблема:** Функция объявлена как возвращающая `None`, но в реальности возвращает строку. При этом `summary_parts` нигде не инициализируется в методе.

**Контекст:** Метод вызывается после цикла dual circuit для сохранения графа через GraphCurator.

### Исправление

**Вариант 1:** Если функция должна возвращать статус (изменяем return type на str):

```python
def _run_graph_curator_after_cycle(self) -> str:
    summary_parts = []
    try:
        # Пробуем получить graph_curator
        graph_curator = getattr(self.brain, 'graph_curator', None)
        
        if graph_curator and hasattr(graph_curator, 'force_curation'):
            logger.info("Запускаем GraphCurator после цикла dual circuit")
            result = graph_curator.force_curation()
            summary_parts.append(f"curator: {result}")
        else:
            # Пробуем напрямую через fractal_graph_v2
            fgv2 = getattr(self.brain, 'fractal_graph_v2', None)
            if fgv2 and hasattr(fgv2, 'save'):
                fgv2.save()
                summary_parts.append("fgv2: saved")
                
    except Exception as e:
        logger.warning(f"GraphCurator error: {e}")
        summary_parts.append(f"error: {e}")
    
    return " | ".join(summary_parts)
```

**Вариант 2:** Если функция должна только выполнять действие (более правильный подход - убираем return):

```python
def _run_graph_curator_after_cycle(self) -> None:
    try:
        graph_curator = getattr(self.brain, 'graph_curator', None)
        
        if graph_curator and hasattr(graph_curator, 'force_curation'):
            logger.info("Запускаем GraphCurator после цикла dual circuit")
            graph_curator.force_curation()
        else:
            fgv2 = getattr(self.brain, 'fractal_graph_v2', None)
            if fgv2 and hasattr(fgv2, 'save'):
                fgv2.save()
                logger.info("FractalGraphV2 сохранен после цикла")
                
    except Exception as e:
        logger.warning(f"GraphCurator error: {e}")
```

---

## H4: SystemMonitor Integration

### Текущая архитектура

**SystemMonitor** (`eva_ai/monitoring/system_monitor.py`) работает изолированно:
- Собирает метрики через `MetricsCollector`
- Проверяет здоровье компонентов через `HealthChecker`
- Генерирует алерты через `AlertManager`
- Работает в отдельном потоке с интервалом 30 секунд

**Проблема:** SystemMonitor не подключён к EventBus и не реагирует на события brain/компонентов.

### План подключения к EventBus

#### 1. События для подписки SystemMonitor

SystemMonitor должен подписаться на следующие события из EventBus:

| Событие | Действие |
|---------|----------|
| `system.ready` | Запустить мониторинг |
| `system.stop` | Остановить мониторинг |
| `component.initialized` | Добавить проверку здоровья компонента |
| `component.error` | Записать метрику ошибки |
| `memory.warning` | Записать алерт о памяти |
| `learning.failed` | Записать алерт об ошибке обучения |

#### 2. Интеграция в CoreBrain

В `eva_ai/core/init_factories.py` добавить:

```python
def _setup_system_monitor(event_bus: EventBus):
    """Подключает SystemMonitor к EventBus."""
    from eva_ai.monitoring.system_monitor import get_system_monitor
    
    monitor = get_system_monitor()
    
    # Подписки на события
    event_bus.subscribe(EventTypes.SYSTEM_READY, monitor.on_system_ready)
    event_bus.subscribe(EventTypes.SYSTEM_STOP, monitor.on_system_stop)
    event_bus.subscribe(EventTypes.COMPONENT_ERROR, monitor.on_component_error)
    event_bus.subscribe(EventTypes.MEMORY_WARNING, monitor.on_memory_warning)
    
    # Запуск мониторинга
    if not monitor.running:
        monitor.start_monitoring()
```

Добавить методы в SystemMonitor:

```python
# В SystemMonitor добавить:
def on_system_ready(self, event):
    """Обработчик system.ready."""
    logger.info("SystemMonitor: system ready received")
    if not self.running:
        self.start_monitoring()

def on_system_stop(self, event):
    """Обработчик system.stop."""
    logger.info("SystemMonitor: system stop received")
    self.stop_monitoring()

def on_component_error(self, event):
    """Обработчик component.error."""
    self.metrics_collector.record_metric(
        "component.error",
        1,
        {"component": event.data.get("component", "unknown")}
    )

def on_memory_warning(self, event):
    """Обработчик memory.warning."""
    self.metrics_collector.record_metric(
        "memory.warning",
        event.data.get("level", 1),
        event.data
    )
```

#### 3. Публикация событий из SystemMonitor

SystemMonitor должен публиковать события при обнаружении проблем:

```python
# В HealthChecker.check_all_components() добавить:
def check_all_components(self) -> Dict[str, Dict[str, Any]]:
    results = {}
    # ... существующий код ...
    
    # Публикуем событие при ошибках
    from eva_ai.core.event_bus import Event, EventTypes, EventPriority
    
    if error_count > 0:
        event_bus.publish(Event(
            event_type=EventTypes.SYSTEM_ERROR,
            source="system_monitor",
            data={"errors": error_count, "components": list(results.keys())}
        ))
    
    return results
```

#### 4. Регистрация проверок компонентов

При инициализации компонентов в CoreBrain:

```python
# В brain_components.py или init_factories.py
def register_component_checks(system_monitor, brain):
    """Регистрирует проверки здоровья для компонентов brain."""
    
    # Проверка fractal_graph_v2
    system_monitor.register_component_check("fractal_graph_v2", lambda: {
        "status": "healthy" if brain.fractal_graph_v2 else "error",
        "nodes": len(getattr(brain.fractal_graph_v2, 'nodes', {}))
    })
    
    # Проверка concept_miner
    if hasattr(brain, 'concept_miner'):
        system_monitor.register_component_check("concept_miner", lambda: {
            "status": "healthy" if brain.concept_miner else "error"
        })
```

### Итоговая архитектура после интеграции

```
EventBus
├── SystemMonitor (подписки)
│   ├── system.ready → start_monitoring()
│   ├── system.stop → stop_monitoring()
│   ├── component.error → record_metric()
│   └── memory.warning → record_metric()
│
├── SystemMonitor (публикации)
│   └── system.error → компоненты получают событие
│
└── Другие компоненты (SelfDialogLearning, ConceptMiner, etc.)
```

### Priority задачи

1. **Medium Priority**: Добавить методы-обработчики в SystemMonitor
2. **Medium Priority**: Подключить в init_factories.py после создания EventBus
3. **Low Priority**: Добавить публикацию событий при обнаружении проблем
