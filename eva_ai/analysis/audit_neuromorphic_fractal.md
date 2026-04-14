# Neuromorphic & Fractal Store Audit

**Дата аудита:** 14.04.2026  
**Аудитор:** EVA AI System  
**Ревизия кода:** Текущая

---

## 1. Neuromorphic System

### 1.1 Architecture

**Компоненты:**
| Файл | Строк | Назначение |
|------|-------|------------|
| `sim_core.py` | 842 | NeuromorphicSimulator - ядро симулятора |
| `sim_neurons.py` | 51 | FallbackNeuralNetwork - нейронная сеть |
| `sim_spikes.py` | 208 | SpikeEvent, NeuralActivity, SpikeGenerator, SpikePropagator |
| `sim_synapses.py` | 55 | SynapseManager - синаптические связи |
| `sim_plasticity.py` | 123 | STDPPlasticity, AdaptiveThreshold, HomeostaticPlasticity |
| `neuromorphic_memory.py` | 62 | NeuromorphicMemory - high-level интерфейс |
| `neuromorphic_simulator.py` | 27 | Экспорт модуля |

**Архитектурная схема:**
```
NeuromorphicSimulator
├── FallbackNeuralNetwork (x3: working, semantic, episodic)
│   └── neuron_states, connections matrix
├── NeuralActivity history
├── SpikeGenerator / SpikePropagator
├── SynapseManager
├── STDPPlasticity
├── AdaptiveThreshold
└── HomeostaticPlasticity
```

**Интеграция с NEST:**
- Пытается импортировать `nest` (симулятор нейроморфных сетей)
- NEST_AVAILABLE = False если не установлен
- FallbackNeuralNetwork используется если NEST недоступен

### 1.2 Implementation

**Ядро (sim_core.py):**
- Класс `NeuromorphicSimulator` с 3 нейросетями (working=100, semantic=500, episodic=300 нейронов)
- Методы:
  - `simulate_activity()` - симуляция активности
  - `consolidate_activity()` - консолидация через fractal_store
  - `analyze_neural_activity()` - анализ паттернов
  - `get_system_health()` - оценка здоровья системы
- Фоновые потоки:
  - `_simulation_worker` - каждые 1.0 сек
  - `_consolidation_worker` - каждые 300.0 сек
- Кэширование активности в JSON

**Нейроны (sim_neurons.py):**
```python
class FallbackNeuralNetwork:
    neuron_states = np.random.rand(num_neurons)  # [0,1]
    connections = np.random.rand(num_neurons, num_neurons)
    
    def simulate_step(input_stimulus):
        influence = np.dot(connections, neuron_states)
        noise = np.random.normal(0, 0.01)
        new_states = clip(neuron_states + influence * 0.1 + noise, 0, 1)
```

**STDP Пластичность (sim_plasticity.py):**
```python
class STDPPlasticity:
    # Классическая STDP: Δt > 0 → potentiation, Δt < 0 → depression
    delta = learning_rate * exp(-|dt| / tau)
```

**Синапсы (sim_synapses.py):**
- Матрица весов `weights[num_neurons, num_neurons]`
- `update_weights()` - обновление по спайкам
- `prune_weak_connections()` - удаление слабых связей (threshold=0.05)

### 1.3 Integration

**Где используется:**

1. **graph_learning.py** - `NeuromorphicRanker`:
   - Ранжирование контекста через нейроморфные волны
   - Метод `rank_context_nodes()` использует `simulate_activity()`
   - `simple_wave_propagation()` - активация по смежным узлам

2. **health_monitor.py**:
   - Анализирует здоровье NeuromorphicSimulator

3. **mlearning/neuromorphic_memory.py**:
   - `NeuromorphicMemoryLayer(nn.Module)` - PyTorch слой
   - Используется в `fractal_trainer.py`

**Интеграция с FractalStore:**
- `_on_container_accessed()` - реагирует на доступ к контейнерам
- `_on_hot_window_updated()` - обновляет сеть на основе горячего окна
- `_strengthen_container_connections()` - усиление связей
- `_integrate_new_pattern()` - интеграция новых паттернов

### 1.4 Issues

**ПРОБЛЕМЫ:**

1. **NEST не установлен** - система работает на fallback реализации
   - FallbackNeuralNetwork не является настоящей нейроморфной симуляцией
   - Это просто мат.модель с шумом

2. **NeuromorphicSimulator не инициализируется в brain_components.py**
   - НЕТ прямой инициализации в основной системе
   - Создаётся только при импорте в graph_learning.py

3. **Отсутствие связи с основным циклом генерации**
   - Не используется в brain_query.py
   - Не влияет на ответы модели напрямую

4. **Консолидация через fractal_store не проверена**
   - `_update_fractal_structure()` вызывает методы fractal_store которых нет
   - `fractal_store.containers` может не существовать

5. **Метрики health_score генерируются рандомно**
   - `get_neuromorphic_dashboard_data()` генерирует `trends` из `np.random.randn()`

**ВЕРДИКТ: ЭКСПЕРИМЕНТАЛЬНАЯ ЗАГЛУШКА**

---

## 2. Fractal Store (vs FractalGraphV2)

### 2.1 Две РАЗНЫЕ фрактальные системы

| Характеристика | FractalStore | FractalGraphV2 |
|----------------|--------------|----------------|
| **Путь** | `eva_ai/fractal/fractal_store.py` | `eva_ai/memory/fractal_graph_v2/` |
| **Назначение** | Хранение весов модели | Хранение знаний и графа памяти |
| **Активность** | Используется фрагментарно | ОСНОВНАЯ СИСТЕМА |
| **Размер** | 708 строк | 1262 строки (API) + storage.py |

### 2.2 FractalStore (eva_ai/fractal/)

**Назначение:** Pack/unpack весов модели в фрактальные контейнеры

**Архитектура:**
```python
class FractalStore:
    containers: Dict[str, FractalContainer]  # Блоки данных
    fractal_tree: Dict[int, List[str]]       # Иерархия уровней
    hot_window: OrderedDict                   # Горячие контейнеры (LRU)
    
    # Уровни фрактальной иерархии:
    # Level 0: float64/float32 (базовые блоки)
    # Level 1: float32
    # Level 2: float16  
    # Level 3+: int8 (квантизация)
```

**Методы:**
- `pack_model_weights()` / `pack_state_dict()` - упаковка весов
- `get_container()` - получить контейнер с горячим окном
- `save_to_disk()` / `load_from_disk()` - персистентность
- `_build_fractal_hierarchy()` - построение иерархии
- `_safe_quantize_to_int8()` - квантизация

**FractalContainer:**
```python
@dataclass
class FractalContainer:
    id: str
    level: int
    position: Tuple[int, ...]
    data: np.ndarray
    metadata: Dict  # layer_name, is_critical, quant_scale...
    parent: Optional[str]
    children: List[str]
```

**Интеграция:**
- Подключается к `neuromorphic_simulator.py` для консолидации
- В `mlearning/storage/` есть `fractal_weight_store.py` (другая система)

### 2.3 EntityFractalStore (eva_ai/fractal/entity_fractal_store.py)

**Назначение:** Мультиуровневое хранение СУЩНОСТЕЙ

**5 уровней абстракции:**
| Уровень | Название | Описание |
|---------|----------|----------|
| 0 | raw_tokens | Сырые токены |
| 1 | ambiguous_terms | Неоднозначные термины |
| 2 | clarified_meanings | Уточнённые значения |
| 3 | concept_definitions | Концептуальные определения |
| 4 | full_understanding | Полное понимание |

**Интеграция:**
- Используется в `graph_ml_core.py` (MemoryGraphML)
- `store_entity()` - сохранение сущности
- `update_clarification()` - обновление при уточнении
- `search_similar_entities()` - поиск по эмбеддингам

### 2.4 FractalGraphV2 (eva_ai/memory/fractal_graph_v2/)

**ЭТО ОСНОВНАЯ СИСТЕМА ПАМЯТИ**

**Компоненты:**
- `FractalMemoryGraph` - главный API
- `FractalGraphV2` (storage.py) - хранилище
- `EmbeddingsManager` - эмбеддинги
- `GGUFExtractor` - извлечение из GGUF
- `EVAGenerator` - генерация
- `SemanticContextCache` - кэш поиска
- `SnapshotManager` - снапшоты
- `VirtualTokenHandler` - виртуальные токены

**Использование:**
- `brain_query.py` - FG-only mode
- `unified_generator.py` - получение контекста
- `dialog_core.py` - сохранение опыта
- `brain_components.py` - инициализация
- `init_factories.py` - создание

### 2.5 Issues

**ПРОБЛЕМЫ FractalStore:**

1. **Не используется основной системой**
   - FractalGraphV2 полностью заменил его функции
   - FractalStore не инициализируется в brain_components.py

2. **Дублирование функциональности**
   - В `mlearning/storage/` есть `fractal_weight_store.py`
   - В `mlearning/storage/` есть `fractal_store.py`, `fractal_store_utils.py`, `fractal_store_core.py`
   - Три разных fractal_store!

3. **Интеграция с neuromorphic сомнительна**
   - Обработчики событий могут не работать
   - `_on_hot_window_updated()` вызывает методы которые могут не существовать

**ПРОБЛЕМЫ EntityFractalStore:**

1. **Использует random embeddings**
   - `_compute_level_embedding()` генерирует случайные вектора
   - `np.random.seed(hash(text) % 2**32)` - НЕ семантические!

2. **Не интегрирован с FractalGraphV2**
   - Параллельное хранение вместо интеграции

**ВЕРДИКТ: Fragmented Development**

---

## 3. Overall Assessment

### 3.1 Neuromorphic System

| Критерий | Оценка |
|----------|--------|
| Полнота реализации | 6/10 (есть все компоненты) |
| Интеграция | 2/10 (изолирован) |
| Производительность | 3/10 (fallback без GPU) |
| Практическая ценность | 1/10 (не влияет на генерацию) |

**Статус:** ЭКСПЕРИМЕНТАЛЬНАЯ ЗАГЛУШКА

### 3.2 Fractal Systems

| Критерий | FractalStore | EntityFractalStore | FractalGraphV2 |
|----------|--------------|---------------------|----------------|
| Полнота | 5/10 | 4/10 | 9/10 |
| Интеграция | 2/10 | 3/10 | 10/10 |
| Использование | 1/10 | 3/10 | 10/10 |

**Статус FractalStore:** ПРОПАВШИЙ КОД (не используется)  
**Статус EntityFractalStore:** ЧАСТИЧНО ИНТЕГРИРОВАН  
**Статус FractalGraphV2:** ОСНОВНАЯ СИСТЕМА

### 3.3 Архитектурные Проблемы

1. **Три разных fractal_store:**
   - `eva_ai/fractal/fractal_store.py`
   - `eva_ai/mlearning/storage/fractal_store.py`
   - `eva_ai/mlearning/storage/fractal_weight_store.py`

2. **Параллельное развитие:**
   - FractalGraphV2 развивается активно
   - FractalStore заброшен
   - EntityFractalStore частично интегрирован

3. **Neuromorphic изолирован:**
   - Не влияет на основной цикл генерации
   - Создаётся только по требованию в graph_learning

### 3.4 Рекомендации

**Немедленные:**
1. Удалить `eva_ai/fractal/fractal_store.py` или документировать его статус
2. Интегрировать EntityFractalStore с FractalGraphV2 или удалить
3. Добавить NEST установку или удалить neuromorphic

**Среднесрочные:**
1. Создать единый FractalStorage интерфейс
2. Интегрировать NeuromorphicRanker в brain_query

**Долгосрочные:**
1. Реализовать настоящую нейроморфную симуляцию на NEST
2. Использовать neuromorphic для adaptive context ranking

---

## 4. Выводы

**Neuromorphic System:**
- Создавалась как "нейроморфный симулятор для EVA"
- Реально: математическая модель с шумом, без GPU/NEST
- Не интегрирована в основной цикл
- **Вердикт: Экспериментальный прототип**

**Fractal Systems:**
- FractalGraphV2 - ОСНОВНАЯ система памяти (работает)
- FractalStore - заброшенный код для упаковки весов
- EntityFractalStore - частичная интеграция для сущностей
- **Вердикт: Fragmented, требует консолидации**

**Общая оценка кодовой базы:**
- Активно развивающиеся части: FractalGraphV2, ConceptExtractor, ContradictionMiner
- Экспериментальные части: Neuromorphic, FractalStore
- Технический долг: дублирование fractal_store, изоляция neuromorphic

---

*Отчёт сгенерирован EVA AI System Auditor*
