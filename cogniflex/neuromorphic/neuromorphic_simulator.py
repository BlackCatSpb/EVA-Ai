"""Модуль нейроморфного симулятора для CogniFlex"""
import os
import logging
import time
import threading
import queue
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64
from dataclasses import dataclass, field
logger = logging.getLogger("cogniflex.neuromorphic")

# Проверка наличия NEST
NEST_AVAILABLE = False
try:
    import nest
    NEST_AVAILABLE = True
    logger.info("Модуль NEST для нейроморфного моделирования доступен")
except ImportError:
    logger.warning("Модуль NEST не найден. Нейроморфное моделирование будет недоступно")
except Exception as e:
    logger.warning(f"Ошибка при импорте NEST: {e}. Нейроморфное моделирование будет недоступно")

@dataclass
class NeuralActivity:
    """Представляет данные о нейронной активности."""
    timestamp: float = field(default_factory=time.time)
    activity_pattern: List[float] = field(default_factory=list)
    memory_type: str = "working"
    strength: float = 0.5
    importance: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует активность в словарь."""
        return {
            "timestamp": self.timestamp,
            "activity_pattern": self.activity_pattern,
            "memory_type": self.memory_type,
            "strength": self.strength,
            "importance": self.importance,
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralActivity':
        """Создает активность из словаря."""
        return cls(
            timestamp=data["timestamp"],
            activity_pattern=data["activity_pattern"],
            memory_type=data["memory_type"],
            strength=data["strength"],
            importance=data["importance"],
            context=data["context"],
            metadata=data["metadata"]
        )

    def get_analysis(self) -> Dict[str, Any]:
        """
        Возвращает анализ активности.
        Returns:
            Dict: Результаты анализа
        """
        if not self.activity_pattern:
            return {"status": "no_data"}
        # Рассчитываем статистику
        mean_activity = np.mean(self.activity_pattern)
        std_activity = np.std(self.activity_pattern)
        max_activity = np.max(self.activity_pattern)
        min_activity = np.min(self.activity_pattern)
        # Рассчитываем когерентность (простая метрика)
        coherence = 1.0 - (std_activity / (mean_activity + 1e-8))  # Избегаем деления на ноль
        coherence = max(0.0, min(1.0, coherence))
        return {
            "mean_activity": float(mean_activity),
            "std_activity": float(std_activity),
            "max_activity": float(max_activity),
            "min_activity": float(min_activity),
            "coherence": float(coherence),
            "pattern_length": len(self.activity_pattern),
            "strength": self.strength,
            "importance": self.importance
        }

class FallbackNeuralNetwork:
    """Альтернативная реализация нейронной сети без NEST."""
    def __init__(self, num_neurons: int, memory_type: str):
        """
        Инициализирует альтернативную нейронную сеть.
        Args:
            num_neurons: Количество нейронов
            memory_type: Тип памяти
        """
        self.num_neurons = num_neurons
        self.memory_type = memory_type
        self.neuron_states = np.random.rand(num_neurons)  # Состояния нейронов (0-1)
        self.connections = np.random.rand(num_neurons, num_neurons)  # Матрица связей
        self.activity_history: List[np.ndarray] = []  # История активности
        self.simulation_steps = 0
        logger.info(f"Альтернативная нейронная сеть инициализирована для {memory_type} памяти с {num_neurons} нейронами")

    def simulate_step(self, input_stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Выполняет один шаг симуляции.
        Args:
            input_stimulus: Входной стимул (опционально)
        Returns:
            np.ndarray: Новое состояние активности
        """
        # Применяем входной стимул
        if input_stimulus is not None and len(input_stimulus) == self.num_neurons:
            self.neuron_states = np.maximum(self.neuron_states, input_stimulus)
        # Обновляем состояния нейронов
        # Простая модель: новое состояние = текущее + влияние связей + шум
        influence = np.dot(self.connections, self.neuron_states)
        noise = np.random.normal(0, 0.01, self.num_neurons)  # Небольшой шум
        new_states = self.neuron_states + influence * 0.1 + noise
        # Ограничиваем значения
        self.neuron_states = np.clip(new_states, 0.0, 1.0)
        # Сохраняем историю
        self.activity_history.append(self.neuron_states.copy())
        # Ограничиваем историю
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
        self.simulation_steps += 1
        return self.neuron_states

    def get_activity_pattern(self) -> List[float]:
        """
        Возвращает текущий паттерн активности.
        Returns:
            List[float]: Паттерн активности
        """
        return self.neuron_states.tolist()

    def connect_neurons(self, source: int, target: int, weight: float = 0.5):
        """
        Соединяет два нейрона.
        Args:
            source: Источник (индекс нейрона)
            target: Цель (индекс нейрона)
            weight: Вес связи
        """
        if 0 <= source < self.num_neurons and 0 <= target < self.num_neurons:
            self.connections[target, source] = max(0.0, min(1.0, weight))
        else:
            logger.warning(f"Неверные индексы нейронов для соединения: {source} -> {target}")

    def get_network_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о сети.
        Returns:
            Dict: Информация о сети
        """
        return {
            "num_neurons": self.num_neurons,
            "memory_type": self.memory_type,
            "simulation_steps": self.simulation_steps,
            "activity_history_length": len(self.activity_history)
        }

class NESTNeuralNetwork:
    """Реализация нейронной сети с использованием NEST."""
    def __init__(self, num_neurons: int, memory_type: str):
        """
        Инициализирует нейронную сеть NEST.
        Args:
            num_neurons: Количество нейронов
            memory_type: Тип памяти
        """
        self.num_neurons = num_neurons
        self.memory_type = memory_type
        self.neuron_population = None
        self.spike_generator = None
        self.spike_detector = None
        self.simulation_steps = 0
        # Инициализируем NEST
        self._initialize_nest()

    def _initialize_nest(self):
        """Инициализирует ядро NEST."""
        try:
            nest.ResetKernel()
            # Создаем популяцию нейронов (например, модель integrate-and-fire)
            self.neuron_population = nest.Create("iaf_psc_alpha", self.num_neurons)
            # Создаем генератор спайков для входных сигналов
            self.spike_generator = nest.Create("spike_generator")
            # Создаем детектор спайков для записи активности
            self.spike_detector = nest.Create("spike_detector")
            # Соединяем генератор с популяцией
            nest.Connect(self.spike_generator, self.neuron_population)
            # Соединяем популяцию с детектором
            nest.Connect(self.neuron_population, self.spike_detector)
            logger.info(f"Нейроморфный симулятор NEST инициализирован для {self.memory_type} памяти с {self.num_neurons} нейронами")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации нейроморфного симулятора NEST: {e}")
            self.neuron_population = None
            self.spike_generator = None
            self.spike_detector = None
            return False

    def simulate_step(self, input_stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Выполняет один шаг симуляции.
        Args:
            input_stimulus: Входной стимул (опционально)
        Returns:
            np.ndarray: Новое состояние активности (простая эмуляция)
        """
        if self.neuron_population is None:
            return np.zeros(self.num_neurons)
        try:
            # Применяем входной стимул
            if input_stimulus is not None and len(input_stimulus) == self.num_neurons:
                # Преобразуем активность в спайки
                spike_times = []
                spike_senders = []
                for i, activity in enumerate(input_stimulus):
                    if activity > 0.5:  # Порог активации
                        spike_times.append(self.simulation_steps * 10 + 1)  # Время спайка
                        spike_senders.append(i + 1)  # ID нейрона (1-based)
                if spike_times:
                    nest.SetStatus(self.spike_generator, {"spike_times": spike_times, "spike_senders": spike_senders})
            # Выполняем шаг симуляции
            nest.Simulate(10.0)  # 10 ms
            self.simulation_steps += 1
            # Получаем активность (простая эмуляция)
            activity = np.random.rand(self.num_neurons)  # Заглушка
            return activity
        except Exception as e:
            logger.error(f"Ошибка симуляции NEST: {e}")
            return np.zeros(self.num_neurons)

    def get_activity_pattern(self) -> List[float]:
        """
        Возвращает текущий паттерн активности.
        Returns:
            List[float]: Паттерн активности
        """
        # В реальной системе здесь будет получение данных из spike_detector
        return np.random.rand(self.num_neurons).tolist()  # Заглушка

    def connect_neurons(self, source: int, target: int, weight: float = 0.5, delay: float = 1.0):
        """
        Соединяет два нейрона с использованием NEST.
        Args:
            source: Источник (индекс нейрона)
            target: Цель (индекс нейрона)
            weight: Вес связи
            delay: Задержка (мс)
        """
        if self.neuron_population is None:
            return
        try:
            if 0 <= source < self.num_neurons and 0 <= target < self.num_neurons:
                # ID нейронов в NEST (1-based)
                source_id = self.neuron_population[source]
                target_id = self.neuron_population[target]
                nest.Connect([source_id], [target_id], syn_spec={"weight": weight, "delay": delay})
            else:
                logger.warning(f"Неверные индексы нейронов для соединения NEST: {source} -> {target}")
        except Exception as e:
            logger.error(f"Ошибка соединения нейронов NEST: {e}")

    def get_network_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о сети.
        Returns:
            Dict: Информация о сети
        """
        return {
            "num_neurons": self.num_neurons,
            "memory_type": self.memory_type,
            "simulation_steps": self.simulation_steps,
            "nest_available": self.neuron_population is not None
        }

class NeuromorphicSimulator:
    """Симулятор нейроморфных сетей для CogniFlex с поддержкой NEST и альтернативных методов."""
    def __init__(self, cache_dir: Optional[str] = None, brain=None):
        """
        Инициализирует нейроморфный симулятор.
        Args:
            cache_dir: Путь к директории кэша
            brain: Ссылка на ядро CogniFlex (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir or "cogniflex_neuromorphic_cache"
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        # Создаем директорию кэша
        os.makedirs(self.cache_dir, exist_ok=True)
        # Пути к файлам
        self.activity_cache_file = os.path.join(self.cache_dir, "neural_activity.json")
        self.network_cache_file = os.path.join(self.cache_dir, "neural_networks.json")
        # Параметры симуляции
        self.simulation_interval = 1.0  # Интервал симуляции (секунды)
        self.consolidation_interval = 300.0  # Интервал консолидации (секунды)
        self.use_nest = NEST_AVAILABLE  # Использовать ли NEST
        # Нейронные сети для разных типов памяти
        self.neural_networks: Dict[str, Union[FallbackNeuralNetwork, NESTNeuralNetwork]] = {}
        # История активности
        self.activity_history: List[NeuralActivity] = []
        # Потоки
        self.simulation_thread: Optional[threading.Thread] = None
        self.consolidation_thread: Optional[threading.Thread] = None
        # Инициализируем
        self.initialize()
        logger.info(f"Нейроморфный симулятор инициализирован. Использование NEST: {self.use_nest}")

    def initialize(self):
        """Инициализирует нейроморфный симулятор."""
        logger.info("Инициализация нейроморфного симулятора...")
        # Загружаем сохраненные данные
        self._load_activity_cache()
        self._load_network_cache()
        # Инициализируем нейронные сети для разных типов памяти
        self._init_neural_networks()
        self.initialized = True
        logger.info("Нейроморфный симулятор успешно инициализирован")

    def _init_neural_networks(self):
        """Инициализирует нейронные сети для различных типов памяти."""
        memory_types = {
            "working": 100,    # Рабочая память
            "semantic": 500,   # Семантическая память
            "episodic": 300    # Эпизодическая память
        }
        for memory_type, num_neurons in memory_types.items():
            try:
                if self.use_nest:
                    try:
                        network = NESTNeuralNetwork(num_neurons, memory_type)
                        if network.neuron_population is not None:
                            self.neural_networks[memory_type] = network
                            logger.info(f"Создана NEST-сеть для {memory_type} памяти с {num_neurons} нейронами")
                        else:
                            # Если NEST не работает, используем альтернативу
                            logger.warning(f"Ошибка создания NEST-сети для {memory_type} памяти: {e}. Используется альтернатива.")
                            self.neural_networks[memory_type] = FallbackNeuralNetwork(num_neurons, memory_type)
                            self.use_nest = False
                    except Exception as e:
                        logger.warning(f"Ошибка создания NEST-сети для {memory_type} памяти: {e}. Используется альтернатива.")
                        self.neural_networks[memory_type] = FallbackNeuralNetwork(num_neurons, memory_type)
                        self.use_nest = False
                else:
                    self.neural_networks[memory_type] = FallbackNeuralNetwork(num_neurons, memory_type)
                    logger.info(f"Создана альтернативная сеть для {memory_type} памяти с {num_neurons} нейронами")
            except Exception as e:
                logger.error(f"Ошибка инициализации нейронной сети для {memory_type} памяти: {e}")

    def _load_activity_cache(self):
        """Загружает кэш активности из файла."""
        try:
            if os.path.exists(self.activity_cache_file):
                with open(self.activity_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                # Восстанавливаем историю активности
                for activity_data in cache_data.get("activity_history", []):
                    self.activity_history.append(NeuralActivity.from_dict(activity_data))
                logger.info(f"Загружено {len(self.activity_history)} записей активности из кэша")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша активности: {e}")

    def _save_activity_cache(self):
        """Сохраняет кэш активности в файл."""
        try:
            cache_data = {
                "activity_history": [act.to_dict() for act in self.activity_history[-1000:]]  # Ограничиваем 1000 записями
            }
            with open(self.activity_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша активности: {e}")

    def _load_network_cache(self):
        """Загружает кэш сетей из файла."""
        # В реальной системе здесь будет загрузка параметров сетей
        pass

    def _save_network_cache(self):
        """Сохраняет кэш сетей в файл."""
        # В реальной системе здесь будет сохранение параметров сетей
        pass

    def start(self):
        """Запускает нейроморфный симулятор."""
        if not self.initialized:
            logger.warning("Попытка запуска неинициализированного нейроморфного симулятора")
            return
        if self.running:
            logger.warning("Нейроморфный симулятор уже запущен")
            return
        self.running = True
        self.stop_event.clear()
        # Запускаем поток симуляции
        self.simulation_thread = threading.Thread(
            target=self._simulation_worker,
            daemon=True,
            name="NeuromorphicSimulationThread"
        )
        self.simulation_thread.start()
        # Запускаем поток консолидации
        self.consolidation_thread = threading.Thread(
            target=self._consolidation_worker,
            daemon=True,
            name="NeuromorphicConsolidationThread"
        )
        self.consolidation_thread.start()
        logger.info("Нейроморфный симулятор запущен")

    def stop(self):
        """Останавливает нейроморфный симулятор."""
        self.running = False
        self.stop_event.set()
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5.0)
        # Сохраняем данные перед остановкой
        self._save_activity_cache()
        self._save_network_cache()
        logger.info("Нейроморфный симулятор остановлен")

    def _simulation_worker(self):
        """Рабочий поток для симуляции нейронной активности."""
        last_consolidation = time.time()
        while self.running and not self.stop_event.is_set():
            try:
                # Выполняем симуляцию
                self._simulate_step()
                # Проверяем, пора ли консолидировать
                current_time = time.time()
                if current_time - last_consolidation > self.consolidation_interval:
                    self.consolidate_activity()
                    last_consolidation = current_time
                # Ждем до следующего шага
                if self.stop_event.wait(self.simulation_interval):
                    break
            except Exception as e:
                logger.error(f"Ошибка в рабочем потоке симуляции: {e}")
                time.sleep(10)

    def _consolidation_worker(self):
        """Рабочий поток для консолидации активности."""
        while self.running and not self.stop_event.is_set():
            try:
                # Ждем до следующей консолидации
                if self.stop_event.wait(self.consolidation_interval):
                    break
                # Выполняем консолидацию
                if self.running:
                    self.consolidate_activity()
            except Exception as e:
                logger.error(f"Ошибка в рабочем потоке консолидации: {e}")
                time.sleep(60)

    def _simulate_step(self):
        """Выполняет один шаг симуляции для всех сетей."""
        for memory_type, network in self.neural_networks.items():
            try:
                # Генерируем входной стимул (простая эмуляция)
                input_stimulus = np.random.rand(network.num_neurons) * 0.1  # Небольшой стимул
                # Выполняем симуляцию
                activity_pattern = network.simulate_step(input_stimulus)
                # Создаем запись активности
                activity = NeuralActivity(
                    activity_pattern=activity_pattern.tolist(),
                    memory_type=memory_type,
                    strength=np.mean(activity_pattern),
                    importance=np.std(activity_pattern)
                )
                # Добавляем в историю
                self.activity_history.append(activity)
                # Ограничиваем историю
                if len(self.activity_history) > 5000:
                    self.activity_history = self.activity_history[-5000:]
            except Exception as e:
                logger.error(f"Ошибка симуляции для сети {memory_type}: {e}")

    def simulate_activity(self, duration: float = 10.0, memory_type: str = "working") -> NeuralActivity:
        """
        Симулирует нейронную активность.
        Args:
            duration: Длительность симуляции (секунды)
            memory_type: Тип памяти
        Returns:
            NeuralActivity: Данные о симуляции
        """
        if not self.running:
            logger.warning("Симулятор не запущен. Запускаем...")
            self.start()
        # Определяем количество шагов
        steps = int(duration / self.simulation_interval)
        if steps <= 0:
            steps = 1
        # Получаем сеть
        network = self.neural_networks.get(memory_type)
        if not network:
            logger.error(f"Сеть для типа памяти {memory_type} не найдена")
            return NeuralActivity(memory_type=memory_type, simulation_duration=duration)
        # Выполняем симуляцию
        patterns = []
        for _ in range(steps):
            # Генерируем входной стимул
            input_stimulus = np.random.rand(network.num_neurons) * 0.2
            # Выполняем шаг
            pattern = network.simulate_step(input_stimulus)
            patterns.append(pattern)
        # Формируем результат
        if patterns:
            # Объединяем паттерны
            combined_pattern = np.mean(patterns, axis=0)
            activity = NeuralActivity(
                activity_pattern=combined_pattern.tolist(),
                memory_type=memory_type,
                strength=np.mean(combined_pattern),
                importance=np.std(combined_pattern),
                metadata={
                    "duration": duration,
                    "steps": steps,
                    "network_info": network.get_network_info()
                }
            )
        else:
            activity = NeuralActivity(
                memory_type=memory_type,
                metadata={"duration": duration, "steps": steps, "error": "No patterns generated"}
            )
        # Добавляем в историю
        self.activity_history.append(activity)
        # Ограничиваем историю
        if len(self.activity_history) > 5000:
            self.activity_history = self.activity_history[-5000:]
        # Сохраняем кэш
        if len(self.activity_history) % 100 == 0:
            self._save_activity_cache()
        return activity

    def consolidate_activity(self) -> bool:
        """
        Консолидирует нейронную активность в долгосрочную память.
        Returns:
            bool: Успешно ли выполнена консолидация
        """
        logger.info("Консолидация нейронной активности...")
        try:
            # Анализируем активность
            analysis = self.analyze_neural_activity()
            # Обновляем память на основе анализа
            if self.brain and hasattr(self.brain, "memory_manager"):
                # Получаем наиболее значимые паттерны
                significant_patterns = [
                    act for act in self.activity_history
                    if act.strength * act.importance > 0.5
                ]
                # Переносим значимые паттерны в долгосрочную память
                for pattern in significant_patterns:
                    # Формируем содержимое для памяти
                    content = f"Нейронный паттерн ({pattern.memory_type}): {pattern.activity_pattern[:5]}..."
                    # Определяем важность
                    importance_value = pattern.strength * 0.7 + pattern.importance * 0.3
                    # Записываем в память
                    self.brain.memory_manager.record_interaction(
                        user_id="neuromorphic_system",
                        query=content,
                        response="Консолидировано через нейроморфный симулятор",
                        importance=importance_value
                    )
                logger.info(f"Консолидировано {len(significant_patterns)} нейронных паттернов")
                # Очищаем историю активности (сохраняем только последние 5 минут)
                current_time = time.time()
                self.activity_history = [
                    act for act in self.activity_history
                    if current_time - act.timestamp < 300  # Сохраняем только последние 5 минут
                ]
                return True
            else:
                logger.warning("Невозможно консолидировать активность: менеджер памяти недоступен")
                return False
        except Exception as e:
            logger.error(f"Ошибка консолидации нейронной активности: {e}")
            return False

    def analyze_neural_activity(self) -> Dict[str, Any]:
        """
        Анализирует нейронную активность.
        Returns:
            Dict: Результаты анализа
        """
        if not self.activity_history:
            return {"status": "no_data"}
        # Получаем последние 100 записей
        recent_activity = self.activity_history[-100:]
        # Анализируем по типам памяти
        memory_analysis = {}
        for activity in recent_activity:
            mem_type = activity.memory_type
            if mem_type not in memory_analysis:
                memory_analysis[mem_type] = {
                    "activities": [],
                    "total_strength": 0.0,
                    "total_importance": 0.0
                }
            memory_analysis[mem_type]["activities"].append(activity)
            memory_analysis[mem_type]["total_strength"] += activity.strength
            memory_analysis[mem_type]["total_importance"] += activity.importance
        # Рассчитываем статистику для каждого типа
        for mem_type, data in memory_analysis.items():
            count = len(data["activities"])
            if count > 0:
                data["average_activity"] = np.mean([np.mean(a.activity_pattern) for a in data["activities"]])
                data["average_strength"] = data["total_strength"] / count
                data["average_importance"] = data["total_importance"] / count
                # Анализируем когерентность последней активности
                if data["activities"]:
                    last_activity = data["activities"][-1]
                    coherence_data = last_activity.get_analysis()
                    data["latest_coherence"] = coherence_data.get("coherence", 0.5)
                else:
                    data["latest_coherence"] = 0.5
            else:
                data["average_activity"] = 0.0
                data["average_strength"] = 0.0
                data["average_importance"] = 0.0
                data["latest_coherence"] = 0.5
        # Анализируем взаимодействие между типами памяти
        interaction_strength = 0.0
        mem_types = list(memory_analysis.keys())
        if len(mem_types) > 1:
            # Простая метрика: средняя корреляция между активностями разных типов
            correlations = []
            for i in range(len(mem_types)):
                for j in range(i + 1, len(mem_types)):
                    type1 = mem_types[i]
                    type2 = mem_types[j]
                    # Получаем активности для сравнения
                    acts1 = [a.strength for a in memory_analysis[type1]["activities"][-10:]]  # Последние 10
                    acts2 = [a.strength for a in memory_analysis[type2]["activities"][-10:]]  # Последние 10
                    # Выравниваем длины
                    min_len = min(len(acts1), len(acts2))
                    if min_len > 1:
                        corr = np.corrcoef(acts1[:min_len], acts2[:min_len])[0, 1]
                        correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            if correlations:
                interaction_strength = np.mean(correlations)
        # Анализируем последнюю активность
        latest_activity = recent_activity[-1] if recent_activity else None
        latest_analysis = latest_activity.get_analysis() if latest_activity else {}
        return {
            "memory_types": memory_analysis,
            "interaction_strength": interaction_strength,
            "latest_activity": latest_analysis,
            "total_activities": len(self.activity_history),
            "analysis_time": time.time()
        }

    def _visualize_activity(self, memory_type: str) -> str:
        """
        Создает визуализацию активности для типа памяти.
        Args:
            memory_type: Тип памяти
        Returns:
            str: Изображение в формате base64
        """
        # Получаем историю активности для типа памяти
        type_activities = [act for act in self.activity_history if act.memory_type == memory_type][-50:]  # Последние 50
        if not type_activities:
            return ""
        # Создаем матрицу активности
        activity_matrix = np.array([act.activity_pattern for act in type_activities])
        # Создаем график
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(activity_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Активность')
        plt.title(f"Нейронная активность: {memory_type}")
        plt.xlabel('Время (шаги)')
        plt.ylabel('Нейроны')
        # Сохраняем в буфер
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        # Конвертируем в base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"

    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает информацию о здоровье нейроморфной системы.
        Returns:
            Dict: Информация о здоровье
        """
        analysis = self.analyze_neural_activity()
        # Рассчитываем оценку здоровья
        health_score = 100
        # Учитываем активность по типам памяти
        for memory_type, data in analysis["memory_types"].items():
            avg_activity = data["average_activity"]
            # Оптимальный диапазон: 0.3 - 0.7
            if avg_activity < 0.2 or avg_activity > 0.8:
                health_score -= 15
        # Учитываем взаимодействие между типами
        if analysis["interaction_strength"] < 0.2:
            health_score -= 20
        elif analysis["interaction_strength"] < 0.4:
            health_score -= 10
        # Учитываем когерентность
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            coherence = analysis["latest_activity"]["coherence"]
            if coherence < 0.3:
                health_score -= 25
            elif coherence < 0.5:
                health_score -= 15
        # Ограничиваем диапазон
        health_score = max(0, min(100, health_score))
        # Определяем статус
        if health_score > 80:
            status = "healthy"
        elif health_score > 50:
            status = "warning"
        else:
            status = "critical"
        return {
            "status": status,
            "health_score": health_score,
            "analysis": analysis,
            "networks": {name: net.get_network_info() for name, net in self.neural_networks.items()},
            "timestamp": time.time()
        }

    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье нейроморфной системы.
        Returns:
            Dict: Отчет о здоровье
        """
        health = self.get_system_health()
        analysis = health["analysis"]
        # Формируем рекомендации
        recommendations = self._generate_health_recommendations(analysis, health["health_score"])
        return {
            "health": health,
            "recommendations": recommendations,
            "timestamp": time.time()
        }

    def _generate_health_recommendations(self, analysis: Dict, health_score: float) -> List[str]:
        """
        Генерирует рекомендации на основе состояния системы.
        Args:
            analysis: Анализ активности
            health_score: Оценка здоровья
        Returns:
            List[str]: Рекомендации
        """
        recommendations = []
        # Рекомендации по активности
        for memory_type, data in analysis["memory_types"].items():
            avg_activity = data["average_activity"]
            if avg_activity < 0.2:
                recommendations.append(
                    f"Низкая активность {memory_type} памяти. Рассмотрите возможность увеличения стимуляции."
                )
            elif avg_activity > 0.8:
                recommendations.append(
                    f"Высокая активность {memory_type} памяти. Рассмотрите возможность снижения нагрузки."
                )
        # Рекомендации по взаимодействию
        if analysis["interaction_strength"] < 0.3:
            recommendations.append(
                "Низкое взаимодействие между типами памяти. Рассмотрите улучшение связей между сетями."
            )
        # Рекомендации по когерентности
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            coherence = analysis["latest_activity"]["coherence"]
            if coherence < 0.4:
                recommendations.append(
                    "Низкая когерентность нейронной активности. Рассмотрите корректировку параметров сети."
                )
        # Общие рекомендации
        if health_score < 50:
            recommendations.append(
                "Общее состояние системы критическое. Проведите полную диагностику и перезапуск."
            )
        elif health_score < 70:
            recommendations.append(
                "Общее состояние системы требует внимания. Проверьте параметры симуляции."
            )
        if not recommendations:
            recommendations.append(
                "Система находится в хорошем состоянии. Нет критических проблем."
            )
        return recommendations

    def get_fix_suggestions(self) -> List[Dict[str, Any]]:
        """
        Возвращает предложения по исправлению проблем с нейроморфной системой.
        Returns:
            List[Dict[str, Any]]: Список предложений
        """
        health = self.get_system_health()
        analysis = health["analysis"]
        fixes = []
        # Добавляем исправления для перегруженной памяти
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                fixes.append({
                    "id": f"reduce_{memory_type}_activity",
                    "title": f"Снизить активность {memory_type} памяти",
                    "description": f"Активность {memory_type} памяти превышает оптимальный уровень",
                    "severity": "high",
                    "action": lambda mt=memory_type: self._adjust_memory_activity(mt, -0.2)
                })
        # Добавляем исправления для низкой активности
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] < 0.2:
                fixes.append({
                    "id": f"increase_{memory_type}_activity",
                    "title": f"Увеличить активность {memory_type} памяти",
                    "description": f"Активность {memory_type} памяти ниже оптимального уровня",
                    "severity": "medium",
                    "action": lambda mt=memory_type: self._adjust_memory_activity(mt, 0.2)
                })
        # Добавляем исправления для низкого взаимодействия
        if analysis["interaction_strength"] < 0.3:
            fixes.append({
                "id": "improve_interaction",
                "title": "Улучшить взаимодействие между типами памяти",
                "description": "Низкая связь между различными типами памяти",
                "severity": "high",
                "action": self._improve_memory_interaction
            })
        # Добавляем исправления для низкой когерентности
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            if analysis["latest_activity"]["coherence"] < 0.4:
                fixes.append({
                    "id": "improve_coherence",
                    "title": "Улучшить когерентность нейронной активности",
                    "description": "Низкая согласованность нейронной активности",
                    "severity": "medium",
                    "action": self._improve_coherence
                })
        # Добавляем исправление для отсутствия NEST
        if not self.use_nest:
            fixes.append({
                "id": "install_nest",
                "title": "Установить NEST",
                "description": "Позволит использовать полную функциональность нейроморфного моделирования",
                "severity": "high",
                "action": self._install_nest_instructions
            })
        return fixes

    def _adjust_memory_activity(self, memory_type: str, delta: float):
        """
        Корректирует активность памяти.
        Args:
            memory_type: Тип памяти
            delta: Изменение активности
        """
        # В реальной системе здесь будет сложная логика
        # Для упрощения просто увеличиваем коэффициенты
        logger.info(f"Корректировка активности {memory_type} памяти на {delta}")
        # Пример: изменяем параметры симуляции
        # В реальной системе здесь будет модификация параметров сети
        pass

    def _improve_memory_interaction(self):
        """Улучшает взаимодействие между типами памяти."""
        # В реальной системе здесь будет сложная логика
        # Для упрощения просто увеличиваем коэффициенты
        logger.info("Улучшение взаимодействия между типами памяти")
        # Пример: увеличиваем веса связей между сетями
        if "working" in self.neural_networks and "semantic" in self.neural_networks:
            # В реальной системе здесь будет модификация связей
            pass

    def _improve_coherence(self):
        """Улучшает когерентность нейронной активности."""
        logger.info("Улучшение когерентности нейронной активности")
        # В реальной системе здесь будет корректировка параметров сети
        for network in self.neural_networks.values():
            # Пример: уменьшаем шум
            # В реальной системе здесь будут конкретные изменения параметров
            pass

    def _install_nest_instructions(self):
        """Выводит инструкции по установке NEST."""
        instructions = """
Для установки NEST выполните следующие команды:

1. Установите зависимости:
   Ubuntu/Debian: sudo apt-get install build-essential cmake python3-dev
   CentOS/RHEL: sudo yum install gcc gcc-c++ make cmake python3-devel

2. Скачайте и установите NEST:
   pip install nest-simulator

3. Проверьте установку:
   python -c "import nest; nest.ResetKernel(); print('NEST успешно установлен')"

После установки перезапустите CogniFlex для активации нейроморфного симулятора.
"""
        logger.info("Инструкции по установке NEST:" + instructions)

    def get_neuromorphic_dashboard_data(self) -> Dict[str, Any]:
        """
        Возвращает данные для дашборда нейроморфной системы.
        Returns:
            Dict: Данные для дашборда
        """
        health = self.get_system_health()
        analysis = health["analysis"]
        # Получаем данные для временных рядов
        trends = {
            "working_activity": [],
            "semantic_activity": [],
            "episodic_activity": [],
            "coherence": []
        }
        # Генерируем данные за последние 5 минут (300 секунд)
        current_time = time.time()
        for i in range(30):  # 30 точек данных
            timestamp = current_time - (i * 10)  # Каждые 10 секунд
            # Генерируем данные для рабочей памяти
            working_activity = max(0.0, min(1.0, 0.5 + np.random.randn() * 0.1))
            trends["working_activity"].append({
                "timestamp": timestamp,
                "value": working_activity
            })
            # Генерируем данные для семантической памяти
            semantic_activity = max(0.0, min(1.0, 0.6 + np.random.randn() * 0.1))
            trends["semantic_activity"].append({
                "timestamp": timestamp,
                "value": semantic_activity
            })
            # Генерируем данные для эпизодической памяти
            episodic_activity = max(0.0, min(1.0, 0.4 + np.random.randn() * 0.1))
            trends["episodic_activity"].append({
                "timestamp": timestamp,
                "value": episodic_activity
            })
            # Генерируем данные когерентности
            coherence = max(0.0, min(1.0, 0.7 + np.random.randn() * 0.1))
            trends["coherence"].append({
                "timestamp": timestamp,
                "value": coherence
            })
        # Сортируем по времени
        for key in trends:
            trends[key].sort(key=lambda x: x["timestamp"])
        return {
            "health": health,
            "trends": trends,
            "recent_activities": [act.to_dict() for act in self.activity_history[-10:]],
            "networks": {name: net.get_network_info() for name, net in self.neural_networks.items()},
            "timestamp": time.time()
        }

    def export_neuromorphic_dashboard_data(self, file_path: str) -> bool:
        """
        Экспортирует данные дашборда нейроморфной системы в файл.
        Args:
            file_path: Путь к файлу для экспорта
        Returns:
            bool: Успешно ли экспортировано
        """
        try:
            # Получаем данные дашборда
            dashboard_data = {
                "metadata": {
                    "export_time": time.time(),
                    "export_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0",
                    "format": "neuromorphic_dashboard"
                },
                "dashboard": self.get_neuromorphic_dashboard_data()
            }
            # Сохраняем в JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Данные дашборда нейроморфной системы экспортированы в {file_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта данных дашборда нейроморфной системы: {e}")
            return False

    def get_learning_opportunities(self) -> List[Dict[str, Any]]:
        """
        Возвращает возможности для обучения на основе анализа нейронной активности.
        Returns:
            List[Dict[str, Any]]: Возможности для обучения
        """
        opportunities = []
        # Анализируем активность
        analysis = self.analyze_neural_activity()
        # 1. Низкая когерентность
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            if analysis["latest_activity"]["coherence"] < 0.4:
                opportunities.append({
                    "type": "coherence_training",
                    "priority": 0.8,
                    "description": "Низкая когерентность нейронной активности",
                    "suggested_action": "Провести обучение для улучшения согласованности активности"
                })
        # 2. Низкое взаимодействие между типами памяти
        if analysis["interaction_strength"] < 0.3:
            opportunities.append({
                "type": "cross_memory_training",
                "priority": 0.9,
                "description": "Низкая связь между различными типами памяти",
                "suggested_action": "Провести обучение для улучшения взаимодействия между типами памяти"
            })
        # 3. Дисбаланс активности
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                opportunities.append({
                    "type": "activity_reduction",
                    "priority": 0.7,
                    "description": f"Перегрузка {memory_type} памяти",
                    "suggested_action": f"Снизить активность {memory_type} памяти"
                })
            elif data["average_activity"] < 0.2:
                opportunities.append({
                    "type": "activity_increase",
                    "priority": 0.6,
                    "description": f"Недостаточная активность {memory_type} памяти",
                    "suggested_action": f"Увеличить активность {memory_type} памяти"
                })
        # Сортируем по приоритету
        opportunities.sort(key=lambda x: x["priority"], reverse=True)
        return opportunities

    def generate_learning_plan(self) -> List[Dict[str, Any]]:
        """
        Генерирует план обучения на основе возможностей.
        Returns:
            List[Dict[str, Any]]: План обучения
        """
        opportunities = self.get_learning_opportunities()
        learning_plan = []
        for i, opportunity in enumerate(opportunities[:5]):  # Ограничиваем 5 возможностями
            # Определяем тип задачи
            opp_type = opportunity["type"]
            if opp_type == "coherence_training":
                task_type = "coherence_improvement"
                description = "Улучшение когерентности нейронной активности"
            elif opp_type == "cross_memory_training":
                task_type = "cross_memory_integration"
                description = "Улучшение взаимодействия между типами памяти"
            else:  # activity_balance
                task_type = "activity_calibration"
                description = opportunity["suggested_action"]
            # Формируем задачу
            task = {
                "id": f"task_{i}_{hash(description) % 1000000}",
                "type": task_type,
                "description": description,
                "priority": opportunity["priority"],
                "estimated_time": "30-60 минут",
                "resources": ["Нейроморфный симулятор", "Анализ активности", "Консолидация"],
                "dependencies": []
            }
            learning_plan.append(task)
        return learning_plan

    def get_system_insights(self) -> Dict[str, Any]:
        """
        Возвращает аналитические данные по нейроморфной системе.
        Returns:
            Dict: Аналитические данные
        """
        # Получаем анализ
        analysis = self.analyze_neural_activity()
        # Определяем проблемные области
        problem_areas = []
        if analysis["interaction_strength"] < 0.3:
            problem_areas.append("Низкая связь между типами памяти")
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                problem_areas.append(f"Перегрузка {memory_type} памяти")
            elif data["average_activity"] < 0.2:
                problem_areas.append(f"Недостаточная активность {memory_type} памяти")
        # Формируем рекомендации
        recommendations = []
        if analysis["interaction_strength"] < 0.3:
            recommendations.append(
                "Улучшить взаимодействие между различными типами памяти через установление дополнительных связей."
            )
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                recommendations.append(
                    f"Снизить нагрузку на {memory_type} память для предотвращения перегрузки."
                )
            elif data["average_activity"] < 0.2:
                recommendations.append(
                    f"Увеличить стимуляцию {memory_type} памяти для повышения активности."
                )
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            if analysis["latest_activity"]["coherence"] < 0.4:
                recommendations.append(
                    "Провести корректировку параметров сети для повышения когерентности активности."
                )
        return {
            "total_activities": analysis["total_activities"],
            "interaction_strength": analysis["interaction_strength"],
            "problem_areas": problem_areas,
            "recommendations": recommendations,
            "timestamp": time.time()
        }

    def get_system_summary(self) -> str:
        """
        Возвращает краткую сводку о системе.
        Returns:
            str: Сводка о системе
        """
        health = self.get_system_health()
        analysis = health["analysis"]
        summary = (
            f"Нейроморфная система: активность {analysis['total_activities']}, "
            f"взаимодействие {analysis['interaction_strength']:.2f}, "
            f"статус: {health['status']}\n"
            f"Здоровье: {health['health_score']}/100"
        )
        return summary

# Пример использования
if __name__ == "__main__":
    # Создаем симулятор
    simulator = NeuromorphicSimulator()
    # Запускаем симулятор
    simulator.start()
    # Симулируем активность
    activity = simulator.simulate_activity(duration=5.0, memory_type="working")
    print(f"Симулированная активность: {activity.get_analysis()}")
    # Получаем отчет о здоровье
    health_report = simulator.get_system_health_report()
    print(f"Отчет о здоровье: {health_report}")
    # Останавливаем симулятор
    simulator.stop()