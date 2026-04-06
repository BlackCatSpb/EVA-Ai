"""Нейроморфный симулятор — ядро, инициализация, жизненный цикл."""
import os
import logging
import time
import threading
import json
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva.neuromorphic")

import base64
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False
    plt = None

NEST_AVAILABLE = False
try:
    import nest
    NEST_AVAILABLE = True
    logger.info("Модуль NEST для нейроморфного моделирования доступен")
except ImportError:
    logger.warning("Модуль NEST не найден.")
except Exception as e:
    logger.warning(f"Ошибка при импорте NEST: {e}")

from io import BytesIO
from dataclasses import dataclass, field

from .sim_neurons import FallbackNeuralNetwork
from .sim_spikes import NeuralActivity


class NeuromorphicSimulator:
    """Симулятор нейроморфных сетей для ЕВА с интеграцией фрактального хранилища."""

    def __init__(self, cache_dir: Optional[str] = None, brain=None, fractal_store=None):
        self.brain = brain
        self.fractal_store = fractal_store
        self.cache_dir = cache_dir or "eva_neuromorphic_cache"
        self.use_nest = NEST_AVAILABLE

        os.makedirs(self.cache_dir, exist_ok=True)

        self.activity_cache_file = os.path.join(self.cache_dir, "neural_activity.json")
        self.network_cache_file = os.path.join(self.cache_dir, "neural_networks.json")

        self.simulation_interval = 1.0
        self.consolidation_interval = 300.0

        self.neural_networks: Dict[str, FallbackNeuralNetwork] = {}
        self.activity_history: List[NeuralActivity] = []

        self.consolidation_cycles = 0
        self.last_consolidation = time.time()
        self.activation_threshold = 0.7
        self.consolidation_strength = 0.3

        self.simulation_thread: Optional[threading.Thread] = None
        self.consolidation_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.initialized = False
        self.running = False

        self.initialize()
        logger.info("Нейроморфный симулятор инициализирован")

    def initialize(self):
        logger.info("Инициализация нейроморфного симулятора...")
        self._load_activity_cache()
        self._init_neural_networks()
        if self.fractal_store:
            self._register_fractal_store_handlers()
        self.initialized = True
        return True

    def _register_fractal_store_handlers(self):
        try:
            if hasattr(self.fractal_store, 'register_event_handler'):
                self.fractal_store.register_event_handler("container_accessed", self._on_container_accessed)
                self.fractal_store.register_event_handler("hot_window_updated", self._on_hot_window_updated)
                logger.debug("Обработчики событий фрактального хранилища зарегистрированы")
        except Exception as e:
            logger.error(f"Ошибка регистрации обработчиков фрактального хранилища: {e}")

    def _on_container_accessed(self, container_id: str):
        try:
            if self.fractal_store and container_id in self.fractal_store.containers:
                container = self.fractal_store.containers[container_id]
                activity = NeuralActivity(
                    activity_pattern=[container.metadata.get("strength", 0.5)] * 10,
                    memory_type="working",
                    strength=container.metadata.get("strength", 0.5),
                    importance=0.7,
                    context={"container_id": container_id, "event": "container_accessed"},
                    metadata={"level": container.level, "domain": container.metadata.get("domain", "general")}
                )
                self.activity_history.append(activity)
                logger.debug(f"Зарегистрирована активность для контейнера {container_id}")
        except Exception as e:
            logger.error(f"Ошибка обработки доступа к контейнеру {container_id}: {e}")

    def _on_hot_window_updated(self, updated_containers: Dict[str, float]):
        try:
            if "working" in self.neural_networks:
                network = self.neural_networks["working"]
                input_stimulus = np.zeros(network.num_neurons)
                for i, (container_id, priority) in enumerate(list(updated_containers.items())[:network.num_neurons]):
                    input_stimulus[i] = priority
                network.simulate_step(input_stimulus)
                logger.debug("Нейронная сеть обновлена на основе горячего окна")
        except Exception as e:
            logger.error(f"Ошибка обработки обновления горячего окна: {e}")

    def _init_neural_networks(self):
        memory_types = {"working": 100, "semantic": 500, "episodic": 300}
        for memory_type, num_neurons in memory_types.items():
            try:
                self.neural_networks[memory_type] = FallbackNeuralNetwork(num_neurons, memory_type)
                logger.info(f"Создана сеть для {memory_type} памяти с {num_neurons} нейронами")
            except Exception as e:
                logger.error(f"Ошибка инициализации сети для {memory_type} памяти: {e}")

    def simulate_activity(self, duration: float = 10.0, memory_type: str = "working") -> NeuralActivity:
        if not self.running:
            logger.warning("Симулятор не запущен. Запускаем...")
            self.start()

        steps = int(duration / self.simulation_interval)
        if steps <= 0:
            steps = 1

        network = self.neural_networks.get(memory_type)
        if not network:
            logger.error(f"Сеть для типа памяти {memory_type} не найдена")
            return NeuralActivity(memory_type=memory_type)

        patterns = []
        for _ in range(steps):
            input_stimulus = self._generate_input_stimulus(memory_type, network.num_neurons)
            pattern = network.simulate_step(input_stimulus)
            patterns.append(pattern)

        if patterns:
            combined_pattern = np.mean(patterns, axis=0)
            activity = NeuralActivity(
                activity_pattern=combined_pattern.tolist(),
                memory_type=memory_type,
                strength=np.mean(combined_pattern),
                importance=np.std(combined_pattern)
            )
        else:
            activity = NeuralActivity(memory_type=memory_type)

        self.activity_history.append(activity)
        if len(self.activity_history) > 5000:
            self.activity_history = self.activity_history[-5000:]
        if len(self.activity_history) % 100 == 0:
            self._save_activity_cache()

        return activity

    def _generate_input_stimulus(self, memory_type: str, num_neurons: int) -> np.ndarray:
        try:
            if not self.fractal_store:
                return np.random.rand(num_neurons) * 0.2
            hot_window = getattr(self.fractal_store, 'hot_window', {})
            if not hot_window:
                return np.random.rand(num_neurons) * 0.2
            stimulus = np.zeros(num_neurons)
            sorted_containers = sorted(hot_window.items(), key=lambda x: x[1], reverse=True)
            for i, (container_id, priority) in enumerate(sorted_containers[:num_neurons]):
                stimulus[i] = priority * 0.8
            return stimulus
        except Exception:
            return np.random.rand(num_neurons) * 0.2

    def consolidate_activity(self) -> bool:
        logger.info("Консолидация нейронной активности через фрактальное хранилище...")
        try:
            current_time = time.time()
            if current_time - self.last_consolidation < self.consolidation_interval:
                return False
            self.last_consolidation = current_time
            self.consolidation_cycles += 1
            activation_patterns = self._analyze_activation_patterns()
            if activation_patterns["stable_patterns"] or activation_patterns["new_patterns"]:
                self._update_fractal_structure(activation_patterns)
            logger.info(f"Консолидация памяти завершена (цикл {self.consolidation_cycles})")
            return True
        except Exception as e:
            logger.error(f"Ошибка консолидации: {e}")
            return False

    def _analyze_activation_patterns(self) -> Dict:
        if len(self.activity_history) < 2:
            return {"stable_patterns": [], "new_patterns": [], "weak_zones": []}
        current_activation = self.activity_history[-1]
        previous_activation = self.activity_history[-2]
        current_pattern = current_activation.activity_pattern or []
        previous_pattern = previous_activation.activity_pattern or []

        stable_patterns = []
        for idx, current_val in enumerate(current_pattern):
            if current_val > 0.7 and idx < len(previous_pattern):
                prev_val = previous_pattern[idx]
                if prev_val > 0.5:
                    stability = min(current_val, prev_val)
                    stable_patterns.append((idx, stability))

        new_patterns = []
        for idx, current_val in enumerate(current_pattern):
            if current_val > 0.7:
                prev_val = previous_pattern[idx] if idx < len(previous_pattern) else 0.0
                if current_val - prev_val > 0.4:
                    novelty = current_val - prev_val
                    new_patterns.append((idx, novelty))

        return {
            "stable_patterns": stable_patterns,
            "new_patterns": new_patterns,
            "weak_zones": current_activation.metadata.get("weak_zones", [])
        }

    def _update_fractal_structure(self, activation_patterns: Dict):
        if not self.fractal_store:
            return
        for container_id, stability in activation_patterns["stable_patterns"]:
            self._strengthen_container_connections(container_id, stability)
        for container_id, novelty in activation_patterns["new_patterns"]:
            self._integrate_new_pattern(container_id, novelty)
        for weak_zone in activation_patterns["weak_zones"]:
            self._reinforce_weak_zone(weak_zone)

    def _strengthen_container_connections(self, container_id: str, strength: float):
        if not self.fractal_store or container_id not in self.fractal_store.containers:
            return
        container = self.fractal_store.containers[container_id]
        if container.metadata and isinstance(container.metadata, dict) and "strength" in container.metadata:
            container.metadata["strength"] = min(1.0, container.metadata["strength"] + strength * 0.2)

    def _integrate_new_pattern(self, container_id: str, novelty: float):
        if not self.fractal_store or container_id not in self.fractal_store.containers:
            return
        container = self.fractal_store.containers[container_id]
        container.metadata = container.metadata or {}
        container.metadata["strength"] = min(1.0, container.metadata.get("strength", 0.3) + novelty * 0.4)
        if container.metadata.get("strength", 0) > 0.7:
            if hasattr(self.fractal_store, '_add_to_hot_window'):
                self.fractal_store._add_to_hot_window([(container_id, container.metadata["strength"], "neural_activation")])

    def _reinforce_weak_zone(self, weak_zone: Dict):
        if not self.fractal_store:
            return
        container_id = weak_zone.get("container_id")
        if not container_id or container_id not in self.fractal_store.containers:
            return
        container = self.fractal_store.containers[container_id]
        if not hasattr(container, 'metadata') or container.metadata is None:
            container.metadata = {}
        current_strength = container.metadata.get("strength", 0.3)
        container.metadata["strength"] = min(1.0, current_strength + 0.2)

    def start(self):
        if not self.initialized:
            logger.warning("Попытка запуска неинициализированного симулятора")
            return
        if self.running:
            logger.warning("Нейроморфный симулятор уже запущен")
            return
        self.running = True
        self.stop_event.clear()

        self.simulation_thread = threading.Thread(
            target=self._simulation_worker, daemon=True, name="NeuromorphicSimulationThread"
        )
        self.simulation_thread.start()

        self.consolidation_thread = threading.Thread(
            target=self._consolidation_worker, daemon=True, name="NeuromorphicConsolidationThread"
        )
        self.consolidation_thread.start()

        logger.info("Нейроморфный симулятор запущен")

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5.0)
        self._save_activity_cache()
        logger.info("Нейроморфный симулятор остановлен")

    def _simulation_worker(self):
        while self.running and not self.stop_event.is_set():
            try:
                self._simulate_step()
                if self.stop_event.wait(self.simulation_interval):
                    break
            except Exception as e:
                logger.error(f"Ошибка в потоке симуляции: {e}")
                time.sleep(10)

    def _consolidation_worker(self):
        while self.running and not self.stop_event.is_set():
            try:
                if self.stop_event.wait(self.consolidation_interval):
                    break
                if self.running:
                    self.consolidate_activity()
            except Exception as e:
                logger.error(f"Ошибка в потоке консолидации: {e}")
                time.sleep(60)

    def _simulate_step(self):
        for memory_type, network in self.neural_networks.items():
            try:
                input_stimulus = self._generate_input_stimulus(memory_type, network.num_neurons)
                activity_pattern = network.simulate_step(input_stimulus)
                activity = NeuralActivity(
                    activity_pattern=activity_pattern.tolist(),
                    memory_type=memory_type,
                    strength=np.mean(activity_pattern),
                    importance=np.std(activity_pattern)
                )
                self.activity_history.append(activity)
                if len(self.activity_history) > 5000:
                    self.activity_history = self.activity_history[-5000:]
            except Exception as e:
                logger.error(f"Ошибка симуляции для сети {memory_type}: {e}")

    def _load_activity_cache(self):
        try:
            if os.path.exists(self.activity_cache_file):
                with open(self.activity_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                for activity_data in cache_data.get("activity_history", []):
                    self.activity_history.append(NeuralActivity.from_dict(activity_data))
                logger.info(f"Загружено {len(self.activity_history)} записей активности")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша активности: {e}")

    def _save_activity_cache(self):
        try:
            cache_data = {
                "activity_history": [act.to_dict() for act in self.activity_history[-1000:]]
            }
            with open(self.activity_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша активности: {e}")

    def analyze_neural_activity(self) -> Dict[str, Any]:
        if not self.activity_history:
            return {"status": "no_data"}

        recent_activity = self.activity_history[-100:]
        memory_analysis = {}

        for activity in recent_activity:
            mem_type = activity.memory_type
            if mem_type not in memory_analysis:
                memory_analysis[mem_type] = {
                    "activities": [], "total_strength": 0.0, "total_importance": 0.0
                }
            memory_analysis[mem_type]["activities"].append(activity)
            memory_analysis[mem_type]["total_strength"] += activity.strength
            memory_analysis[mem_type]["total_importance"] += activity.importance

        for mem_type, data in memory_analysis.items():
            count = len(data["activities"])
            if count > 0:
                data["average_activity"] = np.mean([np.mean(a.activity_pattern) for a in data["activities"]])
                data["average_strength"] = data["total_strength"] / count
                data["average_importance"] = data["total_importance"] / count
                if data["activities"]:
                    last_activity = data["activities"][-1]
                    coherence_data = last_activity.get_analysis()
                    data["latest_coherence"] = coherence_data.get("coherence", 0.5)
            else:
                data["average_activity"] = 0.0
                data["average_strength"] = 0.0
                data["average_importance"] = 0.0
                data["latest_coherence"] = 0.5

        interaction_strength = 0.0
        mem_types = list(memory_analysis.keys())
        if len(mem_types) > 1:
            correlations = []
            for i in range(len(mem_types)):
                for j in range(i + 1, len(mem_types)):
                    type1 = mem_types[i]
                    type2 = mem_types[j]
                    acts1 = [a.strength for a in memory_analysis[type1]["activities"][-10:]]
                    acts2 = [a.strength for a in memory_analysis[type2]["activities"][-10:]]
                    min_len = min(len(acts1), len(acts2))
                    if min_len > 1:
                        a1 = np.asarray(acts1[:min_len], dtype=float)
                        a2 = np.asarray(acts2[:min_len], dtype=float)
                        corr = np.corrcoef(a1, a2)[0, 1]
                        correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            if correlations:
                interaction_strength = np.mean(correlations)

        latest_activity = recent_activity[-1] if recent_activity else None
        latest_analysis = latest_activity.get_analysis() if latest_activity else {}

        return {
            "memory_types": memory_analysis,
            "interaction_strength": interaction_strength,
            "latest_activity": latest_analysis,
            "total_activities": len(self.activity_history),
            "analyzed_activities": len(recent_activity)
        }

    def _visualize_activity(self, memory_type: str) -> str:
        try:
            activities = [act for act in self.activity_history[-50:] if act.memory_type == memory_type]
            if not activities:
                return ""
            fig, ax = plt.subplots(figsize=(10, 6))
            timestamps = [act.timestamp for act in activities]
            strengths = [act.strength for act in activities]
            ax.plot(timestamps, strengths, 'b-', linewidth=2, label='Сила активности')
            ax.set_xlabel('Время')
            ax.set_ylabel('Сила')
            ax.set_title(f'Нейронная активность: {memory_type}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Ошибка создания визуализации для {memory_type}: {e}")
            return ""

    def get_recent_weak_zones(self, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            weak_zones = []
            recent_activities = self.activity_history[-100:]
            container_activities = {}
            for activity in recent_activities:
                container_id = activity.context.get("container_id")
                if container_id:
                    if container_id not in container_activities:
                        container_activities[container_id] = []
                    container_activities[container_id].append(activity)

            for container_id, activities in container_activities.items():
                if len(activities) >= 3:
                    strengths = [act.strength for act in activities[-10:]]
                    avg_strength = np.mean(strengths)
                    min_strength = np.min(strengths)
                    if avg_strength < 0.4:
                        domain = activities[0].context.get("domain", "general")
                        weak_zones.append({
                            "container_id": container_id, "domain": domain,
                            "activation": avg_strength, "min_activation": min_strength,
                            "measurements": len(strengths), "last_updated": activities[-1].timestamp
                        })

            weak_zones.sort(key=lambda x: x["activation"])
            return weak_zones[:limit]
        except Exception as e:
            logger.error(f"Ошибка получения слабых зон: {e}")
            return []

    def get_network_stats(self) -> Dict[str, Any]:
        try:
            stats = {
                "networks": {},
                "total_neurons": 0,
                "active_networks": len(self.neural_networks),
                "simulation_cycles": sum(
                    getattr(network, 'simulation_steps', 0)
                    for network in self.neural_networks.values()
                )
            }
            for memory_type, network in self.neural_networks.items():
                network_info = network.get_network_info()
                stats["networks"][memory_type] = network_info
                stats["total_neurons"] += network_info.get("num_neurons", 0)
            return stats
        except Exception as e:
            logger.error(f"Ошибка получения статистики сетей: {e}")
            return {"error": str(e)}

    def reset_simulation(self):
        try:
            logger.info("Сброс нейроморфной симуляции...")
            if self.running:
                self.stop()
            self.activity_history.clear()
            for network in self.neural_networks.values():
                if hasattr(network, 'neuron_states'):
                    network.neuron_states = np.random.rand(network.num_neurons) * 0.1
                if hasattr(network, 'simulation_steps'):
                    network.simulation_steps = 0
            self.consolidation_cycles = 0
            self.last_consolidation = time.time()
            logger.info("Нейроморфная симуляция сброшена")
        except Exception as e:
            logger.error(f"Ошибка сброса симуляции: {e}")

    def export_data(self, output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "total_activities": len(self.activity_history),
                    "networks": self.get_network_stats(),
                    "simulation_config": {
                        "use_nest": self.use_nest,
                        "simulation_interval": self.simulation_interval,
                        "consolidation_interval": self.consolidation_interval
                    }
                },
                "activity_history": [act.to_dict() for act in self.activity_history[-1000:]],
                "neural_networks": {
                    mem_type: network.get_network_info()
                    for mem_type, network in self.neural_networks.items()
                }
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Данные симуляции экспортированы в {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта данных: {e}")
            return False

    def get_system_health(self) -> Dict[str, Any]:
        analysis = self.analyze_neural_activity()
        if analysis.get("status") == "no_data":
            return {"health_score": 100, "status": "no_data", "message": "Недостаточно данных для анализа"}

        health_score = 100
        for memory_type, data in analysis.get("memory_types", {}).items():
            avg_activity = data["average_activity"]
            if avg_activity < 0.2 or avg_activity > 0.8:
                health_score -= 15
        if analysis.get("interaction_strength", 0.5) < 0.2:
            health_score -= 20
        elif analysis["interaction_strength"] < 0.4:
            health_score -= 10
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            coherence = analysis["latest_activity"]["coherence"]
            if coherence < 0.3:
                health_score -= 25
            elif coherence < 0.5:
                health_score -= 15
        health_score = max(0, min(100, health_score))
        if health_score > 80:
            status = "healthy"
        elif health_score > 50:
            status = "warning"
        else:
            status = "critical"
        return {
            "status": status, "health_score": health_score, "analysis": analysis,
            "networks": {name: net.get_network_info() for name, net in self.neural_networks.items()},
            "timestamp": time.time()
        }

    def get_system_health_report(self) -> Dict[str, Any]:
        health = self.get_system_health()
        analysis = health["analysis"]
        recommendations = self._generate_health_recommendations(analysis, health["health_score"])
        return {"health": health, "recommendations": recommendations, "timestamp": time.time()}

    def _generate_health_recommendations(self, analysis: Dict, health_score: float) -> List[str]:
        recommendations = []
        for memory_type, data in analysis["memory_types"].items():
            avg_activity = data["average_activity"]
            if avg_activity < 0.2:
                recommendations.append(f"Низкая активность {memory_type} памяти. Рассмотрите возможность увеличения стимуляции.")
            elif avg_activity > 0.8:
                recommendations.append(f"Высокая активность {memory_type} памяти. Рассмотрите возможность снижения нагрузки.")
        if analysis["interaction_strength"] < 0.3:
            recommendations.append("Низкое взаимодействие между типами памяти. Рассмотрите улучшение связей между сетями.")
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            coherence = analysis["latest_activity"]["coherence"]
            if coherence < 0.4:
                recommendations.append("Низкая когерентность нейронной активности. Рассмотрите корректировку параметров сети.")
        if health_score < 50:
            recommendations.append("Общее состояние системы критическое. Проведите полную диагностику и перезапуск.")
        elif health_score < 70:
            recommendations.append("Общее состояние системы требует внимания. Проверьте параметры симуляции.")
        if not recommendations:
            recommendations.append("Система находится в хорошем состоянии. Нет критических проблем.")
        return recommendations

    def get_fix_suggestions(self) -> List[Dict[str, Any]]:
        health = self.get_system_health()
        analysis = health["analysis"]
        fixes = []
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                fixes.append({
                    "id": f"reduce_{memory_type}_activity",
                    "title": f"Снизить активность {memory_type} памяти",
                    "description": f"Активность {memory_type} памяти превышает оптимальный уровень",
                    "severity": "high",
                    "action": lambda mt=memory_type: self._adjust_memory_activity(mt, -0.2)
                })
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] < 0.2:
                fixes.append({
                    "id": f"increase_{memory_type}_activity",
                    "title": f"Увеличить активность {memory_type} памяти",
                    "description": f"Активность {memory_type} памяти ниже оптимального уровня",
                    "severity": "medium",
                    "action": lambda mt=memory_type: self._adjust_memory_activity(mt, 0.2)
                })
        if analysis["interaction_strength"] < 0.3:
            fixes.append({
                "id": "improve_interaction", "title": "Улучшить взаимодействие между типами памяти",
                "description": "Низкая связь между различными типами памяти",
                "severity": "high", "action": self._improve_memory_interaction
            })
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            if analysis["latest_activity"]["coherence"] < 0.4:
                fixes.append({
                    "id": "improve_coherence", "title": "Улучшить когерентность нейронной активности",
                    "description": "Низкая согласованность нейронной активности",
                    "severity": "medium", "action": self._improve_coherence
                })
        if not self.use_nest:
            fixes.append({
                "id": "install_nest", "title": "Установить NEST",
                "description": "Позволит использовать полную функциональность нейроморфного моделирования",
                "severity": "high", "action": self._install_nest_instructions
            })
        return fixes

    def _adjust_memory_activity(self, memory_type: str, delta: float):
        if memory_type in self.neural_networks:
            network = self.neural_networks[memory_type]
            for i in range(len(network.neuron_states)):
                network.neuron_states[i] = max(0.0, min(1.0, network.neuron_states[i] + delta))
            logger.info(f"Скорректирована активность {memory_type} памяти на {delta}")
        else:
            logger.warning(f"Неизвестный тип памяти для корректировки: {memory_type}")

    def _improve_memory_interaction(self):
        for src_net in self.neural_networks.values():
            for tgt_net in self.neural_networks.values():
                if src_net != tgt_net:
                    for _ in range(5):
                        i = np.random.randint(0, src_net.num_neurons)
                        j = np.random.randint(0, tgt_net.num_neurons)
                        src_net.connections[i, j] = min(1.0, src_net.connections[i, j] * 1.1)
        logger.info("Улучшено взаимодействие между типами памяти")

    def _improve_coherence(self):
        logger.info("Улучшение когерентности нейронной активности")
        coherence_count = 0
        for network in self.neural_networks.values():
            if hasattr(network, 'noise_level') and network.noise_level > 0.1:
                network.noise_level = max(0.01, network.noise_level * 0.9)
                coherence_count += 1
            if hasattr(network, 'connections'):
                conn_sum = network.connections.sum() if hasattr(network.connections, 'sum') else 0
                if conn_sum > 0:
                    diag_mask = np.eye(network.connections.shape[0])
                    network.connections = network.connections * 0.95 + diag_mask * network.connections * 0.05
                    coherence_count += 1
        logger.info(f"Улучшена когерентность в {coherence_count} сетях")

    def _install_nest_instructions(self):
        instructions = """
Для установки NEST выполните следующие команды:

1. Установите зависимости:
    Ubuntu/Debian: sudo apt-get install build-essential cmake python3-dev
    CentOS/RHEL: sudo yum install gcc gcc-c++ make cmake python3-devel

2. Скачайте и установите NEST:
    pip install nest-simulator

3. Проверьте установку:
    python -c "import nest; nest.ResetKernel(); print('NEST успешно установлен')"

После установки перезапустите ЕВА для активации нейроморфного симулятора.
"""
        logger.info("Инструкции по установке NEST:" + instructions)

    def get_neuromorphic_dashboard_data(self) -> Dict[str, Any]:
        health = self.get_system_health()
        analysis = health["analysis"]
        trends = {
            "working_activity": [], "semantic_activity": [],
            "episodic_activity": [], "coherence": []
        }
        current_time = time.time()
        for i in range(30):
            timestamp = current_time - (i * 10)
            trends["working_activity"].append({"timestamp": timestamp, "value": max(0.0, min(1.0, 0.5 + np.random.randn() * 0.1))})
            trends["semantic_activity"].append({"timestamp": timestamp, "value": max(0.0, min(1.0, 0.6 + np.random.randn() * 0.1))})
            trends["episodic_activity"].append({"timestamp": timestamp, "value": max(0.0, min(1.0, 0.4 + np.random.randn() * 0.1))})
            trends["coherence"].append({"timestamp": timestamp, "value": max(0.0, min(1.0, 0.7 + np.random.randn() * 0.1))})
        for key in trends:
            trends[key].sort(key=lambda x: x["timestamp"])
        return {
            "health": health, "trends": trends,
            "recent_activities": [act.to_dict() for act in self.activity_history[-10:]],
            "networks": {name: net.get_network_info() for name, net in self.neural_networks.items()},
            "timestamp": time.time()
        }

    def export_neuromorphic_dashboard_data(self, file_path: str) -> bool:
        try:
            dashboard_data = {
                "metadata": {
                    "export_time": time.time(),
                    "export_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0", "format": "neuromorphic_dashboard"
                },
                "dashboard": self.get_neuromorphic_dashboard_data()
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Данные дашборда нейроморфной системы экспортированы в {file_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта данных дашборда нейроморфной системы: {e}")
            return False

    def get_learning_opportunities(self) -> List[Dict[str, Any]]:
        opportunities = []
        analysis = self.analyze_neural_activity()
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            if analysis["latest_activity"]["coherence"] < 0.4:
                opportunities.append({
                    "type": "coherence_training", "priority": 0.8,
                    "description": "Низкая когерентность нейронной активности",
                    "suggested_action": "Провести обучение для улучшения согласованности активности"
                })
        if analysis["interaction_strength"] < 0.3:
            opportunities.append({
                "type": "cross_memory_training", "priority": 0.9,
                "description": "Низкая связь между различными типами памяти",
                "suggested_action": "Провести обучение для улучшения взаимодействия между типами памяти"
            })
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                opportunities.append({
                    "type": "activity_reduction", "priority": 0.7,
                    "description": f"Перегрузка {memory_type} памяти",
                    "suggested_action": f"Снизить активность {memory_type} памяти"
                })
            elif data["average_activity"] < 0.2:
                opportunities.append({
                    "type": "activity_increase", "priority": 0.6,
                    "description": f"Недостаточная активность {memory_type} памяти",
                    "suggested_action": f"Увеличить активность {memory_type} памяти"
                })
        opportunities.sort(key=lambda x: x["priority"], reverse=True)
        return opportunities

    def generate_learning_plan(self) -> List[Dict[str, Any]]:
        opportunities = self.get_learning_opportunities()
        learning_plan = []
        for i, opportunity in enumerate(opportunities[:5]):
            opp_type = opportunity["type"]
            if opp_type == "coherence_training":
                task_type = "coherence_improvement"
                description = "Улучшение когерентности нейронной активности"
            elif opp_type == "cross_memory_training":
                task_type = "cross_memory_integration"
                description = "Улучшение взаимодействия между типами памяти"
            else:
                task_type = "activity_calibration"
                description = opportunity["suggested_action"]
            task = {
                "id": f"task_{i}_{hash(description) % 1000000}",
                "type": task_type, "description": description,
                "priority": opportunity["priority"],
                "estimated_time": "30-60 минут",
                "resources": ["Нейроморфный симулятор", "Анализ активности", "Консолидация"],
                "dependencies": []
            }
            learning_plan.append(task)
        return learning_plan

    def get_system_insights(self) -> Dict[str, Any]:
        analysis = self.analyze_neural_activity()
        problem_areas = []
        if analysis["interaction_strength"] < 0.3:
            problem_areas.append("Низкая связь между типами памяти")
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                problem_areas.append(f"Перегрузка {memory_type} памяти")
            elif data["average_activity"] < 0.2:
                problem_areas.append(f"Недостаточная активность {memory_type} памяти")
        recommendations = []
        if analysis["interaction_strength"] < 0.3:
            recommendations.append("Улучшить взаимодействие между различными типами памяти через установление дополнительных связей.")
        for memory_type, data in analysis["memory_types"].items():
            if data["average_activity"] > 0.8:
                recommendations.append(f"Снизить нагрузку на {memory_type} память для предотвращения перегрузки.")
            elif data["average_activity"] < 0.2:
                recommendations.append(f"Увеличить стимуляцию {memory_type} памяти для повышения активности.")
        if "latest_activity" in analysis and "coherence" in analysis["latest_activity"]:
            if analysis["latest_activity"]["coherence"] < 0.4:
                recommendations.append("Провести корректировку параметров сети для повышения когерентности активности.")
        return {
            "total_activities": analysis["total_activities"],
            "interaction_strength": analysis["interaction_strength"],
            "problem_areas": problem_areas, "recommendations": recommendations,
            "timestamp": time.time()
        }

    def get_system_summary(self) -> str:
        health = self.get_system_health()
        analysis = health["analysis"]
        return (
            f"Нейроморфная система: активность {analysis['total_activities']}, "
            f"взаимодействие {analysis['interaction_strength']:.2f}, "
            f"статус: {health['status']}\n"
            f"Здоровье: {health['health_score']}/100"
        )


if __name__ == "__main__":
    simulator = NeuromorphicSimulator()
    simulator.start()
    activity = simulator.simulate_activity(duration=5.0, memory_type="working")
    print(f"Симулированная активность: {activity.get_analysis()}")
    health_report = simulator.get_system_health_report()
    print(f"Отчет о здоровье: {health_report}")
    simulator.stop()
