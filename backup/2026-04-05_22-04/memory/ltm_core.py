"""Core module for LongTermMemory - main classes, initialization, lifecycle."""
import logging
import time
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
import re

from .memory_core import MemoryNeuron, MemoryField, MemoryDatabase
from .memory_working import WorkingMemory

logger = logging.getLogger("eva.memory.long_term")


class SemanticMemory:
    """Семантическая долгосрочная память для хранения общих знаний и фактов."""

    def __init__(self, capacity: int = 5000, consolidation_interval: int = 3600,
                 db: Optional[MemoryDatabase] = None, working_memory: Optional[WorkingMemory] = None):
        self.capacity = capacity
        self.consolidation_interval = consolidation_interval
        self.db = db or MemoryDatabase()
        self.working_memory = working_memory

        self.neurons: Dict[str, MemoryNeuron] = {}
        self.fields: Dict[str, MemoryField] = {}
        self.knowledge_graph = defaultdict(list)

        self.running = False
        self.stop_event = threading.Event()
        self.last_consolidation = 0

        self._load_from_db()

        logger.info(f"Семантическая память инициализирована (емкость: {capacity})")

    def _load_from_db(self):
        from .ltm_storage import _load_semantic_from_db
        _load_semantic_from_db(self)

    def _update_knowledge_graph(self, neuron: MemoryNeuron):
        from .ltm_storage import _update_knowledge_graph_impl
        _update_knowledge_graph_impl(self, neuron)

    def consolidate_from_working(self):
        from .ltm_consolidation import _consolidate_from_working_impl
        _consolidate_from_working_impl(self)

    def _is_duplicate(self, neuron: MemoryNeuron) -> bool:
        from .ltm_consolidation import _is_duplicate_impl
        return _is_duplicate_impl(self, neuron)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        set1 = set(re.findall(r'\w+', text1.lower()))
        set2 = set(re.findall(r'\w+', text2.lower()))

        if not set1 or not set2:
            return 0.0

        return len(set1 & set2) / len(set1 | set2)

    def retrieve_by_concept(self, concept: str, nlp_model=None,
                          max_distance: int = 2, top_k: int = 10) -> List[MemoryNeuron]:
        from .ltm_retrieval import _retrieve_by_concept_impl
        return _retrieve_by_concept_impl(self, concept, nlp_model, max_distance, top_k)

    def retrieve_by_similarity(self, query: Any, nlp_model,
                             threshold: float = 0.6, top_k: int = 5) -> List[MemoryNeuron]:
        from .ltm_retrieval import _retrieve_by_similarity_impl
        return _retrieve_by_similarity_impl(self, query, nlp_model, threshold, top_k)

    def get_statistics(self) -> Dict[str, Any]:
        total_neurons = len(self.neurons)
        total_capacity = sum(field.capacity for field in self.fields.values())
        used_capacity = sum(field.current_size for field in self.fields.values())

        avg_importance = np.mean([n.importance for n in self.neurons.values()]) if self.neurons else 0.0

        return {
            "total_neurons": total_neurons,
            "total_capacity": total_capacity,
            "used_capacity": used_capacity,
            "usage": min(1.0, used_capacity / total_capacity) if total_capacity > 0 else 1.0,
            "avg_importance": avg_importance,
            "knowledge_graph_size": len(self.knowledge_graph),
            "fields": {name: field.get_usage_stats() for name, field in self.fields.items()}
        }

    def start(self):
        if self.running:
            return

        self.running = True
        self.stop_event.clear()

        self.consolidation_thread = threading.Thread(target=self._consolidation_worker, daemon=True)
        self.consolidation_thread.start()

        logger.info("Фоновые процессы семантической памяти запущены")

    def _consolidation_worker(self):
        while not self.stop_event.is_set():
            try:
                time.sleep(max(1, self.consolidation_interval - (time.time() - self.last_consolidation)))
                self.consolidate_from_working()
            except Exception as e:
                logger.error(f"Ошибка в процессе консолидации: {e}")
                time.sleep(60)

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        if hasattr(self, 'consolidation_thread') and self.consolidation_thread.is_alive():
            self.consolidation_thread.join(timeout=5.0)

        for neuron in self.neurons.values():
            self.db.save_neuron(neuron)
        for field in self.fields.values():
            self.db.save_field(field)

        logger.info("Фоновые процессы семантической памяти остановлены")

    def get_health_status(self) -> Dict[str, Any]:
        stats = self.get_statistics()

        usage_score = 1.0 - abs(stats["usage"] - 0.6)
        importance_score = min(1.0, stats["avg_importance"] * 1.5)
        graph_density = min(1.0, len(self.knowledge_graph) / (len(self.neurons) * 0.5 + 1))

        health_score = (
            usage_score * 0.4 +
            importance_score * 0.3 +
            graph_density * 0.3
        )

        if health_score > 0.7:
            status = "healthy"
        elif health_score > 0.4:
            status = "warning"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "usage": stats["usage"],
            "avg_importance": stats["avg_importance"],
            "knowledge_graph_size": stats["knowledge_graph_size"],
            "recommendations": self._generate_health_recommendations(stats)
        }

    def _generate_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        from .ltm_consolidation import _generate_health_recommendations_impl
        return _generate_health_recommendations_impl(self, stats)

    def consolidate_memory(self, contradiction_manager=None):
        from .ltm_consolidation import _consolidate_memory_impl
        _consolidate_memory_impl(self, contradiction_manager)

    def _check_for_contradictions(self, contradiction_manager):
        from .ltm_consolidation import _check_for_contradictions_impl
        _check_for_contradictions_impl(self, contradiction_manager)

    def _optimize_knowledge_graph(self):
        from .ltm_consolidation import _optimize_knowledge_graph_impl
        _optimize_knowledge_graph_impl(self)

    def _merge_similar_concepts(self):
        from .ltm_consolidation import _merge_similar_concepts_impl
        _merge_similar_concepts_impl(self)

    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        set1 = set(re.findall(r'\w+', concept1.lower()))
        set2 = set(re.findall(r'\w+', concept2.lower()))

        if not set1 or not set2:
            return 0.0

        return len(set1 & set2) / len(set1 | set2)

    def _merge_concepts(self, target: str, source: str):
        from .ltm_consolidation import _merge_concepts_impl
        _merge_concepts_impl(self, target, source)


class EpisodicMemory:
    """Эпизодическая долгосрочная память для хранения событий и опыта."""

    def __init__(self, capacity: int = 2000, retention_period: int = 31536000,
                 db: Optional[MemoryDatabase] = None, working_memory: Optional[WorkingMemory] = None):
        self.capacity = capacity
        self.retention_period = retention_period
        self.db = db or MemoryDatabase()
        self.working_memory = working_memory

        self.episodes: Dict[str, MemoryNeuron] = {}
        self.user_episodes: Dict[str, List[str]] = defaultdict(list)
        self.temporal_index = []

        self.running = False
        self.stop_event = threading.Event()
        self.last_cleanup = 0

        self._load_from_db()

        logger.info(f"Эпизодическая память инициализирована (емкость: {capacity})")

    def _load_from_db(self):
        from .ltm_storage import _load_episodic_from_db
        _load_episodic_from_db(self)

    def store_episode(self, content: Any, user_id: str = "system",
                     context: Optional[Dict[str, Any]] = None) -> str:
        from .ltm_storage import _store_episode_impl
        return _store_episode_impl(self, content, user_id, context)

    def retrieve_by_time(self, start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        user_id: Optional[str] = None,
                        top_k: int = 10) -> List[MemoryNeuron]:
        from .ltm_retrieval import _retrieve_by_time_impl
        return _retrieve_by_time_impl(self, start_time, end_time, user_id, top_k)

    def retrieve_by_similarity(self, query: Any, user_id: Optional[str] = None,
                             nlp_model=None, threshold: float = 0.6,
                             top_k: int = 5) -> List[MemoryNeuron]:
        from .ltm_retrieval import _retrieve_episodic_by_similarity_impl
        return _retrieve_episodic_by_similarity_impl(self, query, user_id, nlp_model, threshold, top_k)

    def get_user_history(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        from .ltm_retrieval import _get_user_history_impl
        return _get_user_history_impl(self, user_id, days)

    def cleanup_old_episodes(self):
        from .ltm_consolidation import _cleanup_old_episodes_impl
        _cleanup_old_episodes_impl(self)

    def get_statistics(self) -> Dict[str, Any]:
        total_episodes = len(self.episodes)
        users_with_episodes = len(self.user_episodes)

        avg_history_length = (
            np.mean([len(episodes) for episodes in self.user_episodes.values()])
            if self.user_episodes else 0
        )

        return {
            "total_episodes": total_episodes,
            "capacity": self.capacity,
            "usage": min(1.0, total_episodes / self.capacity) if self.capacity > 0 else 1.0,
            "users_with_episodes": users_with_episodes,
            "avg_history_length": avg_history_length,
            "retention_period": self.retention_period
        }

    def start(self):
        if self.running:
            return

        self.running = True
        self.stop_event.clear()

        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

        logger.info("Фоновые процессы эпизодической памяти запущены")

    def _cleanup_worker(self):
        while not self.stop_event.is_set():
            try:
                time.sleep(86400)
                self.cleanup_old_episodes()
            except Exception as e:
                logger.error(f"Ошибка в процессе очистки эпизодической памяти: {e}")
                time.sleep(3600)

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        if hasattr(self, 'cleanup_thread') and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)

        for neuron in self.episodes.values():
            self.db.save_neuron(neuron)

        logger.info("Фоновые процессы эпизодической памяти остановлены")

    def get_health_status(self) -> Dict[str, Any]:
        stats = self.get_statistics()

        usage_score = 1.0 - abs(stats["usage"] - 0.5)
        history_score = min(1.0, stats["avg_history_length"] / 10)

        health_score = (
            usage_score * 0.6 +
            history_score * 0.4
        )

        if health_score > 0.7:
            status = "healthy"
        elif health_score > 0.4:
            status = "warning"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "usage": stats["usage"],
            "users_with_episodes": stats["users_with_episodes"],
            "avg_history_length": stats["avg_history_length"],
            "recommendations": self._generate_health_recommendations(stats)
        }

    def _generate_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        from .ltm_consolidation import _generate_episodic_health_recommendations_impl
        return _generate_episodic_health_recommendations_impl(self, stats)

    def consolidate_from_working(self, user_id: str = "system"):
        from .ltm_consolidation import _episodic_consolidate_from_working_impl
        _episodic_consolidate_from_working_impl(self, user_id)


class LongTermMemory:
    """Unified interface for long-term memory (semantic + episodic)."""

    def __init__(self, semantic_capacity: int = 5000, episodic_capacity: int = 2000,
                 consolidation_interval: int = 3600, retention_period: int = 31536000,
                 db: Optional[MemoryDatabase] = None, working_memory: Optional[WorkingMemory] = None):
        self.semantic = SemanticMemory(
            capacity=semantic_capacity,
            consolidation_interval=consolidation_interval,
            db=db,
            working_memory=working_memory
        )
        self.episodic = EpisodicMemory(
            capacity=episodic_capacity,
            retention_period=retention_period,
            db=db,
            working_memory=working_memory
        )
        logger.info("Долгосрочная память инициализирована")

    def start(self):
        self.semantic.start()
        self.episodic.start()

    def stop(self):
        self.semantic.stop()
        self.episodic.stop()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "semantic": self.semantic.get_statistics(),
            "episodic": self.episodic.get_statistics()
        }

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "semantic": self.semantic.get_health_status(),
            "episodic": self.episodic.get_health_status()
        }
