"""Модуль долгосрочной памяти ЕВА"""
import os
import logging
import time
import threading
import json
import re
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from .memory_core import MemoryNeuron, MemoryField, MemoryDatabase
from .memory_working import WorkingMemory

logger = logging.getLogger("eva.memory.long_term")

class SemanticMemory:
    """Семантическая долгосрочная память для хранения общих знаний и фактов."""
    
    def __init__(self, capacity: int = 5000, consolidation_interval: int = 3600,
                 db: Optional[MemoryDatabase] = None, working_memory: Optional[WorkingMemory] = None):
        """
        Инициализирует семантическую память.
        
        Args:
            capacity: Емкость семантической памяти
            consolidation_interval: Интервал консолидации в секундах
            db: База данных для сохранения
            working_memory: Ссылка на рабочую память
        """
        self.capacity = capacity
        self.consolidation_interval = consolidation_interval
        self.db = db or MemoryDatabase()
        self.working_memory = working_memory
        
        # Основные структуры данных
        self.neurons: Dict[str, MemoryNeuron] = {}
        self.fields: Dict[str, MemoryField] = {}
        self.knowledge_graph = defaultdict(list)  # {concept: [neuron_id]}
        
        # Параметры работы
        self.running = False
        self.stop_event = threading.Event()
        self.last_consolidation = 0
        
        # Загружаем сохраненные данные
        self._load_from_db()
        
        logger.info(f"Семантическая память инициализирована (емкость: {capacity})")
    
    def _load_from_db(self):
        """Загружает данные семантической памяти из базы данных."""
        try:
            # Загружаем поля
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT name FROM memory_fields")
            for (field_name,) in cursor.fetchall():
                field = self.db.load_field(field_name)
                if field and field.metadata.get("memory_type") == "semantic":
                    self.fields[field_name] = field
            
            # Загружаем нейроны
            cursor.execute("SELECT id FROM memory_neurons")
            for (neuron_id,) in cursor.fetchall():
                neuron = self.db.load_neuron(neuron_id)
                if neuron and neuron.metadata.get("memory_type") == "semantic":
                    self.neurons[neuron_id] = neuron
                    # Обновляем граф знаний
                    self._update_knowledge_graph(neuron)
            
            logger.info(f"Загружено {len(self.neurons)} нейронов в семантическую память")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных семантической памяти: {e}")
    
    def _update_knowledge_graph(self, neuron: MemoryNeuron):
        """Обновляет граф знаний на основе нейрона."""
        if neuron.content_type == "fact" and isinstance(neuron.content, dict):
            # Для фактов обновляем связи между концептами
            subject = neuron.content.get("subject")
            predicate = neuron.content.get("predicate")
            obj = neuron.content.get("object")
            
            if subject and obj:
                # Добавляем связь в граф
                self.knowledge_graph[subject].append((obj, predicate, neuron.id))
                self.knowledge_graph[obj].append((subject, f"reverse_{predicate}", neuron.id))
    
    def consolidate_from_working(self):
        """Консолидирует информацию из рабочей памяти в семантическую."""
        if not self.working_memory:
            return
        
        # Получаем кандидатов для консолидации
        candidates = self.working_memory.get_consolidation_candidates()
        
        for neuron in candidates:
            # Проверяем, не дублируется ли информация
            if not self._is_duplicate(neuron):
                # Создаем копию для долгосрочной памяти
                long_term_neuron = MemoryNeuron(
                    id=f"semantic_{neuron.id}",
                    content=neuron.content,
                    content_type=neuron.content_type,
                    strength=neuron.strength,
                    importance=neuron.importance,
                    timestamp=neuron.timestamp,
                    metadata={
                        **neuron.metadata,
                        "memory_type": "semantic",
                        "source_working_id": neuron.id
                    }
                )
                
                # Сохраняем в структуры данных
                self.neurons[long_term_neuron.id] = long_term_neuron
                self._update_knowledge_graph(long_term_neuron)
                
                # Сохраняем в базу данных
                self.db.save_neuron(long_term_neuron)
                
                # Обновляем поля
                field_name = neuron.metadata.get("field", "general")
                if field_name not in self.fields:
                    self.fields[field_name] = MemoryField(
                        name=field_name,
                        description=f"Поле семантической памяти: {field_name}",
                        capacity=self.capacity // 10,
                        metadata={"memory_type": "semantic"}
                    )
                
                self.fields[field_name].current_size += 1
                self.db.save_field(self.fields[field_name])
        
        # Обновляем время последней консолидации
        self.last_consolidation = time.time()
    
    def _is_duplicate(self, neuron: MemoryNeuron) -> bool:
        """
        Проверяет, является ли нейрон дубликатом существующего.
        
        Args:
            neuron: Нейрон для проверки
            
        Returns:
            bool: Является ли дубликатом
        """
        # Для фактов проверяем по ключевым полям
        if neuron.content_type == "fact" and isinstance(neuron.content, dict):
            subject = neuron.content.get("subject")
            predicate = neuron.content.get("predicate")
            obj = neuron.content.get("object")
            
            if subject and predicate and obj:
                # Ищем похожие факты
                for n in self.neurons.values():
                    if n.content_type != "fact" or not isinstance(n.content, dict):
                        continue
                    
                    n_subject = n.content.get("subject")
                    n_predicate = n.content.get("predicate")
                    n_obj = n.content.get("object")
                    
                    if subject == n_subject and predicate == n_predicate and obj == n_obj:
                        return True
        
        # Для текста проверяем семантическое сходство
        elif neuron.content_type == "text":
            for n in self.neurons.values():
                if n.content_type != "text":
                    continue
                
                # Проверяем сходство (в реальной системе здесь будет NLP-модель)
                similarity = self._calculate_text_similarity(str(neuron.content), str(n.content))
                if similarity > 0.8:
                    return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисляет сходство между текстами (упрощенная версия).
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            float: Сходство (0.0-1.0)
        """
        # Упрощенное лексическое сходство
        set1 = set(re.findall(r'\w+', text1.lower()))
        set2 = set(re.findall(r'\w+', text2.lower()))
        
        if not set1 or not set2:
            return 0.0
        
        return len(set1 & set2) / len(set1 | set2)
    
    def retrieve_by_concept(self, concept: str, nlp_model=None, 
                          max_distance: int = 2, top_k: int = 10) -> List[MemoryNeuron]:
        """
        Извлекает информацию, связанную с концептом.
        
        Args:
            concept: Концепт для поиска
            nlp_model: NLP-модель для анализа
            max_distance: Максимальное расстояние в графе
            top_k: Количество результатов
            
        Returns:
            List[MemoryNeuron]: Результаты поиска
        """
        # Ищем в графе знаний
        related_neurons = set()
        queue = [(concept, 0)]
        visited = {concept}
        
        while queue and len(related_neurons) < top_k * 2:
            current_concept, distance = queue.pop(0)
            
            if distance > max_distance:
                continue
            
            # Добавляем связанные нейроны
            for neuron_id in self.knowledge_graph.get(current_concept, []):
                related_neurons.add(neuron_id[2])  # neuron_id
            
            # Расширяем поиск
            for neighbor, _, _ in self.knowledge_graph.get(current_concept, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        # Возвращаем топ-K результатов
        results = []
        for neuron_id in list(related_neurons)[:top_k]:
            if neuron_id in self.neurons:
                results.append(self.neurons[neuron_id])
        
        return results
    
    def retrieve_by_similarity(self, query: Any, nlp_model, 
                             threshold: float = 0.6, top_k: int = 5) -> List[MemoryNeuron]:
        """
        Извлекает информацию по семантическому сходству.
        
        Args:
            query: Запрос
            nlp_model: NLP-модель
            threshold: Порог сходства
            top_k: Количество результатов
            
        Returns:
            List[MemoryNeuron]: Результаты поиска
        """
        results = []
        
        for neuron in self.neurons.values():
            similarity = neuron.get_similarity(MemoryNeuron(
                id="query",
                content=query,
                content_type="text" if isinstance(query, str) else "fact"
            ), nlp_model)
            
            if similarity >= threshold:
                results.append((similarity, neuron))
        
        # Сортируем по сходству
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [neuron for _, neuron in results[:top_k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику семантической памяти.
        
        Returns:
            Dict: Статистика
        """
        total_neurons = len(self.neurons)
        total_capacity = sum(field.capacity for field in self.fields.values())
        used_capacity = sum(field.current_size for field in self.fields.values())
        
        # Вычисляем среднюю важность
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
        """Запускает фоновые процессы семантической памяти."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Запускаем поток консолидации
        self.consolidation_thread = threading.Thread(target=self._consolidation_worker, daemon=True)
        self.consolidation_thread.start()
        
        logger.info("Фоновые процессы семантической памяти запущены")
    
    def _consolidation_worker(self):
        """Рабочий процесс для периодической консолидации."""
        while not self.stop_event.is_set():
            try:
                # Консолидируем каждые consolidation_interval секунд
                time.sleep(max(1, self.consolidation_interval - (time.time() - self.last_consolidation)))
                self.consolidate_from_working()
            except Exception as e:
                logger.error(f"Ошибка в процессе консолидации: {e}")
                time.sleep(60)  # Подождать перед повторной попыткой
    
    def stop(self):
        """Останавливает фоновые процессы семантической памяти."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        # Даем время на завершение
        if hasattr(self, 'consolidation_thread') and self.consolidation_thread.is_alive():
            self.consolidation_thread.join(timeout=5.0)
        
        # Сохраняем данные
        for neuron in self.neurons.values():
            self.db.save_neuron(neuron)
        for field in self.fields.values():
            self.db.save_field(field)
        
        logger.info("Фоновые процессы семантической памяти остановлены")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Возвращает статус здоровья семантической памяти.
        
        Returns:
            Dict: Статус здоровья
        """
        stats = self.get_statistics()
        
        # Вычисляем оценку здоровья
        usage_score = 1.0 - abs(stats["usage"] - 0.6)  # Оптимально около 60%
        importance_score = min(1.0, stats["avg_importance"] * 1.5)
        graph_density = min(1.0, len(self.knowledge_graph) / (len(self.neurons) * 0.5 + 1))
        
        health_score = (
            usage_score * 0.4 +
            importance_score * 0.3 +
            graph_density * 0.3
        )
        
        # Определяем статус
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
        """
        Генерирует рекомендации на основе статуса здоровья.
        
        Args:
            stats: Статистика памяти
            
        Returns:
            List[str]: Рекомендации
        """
        recommendations = []
        
        # Проверяем использование
        if stats["usage"] > 0.9:
            recommendations.append(
                "Семантическая память переполнена. Рассмотрите увеличение емкости "
                "или улучшение фильтрации знаний."
            )
        elif stats["usage"] < 0.3:
            recommendations.append(
                "Семантическая память недостаточно заполнена. Проверьте настройки "
                "порога консолидации."
            )
        
        # Проверяем среднюю важность
        if stats["avg_importance"] < 0.4:
            recommendations.append(
                "Низкая средняя важность знаний. Проверьте критерии отбора "
                "информации для долгосрочного хранения."
            )
        
        # Проверяем плотность графа знаний
        if stats["knowledge_graph_size"] < len(self.neurons) * 0.3:
            recommendations.append(
                "Низкая плотность графа знаний. Рассмотрите улучшение процесса "
                "связывания новых знаний с существующими."
            )
        
        return recommendations
    
    def consolidate_memory(self, contradiction_manager=None):
        """
        Выполняет глубокую консолидацию памяти с проверкой на противоречия.
        
        Args:
            contradiction_manager: Менеджер противоречий
        """
        logger.info("Начало глубокой консолидации семантической памяти")
        
        # Проверяем противоречия, если доступен менеджер противоречий
        if contradiction_manager:
            self._check_for_contradictions(contradiction_manager)
        
        # Оптимизируем связи в графе знаний
        self._optimize_knowledge_graph()
        
        logger.info("Глубокая консолидация семантической памяти завершена")
    
    def _check_for_contradictions(self, contradiction_manager):
        """
        Проверяет знания на наличие противоречий.
        
        Args:
            contradiction_manager: Менеджер противоречий
        """
        # Группируем факты по концептам
        facts_by_concept = defaultdict(list)
        for neuron_id, neuron in self.neurons.items():
            if neuron.content_type == "fact" and isinstance(neuron.content, dict):
                concept = neuron.content.get("subject")
                if concept:
                    facts_by_concept[concept].append({
                        "id": neuron_id,
                        "fact": neuron.content,
                        "strength": neuron.strength,
                        "importance": neuron.importance
                    })
        
        # Проверяем каждую группу на противоречия
        for concept, facts in facts_by_concept.items():
            if len(facts) < 2:
                continue
            
            # Создаем временный факт для проверки
            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    fact1 = facts[i]["fact"]
                    fact2 = facts[j]["fact"]
                    
                    # Пропускаем идентичные факты
                    if fact1 == fact2:
                        continue
                    
                    # Проверяем на противоречие
                    contradiction = contradiction_manager.detector._create_contradiction(
                        concept,
                        [fact1, fact2],
                        0.5,  # Начальный уровень расхождения
                        relation_type=fact1.get("predicate", "related_to")
                    )
                    
                    # Добавляем в менеджер противоречий
                    contradiction_manager.add_contradiction(contradiction)
    
    def _optimize_knowledge_graph(self):
        """Оптимизирует структуру графа знаний."""
        # Удаляем слабые связи
        for concept, connections in list(self.knowledge_graph.items()):
            # Оставляем только сильные связи
            strong_connections = [
                conn for conn in connections 
                if self.neurons.get(conn[2], MemoryNeuron("", "", "")).strength > 0.3
            ]
            if strong_connections:
                self.knowledge_graph[concept] = strong_connections
            else:
                del self.knowledge_graph[concept]
        
        # Объединяем похожие концепты
        self._merge_similar_concepts()
    
    def _merge_similar_concepts(self):
        """Объединяет похожие концепты в графе знаний."""
        concepts = list(self.knowledge_graph.keys())
        merged = set()
        
        for i in range(len(concepts)):
            if concepts[i] in merged:
                continue
            
            for j in range(i + 1, len(concepts)):
                if concepts[j] in merged:
                    continue
                
                # Проверяем сходство концептов
                similarity = self._calculate_concept_similarity(concepts[i], concepts[j])
                if similarity > 0.8:
                    # Объединяем концепты
                    self._merge_concepts(concepts[i], concepts[j])
                    merged.add(concepts[j])
    
    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """
        Вычисляет сходство между концептами.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            
        Returns:
            float: Сходство (0.0-1.0)
        """
        # Упрощенное лексическое сходство
        set1 = set(re.findall(r'\w+', concept1.lower()))
        set2 = set(re.findall(r'\w+', concept2.lower()))
        
        if not set1 or not set2:
            return 0.0
        
        return len(set1 & set2) / len(set1 | set2)
    
    def _merge_concepts(self, target: str, source: str):
        """
        Объединяет два концепта в графе знаний.
        
        Args:
            target: Целевой концепт
            source: Исходный концепт
        """
        # Переносим связи
        if source in self.knowledge_graph:
            for connection in self.knowledge_graph[source]:
                self.knowledge_graph[target].append(connection)
            del self.knowledge_graph[source]
        
        # Обновляем факты
        for neuron in self.neurons.values():
            if neuron.content_type == "fact" and isinstance(neuron.content, dict):
                if neuron.content.get("subject") == source:
                    neuron.content["subject"] = target
                if neuron.content.get("object") == source:
                    neuron.content["object"] = target

class EpisodicMemory:
    """Эпизодическая долгосрочная память для хранения событий и опыта."""
    
    def __init__(self, capacity: int = 2000, retention_period: int = 31536000,  # 1 год
                 db: Optional[MemoryDatabase] = None, working_memory: Optional[WorkingMemory] = None):
        """
        Инициализирует эпизодическую память.
        
        Args:
            capacity: Емкость эпизодической памяти
            retention_period: Период хранения в секундах
            db: База данных для сохранения
            working_memory: Ссылка на рабочую память
        """
        self.capacity = capacity
        self.retention_period = retention_period
        self.db = db or MemoryDatabase()
        self.working_memory = working_memory
        
        # Основные структуры данных
        self.episodes: Dict[str, MemoryNeuron] = {}  # {episode_id: neuron}
        self.user_episodes: Dict[str, List[str]] = defaultdict(list)  # {user_id: [episode_id]}
        self.temporal_index = []  # [(timestamp, episode_id)]
        
        # Параметры работы
        self.running = False
        self.stop_event = threading.Event()
        self.last_cleanup = 0
        
        # Загружаем сохраненные данные
        self._load_from_db()
        
        logger.info(f"Эпизодическая память инициализирована (емкость: {capacity})")
    
    def _load_from_db(self):
        """Загружает данные эпизодической памяти из базы данных."""
        try:
            # Загружаем эпизоды
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT id FROM memory_neurons WHERE content_type = 'episode'")
            for (episode_id,) in cursor.fetchall():
                neuron = self.db.load_neuron(episode_id)
                if neuron:
                    self.episodes[episode_id] = neuron
                    self.temporal_index.append((neuron.timestamp, episode_id))
                    
                    # Индексируем по пользователю
                    user_id = neuron.metadata.get("user_id", "system")
                    self.user_episodes[user_id].append(episode_id)
            
            # Сортируем временной индекс
            self.temporal_index.sort(key=lambda x: x[0])
            
            logger.info(f"Загружено {len(self.episodes)} эпизодов в эпизодическую память")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных эпизодической памяти: {e}")
    
    def store_episode(self, content: Any, user_id: str = "system", 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Сохраняет эпизод в долгосрочную память.
        
        Args:
            content: Содержимое эпизода
            user_id: ID пользователя
            context: Контекст эпизода
            
        Returns:
            str: ID сохраненного эпизода
        """
        # Создаем уникальный ID
        timestamp = int(time.time() * 1000)
        content_hash = hash(str(content)) % 1000000
        episode_id = f"episode_{content_hash}_{timestamp}"
        
        # Создаем нейрон
        neuron = MemoryNeuron(
            id=episode_id,
            content=content,
            content_type="episode",
            metadata={
                "user_id": user_id,
                "context": context or {},
                "memory_type": "episodic"
            }
        )
        
        # Сохраняем в структуры данных
        self.episodes[episode_id] = neuron
        self.user_episodes[user_id].append(episode_id)
        self.temporal_index.append((neuron.timestamp, episode_id))
        self.temporal_index.sort(key=lambda x: x[0])
        
        # Сохраняем в базу данных
        self.db.save_neuron(neuron)
        
        return episode_id
    
    def retrieve_by_time(self, start_time: Optional[float] = None, 
                        end_time: Optional[float] = None, 
                        user_id: Optional[str] = None, 
                        top_k: int = 10) -> List[MemoryNeuron]:
        """
        Извлекает эпизоды по временному диапазону.
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
            user_id: ID пользователя
            top_k: Количество результатов
            
        Returns:
            List[MemoryNeuron]: Эпизоды в хронологическом порядке
        """
        start_time = start_time or (time.time() - 86400)  # Последние 24 часа по умолчанию
        end_time = end_time or time.time()
        
        # Фильтруем по времени
        episodes = [
            self.episodes[episode_id] for ts, episode_id in self.temporal_index
            if start_time <= ts <= end_time
        ]
        
        # Фильтруем по пользователю
        if user_id:
            episodes = [
                e for e in episodes 
                if e.metadata.get("user_id") == user_id
            ]
        
        return episodes[:top_k]
    
    def retrieve_by_similarity(self, query: Any, user_id: Optional[str] = None,
                             nlp_model=None, threshold: float = 0.6, 
                             top_k: int = 5) -> List[MemoryNeuron]:
        """
        Извлекает эпизоды по семантическому сходству.
        
        Args:
            query: Запрос
            user_id: ID пользователя
            nlp_model: NLP-модель
            threshold: Порог сходства
            top_k: Количество результатов
            
        Returns:
            List[MemoryNeuron]: Результаты поиска
        """
        candidates = []
        
        for episode in self.episodes.values():
            # Фильтруем по пользователю
            if user_id and episode.metadata.get("user_id") != user_id:
                continue
            
            # Вычисляем сходство
            similarity = episode.get_similarity(MemoryNeuron(
                id="query",
                content=query,
                content_type="text" if isinstance(query, str) else "fact"
            ), nlp_model)
            
            if similarity >= threshold:
                candidates.append((similarity, episode))
        
        # Сортируем по сходству
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [episode for _, episode in candidates[:top_k]]
    
    def get_user_history(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Возвращает историю взаимодействий пользователя.
        
        Args:
            user_id: ID пользователя
            days: Количество дней
            
        Returns:
            List[Dict]: История взаимодействий
        """
        cutoff_time = time.time() - (days * 86400)
        
        history = []
        for episode_id in self.user_episodes.get(user_id, []):
            episode = self.episodes.get(episode_id)
            if episode and episode.timestamp >= cutoff_time:
                history.append({
                    "timestamp": episode.timestamp,
                    "content": episode.content,
                    "context": episode.metadata.get("context", {}),
                    "importance": episode.importance
                })
        
        # Сортируем по времени
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return history
    
    def cleanup_old_episodes(self):
        """Удаляет устаревшие эпизоды."""
        current_time = time.time()
        cutoff_time = current_time - self.retention_period
        
        to_remove = [
            episode_id for episode_id, episode in self.episodes.items()
            if episode.timestamp < cutoff_time
        ]
        
        # Удаляем эпизоды
        for episode_id in to_remove:
            user_id = self.episodes[episode_id].metadata.get("user_id", "system")
            if user_id in self.user_episodes:
                self.user_episodes[user_id] = [
                    eid for eid in self.user_episodes[user_id] if eid != episode_id
                ]
                if not self.user_episodes[user_id]:
                    del self.user_episodes[user_id]
            
            if episode_id in dict(self.temporal_index):
                self.temporal_index = [
                    (ts, eid) for ts, eid in self.temporal_index if eid != episode_id
                ]
            
            del self.episodes[episode_id]
        
        # Сохраняем изменения
        logger.info(f"Удалено {len(to_remove)} устаревших эпизодов")
        self.last_cleanup = current_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику эпизодической памяти.
        
        Returns:
            Dict: Статистика
        """
        total_episodes = len(self.episodes)
        users_with_episodes = len(self.user_episodes)
        
        # Вычисляем среднюю длину истории на пользователя
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
        """Запускает фоновые процессы эпизодической памяти."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Запускаем поток очистки
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Фоновые процессы эпизодической памяти запущены")
    
    def _cleanup_worker(self):
        """Рабочий процесс для периодической очистки устаревших эпизодов."""
        while not self.stop_event.is_set():
            try:
                # Проверяем каждые 24 часа
                time.sleep(86400)
                self.cleanup_old_episodes()
            except Exception as e:
                logger.error(f"Ошибка в процессе очистки эпизодической памяти: {e}")
                time.sleep(3600)  # Подождать час перед повторной попыткой
    
    def stop(self):
        """Останавливает фоновые процессы эпизодической памяти."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        # Даем время на завершение
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        # Сохраняем данные
        for neuron in self.episodes.values():
            self.db.save_neuron(neuron)
        
        logger.info("Фоновые процессы эпизодической памяти остановлены")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Возвращает статус здоровья эпизодической памяти.
        
        Returns:
            Dict: Статус здоровья
        """
        stats = self.get_statistics()
        
        # Вычисляем оценку здоровья
        usage_score = 1.0 - abs(stats["usage"] - 0.5)  # Оптимально около 50%
        history_score = min(1.0, stats["avg_history_length"] / 10)  # Оптимально около 10 эпизодов на пользователя
        
        health_score = (
            usage_score * 0.6 +
            history_score * 0.4
        )
        
        # Определяем статус
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
        """
        Генерирует рекомендации на основе статуса здоровья.
        
        Args:
            stats: Статистика памяти
            
        Returns:
            List[str]: Рекомендации
        """
        recommendations = []
        
        # Проверяем использование
        if stats["usage"] > 0.9:
            recommendations.append(
                "Эпизодическая память переполнена. Рассмотрите увеличение емкости "
                "или сокращение периода хранения."
            )
        elif stats["usage"] < 0.2:
            recommendations.append(
                "Эпизодическая память недостаточно используется. Проверьте настройки "
                "периода хранения."
            )
        
        # Проверяем длину истории
        if stats["avg_history_length"] < 2:
            recommendations.append(
                "Слишком короткая история взаимодействий. Рассмотрите увеличение "
                "периода хранения эпизодов."
            )
        
        return recommendations
    
    def consolidate_from_working(self, user_id: str = "system"):
        """
        Консолидирует информацию из рабочей памяти в эпизодическую.
        
        Args:
            user_id: ID пользователя
        """
        if not self.working_memory:
            return
        
        # Ищем эпизоды в рабочей памяти
        episodes = self.working_memory.retrieve(
            {"content_type": "episode"},
            nlp_model=None,
            top_k=100
        )
        
        for episode in episodes:
            # Проверяем, не сохранен ли уже этот эпизод
            if episode.id not in self.episodes:
                # Создаем копию для эпизодической памяти
                episodic_neuron = MemoryNeuron(
                    id=f"episodic_{episode.id}",
                    content=episode.content,
                    content_type="episode",
                    strength=episode.strength,
                    importance=episode.importance,
                    timestamp=episode.timestamp,
                    metadata={
                        **episode.metadata,
                        "memory_type": "episodic",
                        "source_working_id": episode.id,
                        "user_id": user_id
                    }
                )
                
                # Сохраняем
                self.store_episode(
                    content=episodic_neuron.content,
                    user_id=user_id,
                    context=episodic_neuron.metadata.get("context")
                )