"""Модуль рабочей (краткосрочной) памяти ЕВА"""
import os
import logging
import time
import threading
import queue
import json
import re
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable, Deque
from dataclasses import dataclass, field
import numpy as np
from collections import deque, defaultdict

from .memory_core import MemoryNeuron, MemoryField, MemoryDatabase

logger = logging.getLogger("eva.memory.working")

class WorkingMemory:
    """Рабочая (краткосрочная) память для временного хранения информации."""
    
    def __init__(self, capacity: int = 1000, decay_rate: float = 0.95, 
                 consolidation_threshold: float = 0.7, db: Optional[MemoryDatabase] = None):
        """
        Инициализирует рабочую память.
        
        Args:
            capacity: Емкость рабочей памяти
            decay_rate: Коэффициент затухания информации
            consolidation_threshold: Порог для консолидации в долгосрочную память
            db: База данных для сохранения
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        self.db = db or MemoryDatabase()
        
        # Основные структуры данных
        self.neurons: Dict[str, MemoryNeuron] = {}
        self.fields: Dict[str, MemoryField] = {}
        self.priority_queue = []
        
        # Thread lock for shared state protection
        self._lock = threading.RLock()
        
        # Параметры работы
        self.running = False
        self.stop_event = threading.Event()
        
        # Загружаем сохраненные данные
        self._load_from_db()
        
        logger.info(f"Рабочая память инициализирована (емкость: {capacity})")
    
    def _load_from_db(self):
        """Загружает данные рабочей памяти из базы данных."""
        try:
            # Загружаем нейроны
            for field_name in self.db.conn.execute("SELECT name FROM memory_fields"):
                field = self.db.load_field(field_name[0])
                if field:
                    self.fields[field.name] = field
            
            # Загружаем нейроны
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT id FROM memory_neurons")
            for (neuron_id,) in cursor.fetchall():
                neuron = self.db.load_neuron(neuron_id)
                if neuron:
                    self.neurons[neuron_id] = neuron
            
            logger.info(f"Загружено {len(self.neurons)} нейронов в рабочую память")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных рабочей памяти: {e}")
    
    def store(self, content: Any, content_type: str = "text", 
             field_name: str = "general", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Сохраняет информацию в рабочую память.
        
        Args:
            content: Содержимое для сохранения
            content_type: Тип содержимого
            field_name: Имя поля памяти
            metadata: Дополнительные метаданные
            
        Returns:
            str: ID сохраненного нейрона
        """
        with self._lock:
            # Создаем уникальный ID
            timestamp = int(time.time() * 1000)
            content_hash = hash(str(content)) % 1000000
            neuron_id = f"working_{content_hash}_{timestamp}"
            
            # Проверяем или создаем поле памяти
            if field_name not in self.fields:
                self.fields[field_name] = MemoryField(
                    name=field_name,
                    description=f"Поле рабочей памяти: {field_name}",
                    capacity=self.capacity // 5
                )
            
            field = self.fields[field_name]
            
            # Проверяем емкость
            if field.current_size >= field.capacity:
                # Удаляем наименее важные нейроны
                self._evict_least_important(field_name)
            
            # Создаем нейрон
            neuron = MemoryNeuron(
                id=neuron_id,
                content=content,
                content_type=content_type,
                metadata=metadata or {},
                strength=1.0
            )
            
            # Сохраняем в структуры данных
            self.neurons[neuron_id] = neuron
            field.current_size += 1
            field.update_access("system")
            
            # Сохраняем в базу данных
            self.db.save_neuron(neuron)
            self.db.save_field(field)
            
            return neuron_id
    
    def _evict_least_important(self, field_name: str):
        """Удаляет наименее важные нейроны из поля памяти."""
        field = self.fields[field_name]
        neurons_in_field = [
            (nid, neuron) for nid, neuron in self.neurons.items()
            if neuron.metadata.get("field") == field_name
        ]
        
        # Сортируем по важности и времени последнего доступа
        neurons_in_field.sort(key=lambda x: (
            x[1].importance * 0.7 + 
            (time.time() - x[1].last_accessed) / 86400 * 0.3
        ))
        
        # Удаляем 10% наименее важных
        evict_count = max(1, len(neurons_in_field) // 10)
        for i in range(evict_count):
            nid, _ = neurons_in_field[i]
            del self.neurons[nid]
            field.current_size -= 1
        
        self.db.save_field(field)
    
    def retrieve(self, query: Any, field_name: Optional[str] = None, 
                nlp_model=None, top_k: int = 5) -> List[MemoryNeuron]:
        """
        Извлекает информацию из рабочей памяти на основе запроса.
        
        Args:
            query: Запрос для поиска
            field_name: Имя поля для поиска (если None, ищет во всех полях)
            nlp_model: NLP-модель для семантического поиска
            top_k: Количество результатов
            
        Returns:
            List[MemoryNeuron]: Топ-K наиболее релевантных нейронов
        """
        with self._lock:
            candidates = []
            
            # Фильтруем по полю, если указано
            for neuron in self.neurons.values():
                if field_name and neuron.metadata.get("field") != field_name:
                    continue
                
                # Вычисляем релевантность
                similarity = self._calculate_similarity(query, neuron, nlp_model)
                if similarity > 0.2:  # Порог минимальной релевантности
                    candidates.append((similarity, neuron))
            
            # Сортируем по релевантности
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Возвращаем топ-K результатов
            return [neuron for _, neuron in candidates[:top_k]]
    
    def _calculate_similarity(self, query: Any, neuron: MemoryNeuron, nlp_model) -> float:
        """
        Вычисляет релевантность запроса к нейрону.
        
        Args:
            query: Запрос
            neuron: Нейрон для сравнения
            nlp_model: NLP-модель
            
        Returns:
            float: Степень релевантности (0.0-1.0)
        """
        # Для текстовых запросов используем семантическое сходство
        if isinstance(query, str) and neuron.content_type == "text" and nlp_model:
            return neuron.get_similarity(MemoryNeuron(
                id="query",
                content=query,
                content_type="text"
            ), nlp_model)
        
        # Для фактов сравниваем ключевые поля
        if isinstance(query, dict) and isinstance(neuron.content, dict):
            common_keys = set(query.keys()) & set(neuron.content.keys())
            if not common_keys:
                return 0.0
            
            # Вычисляем сходство по общим полям
            similarities = []
            for key in common_keys:
                val1 = query[key]
                val2 = neuron.content[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Для числовых значений вычисляем относительное сходство
                    max_val = max(abs(val1), abs(val2), 1)
                    similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
                elif isinstance(val1, str) and isinstance(val2, str):
                    # Для строк используем лексическое сходство
                    if val1 == val2:
                        similarities.append(1.0)
                    else:
                        # Упрощенное лексическое сходство
                        set1 = set(re.findall(r'\w+', val1.lower()))
                        set2 = set(re.findall(r'\w+', val2.lower()))
                        if not set1 or not set2:
                            similarities.append(0.0)
                        else:
                            jaccard = len(set1 & set2) / len(set1 | set2)
                            similarities.append(jaccard)
                else:
                    similarities.append(1.0 if val1 == val2 else 0.0)
            
            return np.mean(similarities) if similarities else 0.0
        
        # Для других типов используем простое сравнение
        return 1.0 if query == neuron.content else 0.0
    
    def update_importance(self, neuron_id: str, delta: float):
        """
        Обновляет важность нейрона.
        
        Args:
            neuron_id: ID нейрона
            delta: Изменение важности
        """
        if neuron_id in self.neurons:
            neuron = self.neurons[neuron_id]
            neuron.importance = max(0.0, min(1.0, neuron.importance + delta))
            self.db.save_neuron(neuron)
    
    def decay_memory(self):
        """Применяет затухание ко всем нейронам в рабочей памяти."""
        with self._lock:
            for neuron in self.neurons.values():
                neuron.update_strength(-0.1 * (1.0 - neuron.importance))
                if neuron.strength <= 0.1:
                    # Помечаем на удаление
                    if neuron.metadata:
                        neuron.metadata["to_remove"] = True
                    else:
                        neuron.metadata = {"to_remove": True}
            
            # Удаляем слабые нейроны
            to_remove = [nid for nid, neuron in self.neurons.items() 
                        if neuron.metadata.get("to_remove")]
            for nid in to_remove:
                field_name = self.neurons[nid].metadata.get("field", "general")
                if field_name in self.fields:
                    self.fields[field_name].current_size -= 1
                del self.neurons[nid]
            
            # Сохраняем изменения
            for field in self.fields.values():
                self.db.save_field(field)
    
    def get_consolidation_candidates(self) -> List[MemoryNeuron]:
        """
        Возвращает кандидатов для консолидации в долгосрочную память.
        
        Returns:
            List[MemoryNeuron]: Нейроны, готовые к консолидации
        """
        candidates = []
        for neuron in self.neurons.values():
            # Нейроны с высокой важностью и силой
            if neuron.importance >= self.consolidation_threshold and \
               neuron.strength >= self.consolidation_threshold:
                candidates.append(neuron)
        return candidates
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику рабочей памяти.
        
        Returns:
            Dict: Статистика
        """
        total_neurons = len(self.neurons)
        total_capacity = sum(field.capacity for field in self.fields.values())
        used_capacity = sum(field.current_size for field in self.fields.values())
        
        # Вычисляем среднюю важность
        avg_importance = np.mean([n.importance for n in self.neurons.values()]) if self.neurons else 0.0
        
        # Вычисляем среднюю силу
        avg_strength = np.mean([n.strength for n in self.neurons.values()]) if self.neurons else 0.0
        
        return {
            "total_neurons": total_neurons,
            "total_capacity": total_capacity,
            "used_capacity": used_capacity,
            "usage": min(1.0, used_capacity / total_capacity) if total_capacity > 0 else 1.0,
            "avg_importance": avg_importance,
            "avg_strength": avg_strength,
            "fields": {name: field.get_usage_stats() for name, field in self.fields.items()}
        }
    
    def start(self):
        """Запускает фоновые процессы рабочей памяти."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Запускаем поток затухания
        self.decay_thread = threading.Thread(target=self._decay_worker, daemon=True)
        self.decay_thread.start()
        
        logger.info("Фоновые процессы рабочей памяти запущены")
    
    def _decay_worker(self):
        """Рабочий процесс для периодического затухания памяти."""
        while not self.stop_event.is_set():
            try:
                # Затухание каждые 5 минут
                time.sleep(300)
                # Call decay_memory which already has locking
                self.decay_memory()
            except Exception as e:
                logger.error(f"Ошибка в процессе затухания памяти: {e}")
                time.sleep(60)  # Подождать перед повторной попыткой
    
    def stop(self):
        """Останавливает фоновые процессы рабочей памяти."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        # Даем время на завершение
        if hasattr(self, 'decay_thread') and self.decay_thread.is_alive():
            self.decay_thread.join(timeout=5.0)
        
        # Сохраняем данные
        for neuron in self.neurons.values():
            self.db.save_neuron(neuron)
        for field in self.fields.values():
            self.db.save_field(field)
        
        logger.info("Фоновые процессы рабочей памяти остановлены")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Возвращает статус здоровья рабочей памяти.
        
        Returns:
            Dict: Статус здоровья
        """
        stats = self.get_statistics()
        
        # Вычисляем оценку здоровья
        usage_score = 1.0 - abs(stats["usage"] - 0.7)  # Оптимально около 70%
        importance_score = min(1.0, stats["avg_importance"] * 1.5)
        strength_score = min(1.0, stats["avg_strength"] * 1.5)
        
        health_score = (
            usage_score * 0.4 +
            importance_score * 0.3 +
            strength_score * 0.3
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
            "avg_strength": stats["avg_strength"],
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
                "Рабочая память переполнена. Рассмотрите увеличение емкости или "
                "более частую консолидацию."
            )
        elif stats["usage"] < 0.3:
            recommendations.append(
                "Рабочая память недостаточно используется. Проверьте настройки "
                "порога консолидации."
            )
        
        # Проверяем среднюю важность
        if stats["avg_importance"] < 0.3:
            recommendations.append(
                "Низкая средняя важность информации. Проверьте критерии отбора "
                "информации для хранения."
            )
        
        # Проверяем среднюю силу
        if stats["avg_strength"] < 0.4:
            recommendations.append(
                "Низкая средняя сила связей. Возможно, требуется корректировка "
                "процесса затухания."
            )
        
        return recommendations