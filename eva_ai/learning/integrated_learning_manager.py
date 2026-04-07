"""
Интегрированный модуль самообучения для ЕВА с фрактальным хранилищем
"""
import os
import logging
import time
import threading
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import torch
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.integrated_learning")

@dataclass
class LearningConfig:
    """Конфигурация интегрированного обучения."""
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 5
    max_sequence_length: int = 512
    save_interval: int = 100  # Сохранять каждые N шагов
    use_fractal_storage: bool = True
    cache_embeddings: bool = True
    adaptive_batch_size: bool = True

class IntegratedLearningManager:
    """Интегрированный менеджер обучения с фрактальным хранилищем и гибридным кэшем."""
    
    def __init__(self, brain=None, config: Optional[LearningConfig] = None):
        self.brain = brain
        self.config = config or LearningConfig()
        
        # Компоненты
        self.fractal_model_manager = None
        self.hybrid_cache = None
        self.knowledge_graph = None
        self.training_orchestrator = None
        
        # Состояние обучения
        self.is_learning = False
        self.learning_thread = None
        self.learning_stats = {
            "total_steps": 0,
            "saved_checkpoints": 0,
            "last_checkpoint": None,
            "learning_start_time": None,
            "total_loss": 0.0,
            "accuracy": 0.0
        }
        
        # Очередь обучения
        self.learning_queue = []
        self.queue_lock = threading.Lock()
        
        logger.info("IntegratedLearningManager инициализирован")
    
    def initialize(self) -> bool:
        """Инициализирует компоненты обучения."""
        try:
            # Получаем компоненты из brain
            if self.brain:
                # Пробуем получить model_manager (новый RuGPT3ModelManager)
                self.fractal_model_manager = getattr(self.brain, 'model_manager', None)
                # Если нет, пробуем старый fractal_model_manager
                if not self.fractal_model_manager:
                    self.fractal_model_manager = getattr(self.brain, 'fractal_model_manager', None)
                
                self.knowledge_graph = getattr(self.brain, 'knowledge_graph', None)
                self.training_orchestrator = getattr(self.brain, 'training_orchestrator', None)
                
                # Ищем гибридный кэш (необязательный компонент)
                # Сначала пробуем через brain.components
                if hasattr(self.brain, 'components') and 'hybrid_cache' in self.brain.components:
                    self.hybrid_cache = self.brain.components['hybrid_cache']
                    logger.debug("Гибридный кэш найден через brain.components")
                
                # Потом пробуем через memory_manager
                if not self.hybrid_cache and hasattr(self.brain, 'memory_manager'):
                    self.hybrid_cache = getattr(self.brain.memory_manager, 'hybrid_cache', None)
                    if self.hybrid_cache:
                        logger.debug("Гибридный кэш найден через memory_manager")
                
                # Потом пробуем через ml_unit
                if not self.hybrid_cache and hasattr(self.brain, 'ml_unit'):
                    ml_unit = self.brain.ml_unit
                    self.hybrid_cache = getattr(ml_unit, 'hybrid_cache', None)
                    if self.hybrid_cache:
                        logger.debug("Гибридный кэш найден через ml_unit")
                
                # Дополнительные источники гибридного кэша
                if not self.hybrid_cache and hasattr(self.brain, 'component_initializer'):
                    initializer = self.brain.component_initializer
                    if hasattr(initializer, 'get_component'):
                        self.hybrid_cache = initializer.get_component('hybrid_cache')
                        if self.hybrid_cache:
                            logger.debug("Гибридный кэш найден через component_initializer")
            
            # Проверяем доступность основных компонентов
            core_components_ready = bool(
                self.fractal_model_manager and 
                self.knowledge_graph
            )
            
            # Гибридный кэш - необязательный компонент
            if self.hybrid_cache:
                logger.info("Гибридный кэш доступен для обучения")
                self.config.cache_embeddings = True
            else:
                logger.warning("Гибридный кэш недоступен, кэширование эмбеддингов отключено")
                self.config.cache_embeddings = False
            
            if core_components_ready:
                logger.info("Основные компоненты обучения доступны")
                return True
            else:
                logger.warning("Некоторые основные компоненты обучения недоступны")
                logger.info(f"  model_manager: {self.fractal_model_manager is not None}")
                logger.info(f"  knowledge_graph: {self.knowledge_graph is not None}")
                logger.info(f"  hybrid_cache: {self.hybrid_cache is not None} (необязательный)")
                
                # Работаем с доступными компонентами
                if self.fractal_model_manager or self.knowledge_graph:
                    logger.info("Работаем с доступными компонентами")
                    return True
                else:
                    logger.error("Недостаточно компонентов для обучения")
                    return False
                
        except Exception as e:
            logger.error(f"Ошибка инициализации IntegratedLearningManager: {e}")
            return False
    
    def add_learning_task(self, task_data: Dict[str, Any]) -> bool:
        """Добавляет задачу в очередь обучения."""
        try:
            with self.queue_lock:
                task = {
                    "id": f"task_{int(time.time())}_{len(self.learning_queue)}",
                    "data": task_data,
                    "timestamp": time.time(),
                    "status": "pending",
                    "priority": task_data.get("priority", 0.5)
                }
                self.learning_queue.append(task)
                logger.info(f"Добавлена задача обучения: {task['id']}")
                return True
        except Exception as e:
            logger.error(f"Ошибка добавления задачи обучения: {e}")
            return False
    
    def start_learning(self) -> bool:
        """Запускает процесс обучения."""
        if self.is_learning:
            logger.warning("Обучение уже запущено")
            return False
        
        if not self.initialize():
            logger.error("Не удалось инициализировать компоненты обучения")
            return False
        
        try:
            self.learning_thread = threading.Thread(
                target=self._learning_worker,
                daemon=True
            )
            self.learning_thread.start()
            self.is_learning = True
            self.learning_stats["learning_start_time"] = time.time()
            logger.info("Процесс обучения запущен")
            return True
        except Exception as e:
            logger.error(f"Ошибка запуска обучения: {e}")
            return False
    
    def stop_learning(self) -> bool:
        """Останавливает процесс обучения."""
        if not self.is_learning:
            return False
        
        self.is_learning = False
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5.0)
        
        logger.info("Процесс обучения остановлен")
        return True
    
    def _learning_worker(self):
        """Рабочий метод для обучения в отдельном потоке."""
        try:
            logger.info("Начало рабочего цикла обучения")
            
            while self.is_learning:
                # Получаем следующую задачу
                task = self._get_next_task()
                if not task:
                    time.sleep(1.0)
                    continue
                
                # Обрабатываем задачу
                self._process_learning_task(task)
                
        except Exception as e:
            logger.error(f"Ошибка в рабочем цикле обучения: {e}")
        finally:
            self.is_learning = False
            logger.info("Рабочий цикл обучения завершен")
    
    def _get_next_task(self) -> Optional[Dict[str, Any]]:
        """Получает следующую задачу из очереди."""
        try:
            with self.queue_lock:
                if not self.learning_queue:
                    return None
                
                # Сортируем по приоритету
                self.learning_queue.sort(key=lambda x: x["priority"], reverse=True)
                task = self.learning_queue.pop(0)
                task["status"] = "processing"
                return task
        except Exception as e:
            logger.error(f"Ошибка получения задачи: {e}")
            return None
    
    def _process_learning_task(self, task: Dict[str, Any]):
        """Обрабатывает задачу обучения."""
        try:
            task_id = task["id"]
            task_data = task["data"]
            
            logger.info(f"Обработка задачи {task_id}")
            
            # Определяем тип задачи
            task_type = task_data.get("type", "unknown")
            
            if task_type == "document_training":
                self._train_from_document(task_data)
            elif task_type == "knowledge_graph_training":
                self._train_knowledge_graph(task_data)
            elif task_type == "model_fine_tuning":
                self._fine_tune_model(task_data)
            else:
                logger.warning(f"Неизвестный тип задачи: {task_type}")
            
            # Обновляем статистику
            task["status"] = "completed"
            task["completed_time"] = time.time()
            
        except Exception as e:
            logger.error(f"Ошибка обработки задачи {task.get('id', 'unknown')}: {e}")
            task["status"] = "error"
            task["error"] = str(e)
    
    def _train_from_document(self, task_data: Dict[str, Any]):
        """Обучает модель на документе."""
        try:
            document = task_data.get("document")
            model_id = task_data.get("model_id", "text-generation")
            
            if not document:
                logger.warning("Документ не предоставлен для обучения")
                return
            
            # Обучение через самодиалог
            if self.training_orchestrator:
                result = self.training_orchestrator.train_from_document(
                    imported_doc=document,
                    model_id=model_id,
                    use_fractal=self.config.use_fractal_storage
                )
                logger.info(f"Обучение завершено: {result.get('status', 'unknown')}")
            else:
                # Fallback - простое обучение
                self._simple_document_training(document, model_id)
                
        except Exception as e:
            logger.error(f"Ошибка обучения на документе: {e}")
    
    def _train_knowledge_graph(self, task_data: Dict[str, Any]):
        """Обучает на графе знаний."""
        try:
            if not self.knowledge_graph:
                logger.warning("KnowledgeGraph недоступен")
                return
            
            # Получаем узлы и связи из графа
            nodes = self.knowledge_graph.get_all_nodes()
            edges = self.knowledge_graph.get_all_edges()
            
            if not nodes:
                logger.warning("Граф знаний пуст")
                return
            
            # Создаем обучающие данные
            training_data = self._create_graph_training_data(nodes, edges)
            
            # Сохраняем во фрактальное хранилище
            if self.config.use_fractal_storage and self.fractal_model_manager:
                self._save_training_data(training_data, "knowledge_graph")
            
            logger.info(f"Обучение на графе знаний: {len(nodes)} узлов, {len(edges)} связей")
            
        except Exception as e:
            logger.error(f"Ошибка обучения на графе знаний: {e}")
    
    def _fine_tune_model(self, task_data: Dict[str, Any]):
        """Дообучает модель."""
        try:
            if not self.fractal_model_manager:
                logger.warning("FractalModelManager недоступен")
                return
            
            training_data = task_data.get("training_data")
            model_id = task_data.get("model_id", "text-generation")
            
            if not training_data:
                logger.warning("Обучающие данные не предоставлены")
                return
            
            # Кэшируем эмбеддинги если включено
            if self.config.cache_embeddings:
                if not self.hybrid_cache:
                    # Пробуем найти динамически
                    self._find_hybrid_cache_dynamically()
                
                if self.hybrid_cache:
                    training_data = self._cache_embeddings(training_data)
                else:
                    logger.warning("Кэширование эмбеддингов не удалось - гибридный кэш недоступен")
            
            # Сохраняем checkpoint
            self._save_checkpoint(model_id, training_data)
            
            logger.info(f"Дообучение модели {model_id} завершено")
            
        except Exception as e:
            logger.error(f"Ошибка дообучения модели: {e}")
    
    def _simple_document_training(self, document, model_id: str):
        """Простое обучение на документе (fallback)."""
        try:
            # Извлекаем текст из документа
            text_content = ""
            if hasattr(document, 'iter_segments'):
                for segment in document.iter_segments():
                    text_content += str(segment) + " "
            else:
                text_content = str(document)
            
            # Кэшируем текст
            if self.hybrid_cache:
                cache_key = f"training_doc_{hash(text_content) % 1000000}"
                self.hybrid_cache.set(cache_key, {
                    "text": text_content,
                    "model_id": model_id,
                    "timestamp": time.time()
                })
            
            # Сохраняем во фрактальное хранилище
            if self.config.use_fractal_storage and self.fractal_model_manager:
                self._save_training_data({"text": text_content}, f"document_{model_id}")
            
            logger.info(f"Простое обучение завершено: {len(text_content)} символов")
            
        except Exception as e:
            logger.error(f"Ошибка простого обучения: {e}")
    
    def _create_graph_training_data(self, nodes: List, edges: List) -> Dict[str, Any]:
        """Создает обучающие данные из графа."""
        try:
            # Извлекаем текст из узлов
            node_texts = []
            for node in nodes:
                if hasattr(node, 'content'):
                    node_texts.append(str(node.content))
                elif hasattr(node, 'text'):
                    node_texts.append(str(node.text))
                else:
                    node_texts.append(str(node))
            
            # Извлекаем связи
            edge_data = []
            for edge in edges:
                edge_info = {
                    "source": str(edge.get('source', '')),
                    "target": str(edge.get('target', '')),
                    "type": str(edge.get('type', 'unknown'))
                }
                edge_data.append(edge_info)
            
            return {
                "nodes": node_texts,
                "edges": edge_data,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка создания обучающих данных из графа: {e}")
            return {}
    
    def _cache_embeddings(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Кэширует эмбеддинги в гибридном кэше."""
        try:
            if not self.hybrid_cache:
                return training_data
            
            # Кэшируем текстовые данные
            for key, value in training_data.items():
                if isinstance(value, str):
                    cache_key = f"embed_{hash(value) % 1000000}"
                    self.hybrid_cache.set(cache_key, {
                        "text": value,
                        "cached_at": time.time()
                    })
            
            return training_data
            
        except Exception as e:
            logger.error(f"Ошибка кэширования эмбеддингов: {e}")
            return training_data
    
    def _save_training_data(self, data: Dict[str, Any], data_type: str):
        """Сохраняет обучающие данные во фрактальное хранилище."""
        try:
            if not self.fractal_model_manager:
                return
            
            # Создаем имя файла
            timestamp = int(time.time())
            filename = f"training_{data_type}_{timestamp}.json"
            
            # Сохраняем данные
            cache_dir = getattr(self.fractal_model_manager, 'cache_dir', './cache')
            training_dir = os.path.join(cache_dir, 'training_data')
            os.makedirs(training_dir, exist_ok=True)
            
            filepath = os.path.join(training_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Обучающие данные сохранены: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения обучающих данных: {e}")
    
    def _save_checkpoint(self, model_id: str, training_data: Dict[str, Any]):
        """Сохраняет checkpoint обучения."""
        try:
            checkpoint = {
                "model_id": model_id,
                "timestamp": time.time(),
                "training_stats": self.learning_stats,
                "config": {
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "epochs": self.config.epochs
                },
                "data_sample": training_data
            }
            
            # Сохраняем checkpoint
            self._save_training_data(checkpoint, f"checkpoint_{model_id}")
            
            # Обновляем статистику
            self.learning_stats["saved_checkpoints"] += 1
            self.learning_stats["last_checkpoint"] = time.time()
            
            logger.info(f"Checkpoint сохранен для модели {model_id}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения checkpoint: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Возвращает статистику обучения."""
        stats = self.learning_stats.copy()
        
        # Добавляем информацию о компонентах
        stats["components"] = {
            "fractal_model_manager": self.fractal_model_manager is not None,
            "hybrid_cache": self.hybrid_cache is not None,
            "knowledge_graph": self.knowledge_graph is not None,
            "training_orchestrator": self.training_orchestrator is not None
        }
        
        # Добавляем информацию об очереди
        with self.queue_lock:
            stats["queue_length"] = len(self.learning_queue)
            stats["pending_tasks"] = len([t for t in self.learning_queue if t["status"] == "pending"])
            stats["processing_tasks"] = len([t for t in self.learning_queue if t["status"] == "processing"])
        
        # Добавляем время работы
        if stats["learning_start_time"]:
            stats["learning_duration"] = time.time() - stats["learning_start_time"]
        
        return stats
    
    def add_document_for_training(self, document, model_id: str = "text-generation", priority: float = 0.5) -> bool:
        """Добавляет документ для обучения."""
        task_data = {
            "type": "document_training",
            "document": document,
            "model_id": model_id,
            "priority": priority
        }
        return self.add_learning_task(task_data)
    
    def add_knowledge_graph_training(self, priority: float = 0.7) -> bool:
        """Добавляет задачу обучения на графе знаний."""
        task_data = {
            "type": "knowledge_graph_training",
            "priority": priority
        }
        return self.add_learning_task(task_data)
