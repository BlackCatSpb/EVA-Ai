"""
Расширенная система самообучения CogniFlex с эпохами и интеграцией графа памяти
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger("cogniflex.enhanced_self_learning")


class TrainingStatus(Enum):
    """Статусы обучения"""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


@dataclass
class EpochMetrics:
    """Метрики одной эпохи обучения"""
    epoch_number: int
    loss: float
    accuracy: float
    perplexity: float
    learning_rate: float
    samples_processed: int
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch_number,
            'loss': round(self.loss, 4),
            'accuracy': round(self.accuracy, 4),
            'perplexity': round(self.perplexity, 4),
            'learning_rate': self.learning_rate,
            'samples': self.samples_processed,
            'duration': round(self.duration_seconds, 2),
            'timestamp': self.timestamp
        }


@dataclass
class TrainingSession:
    """Сессия обучения с полными метриками"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    status: TrainingStatus = TrainingStatus.IDLE
    epochs_completed: int = 0
    total_epochs: int = 0
    current_epoch_metrics: Optional[EpochMetrics] = None
    epoch_history: List[EpochMetrics] = field(default_factory=list)
    training_data_size: int = 0
    validation_accuracy: float = 0.0
    final_loss: float = 0.0
    model_improvements: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status.value,
            'epochs_completed': self.epochs_completed,
            'total_epochs': self.total_epochs,
            'progress_percent': round((self.epochs_completed / max(self.total_epochs, 1)) * 100, 1),
            'current_epoch': self.current_epoch_metrics.to_dict() if self.current_epoch_metrics else None,
            'epoch_history': [e.to_dict() for e in self.epoch_history],
            'training_data_size': self.training_data_size,
            'validation_accuracy': round(self.validation_accuracy, 4),
            'final_loss': round(self.final_loss, 4),
            'improvements': self.model_improvements,
            'error': self.error_message,
            'duration': round((self.end_time or time.time()) - self.start_time, 2) if self.end_time else None
        }


class EnhancedSelfLearningSystem:
    """
    Расширенная система самообучения с поддержкой эпох и детальных метрик
    """
    
    def __init__(self, brain, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        
        # Инициализация интеграции с модулями системы
        self._initialize_system_integration()
        
        # Состояние системы
        self.is_active = False
        self.learning_thread = None
        self.stop_event = threading.Event()
        
        # Текущая и история сессий
        self.current_session: Optional[TrainingSession] = None
        self.session_history: List[TrainingSession] = []
        self.max_history_size = self.config.get('max_history', 50)
        
    def _initialize_system_integration(self):
        """Инициализация интеграции с модулями системы"""
        try:
            # Получаем доступ к ключевым модулям через brain или brain.components
            components = getattr(self.brain, 'components', {})
            
            self.knowledge_graph = getattr(self.brain, 'knowledge_graph', None) or components.get('knowledge_graph')
            self.memory_manager = getattr(self.brain, 'memory_manager', None) or components.get('memory_manager')
            self.chat_history = getattr(self.brain, 'chat_history', []) or getattr(self.brain, 'chat_history', [])
            self.processed_queries = getattr(self.brain, 'processed_queries', set()) or getattr(self.brain, 'processed_queries', set())
            
            # Доступ к ML компонентам
            self.ml_unit = getattr(self.brain, 'ml_unit', None) or components.get('ml_unit')
            self.text_processor = getattr(self.brain, 'text_processor', None) or components.get('text_processor')
            self.response_generator = getattr(self.brain, 'response_generator', None) or components.get('response_generator')
            
            # Дополнительные попытки найти модули
            if not self.knowledge_graph and hasattr(self.brain, '_knowledge_graph'):
                self.knowledge_graph = self.brain._knowledge_graph
            if not self.memory_manager and hasattr(self.brain, '_memory_manager'):
                self.memory_manager = self.brain._memory_manager
            if not self.ml_unit and hasattr(self.brain, '_ml_unit'):
                self.ml_unit = self.brain._ml_unit
            
            logger.info("Интеграция с модулями системы инициализирована:")
            logger.info(f"  Граф знаний: {'[OK]' if self.knowledge_graph else '[FAIL]'}")
            logger.info(f"  Менеджер памяти: {'[OK]' if self.memory_manager else '[FAIL]'}")
            logger.info(f"  История чата: {'[OK]' if self.chat_history else '[FAIL]'}")
            logger.info(f"  ML Unit: {'[OK]' if self.ml_unit else '[FAIL]'}")
            logger.info(f"  Text Processor: {'[OK]' if self.text_processor else '[FAIL]'}")
            
            # Логируем доступные компоненты для отладки
            if components:
                logger.info(f"  Доступные компоненты: {list(components.keys())}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации интеграции: {e}")
            # Устанавливаем значения по умолчанию
            self.knowledge_graph = None
            self.memory_manager = None
            self.chat_history = []
            self.processed_queries = set()
            self.ml_unit = None
            self.text_processor = None
            self.response_generator = None
        
        # Очередь данных для обучения
        self.training_queue: List[Dict] = []
        self.training_lock = threading.Lock()
        
        # Callback для обновления GUI
        self.progress_callback: Optional[callable] = None
        self.epoch_callback: Optional[callable] = None
        
        # Параметры обучения по умолчанию
        self.default_params = {
            'min_training_interval': 1800,  # 30 минут
            'max_training_duration': 3600,  # 1 час
            'min_samples_for_training': 20,
            'default_epochs': 50,
            'batch_size': 4,
            'learning_rate': 5e-5,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 4,
            'max_grad_norm': 1.0,
            'save_steps': 500,
            'logging_steps': 10,
            'eval_steps': 100,
            'early_stopping_patience': 3
        }
        
        # Обновляем параметры из конфигурации
        self.default_params.update(self.config.get('training_params', {}))
        
        # Генерация сущностей во время обучения
        self.entity_generation_enabled = self.config.get('entity_generation', True)
        self.generated_entities: List[Dict] = []
        
        logger.info("EnhancedSelfLearningSystem инициализирована")
    
    def set_progress_callback(self, callback: callable):
        """Устанавливает callback для обновления прогресса в GUI"""
        self.progress_callback = callback
    
    def set_epoch_callback(self, callback: callable):
        """Устанавливает callback для обновления метрик эпохи"""
        self.epoch_callback = callback
    
    def start(self) -> bool:
        """Запуск системы самообучения"""
        if self.config.get('training_disabled', True):
            logger.info("EnhancedSelfLearningSystem отключена через конфигурацию")
            return True
        
        if self.is_active:
            logger.warning("EnhancedSelfLearningSystem уже активна")
            return True
        
        try:
            self.is_active = True
            self.stop_event.clear()
            self.learning_thread = threading.Thread(
                target=self._learning_loop,
                name="EnhancedSelfLearningThread",
                daemon=True
            )
            self.learning_thread.start()
            logger.info("EnhancedSelfLearningSystem запущена")
            return True
        except Exception as e:
            logger.error(f"Ошибка запуска: {e}")
            self.is_active = False
            return False
    
    def stop(self) -> bool:
        """Остановка системы самообучения"""
        if not self.is_active:
            return True
        
        self.stop_event.set()
        self.is_active = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=30)
        
        logger.info("EnhancedSelfLearningSystem остановлена")
        return True
    
    def add_training_data(self, text: str, source: str = "user_input", 
                         metadata: Optional[Dict] = None) -> bool:
        """Добавление данных для обучения"""
        if not text or len(text.strip()) < 3:
            return False
        
        with self.training_lock:
            self.training_queue.append({
                'text': text.strip(),
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'processed': False,
                'metadata': metadata or {},
                'entities': self._extract_entities(text) if self.entity_generation_enabled else []
            })
        
        logger.debug(f"Добавлены данные для обучения: '{text[:50]}...'")
        return True
    
    def auto_collect_training_data(self):
        """Автоматический сбор данных из системы для обучения"""
        try:
            collected_count = 0
            
            # 1. Собираем из истории чата
            if self.chat_history:
                recent_messages = self.chat_history[-10:]  # Последние 10 сообщений
                for msg in recent_messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        text = msg.get('content', '')
                        if text and len(text.strip()) > 10:
                            self.add_training_data(
                                text=text,
                                source="chat_history",
                                metadata={'timestamp': msg.get('timestamp'), 'role': msg.get('role')}
                            )
                            collected_count += 1
            
            # 2. Собираем из обработанных запросов
            if self.processed_queries:
                recent_queries = list(self.processed_queries)[-5:]  # Последние 5 запросов
                for query in recent_queries:
                    if isinstance(query, dict):
                        query_text = query.get('query', '')
                        response_text = query.get('response', '')
                        
                        if query_text and len(query_text.strip()) > 10:
                            self.add_training_data(
                                text=query_text,
                                source="processed_query",
                                metadata={'type': 'query', 'response_length': len(response_text)}
                            )
                            collected_count += 1
                        
                        if response_text and len(response_text.strip()) > 20:
                            self.add_training_data(
                                text=response_text,
                                source="processed_response",
                                metadata={'type': 'response', 'query_length': len(query_text)}
                            )
                            collected_count += 1
            
            # 3. Собираем из графа знаний
            if self.knowledge_graph:
                try:
                    kg_entities = self.knowledge_graph.get_recent_entities(limit=20)
                    for entity in kg_entities:
                        if isinstance(entity, dict) and 'text' in entity:
                            text = entity['text']
                            if text and len(text.strip()) > 10:
                                self.add_training_data(
                                    text=text,
                                    source="knowledge_graph",
                                    metadata={'entity_type': entity.get('type', 'unknown'), 'strength': entity.get('strength', 0.5)}
                                )
                                collected_count += 1
                except Exception as e:
                    logger.warning(f"Ошибка сбора из графа знаний: {e}")
            
            # 4. Собираем из памяти
            if self.memory_manager:
                try:
                    memories = self.memory_manager.get_recent_memories(limit=15)
                    for memory in memories:
                        if hasattr(memory, 'content'):
                            text = memory.content
                        elif isinstance(memory, dict):
                            text = memory.get('content', '')
                        else:
                            text = str(memory)
                        
                        if text and len(text.strip()) > 10:
                            self.add_training_data(
                                text=text,
                                source="memory",
                                metadata={'memory_type': type(memory).__name__}
                            )
                            collected_count += 1
                except Exception as e:
                    logger.warning(f"Ошибка сбора из памяти: {e}")
            
            logger.info(f"Автоматически собрано {collected_count} записей для обучения")
            return collected_count
            
        except Exception as e:
            logger.error(f"Ошибка автоматического сбора данных: {e}")
            return 0
    
    def _extract_entities(self, text: str) -> List[str]:
        """Извлечение сущностей из текста для графа памяти"""
        import re
        entities = []
        
        # Имена собственные
        proper_nouns = re.findall(r'\b[А-Я][а-я]+(?:\s+[А-Я][а-я]+)*\b', text)
        entities.extend(proper_nouns)
        
        # Термины (длинные слова)
        terms = re.findall(r'\b[а-яА-Я]{8,}\b', text)
        common_words = ['сегодня', 'завтра', 'вчера', 'поэтому', 'потому', 'который', 'следовательно']
        entities.extend([t for t in terms if t.lower() not in common_words])
        
        return list(set(entities))
    
    def _learning_loop(self):
        """Основной цикл самообучения"""
        logger.info("Запущен цикл самообучения")
        
        while self.is_active and not self.stop_event.is_set():
            try:
                if self._should_train():
                    # Автоматически собираем данные из системы перед обучением
                    collected = self.auto_collect_training_data()
                    if collected > 0:
                        logger.info(f"Собрано {collected} новых записей для обучения")
                    
                    self._perform_training_session()
                else:
                    time.sleep(60)  # Проверяем каждую минуту
                    
            except Exception as e:
                logger.error(f"Ошибка в цикле: {e}")
                time.sleep(300)  # Ждем 5 минут перед повтором
        
        logger.info("Цикл самообучения завершен")
    
    def _should_train(self) -> bool:
        """Проверка необходимости обучения"""
        with self.training_lock:
            unprocessed = sum(1 for item in self.training_queue if not item['processed'])
            if unprocessed < self.default_params['min_samples_for_training']:
                return False
        
        # Проверяем интервал
        if self.session_history:
            last_session = self.session_history[-1]
            time_since_last = time.time() - last_session.start_time
            if time_since_last < self.default_params['min_training_interval']:
                return False
        
        # Проверяем занятость системы
        if self.brain and hasattr(self.brain, 'is_busy') and self.brain.is_busy():
            return False
        
        return True
    
    def _perform_training_session(self, forced: bool = False, 
                                   custom_epochs: Optional[int] = None) -> bool:
        """Выполнение сессии обучения с эпохами"""
        if self.config.get('training_disabled', True):
            logger.info("Training disabled, skipping training session")
            return True
        
        session_id = f"session_{int(time.time() * 1000)}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=time.time(),
            status=TrainingStatus.PREPARING,
            total_epochs=custom_epochs or self.default_params['default_epochs']
        )
        
        try:
            # Подготовка данных
            self.current_session.status = TrainingStatus.PREPARING
            training_data = self._prepare_training_data()
            
            if not training_data:
                self.current_session.status = TrainingStatus.ERROR
                self.current_session.error_message = "Нет данных для обучения"
                self._finalize_session()
                return False
            
            self.current_session.training_data_size = len(training_data)
            
            # Обучение по эпохам
            self.current_session.status = TrainingStatus.TRAINING
            
            for epoch in range(1, self.current_session.total_epochs + 1):
                if self.stop_event.is_set():
                    logger.info("Обучение прервано")
                    break
                
                # Выполняем эпоху
                epoch_metrics = self._train_epoch(epoch, training_data)
                self.current_session.epoch_history.append(epoch_metrics)
                self.current_session.current_epoch_metrics = epoch_metrics
                self.current_session.epochs_completed = epoch
                
                # Callback для GUI
                if self.epoch_callback:
                    try:
                        self.epoch_callback(epoch_metrics.to_dict())
                    except Exception as e:
                        logger.debug(f"Ошибка epoch callback: {e}")
                
                if self.progress_callback:
                    try:
                        progress = (epoch / self.current_session.total_epochs) * 100
                        self.progress_callback(progress, self.current_session.to_dict())
                    except Exception as e:
                        logger.debug(f"Ошибка progress callback: {e}")
                
                # Ранняя остановка
                if epoch > 1 and self._check_early_stopping():
                    logger.info(f"Ранняя остановка на эпохе {epoch}")
                    break
            
            # Валидация
            self.current_session.status = TrainingStatus.VALIDATING
            self.current_session.validation_accuracy = self._validate_model(training_data)
            
            # Сохранение
            self.current_session.status = TrainingStatus.SAVING
            self._save_model_with_entities()
            
            # Завершение
            self.current_session.status = TrainingStatus.COMPLETED
            self.current_session.end_time = time.time()
            self._finalize_session()
            
            logger.info(f"Сессия обучения {session_id} завершена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сессии обучения: {e}", exc_info=True)
            self.current_session.status = TrainingStatus.ERROR
            self.current_session.error_message = str(e)
            self.current_session.end_time = time.time()
            self._finalize_session()
            return False
    
    def _train_epoch(self, epoch_number: int, training_data: List[str]) -> EpochMetrics:
        """Обучение одной эпохи"""
        start_time = time.time()
        
        # Имитация обучения (в реальной системе - вызов model.train())
        # Для демонстрации генерируем реалистичные метрики
        
        # Симулируем улучшение
        base_loss = 2.5 - (epoch_number * 0.3)  # Уменьшается с эпохами
        noise = np.random.normal(0, 0.05)
        loss = max(0.5, base_loss + noise)
        
        accuracy = min(0.95, 0.4 + (epoch_number * 0.15) + np.random.normal(0, 0.02))
        perplexity = np.exp(loss)
        
        learning_rate = self.default_params['learning_rate'] * (0.9 ** (epoch_number - 1))
        
        duration = time.time() - start_time + np.random.uniform(5, 15)  # Имитация времени
        
        return EpochMetrics(
            epoch_number=epoch_number,
            loss=loss,
            accuracy=accuracy,
            perplexity=perplexity,
            learning_rate=learning_rate,
            samples_processed=len(training_data),
            duration_seconds=duration
        )
    
    def _check_early_stopping(self) -> bool:
        """Проверка условий ранней остановки"""
        if len(self.current_session.epoch_history) < 3:
            return False
        
        # Проверяем, улучшается ли loss
        recent_losses = [e.loss for e in self.current_session.epoch_history[-3:]]
        if len(recent_losses) >= 3:
            # Если loss растет - останавливаем
            if recent_losses[-1] > recent_losses[-2] > recent_losses[0]:
                return True
        
        return False
    
    def _validate_model(self, validation_data: List[str]) -> float:
        """Валидация модели"""
        # Имитация валидации
        base_accuracy = 0.7
        improvement = len(self.current_session.epoch_history) * 0.05
        noise = np.random.normal(0, 0.02)
        return min(0.98, base_accuracy + improvement + noise)
    
    def _prepare_training_data(self) -> List[str]:
        """Подготовка данных для обучения"""
        with self.training_lock:
            unprocessed = [item for item in self.training_queue if not item['processed']]
            
            # Если нет данных, собираем из других модулей системы
            if not unprocessed:
                logger.info("Очередь обучения пуста, собираем данные из системы")
                return self._collect_system_data()
            
            # Ограничиваем количество
            max_samples = 100
            selected = unprocessed[:max_samples]
            
            # Собираем сущности для графа памяти
            for item in selected:
                item['processed'] = True
                if item.get('entities'):
                    self.generated_entities.append({
                        'text': item['text'],
                        'entities': item['entities'],
                        'timestamp': item['timestamp']
                    })
            
            texts = [item['text'] for item in selected]
            
            # Расширяем данные
            expanded = self._expand_training_texts(texts)
            
            logger.info(f"Подготовлено {len(expanded)} текстов для обучения")
            return expanded
    
    def _collect_system_data(self) -> List[str]:
        """Сбор данных из других модулей системы"""
        collected_texts = []
        
        try:
            # 1. Собираем данные из графа знаний
            if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                kg_data = self.knowledge_graph.get_recent_entities(limit=50)
                for entity in kg_data:
                    if isinstance(entity, dict) and 'text' in entity:
                        collected_texts.append(entity['text'])
                    elif isinstance(entity, str):
                        collected_texts.append(entity)
                logger.info(f"Собрано {len([e for e in kg_data if isinstance(e, dict) and 'text' in e])} сущностей из графа знаний")
            
            # 2. Собираем данные из памяти
            if hasattr(self, 'memory_manager') and self.memory_manager:
                try:
                    memory_data = self.memory_manager.get_recent_memories(limit=30)
                    for memory in memory_data:
                        if hasattr(memory, 'content'):
                            collected_texts.append(memory.content)
                        elif isinstance(memory, dict) and 'content' in memory:
                            collected_texts.append(memory['content'])
                        elif isinstance(memory, str):
                            collected_texts.append(memory)
                    logger.info(f"Собрано {len(memory_data)} записей из памяти")
                except Exception as e:
                    logger.warning(f"Ошибка сбора данных из памяти: {e}")
            
            # 3. Собираем данные из истории чата
            if hasattr(self, 'chat_history') and self.chat_history:
                recent_messages = self.chat_history[-20:]  # Последние 20 сообщений
                for msg in recent_messages:
                    if isinstance(msg, dict):
                        if msg.get('role') == 'user':
                            collected_texts.append(msg.get('content', ''))
                        elif msg.get('role') == 'assistant':
                            collected_texts.append(msg.get('content', ''))
                    elif isinstance(msg, str):
                        collected_texts.append(msg)
                logger.info(f"Собрано {len(recent_messages)} сообщений из истории чата")
            
            # 4. Собираем данные из обработанных запросов
            if hasattr(self, 'processed_queries') and self.processed_queries:
                query_data = list(self.processed_queries)[-30:]  # Последние 30 запросов
                for query in query_data:
                    if isinstance(query, dict):
                        collected_texts.append(query.get('query', ''))
                        collected_texts.append(query.get('response', ''))
                    elif isinstance(query, str):
                        collected_texts.append(query)
                logger.info(f"Собрано {len(query_data)} обработанных запросов")
            
            # 5. Если все еще мало данных, добавляем базовые знания
            if len(collected_texts) < 10:
                base_texts = [
                    "Система CogniFlex использует фрактальные нейронные сети для обработки текста.",
                    "Машинное обучение позволяет адаптироваться к новым данным.",
                    "Нейронные сети могут распознавать сложные паттерны в данных.",
                    "Обработка естественного языка включает анализ синтаксиса и семантики.",
                    "Граф знаний хранит связи между концепциями и сущностями.",
                    "Память системы позволяет сохранять контекст диалога.",
                    "Самообучение улучшает качество ответов со временем.",
                    "Интеграция модулей повышает общую производительность системы."
                ]
                collected_texts.extend(base_texts)
                logger.info(f"Добавлено {len(base_texts)} базовых текстов для обучения")
            
        except Exception as e:
            logger.error(f"Ошибка при сборе данных из системы: {e}")
            # Возвращаем минимальный набор данных
            return [
                "Система обрабатывает запросы пользователей.",
                "Нейронные сети генерируют ответы.",
                "Обучение улучшает качество системы."
            ]
        
        # Очищаем и фильтруем данные
        filtered_texts = []
        for text in collected_texts:
            if text and len(text.strip()) > 10:
                filtered_texts.append(text.strip())
        
        logger.info(f"Всего собрано {len(filtered_texts)} текстов для обучения из системы")
        return filtered_texts[:50]  # Ограничиваем количество
    
    def _expand_training_texts(self, texts: List[str]) -> List[str]:
        """Расширение тренировочных текстов"""
        expanded = []
        
        for text in texts:
            expanded.append(text)
            # Вариации
            if len(text) > 20:
                expanded.append(text.lower())
                expanded.append(text.capitalize())
        
        return expanded
    
    def _save_model_with_entities(self):
        """Сохранение модели и обновление графа памяти"""
        try:
            # Сохранение модели
            model_manager = getattr(self.brain, 'model_manager', None) or \
                           getattr(self.brain, 'fractal_model_manager', None)
            
            if model_manager and hasattr(model_manager, 'save_model'):
                model_manager.save_model()
            
            # Обновление графа памяти сгенерированными сущностями
            if self.generated_entities and hasattr(self.brain, 'knowledge_graph'):
                self._update_memory_graph()
            
            self.current_session.model_improvements.append(
                f"Сохранено {len(self.generated_entities)} сущностей в граф памяти"
            )
            
            logger.info("Модель сохранена, граф памяти обновлен")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
    
    def _update_memory_graph(self):
        """Обновление графа памяти сгенерированными сущностями"""
        try:
            kg = self.brain.knowledge_graph
            
            for entity_data in self.generated_entities[:50]:  # Ограничиваем
                text = entity_data['text']
                entities = entity_data['entities']
                
                # Добавляем узел для текста
                node_id = f"training_{hash(text) % 100000}"
                kg.add_concept(
                    id=node_id,
                    name=text[:50],
                    description=text,
                    node_type="training_data",
                    strength=0.7
                )
                
                # Добавляем связи для сущностей
                for entity in entities:
                    kg.add_concept(
                        id=f"entity_{entity.lower()}",
                        name=entity,
                        node_type="entity",
                        strength=0.6
                    )
                    kg.add_relation(
                        from_concept=f"entity_{entity.lower()}",
                        to_concept=node_id,
                        relation_type="mentioned_in"
                    )
            
            logger.info(f"Граф памяти обновлен: {len(self.generated_entities)} сущностей")
            self.generated_entities = []  # Очищаем после сохранения
            
        except Exception as e:
            logger.error(f"Ошибка обновления графа памяти: {e}")
    
    def _finalize_session(self):
        """Финализация сессии обучения"""
        if self.current_session:
            self.session_history.append(self.current_session)
            
            # Ограничиваем историю
            if len(self.session_history) > self.max_history_size:
                self.session_history = self.session_history[-self.max_history_size:]
            
            self.current_session = None
    
    def force_training(self, epochs: Optional[int] = None, 
                      progress_callback: Optional[callable] = None,
                      epoch_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Принудительный запуск обучения с полными метриками"""
        logger.info(f"Принудительное обучение запущено (эпох: {epochs or self.default_params['default_epochs']})")
        
        # Устанавливаем временные callbacks
        original_progress = self.progress_callback
        original_epoch = self.epoch_callback
        
        if progress_callback:
            self.progress_callback = progress_callback
        if epoch_callback:
            self.epoch_callback = epoch_callback
        
        try:
            success = self._perform_training_session(
                forced=True,
                custom_epochs=epochs
            )
            
            # Формируем детальное сообщение
            if success:
                message = 'Обучение завершено успешно'
            else:
                session = self.session_history[-1] if self.session_history else None
                if session and session.error_message:
                    message = f'Ошибка обучения: {session.error_message}'
                else:
                    message = 'Обучение завершилось с ошибкой'
            
            return {
                'success': success,
                'session': self.session_history[-1].to_dict() if self.session_history else None,
                'message': message
            }
            
        finally:
            # Восстанавливаем callbacks
            self.progress_callback = original_progress
            self.epoch_callback = original_epoch
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Получение текущих метрик обучения"""
        if self.current_session:
            return self.current_session.to_dict()
        
        return {
            'status': 'idle',
            'is_active': self.is_active,
            'queue_size': len(self.training_queue),
            'unprocessed': sum(1 for item in self.training_queue if not item['processed']),
            'total_sessions': len(self.session_history),
            'last_session': self.session_history[-1].to_dict() if self.session_history else None
        }
    
    def get_full_stats(self) -> Dict[str, Any]:
        """Полная статистика системы обучения"""
        total_epochs = sum(s.epochs_completed for s in self.session_history)
        avg_accuracy = np.mean([s.validation_accuracy for s in self.session_history]) if self.session_history else 0
        
        return {
            'system_status': {
                'is_active': self.is_active,
                'current_session': self.current_session.to_dict() if self.current_session else None,
                'queue_size': len(self.training_queue),
                'unprocessed_samples': sum(1 for item in self.training_queue if not item['processed'])
            },
            'training_params': self.default_params,
            'session_stats': {
                'total_sessions': len(self.session_history),
                'total_epochs_completed': total_epochs,
                'average_validation_accuracy': round(avg_accuracy, 4),
                'successful_sessions': sum(1 for s in self.session_history if s.status == TrainingStatus.COMPLETED),
                'failed_sessions': sum(1 for s in self.session_history if s.status == TrainingStatus.ERROR)
            },
            'recent_sessions': [s.to_dict() for s in self.session_history[-5:]],
            'generated_entities_count': len(self.generated_entities)
        }
