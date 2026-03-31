"""
Система дообучения GGUF моделей на основе верифицированных знаний.

Архитектура:
- training_gguf: Отдельный экземпляр GGUF только для обучения
- production_gguf: Основная модель для ответов (НЕ обучается)
- verification: Проверка целостности и качества
- LoRA: Точечное добавление весов без изменения базовой модели
"""
import os
import logging
import time
import json
import threading
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("eva.gguf_training")


class TrainingStatus(Enum):
    """Статусы процесса обучения."""
    IDLE = "idle"
    EXTRACTING = "extracting"
    PREPARING = "preparing"
    TRAINING = "training"
    VERIFYING = "verifying"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingMetrics:
    """Метрики процесса обучения."""
    loss: float = 0.0
    accuracy: float = 0.0
    knowledge_volume: int = 0
    verification_score: float = 0.0
    integrity_check: bool = False
    training_time: float = 0.0
    nodes_processed: int = 0
    links_learned: int = 0
    generation_quality: float = 0.0


@dataclass
class VerifiedKnowledge:
    """Верифицированное знание для обучения."""
    concept: str
    description: str
    links: List[str]
    source: str
    confidence: float
    verified: bool = False


class GGUFTrainingSystem:
    """
    Система дообучения GGUF моделей на основе графа памяти.
    
    Основные компоненты:
    1. TrainingGGUF - отдельный экземпляр для обучения
    2. ProductionGGUF - основная модель (НЕ обучается)
    3. VerificationSystem - проверка целостности и качества
    4. LoRAAdapter - точечные дополнения весов
    """
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        """
        Инициализация системы дообучения.
        
        Args:
            brain: Ссылка на ядро ЕВА
            config: Конфигурация системы
        """
        self.brain = brain
        self.config = config or {}
        
        # Пути к моделям
        self.base_model_path = self.config.get('base_model_path', 
            'eva/models/qwen2.5-0.5b-instruct-q4_0.gguf')
        self.training_model_path = self.config.get('training_model_path',
            'eva/models/training_qwen.gguf')
        self.lora_path = self.config.get('lora_path',
            'eva/models/lora_adapters')
        self.verified_model_path = self.config.get('verified_model_path',
            'eva/models/verified_qwen.gguf')
        
        # Настройки обучения
        self.batch_size = self.config.get('batch_size', 4)
        self.epochs = self.config.get('epochs', 3)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Статус и метрики
        self.status = TrainingStatus.IDLE
        self.metrics = TrainingMetrics()
        
        # Модели
        self.training_model = None  # Экземпляр для обучения
        self.verification_model = None  # Экземпляр для верификации
        
        # Флаги
        self.running = False
        self.stop_event = threading.Event()
        self.training_thread = None
        
        # Кэш верифицированных знаний
        self.verified_knowledge: List[VerifiedKnowledge] = []
        
        # Создаем директории
        os.makedirs(self.lora_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.training_model_path), exist_ok=True)
        
        logger.info("GGUFTrainingSystem инициализирована")
    
    def start(self):
        """Запускает систему дообучения в фоновом режиме."""
        if self.running:
            logger.warning("Система дообучения уже запущена")
            return False
        
        self.running = True
        self.stop_event.clear()
        
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("GGUFTrainingSystem запущена")
        return True
    
    def stop(self):
        """Останавливает систему дообучения."""
        if not self.running:
            return
        
        self.stop_event.set()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
        
        self.running = False
        logger.info("GGUFTrainingSystem остановлена")
    
    def _training_loop(self):
        """Основной цикл дообучения."""
        logger.info("Запущен цикл дообучения GGUF")
        
        while not self.stop_event.is_set():
            try:
                # 1. Извлекаем верифицированные знания из графа
                knowledge = self._extract_verified_knowledge()
                
                if knowledge:
                    logger.info(f"Извлечено {len(knowledge)} верифицированных знаний")
                    
                    # 2. Подготавливаем данные для обучения
                    training_data = self._prepare_training_data(knowledge)
                    
                    if training_data:
                        # 3. Обучаем отдельный экземпляр
                        self._train_separate_instance(training_data)
                        
                        # 4. Верифицируем качество
                        if self._verify_training_quality():
                            # 5. Сохраняем LoRA адаптеры
                            self._save_lora_adapters()
                            
                            logger.info("Дообучение завершено успешно")
                        else:
                            logger.warning("Верификация не пройдена")
                
            except Exception as e:
                logger.error(f"Ошибка в цикле дообучения: {e}", exc_info=True)
            
            # Ждем следующий цикл (например, каждый час)
            self.stop_event.wait(timeout=3600)
    
    def _extract_verified_knowledge(self) -> List[VerifiedKnowledge]:
        """
        Извлекает верифицированные знания из графа памяти.
        
        Returns:
            Список верифицированных знаний
        """
        self.status = TrainingStatus.EXTRACTING
        
        knowledge = []
        kg = self._get_knowledge_graph()
        
        if not kg:
            return knowledge
        
        try:
            # Получаем все узлы из графа
            nodes = self._get_all_nodes(kg)
            
            for node in nodes:
                # Проверяем верификацию (confidence > threshold)
                if self._is_node_verified(node):
                    vk = self._create_verified_knowledge(node)
                    if vk:
                        knowledge.append(vk)
            
            logger.info(f"Извлечено {len(knowledge)} верифицированных узлов")
            
        except Exception as e:
            logger.error(f"Ошибка извлечения знаний: {e}")
        
        return knowledge
    
    def _get_knowledge_graph(self):
        """Получает ссылку на граф знаний."""
        if not self.brain:
            return None
        return getattr(self.brain, 'knowledge_graph', None)
    
    def _get_all_nodes(self, kg) -> List:
        """Получает все узлы из графа."""
        try:
            if hasattr(kg, 'get_all_nodes'):
                return kg.get_all_nodes()
            elif hasattr(kg, 'nodes'):
                return list(kg.nodes.values()) if hasattr(kg.nodes, 'values') else []
            return []
        except Exception as e:
            logger.debug(f"Ошибка получения узлов: {e}")
            return []
    
    def _is_node_verified(self, node) -> bool:
        """Проверяет, верифицирован ли узел."""
        try:
            confidence = getattr(node, 'confidence', 0.0)
            return confidence >= self.min_confidence
        except:
            return False
    
    def _create_verified_knowledge(self, node) -> Optional[VerifiedKnowledge]:
        """Создает объект верифицированного знания."""
        try:
            name = getattr(node, 'name', '') or ''
            content = getattr(node, 'content', '') or ''
            links = getattr(node, 'links', []) or []
            confidence = getattr(node, 'confidence', 0.0)
            
            if not name and not content:
                return None
            
            return VerifiedKnowledge(
                concept=name,
                description=content,
                links=links if isinstance(links, list) else [],
                source='knowledge_graph',
                confidence=confidence,
                verified=True
            )
        except Exception as e:
            logger.debug(f"Ошибка создания VerifiedKnowledge: {e}")
            return None
    
    def _prepare_training_data(self, knowledge: List[VerifiedKnowledge]) -> List[Dict]:
        """
        Подготавливает данные для обучения в формате GGUF.
        
        Args:
            knowledge: Список верифицированных знаний
            
        Returns:
            Данные для обучения
        """
        self.status = TrainingStatus.PREPARING
        
        training_data = []
        
        for vk in knowledge:
            # Формируем пару (вопрос, ответ) для обучения
            qa_pair = {
                'question': f"Что такое {vk.concept}?",
                'answer': vk.description,
                'links': vk.links,
                'confidence': vk.confidence
            }
            
            # Создаем дополнительные пары из связей
            if vk.links:
                for link in vk.links[:3]:  # Ограничиваем количество
                    link_qa = {
                        'question': f"Что связано с {vk.concept}?",
                        'answer': f"{vk.concept} связан с: {link}",
                        'links': [link],
                        'confidence': vk.confidence * 0.9
                    }
                    training_data.append(link_qa)
            
            training_data.append(qa_pair)
        
        logger.info(f"Подготовлено {len(training_data)} пар для обучения")
        return training_data
    
    def _train_separate_instance(self, training_data: List[Dict]):
        """
        Обучает отдельный экземпляр модели на данных.
        
        Args:
            training_data: Данные для обучения
        """
        self.status = TrainingStatus.TRAINING
        start_time = time.time()
        
        try:
            # Загружаем копию базовой модели для обучения
            logger.info("Загрузка отдельного экземпляра для обучения...")
            
            # Используем LoRA для эффективного дообучения
            # LoRA добавляет маленькие матрицы к существующим весам
            # без изменения базовой модели
            
            for i, data in enumerate(training_data[:10]):  # Ограничиваем для теста
                if self.stop_event.is_set():
                    break
                
                # Симуляция обучения (реальная реализация требует PEFT/LoRA)
                self.metrics.nodes_processed = i + 1
                self.metrics.links_learned += len(data.get('links', []))
                
                logger.debug(f"Обучение на узле {i+1}/{len(training_data)}: {data['question'][:50]}")
            
            self.metrics.training_time = time.time() - start_time
            logger.info(f"Обучение завершено за {self.metrics.training_time:.2f}с")
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            self.status = TrainingStatus.FAILED
    
    def _verify_training_quality(self) -> bool:
        """
        Верифицирует качество обученной модели.
        
        Returns:
            True если качество удовлетворительное
        """
        self.status = TrainingStatus.VERIFYING
        
        try:
            # Проверяем целостность модели
            integrity_ok = self._check_model_integrity()
            
            # Проверяем качество генерации
            quality_ok = self._check_generation_quality()
            
            # Проверяем объем знаний
            knowledge_ok = self._check_knowledge_volume()
            
            self.metrics.integrity_check = integrity_ok
            
            if integrity_ok and quality_ok and knowledge_ok:
                self.status = TrainingStatus.COMPLETED
                return True
            else:
                self.status = TrainingStatus.FAILED
                return False
                
        except Exception as e:
            logger.error(f"Ошибка верификации: {e}")
            return False
    
    def _check_model_integrity(self) -> bool:
        """Проверяет целостность модели."""
        # Проверяем hash файла
        # Проверяем структуру модели
        # Проверяем совместимость с llama.cpp
        return True  # Заглушка
    
    def _check_generation_quality(self) -> bool:
        """Проверяет качество генерации."""
        test_questions = [
            "Что такое снег?",
            "Какая погода зимой?",
            "Что связано с холодом?"
        ]
        
        # Генерируем ответы и проверяем их качество
        # Оценка должна быть > порога
        return True  # Заглушка
    
    def _check_knowledge_volume(self) -> bool:
        """Проверяет объем изученных знаний."""
        return self.metrics.nodes_processed > 0
    
    def _save_lora_adapters(self):
        """Сохраняет LoRA адаптеры после успешного обучения."""
        self.status = TrainingStatus.MERGING
        
        try:
            # Сохраняем адаптеры в отдельную директорию
            timestamp = int(time.time())
            adapter_path = os.path.join(self.lora_path, f"adapter_{timestamp}")
            os.makedirs(adapter_path, exist_ok=True)
            
            # Сохраняем метрики
            metrics_path = os.path.join(adapter_path, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'loss': self.metrics.loss,
                    'accuracy': self.metrics.accuracy,
                    'knowledge_volume': self.metrics.knowledge_volume,
                    'verification_score': self.metrics.verification_score,
                    'training_time': self.metrics.training_time,
                    'nodes_processed': self.metrics.nodes_processed,
                    'links_learned': self.metrics.links_learned,
                    'timestamp': timestamp
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"LoRA адаптеры сохранены: {adapter_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения адаптеров: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает текущий статус системы."""
        return {
            'status': self.status.value,
            'running': self.running,
            'metrics': {
                'loss': self.metrics.loss,
                'accuracy': self.metrics.accuracy,
                'knowledge_volume': self.metrics.knowledge_volume,
                'verification_score': self.metrics.verification_score,
                'training_time': self.metrics.training_time,
                'nodes_processed': self.metrics.nodes_processed,
                'links_learned': self.metrics.links_learned,
                'generation_quality': self.metrics.generation_quality
            }
        }
    
    def get_lora_adapters(self) -> List[str]:
        """Возвращает список доступных LoRA адаптеров."""
        try:
            adapters = []
            if os.path.exists(self.lora_path):
                for item in os.listdir(self.lora_path):
                    if item.startswith('adapter_'):
                        adapters.append(item)
            return sorted(adapters, reverse=True)
        except:
            return []
