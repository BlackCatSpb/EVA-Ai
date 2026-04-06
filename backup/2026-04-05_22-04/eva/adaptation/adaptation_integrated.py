"""
Интегрированный адаптационный менеджер ЕВА
Поддерживает BaseComponent и EventBus
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("eva.adaptation")

from eva.core.base_component import BaseComponent, ComponentState
from eva.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный AdaptationManager
try:
    from eva.adaptation.adaptation_core import AdaptationManager
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный AdaptationManager недоступен")


class IntegratedAdaptationManager(BaseComponent):
    """Интегрированный менеджер адаптации с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("adaptation_manager", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'adaptation_cache')
        
        # Инициализируем оригинальный менеджер если доступен
        self._original_manager = None
        if ORIGINAL_AVAILABLE:
            try:
                self._original_manager = AdaptationManager(brain, cache_dir)
                logger.info("Оригинальный AdaptationManager инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального менеджера: {e}")
        
        # Статистика
        self.stats = {
            "adaptations_performed": 0,
            "profiles_created": 0,
            "feedback_processed": 0,
            "errors": 0
        }
        
        logger.info(f"IntegratedAdaptationManager {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация адаптационного менеджера...")
            
            # Инициализируем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'initialize'):
                self._original_manager.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Публикуем событие инициализации
            self._emit_event("adaptation_manager.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации адаптационного менеджера: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск адаптационного менеджера...")
            
            # Запускаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'start'):
                self._original_manager.start()
            
            # Публикуем событие запуска
            self._emit_event("adaptation_manager.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска адаптационного менеджера: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка адаптационного менеджера...")
            
            # Останавливаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'stop'):
                self._original_manager.stop()
            
            # Публикуем событие остановки
            self._emit_event("adaptation_manager.stopped", {
                'component': self.name,
                'stats': self.stats
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки адаптационного менеджера: {e}")
            return False
    
    def adapt_response(self, query: str, response: str, user_profile: Optional[Dict] = None) -> str:
        """Адаптирует ответ под профиль пользователя"""
        start_time = time.time()
        
        try:
            if self._original_manager and hasattr(self._original_manager, 'adapt_response'):
                adapted_response = self._original_manager.adapt_response(query, response, user_profile)
            else:
                # Базовая адаптация
                adapted_response = self._basic_adaptation(query, response, user_profile)
            
            # Обновляем статистику
            self.stats["adaptations_performed"] += 1
            
            # Публикуем событие адаптации
            self._emit_event("adaptation_manager.response_adapted", {
                'query_length': len(query),
                'response_length': len(response),
                'adapted_length': len(adapted_response),
                'processing_time': time.time() - start_time
            })
            
            return adapted_response
            
        except Exception as e:
            logger.error(f"Ошибка адаптации ответа: {e}")
            self.stats["errors"] += 1
            return response  # Возвращаем оригинальный ответ при ошибке
    
    def _basic_adaptation(self, query: str, response: str, user_profile: Optional[Dict] = None) -> str:
        """Базовая адаптация ответа"""
        # Простая логика адаптации
        if user_profile:
            # Если профиль указывает на формальный стиль
            if user_profile.get('style') == 'formal':
                response = response.replace("Привет", "Здравствуйте")
            elif user_profile.get('style') == 'casual':
                response = response.replace("Здравствуйте", "Привет")
        
        return response
    
    def create_user_profile(self, user_id: str, preferences: Dict) -> bool:
        """Создает профиль пользователя"""
        try:
            if self._original_manager and hasattr(self._original_manager, 'create_user_profile'):
                success = self._original_manager.create_user_profile(user_id, preferences)
            else:
                # Базовое создание профиля
                success = self._create_basic_profile(user_id, preferences)
            
            if success:
                self.stats["profiles_created"] += 1
                self._emit_event("adaptation_manager.profile_created", {
                    'user_id': user_id,
                    'preferences': preferences
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка создания профиля: {e}")
            self.stats["errors"] += 1
            return False
    
    def _create_basic_profile(self, user_id: str, preferences: Dict) -> bool:
        """Базовое создание профиля"""
        profile_path = os.path.join(self.cache_dir, f"profile_{user_id}.json")
        try:
            import json
            with open(profile_path, 'w') as f:
                json.dump({
                    'user_id': user_id,
                    'preferences': preferences,
                    'created_at': datetime.now().isoformat()
                }, f)
            return True
        except Exception:
            return False
    
    def process_feedback(self, user_id: str, feedback: Dict) -> bool:
        """Обрабатывает обратную связь"""
        try:
            if self._original_manager and hasattr(self._original_manager, 'process_feedback'):
                success = self._original_manager.process_feedback(user_id, feedback)
            else:
                # Базовая обработка
                success = self._process_basic_feedback(user_id, feedback)
            
            if success:
                self.stats["feedback_processed"] += 1
                self._emit_event("adaptation_manager.feedback_processed", {
                    'user_id': user_id,
                    'feedback_type': feedback.get('type'),
                    'rating': feedback.get('rating')
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка обработки обратной связи: {e}")
            self.stats["errors"] += 1
            return False
    
    def _process_basic_feedback(self, user_id: str, feedback: Dict) -> bool:
        """Базовая обработка обратной связи"""
        feedback_path = os.path.join(self.cache_dir, f"feedback_{user_id}.json")
        try:
            import json
            # Загружаем существующие отзывы
            existing_feedback = []
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    existing_feedback = json.load(f)
            
            # Добавляем новый отзыв
            existing_feedback.append({
                **feedback,
                'timestamp': datetime.now().isoformat()
            })
            
            # Сохраняем
            with open(feedback_path, 'w') as f:
                json.dump(existing_feedback, f)
            
            return True
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы"""
        stats = self.stats.copy()
        
        # Добавляем статистику из оригинального менеджера
        if self._original_manager and hasattr(self._original_manager, 'get_statistics'):
            original_stats = self._original_manager.get_statistics()
            stats.update(original_stats)
        
        return stats
    
    def _extract_concept_from_query(self, query: str) -> Optional[str]:
        """
        Извлечение концепта из запроса пользователя.
        
        Args:
            query: Текст запроса
            
        Returns:
            Optional[str]: Идентификатор концепта или None
        """
        try:
            # Если оригинальный менеджер имеет этот метод, используем его
            if self._original_manager and hasattr(self._original_manager, '_extract_concept_from_query'):
                return self._original_manager._extract_concept_from_query(query)
            
            # Иначе используем базовую реализацию
            query_lower = query.lower()
            
            # Список известных концептов
            concepts = [
                {"name": "нейроэстетика", "keywords": ["нейроэстетика", "нейроэстетический"], "weight": 1.0},
                {"name": "этика_искусственного_интеллекта", "keywords": ["этика ии", "этика искусственного интеллекта"], "weight": 0.8},
                {"name": "автономия_человека", "keywords": ["автономия человека", "человеческая автономия"], "weight": 0.9},
                {"name": "когнитивные_библиотеки", "keywords": ["когнитивные библиотеки", "библиотеки знаний"], "weight": 0.7},
                {"name": "нейроморфные_системы", "keywords": ["нейроморфные системы", "нейроморфный"], "weight": 1.0},
                {"name": "эмоциональный_интеллект", "keywords": ["эмоциональный интеллект", "эмоциональный ии"], "weight": 0.9},
                {"name": "квантовые_вычисления", "keywords": ["квантовые вычисления", "квантовый компьютер"], "weight": 0.8},
                {"name": "нейронные_сети", "keywords": ["нейронные сети", "глубокое обучение"], "weight": 0.9},
                {"name": "машинное_обучение", "keywords": ["машинное обучение", "обучение с учителем"], "weight": 0.8},
                {"name": "обработка_естественного_языка", "keywords": ["обработка естественного языка", "нлп"], "weight": 0.9}
            ]
            
            # Поиск лучшего совпадения
            best_match = None
            best_score = 0.0
            
            for concept in concepts:
                score = 0.0
                for keyword in concept["keywords"]:
                    if keyword.lower() in query_lower:
                        score += concept["weight"]
                
                if concept["name"].lower() in query_lower:
                    score += concept["weight"] * 1.5
                
                if score > best_score:
                    best_score = score
                    best_match = concept["name"]
            
            return best_match if best_score > 0.5 else None
            
        except Exception as e:
            logger.error(f"Ошибка извлечения концепта: {e}", exc_info=True)
            return None
