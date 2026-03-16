"""
Улучшенная интеграция веб-поиска и самообучения без конфликтов методов
"""
import os
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger("cogniflex.enhanced_learning")

@dataclass
class LearningSession:
    """Сессия обучения"""
    session_id: str
    topics: List[str]
    start_time: float
    end_time: Optional[float] = None
    training_texts: List[str] = field(default_factory=list)
    quality_before: float = 0.0
    quality_after: float = 0.0
    status: str = "pending"
    web_searches: int = 0
    knowledge_extracted: int = 0

class EnhancedLearningIntegration:
    """Улучшенная интеграция обучения с веб-поиском без конфликтов"""
    
    def __init__(self, fractal_model_manager, config_path: Optional[str] = None):
        """
        Инициализирует улучшенную интеграцию обучения
        
        Args:
            fractal_model_manager: Менеджер фрактальной модели
            config_path: Путь к конфигурации
        """
        self.fractal_model_manager = fractal_model_manager
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 
            "..", "config", "enhanced_learning_config.json"
        )
        
        # Загружаем конфигурацию
        self.config = self._load_config()
        
        # Проверяем существующие методы в менеджере
        self._check_existing_methods()
        
        # Инициализируем компоненты
        self._initialize_components()
        
        # Сессии обучения
        self.learning_sessions = {}
        self.active_session = None
        
        # Статистика системы
        self.system_stats = {
            "total_sessions": 0,
            "completed_sessions": 0,
            "total_web_searches": 0,
            "total_knowledge_extracted": 0,
            "average_quality_improvement": 0.0,
            "total_training_texts": 0,
            "system_uptime": time.time(),
            "last_update": time.time()
        }
        
        # Фоновые процессы
        self.background_executor = ThreadPoolExecutor(max_workers=2)
        self.monitoring_thread = None
        
        # Запускаем мониторинг
        self._start_monitoring()
        
        logger.info("EnhancedLearningIntegration инициализирована")
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию системы"""
        
        default_config = {
            "enhanced_web_search": {
                "auto_search_threshold": 0.6,
                "max_search_results": 5,
                "search_timeout": 30.0,
                "use_search_cache": True,
                "search_engines": ["google", "yandex"],
                "min_content_length": 100,
                "max_content_length": 1000
            },
            "enhanced_learning": {
                "auto_learning_enabled": True,
                "learning_interval_hours": 6,
                "max_topics_per_session": 10,
                "max_texts_per_topic": 3,
                "quality_threshold": 0.7,
                "min_improvement_threshold": 0.1,
                "training_epochs": 3,
                "batch_size": 2
            },
            "learning_topics": {
                "core_topics": [
                    "машинное обучение",
                    "нейронные сети",
                    "искусственный интеллект",
                    "глубокое обучение",
                    "трансформеры",
                    "NLP",
                    "компьютерное зрение",
                    "рекуррентные сети",
                    "LSTM",
                    "attention механизм"
                ],
                "trending_topics": [
                    "GPT модели",
                    "BERT",
                    "квантовые вычисления",
                    "обучение с подкреплением",
                    "transfer learning",
                    "fine-tuning",
                    "оптимизация гиперпараметров",
                    "свёрточные сети",
                    "генеративные модели",
                    "мультимодальное обучение"
                ],
                "custom_topics": []
            },
            "monitoring": {
                "enable_monitoring": True,
                "stats_update_interval": 60,
                "performance_tracking": True,
                "quality_monitoring": True,
                "auto_cleanup": True,
                "cleanup_interval_hours": 24
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    for section, values in loaded_config.items():
                        if section in default_config:
                            default_config[section].update(values)
                        else:
                            default_config[section] = values
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
        
        # Сохраняем конфигурацию
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        return default_config
    
    def _check_existing_methods(self):
        """Проверяет существующие методы в менеджере для избежания конфликтов"""
        
        self.has_web_search = hasattr(self.fractal_model_manager, 'web_search_integration')
        self.has_generate_with_web = hasattr(self.fractal_model_manager, 'generate_response_with_web_search')
        self.has_training_texts = hasattr(self.fractal_model_manager, 'generate_training_texts_from_web')
        self.has_web_stats = hasattr(self.fractal_model_manager, 'get_web_search_stats')
        self.has_configure_web = hasattr(self.fractal_model_manager, 'configure_web_search')
        self.has_clear_cache = hasattr(self.fractal_model_manager, 'clear_web_search_cache')
        
        logger.info(f"Проверка методов: web_search={self.has_web_search}, "
                   f"generate_with_web={self.has_generate_with_web}, "
                   f"training_texts={self.has_training_texts}")
    
    def _initialize_components(self):
        """Инициализирует компоненты системы"""
        
        # Используем существующую интеграцию веб-поиска если доступна
        if self.has_web_search:
            self.web_search_integration = self.fractal_model_manager.web_search_integration
            logger.info("Используется существующая интеграция веб-поиска")
        else:
            logger.warning("Веб-поиск недоступен")
            self.web_search_integration = None
        
        # Настраиваем веб-поиск если доступен
        if self.web_search_integration:
            self.web_search_integration.configure_integration(**self.config["enhanced_web_search"])
        
        logger.info("Компоненты инициализированы")
    
    def _start_monitoring(self):
        """Запускает мониторинг"""
        
        if self.config["monitoring"]["enable_monitoring"]:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Мониторинг запущен")
    
    def start_enhanced_learning_session(self, topics: Optional[List[str]] = None, 
                                       session_name: Optional[str] = None) -> str:
        """
        Запускает улучшенную сессию обучения
        
        Args:
            topics: Список тем для обучения
            session_name: Название сессии
            
        Returns:
            str: ID сессии
        """
        
        try:
            # Генерируем ID сессии
            session_id = session_name or f"enhanced_session_{int(time.time())}"
            
            # Определяем темы
            if not topics:
                topics = self._get_learning_topics()
            
            # Создаем сессию
            session = LearningSession(
                session_id=session_id,
                topics=topics,
                start_time=time.time()
            )
            
            self.learning_sessions[session_id] = session
            self.active_session = session_id
            
            # Запускаем обучение в фоне
            self.background_executor.submit(self._run_enhanced_learning_session, session_id)
            
            logger.info(f"Улучшенная сессия обучения запущена: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Ошибка запуска улучшенной сессии обучения: {e}")
            raise
    
    def _run_enhanced_learning_session(self, session_id: str):
        """Выполняет улучшенную сессию обучения"""
        
        try:
            session = self.learning_sessions[session_id]
            session.status = "active"
            
            logger.info(f"Выполнение улучшенной сессии {session_id}")
            
            # 1. Анализ качества до обучения
            session.quality_before = self._assess_current_quality()
            
            # 2. Генерация обучающих текстов
            training_texts = self._generate_enhanced_training_texts(session.topics)
            session.training_texts = training_texts
            
            # 3. Обучение модели
            if training_texts:
                improvement_result = self._train_model_enhanced(training_texts)
                
                if improvement_result.get('status') == 'success':
                    # 4. Анализ качества после обучения
                    session.quality_after = self._assess_current_quality()
                    
                    # 5. Обновляем статистику
                    self._update_session_stats(session, improvement_result)
                    
                    session.status = "completed"
                    logger.info(f"Улучшенная сессия {session_id} завершена успешно")
                else:
                    session.status = "failed"
                    logger.error(f"Улучшенная сессия {session_id} завершилась с ошибкой")
            else:
                session.status = "failed"
                logger.warning(f"Улучшенная сессия {session_id}: нет обучающих текстов")
            
            session.end_time = time.time()
            
        except Exception as e:
            logger.error(f"Ошибка выполнения улучшенной сессии {session_id}: {e}")
            session.status = "failed"
            session.end_time = time.time()
    
    def _get_learning_topics(self) -> List[str]:
        """Получает темы для обучения"""
        
        all_topics = []
        all_topics.extend(self.config["learning_topics"]["core_topics"])
        all_topics.extend(self.config["learning_topics"]["trending_topics"])
        all_topics.extend(self.config["learning_topics"]["custom_topics"])
        
        # Выбираем случайные темы
        import random
        max_topics = self.config["enhanced_learning"]["max_topics_per_session"]
        selected_topics = random.sample(all_topics, min(max_topics, len(all_topics)))
        
        return selected_topics
    
    def _generate_enhanced_training_texts(self, topics: List[str]) -> List[str]:
        """Генерирует улучшенные обучающие тексты"""
        
        if not self.web_search_integration:
            logger.warning("Веб-поиск недоступен для генерации текстов")
            return []
        
        try:
            max_texts = self.config["enhanced_learning"]["max_texts_per_topic"]
            
            # Используем существующий метод если доступен
            if self.has_training_texts:
                training_texts = self.fractal_model_manager.generate_training_texts_from_web(
                    topics, max_texts_per_topic=max_texts
                )
            else:
                # Используем прямую интеграцию
                training_texts = self.web_search_integration.generate_training_texts_from_search(
                    topics, max_texts_per_topic=max_texts
                )
            
            logger.info(f"Сгенерировано {len(training_texts)} улучшенных обучающих текстов")
            return training_texts
            
        except Exception as e:
            logger.error(f"Ошибка генерации улучшенных обучающих текстов: {e}")
            return []
    
    def _train_model_enhanced(self, training_texts: List[str]) -> Dict[str, Any]:
        """Обучает модель с улучшенными параметрами"""
        
        try:
            epochs = self.config["enhanced_learning"]["training_epochs"]
            batch_size = self.config["enhanced_learning"]["batch_size"]
            
            improvement_result = self.fractal_model_manager.improve_quality(
                training_texts=training_texts,
                epochs=epochs,
                batch_size=batch_size
            )
            
            return improvement_result
            
        except Exception as e:
            logger.error(f"Ошибка улучшенного обучения модели: {e}")
            return {"status": "error", "message": str(e)}
    
    def _assess_current_quality(self) -> float:
        """Оценивает текущее качество модели"""
        
        try:
            quality_metrics = self.fractal_model_manager.get_quality_metrics()
            return quality_metrics.get("overall", 0.0)
        except Exception as e:
            logger.error(f"Ошибка оценки качества: {e}")
            return 0.0
    
    def _update_session_stats(self, session: LearningSession, improvement_result: Dict[str, Any]):
        """Обновляет статистику сессии"""
        
        # Статистика веб-поиска
        if self.web_search_integration:
            web_stats = self.web_search_integration.get_integration_stats()
            session.web_searches = web_stats.get("search_queries", 0)
            session.knowledge_extracted = web_stats.get("knowledge_extracted", 0)
        
        # Обновляем общую статистику
        self.system_stats["total_sessions"] += 1
        self.system_stats["total_web_searches"] += session.web_searches
        self.system_stats["total_knowledge_extracted"] += session.knowledge_extracted
        self.system_stats["total_training_texts"] += len(session.training_texts)
        
        if session.status == "completed":
            self.system_stats["completed_sessions"] += 1
            
            # Улучшение качества
            quality_improvement = session.quality_after - session.quality_before
            if quality_improvement > self.config["enhanced_learning"]["min_improvement_threshold"]:
                completed = self.system_stats["completed_sessions"]
                current_avg = self.system_stats["average_quality_improvement"]
                new_avg = (current_avg * (completed - 1) + quality_improvement) / completed
                self.system_stats["average_quality_improvement"] = new_avg
        
        self.system_stats["last_update"] = time.time()
    
    def generate_enhanced_response(self, query: str, max_tokens: int = 100, 
                                 use_web_search: bool = True) -> Dict[str, Any]:
        """
        Генерирует улучшенный ответ с использованием веб-поиска
        
        Args:
            query: Запрос пользователя
            max_tokens: Максимальное количество токенов
            use_web_search: Использовать веб-поиск
            
        Returns:
            Dict[str, Any]: Результат генерации
        """
        
        try:
            if use_web_search and self.has_generate_with_web:
                # Используем существующий метод
                result = self.fractal_model_manager.generate_response_with_web_search(
                    query, max_tokens, use_web_search=True
                )
                
                # Добавляем информацию об улучшенной системе
                result["enhanced_system_info"] = {
                    "active_session": self.active_session,
                    "total_sessions": self.system_stats["total_sessions"],
                    "system_uptime": time.time() - self.system_stats["system_uptime"],
                    "enhanced_learning": True
                }
                
                return result
            else:
                # Fallback на обычную генерацию
                response = self.fractal_model_manager.generate_response(query, max_tokens)
                return {
                    "status": "completed",
                    "response": response,
                    "web_search_used": False,
                    "enhanced_system_info": {
                        "active_session": self.active_session,
                        "total_sessions": self.system_stats["total_sessions"],
                        "enhanced_learning": True
                    }
                }
                
        except Exception as e:
            logger.error(f"Ошибка генерации улучшенного ответа: {e}")
            return {
                "status": "error",
                "error": str(e),
                "response": self.fractal_model_manager.generate_response(query, max_tokens)
            }
    
    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Возвращает статус улучшенной системы"""
        
        status = {
            "enhanced_system": {
                "uptime": time.time() - self.system_stats["system_uptime"],
                "last_update": self.system_stats["last_update"],
                "active_session": self.active_session,
                "web_search_available": self.web_search_integration is not None,
                "existing_methods": {
                    "web_search": self.has_web_search,
                    "generate_with_web": self.has_generate_with_web,
                    "training_texts": self.has_training_texts,
                    "web_stats": self.has_web_stats,
                    "configure_web": self.has_configure_web,
                    "clear_cache": self.has_clear_cache
                }
            },
            "statistics": self.system_stats.copy(),
            "sessions": {},
            "performance": {}
        }
        
        # Информация о сессиях
        for session_id, session in self.learning_sessions.items():
            status["sessions"][session_id] = {
                "status": session.status,
                "topics_count": len(session.topics),
                "training_texts": len(session.training_texts),
                "quality_before": session.quality_before,
                "quality_after": session.quality_after,
                "improvement": session.quality_after - session.quality_before,
                "web_searches": session.web_searches,
                "knowledge_extracted": session.knowledge_extracted,
                "duration": (session.end_time or time.time()) - session.start_time
            }
        
        # Производительность
        if hasattr(self.fractal_model_manager, 'get_performance_stats'):
            status["performance"] = self.fractal_model_manager.get_performance_stats()
        
        # Статистика веб-поиска
        if self.has_web_stats:
            status["web_search_stats"] = self.fractal_model_manager.get_web_search_stats()
        elif self.web_search_integration:
            status["web_search_stats"] = self.web_search_integration.get_integration_stats()
        
        return status
    
    def add_enhanced_topics(self, topics: List[str]):
        """Добавляет улучшенные темы"""
        
        self.config["learning_topics"]["custom_topics"].extend(topics)
        self.config["learning_topics"]["custom_topics"] = list(set(self.config["learning_topics"]["custom_topics"]))
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Добавлено {len(topics)} улучшенных тем")
    
    def configure_enhanced_learning(self, **settings):
        """Настраивает параметры улучшенного обучения"""
        
        self.config["enhanced_learning"].update(settings)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        # Применяем настройки
        if self.web_search_integration:
            self.web_search_integration.configure_integration(**self.config["enhanced_web_search"])
        
        logger.info(f"Параметры улучшенного обучения обновлены: {settings}")
    
    def _monitoring_loop(self):
        """Фоновый цикл мониторинга"""
        
        interval = self.config["monitoring"]["stats_update_interval"]
        
        while True:
            try:
                time.sleep(interval)
                self._update_system_stats()
                
                if self.config["monitoring"]["auto_cleanup"]:
                    self._perform_cleanup()
                
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                time.sleep(interval)
    
    def _update_system_stats(self):
        """Обновляет статистику системы"""
        
        try:
            self.system_stats["system_uptime"] = time.time() - self.system_stats["system_uptime"]
            self.system_stats["last_update"] = time.time()
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def _perform_cleanup(self):
        """Выполняет очистку системы"""
        
        try:
            current_time = time.time()
            cleanup_interval = self.config["monitoring"]["cleanup_interval_hours"] * 3600
            
            sessions_to_remove = []
            for session_id, session in self.learning_sessions.items():
                if session.end_time and (current_time - session.end_time) > cleanup_interval:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.learning_sessions[session_id]
                logger.info(f"Сессия {session_id} удалена при очистке")
            
            if self.web_search_integration:
                self.web_search_integration.clear_cache()
            
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
    
    def __del__(self):
        """Очистка при удалении"""
        
        try:
            if self.background_executor:
                self.background_executor.shutdown(wait=False)
            
            logger.info("EnhancedLearningIntegration очищена")
        except Exception:
            pass
