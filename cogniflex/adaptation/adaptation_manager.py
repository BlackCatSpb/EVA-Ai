"""Модуль адаптации для CogniFlex - динамическая адаптация к пользовательским потребностям"""
import os
import logging
import time
import json
import numpy as np
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta

logger = logging.getLogger("cogniflex.adaptation.core")

class AdaptationManager:
    """Менеджер адаптации системы к пользовательским потребностям и контексту."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует менеджер адаптации.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            cache_dir: Путь к директории кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        
        # Создаем директорию кэша
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Пути к файлам
        self.user_profiles_file = os.path.join(self.cache_dir, "user_profiles.json") if self.cache_dir else None
        self.concept_cache_file = os.path.join(self.cache_dir, "concept_cache.json") if self.cache_dir else None
        self.adaptation_metrics_file = os.path.join(self.cache_dir, "adaptation_metrics.json") if self.cache_dir else None
        
        # Внутренние структуры данных
        self.user_profiles = {}
        self._concept_cache = {}
        self._concept_usage = defaultdict(int)
        self._user_interaction_patterns = defaultdict(lambda: defaultdict(int))
        self._adaptation_history = []
        
        # Параметры адаптации
        self.adaptation_params = {
            "min_user_interactions": 5,      # Минимальное количество взаимодействий для адаптации
            "concept_threshold": 0.7,        # Порог для важности концепта
            "pattern_recognition_window": 7, # Окно для распознавания паттернов (дней)
            "adaptation_interval": 3600,     # Интервал адаптации (сек)
            "max_history_size": 1000,        # Максимальный размер истории
            "user_profile_expiration": 30    # Срок хранения профиля пользователя (дней)
        }
        
        # Загружаем данные
        self._load_user_profiles()
        self._load_concept_cache()
        
        # Инициализируем
        self._initialize()
        logger.info("AdaptationManager инициализирован")
    
    def _initialize(self):
        """Инициализирует внутренние компоненты менеджера адаптации."""
        logger.info("Инициализация менеджера адаптации...")
        try:
            self.initialized = True
            logger.info("Менеджер адаптации полностью инициализирован")
        except Exception as e:
            logger.error(f"Ошибка полной инициализации менеджера адаптации: {e}", exc_info=True)
            self.initialized = False
    
    def start(self):
        """Запускает фоновые процессы менеджера адаптации."""
        if not self.initialized:
            logger.error("Невозможно запустить неинициализированный менеджер адаптации")
            return False
        
        self.running = True
        self.stop_event.clear()
        
        # Запускаем фоновый анализ
        self.start_background_analysis()
        
        logger.info("Менеджер адаптации запущен")
        return True
    
    def stop(self):
        """Останавливает фоновые процессы менеджера адаптации."""
        self.running = False
        self.stop_event.set()
        logger.info("Менеджер адаптации остановлен")
    
    def initialize(self) -> bool:
        """Дополнительная инициализация после создания экземпляра."""
        # Здесь можно добавить дополнительную логику инициализации
        return self.initialized
    
    def start_background_analysis(self):
        """Запускает фоновый анализ для выявления паттернов адаптации."""
        if not self.running:
            logger.warning("Попытка запуска фонового анализа при остановленном менеджере адаптации")
            return
        
        # Здесь можно запустить фоновый поток для анализа
        logger.info("Фоновый анализ адаптации запущен")
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """
        Возвращает текущие метрики адаптации.
        
        Returns:
            Dict[str, Any]: Метрики адаптации
        """
        # Рассчитываем уровень адаптации
        total_users = len(self.user_profiles)
        active_users = sum(1 for profile in self.user_profiles.values() 
                          if self._is_user_active(profile))
        
        # Собираем статистику по отзывам
        feedback_stats = {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        for profile in self.user_profiles.values():
            if "feedback" in profile:
                for feedback in profile["feedback"]:
                    feedback_stats["total"] += 1
                    if feedback["sentiment"] > 0.6:
                        feedback_stats["positive"] += 1
                    elif feedback["sentiment"] < 0.4:
                        feedback_stats["negative"] += 1
                    else:
                        feedback_stats["neutral"] += 1
        
        # Рассчитываем эффективность адаптации
        adaptation_efficiency = 0.0
        if feedback_stats["total"] > 0:
            adaptation_efficiency = (feedback_stats["positive"] / feedback_stats["total"]) * 100
        
        # Определяем время последнего обновления
        last_update = "N/A"
        if self._adaptation_history:
            last_update = datetime.fromtimestamp(self._adaptation_history[-1]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        
        # Рассчитываем уровень адаптации без рекурсивного вызова
        adaptation_level = self._calculate_adaptation_level_directly(
            total_users, active_users, feedback_stats, adaptation_efficiency
        )
        
        return {
            "adaptation_level": adaptation_level,
            "total_users": total_users,
            "active_users": active_users,
            "feedback_stats": feedback_stats,
            "last_update": last_update,
            "user_patterns_analyzed": len(self._user_interaction_patterns),
            "adaptation_efficiency": adaptation_efficiency,
            "timestamp": time.time()
        }
    
    def _calculate_adaptation_level_directly(
        self, 
        total_users: int, 
        active_users: int, 
        feedback_stats: Dict[str, int], 
        adaptation_efficiency: float
    ) -> float:
        """
        Рассчитывает общий уровень адаптации системы без рекурсивных вызовов.
        
        Args:
            total_users: Общее количество пользователей
            active_users: Количество активных пользователей
            feedback_stats: Статистика по отзывам
            adaptation_efficiency: Эффективность адаптации
            
        Returns:
            float: Уровень адаптации (0.0-1.0)
        """
        # Веса для различных факторов
        weights = {
            "user_engagement": 0.3,
            "feedback_quality": 0.4,
            "concept_relevance": 0.3
        }
        
        # Рассчитываем компоненты
        user_engagement = min(1.0, active_users / max(1, total_users))
        
        feedback_quality = 0.0
        if feedback_stats["total"] > 0:
            feedback_quality = feedback_stats["positive"] / feedback_stats["total"]
        
        concept_relevance = self._calculate_concept_relevance()
        
        # Собираем общий показатель
        adaptation_level = (
            weights["user_engagement"] * user_engagement +
            weights["feedback_quality"] * feedback_quality +
            weights["concept_relevance"] * concept_relevance
        )
        
        return adaptation_level
    
    def _calculate_adaptation_level(self) -> float:
        """
        Рассчитывает общий уровень адаптации системы.
        
        Returns:
            float: Уровень адаптации (0.0-1.0)
        """
        # Получаем необходимые данные напрямую
        total_users = len(self.user_profiles)
        active_users = sum(1 for profile in self.user_profiles.values() 
                          if self._is_user_active(profile))
        
        # Собираем статистику по отзывам
        feedback_stats = {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        for profile in self.user_profiles.values():
            if "feedback" in profile:
                for feedback in profile["feedback"]:
                    feedback_stats["total"] += 1
                    if feedback["sentiment"] > 0.6:
                        feedback_stats["positive"] += 1
                    elif feedback["sentiment"] < 0.4:
                        feedback_stats["negative"] += 1
                    else:
                        feedback_stats["neutral"] += 1
        
        # Рассчитываем эффективность адаптации
        adaptation_efficiency = 0.0
        if feedback_stats["total"] > 0:
            adaptation_efficiency = (feedback_stats["positive"] / feedback_stats["total"]) * 100
        
        # Используем метод без рекурсии
        return self._calculate_adaptation_level_directly(
            total_users, active_users, feedback_stats, adaptation_efficiency
        )
    
    def _calculate_concept_relevance(self) -> float:
        """
        Рассчитывает релевантность используемых концептов.
        
        Returns:
            float: Уровень релевантности (0.0-1.0)
        """
        if not self._concept_usage:
            return 0.0
        
        # Находим самые используемые концепты
        total_usage = sum(self._concept_usage.values())
        if total_usage == 0:
            return 0.0
        
        # Рассчитываем распределение
        usage_values = list(self._concept_usage.values())
        usage_values.sort(reverse=True)
        
        # Если топ-3 концепта составляют более 70% использования, считаем это хорошим показателем
        top_3_usage = sum(usage_values[:3]) / total_usage if len(usage_values) >= 3 else sum(usage_values) / total_usage
        return min(1.0, top_3_usage / 0.7)
    
    def _is_user_active(self, profile: Dict[str, Any]) -> bool:
        """
        Проверяет, является ли пользователь активным.
        
        Args:
            profile: Профиль пользователя
            
        Returns:
            bool: Активен ли пользователь
        """
        last_active = profile.get("last_active", 0)
        return (time.time() - last_active) < (self.adaptation_params["user_profile_expiration"] * 86400)
    
    def record_user_interaction(self, user_id: str, query: str, response: str, context: Optional[Dict] = None) -> str:
        """
        Записывает взаимодействие пользователя для анализа.
        
        Args:
            user_id: ID пользователя
            query: Запрос пользователя
            response: Ответ системы
            context: Контекст взаимодействия
            
        Returns:
            str: ID взаимодействия
        """
        interaction_id = f"inter_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Создаем запись взаимодействия
        interaction = {
            "id": interaction_id,
            "user_id": user_id,
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "context": context or {}
        }
        
        # Обновляем профиль пользователя
        self._update_user_profile(user_id, interaction)
        
        # Анализируем концепты в запросе
        self._analyze_concepts(user_id, query)
        
        # Сохраняем изменения
        self._save_user_profiles()
        
        logger.debug(f"Взаимодействие пользователя записано: {interaction_id}")
        return interaction_id
    
    def record_user_feedback(self, user_id: str, interaction_id: str, feedback: str, sentiment: float) -> bool:
        """
        Записывает обратную связь пользователя.
        
        Args:
            user_id: ID пользователя
            interaction_id: ID взаимодействия
            feedback: Текст обратной связи
            sentiment: Оценка тональности (0.0-1.0)
            
        Returns:
            bool: Успешно ли записано
        """
        if user_id not in self.user_profiles:
            logger.warning(f"Попытка записи обратной связи для неизвестного пользователя: {user_id}")
            return False
        
        # Создаем запись обратной связи
        feedback_record = {
            "interaction_id": interaction_id,
            "feedback": feedback,
            "sentiment": sentiment,
            "timestamp": time.time()
        }
        
        # Добавляем в профиль пользователя
        if "feedback" not in self.user_profiles[user_id]:
            self.user_profiles[user_id]["feedback"] = []
        
        self.user_profiles[user_id]["feedback"].append(feedback_record)
        
        # Сохраняем изменения
        self._save_user_profiles()
        
        logger.debug(f"Обратная связь пользователя записана: {user_id} - {interaction_id}")
        return True
    
    def _update_user_profile(self, user_id: str, interaction: Dict[str, Any]):
        """
        Обновляет профиль пользователя на основе взаимодействия.
        
        Args:
            user_id: ID пользователя
            interaction: Данные взаимодействия
        """
        # Создаем профиль, если его нет
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "id": user_id,
                "preferences": {
                    "preferred_concepts": {},
                    "response_style": "neutral",
                    "interaction_frequency": "regular"
                },
                "interaction_history": [],
                "created_at": time.time(),
                "last_active": time.time()
            }
        
        # Обновляем историю взаимодействий
        self.user_profiles[user_id]["interaction_history"].append(interaction)
        
        # Ограничиваем размер истории
        if len(self.user_profiles[user_id]["interaction_history"]) > self.adaptation_params["max_history_size"]:
            self.user_profiles[user_id]["interaction_history"] = self.user_profiles[user_id]["interaction_history"][-self.adaptation_params["max_history_size"]:]
        
        # Обновляем время последней активности
        self.user_profiles[user_id]["last_active"] = time.time()
    
    def _analyze_concepts(self, user_id: str, query: str):
        """
        Анализирует концепты в запросе пользователя.
        
        Args:
            user_id: ID пользователя
            query: Текст запроса
        """
        concept = self._extract_concept_from_query(query)
        if concept:
            # Обновляем кэш концептов
            self._concept_cache[query] = concept
            self._concept_usage[concept] += 1
            
            # Обновляем профиль пользователя
            if user_id in self.user_profiles:
                if "preferred_concepts" not in self.user_profiles[user_id]["preferences"]:
                    self.user_profiles[user_id]["preferences"]["preferred_concepts"] = {}
                
                if concept not in self.user_profiles[user_id]["preferences"]["preferred_concepts"]:
                    self.user_profiles[user_id]["preferences"]["preferred_concepts"][concept] = 0
                
                self.user_profiles[user_id]["preferences"]["preferred_concepts"][concept] += 1
                
                # Анализируем паттерны взаимодействия
                self._analyze_interaction_pattern(user_id, concept)
    
    def _analyze_interaction_pattern(self, user_id: str, concept: str):
        """
        Анализирует паттерны взаимодействия пользователя.
        
        Args:
            user_id: ID пользователя
            concept: Концепт
        """
        # Создаем ключ для паттерна
        pattern_key = f"{user_id}_{concept}"
        
        # Обновляем счетчик
        self._user_interaction_patterns[pattern_key][time.time()] = 1
    
    def _extract_concept_from_query(self, query: str) -> Optional[str]:
        """
        Извлекает концепт из запроса пользователя с учетом контекста и весов.
        
        Args:
            query: Текст запроса
            
        Returns:
            Optional[str]: Идентификатор концепта или None
        """
        # Проверяем кэш
        if query in self._concept_cache:
            return self._concept_cache[query]
        
        # Список известных концептов с их синонимами и ключевыми словами
        concepts = [
            {
                "name": "нейроэстетика",
                "keywords": ["нейроэстетика", "нейроэстетический"],
                "weight": 1.0
            },
            {
                "name": "этика_искусственного_интеллекта",
                "keywords": ["этика ии", "этика искуственного интеллекта", "этические нормы ии"],
                "weight": 0.8
            },
            {
                "name": "автономия_человека",
                "keywords": ["автономия человека", "человеческая автономия", "свобода выбора"],
                "weight": 0.9
            },
            {
                "name": "когнитивные_библиотеки",
                "keywords": ["когнитивные библиотеки", "библиотеки знаний", "когнитивные структуры"],
                "weight": 0.7
            },
            {
                "name": "нейроморфные_системы",
                "keywords": ["нейроморфные системы", "нейроморфный", "нейроморфные чипы", "нейроморфные вычисления"],
                "weight": 1.0
            },
            {
                "name": "эмоциональный_интеллект",
                "keywords": ["эмоциональный интеллект", "эмоциональный ии", "распознавание эмоций"],
                "weight": 0.9
            },
            {
                "name": "квантовые_вычисления",
                "keywords": ["квантовые вычисления", "квантовый компьютер", "квантовые алгоритмы"],
                "weight": 0.8
            },
            {
                "name": "нейронные_сети",
                "keywords": ["нейронные сети", "глубокое обучение", "искусственные нейронные сети"],
                "weight": 0.9
            },
            {
                "name": "машинное_обучение",
                "keywords": ["машинное обучение", "обучение с учителем", "обучение без учителя"],
                "weight": 0.8
            },
            {
                "name": "обработка_естественного_языка",
                "keywords": ["обработка естественного языка", "нлп", "natural language processing"],
                "weight": 0.9
            }
        ]
        
        # Приводим запрос к нижнему регистру
        query_lower = query.lower()
        
        # Подсчитываем совпадения для каждого концепта
        concept_scores = {}
        
        for concept in concepts:
            score = 0.0
            exact_match = False
            
            # Проверяем точные совпадения (дают максимальный балл)
            for keyword in concept["keywords"]:
                if keyword in query_lower:
                    # Точное совпадение дает полный вес
                    score += concept["weight"]
                    exact_match = True
            
            # Если есть точное совпадение, проверяем контекстные ключевые слова
            if exact_match:
                # Добавляем баллы за связанные слова в контексте
                context_keywords = {
                    "нейроморфные_системы": ["мозг", "нейрон", "нейроморфный", "биоморфный", "нейрокомпьютер"],
                    "этика_искусственного_интеллекта": ["норма", "мораль", "принцип", "ответственность", "право"],
                    "нейронные_сети": ["архитектура", "обучение", "веса", "сверточные", "рекуррентные"]
                }
                
                if concept["name"] in context_keywords:
                    for ctx_word in context_keywords[concept["name"]]:
                        if ctx_word in query_lower:
                            score += 0.2  # Добавляем баллы за контекст
            
            # Сохраняем результат
            if score > 0:
                concept_scores[concept["name"]] = score
        
        # Выбираем концепт с наибольшим баллом
        if concept_scores:
            best_concept = max(concept_scores.items(), key=lambda x: x[1])[0]
            self._concept_cache[query] = best_concept
            return best_concept
        
        return None
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Возвращает предпочтения пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Dict[str, Any]: Предпочтения пользователя
        """
        if user_id not in self.user_profiles:
            return {
                "preferred_concepts": {},
                "response_style": "neutral",
                "interaction_frequency": "regular"
            }
        
        return self.user_profiles[user_id].get("preferences", {
            "preferred_concepts": {},
            "response_style": "neutral",
            "interaction_frequency": "regular"
        })
    
    def adapt_response(self, user_id: str, response: str, context: Optional[Dict] = None) -> str:
        """
        Адаптирует ответ системы под пользователя.
        
        Args:
            user_id: ID пользователя
            response: Исходный ответ
            context: Контекст
            
        Returns:
            str: Адаптированный ответ
        """
        # Получаем предпочтения пользователя
        preferences = self.get_user_preferences(user_id)
        
        # Адаптируем ответ в зависимости от предпочтений
        adapted_response = response
        
        # Изменяем стиль ответа
        if preferences.get("response_style") == "formal":
            # Сделать ответ более формальным
            pass
        elif preferences.get("response_style") == "casual":
            # Сделать ответ более неформальным
            pass
        
        # Добавляем релевантные концепты
        preferred_concepts = preferences.get("preferred_concepts", {})
        if preferred_concepts:
            top_concept = max(preferred_concepts.items(), key=lambda x: x[1])[0]
            # Можно добавить упоминание концепта в ответ
            adapted_response += f" [Контекст: {top_concept.replace('_', ' ')}]"
        
        return adapted_response
    
    def _load_user_profiles(self):
        """Загружает профили пользователей из файла."""
        if not self.user_profiles_file or not os.path.exists(self.user_profiles_file):
            return
        
        try:
            with open(self.user_profiles_file, 'r', encoding='utf-8') as f:
                self.user_profiles = json.load(f)
            logger.info(f"Загружено {len(self.user_profiles)} профилей пользователей")
        except Exception as e:
            logger.error(f"Ошибка загрузки профилей пользователей: {e}")
    
    def _save_user_profiles(self):
        """Сохраняет профили пользователей в файл."""
        if not self.user_profiles_file:
            return
        
        try:
            with open(self.user_profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения профилей пользователей: {e}")
    
    def _load_concept_cache(self):
        """Загружает кэш концептов из файла."""
        if not self.concept_cache_file or not os.path.exists(self.concept_cache_file):
            return
        
        try:
            with open(self.concept_cache_file, 'r', encoding='utf-8') as f:
                self._concept_cache = json.load(f)
            logger.info(f"Загружен кэш концептов с {len(self._concept_cache)} записями")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша концептов: {e}")
    
    def _save_concept_cache(self):
        """Сохраняет кэш концептов в файл."""
        if not self.concept_cache_file:
            return
        
        try:
            with open(self.concept_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._concept_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша концептов: {e}")
    
    def set_default_profile_type(self, profile_type: str):
        """
        Устанавливает тип профиля по умолчанию.
        
        Args:
            profile_type: Тип профиля (например, 'standard', 'advanced', 'beginner')
        """
        self.adaptation_params["default_profile_type"] = profile_type
        logger.info(f"Установлен тип профиля по умолчанию: {profile_type}")

    def set_default_detail_level(self, detail_level: int):
        """
        Устанавливает уровень детализации по умолчанию.
        
        Args:
            detail_level: Уровень детализации (1-5)
        """
        self.adaptation_params["default_detail_level"] = detail_level
        logger.info(f"Установлен уровень детализации по умолчанию: {detail_level}")

    def set_emotional_tone(self, emotional_tone: str):
        """
        Устанавливает эмоциональный тон по умолчанию.
        
        Args:
            emotional_tone: Эмоциональный тон
        """
        self.adaptation_params["emotional_tone"] = emotional_tone
        logger.info(f"Установлен эмоциональный тон по умолчанию: {emotional_tone}")

    def set_knowledge_level(self, knowledge_level: str):
        """
        Устанавливает уровень знаний по умолчанию.
        
        Args:
            knowledge_level: Уровень знаний
        """
        self.adaptation_params["knowledge_level"] = knowledge_level
        logger.info(f"Установлен уровень знаний по умолчанию: {knowledge_level}")

    def set_cultural_context(self, region: str, communication_style: str):
        """
        Устанавливает культурный контекст по умолчанию.
        
        Args:
            region: Регион
            communication_style: Стиль общения
        """
        self.adaptation_params["cultural_context"] = {
            "region": region,
            "communication_style": communication_style
        }
        logger.info(f"Установлен культурный контекст по умолчанию: регион={region}, стиль общения={communication_style}")

    def set_learning_preferences(self, learning_style: str, learning_speed: float):
        """
        Устанавливает предпочтения в обучении по умолчанию.
        
        Args:
            learning_style: Стиль обучения
            learning_speed: Скорость обучения
        """
        self.adaptation_params["learning_preferences"] = {
            "learning_style": learning_style,
            "learning_speed": learning_speed
        }
        logger.info(f"Установлены предпочтения в обучении по умолчанию: стиль={learning_style}, скорость={learning_speed}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье системы адаптации.
        
        Returns:
            Dict: Отчет о здоровье
        """
        try:
            # Рассчитываем общий показатель здоровья
            health_score = 100.0
            
            # Учитываем количество пользователей
            total_users = len(self.user_profiles)
            if total_users < 10:
                health_score -= min(30, (10 - total_users) * 3)
            
            # Учитываем активность пользователей
            active_users = sum(1 for profile in self.user_profiles.values() 
                             if self._is_user_active(profile))
            active_ratio = active_users / total_users if total_users > 0 else 0
            if active_ratio < 0.3:
                health_score -= 25
            elif active_ratio < 0.5:
                health_score -= 15
            
            # Учитываем обратную связь
            feedback_count = 0
            positive_feedback = 0
            
            for profile in self.user_profiles.values():
                if "feedback" in profile:
                    feedback_count += len(profile["feedback"])
                    positive_feedback += sum(1 for f in profile["feedback"] if f["sentiment"] > 0.6)
            
            if feedback_count > 0:
                positive_ratio = positive_feedback / feedback_count
                if positive_ratio < 0.4:
                    health_score -= 20
                elif positive_ratio < 0.6:
                    health_score -= 10
            
            # Анализируем проблемы
            problem_areas = []
            if total_users < 10:
                problem_areas.append("Недостаточно пользователей для адаптации")
            
            if active_ratio < 0.3:
                problem_areas.append("Низкая активность пользователей")
            
            if feedback_count > 0 and positive_ratio < 0.4:
                problem_areas.append("Низкий уровень положительной обратной связи")
            
            # Формируем рекомендации
            recommendations = []
            if total_users < 10:
                recommendations.append("Привлеките больше пользователей для эффективной адаптации")
            
            if active_ratio < 0.3:
                recommendations.append("Увеличьте вовлеченность пользователей через персонализированные функции")
            
            if feedback_count > 0 and positive_ratio < 0.4:
                recommendations.append("Анализируйте причины низкой удовлетворенности пользователей")
            
            if not recommendations:
                recommendations.append("Система адаптации работает стабильно. Продолжайте сбор данных для улучшения точности адаптации")
            
            return {
                "health_score": max(0, min(100, health_score)),
                "total_users": total_users,
                "active_users": active_users,
                "feedback_count": feedback_count,
                "problem_areas": problem_areas,
                "recommendations": recommendations,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о здоровье системы адаптации: {e}", exc_info=True)
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": time.time()
            }