"""
Модуль адаптации системы для CogniFlex
Управление профилями пользователей, обратная связь и адаптация поведения системы
"""

import logging
import time
import os
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict, is_dataclass
from collections import defaultdict, deque

logger = logging.getLogger("cogniflex.adaptation")


# ============================================================================
# Классы данных
# ============================================================================

@dataclass
class UserProfile:
    """Профиль пользователя для персонализации взаимодействия."""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_level: float = 0.5
    learning_style: str = "balanced"
    knowledge_level: float = 0.5
    response_preferences: Dict[str, float] = field(default_factory=lambda: {"formal": 0.5, "casual": 0.5})
    cultural_profile: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Создание из словаря."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class UserFeedback:
    """Запись обратной связи от пользователя."""
    id: str
    user_id: str
    query: str
    response: str
    feedback_type: str  # positive, negative, neutral
    feedback_text: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeedback':
        """Создание из словаря."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ============================================================================
# Менеджер адаптации
# ============================================================================

class AdaptationManager:
    """
    Основной класс управления адаптацией системы.
    Отвечает за персонализацию, обучение и адаптацию поведения системы.
    """
    
    def __init__(self, data_dir: str = "adaptation_data", brain=None):
        """
        Инициализация менеджера адаптации.
        
        Args:
            data_dir: Директория для хранения данных адаптации
            brain: Ссылка на ядро CogniFlex (опционально)
        """
        self.data_dir = data_dir
        self.brain = brain
        os.makedirs(data_dir, exist_ok=True)
        
        # Инициализация хранилищ
        self.user_profiles: Dict[str, UserProfile] = {}
        self.feedback_history: deque = deque(maxlen=10000)
        self.concept_cache: Dict[str, str] = {}
        self.concept_usage: defaultdict = defaultdict(int)
        
        # Параметры адаптации
        self.adaptation_params = {
            "default_profile_type": "standard",
            "default_detail_level": 3,
            "emotional_tone": "neutral",
            "knowledge_level": "intermediate",
            "cultural_context": {
                "region": "international",
                "communication_style": "professional"
            },
            "learning_preferences": {
                "learning_style": "balanced",
                "learning_speed": 1.0
            }
        }
        
        # Статистика
        self.stats = {
            "total_users": 0,
            "active_users": 0,
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "neutral_feedback": 0,
            "last_update": time.time(),
            "user_patterns_analyzed": 0,
            "adaptation_efficiency": 0.0,
            "adaptation_level": 0.0
        }
        
        # Фоновые процессы
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()
        self.initialized = False
        self.running = False
        
        # Загрузка данных
        self._load_data()
        
        # Запуск фонового анализа
        self._start_background_analysis()
        
        self.initialized = True
        logger.info("Менеджер адаптации инициализирован")
    
    def _load_data(self):
        """Загрузка данных из файлов."""
        try:
            # Загрузка профилей пользователей
            profiles_file = os.path.join(self.data_dir, "user_profiles.json")
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    for user_id, profile_data in profiles_data.items():
                        self.user_profiles[user_id] = UserProfile.from_dict(profile_data)
            
            # Загрузка кэша концептов
            cache_file = os.path.join(self.data_dir, "concept_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.concept_cache = cache_data.get("concepts", {})
                    usage_data = cache_data.get("usage", {})
                    self.concept_usage = defaultdict(int, usage_data)
            
            # Загрузка метрик
            metrics_file = os.path.join(self.data_dir, "adaptation_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
            
            logger.info(f"Данные адаптации загружены: {len(self.user_profiles)} пользователей")
            
        except FileNotFoundError:
            logger.debug("Файлы данных не найдены, создаются новые")
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных адаптации: {e}", exc_info=True)
    
    def _save_data(self):
        """Сохранение данных в файлы."""
        try:
            # Сохранение профилей пользователей
            profiles_file = os.path.join(self.data_dir, "user_profiles.json")
            profiles_data = {user_id: profile.to_dict() for user_id, profile in self.user_profiles.items()}
            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
            
            # Сохранение кэша концептов
            cache_file = os.path.join(self.data_dir, "concept_cache.json")
            cache_data = {
                "concepts": self.concept_cache,
                "usage": dict(self.concept_usage)
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            # Сохранение метрик
            metrics_file = os.path.join(self.data_dir, "adaptation_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            
            logger.debug("Данные адаптации сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения данных адаптации: {e}", exc_info=True)
    
    def _start_background_analysis(self):
        """Запуск фонового потока анализа."""
        if self._background_thread is None or not self._background_thread.is_alive():
            self._stop_background.clear()
            self._background_thread = threading.Thread(
                target=self._background_analysis_worker,
                name="AdaptationBackground",
                daemon=True
            )
            self._background_thread.start()
            self.running = True
            logger.info("Фоновый анализ адаптации запущен")
    
    def _background_analysis_worker(self):
        """Рабочий процесс фонового анализа."""
        while not self._stop_background.wait(300):  # Каждые 5 минут
            try:
                self._analyze_user_patterns()
                self._update_statistics()
                self._save_data()
            except Exception as e:
                logger.error(f"Ошибка в фоновом анализе: {e}", exc_info=True)
    
    def _analyze_user_patterns(self):
        """Анализ паттернов пользователей."""
        try:
            # Обновление адаптационных уровней
            for user_id, profile in self.user_profiles.items():
                if len(profile.interaction_history) > 10:
                    # Анализ истории взаимодействий
                    recent_interactions = profile.interaction_history[-50:]
                    
                    # Расчет адаптационного уровня
                    adaptation_score = 0.0
                    feedback_count = 0
                    for interaction in recent_interactions:
                        if 'feedback' in interaction:
                            feedback_score = interaction['feedback'].get('sentiment', 0.5)
                            adaptation_score += feedback_score
                            feedback_count += 1
                    
                    if feedback_count > 0:
                        profile.adaptation_level = min(1.0, adaptation_score / feedback_count)
            
            self.stats["user_patterns_analyzed"] += 1
            logger.debug("Анализ паттернов пользователей завершен")
            
        except Exception as e:
            logger.error(f"Ошибка анализа паттернов: {e}", exc_info=True)
    
    def _update_statistics(self):
        """Обновление статистики."""
        try:
            self.stats["total_users"] = len(self.user_profiles)
            self.stats["active_users"] = sum(
                1 for profile in self.user_profiles.values()
                if self._is_user_active(profile)
            )
            self.stats["last_update"] = time.time()
            
            # Расчет эффективности адаптации
            if self.stats["total_feedback"] > 0:
                self.stats["adaptation_efficiency"] = (
                    self.stats["positive_feedback"] / self.stats["total_feedback"]
                )
            
            # Расчет общего уровня адаптации
            if self.user_profiles:
                avg_adaptation = sum(
                    p.adaptation_level for p in self.user_profiles.values()
                ) / len(self.user_profiles)
                self.stats["adaptation_level"] = avg_adaptation
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}", exc_info=True)
    
    def _is_user_active(self, profile: UserProfile) -> bool:
        """Проверка активности пользователя."""
        if not profile.interaction_history:
            return False
        
        try:
            last_interaction = profile.interaction_history[-1]
            last_time = last_interaction.get('timestamp', 0)
            current_time = time.time()
            
            # Пользователь активен, если было взаимодействие за последние 7 дней
            return (current_time - last_time) < 7 * 24 * 3600
        except Exception:
            return False
    
    def initialize(self) -> bool:
        """Дополнительная инициализация после создания экземпляра."""
        return self.initialized
    
    def start(self) -> bool:
        """Запуск менеджера адаптации."""
        if not self.initialized:
            logger.error("Невозможно запустить неинициализированный менеджер адаптации")
            return False
        
        self._start_background_analysis()
        self.running = True
        logger.info("Менеджер адаптации запущен")
        return True
    
    def stop(self):
        """Остановка менеджера адаптации."""
        self.running = False
        self._stop_background.set()
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=10)
        logger.info("Менеджер адаптации остановлен")
    
    def shutdown(self):
        """Завершение работы менеджера адаптации."""
        self.stop()
        self._save_data()
        logger.info("Менеджер адаптации завершил работу")
    
    # ========================================================================
    # Методы работы с профилями пользователей
    # ========================================================================
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Получение профиля пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            UserProfile or None
        """
        return self.user_profiles.get(user_id)
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Обновление профиля пользователя.
        
        Args:
            user_id: ID пользователя
            updates: Обновления профиля
            
        Returns:
            bool: Успешно ли обновлено
        """
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    preferences={},
                    interaction_history=[],
                    adaptation_level=0.5,
                    learning_style="balanced",
                    knowledge_level=0.5,
                    response_preferences={"formal": 0.5, "casual": 0.5},
                    cultural_profile={}
                )
            
            profile = self.user_profiles[user_id]
            
            # Применение обновлений
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
                elif key == 'preferences' and isinstance(value, dict):
                    profile.preferences.update(value)
            
            profile.last_updated = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления профиля пользователя: {e}", exc_info=True)
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Получение предпочтений пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Dict[str, Any]: Предпочтения пользователя
        """
        profile = self.user_profiles.get(user_id)
        if profile:
            return profile.preferences
        return {
            "preferred_concepts": {},
            "response_style": "neutral",
            "interaction_frequency": "regular"
        }
    
    # ========================================================================
    # Методы записи взаимодействий
    # ========================================================================
    
    def record_user_interaction(self, user_id: str, query: str, response: str, 
                                context: Optional[Dict] = None) -> str:
        """
        Запись взаимодействия пользователя.
        
        Args:
            user_id: ID пользователя
            query: Запрос пользователя
            response: Ответ системы
            context: Контекст взаимодействия
            
        Returns:
            str: ID взаимодействия
        """
        try:
            interaction_id = f"inter_{int(time.time())}_{os.urandom(4).hex()}"
            
            # Создание записи взаимодействия
            interaction = {
                "id": interaction_id,
                "user_id": user_id,
                "query": query,
                "response": response,
                "timestamp": time.time(),
                "context": context or {}
            }
            
            # Обновление профиля пользователя
            self._update_user_profile_from_interaction(user_id, interaction)
            
            # Извлечение концепта
            concept = self._extract_concept_from_query(query)
            if concept:
                self.concept_usage[concept] += 1
            
            logger.debug(f"Взаимодействие записано: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Ошибка записи взаимодействия: {e}", exc_info=True)
            return f"error_{int(time.time())}"
    
    def record_user_feedback(self, user_id: str, interaction_id: str, 
                            feedback: str, sentiment: float) -> bool:
        """
        Запись обратной связи пользователя.
        
        Args:
            user_id: ID пользователя
            interaction_id: ID взаимодействия
            feedback: Текст обратной связи
            sentiment: Оценка тональности (0.0-1.0)
            
        Returns:
            bool: Успешно ли записано
        """
        try:
            feedback_record = UserFeedback(
                id=f"fb_{int(time.time())}_{os.urandom(4).hex()}",
                user_id=user_id,
                query="",
                response="",
                feedback_type="positive" if sentiment > 0.6 else "negative" if sentiment < 0.4 else "neutral",
                feedback_text=feedback,
                timestamp=time.time(),
                context={"interaction_id": interaction_id},
                metadata={"sentiment": sentiment}
            )
            
            self.feedback_history.append(feedback_record)
            
            # Обновление статистики
            self.stats["total_feedback"] += 1
            if sentiment > 0.6:
                self.stats["positive_feedback"] += 1
            elif sentiment < 0.4:
                self.stats["negative_feedback"] += 1
            else:
                self.stats["neutral_feedback"] += 1
            
            logger.debug(f"Обратная связь записана: {user_id} - {interaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка записи обратной связи: {e}", exc_info=True)
            return False
    
    def _update_user_profile_from_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """Обновление профиля пользователя на основе взаимодействия."""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    preferences={},
                    interaction_history=[],
                    adaptation_level=0.5,
                    learning_style="balanced",
                    knowledge_level=0.5,
                    response_preferences={"formal": 0.5, "casual": 0.5},
                    cultural_profile={}
                )
            
            profile = self.user_profiles[user_id]
            profile.interaction_history.append(interaction)
            
            # Ограничение размера истории
            max_history = 1000
            if len(profile.interaction_history) > max_history:
                profile.interaction_history = profile.interaction_history[-max_history:]
            
            profile.last_updated = time.time()
            
        except Exception as e:
            logger.error(f"Ошибка обновления профиля из взаимодействия: {e}", exc_info=True)
    
    # ========================================================================
    # Методы извлечения концептов
    # ========================================================================
    
    def _extract_concept_from_query(self, query: str) -> Optional[str]:
        """
        Извлечение концепта из запроса пользователя.
        
        Args:
            query: Текст запроса
            
        Returns:
            Optional[str]: Идентификатор концепта или None
        """
        try:
            # Проверка кэша
            query_lower = query.lower()
            if query_lower in self.concept_cache:
                return self.concept_cache[query_lower]
            
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
            
            # Кэширование результата
            if best_match:
                self.concept_cache[query_lower] = best_match
            
            return best_match if best_score > 0.5 else None
            
        except Exception as e:
            logger.error(f"Ошибка извлечения концепта: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # Методы получения метрик и статистики
    # ========================================================================
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик адаптации.
        
        Returns:
            Dict[str, Any]: Метрики адаптации
        """
        try:
            return {
                "adaptation_level": self.stats.get("adaptation_level", 0.0),
                "total_users": self.stats.get("total_users", 0),
                "active_users": self.stats.get("active_users", 0),
                "feedback_stats": {
                    "total": self.stats.get("total_feedback", 0),
                    "positive": self.stats.get("positive_feedback", 0),
                    "negative": self.stats.get("negative_feedback", 0),
                    "neutral": self.stats.get("neutral_feedback", 0)
                },
                "last_update": datetime.fromtimestamp(
                    self.stats.get("last_update", time.time())
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "user_patterns_analyzed": self.stats.get("user_patterns_analyzed", 0),
                "adaptation_efficiency": self.stats.get("adaptation_efficiency", 0.0),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка получения метрик адаптации: {e}", exc_info=True)
            return {"error": str(e), "timestamp": time.time()}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Получение отчета о здоровье системы адаптации.
        
        Returns:
            Dict: Отчет о здоровье
        """
        try:
            health_score = 100.0
            total_users = len(self.user_profiles)
            
            if total_users < 10:
                health_score -= min(30, (10 - total_users) * 3)
            
            active_users = sum(
                1 for profile in self.user_profiles.values()
                if self._is_user_active(profile)
            )
            active_ratio = active_users / total_users if total_users > 0 else 0
            
            if active_ratio < 0.3:
                health_score -= 25
            elif active_ratio < 0.5:
                health_score -= 15
            
            feedback_count = len(self.feedback_history)
            positive_feedback = sum(
                1 for f in self.feedback_history 
                if f.metadata.get("sentiment", 0) > 0.6
            )
            
            if feedback_count > 0:
                positive_ratio = positive_feedback / feedback_count
                if positive_ratio < 0.4:
                    health_score -= 20
                elif positive_ratio < 0.6:
                    health_score -= 10
            
            problem_areas = []
            recommendations = []
            
            if total_users < 10:
                problem_areas.append("Недостаточно пользователей для адаптации")
                recommendations.append("Привлеките больше пользователей для эффективной адаптации")
            
            if active_ratio < 0.3:
                problem_areas.append("Низкая активность пользователей")
                recommendations.append("Увеличьте вовлеченность пользователей через персонализированные функции")
            
            if feedback_count > 0 and positive_ratio < 0.4:
                problem_areas.append("Низкий уровень положительной обратной связи")
                recommendations.append("Анализируйте причины низкой удовлетворенности пользователей")
            
            if not recommendations:
                recommendations.append("Система адаптации работает стабильно")
            
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
            logger.error(f"Ошибка получения информации о здоровье системы: {e}", exc_info=True)
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    # ========================================================================
    # Методы настройки параметров
    # ========================================================================
    
    def set_default_profile_type(self, profile_type: str):
        """Установка типа профиля по умолчанию."""
        self.adaptation_params["default_profile_type"] = profile_type
        logger.info(f"Установлен тип профиля по умолчанию: {profile_type}")
    
    def set_default_detail_level(self, detail_level: int):
        """Установка уровня детализации по умолчанию."""
        self.adaptation_params["default_detail_level"] = max(1, min(5, detail_level))
        logger.info(f"Установлен уровень детализации по умолчанию: {detail_level}")
    
    def set_emotional_tone(self, emotional_tone: str):
        """Установка эмоционального тона по умолчанию."""
        self.adaptation_params["emotional_tone"] = emotional_tone
        logger.info(f"Установлен эмоциональный тон по умолчанию: {emotional_tone}")
    
    def set_knowledge_level(self, knowledge_level: str):
        """Установка уровня знаний по умолчанию."""
        self.adaptation_params["knowledge_level"] = knowledge_level
        logger.info(f"Установлен уровень знаний по умолчанию: {knowledge_level}")
    
    def set_cultural_context(self, region: str, communication_style: str):
        """Установка культурного контекста по умолчанию."""
        self.adaptation_params["cultural_context"] = {
            "region": region,
            "communication_style": communication_style
        }
        logger.info(f"Установлен культурный контекст: регион={region}, стиль={communication_style}")
    
    def set_learning_preferences(self, learning_style: str, learning_speed: float):
        """Установка предпочтений в обучении по умолчанию."""
        self.adaptation_params["learning_preferences"] = {
            "learning_style": learning_style,
            "learning_speed": max(0.1, min(2.0, learning_speed))
        }
        logger.info(f"Установлены предпочтения обучения: стиль={learning_style}, скорость={learning_speed}")


# ============================================================================
# Экспорт для совместимости
# ============================================================================

__all__ = [
    'AdaptationManager',
    'UserProfile',
    'UserFeedback'
]