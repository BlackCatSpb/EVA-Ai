"""Базовая функциональность адаптационного менеджера CogniFlex"""
import os
import logging
import time
import threading
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import re
import hashlib
import json


logger = logging.getLogger("cogniflex.adaptation.core")

# Импортируем классы профилей
from .adaptation_profiles import UserFeedback, UserProfile

class AdaptationManager:
    """Основной менеджер адаптации системы CogniFlex."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует менеджер адаптации.
        
        Args:
            brain: Ссылка на ядро CogniFlex (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        
        # Создаем директорию кэша
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "adaptation.db") if self.cache_dir else None
        
        # Профили пользователей
        self.user_profiles: Dict[str, UserProfile] = {}
        self.profile_lock = threading.Lock()
        
        # Фидбэк
        self.feedback_history: List[UserFeedback] = []
        self.feedback_lock = threading.Lock()
        
        # Статистика адаптации
        self.stats = {
            "total_users": 0,
            "active_users": 0,
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "neutral_feedback": 0,
            "last_update": 0
        }
        
        # Дополнительные метрики
        self.adaptation_level = 0.0
        self.user_patterns_analyzed = 0
        self.adaptation_efficiency = 0.0
        
        logger.debug("AdaptationManager инициализирован (базовая структура)")
    
    def initialize(self) -> bool:
        """
        Полностью инициализирует менеджер адаптации.
        
        Returns:
            bool: Успешно ли инициализировано
        """
        try:
            logger.info("Полная инициализация менеджера адаптации...")
            
            # Инициализируем базу данных
            self._init_db()
            
            # Загружаем данные
            self._load_profiles()
            self._load_feedback()
            
            # Инициализируем фоновые процессы
            self._init_background_processes()
            
            self.initialized = True
            logger.info("Менеджер адаптации успешно инициализирован")
            return True
        except Exception as e:
            logger.error(f"Ошибка полной инициализации AdaptationManager: {e}", exc_info=True)
            return False
    
    def _init_db(self):
        """Инициализирует базу данных для адаптации."""
        if not self.db_path:
            return
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Таблица профилей пользователей
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences TEXT NOT NULL,
                interaction_history TEXT NOT NULL,
                adaptation_level REAL NOT NULL,
                last_updated REAL NOT NULL,
                learning_style TEXT NOT NULL,
                knowledge_level REAL NOT NULL,
                response_preferences TEXT NOT NULL,
                cultural_profile TEXT NOT NULL
            )
            """)
            
            # Таблица фидбэка
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                feedback_text TEXT NOT NULL,
                timestamp REAL NOT NULL,
                context TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """)
            
            # Таблица для хранения аналитики адаптации
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptation_analytics (
                timestamp REAL PRIMARY KEY,
                total_users INTEGER NOT NULL,
                active_users INTEGER NOT NULL,
                total_feedback INTEGER NOT NULL,
                positive_feedback INTEGER NOT NULL,
                negative_feedback INTEGER NOT NULL,
                neutral_feedback INTEGER NOT NULL
            )
            """)
            
            conn.commit()
            conn.close()
            logger.debug("База данных адаптации инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных адаптации: {e}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает новое соединение с базой данных."""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def _load_profiles(self):
        """Загружает профили пользователей из базы данных."""
        if not self.db_path:
            return
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM user_profiles")
            for row in cursor.fetchall():
                user_id = row[0]
                profile = UserProfile(
                    user_id=user_id,
                    preferences=json.loads(row[1]) if row[1] else {},
                    interaction_history=json.loads(row[2]),
                    adaptation_level=row[3],
                    last_updated=row[4],
                    learning_style=row[5],
                    knowledge_level=row[6],
                    response_preferences=json.loads(row[7]),
                    cultural_profile=json.loads(row[8])
                )
                self.user_profiles[user_id] = profile
            
            # Обновляем статистику
            self._update_user_statistics()
            
            logger.debug(f"Загружено {len(self.user_profiles)} профилей пользователей")
        except Exception as e:
            logger.error(f"Ошибка загрузки профилей пользователей: {e}")
        finally:
            conn.close()
    
    def _save_profiles(self):
        """Сохраняет профили пользователей в базу данных."""
        if not self.db_path:
            return
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Очищаем таблицу
            cursor.execute("DELETE FROM user_profiles")
            
            # Сохраняем профили
            for profile in self.user_profiles.values():
                cursor.execute("""
                INSERT INTO user_profiles (
                    user_id, preferences, interaction_history, adaptation_level, 
                    last_updated, learning_style, knowledge_level, 
                    response_preferences, cultural_profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    json.dumps(profile.preferences),
                    json.dumps(profile.interaction_history),
                    profile.adaptation_level,
                    profile.last_updated,
                    profile.learning_style,
                    profile.knowledge_level,
                    json.dumps(profile.response_preferences),
                    json.dumps(profile.cultural_profile)
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка сохранения профилей пользователей: {e}")
    
    def _load_feedback(self):
        """Загружает историю фидбэка из базы данных."""
        if not self.db_path:
            return
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM feedback")
            for row in cursor.fetchall():
                feedback = UserFeedback(
                    id=row[0],
                    user_id=row[1],
                    query=row[2],
                    response=row[3],
                    feedback_type=row[4],
                    feedback_text=row[5],
                    timestamp=row[6],
                    context=json.loads(row[7]),
                    metadata=json.loads(row[8])
                )
                self.feedback_history.append(feedback)
            
            # Обновляем статистику
            self._update_feedback_statistics()
            
            logger.debug(f"Загружено {len(self.feedback_history)} записей фидбэка")
        except Exception as e:
            logger.error(f"Ошибка загрузки фидбэка: {e}")
        finally:
            conn.close()
    
    def _save_feedback(self):
        """Сохраняет историю фидбэка в базу данных."""
        if not self.db_path:
            return
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Очищаем таблицу
            cursor.execute("DELETE FROM feedback")
            
            # Сохраняем фидбэк
            for feedback in self.feedback_history:
                cursor.execute("""
                INSERT INTO feedback (
                    id, user_id, query, response, feedback_type, 
                    feedback_text, timestamp, context, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.id,
                    feedback.user_id,
                    feedback.query,
                    feedback.response,
                    feedback.feedback_type,
                    feedback.feedback_text,
                    feedback.timestamp,
                    json.dumps(feedback.context),
                    json.dumps(feedback.metadata)
                ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения фидбэка: {e}")
        finally:
            conn.close()
    
    def _update_user_statistics(self):
        """Обновляет статистику по пользователям."""
        current_time = time.time()
        active_threshold = 86400  # 24 часа
        
        active_users = 0
        for profile in self.user_profiles.values():
            if current_time - profile.last_updated < active_threshold:
                active_users += 1
        
        self.stats["total_users"] = len(self.user_profiles)
        self.stats["active_users"] = active_users
        self.stats["last_update"] = current_time
    
    def _update_feedback_statistics(self):
        """Обновляет статистику по фидбэку."""
        self.stats["total_feedback"] = len(self.feedback_history)
        
        # Сбрасываем счетчики
        self.stats["positive_feedback"] = 0
        self.stats["negative_feedback"] = 0
        self.stats["neutral_feedback"] = 0
        
        # Подсчитываем типы фидбэка
        for feedback in self.feedback_history:
            if feedback.feedback_type == "positive":
                self.stats["positive_feedback"] += 1
            elif feedback.feedback_type == "negative":
                self.stats["negative_feedback"] += 1
            else:
                self.stats["neutral_feedback"] += 1
        
        self.stats["last_update"] = time.time()
        
        # Обновляем дополнительные метрики
        total = self.stats["total_feedback"]
        if total > 0:
            self.adaptation_level = self.stats["positive_feedback"] / total
            self.adaptation_efficiency = self.adaptation_level
        else:
            self.adaptation_level = 0.0
            self.adaptation_efficiency = 0.0
        
        self.user_patterns_analyzed = len(self.user_profiles)
    
    def _save_analytics_snapshot(self):
        """Сохраняет снимок статистики адаптации для анализа трендов."""
        if not self.db_path:
            return
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO adaptation_analytics(
                timestamp, total_users, active_users, total_feedback,
                positive_feedback, negative_feedback, neutral_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                self.stats["total_users"],
                self.stats["active_users"],
                self.stats["total_feedback"],
                self.stats["positive_feedback"],
                self.stats["negative_feedback"],
                self.stats["neutral_feedback"]
            ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения снимка статистики адаптации: {e}")
        finally:
            conn.close()
    
    def _init_background_processes(self):
        """Инициализирует фоновые процессы менеджера адаптации."""
        # Создаем поток для фонового анализа
        self.analysis_thread = None
        logger.debug("Фоновые процессы адаптации инициализированы")
    
    def apply_adaptation_insights(self):
        """Применяет инсайты адаптации для улучшения системы."""
        try:
            # Обновляем статистику
            self._update_user_statistics()
            self._update_feedback_statistics()
            
            # Анализируем паттерны пользователей
            self._analyze_user_patterns()
            
            logger.debug("Инсайты адаптации применены")
        except Exception as e:
            logger.error(f"Ошибка применения инсайтов адаптации: {e}")
    
    def _analyze_user_patterns(self):
        """Анализирует паттерны поведения пользователей."""
        try:
            with self.profile_lock:
                # Простой анализ активности пользователей
                current_time = time.time()
                for profile in self.user_profiles.values():
                    # Обновляем уровень адаптации на основе активности
                    time_since_update = current_time - profile.last_updated
                    if time_since_update < 3600:  # Активен в последний час
                        profile.adaptation_level = min(1.0, profile.adaptation_level + 0.01)
                    elif time_since_update > 86400:  # Неактивен более суток
                        profile.adaptation_level = max(0.0, profile.adaptation_level - 0.005)
            
            logger.debug("Анализ паттернов пользователей завершен")
        except Exception as e:
            logger.error(f"Ошибка анализа паттернов пользователей: {e}")
    
    def start(self):
        """Запускает менеджер адаптации."""
        if not self.initialized:
            logger.warning("Попытка запуска неинициализированного AdaptationManager")
            if not self.initialize():
                logger.error("Не удалось инициализировать AdaptationManager перед запуском")
                return
        
        if self.running:
            return
            
        self.running = True
        self.stop_event.clear()
        
        # Запускаем фоновый анализ
        self.start_background_analysis()
        
        logger.info("AdaptationManager запущен")
    
    def start_background_analysis(self):
        """Запускает фоновый анализ адаптации."""
        if not self.running:
            logger.warning("Попытка запуска фонового анализа при остановленном AdaptationManager")
            return
            
        if self.analysis_thread and self.analysis_thread.is_alive():
            logger.debug("Фоновый анализ адаптации уже запущен")
            return
            
        interval = 3600  # Интервал анализа в секундах (по умолчанию 1 час)
        
        def analysis_loop():
            nonlocal interval  # Используем нелокальную переменную interval
            
            while not self.stop_event.is_set():
                try:
                    # Применяем инсайты
                    self.apply_adaptation_insights()
                    
                    # Сохраняем снимок статистики
                    self._save_analytics_snapshot()
                    
                    # Ждем до следующего анализа
                    self.stop_event.wait(interval)
                except Exception as e:
                    logger.error(f"Ошибка в фоновом анализе адаптации: {e}")
                    # Пауза перед повторной попыткой
                    self.stop_event.wait(60)
        
        # Запускаем в отдельном потоке
        self.analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        self.analysis_thread.start()
        logger.debug(f"Фоновый анализ адаптации запущен (интервал: {interval} сек)")
    
    def stop(self):
        """Останавливает менеджер адаптации."""
        if not self.running:
            return
            
        self.running = False
        self.stop_event.set()
        
        # Сохраняем данные перед остановкой
        self._save_profiles()
        self._save_feedback()
        
        logger.info("AdaptationManager остановлен")
    
    def close(self):
        """Закрывает менеджер адаптации и освобождает ресурсы."""
        self.stop()
        
        logger.info("AdaptationManager закрыт")

def _extract_concept_from_query(self, query: str) -> Optional[str]:
    """Извлекает концепт из запроса пользователя с использованием полноценной NLP обработки."""
    # Проверяем кэш для этого запроса
    query_hash = hashlib.md5(query.lower().encode()).hexdigest()
    if not hasattr(self, '_concept_cache'):
        self._concept_cache = {}
    if query_hash in self._concept_cache:
        return self._concept_cache[query_hash]
    
    logger.debug(f"Анализ запроса для извлечения концептов: '{query}'")
    
    # 1. Проверяем доступность NLP-процессора
    if self.brain and hasattr(self.brain, 'nlp_processor'):
        try:
            # Обрабатываем текст с помощью NLP-процессора
            analysis = self.brain.nlp_processor.process_text(query)
            
            # 2. Извлекаем ключевые концепты из анализа
            candidate_concepts = []
            
            # Добавляем именованные сущности
            for entity in analysis.entities:
                if entity["type"] in ["CONCEPT", "TERM", "ORG", "PRODUCT", "TECH", "SCI", "PHIL"]:
                    candidate_concepts.append({
                        "concept": entity["text"],
                        "type": entity["type"],
                        "score": entity["confidence"]
                    })
            
            # Добавляем ключевые слова из анализа
            for keyword in analysis.keywords:
                if keyword["relevance"] > 0.6:
                    candidate_concepts.append({
                        "concept": keyword["text"],
                        "type": "KEYWORD",
                        "score": keyword["relevance"]
                    })
            
            # 3. Проверяем концепты в графе знаний
            if self.brain and hasattr(self.brain, 'knowledge_graph') and candidate_concepts:
                valid_concepts = []
                
                for concept_data in candidate_concepts:
                    concept = concept_data["concept"]
                    
                    # Проверяем, существует ли концепт в графе знаний
                    node = self.brain.knowledge_graph.get_node_by_name(concept)
                    if node:
                        valid_concepts.append({
                            "concept": concept,
                            "type": concept_data["type"],
                            "score": concept_data["score"] * 1.2,  # Увеличиваем вес для концептов из графа
                            "in_knowledge_graph": True
                        })
                    else:
                        # Проверяем похожие концепты
                        similar_nodes = self.brain.knowledge_graph.find_similar_nodes(concept, threshold=0.75)
                        if similar_nodes:
                            # Берем наиболее похожий узел
                            most_similar = max(similar_nodes, key=lambda x: x[1])
                            valid_concepts.append({
                                "concept": most_similar[0],
                                "type": concept_data["type"],
                                "score": concept_data["score"] * 0.8,  # Снижаем вес для похожих концептов
                                "in_knowledge_graph": True,
                                "similar_to": concept
                            })
                        else:
                            valid_concepts.append(concept_data)
                
                # 4. Сортируем концепты по релевантности
                valid_concepts.sort(key=lambda x: x["score"], reverse=True)
                
                # 5. Возвращаем наиболее релевантный концепт
                if valid_concepts:
                    result = valid_concepts[0]["concept"].replace(" ", "_").lower()
                    self._concept_cache[query_hash] = result
                    return result
            
        except Exception as e:
            logger.error(f"Ошибка при NLP-анализе запроса: {e}")
    
    # Резервные методы, если NLP-обработка недоступна
    # Метод 1: Используем список известных концептов из системы
    if self.brain and hasattr(self.brain, 'contradiction_manager'):
        known_concepts = self.brain.contradiction_manager.get_known_concepts()
        query_lower = query.lower()
        
        for concept in known_concepts:
            normalized_concept = concept.replace("_", " ").lower()
            if normalized_concept in query_lower:
                result = concept
                self._concept_cache[query_hash] = result
                return result
    
    # Метод 2: Используем простой анализ через spaCy, если он доступен
    try:
        import spacy
        if not hasattr(self, 'nlp_spacy'):
            try:
                self.nlp_spacy = spacy.load("ru_core_news_sm")
            except (ImportError, OSError):
                try:
                    spacy.cli.download("ru_core_news_sm")
                    self.nlp_spacy = spacy.load("ru_core_news_sm")
                except:
                    self.nlp_spacy = None
        
        if self.nlp_spacy:
            doc = self.nlp_spacy(query)
            # Ищем составные именные группы
            noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 3]
            
            # Проверяем наиболее вероятные кандидаты
            if noun_phrases:
                # Берем самую длинную фразу (вероятно, наиболее специфичный концепт)
                candidate = max(noun_phrases, key=len)
                result = candidate.replace(" ", "_").lower()
                self._concept_cache[query_hash] = result
                return result
    except Exception as e:
        logger.debug(f"Не удалось использовать spaCy для анализа: {e}")
    
    # Метод 3: Используем TF-IDF для выделения ключевых терминов
    try:
        if self.brain and hasattr(self.brain, 'nlp_processor'):
            keywords = self.brain.nlp_processor.extract_keywords(query, top_n=3)
            if keywords:
                # Берем наиболее релевантное ключевое слово
                result = keywords[0][0].replace(" ", "_").lower()
                self._concept_cache[query_hash] = result
                return result
    except Exception as e:
        logger.debug(f"Не удалось использовать TF-IDF анализ: {e}")
    
    # 4. Если все методы не сработали, возвращаем None
    logger.debug(f"Не удалось определить концепт из запроса: '{query}'")
    self._concept_cache[query_hash] = None
    return None