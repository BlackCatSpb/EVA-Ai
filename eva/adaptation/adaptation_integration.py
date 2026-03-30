
"""Интеграционные функции адаптационного менеджера ЕВА"""
import os
import logging
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger("eva.adaptation.integration")

# Импортируем основной класс
from .adaptation_core import AdaptationManager
from .adaptation_profiles import UserProfile, UserFeedback

def get_user_profile(self, user_id: str) -> UserProfile:
    """
    Возвращает профиль пользователя.
    
    Args:
        user_id: Идентификатор пользователя
        
    Returns:
        UserProfile: Профиль пользователя
    """
    with self.profile_lock:
        if user_id not in self.user_profiles:
            # Создаем новый профиль
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences={
                    "response_length": "medium",
                    "formality_level": "neutral",
                    "preferred_domains": []
                },
                interaction_history=[],
                knowledge_level=0.5,
                learning_style="balanced",
                cultural_profile={
                    "language": "ru",
                    "cultural_norms": []
                }
            )
            # Сохраняем новый профиль
            self._save_profiles()
            self._update_user_statistics()
        
        return self.user_profiles[user_id]

def update_user_profile(self, user_id: str, updates: Dict[str, Any]):
    """
    Обновляет профиль пользователя.
    
    Args:
        user_id: Идентификатор пользователя
        updates: Обновления профиля
    """
    with self.profile_lock:
        profile = self.get_user_profile(user_id)
        
        # Обновляем поля
        if "preferences" in updates:
            profile.preferences.update(updates["preferences"])
        if "learning_style" in updates:
            profile.learning_style = updates["learning_style"]
        if "knowledge_level" in updates:
            profile.knowledge_level = updates["knowledge_level"]
        if "response_preferences" in updates:
            profile.response_preferences.update(updates["response_preferences"])
        if "cultural_profile" in updates:
            profile.cultural_profile.update(updates["cultural_profile"])
        
        profile.last_updated = time.time()
        
        # Сохраняем изменения
        self._save_profiles()
        self._update_user_statistics()

def record_feedback(self, user_id: str, query: str, response: str, 
                  feedback_type: str, feedback_text: str, context: Dict[str, Any] = None):
    """
    Записывает пользовательский фидбэк.
    
    Args:
        user_id: Идентификатор пользователя
        query: Запрос пользователя
        response: Ответ системы
        feedback_type: Тип фидбэка (positive, negative, neutral)
        feedback_text: Текст фидбэка
        context: Дополнительный контекст
    """
    with self.feedback_lock:
        # Генерируем уникальный ID
        feedback_id = f"fb_{hashlib.md5(f'{user_id}_{time.time()}'.encode()).hexdigest()[:12]}"
        
        # Создаем запись
        feedback = UserFeedback(
            id=feedback_id,
            user_id=user_id,
            query=query,
            response=response,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            context=context or {},
            metadata={"timestamp": time.time()}
        )
        
        # Добавляем в историю
        self.feedback_history.append(feedback)
        
        # Сохраняем
        self._save_feedback()
        
        # Обновляем статистику
        self._update_feedback_statistics()
        
        # Обновляем профиль пользователя
        self._update_profile_from_feedback(user_id, feedback)
        
        logger.debug(f"Записан фидбэк от пользователя {user_id}: {feedback_type}")

def _update_profile_from_feedback(self, user_id: str, feedback: UserFeedback):
    """Обновляет профиль пользователя на основе фидбэка."""
    profile = self.get_user_profile(user_id)
    
    # Анализируем фидбэк и обновляем профиль
    if feedback.feedback_type == "positive":
        profile.adaptation_level = min(1.0, profile.adaptation_level + 0.05)
    elif feedback.feedback_type == "negative":
        profile.adaptation_level = max(0.0, profile.adaptation_level - 0.1)
    
    # Добавляем в историю взаимодействий
    profile.interaction_history.append({
        "query": feedback.query,
        "response": feedback.response,
        "feedback": feedback.feedback_type,
        "timestamp": feedback.timestamp,
        "context": feedback.context
    })
    
    # Ограничиваем историю
    if len(profile.interaction_history) > 100:
        profile.interaction_history = profile.interaction_history[-100:]
    
    # Обновляем статистику
    self._update_user_statistics()
    self._update_feedback_statistics()
    
    # Сохраняем изменения
    self._save_profiles()

def analyze_user_patterns(self):
    """Анализирует паттерны пользователей для улучшения адаптации."""
    try:
        # Анализируем паттерны на основе фидбэка
        positive_count = sum(1 for fb in self.feedback_history if fb.feedback_type == "positive")
        negative_count = sum(1 for fb in self.feedback_history if fb.feedback_type == "negative")
        total = len(self.feedback_history)
        
        if total > 0:
            success_rate = positive_count / total
            self.adaptation_level = success_rate
            self.adaptation_efficiency = success_rate
            
            # Обновляем уровень адаптации профилей
            for profile in self.user_profiles.values():
                profile.adaptation_level = min(1.0, profile.adaptation_level * 0.9 + success_rate * 0.1)
            
            logger.info(f"Анализ паттернов завершен. Уровень адаптации: {self.adaptation_level:.2f}")
        
        return {
            "adaptation_level": self.adaptation_level,
            "success_rate": self.adaptation_efficiency,
            "total_interactions": total,
            "positive_feedback": positive_count,
            "negative_feedback": negative_count
        }
    except Exception as e:
        logger.error(f"Ошибка анализа паттернов пользователей: {e}")
        return {
            "adaptation_level": self.adaptation_level,
            "error": str(e)
        }

def export_adaptation_data(self, file_path: str, days: int = 30) -> bool:
    """
    Экспортирует данные адаптации в файл.
    
    Args:
        file_path: Путь к файлу для экспорта
        days: Количество дней для экспорта
        
    Returns:
        bool: Успешно ли экспортировано
    """
    try:
        # Собираем данные для экспорта
        export_data = {
            "metadata": {
                "export_time": time.time(),
                "period_days": days,
                "format_version": "1.0"
            },
            "user_profiles": [profile.to_dict() for profile in self.user_profiles.values()],
            "feedback_history": [fb.to_dict() for fb in self.get_feedback_history(days=days)],
            "system_report": self.get_system_adaptation_report()
        }
        
        # Сохраняем в JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Данные адаптации экспортированы в {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка экспорта данных адаптации: {e}")
        return False

def get_feedback_history(self, days: int = 30) -> List[UserFeedback]:
    """
    Возвращает историю фидбэка за указанный период.
    
    Args:
        days: Количество дней
        
    Returns:
        List[UserFeedback]: История фидбэка
    """
    cutoff = time.time() - (days * 86400)
    return [fb for fb in self.feedback_history if fb.timestamp >= cutoff]

def import_adaptation_data(self, file_path: str) -> bool:
    """
    Импортирует данные адаптации из файла.
    
    Args:
        file_path: Путь к файлу для импорта
        
    Returns:
        bool: Успешно ли импортировано
    """
    try:
        # Загружаем данные из JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Валидируем формат
        if "metadata" not in data or "format_version" not in data["metadata"]:
            logger.error("Неверный формат файла адаптации")
            return False
        
        # Импортируем профили пользователей
        with self.profile_lock:
            self.user_profiles = {}
            for profile_data in data.get("user_profiles", []):
                profile = UserProfile.from_dict(profile_data)
                self.user_profiles[profile.user_id] = profile
        
        # Импортируем фидбэк
        with self.feedback_lock:
            self.feedback_history = []
            for feedback_data in data.get("feedback_history", []):
                feedback = UserFeedback.from_dict(feedback_data)
                self.feedback_history.append(feedback)
        
        # Обновляем статистику
        self._update_user_statistics()
        self._update_feedback_statistics()
        
        logger.info(f"Данные адаптации импортированы из {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка импорта данных адаптации: {e}")
        return False

def integrate_with_knowledge_graph(self, user_id: str):
    """
    Интегрирует данные адаптации с графом знаний.
    
    Args:
        user_id: ID пользователя
    """
    if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
        logger.warning("KnowledgeGraph недоступен для интеграции с адаптацией")
        return
    
    profile = self.get_user_profile(user_id)
    
    # Добавляем информацию о пользователе в граф знаний
    self.brain.knowledge_graph.add_node(
        node_id=f"user_{user_id}",
        node_type="user_profile",
        domain="user",
        metadata={
            "user_id": user_id,
            "knowledge_level": profile.knowledge_level,
            "learning_style": profile.learning_style,
            "adaptation_level": profile.adaptation_level
        }
    )
    
    # Добавляем информацию о предыдущих знаниях
    if "known_concepts" in profile.preferences:
        for concept in profile.preferences["known_concepts"]:
            self.brain.knowledge_graph.add_node(
                node_id=f"user_{user_id}_knowledge_{concept}",
                node_type="user_knowledge",
                domain="user",
                metadata={"user_id": user_id}
            )
            self.brain.knowledge_graph.add_edge(
                f"user_{user_id}", 
                f"user_{user_id}_knowledge_{concept}",
                relationship="knows"
            )
    
    logger.info(f"Интегрированы данные о предыдущих знаниях для пользователя {user_id}")

def get_cultural_adaptation(self, user_id: str) -> Dict[str, Any]:
    """
    Возвращает данные о культурной адаптации для пользователя.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Dict: Данные о культурной адаптации
    """
    profile = self.get_user_profile(user_id)
    cultural_profile = profile.cultural_profile
    
    return {
        "language": cultural_profile.get("language", "ru"),
        "communication_style": cultural_profile.get("communication_style", "neutral"),
        "cultural_norms": cultural_profile.get("cultural_norms", []),
        "preferred_examples": cultural_profile.get("preferred_examples", [])
    }

def get_adaptation_progress(self, user_id: str) -> Dict[str, Any]:
    """
    Возвращает прогресс адаптации для пользователя.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Dict: Прогресс адаптации
    """
    profile = self.get_user_profile(user_id)
    
    # Анализируем историю взаимодействий
    positive_count = sum(1 for item in profile.interaction_history 
                       if item.get("feedback") == "positive")
    negative_count = sum(1 for item in profile.interaction_history 
                       if item.get("feedback") == "negative")
    total = len(profile.interaction_history)
    
    # Вычисляем прогресс
    progress = {
        "total_interactions": total,
        "positive_feedback": positive_count,
        "negative_feedback": negative_count,
        "feedback_ratio": {
            "positive": positive_count / total if total > 0 else 0,
            "negative": negative_count / total if total > 0 else 0
        },
        "adaptation_level": profile.adaptation_level,
        "knowledge_progress": profile.knowledge_level,
        "timestamp": time.time()
    }
    
    return progress

def generate_adaptation_report(self) -> str:
    """
    Генерирует текстовый отчет об адаптации.
    
    Returns:
        str: Текстовый отчет
    """
    dashboard = self.get_adaptation_dashboard_data()
    health = dashboard["health"]
    problem_areas = dashboard["health"]["problem_areas"]
    
    report = "ОТЧЕТ О СИСТЕМНОЙ АДАПТАЦИИ\n"
    report += "=" * 50 + "\n\n"
    
    # Общая оценка
    report += f"Оценка здоровья системы: {health['health_score']:.1f}/100\n"
    report += f"Уровень адаптации: {dashboard['metrics']['adaptation_level']:.2f}\n"
    report += f"Активные пользователи: {dashboard['metrics']['active_users']}/{dashboard['metrics']['total_users']}\n\n"
    
    # Проблемные области
    if problem_areas:
        report += "ОБНАРУЖЕННЫЕ ПРОБЛЕМНЫЕ ОБЛАСТИ:\n"
        for i, area in enumerate(problem_areas, 1):
            report += f"{i}. {area}\n"
        report += "\n"
    
    # Анализ фидбэка
    feedback_ratio = dashboard["metrics"]["feedback_stats"]
    total = feedback_ratio["total"]
    if total > 0:
        pos_ratio = feedback_ratio["positive"] / total
        neg_ratio = feedback_ratio["negative"] / total
        
        report += "АНАЛИЗ ФИДБЭКА:\n"
        report += f"- Положительных отзывов: {pos_ratio:.1%}\n"
        report += f"- Отрицательных отзывов: {neg_ratio:.1%}\n"
        
        if neg_ratio > 0.3:
            report += "\nРЕКОМЕНДАЦИИ:\n"
            report += "- Рассмотрите возможность адаптации контента под уровень знаний пользователя\n"
            report += "- Проверьте проблемные области, выявленные через негативный фидбэк\n"
    
    report += "\n" + "=" * 50 + "\n"
    report += "Отчет сгенерирован автоматически"
    
    return report

def get_system_adaptation_report(self) -> Dict[str, Any]:
    """
    Возвращает отчет о системной адаптации.
    
    Returns:
        Dict: Отчет о системной адаптации
    """
    # Получаем здоровье системы через ядро, если доступно
    if self.brain and hasattr(self.brain, 'get_system_health'):
        health = self.brain.get_system_health()
    else:
        health = self.get_system_health() if hasattr(self, 'get_system_health') else {"health_score": 0.0}
    
    # Анализируем фидбэк за последнюю неделю
    feedback_last_week = self.get_feedback_history(days=7)
    negative_feedback = [fb for fb in feedback_last_week if fb.feedback_type == "negative"]
    
    # Анализируем проблемы
    problem_areas = []
    if negative_feedback:
        # Группируем негативный фидбэк по концептам
        concept_issues = {}
        for fb in negative_feedback:
            concept = self._extract_concept_from_query(fb.query)
            if concept:
                if concept not in concept_issues:
                    concept_issues[concept] = []
                concept_issues[concept].append(fb)
        
        # Определяем значимые проблемы
        for concept, feedbacks in concept_issues.items():
            if len(feedbacks) >= 3:  # Если проблема встречается не менее 3 раз
                problem_areas.append(
                    f"Проблема с концептом '{concept}': {len(feedbacks)} негативных отзывов за неделю"
                )
    
    return {
        "health_score": health["health_score"],
        "adaptation_level": self.adaptation_level,
        "total_users": self.stats["total_users"],
        "active_users": self.stats["active_users"],
        "feedback_stats": {
            "total": self.stats["total_feedback"],
            "positive": self.stats["positive_feedback"],
            "negative": self.stats["negative_feedback"],
            "neutral": self.stats["neutral_feedback"]
        },
        "problem_areas": problem_areas,
        "timestamp": time.time()
    }

def adapt_response(self, user_id: str, response: str, context: Dict[str, Any]) -> str:
    """
    Адаптирует ответ под пользователя.
    
    Args:
        user_id: ID пользователя
        response: Исходный ответ
        context: Контекст запроса
        
    Returns:
        str: Адаптированный ответ
    """
    profile = self.get_user_profile(user_id)
    
    # Адаптируем длину ответа
    if "response_length" in profile.preferences:
        if profile.preferences["response_length"] == "short":
            response = self._shorten_response(response)
        elif profile.preferences["response_length"] == "detailed":
            response = self._expand_response(response, context)
    
    # Адаптируем уровень формальности
    if "formality_level" in profile.preferences:
        if profile.preferences["formality_level"] == "formal":
            response = self._make_response_more_formal(response)
        elif profile.preferences["formality_level"] == "casual":
            response = self._make_response_more_casual(response)
    
    # Адаптируем под стиль обучения
    if profile.learning_style == "visual" and "concepts" in context:
        response = self._add_visual_elements(response, context["concepts"])
    
    return response

def _shorten_response(self, response: str) -> str:
    """Сокращает ответ."""
    sentences = response.split(". ")
    if len(sentences) > 3:
        return ". ".join(sentences[:3]) + "."
    return response

def _expand_response(self, response: str, context: Dict[str, Any]) -> str:
    """Расширяет ответ, добавляя контекстные детали (простейшая реализация)."""
    details = []
    if isinstance(context, dict):
        topic = context.get("topic") or context.get("concept")
        if topic:
            details.append(f"Тема: {topic}")
        if context.get("examples"):
            details.append(f"Примеры: {', '.join(map(str, context['examples']))}")
    extra = ("\n\nДополнительно: " + "; ".join(details)) if details else ""
    return response + extra

def _extract_concept_from_query(self, query: str) -> Optional[str]:
    """Извлекает концепт из запроса пользователя с использованием полноценной NLP обработки."""
    # Проверяем кэш для этого запроса
    query_hash = hashlib.md5(query.lower().encode()).hexdigest()
    if query_hash in self._concept_cache:
        return self._concept_cache[query_hash]
    
    logger.debug(f"Анализ запроса для извлечения концептов: '{query}'")
    
    # Инициализируем кэш, если он еще не создан
    if not hasattr(self, '_concept_cache'):
        self._concept_cache = {}
    
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
            
            # 6. Если NLP-анализ не дал результатов, пытаемся использовать семантическое сходство
            if not candidate_concepts and self.brain and hasattr(self.brain, 'nlp_processor'):
                # Получаем эмбеддинг запроса
                query_embedding = self.brain.nlp_processor.get_embedding(query)
                
                # Получаем все концепты из графа знаний
                if self.brain and hasattr(self.brain, 'knowledge_graph'):
                    all_concepts = self.brain.knowledge_graph.get_all_concepts()
                    
                    if all_concepts:
                        # Вычисляем сходство с каждым концептом
                        similarities = []
                        try:
                            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
                        except Exception:
                            cosine_similarity = None  # type: ignore
                        for concept in all_concepts:
                            concept_embedding = self.brain.nlp_processor.get_embedding(concept)
                            if cosine_similarity is None:
                                # Простейшая косинусная мера без sklearn
                                import math
                                def _cos(a, b):
                                    num = sum(x*y for x, y in zip(a, b))
                                    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
                                    return (num / den) if den else 0.0
                                similarity = _cos(query_embedding, concept_embedding)
                            else:
                                similarity = cosine_similarity([query_embedding], [concept_embedding])[0][0]
                            similarities.append((concept, similarity))
                        
                        # Сортируем по сходству
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        
                        # Возвращаем наиболее похожий концепт
                        if similarities and similarities[0][1] > 0.6:
                            result = similarities[0][0].replace(" ", "_").lower()
                            self._concept_cache[query_hash] = result
                            return result
        
        except Exception as e:
            logger.error(f"Ошибка при NLP-анализе запроса: {e}")
    
    # 7. Резервные методы, если NLP-обработка недоступна
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
                except Exception:
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
    
    # 8. Если все методы не сработали, возвращаем None
    logger.debug(f"Не удалось определить концепт из запроса: '{query}'")
    self._concept_cache[query_hash] = None
    return None

def _make_response_more_formal(self, response: str) -> str:
    """Делает ответ более формальным."""
    # Заменяем разговорные выражения на формальные
    replacements = {
        "знать": "ознакомиться",
        "понять": "осознать",
        "сделать": "осуществить"
    }
    
    for informal, formal in replacements.items():
        response = response.replace(informal, formal)
    
    return response

def _make_response_more_casual(self, response: str) -> str:
    """Делает ответ более неформальным."""
    # Заменяем формальные выражения на разговорные
    replacements = {
        "ознакомиться": "знать",
        "осознать": "понять",
        "осуществить": "сделать"
    }
    
    for formal, informal in replacements.items():
        response = response.replace(formal, informal)
    
    return response

def _add_visual_elements(self, response: str, concepts: List[str]) -> str:
    """Добавляет визуальные элементы в ответ."""
    if concepts:
        response += f"\n\nВизуальное представление: {', '.join(concepts)}"
    return response

# Добавляем методы в класс AdaptationManager
AdaptationManager.get_user_profile = get_user_profile
AdaptationManager.update_user_profile = update_user_profile
AdaptationManager.record_feedback = record_feedback
AdaptationManager._update_profile_from_feedback = _update_profile_from_feedback
AdaptationManager.analyze_user_patterns = analyze_user_patterns
AdaptationManager.export_adaptation_data = export_adaptation_data
AdaptationManager.get_feedback_history = get_feedback_history
AdaptationManager.import_adaptation_data = import_adaptation_data
AdaptationManager.integrate_with_knowledge_graph = integrate_with_knowledge_graph
AdaptationManager.get_cultural_adaptation = get_cultural_adaptation
AdaptationManager.get_adaptation_progress = get_adaptation_progress
AdaptationManager.generate_adaptation_report = generate_adaptation_report
AdaptationManager.get_system_adaptation_report = get_system_adaptation_report
AdaptationManager.adapt_response = adapt_response
AdaptationManager._shorten_response = _shorten_response
AdaptationManager._expand_response = _expand_response
AdaptationManager._make_response_more_formal = _make_response_more_formal
AdaptationManager._make_response_more_casual = _make_response_more_casual
