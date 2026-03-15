"""Аналитические функции адаптационного менеджера CogniFlex"""
import os
import logging
import time
import sqlite3
import numpy as np
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger("cogniflex.adaptation.analytics")

# Импортируем основной класс
from .adaptation_core import AdaptationManager

# Добавляем методы аналитики в класс AdaptationManager
def get_adaptation_metrics(self) -> Dict[str, Any]:
    """
    Возвращает метрики адаптации системы.
    
    Returns:
        Dict[str, Any]: Метрики адаптации
    """
    # Подсчитываем статистику по фидбэку
    feedback_stats = {
        "total": self.stats["total_feedback"],
        "positive": self.stats["positive_feedback"],
        "negative": self.stats["negative_feedback"],
        "neutral": self.stats["neutral_feedback"]
    }
    
    # Вычисляем эффективность адаптации
    total_feedback = self.stats["total_feedback"]
    adaptation_efficiency = 0.0
    if total_feedback > 0:
        adaptation_efficiency = self.stats["positive_feedback"] / total_feedback
    
    return {
        "adaptation_level": self.adaptation_level,
        "total_users": self.stats["total_users"],
        "active_users": self.stats["active_users"],
        "feedback_stats": feedback_stats,
        "last_update": datetime.fromtimestamp(self.stats["last_update"]).isoformat() if self.stats["last_update"] > 0 else "N/A",
        "user_patterns_analyzed": self.user_patterns_analyzed,
        "adaptation_efficiency": adaptation_efficiency
    }

def get_system_health(self) -> Dict[str, Any]:
    """
    Возвращает отчет о здоровье системы адаптации.
    
    Returns:
        Dict: Отчет о здоровье
    """
    metrics = self.get_adaptation_metrics()
    
    # Рассчитываем общий показатель здоровья
    health_score = 100.0
    
    # Учитываем уровень адаптации
    if metrics["adaptation_level"] < 0.5:
        health_score -= min(30, (0.5 - metrics["adaptation_level"]) * 100)
    elif metrics["adaptation_level"] < 0.7:
        health_score -= min(15, (0.7 - metrics["adaptation_level"]) * 50)
    
    # Учитываем количество пользователей
    if metrics["total_users"] < 5:
        health_score -= min(20, (5 - metrics["total_users"]) * 5)
    
    # Учитываем активность
    if metrics["total_users"] > 0 and metrics["active_users"] / metrics["total_users"] < 0.3:
        health_score -= 15
    
    # Анализируем проблемы
    problem_areas = []
    if metrics["adaptation_level"] < 0.5:
        problem_areas.append("Низкий уровень адаптации")
    
    if metrics["total_users"] < 5:
        problem_areas.append("Малое количество пользователей")
    
    if metrics["feedback_stats"]["negative"] > metrics["feedback_stats"]["positive"] * 2:
        problem_areas.append("Преобладание негативного фидбэка")
    
    # Формируем рекомендации
    recommendations = []
    if metrics["adaptation_level"] < 0.5:
        recommendations.append(
            "Рекомендуется улучшить алгоритмы адаптации на основе анализа фидбэка"
        )
    
    if metrics["feedback_stats"]["negative"] > metrics["feedback_stats"]["positive"] * 2:
        recommendations.append(
            "Требуется анализ причин негативного фидбэка и корректировка стратегии адаптации"
        )
    
    if not recommendations:
        recommendations.append(
            "Система адаптации работает стабильно. Продолжайте сбор фидбэка "
            "для дальнейшего улучшения."
        )
    
    return {
        "health_score": max(0, min(100, health_score)),
        "metrics": metrics,
        "problem_areas": problem_areas,
        "recommendations": recommendations,
        "timestamp": time.time()
    }

def get_adaptation_dashboard_data(self) -> Dict[str, Any]:
    """
    Возвращает данные для дашборда адаптации.
    
    Returns:
        Dict[str, Any]: Данные для дашборда
    """
    # Получаем данные за последние 7 дней
    end_time = time.time()
    start_time = end_time - (7 * 86400)
    
    # Получаем аналитические данные из БД
    analytics = self._get_analytics_data(start_time, end_time)
    
    # Формируем временные ряды
    trends = {
        "users_over_time": [],
        "feedback_over_time": [],
        "adaptation_level": []
    }
    
    for entry in analytics:
        date_str = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d")
        
        # Пользователи
        trends["users_over_time"].append({
            "date": date_str,
            "total": entry["total_users"],
            "active": entry["active_users"]
        })
        
        # Фидбэк
        total = entry["total_feedback"]
        positive = entry["positive_feedback"]
        negative = entry["negative_feedback"]
        neutral = entry["neutral_feedback"]
        
        trends["feedback_over_time"].append({
            "date": date_str,
            "total": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "positive_ratio": positive / total if total > 0 else 0,
            "negative_ratio": negative / total if total > 0 else 0
        })
        
        # Уровень адаптации
        adaptation_level = positive / total if total > 0 else 0
        trends["adaptation_level"].append({
            "date": date_str,
            "value": adaptation_level
        })
    
    # Получаем текущий отчет о здоровье
    health = self.get_system_health()
    
    return {
        "health": health,
        "metrics": self.get_adaptation_metrics(),
        "trends": trends,
        "timestamp": time.time()
    }

def _get_analytics_data(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
    """Получает аналитические данные из базы данных."""
    if not self.db_path:
        return []
        
    try:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM adaptation_analytics
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """, (start_time, end_time))
        
        analytics = []
        for row in cursor.fetchall():
            analytics.append({
                "timestamp": row[0],
                "total_users": row[1],
                "active_users": row[2],
                "total_feedback": row[3],
                "positive_feedback": row[4],
                "negative_feedback": row[5],
                "neutral_feedback": row[6]
            })
        
        conn.close()
        return analytics
    except Exception as e:
        logger.error(f"Ошибка получения аналитических данных: {e}")
        return []

def get_adaptation_insights(self) -> Dict[str, Any]:
    """
    Возвращает аналитические данные об адаптации.
    
    Returns:
        Dict: Аналитические данные
    """
    # Получаем данные о пользователях
    total_users = self.stats["total_users"]
    if total_users == 0:
        return {"insights": [], "recommendations": ["Нет данных о пользователях"]}
    
    # Анализируем уровни знаний
    knowledge_levels = {"beginner": 0, "intermediate": 0, "advanced": 0, "expert": 0}
    learning_styles = {"visual": 0, "auditory": 0, "reading": 0, "kinesthetic": 0, "balanced": 0}
    
    for profile in self.user_profiles.values():
        # Определяем уровень знаний
        if profile.knowledge_level < 0.25:
            knowledge_levels["beginner"] += 1
        elif profile.knowledge_level < 0.5:
            knowledge_levels["intermediate"] += 1
        elif profile.knowledge_level < 0.75:
            knowledge_levels["advanced"] += 1
        else:
            knowledge_levels["expert"] += 1
        
        # Учитываем стиль обучения
        style = profile.learning_style
        if style in learning_styles:
            learning_styles[style] += 1
    
    # Анализируем фидбэк
    feedback_last_week = [fb for fb in self.feedback_history 
                        if time.time() - fb.timestamp <= 7 * 86400]
    
    # Группируем негативный фидбэк по концептам
    concept_issues = {}
    for fb in feedback_last_week:
        if fb.feedback_type == "negative":
            concept = self._extract_concept_from_query(fb.query)
            if concept:
                if concept not in concept_issues:
                    concept_issues[concept] = []
                concept_issues[concept].append(fb)
    
    # Формируем инсайты
    insights = []
    
    # Анализируем проблемы с концептами
    for concept, feedbacks in concept_issues.items():
        if len(feedbacks) >= 3:  # Если проблема встречается не менее 3 раз
            insights.append(
                f"Проблема с концептом '{concept}': {len(feedbacks)} негативных отзывов за неделю"
            )
    
    # Анализируем распределение уровней знаний
    dominant_knowledge = max(knowledge_levels, key=knowledge_levels.get)
    if knowledge_levels[dominant_knowledge] / total_users > 0.7:
        insights.append(
            f"Большинство пользователей ({knowledge_levels[dominant_knowledge]/total_users:.1%}) "
            f"имеют уровень знаний '{dominant_knowledge}'"
        )
    
    # Анализируем стили обучения
    dominant_style = max(learning_styles, key=learning_styles.get)
    if learning_styles[dominant_style] / total_users > 0.6:
        insights.append(
            f"Преобладающий стиль обучения: '{dominant_style}' "
            f"({learning_styles[dominant_style]/total_users:.1%} пользователей)"
        )
    
    # Формируем рекомендации
    recommendations = []
    if insights:
        recommendations.append("На основе анализа пользователей рекомендуется сфокусироваться на следующих улучшениях:")
        for i, insight in enumerate(insights, 1):
            recommendations.append(f"{i}. {insight}")
    else:
        recommendations.append("Система хорошо адаптирована под потребности пользователей. "
                             "Рекомендуется продолжать текущую стратегию адаптации.")
    
    return {
        "insights": insights,
        "recommendations": recommendations,
        "user_distribution": {
            "knowledge_levels": {k: v/total_users for k, v in knowledge_levels.items()},
            "learning_styles": {k: v/total_users for k, v in learning_styles.items()}
        },
        "feedback_metrics": {
            "positive_ratio": self.stats["positive_feedback"] / max(1, self.stats["total_feedback"]),
            "negative_ratio": self.stats["negative_feedback"] / max(1, self.stats["total_feedback"])
        },
        "timestamp": time.time()
    }

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
                        for concept in all_concepts:
                            concept_embedding = self.brain.nlp_processor.get_embedding(concept)
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
    
    # 8. Если все методы не сработали, возвращаем None
    logger.debug(f"Не удалось определить концепт из запроса: '{query}'")
    self._concept_cache[query_hash] = None
    return None

def apply_adaptation_insights(self):
    """Применяет аналитические данные для улучшения адаптации."""
    insights = self.get_adaptation_insights()
    
    # Если есть инсайты для улучшения
    if insights["insights"]:
        logger.info("Применение инсайтов адаптации для улучшения системы")
        
        # Обрабатываем проблемы с концептами
        for insight in insights["insights"]:
            if "Проблема с концептом" in insight:
                # Извлекаем название концепта
                concept = insight.split("'")[1]
                
                # Генерируем задачу для самоанализа
                if self.brain and hasattr(self.brain, 'self_analyzer'):
                    self.brain.self_analyzer.add_analysis_task(
                        title=f"Углубить знания по концепту '{concept}'",
                        description=f"Частый негативный фидбэк по концепту",
                        priority="high",
                        domain=self.brain.self_analyzer._determine_domain(concept),
                        evidence=[f"Частый негативный фидбэк по концепту ({len(insight.split()[-2])} за неделю)"],
                        suggested_actions=[
                            f"Углубить знания по концепту '{concept}'",
                            f"Проверить источники информации по '{concept}'"
                        ]
                    )

# Добавляем методы в класс AdaptationManager
AdaptationManager.get_adaptation_metrics = get_adaptation_metrics
AdaptationManager.get_system_health = get_system_health
AdaptationManager.get_adaptation_dashboard_data = get_adaptation_dashboard_data
AdaptationManager._get_analytics_data = _get_analytics_data
AdaptationManager.get_adaptation_insights = get_adaptation_insights
AdaptationManager._extract_concept_from_query = _extract_concept_from_query
AdaptationManager.apply_adaptation_insights = apply_adaptation_insights