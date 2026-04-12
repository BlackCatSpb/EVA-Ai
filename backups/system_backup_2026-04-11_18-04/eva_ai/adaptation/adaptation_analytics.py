"""Аналитические функции адаптационного менеджера ЕВА"""
import os
import logging
import time
import sqlite3
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


logger = logging.getLogger("eva_ai.adaptation.analytics")

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
    feedback_last_week = []
    if self.feedback_history:
        feedback_last_week = [fb for fb in self.feedback_history 
                            if hasattr(fb, 'timestamp') and hasattr(fb, 'feedback_type')
                            and time.time() - fb.timestamp <= 7 * 86400]
    
    # Группируем негативный фидбэк по концептам
    concept_issues = {}
    for fb in feedback_last_week:
        if hasattr(fb, 'feedback_type') and fb.feedback_type == "negative":
            if not hasattr(fb, 'query'):
                continue
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
AdaptationManager.apply_adaptation_insights = apply_adaptation_insights