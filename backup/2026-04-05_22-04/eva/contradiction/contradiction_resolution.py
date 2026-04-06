"""Модуль для разрешения противоречий в системе ЕВА"""
import os
import logging
import time
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
import random
import hashlib
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger("eva.contradiction.resolution")

# Инициализация NLP-ресурсов (offline-safe)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception:
        pass

class ContradictionResolution:
    """Класс, содержащий методы разрешения противоречий."""
    
    @staticmethod
    def generate_resolution_report(contradiction) -> Dict[str, Any]:
        """
        Генерирует отчет о разрешении противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            Dict: Отчет о разрешении
        """
        report = {
            "contradiction_id": contradiction.contradiction_id,
            "concept": contradiction.concept,
            "divergence_level": contradiction.divergence_level,
            "type": ContradictionResolution.get_contradiction_type(contradiction),
            "severity": ContradictionResolution.get_severity(contradiction),
            "status": contradiction.status,
            "timestamp": contradiction.timestamp,
            "resolution_summary": ContradictionResolution.get_resolution_summary(contradiction),
            "confidence": contradiction.confidence,
            "resolution_priority": ContradictionResolution.get_resolution_priority(contradiction),
            "impact_score": contradiction.impact_score,
            "analysis": {
                "source_analysis": contradiction.source_analysis,
                "nlp_metrics": contradiction.nlp_metrics
            }
        }
        
        # Добавляем рекомендации
        report["recommendations"] = ContradictionResolution._generate_resolution_recommendations(contradiction)
        
        return report
    
    @staticmethod
    def _generate_resolution_recommendations(contradiction) -> List[str]:
        """
        Генерирует рекомендации по разрешению противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            List[str]: Рекомендации
        """
        recommendations = []
        contradiction_type = ContradictionResolution.get_contradiction_type(contradiction)
        severity = ContradictionResolution.get_severity(contradiction)
        
        # Рекомендации на основе типа противоречия
        if contradiction_type == "numeric_conflict":
            recommendations.append(
                "Проведите дополнительный анализ для определения наиболее точного числового значения. "
                "Рассмотрите возможность усреднения показателей или выявления контекстных условий, "
                "при которых верно каждое значение."
            )
        
        elif contradiction_type == "boolean_conflict":
            recommendations.append(
                "Проверьте условия, при которых каждое утверждение является верным. "
                "Возможно, противоречие возникает из-за различия в контексте или условиях."
            )
        
        elif contradiction_type == "exclusivity_conflict":
            recommendations.append(
                "Проанализируйте, не являются ли утверждения 'только' и 'не только' "
                "применимыми в разных контекстах или подкатегориях."
            )
        
        elif contradiction_type == "hierarchy_conflict":
            recommendations.append(
                "Пересмотрите иерархию для устранения циклических зависимостей или "
                "взаимоисключающих классификаций. Возможно, некоторые связи должны "
                "быть заменены на другие типы отношений."
            )
        
        elif contradiction_type == "response_conflict":
            recommendations.append(
                "Проанализируйте контекст использования каждого ответа. Возможно, "
                "разные ответы применимы в разных сценариях или для разных аудиторий."
            )
        
        # Рекомендации на основе серьезности
        if severity == "high":
            recommendations.append(
                "Это высокосерьезное противоречие требует немедленного внимания. "
                "Рассмотрите возможность привлечения экспертов для разрешения."
            )
        elif severity == "medium":
            recommendations.append(
                "Это среднесерьезное противоречие важно для точности системы. "
                "Планируйте его разрешение в ближайшее время."
            )
        else:
            recommendations.append(
                "Это низкосерьезное противоречие имеет минимальное влияние на систему. "
                "Его можно разрешить в рамках регулярного процесса обновления знаний."
            )
        
        # Добавляем общие рекомендации
        recommendations.append(
            "Соберите дополнительные данные из авторитетных источников для подтверждения или "
            "опровержения конфликтующих утверждений."
        )
        
        recommendations.append(
            "Проведите анализ контекстных условий, при которых проявляется каждое утверждение."
        )
        
        return recommendations
    
    @staticmethod
    def calculate_contradiction_impact(contradiction, knowledge_graph) -> float:
        """
        Вычисляет влияние противоречия на систему знаний.
        
        Args:
            contradiction: Экземпляр противоречия
            knowledge_graph: Ссылка на граф знаний
            
        Returns:
            float: Уровень влияния (0.0-1.0)
        """
        try:
            # Получаем количество связанных узлов
            related_nodes = knowledge_graph.get_related_nodes(contradiction.concept)
            node_count = len(related_nodes)
            
            # Получаем глубину влияния
            depth = knowledge_graph.get_influence_depth(contradiction.concept)
            
            # Оцениваем важность концепта
            importance = ContradictionResolution._calculate_concept_importance(contradiction)
            
            # Влияние = (важность * 0.4) + (количество узлов * 0.3) + (глубина * 0.3)
            impact = (importance * 0.4) + (min(1.0, node_count / 50) * 0.3) + (min(1.0, depth / 10) * 0.3)
            
            contradiction.impact_score = min(1.0, impact)
            return contradiction.impact_score
            
        except Exception as e:
            logger.error(f"Ошибка вычисления влияния противоречия: {e}")
            # Базовая оценка влияния на основе дивергенции
            contradiction.impact_score = contradiction.divergence_level * 0.7
            return contradiction.impact_score
    
    @staticmethod
    def get_learning_opportunity(contradiction) -> Dict[str, Any]:
        """
        Генерирует возможность для обучения на основе противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            Dict: Возможность для обучения
        """
        priority = ContradictionResolution.get_resolution_priority(contradiction)
        impact = contradiction.divergence_level * 0.8 + priority * 0.2
        
        return {
            "id": f"learn_{contradiction.contradiction_id}",
            "concept": contradiction.concept,
            "type": "contradiction_resolution",
            "priority": impact,
            "description": f"Разрешение противоречия в знаниях по '{contradiction.concept}'",
            "evidence": [
                f"Обнаружено противоречие с уровнем расхождения {contradiction.divergence_level:.2f}",
                f"Тип противоречия: {ContradictionResolution.get_contradiction_type(contradiction)}",
                f"Серьезность: {ContradictionResolution.get_severity(contradiction)}"
            ],
            "suggested_actions": ContradictionResolution._generate_resolution_recommendations(contradiction)[:3],
            "required_capabilities": ["knowledge_integration", "source_analysis"]
        }
    
    @staticmethod
    def update_confidence(contradiction, new_confidence: float):
        """
        Обновляет уверенность в противоречии.
        
        Args:
            contradiction: Экземпляр противоречия
            new_confidence: Новая уверенность
        """
        contradiction.confidence = max(0.0, min(1.0, new_confidence))
    
    @staticmethod
    def get_contradiction_type(contradiction) -> str:
        """Определяет тип противоречия."""
        return getattr(contradiction, 'contradiction_type', 'general_conflict')
    
    @staticmethod
    def get_severity(contradiction) -> str:
        """Определяет серьезность противоречия."""
        if contradiction.divergence_level > 0.8:
            return "high"
        elif contradiction.divergence_level > 0.5:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def get_resolution_priority(contradiction) -> float:
        """Вычисляет приоритет разрешения противоречия."""
        severity_weight = {"high": 0.9, "medium": 0.6, "low": 0.3}
        severity = ContradictionResolution.get_severity(contradiction)
        return severity_weight.get(severity, 0.5) * contradiction.divergence_level
    
    @staticmethod
    def get_resolution_summary(contradiction) -> str:
        """Генерирует краткое резюме разрешения противоречия."""
        if hasattr(contradiction, 'resolution') and contradiction.resolution:
            return contradiction.resolution.get('summary', 'Противоречие в процессе разрешения')
        return f"Противоречие по '{contradiction.concept}' требует разрешения"
    
    @staticmethod
    def calculate_resolution_confidence(contradiction) -> float:
        """Вычисляет уверенность в разрешении противоречия."""
        base_confidence = 1.0 - contradiction.divergence_level
        if hasattr(contradiction, 'source_analysis') and contradiction.source_analysis:
            source_reliability = sum(s.get('reliability', 0.5) for s in contradiction.source_analysis.values()) / len(contradiction.source_analysis)
            return min(1.0, base_confidence * 0.7 + source_reliability * 0.3)
        return base_confidence * 0.5
    
    @staticmethod
    def _calculate_concept_importance(contradiction) -> float:
        """Вычисляет важность концепта."""
        base_importance = getattr(contradiction, 'usage_frequency', 0.5)
        type_weights = {
            "numeric_conflict": 0.8,
            "boolean_conflict": 0.9,
            "hierarchy_conflict": 0.95,
            "response_conflict": 0.7,
            "general_conflict": 0.6
        }
        contradiction_type = ContradictionResolution.get_contradiction_type(contradiction)
        type_weight = type_weights.get(contradiction_type, 0.6)
        return min(1.0, base_importance * type_weight)
    
    @staticmethod
    def get_confidence(contradiction) -> float:
        """
        Возвращает текущую уверенность в противоречии.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            float: Уверенность
        """
        return contradiction.confidence
    
    @staticmethod
    def get_resolution_confidence(contradiction) -> float:
        """
        Возвращает уверенность в решении противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            float: Уверенность в решении
        """
        if contradiction.status == "resolved" and "confidence" in contradiction.resolution:
            return contradiction.resolution["confidence"]
        return ContradictionResolution.calculate_resolution_confidence(contradiction)
    
    @staticmethod
    def is_high_priority(contradiction) -> bool:
        """
        Проверяет, является ли противоречие высокоприоритетным.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            bool: Является ли высокоприоритетным
        """
        return ContradictionResolution.get_resolution_priority(contradiction) > 0.7
    
    @staticmethod
    def get_time_since_detection(contradiction) -> float:
        """
        Возвращает время с момента обнаружения противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            float: Время в секундах
        """
        return time.time() - contradiction.timestamp
    
    @staticmethod
    def requires_immediate_attention(contradiction) -> bool:
        """
        Проверяет, требует ли противоречие немедленного внимания.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            bool: Требует ли немедленного внимания
        """
        # Высокосерьезные противоречия требуют внимания в течение 24 часов
        if ContradictionResolution.get_severity(contradiction) == "high" and ContradictionResolution.get_time_since_detection(contradiction) > 86400:
            return True
        
        # Среднесерьезные противоречия требуют внимания в течение 3 дней
        if ContradictionResolution.get_severity(contradiction) == "medium" and ContradictionResolution.get_time_since_detection(contradiction) > 3 * 86400:
            return True
        
        return False
    
    @staticmethod
    def generate_balanced_response(contradiction, language: str = "ru") -> str:
        """
        Генерирует сбалансированный ответ на основе противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            language: Язык ответа
            
        Returns:
            str: Сбалансированный ответ
        """
        contradiction_type = ContradictionResolution.get_contradiction_type(contradiction)
        severity = ContradictionResolution.get_severity(contradiction)
        
        if language == "ru":
            if contradiction_type == "numeric_conflict":
                return ContradictionResolution._format_numeric_conflict_response_ru(contradiction, severity)
            elif contradiction_type == "boolean_conflict":
                return ContradictionResolution._format_boolean_conflict_response_ru(contradiction, severity)
            elif contradiction_type == "response_conflict":
                return ContradictionResolution._format_response_conflict_response_ru(contradiction, severity)
            else:
                return ContradictionResolution._format_general_conflict_response_ru(contradiction, severity)
        else:
            if contradiction_type == "numeric_conflict":
                return ContradictionResolution._format_numeric_conflict_response_en(contradiction, severity)
            elif contradiction_type == "boolean_conflict":
                return ContradictionResolution._format_boolean_conflict_response_en(contradiction, severity)
            elif contradiction_type == "response_conflict":
                return ContradictionResolution._format_response_conflict_response_en(contradiction, severity)
            else:
                return ContradictionResolution._format_general_conflict_response_en(contradiction, severity)
    
    @staticmethod
    def _format_numeric_conflict_response_ru(contradiction, severity: str) -> str:
        """Форматирует ответ для числового противоречия на русском языке."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        value1 = fact1.get("value", "?")
        value2 = fact2.get("value", "?")
        source1 = fact1.get("source", "неизвестный источник")
        source2 = fact2.get("source", "неизвестный источник")
        
        if severity == "high":
            return (
                f"Существует значительное расхождение в данных по '{contradiction.concept}': {value1} согласно {source1} "
                f"и {value2} согласно {source2}. Это противоречие требует дополнительного анализа для определения "
                f"наиболее точного значения. Возможно, различия обусловлены разными методами измерения или "
                f"контекстными условиями."
            )
        elif severity == "medium":
            return (
                f"Наблюдаются различия в данных по '{contradiction.concept}': {value1} согласно {source1} "
                f"и {value2} согласно {source2}. Разница может быть связана с различными условиями измерения или "
                f"методологиями. Рекомендуется уточнить контекст применения каждого значения."
            )
        else:
            return (
                f"Незначительные различия в данных по '{contradiction.concept}': {value1} согласно {source1} "
                f"и {value2} согласно {source2}. Эти расхождения находятся в пределах нормальной вариативности "
                f"и могут быть объяснены различиями в методах измерения."
            )
    
    @staticmethod
    def _format_boolean_conflict_response_ru(contradiction, severity: str) -> str:
        """Форматирует ответ для булева противоречия на русском языке."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        value1 = "верно" if fact1.get("value", False) else "неверно"
        value2 = "верно" if fact2.get("value", False) else "неверно"
        source1 = fact1.get("source", "неизвестный источник")
        source2 = fact2.get("source", "неизвестный источник")
        
        if severity == "high":
            return (
                f"Существует фундаментальное противоречие по '{contradiction.concept}': утверждение считается {value1} "
                f"согласно {source1} и {value2} согласно {source2}. Это указывает на глубокие различия в "
                f"интерпретации концепта. Для разрешения требуется анализ основных принципов и контекста."
            )
        elif severity == "medium":
            return (
                f"Наблюдается противоречие по '{contradiction.concept}': утверждение считается {value1} "
                f"согласно {source1} и {value2} согласно {source2}. Различия, вероятно, связаны с "
                f"контекстными условиями или областью применения."
            )
        else:
            return (
                f"Незначительное расхождение в интерпретации '{contradiction.concept}': утверждение считается {value1} "
                f"согласно {source1} и {value2} согласно {source2}. Эти различия могут быть объяснены "
                f"небольшими отличиями в контексте."
            )
    
    @staticmethod
    def _format_response_conflict_response_ru(contradiction, severity: str) -> str:
        """Форматирует ответ для противоречия в ответах на русском языке."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        snippet1 = fact1.get("value", "")[:100] + "..." if len(fact1.get("value", "")) > 100 else fact1.get("value", "")
        snippet2 = fact2.get("value", "")[:100] + "..." if len(fact2.get("value", "")) > 100 else fact2.get("value", "")
        
        if severity == "high":
            return (
                f"Существуют кардинально различные ответы на вопрос о '{contradiction.concept}':\n\n"
                f"1. {snippet1}\n\n"
                f"2. {snippet2}\n\n"
                f"Эти ответы представляют противоположные точки зрения. Для разрешения рекомендуется "
                f"провести глубокий анализ контекста и целевой аудитории, а также собрать дополнительные данные."
            )
        elif severity == "medium":
            return (
                f"Наблюдаются значительные различия в ответах на вопрос о '{contradiction.concept}':\n\n"
                f"1. {snippet1}\n\n"
                f"2. {snippet2}\n\n"
                f"Эти ответы отражают разные аспекты концепта. Рекомендуется уточнить контекст вопроса "
                f"и определить, какие условия делают каждый ответ верным."
            )
        else:
            return (
                f"Незначительные различия в ответах на вопрос о '{contradiction.concept}':\n\n"
                f"1. {snippet1}\n\n"
                f"2. {snippet2}\n\n"
                f"Эти ответы дополняют друг друга, отражая разные грани одного явления. "
                f"Оба ответа могут быть верны в разных контекстах."
            )
    
    @staticmethod
    def _format_general_conflict_response_ru(contradiction, severity: str) -> str:
        """Форматирует общий ответ на противоречие на русском языке."""
        if severity == "high":
            return (
                f"Обнаружено серьезное противоречие в знаниях по '{contradiction.concept}'. "
                f"Существуют несовместимые утверждения, требующие немедленного разрешения. "
                f"Рекомендуется провести детальный анализ источников и контекста для определения "
                f"наиболее достоверной информации."
            )
        elif severity == "medium":
            return (
                f"Обнаружено умеренное противоречие в знаниях по '{contradiction.concept}'. "
                f"Существуют различающиеся точки зрения, которые могут быть объяснены "
                f"разными контекстами или условиями. Рекомендуется уточнить условия применимости каждого утверждения."
            )
        else:
            return (
                f"Обнаружено незначительное расхождение в знаниях по '{contradiction.concept}'. "
                f"Эти различия находятся в пределах нормальной вариативности и могут быть "
                f"объяснены различными интерпретациями или контекстами."
            )
    
    @staticmethod
    def _format_numeric_conflict_response_en(contradiction, severity: str) -> str:
        """Форматирует ответ для числового противоречия на английском языке."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        value1 = fact1.get("value", "?")
        value2 = fact2.get("value", "?")
        source1 = fact1.get("source", "unknown source")
        source2 = fact2.get("source", "unknown source")
        
        if severity == "high":
            return (
                f"There is a significant discrepancy in data for '{contradiction.concept}': {value1} according to {source1} "
                f"and {value2} according to {source2}. This contradiction requires additional analysis to determine "
                f"the most accurate value. Differences may be due to different measurement methods or contextual conditions."
            )
        elif severity == "medium":
            return (
                f"There are differences in data for '{contradiction.concept}': {value1} according to {source1} "
                f"and {value2} according to {source2}. The discrepancy may be related to different measurement conditions "
                f"or methodologies. It is recommended to clarify the context of each value's application."
            )
        else:
            return (
                f"Minor differences in data for '{contradiction.concept}': {value1} according to {source1} "
                f"and {value2} according to {source2}. These discrepancies are within normal variability "
                f"and may be explained by differences in measurement methods."
            )
    
    @staticmethod
    def _format_boolean_conflict_response_en(contradiction, severity: str) -> str:
        """Форматирует ответ для булева противоречия на английском языке."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        value1 = "true" if fact1.get("value", False) else "false"
        value2 = "true" if fact2.get("value", False) else "false"
        source1 = fact1.get("source", "unknown source")
        source2 = fact2.get("source", "unknown source")
        
        if severity == "high":
            return (
                f"There is a fundamental contradiction regarding '{contradiction.concept}': the statement is considered {value1} "
                f"according to {source1} and {value2} according to {source2}. This indicates deep differences in "
                f"the interpretation of the concept. Resolution requires analysis of fundamental principles and context."
            )
        elif severity == "medium":
            return (
                f"There is a contradiction regarding '{contradiction.concept}': the statement is considered {value1} "
                f"according to {source1} and {value2} according to {source2}. Differences are likely related to "
                f"contextual conditions or area of application."
            )
        else:
            return (
                f"Minor discrepancy in interpretation of '{contradiction.concept}': the statement is considered {value1} "
                f"according to {source1} and {value2} according to {source2}. These differences may be explained "
                f"by minor contextual variations."
            )
    
    @staticmethod
    def _format_response_conflict_response_en(contradiction, severity: str) -> str:
        """Форматирует ответ для противоречия в ответах на английском языке."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        snippet1 = fact1.get("value", "")[:100] + "..." if len(fact1.get("value", "")) > 100 else fact1.get("value", "")
        snippet2 = fact2.get("value", "")[:100] + "..." if len(fact2.get("value", "")) > 100 else fact2.get("value", "")
        
        if severity == "high":
            return (
                f"There are radically different responses to the question about '{contradiction.concept}':\n\n"
                f"1. {snippet1}\n\n"
                f"2. {snippet2}\n\n"
                f"These responses represent opposing viewpoints. To resolve, it is recommended to "
                f"conduct a deep analysis of context and target audience, as well as gather additional data."
            )
        elif severity == "medium":
            return (
                f"There are significant differences in responses to the question about '{contradiction.concept}':\n\n"
                f"1. {snippet1}\n\n"
                f"2. {snippet2}\n\n"
                f"These responses reflect different aspects of the concept. It is recommended to clarify the context of the question "
                f"and determine under what conditions each response is valid."
            )
        else:
            return (
                f"Minor differences in responses to the question about '{contradiction.concept}':\n\n"
                f"1. {snippet1}\n\n"
                f"2. {snippet2}\n\n"
                f"These responses complement each other, reflecting different facets of the same phenomenon. "
                f"Both responses may be valid in different contexts."
            )
    
    @staticmethod
    def _format_general_conflict_response_en(contradiction, severity: str) -> str:
        """Форматирует общий ответ на противоречие на английском языке."""
        if severity == "high":
            return (
                f"A serious contradiction in knowledge about '{contradiction.concept}' has been detected. "
                f"There are incompatible statements that require immediate resolution. "
                f"It is recommended to conduct a detailed analysis of sources and context to determine "
                f"the most reliable information."
            )
        elif severity == "medium":
            return (
                f"A moderate contradiction in knowledge about '{contradiction.concept}' has been detected. "
                f"There are differing viewpoints that may be explained by "
                f"different contexts or conditions. It is recommended to clarify the applicability conditions for each statement."
            )
        else:
            return (
                f"A minor discrepancy in knowledge about '{contradiction.concept}' has been detected. "
                f"These differences are within normal variability and may be "
                f"explained by different interpretations or contexts."
            )