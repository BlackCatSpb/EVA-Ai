
"""Модуль для анализа противоречий в системе ЕВА"""
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

# Централизованные NLP-фоллбеки (без обязательных внешних зависимостей)
from eva_ai.nlp_fallbacks import (
    compute_semantic_similarity,
    get_sentiment_analyzer,
    polarity_scores,
    tokenize,
    get_stopwords,
)

logger = logging.getLogger("eva_ai.contradiction.analysis")

class ContradictionAnalyzer:
    """Класс, содержащий методы анализа противоречий."""
    
    @staticmethod
    def get_contradiction_type(contradiction) -> str:
        """
        Определяет тип противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            str: Тип противоречия
        """
        if hasattr(contradiction, 'metadata') and isinstance(contradiction.metadata, dict) and "type" in contradiction.metadata:
            return contradiction.metadata["type"]
        
        # Определяем тип по конфликтующим фактам
        if len(contradiction.conflicting_facts) == 2:
            fact1 = contradiction.conflicting_facts[0]
            fact2 = contradiction.conflicting_facts[1]
            
            # Проверяем числовые противоречия
            if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                return "numeric_conflict"
            
            # Проверяем булевы противоречия
            if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                return "boolean_conflict"
            
            # Проверяем противоречия в ответах
            if "response" in fact1.get("relation", "") and "response" in fact2.get("relation", ""):
                return "response_conflict"
        
        # Проверяем по метаданным
        if hasattr(contradiction, 'metadata') and isinstance(contradiction.metadata, dict) and "relation_type" in contradiction.metadata:
            relation_type = contradiction.metadata["relation_type"]
            if relation_type.startswith("only_") or relation_type.startswith("not_only_"):
                return "exclusivity_conflict"
            if relation_type in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        
        return "unknown"
    
    @staticmethod
    def get_severity(contradiction) -> str:
        """
        Определяет серьезность противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            str: Серьезность (low, medium, high)
        """
        if contradiction.divergence_level < 0.4:
            return "low"
        elif contradiction.divergence_level < 0.7:
            return "medium"
        else:
            return "high"
    
    @staticmethod
    def get_resolution_priority(contradiction) -> float:
        """
        Определяет приоритет разрешения противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            float: Приоритет (0.0-1.0)
        """
        # Базовый приоритет на основе серьезности
        severity = ContradictionAnalysis.get_severity(contradiction)
        severity_priority = 0.3 if severity == "low" else 0.6 if severity == "medium" else 0.9
        
        # Учитываем важность концепта
        concept_importance = ContradictionAnalysis._calculate_concept_importance(contradiction)
        
        # Учитываем возраст противоречия
        age = time.time() - contradiction.timestamp
        age_priority = min(0.4, age / 86400 * 0.1)  # Увеличивается на 0.1 за день, максимум 0.4
        
        # Общий приоритет
        priority = (severity_priority * 0.5 + 
                   concept_importance * 0.3 + 
                   age_priority * 0.2)
        
        contradiction.resolution_priority = min(1.0, priority)
        return contradiction.resolution_priority
    
    @staticmethod
    def _calculate_concept_importance(contradiction) -> float:
        """
        Вычисляет важность концепта на основе его использования и значимости.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            float: Важность концепта (0.0-1.0)
        """
        # Анализатор тональности и стоп-слова (безопасно)
        sentiment_analyzer = get_sentiment_analyzer()
        stop_words = get_stopwords(("english", "russian"))
        
        # 1. Анализ частоты использования концепта в системе
        usage_frequency = 0.3  # Базовая оценка
        meta = getattr(contradiction, "metadata", None)
        if not isinstance(meta, dict) or not meta:
            meta = getattr(contradiction, "meta", {})
        if isinstance(meta, dict) and "usage_count" in meta:
            # Логарифмическая шкала для учета частоты использования
            try:
                usage_frequency = min(0.5, 0.1 + np.log1p(float(meta["usage_count"])) * 0.1)
            except Exception:
                pass
        
        # 2. Анализ важности концепта через контент
        concept_score = 0.0
        key_terms = ["наука", "технология", "искусственный интеллект", "этика", "безопасность",
                    "человек", "сознание", "разум", "жизнь", "вселенная", "время", "пространство"]
        
        # Проверяем содержимое конфликтующих фактов
        fact_content = " ".join(str(fact.get("value", "")) for fact in contradiction.conflicting_facts)
        fact_content = fact_content.lower()
        
        # Считаем вхождения ключевых терминов
        key_term_count = sum(1 for term in key_terms if term in fact_content)
        concept_score += min(0.3, key_term_count * 0.1)
        
        # 3. Анализ тональности контента
        scores = polarity_scores(fact_content, sentiment_analyzer)
        neutrality = 1.0 - abs(scores.get('compound', 0.0))
        concept_score += neutrality * 0.2
        
        # 4. Анализ структуры текста
        words = tokenize(fact_content)
        words = [word for word in words if word.isalnum() and word not in stop_words]
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words > 0:
            diversity = unique_words / total_words
            concept_score += min(0.2, diversity * 0.2)
        
        # 5. Проверка на общий/специфичный концепт
        common_concepts = [
            "человек", "знание", "информация", "данные", "процесс", "система",
            "время", "пространство", "вселенная", "жизнь", "сознание", "разум"
        ]
        
        if contradiction.concept.lower() in [c.lower() for c in common_concepts]:
            concept_score += 0.2
        
        # Общая важность концепта
        importance = min(1.0, usage_frequency + concept_score)
        return importance
    
    @staticmethod
    def calculate_semantic_divergence(contradiction, nlp_model) -> float:
        """
        Вычисляет семантическое расхождение между конфликтующим фактами.
        
        Args:
            contradiction: Экземпляр противоречия
            nlp_model: NLP-модель для анализа
            
        Returns:
            float: Уровень семантического расхождения (0.0-1.0)
        """
        if len(contradiction.conflicting_facts) < 2:
            return 0.0
        
        # Извлекаем тексты для сравнения
        texts = [str(fact.get("value", "")) for fact in contradiction.conflicting_facts]
        
        # Вычисляем среднее семантическое расхождение, используя централизованный фоллбек
        total_distance = 0.0
        count = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = compute_semantic_similarity([texts[i], texts[j]], nlp_model)
                distance = 1.0 - float(sim)
                total_distance += distance
                count += 1
        
        if count > 0:
            return total_distance / count
        else:
            return 0.0
    
    @staticmethod
    def analyze_source_credibility(contradiction, source_reputation_system) -> Dict[str, Any]:
        """
        Анализирует достоверность источников конфликтующих фактов.
        
        Args:
            contradiction: Экземпляр противоречия
            source_reputation_system: Система репутации источников
            
        Returns:
            Dict: Результаты анализа достоверности
        """
        sources = []
        for fact in contradiction.conflicting_facts:
            source = fact.get("source")
            if source:
                credibility = source_reputation_system.get_source_reputation(source)
                sources.append({
                    "source": source,
                    "credibility": credibility,
                    "domain": source_reputation_system._extract_domain(source)
                })
        
        # Определяем наиболее надежный источник
        if sources:
            best_source = max(sources, key=lambda x: x["credibility"])
            comparison = "neutral"
            
            if len(sources) > 1:
                diff = sources[0]["credibility"] - sources[1]["credibility"]
                if diff > 0.2:
                    comparison = "source1_more_credible"
                elif diff < -0.2:
                    comparison = "source2_more_credible"
                elif diff > 0.05:
                    comparison = "source1_slightly_more_credible"
                elif diff < -0.05:
                    comparison = "source2_slightly_more_credible"
        else:
            best_source = None
            comparison = "no_sources"
        
        contradiction.source_analysis = {
            "sources": sources,
            "best_source": best_source,
            "comparison": comparison,
            "timestamp": time.time()
        }
        
        return contradiction.source_analysis
    
    @staticmethod
    def calculate_nlp_metrics(contradiction, nlp_model) -> Dict[str, Any]:
        """
        Вычисляет NLP-метрики для анализа противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            nlp_model: NLP-модель для анализа
            
        Returns:
            Dict: NLP-метрики
        """
        try:
            conflicting_facts = getattr(contradiction, 'conflicting_facts', [])
            if len(conflicting_facts) < 2:
                return {}
            
            # Извлекаем тексты для сравнения
            texts = [str(fact.get("value", "")) for fact in conflicting_facts]
            
            # Инициализируем метрики
            metrics = {
                "semantic_similarity": 0.0,
                "lexical_overlap": 0.0,
                "sentiment_divergence": 0.0,
                "divergence": 0.0,
                "timestamp": time.time()
            }
            
            # Вычисляем семантическое сходство (централизованный фоллбек)
            try:
                metrics["semantic_similarity"] = float(compute_semantic_similarity(texts, nlp_model))
            except Exception as e:
                logger.warning(f"Ошибка вычисления семантического сходства: {e}")
            
            # Вычисляем лексическое перекрытие
            try:
                def preprocess(text):
                    text = text.lower()
                    text = re.sub(r'[^\w\s]', '', text)
                    return text
                
                words1 = set(preprocess(texts[0]).split())
                words2 = set(preprocess(texts[1]).split())
                lexical_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                metrics["lexical_overlap"] = float(lexical_overlap)
            except Exception as e:
                logger.warning(f"Ошибка вычисления лексического перекрытия: {e}")
            
            # Анализируем тональность (централизованный фоллбек)
            try:
                analyzer = get_sentiment_analyzer()
                s1 = polarity_scores(texts[0], analyzer)
                s2 = polarity_scores(texts[1], analyzer)
                metrics["sentiment_divergence"] = float(abs(s1.get('compound', 0.0) - s2.get('compound', 0.0)))
            except Exception as e:
                logger.warning(f"Ошибка анализа тональности: {e}")
            
            # Вычисляем взвешенное расхождение
            semantic_divergence = 1.0 - metrics["semantic_similarity"]
            lexical_divergence = 1.0 - metrics["lexical_overlap"]
            
            weighted_divergence = (
                semantic_divergence * 0.6 +
                lexical_divergence * 0.3 +
                metrics["sentiment_divergence"] * 0.1
            )
            
            metrics["divergence"] = float(weighted_divergence)
            
            # Сохраняем метрики если возможно
            if hasattr(contradiction, 'nlp_metrics'):
                contradiction.nlp_metrics = metrics
            
            return metrics
        except Exception as e:
            logger.error(f"Ошибка вычисления NLP-метрик: {e}")
            return {"timestamp": time.time(), "error": str(e)}
    
    @staticmethod
    def get_resolution_strategy(contradiction) -> Dict[str, Any]:
        """
        Определяет оптимальную стратегию разрешения противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            
        Returns:
            Dict: Стратегия разрешения
        """
        contradiction_type = ContradictionAnalysis.get_contradiction_type(contradiction)
        severity = ContradictionAnalysis.get_severity(contradiction)
        
        if contradiction_type == "numeric_conflict":
            return ContradictionAnalysis._get_numeric_conflict_strategy(contradiction, severity)
        elif contradiction_type == "boolean_conflict":
            return ContradictionAnalysis._get_boolean_conflict_strategy(contradiction, severity)
        elif contradiction_type == "exclusivity_conflict":
            return ContradictionAnalysis._get_exclusivity_conflict_strategy(contradiction, severity)
        elif contradiction_type == "hierarchy_conflict":
            return ContradictionAnalysis._get_hierarchy_conflict_strategy(contradiction, severity)
        elif contradiction_type == "response_conflict":
            return ContradictionAnalysis._get_response_conflict_strategy(contradiction, severity)
        else:
            return ContradictionAnalysis._get_general_conflict_strategy(contradiction, severity)
    
    @staticmethod
    def _get_numeric_conflict_strategy(contradiction, severity: str) -> Dict[str, Any]:
        """Возвращает стратегию для числового противоречия."""
        if severity == "high":
            return {
                "type": "evidence_based",
                "description": "Требуется сбор дополнительных данных для определения наиболее точного значения",
                "steps": [
                    "Сбор данных из авторитетных источников",
                    "Анализ методов измерения",
                    "Определение контекстных условий",
                    "Усреднение показателей"
                ],
                "expected_outcome": "Определение наиболее точного числового значения с учетом контекста"
            }
        elif severity == "medium":
            return {
                "type": "context_analysis",
                "description": "Требуется анализ контекстных условий, при которых верно каждое значение",
                "steps": [
                    "Определение условий применения",
                    "Анализ методологий",
                    "Сравнение условий измерения"
                ],
                "expected_outcome": "Уточнение контекста применения каждого значения"
            }
        else:
            return {
                "type": "accept_variability",
                "description": "Незначительные различия находятся в пределах нормальной вариативности",
                "steps": [
                    "Подтверждение нормальной вариативности",
                    "Документирование различий"
                ],
                "expected_outcome": "Признание нормальной вариативности числовых значений"
            }
    
    @staticmethod
    def _get_boolean_conflict_strategy(contradiction, severity: str) -> Dict[str, Any]:
        """Возвращает стратегию для булева противоречия."""
        if severity == "high":
            return {
                "type": "philosophical_analysis",
                "description": "Требуется глубокий анализ оснований каждого представления",
                "steps": [
                    "Анализ философских основ",
                    "Исторический контекст",
                    "Поиск точек соприкосновения",
                    "Синтез нового представления"
                ],
                "expected_outcome": "Создание интегрированного представления, учитывающего оба подхода"
            }
        elif severity == "medium":
            return {
                "type": "contextualization",
                "description": "Требуется определение условий, при которых каждое утверждение верно",
                "steps": [
                    "Анализ условий применения",
                    "Определение границ применимости",
                    "Создание условного ответа"
                ],
                "expected_outcome": "Создание контекстно-зависимого ответа"
            }
        else:
            return {
                "type": "minor_refinement",
                "description": "Незначительные расхождения в интерпретации",
                "steps": [
                    "Уточнение формулировок",
                    "Документирование нюансов"
                ],
                "expected_outcome": "Незначительная корректировка формулировок"
            }
    
    @staticmethod
    def _get_exclusivity_conflict_strategy(contradiction, severity: str) -> Dict[str, Any]:
        """Возвращает стратегию для противоречия эксклюзивности."""
        return {
            "type": "category_analysis",
            "description": "Требуется анализ категорий и подкатегорий",
            "steps": [
                "Анализ иерархии категорий",
                "Определение взаимоисключающих условий",
                "Разделение на подкатегории",
                "Переформулирование утверждений"
            ],
            "expected_outcome": "Четкое разделение условий применения 'только' и 'не только'"
        }
    
    @staticmethod
    def _get_hierarchy_conflict_strategy(contradiction, severity: str) -> Dict[str, Any]:
        """Возвращает стратегию для иерархического противоречия."""
        return {
            "type": "taxonomy_revision",
            "description": "Требуется пересмотр иерархии категорий",
            "steps": [
                "Анализ иерархических связей",
                "Выявление циклических зависимостей",
                "Определение взаимоисключающих классификаций",
                "Перестройка иерархии"
            ],
            "expected_outcome": "Создание непротиворечивой иерархической структуры"
        }
    
    @staticmethod
    def _get_response_conflict_strategy(contradiction, severity: str) -> Dict[str, Any]:
        """Возвращает стратегию для противоречия в ответах."""
        if severity == "high":
            return {
                "type": "audience_context_analysis",
                "description": "Требуется анализ контекста и целевой аудитории",
                "steps": [
                    "Анализ целевой аудитории",
                    "Определение контекстных условий",
                    "Создание условных ответов",
                    "Интеграция различных подходов"
                ],
                "expected_outcome": "Создание контекстно-зависимых ответов для разных сценариев"
            }
        else:
            return {
                "type": "response_integration",
                "description": "Требуется интеграция различных ответов",
                "steps": [
                    "Анализ условий применения",
                    "Определение границ применимости",
                    "Создание интегрированного ответа"
                ],
                "expected_outcome": "Создание ответа, объединяющего оба подхода"
            }
    
    @staticmethod
    def _get_general_conflict_strategy(contradiction, severity: str) -> Dict[str, Any]:
        """Возвращает общую стратегию для неопределенного типа противоречия."""
        return {
            "type": "comprehensive_analysis",
            "description": "Требуется комплексный анализ противоречия",
            "steps": [
                "Определение типа противоречия",
                "Анализ источников информации",
                "Исследование контекстных условий",
                "Поиск дополнительных данных",
                "Синтез решения"
            ],
            "expected_outcome": "Полное разрешение противоречия с документированием решения"
        }
    
    @staticmethod
    def get_resolution_recommendations(contradiction, source_reputation_system, nlp_model) -> List[str]:
        """
        Генерирует рекомендации по разрешению противоречия с учетом анализа.
        
        Args:
            contradiction: Экземпляр противоречия
            source_reputation_system: Система репутации источников
            nlp_model: NLP-модель для анализа
            
        Returns:
            List[str]: Рекомендации
        """
        try:
            recommendations = []
            
            # Анализируем источники
            source_analysis = ContradictionAnalysis.analyze_source_credibility(contradiction, source_reputation_system)
            
            # Вычисляем NLP-метрики
            nlp_metrics = ContradictionAnalysis.calculate_nlp_metrics(contradiction, nlp_model)
            
            # Получаем стратегию
            strategy = ContradictionAnalysis.get_resolution_strategy(contradiction)
            
            # Добавляем рекомендации из стратегии
            recommendations.extend([
                f"Рекомендуется {strategy['description']}",
                f"Основные шаги: {', '.join(strategy['steps'])}",
                f"Ожидаемый результат: {strategy['expected_outcome']}"
            ])
            
            # Добавляем рекомендации на основе анализа источников
            comparison = source_analysis.get("comparison", "neutral")
            if comparison in ["source1_more_credible", "source2_more_credible"]:
                source_num = 1 if comparison.startswith("source1") else 2
                recommendations.append(
                    f"Источник {source_num} имеет значительно более высокую репутацию. "
                    f"Рассмотрите возможность отдачи предпочтения информации из этого источника."
                )
            
            # Добавляем рекомендации на основе NLP-метрик
            if nlp_metrics.get("sentiment_divergence", 0) > 0.6:
                recommendations.append(
                    "Высокая разница в тональности указывает на принципиально разные точки зрения. "
                    "Рассмотрите возможность создания контекстно-зависимого ответа."
                )
            
            if nlp_metrics.get("divergence", 0) > 0.7:
                recommendations.append(
                    "Высокое общее расхождение указывает на фундаментальные различия. "
                    "Требуется глубокий анализ оснований каждого представления."
                )
            
            return recommendations
        except Exception as e:
            logger.error(f"Ошибка генерации рекомендаций: {e}")
            return ["Требуется дополнительный анализ противоречия"]
    
    @staticmethod
    def calculate_resolution_confidence_score(contradiction, source_reputation_system, nlp_model) -> float:
        """
        Вычисляет общий показатель уверенности в решении противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            source_reputation_system: Система репутации источников
            nlp_model: NLP-модель для анализа
            
        Returns:
            float: Показатель уверенности (0.0-1.0)
        """
        try:
            # Анализируем источники
            source_analysis = ContradictionAnalysis.analyze_source_credibility(contradiction, source_reputation_system)
            
            # Вычисляем NLP-метрики
            nlp_metrics = ContradictionAnalysis.calculate_nlp_metrics(contradiction, nlp_model)
            
            # Базовая уверенность на основе дивергенции
            divergence_level = getattr(contradiction, 'divergence_level', 0.5)
            base_confidence = 1.0 - divergence_level
            
            # Учитываем репутацию источников
            source_confidence = 0.0
            sources = source_analysis.get("sources", [])
            if sources:
                source_confidence = sum(s["credibility"] for s in sources) / len(sources)
            
            # Учитываем NLP-метрики
            nlp_confidence = 1.0 - nlp_metrics.get("divergence", divergence_level)
            
            # Взвешенное среднее
            confidence = (
                base_confidence * 0.3 +
                source_confidence * 0.4 +
                nlp_confidence * 0.3
            )
            
            # Корректируем на основе серьезности
            severity = ContradictionAnalysis.get_severity(contradiction)
            if severity == "high":
                confidence *= 0.7
            elif severity == "medium":
                confidence *= 0.85
            
            confidence = max(0.0, min(1.0, confidence))
            
            # Сохраняем уверенность если возможно
            if hasattr(contradiction, 'confidence'):
                contradiction.confidence = confidence
            
            return confidence
        except Exception as e:
            logger.error(f"Ошибка вычисления показателя уверенности: {e}")

"""Модуль для анализа противоречий в системе ЕВА"""
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

# Централизованные NLP-фоллбеки (без обязательных внешних зависимостей)
from eva_ai.nlp_fallbacks import (
    compute_semantic_similarity,
    get_sentiment_analyzer,
    polarity_scores,
    tokenize,
    get_stopwords,
)

logger = logging.getLogger("eva_ai.contradiction.analysis")
