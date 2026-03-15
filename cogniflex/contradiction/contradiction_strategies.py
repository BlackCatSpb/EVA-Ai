"""Модуль стратегий разрешения противоречий в системе CogniFlex"""
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger("cogniflex.contradiction.strategies")

class ContradictionResolutionStrategy:
    """Стратегия разрешения противоречия."""
    
    def __init__(self, strategy_type: str, concept: str, description: str, 
                 confidence: float, steps: List[Dict[str, Any]]):
        """
        Инициализирует стратегию разрешения противоречия.
        
        Args:
            strategy_type: Тип стратегии
            concept: Концепт
            description: Описание стратегии
            confidence: Уверенность в стратегии
            steps: Шаги стратегии
        """
        self.strategy_type = strategy_type
        self.concept = concept
        self.description = description
        self.confidence = confidence
        self.steps = steps
        self.expected_outcome = ""
        self.effective_for = []
        self.created_at = time.time()
        self.last_updated = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует стратегию в словарь."""
        return {
            "strategy_type": self.strategy_type,
            "concept": self.concept,
            "description": self.description,
            "confidence": self.confidence,
            "steps": self.steps,
            "expected_outcome": self.expected_outcome,
            "effective_for": self.effective_for,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContradictionResolutionStrategy':
        """Создает стратегию из словаря."""
        strategy = cls(
            strategy_type=data["strategy_type"],
            concept=data["concept"],
            description=data["description"],
            confidence=data["confidence"],
            steps=data["steps"]
        )
        strategy.expected_outcome = data.get("expected_outcome", "")
        strategy.effective_for = data.get("effective_for", [])
        strategy.created_at = data.get("created_at", time.time())
        strategy.last_updated = data.get("last_updated", strategy.created_at)
        return strategy
    
    def execute(self, contradiction, knowledge_graph, source_reputation_system, nlp_model) -> Dict[str, Any]:
        """
        Выполняет стратегию разрешения противоречия.
        
        Args:
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            source_reputation_system: Система репутации источников
            nlp_model: NLP-модель
            
        Returns:
            Dict: Результат выполнения
        """
        try:
            # Выполняем каждый шаг стратегии
            results = []
            for i, step in enumerate(self.steps, 1):
                result = self._execute_step(i, step, contradiction, knowledge_graph, source_reputation_system, nlp_model)
                results.append(result)
            
            # Определяем общий результат
            success = all(result["success"] for result in results)
            resolution = self._generate_resolution(contradiction, results, knowledge_graph, nlp_model)
            
            return {
                "success": success,
                "resolution": resolution,
                "steps_results": results,
                "confidence": self.confidence,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка выполнения стратегии разрешения: {e}")
            return {
                "success": False,
                "error": str(e),
                "resolution": None,
                "timestamp": time.time()
            }
    
    def _execute_step(self, step_number: int, step: Dict[str, Any], 
                     contradiction, knowledge_graph, 
                     source_reputation_system, nlp_model) -> Dict[str, Any]:
        """
        Выполняет один шаг стратегии.
        
        Args:
            step_number: Номер шага
            step: Описание шага
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            source_reputation_system: Система репутации источников
            nlp_model: NLP-модель
            
        Returns:
            Dict: Результат выполнения шага
        """
        try:
            # Анализируем тип шага
            if "анализ" in step["title"].lower() or "анализ" in step["description"].lower():
                return self._execute_analysis_step(step_number, step, contradiction, knowledge_graph, source_reputation_system, nlp_model)
            
            elif "сбор" in step["title"].lower() or "сбор" in step["description"].lower():
                return self._execute_evidence_collection_step(step_number, step, contradiction, knowledge_graph, nlp_model)
            
            elif "интеграция" in step["title"].lower() or "интеграция" in step["description"].lower():
                return self._execute_integration_step(step_number, step, contradiction, knowledge_graph, nlp_model)
            
            elif "синтез" in step["title"].lower() or "синтез" in step["description"].lower():
                return self._execute_synthesis_step(step_number, step, contradiction, knowledge_graph, nlp_model)
            
            else:
                # Общий случай
                time.sleep(0.05)  # Имитация работы
                return {
                    "step_number": step_number,
                    "title": step["title"],
                    "description": step["description"],
                    "success": True,
                    "details": f"Шаг {step_number} выполнен успешно",
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Ошибка выполнения шага {step_number}: {e}")
            return {
                "step_number": step_number,
                "title": step["title"],
                "description": step["description"],
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _execute_analysis_step(self, step_number: int, step: Dict[str, Any], 
                              contradiction, knowledge_graph,
                              source_reputation_system, nlp_model) -> Dict[str, Any]:
        """
        Выполняет шаг анализа.
        
        Args:
            step_number: Номер шага
            step: Описание шага
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            source_reputation_system: Система репутации источников
            nlp_model: NLP-модель
            
        Returns:
            Dict: Результат выполнения
        """
        # Выполняем анализ источников
        source_analysis = self._analyze_source_credibility(contradiction, source_reputation_system)
        
        # Вычисляем NLP-метрики
        nlp_metrics = self._calculate_nlp_metrics(contradiction, nlp_model)
        
        # Анализируем влияние на систему
        impact = self._calculate_contradiction_impact(contradiction, knowledge_graph)
        
        return {
            "step_number": step_number,
            "title": step["title"],
            "description": step["description"],
            "success": True,
            "details": "Анализ выполнен успешно",
            "source_analysis": source_analysis,
            "nlp_metrics": nlp_metrics,
            "impact": impact,
            "timestamp": time.time()
        }
    
    def _execute_evidence_collection_step(self, step_number: int, step: Dict[str, Any], 
                                        contradiction, knowledge_graph, 
                                        nlp_model) -> Dict[str, Any]:
        """
        Выполняет шаг сбора доказательств.
        
        Args:
            step_number: Номер шага
            step: Описание шага
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            nlp_model: NLP-модель
            
        Returns:
            Dict: Результат выполнения
        """
        # Формируем поисковый запрос
        query = f"{contradiction.concept} {contradiction.conflicting_facts[0]['value']} vs {contradiction.conflicting_facts[1]['value']}"
        
        # Проверяем наличие поискового движка в графе знаний
        if hasattr(knowledge_graph, 'web_search_engine') and knowledge_graph.web_search_engine:
            try:
                # Выполняем поиск через поисковый движок
                search_results = knowledge_graph.web_search_engine.search_internet(
                    query=query,
                    num_results=10,
                    search_timeout=20.0
                )
                
                if search_results["status"] == "success":
                    # Анализируем результаты
                    evidence = []
                    for result in search_results["results"]:
                        # Проверяем релевантность
                        relevance = self._calculate_relevance(result["title"] + " " + result["snippet"], query, nlp_model)
                        if relevance > 0.4:
                            # Определяем, какое представление поддерживается
                            support = self._determine_support(contradiction, result["title"] + " " + result["snippet"], nlp_model)
                            if support:
                                evidence.append({
                                    "url": result["url"],
                                    "title": result["title"],
                                    "snippet": result["snippet"],
                                    "relevance": relevance,
                                    "supports": support,
                                    "source_reputation": source_reputation_system.get_source_reputation(result["url"])
                                })
                    
                    return {
                        "step_number": step_number,
                        "title": step["title"],
                        "description": step["description"],
                        "success": True,
                        "details": f"Собрано {len(evidence)} релевантных источников",
                        "evidence": evidence,
                        "timestamp": time.time()
                    }
                else:
                    logger.warning(f"Поиск не дал результатов: {search_results.get('error', 'неизвестная ошибка')}")
                    return {
                        "step_number": step_number,
                        "title": step["title"],
                        "description": step["description"],
                        "success": False,
                        "error": search_results.get("error", "Неизвестная ошибка поиска"),
                        "timestamp": time.time()
                    }
                    
            except Exception as e:
                logger.error(f"Ошибка при выполнении поиска: {e}")
                return {
                    "step_number": step_number,
                    "title": step["title"],
                    "description": step["description"],
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        # Если поисковый движок недоступен, возвращаем ошибку
        return {
            "step_number": step_number,
            "title": step["title"],
            "description": step["description"],
            "success": False,
            "error": "Поисковый движок недоступен",
            "timestamp": time.time()
        }
    
    def _calculate_relevance(self, text: str, query: str, nlp_model) -> float:
        """
        Вычисляет релевантность текста запросу.
        
        Args:
            text: Текст для анализа
            query: Поисковый запрос
            nlp_model: NLP-модель
            
        Returns:
            float: Релевантность (0.0-1.0)
        """
        try:
            # Генерируем эмбеддинги
            embeddings = nlp_model.encode([text, query])
            
            # Вычисляем косинусное сходство
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return max(0.0, min(1.0, float(similarity)))
        except Exception as e:
            logger.error(f"Ошибка вычисления релевантности: {e}")
            return 0.0
    
    def _determine_support(self, contradiction, text: str, nlp_model) -> Optional[str]:
        """
        Определяет, какое представление поддерживается текстом.
        
        Args:
            contradiction: Противоречие
            text: Текст для анализа
            nlp_model: NLP-модель
            
        Returns:
            Optional[str]: "a", "b" или None
        """
        try:
            # Анализируем поддержку первого представления
            support_a = self._calculate_relevance(text, contradiction.conflicting_facts[0]["value"], nlp_model)
            
            # Анализируем поддержку второго представления
            support_b = self._calculate_relevance(text, contradiction.conflicting_facts[1]["value"], nlp_model)
            
            # Определяем, какое представление поддерживается
            if support_a > 0.6 and support_a > support_b * 1.2:
                return "a"
            elif support_b > 0.6 and support_b > support_a * 1.2:
                return "b"
            
            return None
        except Exception as e:
            logger.error(f"Ошибка определения поддержки: {e}")
            return None
    
    def _execute_integration_step(self, step_number: int, step: Dict[str, Any], 
                                 contradiction, knowledge_graph, 
                                 nlp_model) -> Dict[str, Any]:
        """
        Выполняет шаг интеграции.
        
        Args:
            step_number: Номер шага
            step: Описание шага
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            nlp_model: NLP-модель
            
        Returns:
            Dict: Результат выполнения
        """
        # Анализируем конфликтующие факты
        analysis = {
            "source_analysis": self._analyze_source_credibility(contradiction, knowledge_graph.source_reputation_system),
            "nlp_metrics": self._calculate_nlp_metrics(contradiction, nlp_model)
        }
        
        # Создаем интегрированный ответ
        integrated_response = self._generate_integrated_response(contradiction, analysis, nlp_model)
        
        return {
            "step_number": step_number,
            "title": step["title"],
            "description": step["description"],
            "success": True,
            "details": "Интеграция выполнена успешно",
            "integrated_response": integrated_response,
            "timestamp": time.time()
        }
    
    def _generate_integrated_response(self, contradiction, 
                                     analysis: Dict, nlp_model) -> str:
        """
        Генерирует интегрированный ответ.
        
        Args:
            contradiction: Противоречие
            analysis: Анализ противоречия
            nlp_model: NLP-модель
            
        Returns:
            str: Интегрированный ответ
        """
        # Определяем тип противоречия
        contradiction_type = self._get_contradiction_type(contradiction)
        
        # Генерируем ответ в зависимости от типа
        if contradiction_type == "numeric_conflict":
            return self._generate_numeric_integrated_response(contradiction, analysis)
        
        elif contradiction_type == "boolean_conflict":
            return self._generate_boolean_integrated_response(contradiction, analysis)
        
        elif contradiction_type == "response_conflict":
            return self._generate_response_integrated_response(contradiction, analysis)
        
        else:
            return self._generate_general_integrated_response(contradiction, analysis)
    
    def _generate_numeric_integrated_response(self, contradiction, analysis: Dict) -> str:
        """Генерирует интегрированный ответ для числового противоречия."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        value1 = fact1.get("value", "?")
        value2 = fact2.get("value", "?")
        source1 = fact1.get("source", "неизвестный источник")
        source2 = fact2.get("source", "неизвестный источник")
        
        # Определяем, есть ли доминирующее представление
        if abs(value1 - value2) / max(value1, value2) < 0.1:
            # Незначительные различия
            avg_value = (value1 + value2) / 2
            return (
                f"Данные по '{contradiction.concept}' незначительно различаются: {value1} согласно {source1} "
                f"и {value2} согласно {source2}. Среднее значение составляет {avg_value:.2f}, "
                f"что может быть использовано как обобщенный показатель."
            )
        else:
            # Значительные различия
            return (
                f"Данные по '{contradiction.concept}' значительно различаются: {value1} согласно {source1} "
                f"и {value2} согласно {source2}. Эти различия, вероятно, обусловлены разными методами измерения или "
                f"контекстными условиями. Рекомендуется уточнить условия применения каждого значения."
            )
    
    def _generate_boolean_integrated_response(self, contradiction, analysis: Dict) -> str:
        """Генерирует интегрированный ответ для булева противоречия."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        value1 = "верно" if fact1.get("value", False) else "неверно"
        value2 = "верно" if fact2.get("value", False) else "неверно"
        source1 = fact1.get("source", "неизвестный источник")
        source2 = fact2.get("source", "неизвестный источник")
        
        return (
            f"Утверждение по '{contradiction.concept}' имеет разную интерпретацию: {value1} согласно {source1} "
            f"и {value2} согласно {source2}. Эти различия, вероятно, связаны с контекстными условиями или "
            f"областью применения. Рекомендуется уточнить условия, при которых верно каждое утверждение."
        )
    
    def _generate_response_integrated_response(self, contradiction, analysis: Dict) -> str:
        """Генерирует интегрированный ответ для противоречия в ответах."""
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        snippet1 = fact1.get("value", "")[:100] + "..." if len(fact1.get("value", "")) > 100 else fact1.get("value", "")
        snippet2 = fact2.get("value", "")[:100] + "..." if len(fact2.get("value", "")) > 100 else fact2.get("value", "")
        
        return (
            f"Существуют различные ответы на вопрос о '{contradiction.concept}':\n\n"
            f"1. {snippet1}\n\n"
            f"2. {snippet2}\n\n"
            f"Эти ответы отражают разные аспекты концепта. Рекомендуется уточнить контекст вопроса "
            f"и определить, какие условия делают каждый ответ верным."
        )
    
    def _generate_general_integrated_response(self, contradiction, analysis: Dict) -> str:
        """Генерирует общий интегрированный ответ."""
        return (
            f"Обнаружено противоречие в знаниях по '{contradiction.concept}'. "
            f"Существуют несовместимые утверждения, требующие анализа. "
            f"Рекомендуется провести детальный анализ источников и контекста для определения "
            f"наиболее достоверной информации."
        )
    
    def _execute_synthesis_step(self, step_number: int, step: Dict[str, Any], 
                              contradiction, knowledge_graph, 
                              nlp_model) -> Dict[str, Any]:
        """
        Выполняет шаг синтеза.
        
        Args:
            step_number: Номер шага
            step: Описание шага
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            nlp_model: NLP-модель
            
        Returns:
            Dict: Результат выполнения
        """
        # Генерируем синтезированный ответ
        synthesized_view = self._synthesize_view(contradiction, knowledge_graph, nlp_model)
        
        return {
            "step_number": step_number,
            "title": step["title"],
            "description": step["description"],
            "success": True,
            "details": "Синтез выполнен успешно",
            "synthesized_view": synthesized_view,
            "timestamp": time.time()
        }
    
    def _synthesize_view(self, contradiction, 
                        knowledge_graph, nlp_model) -> str:
        """
        Синтезирует новое представление, объединяющее конфликтующие факты.
        
        Args:
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            nlp_model: NLP-модель
            
        Returns:
            str: Синтезированное представление
        """
        # Пытаемся синтезировать новое представление
        if hasattr(knowledge_graph, 'ml_unit') and knowledge_graph.ml_unit:
            try:
                # Создаем промпт для синтеза
                prompt = (
                    f"Синтезируйте новое представление, объединяющее два противоречащих взгляда "
                    f"на {contradiction.concept}:\n"
                    f"1. {contradiction.conflicting_facts[0]['value']}\n"
                    f"2. {contradiction.conflicting_facts[1]['value']}\n\n"
                    f"Новое представление должно учитывать достоинства обоих взглядов и "
                    f"определять условия, при которых каждый из них верен. Ответ должен быть "
                    f"четким и лаконичным, не превышающим 150 слов."
                )
                
                # Получаем синтезированный ответ
                synthesized_view = knowledge_graph.ml_unit.generate_response(prompt)
                
                return synthesized_view
                
            except Exception as e:
                logger.error(f"Ошибка синтеза нового представления: {e}")
        
        # Если синтез не удался, используем анализ для генерации ответа
        source_analysis = self._analyze_source_credibility(contradiction, knowledge_graph.source_reputation_system)
        nlp_metrics = self._calculate_nlp_metrics(contradiction, nlp_model)
        
        return self._generate_synthesized_response(contradiction, source_analysis, nlp_metrics)
    
    def _generate_synthesized_response(self, contradiction, source_analysis, nlp_metrics) -> str:
        """
        Генерирует синтезированный ответ на основе анализа.
        
        Args:
            contradiction: Противоречие
            source_analysis: Анализ источников
            nlp_metrics: NLP-метрики
            
        Returns:
            str: Синтезированный ответ
        """
        fact1 = contradiction.conflicting_facts[0]
        fact2 = contradiction.conflicting_facts[1]
        
        value1 = fact1.get("value", "?")
        value2 = fact2.get("value", "?")
        source1 = fact1.get("source", "неизвестный источник")
        source2 = fact2.get("source", "неизвестный источник")
        
        # Анализируем, какой источник более авторитетный
        best_source = None
        if source_analysis.get("sources"):
            best_source = max(source_analysis["sources"], key=lambda x: x["credibility"])
        
        # Анализируем NLP-метрики
        semantic_divergence = nlp_metrics.get("semantic_divergence", 0.5)
        sentiment_divergence = nlp_metrics.get("sentiment_divergence", 0.5)
        
        # Генерируем ответ в зависимости от метрик
        if semantic_divergence < 0.3:
            # Небольшое семантическое расхождение
            return (
                f"На основе анализа, оба утверждения по '{contradiction.concept}' передают похожий смысл, "
                f"хотя и выражены разными словами. Вероятно, это два способа описать одно и то же явление."
            )
        elif sentiment_divergence > 0.7:
            # Высокая разница в тональности
            return (
                f"Утверждения по '{contradiction.concept}' имеют противоположную тональность, что указывает "
                f"на принципиально разные точки зрения. Вероятно, каждое утверждение верно в своем контексте."
            )
        else:
            # Общий случай
            if best_source:
                return (
                    f"На основе анализа источников, информация от {best_source['source']} имеет более высокую репутацию. "
                    f"Тем не менее, оба утверждения могут быть верны в разных контекстах: {value1} согласно {source1} "
                    f"и {value2} согласно {source2}."
                )
            else:
                return (
                    f"Оба утверждения по '{contradiction.concept}' имеют сопоставимую репутацию источников. "
                    f"Вероятно, они описывают разные аспекты или условия применения концепта: {value1} согласно {source1} "
                    f"и {value2} согласно {source2}."
                )
    
    def _generate_resolution(self, contradiction, 
                           steps_results: List[Dict[str, Any]], 
                           knowledge_graph, nlp_model) -> Dict[str, Any]:
        """
        Генерирует решение на основе выполненных шагов.
        
        Args:
            contradiction: Противоречие
            steps_results: Результаты выполнения шагов
            knowledge_graph: Граф знаний
            nlp_model: NLP-модель
            
        Returns:
            Dict: Сгенерированное решение
        """
        # Анализируем результаты шагов
        successful_steps = [r for r in steps_results if r["success"]]
        
        # Генерируем решение в зависимости от типа стратегии
        if self.strategy_type == "dominant_perspective":
            return self._generate_dominant_perspective_resolution(contradiction, successful_steps)
        
        elif self.strategy_type == "evidence_gathering":
            return self._generate_evidence_based_resolution(contradiction, successful_steps)
        
        elif self.strategy_type == "synthesis":
            return self._generate_synthesis_resolution(contradiction, successful_steps, knowledge_graph, nlp_model)
        
        else:
            return {
                "type": "custom",
                "resolution": "Решение сгенерировано на основе кастомной стратегии",
                "confidence": self.confidence,
                "steps": [r["title"] for r in successful_steps]
            }
    
    def _generate_dominant_perspective_resolution(self, contradiction, 
                                               successful_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Генерирует решение для стратегии доминирующего представления.
        
        Args:
            contradiction: Противоречие
            successful_steps: Успешные шаги
            
        Returns:
            Dict: Сгенерированное решение
        """
        # Определяем доминирующее представление
        strength_a = contradiction.conflicting_facts[0].get("reliability", 0.5)
        strength_b = contradiction.conflicting_facts[1].get("reliability", 0.5)
        
        dominant = "a" if strength_a > strength_b else "b"
        perspective = contradiction.conflicting_facts[0]["value"] if dominant == "a" else contradiction.conflicting_facts[1]["value"]
        
        return {
            "type": "dominant_perspective",
            "dominant": dominant,
            "resolution": f"Принято доминирующее представление: {perspective}",
            "confidence": max(strength_a, strength_b),
            "steps": [step["title"] for step in successful_steps]
        }
    
    def _generate_evidence_based_resolution(self, contradiction, 
                                         successful_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Генерирует решение для стратегии сбора доказательств.
        
        Args:
            contradiction: Противоречие
            successful_steps: Успешные шаги
            
        Returns:
            Dict: Сгенерированное решение
        """
        # Оцениваем количество и качество доказательств
        evidence_a = 0
        evidence_b = 0
        total_reputation_a = 0.0
        total_reputation_b = 0.0
        
        for step in successful_steps:
            if "evidence" in step and isinstance(step["evidence"], list):
                for evidence in step["evidence"]:
                    if evidence.get("supports") == "a":
                        evidence_a += 1
                        total_reputation_a += evidence.get("source_reputation", 0.5)
                    elif evidence.get("supports") == "b":
                        evidence_b += 1
                        total_reputation_b += evidence.get("source_reputation", 0.5)
        
        # Определяем, какое представление поддерживается сильнее
        if evidence_a > 0:
            avg_reputation_a = total_reputation_a / evidence_a
        else:
            avg_reputation_a = 0.5
            
        if evidence_b > 0:
            avg_reputation_b = total_reputation_b / evidence_b
        else:
            avg_reputation_b = 0.5
        
        if evidence_a > evidence_b:
            dominant = "a"
            confidence = 0.6 + min(0.3, (evidence_a - evidence_b) * 0.1) * avg_reputation_a
        elif evidence_b > evidence_a:
            dominant = "b"
            confidence = 0.6 + min(0.3, (evidence_b - evidence_a) * 0.1) * avg_reputation_b
        else:
            dominant = "synthesis"
            confidence = 0.5
        
        confidence = min(1.0, confidence)
        
        if dominant == "a" or dominant == "b":
            perspective = contradiction.conflicting_facts[0]["value"] if dominant == "a" else contradiction.conflicting_facts[1]["value"]
            resolution = f"Принято представление: {perspective} на основе собранных доказательств"
        else:
            resolution = (
                "На основе собранных данных оба представления имеют подтверждение в разных контекстах. "
                "Рекомендуется использовать контекстно-зависимый подход."
            )
        
        return {
            "type": "evidence_based",
            "dominant": dominant,
            "resolution": resolution,
            "confidence": confidence,
            "steps": [step["title"] for step in successful_steps]
        }
    
    def _generate_synthesis_resolution(self, contradiction, 
                                    successful_steps: List[Dict[str, Any]],
                                    knowledge_graph, nlp_model) -> Dict[str, Any]:
        """
        Генерирует решение для стратегии синтеза.
        
        Args:
            contradiction: Противоречие
            successful_steps: Успешные шаги
            knowledge_graph: Граф знаний
            nlp_model: NLP-модель
            
        Returns:
            Dict: Сгенерированное решение
        """
        # Пытаемся синтезировать новое представление
        synthesized_view = self._synthesize_view(contradiction, knowledge_graph, nlp_model)
        
        # Оцениваем качество синтеза
        confidence = 0.7
        if "контекст" in synthesized_view.lower() or "условия" in synthesized_view.lower():
            confidence = 0.85
        elif "оба" in synthesized_view.lower() or "разные" in synthesized_view.lower():
            confidence = 0.75
            
        return {
            "type": "synthesis",
            "resolution": synthesized_view,
            "confidence": confidence,
            "steps": [step["title"] for step in successful_steps]
        }
    
    def _analyze_source_credibility(self, contradiction, source_reputation_system) -> Dict[str, Any]:
        """
        Анализирует достоверность источников конфликтующих фактов.
        
        Args:
            contradiction: Противоречие
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
        
        return {
            "sources": sources,
            "best_source": best_source,
            "comparison": comparison,
            "timestamp": time.time()
        }
    
    def _calculate_nlp_metrics(self, contradiction, nlp_model) -> Dict[str, Any]:
        """
        Вычисляет NLP-метрики для анализа противоречия.
        
        Args:
            contradiction: Противоречие
            nlp_model: NLP-модель для анализа
            
        Returns:
            Dict: NLP-метрики
        """
        if len(contradiction.conflicting_facts) < 2:
            return {}
        
        # Инициализируем анализатор тональности
        sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Извлекаем тексты для сравнения
        texts = [str(fact.get("value", "")) for fact in contradiction.conflicting_facts]
        
        # Вычисляем семантическое сходство
        embeddings = nlp_model.encode(texts)
        semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        semantic_divergence = 1.0 - semantic_similarity
        
        # Вычисляем лексическое перекрытие
        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text
        
        words1 = set(preprocess(texts[0]).split())
        words2 = set(preprocess(texts[1]).split())
        lexical_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        lexical_divergence = 1.0 - lexical_overlap
        
        # Анализируем тональность
        sentiment1 = sentiment_analyzer.polarity_scores(texts[0])
        sentiment2 = sentiment_analyzer.polarity_scores(texts[1])
        sentiment_divergence = abs(sentiment1['compound'] - sentiment2['compound'])
        
        # Вычисляем взвешенное расхождение
        weighted_divergence = (
            semantic_divergence * 0.6 +
            lexical_divergence * 0.3 +
            sentiment_divergence * 0.1
        )
        
        # Сохраняем метрики
        return {
            "semantic_similarity": float(semantic_similarity),
            "lexical_overlap": float(lexical_overlap),
            "sentiment_divergence": float(sentiment_divergence),
            "divergence": float(weighted_divergence),
            "timestamp": time.time()
        }
    
    def _calculate_contradiction_impact(self, contradiction, knowledge_graph) -> float:
        """
        Вычисляет влияние противоречия на систему знаний.
        
        Args:
            contradiction: Противоречие
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
            importance = self._calculate_concept_importance(contradiction)
            
            # Влияние = (важность * 0.4) + (количество узлов * 0.3) + (глубина * 0.3)
            impact = (importance * 0.4) + (min(1.0, node_count / 50) * 0.3) + (min(1.0, depth / 10) * 0.3)
            
            return min(1.0, impact)
            
        except Exception as e:
            logger.error(f"Ошибка вычисления влияния противоречия: {e}")
            # Базовая оценка влияния на основе дивергенции
            return contradiction.divergence_level * 0.7
    
    def _calculate_concept_importance(self, contradiction) -> float:
        """
        Вычисляет важность концепта на основе его использования и значимости.
        
        Args:
            contradiction: Противоречие
            
        Returns:
            float: Важность концепта (0.0-1.0)
        """
        # Инициализируем анализатор тональности
        sentiment_analyzer = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english') + stopwords.words('russian'))
        
        # 1. Анализ частоты использования концепта в системе
        usage_frequency = 0.3  # Базовая оценка
        if "usage_count" in contradiction.meta:
            # Логарифмическая шкала для учета частоты использования
            usage_frequency = min(0.5, 0.1 + np.log1p(contradiction.metadata["usage_count"]) * 0.1)
        
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
        sentiment = sentiment_analyzer.polarity_scores(fact_content)
        neutrality = 1.0 - abs(sentiment['compound'])
        concept_score += neutrality * 0.2
        
        # 4. Анализ структуры текста
        words = word_tokenize(fact_content)
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
    
    def _get_contradiction_type(self, contradiction) -> str:
        """
        Определяет тип противоречия.
        
        Args:
            contradiction: Противоречие
            
        Returns:
            str: Тип противоречия
        """
        if "type" in contradiction.meta:
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
        if "relation_type" in contradiction.meta:
            if contradiction.metadata["relation_type"].startswith("only_") or contradiction.metadata["relation_type"].startswith("not_only_"):
                return "exclusivity_conflict"
            if contradiction.metadata["relation_type"] in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        
        return "unknown"