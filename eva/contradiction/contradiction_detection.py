
"""Модуль обнаружения противоречий в системе ЕВА"""
import os
import logging
import time
import json
import re
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
import random
import hashlib
import numpy as np

# Централизованные NLP-фоллбеки
from eva.nlp_fallbacks import (
    compute_semantic_similarity,
    get_sentiment_analyzer,
    polarity_scores,
    tokenize,
    get_stopwords,
)

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # ImportError or any runtime import issue
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Singleton для sentence-transformers моделей
try:
    from eva.mlearning.sentence_transformers_cache import get_sentence_transformer
except ImportError:
    get_sentence_transformer = None

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger("eva.contradiction.detection")

class ContradictionDetector:
    """Класс для обнаружения противоречий в знаниях системы."""
    
    def __init__(self, knowledge_graph, nlp_model=None, 
                 source_reputation_system=None, detection_threshold: float = 0.65,
                 max_conflicting_facts: int = 5):
        """
        Инициализирует детектор противоречий.
        
        Args:
            knowledge_graph: Граф знаний системы
            nlp_model: NLP-модель для анализа
            source_reputation_system: Система репутации источников
            detection_threshold: Порог обнаружения противоречий (0.0-1.0)
            max_conflicting_facts: Максимальное количество конфликтующих фактов для анализа
        """
        self.knowledge_graph = knowledge_graph
        self.nlp_model = nlp_model or self._init_nlp_model()
        self.active = False
        self.detection_threshold = detection_threshold
        self.max_conflicting_facts = max_conflicting_facts

        # Инициализируем анализатор тональности и стоп-слова (безопасно)
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.stop_words = get_stopwords(("english", "russian"))
        
        logger.info("Детектор противоречий инициализирован")
    
    def _init_nlp_model(self):
        """Инициализирует NLP-модель для анализа противоречий."""
        try:
            logger.info("Инициализация NLP-модели для обнаружения противоречий...")
            
            # Используем singleton для загрузки sentence-transformers модели
            if get_sentence_transformer is not None:
                model = get_sentence_transformer('intfloat/multilingual-e5-small')
                if model is not None:
                    logger.info("NLP-модель загружена через singleton")
                    return model
            
            # Fallback на прямую загрузку
            if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
                return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            else:
                raise ImportError("sentence_transformers not available")
        except Exception as e:
            logger.error(f"Ошибка инициализации NLP-модели: {e}")
            # Возвращаем заглушку
            class DummyNLPModel:
                def encode(self, texts):
                    return np.random.rand(len(texts), 384)
            return DummyNLPModel()
    
    def detect_contradictions(self, concept: Optional[str] = None,
                            force: bool = False) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия в знаниях системы.
        
        Args:
            concept: Концепт для анализа (если None, анализируются все концепты)
            force: Принудительно запустить анализ, даже если недавно выполнялся
            
        Returns:
            List: Список обнаруженных противоречий
        """
        start_time = time.time()
        
        # Проверяем, не был ли недавно выполнен анализ
        if not force and (time.time() - self.last_detection_time < 60):
            logger.info("Обнаружение противоречий пропущено - недавно выполнено")
            return self.detected_contradictions
        
        logger.info(f"Начало обнаружения противоречий для {'всех концептов' if concept is None else concept}")
        
        # Сбрасываем обнаруженные противоречия
        self.detected_contradictions = []
        
        try:
            if concept:
                # Анализируем конкретный концепт
                contradictions = self._detect_contradictions_for_concept(concept)
                self.detected_contradictions.extend(contradictions)
            else:
                # Анализируем все концепты
                concepts = self.knowledge_graph.get_all_concepts()
                for i, c in enumerate(concepts):
                    contradictions = self._detect_contradictions_for_concept(c)
                    self.detected_contradictions.extend(contradictions)
                    
                    # Логируем прогресс каждые 100 концептов
                    if (i + 1) % 100 == 0:
                        logger.info(f"Проанализировано {i + 1}/{len(concepts)} концептов")
            
            # Сохраняем время последнего обнаружения
            self.last_detection_time = time.time()
            
            # Добавляем в историю
            self.detection_history.append({
                "timestamp": self.last_detection_time,
                "concept": concept,
                "contradictions_found": len(self.detected_contradictions),
                "duration": time.time() - start_time
            })
            
            # Ограничиваем историю
            if len(self.detection_history) > 100:
                self.detection_history = self.detection_history[-100:]
            
            logger.info(f"Обнаружено {len(self.detected_contradictions)} противоречий")
            return self.detected_contradictions
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения противоречий: {e}")
            return []
    
    def _detect_contradictions_for_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия для конкретного концепта.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            List: Список обнаруженных противоречий
        """
        # Получаем факты по концепту
        facts = self.knowledge_graph.get_facts_by_concept(concept)
        
        if len(facts) < 2:
            return []  # Недостаточно фактов для обнаружения противоречий
        
        # Группируем факты по типу отношения
        facts_by_relation = defaultdict(list)
        for fact in facts:
            relation_type = fact.get("relation_type", "related_to")
            facts_by_relation[relation_type].append(fact)
        
        contradictions = []
        
        # Проверяем каждую группу фактов
        for relation_type, fact_group in facts_by_relation.items():
            if len(fact_group) < 2:
                continue
            
            # Определяем, есть ли потенциальные противоречия в группе
            potential_contradictions = self._find_potential_contradictions(fact_group)
            
            # Анализируем каждую потенциальную пару
            for i, j in potential_contradictions:
                fact1 = fact_group[i]
                fact2 = fact_group[j]
                
                # Вычисляем уровень расхождения
                divergence = self._calculate_divergence(fact1, fact2)
                
                # Проверяем, превышает ли расхождение порог
                if divergence > self.detection_threshold:
                    # Создаем противоречие
                    contradiction = self._create_contradiction(
                        concept, 
                        [fact1, fact2], 
                        divergence,
                        relation_type=relation_type
                    )
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _find_potential_contradictions(self, facts: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        Находит потенциальные пары противоречивых фактов.
        
        Args:
            facts: Список фактов
            
        Returns:
            List[Tuple[int, int]]: Список индексов потенциальных противоречий
        """
        potential_pairs = []
        
        # Для числовых значений ищем значительные различия
        numeric_facts = []
        for i, fact in enumerate(facts):
            if isinstance(fact.get("value"), (int, float)):
                numeric_facts.append((i, fact["value"]))
        
        if numeric_facts:
            values = [v for _, v in numeric_facts]
            mean = np.mean(values)
            std = np.std(values)
            
            # Находим значения, которые значительно отличаются от среднего
            for i, (idx, value) in enumerate(numeric_facts):
                if std > 0 and abs(value - mean) > 2 * std:
                    for j, (other_idx, _) in enumerate(numeric_facts):
                        if i != j:
                            potential_pairs.append((idx, other_idx))
        
        # Для булевых значений ищем противоположные утверждения
        boolean_facts = []
        for i, fact in enumerate(facts):
            if isinstance(fact.get("value"), bool):
                boolean_facts.append((i, fact["value"]))
        
        if len(boolean_facts) > 1:
            true_count = sum(1 for _, v in boolean_facts if v)
            false_count = len(boolean_facts) - true_count
            
            if true_count > 0 and false_count > 0:
                for i, (idx1, val1) in enumerate(boolean_facts):
                    for j, (idx2, val2) in enumerate(boolean_facts):
                        if i < j and val1 != val2:
                            potential_pairs.append((idx1, idx2))
        
        # Для текстовых ответов ищем значительные семантические различия
        text_facts = []
        for i, fact in enumerate(facts):
            if isinstance(fact.get("value"), str) and "response" in fact.get("relation_type", ""):
                text_facts.append((i, fact["value"]))
        
        if len(text_facts) > 1:
            # Вычисляем семантическое сходство с фоллбеком
            texts = [text for _, text in text_facts]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    try:
                        sim = float(compute_semantic_similarity([texts[i], texts[j]], self.nlp_model))
                    except Exception:
                        sim = 0.5
                    if 1.0 - sim > self.detection_threshold:
                        potential_pairs.append((text_facts[i][0], text_facts[j][0]))
        
        # Уникализируем пары
        unique_pairs = list(set(potential_pairs))
        
        # Ограничиваем количество проверяемых пар
        if len(unique_pairs) > self.max_conflicting_facts:
            unique_pairs = random.sample(unique_pairs, self.max_conflicting_facts)
        
        return unique_pairs
    
    def _calculate_divergence(self, fact1: Dict[str, Any], 
                            fact2: Dict[str, Any]) -> float:
        """
        Вычисляет уровень расхождения между двумя фактами.
        
        Args:
            fact1: Первый факт
            fact2: Второй факт
            
        Returns:
            float: Уровень расхождения (0.0-1.0)
        """
        # Для числовых значений вычисляем относительное различие
        if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
            v1, v2 = fact1["value"], fact2["value"]
            if v1 == 0 and v2 == 0:
                return 0.0
            elif v1 == 0 or v2 == 0:
                return 1.0
            else:
                return abs(v1 - v2) / max(abs(v1), abs(v2))
        
        # Для булевых значений противоречие, если значения разные
        if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
            return 1.0 if fact1["value"] != fact2["value"] else 0.0
        
        # Для текстовых значений вычисляем семантическое расхождение
        if isinstance(fact1.get("value"), str) and isinstance(fact2.get("value"), str):
            return self._calculate_text_divergence(fact1["value"], fact2["value"])
        
        # Для других типов используем лексическое сравнение
        return self._calculate_lexical_divergence(str(fact1.get("value", "")), 
                                               str(fact2.get("value", "")))
    
    def _calculate_text_divergence(self, text1: str, text2: str) -> float:
        """
        Вычисляет семантическое расхождение между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            float: Уровень расхождения (0.0-1.0)
        """
        try:
            sim = float(compute_semantic_similarity([text1, text2], self.nlp_model))
            divergence = 1.0 - sim
            return max(0.0, min(1.0, divergence))
        except Exception as e:
            logger.error(f"Ошибка вычисления семантического расхождения: {e}")
            return 0.5  # Нейтральное значение при ошибке
    
    def _calculate_lexical_divergence(self, text1: str, text2: str) -> float:
        """
        Вычисляет лексическое расхождение между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            float: Уровень расхождения (0.0-1.0)
        """
        # Предобработка текстов
        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            words = [w for w in tokenize(text) if w.isalnum() and w not in self.stop_words]
            return set(words)
        
        words1 = preprocess(text1)
        words2 = preprocess(text2)
        
        # Вычисляем Jaccard расстояние
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 1.0  # Полное расхождение для пустых текстов
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    
    def _create_contradiction(self, concept: str, conflicting_facts: List[Dict[str, Any]], 
                            divergence_level: float, **kwargs) -> Dict[str, Any]:
        """
        Создает объект противоречия.
        
        Args:
            concept: Концепт
            conflicting_facts: Конфликтующие факты
            divergence_level: Уровень расхождения
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict: Объект противоречия
        """
        # Генерируем уникальный ID
        timestamp = int(time.time() * 1000)
        hash_str = f"{concept}{divergence_level}{timestamp}"
        contradiction_id = f"contradiction_{hashlib.md5(hash_str.encode()).hexdigest()[:8]}"
        
        # Создаем метаданные
        metadata = {
            "relation_type": kwargs.get("relation_type", "related_to"),
            "detection_method": "automatic",
            "timestamp": timestamp
        }
        
        # Добавляем информацию о репутации источников, если доступно
        if self.source_reputation_system:
            for i, fact in enumerate(conflicting_facts):
                source = fact.get("source")
                if source:
                    reputation = self.source_reputation_system.get_source_reputation(source)
                    metadata[f"source_reputation_{i}"] = reputation
        
        # Создаем противоречие
        contradiction = {
            "contradiction_id": contradiction_id,
            "concept": concept,
            "conflicting_facts": conflicting_facts,
            "divergence_level": divergence_level,
            "timestamp": time.time(),
            "status": "detected",
            "resolution": {},
            "metadata": metadata,
            "assigned_to": None,
            "resolved_at": None,
            "confidence": 0.0,
            "resolution_history": [],
            "resolution_confidence": 0.0,
            "impact_score": 0.0,
            "resolution_priority": 0.0,
            "source_analysis": {},
            "nlp_metrics": {}
        }
        
        return contradiction
    
    def detect_contradictions_in_new_fact(self, new_fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия для нового факта.
        
        Args:
            new_fact: Новый факт
            
        Returns:
            List: Список обнаруженных противоречий
        """
        concept = new_fact.get("concept")
        if not concept:
            return []
        
        # Получаем существующие факты по концепту
        existing_facts = self.knowledge_graph.get_facts_by_concept(concept)
        
        contradictions = []
        
        # Сравниваем новый факт со всеми существующими
        for fact in existing_facts:
            # Пропускаем идентичные факты
            if self._are_facts_equivalent(new_fact, fact):
                continue
            
            # Вычисляем уровень расхождения
            divergence = self._calculate_divergence(new_fact, fact)
            
            # Проверяем, превышает ли расхождение порог
            if divergence > self.detection_threshold:
                contradiction = self._create_contradiction(
                    concept,
                    [new_fact, fact],
                    divergence,
                    relation_type=new_fact.get("relation_type", "related_to")
                )
                contradictions.append(contradiction)
        
        return contradictions
    
    def _are_facts_equivalent(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> bool:
        """
        Проверяет, являются ли два факта эквивалентными.
        
        Args:
            fact1: Первый факт
            fact2: Второй факт
            
        Returns:
            bool: Являются ли факты эквивалентными
        """
        # Проверяем основные поля
        if fact1.get("concept") != fact2.get("concept"):
            return False
        
        if fact1.get("relation_type") != fact2.get("relation_type"):
            return False
        
        # Для числовых значений проверяем близость
        if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
            return abs(fact1["value"] - fact2["value"]) < 0.01 * max(abs(fact1["value"]), abs(fact2["value"]), 1)
        
        # Для других типов проверяем точное совпадение
        return str(fact1.get("value", "")) == str(fact2.get("value", ""))
    
    def analyze_fact_consistency(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует согласованность факта с существующими знаниями.
        
        Args:
            fact: Факт для анализа
            
        Returns:
            Dict: Результаты анализа
        """
        concept = fact.get("concept")
        if not concept:
            return {
                "consistent": False,
                "reason": "Concept is missing",
                "divergence": 1.0,
                "conflicting_facts": []
            }
        
        # Получаем существующие факты по концепту
        existing_facts = self.knowledge_graph.get_facts_by_concept(concept)
        
        if not existing_facts:
            return {
                "consistent": True,
                "reason": "No existing facts to compare with",
                "divergence": 0.0,
                "conflicting_facts": []
            }
        
        # Вычисляем среднее расхождение
        divergences = []
        conflicting_facts = []
        
        for existing_fact in existing_facts:
            if self._are_facts_equivalent(fact, existing_fact):
                continue
            
            divergence = self._calculate_divergence(fact, existing_fact)
            divergences.append(divergence)
            
            if divergence > self.detection_threshold:
                conflicting_facts.append(existing_fact)
        
        avg_divergence = np.mean(divergences) if divergences else 0.0
        
        return {
            "consistent": avg_divergence <= self.detection_threshold,
            "reason": "High divergence with existing facts" if avg_divergence > self.detection_threshold else "Consistent with existing knowledge",
            "divergence": avg_divergence,
            "conflicting_facts": conflicting_facts
        }
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику обнаружения противоречий.
        
        Returns:
            Dict: Статистика обнаружения
        """
        total_contradictions = len(self.detected_contradictions)
        
        # Группируем противоречия по концептам
        contradictions_by_concept = defaultdict(int)
        for contradiction in self.detected_contradictions:
            contradictions_by_concept[contradiction["concept"]] += 1
        
        # Находим концепты с наибольшим количеством противоречий
        top_concepts = sorted(
            contradictions_by_concept.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Группируем по типу противоречия
        contradiction_types = defaultdict(int)
        for contradiction in self.detected_contradictions:
            # Определяем тип противоречия
            if len(contradiction["conflicting_facts"]) == 2:
                fact1 = contradiction["conflicting_facts"][0]
                fact2 = contradiction["conflicting_facts"][1]
                
                # Проверяем числовые противоречия
                if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                    contradiction_types["numeric"] += 1
                
                # Проверяем булевы противоречия
                elif isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                    contradiction_types["boolean"] += 1
                
                # Проверяем противоречия в ответах
                elif "response" in fact1.get("relation_type", "") and "response" in fact2.get("relation_type", ""):
                    contradiction_types["response"] += 1
            
            # Проверяем по метаданным
            if "relation_type" in contradiction["metadata"]:
                relation_type = contradiction["metadata"]["relation_type"]
                if relation_type.startswith("only_") or relation_type.startswith("not_only_"):
                    contradiction_types["exclusivity"] += 1
                if relation_type in ["is_a", "part_of", "member_of"]:
                    contradiction_types["hierarchy"] += 1
        
        # Анализируем серьезность
        severity_counts = {
            "low": 0,
            "medium": 0,
            "high": 0
        }
        
        for contradiction in self.detected_contradictions:
            if contradiction["divergence_level"] < 0.4:
                severity_counts["low"] += 1
            elif contradiction["divergence_level"] < 0.7:
                severity_counts["medium"] += 1
            else:
                severity_counts["high"] += 1
        
        return {
            "total_contradictions": total_contradictions,
            "top_concepts": [{"concept": c, "count": count} for c, count in top_concepts],
            "contradiction_types": dict(contradiction_types),
            "severity_distribution": severity_counts,
            "last_detection_time": self.last_detection_time,
            "detection_history": self.detection_history[-5:]  # Последние 5 проверок
        }
    
    def start_background_detection(self, interval: int = 3600):
        """
        Запускает фоновое обнаружение противоречий.
        
        Args:
            interval: Интервал в секундах
        """
        if not self.active:
            return
        
        def detection_loop():
            while self.active:
                try:
                    self.detect_contradictions()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Ошибка в фоновом обнаружении противоречий: {e}")
                    time.sleep(60)  # Подождать перед повторной попыткой
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=detection_loop, daemon=True)
        thread.start()
        logger.info(f"Фоновое обнаружение противоречий запущено (интервал: {interval} секунд)")
    
    def stop_background_detection(self):
        """Останавливает фоновое обнаружение противоречий."""
        self.active = False
        logger.info("Фоновое обнаружение противоречий остановлено")
    
    def get_recent_contradictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Возвращает недавно обнаруженные противоречия.
        
        Args:
            limit: Ограничение количества
            
        Returns:
            List: Список противоречий
        """
        return self.detected_contradictions[-limit:]
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает противоречие по ID.
        
        Args:
            contradiction_id: ID противоречия
            
        Returns:
            Optional[Dict]: Противоречие или None
        """
        for contradiction in self.detected_contradictions:
            if contradiction["contradiction_id"] == contradiction_id:
                return contradiction
        return None
    
    def mark_contradiction_resolved(self, contradiction_id: str, resolution: Dict[str, Any]):
        """
        Отмечает противоречие как разрешенное.
        
        Args:
            contradiction_id: ID противоречия
            resolution: Решение
        """
        for contradiction in self.detected_contradictions:
            if contradiction["contradiction_id"] == contradiction_id:
                contradiction["status"] = "resolved"
                contradiction["resolution"] = resolution
                contradiction["resolved_at"] = time.time()
                contradiction["resolution_history"].append({
                    "resolver": "system",
                    "timestamp": time.time(),
                    "resolution": resolution,
                    "confidence": resolution.get("confidence", 0.8)
                })
                break
    
    def get_high_priority_contradictions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Возвращает высокоприоритетные противоречия.
        
        Args:
            limit: Ограничение количества
            
        Returns:
            List: Список высокоприоритетных противоречий
        """
        # Сортируем по приоритету разрешения (вычисляем на лету)
        def get_priority(contradiction):
            # Базовый приоритет на основе серьезности
            severity_priority = 0.3 if contradiction["divergence_level"] < 0.4 else 0.6 if contradiction["divergence_level"] < 0.7 else 0.9
            
            # Учитываем возраст противоречия
            age = time.time() - contradiction["timestamp"]
            age_priority = min(0.4, age / 86400 * 0.1)
            
            return severity_priority * 0.7 + age_priority * 0.3
        
        # Фильтруем неразрешенные противоречия
        unresolved = [c for c in self.detected_contradictions if c["status"] != "resolved"]
        
        # Сортируем по приоритету
        sorted_contradictions = sorted(unresolved, key=get_priority, reverse=True)
        
        return sorted_contradictions[:limit]
    
    def generate_contradiction_report(self) -> str:
        """
        Генерирует отчет об обнаруженных противоречиях.
        
        Returns:
            str: Текст отчета
        """
        stats = self.get_detection_statistics()
        
        report = "ОТЧЕТ ОБ ОБНАРУЖЕННЫХ ПРОТИВОРЕЧИЯХ\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Всего обнаружено противоречий: {stats['total_contradictions']}\n"
        report += f"Последнее обнаружение: {datetime.fromtimestamp(stats['last_detection_time']).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Топ концептов с противоречиями
        report += "ТОП-10 КОНЦЕПТОВ С ПРОТИВОРЕЧИЯМИ:\n"
        for i, item in enumerate(stats["top_concepts"], 1):
            report += f"{i}. {item['concept']} - {item['count']} противоречий\n"
        report += "\n"
        
        # Распределение по типам
        report += "РАСПРЕДЕЛЕНИЕ ПО ТИПАМ:\n"
        for ctype, count in stats["contradiction_types"].items():
            report += f"- {ctype}: {count}\n"
        report += "\n"
        
        # Распределение по серьезности
        report += "РАСПРЕДЕЛЕНИЕ ПО СЕРЬЕЗНОСТИ:\n"
        report += f"- Высокая: {stats['severity_distribution']['high']}\n"
        report += f"- Средняя: {stats['severity_distribution']['medium']}\n"
        report += f"- Низкая: {stats['severity_distribution']['low']}\n"
        report += "\n"
        
        # Высокоприоритетные противоречия
        high_priority = self.get_high_priority_contradictions(3)
        report += "ВЫСОКОПРИОРИТЕТНЫЕ ПРОТИВОРЕЧИЯ:\n"
        if high_priority:
            for i, contradiction in enumerate(high_priority, 1):
                severity = "высокая" if contradiction["divergence_level"] >= 0.7 else "средняя" if contradiction["divergence_level"] >= 0.4 else "низкая"
                report += f"{i}. {contradiction['concept']} (серьезность: {severity}, расхождение: {contradiction['divergence_level']:.2f})\n"
        else:
            report += "Нет высокоприоритетных противоречий\n"
        
        return report
    
    def integrate_contradiction_resolution(self, contradiction_id: str, 
                                        resolved_fact: Dict[str, Any],
                                        removed_facts: List[Dict[str, Any]]):
        """
        Интегрирует решение противоречия в граф знаний.
        
        Args:
            contradiction_id: ID противоречия
            resolved_fact: Разрешенный факт
            removed_facts: Удаленные факты
        """
        # Находим противоречие
        contradiction = self.get_contradiction_by_id(contradiction_id)
        if not contradiction:
            logger.warning(f"Противоречие {contradiction_id} не найдено")
            return
        
        # Получаем концепт
        concept = contradiction["concept"]
        
        # Удаляем конфликтующие факты из графа знаний
        for fact in removed_facts:
            self.knowledge_graph.remove_fact(fact)
        
        # Добавляем разрешенный факт
        if resolved_fact not in self.knowledge_graph.get_facts_by_concept(concept):
            self.knowledge_graph.add_fact(resolved_fact)
        
        # Отмечаем противоречие как разрешенное
        resolution = {
            "resolved_fact": resolved_fact,
            "removed_facts": removed_facts,
            "timestamp": time.time(),
            "confidence": 0.9
        }
        self.mark_contradiction_resolved(contradiction_id, resolution)
        
        logger.info(f"Противоречие {contradiction_id} интегрировано в граф знаний")
    
    def detect_hierarchy_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия в иерархии концептов.
        
        Returns:
            List: Список обнаруженных иерархических противоречий
        """
        contradictions = []
        
        # Получаем все концепты
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            # Получаем родительские концепты
            parents = self.knowledge_graph.get_relations(concept, "is_a")
            if len(parents) > 1:
                # Проверяем, не являются ли родители взаимоисключающими
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        parent1 = parents[i]
                        parent2 = parents[j]
                        
                        # Проверяем, не являются ли родители взаимоисключающими
                        if self._are_concepts_mutually_exclusive(parent1, parent2):
                            contradiction = self._create_contradiction(
                                concept,
                                [{"concept": concept, "relation": "is_a", "value": parent1},
                                 {"concept": concept, "relation": "is_a", "value": parent2}],
                                0.8,  # Высокий уровень расхождения для иерархических противоречий
                                relation_type="hierarchy"
                            )
                            contradictions.append(contradiction)
            
            # Проверяем на циклические зависимости
            if self._has_cyclic_dependency(concept):
                # Получаем цикл
                cycle = self._find_cyclic_dependency(concept)
                
                if cycle:
                    # Создаем противоречие
                    facts = [{"concept": cycle[i], "relation": "is_a", "value": cycle[(i+1) % len(cycle)]} 
                            for i in range(len(cycle))]
                    
                    contradiction = self._create_contradiction(
                        concept,
                        facts,
                        0.9,  # Очень высокий уровень расхождения для циклических зависимостей
                        relation_type="hierarchy_cycle"
                    )
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _are_concepts_mutually_exclusive(self, concept1: str, concept2: str) -> bool:
        """
        Проверяет, являются ли два концепта взаимоисключающими.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            
        Returns:
            bool: Являются ли концепты взаимоисключающими
        """
        # Проверяем по ключевым словам
        exclusive_pairs = [
            ("мужчина", "женщина"),
            ("живой", "мертвый"),
            ("истина", "ложь"),
            ("включение", "выключение"),
            ("свет", "тьма"),
            ("тепло", "холод")
        ]
        
        for pair in exclusive_pairs:
            if (concept1.lower() in pair and concept2.lower() in pair) and (pair[0] != pair[1]):
                return True
        
        # В реальной системе здесь будет более глубокий анализ
        return False
    
    def _has_cyclic_dependency(self, concept: str) -> bool:
        """
        Проверяет, есть ли циклическая зависимость для концепта.
        
        Args:
            concept: Концепт
            
        Returns:
            bool: Есть ли циклическая зависимость
        """
        visited = set()
        path = []
        
        def dfs(node):
            if node in visited:
                return False
            if node in path:
                return True
            
            visited.add(node)
            path.append(node)
            
            # Получаем дочерние концепты
            children = self.knowledge_graph.get_relations(node, "is_a")
            for child in children:
                if dfs(child):
                    return True
            
            path.pop()
            return False
        
        return dfs(concept)
    
    def _find_cyclic_dependency(self, concept: str) -> Optional[List[str]]:
        """
        Находит циклическую зависимость для концепта.
        
        Args:
            concept: Концепт
            
        Returns:
            List: Цикл зависимостей
        """
        path = []
        
        def dfs(node, current_path):
            if node in current_path:
                # Найден цикл
                idx = current_path.index(node)
                return current_path[idx:]
            
            current_path.append(node)
            
            # Получаем дочерние концепты
            children = self.knowledge_graph.get_relations(node, "is_a")
            for child in children:
                cycle = dfs(child, current_path.copy())
                if cycle:
                    return cycle
            
            return None
        
        return dfs(concept, [])
    
    def detect_exclusivity_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия эксклюзивности.
        
        Returns:
            List: Список обнаруженных противоречий эксклюзивности
        """
        contradictions = []
        
        # Получаем все концепты
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            # Проверяем утверждения "только"
            only_relations = self.knowledge_graph.get_relations(concept, "only_in")
            not_only_relations = self.knowledge_graph.get_relations(concept, "not_only_in")
            
            # Проверяем пересечение
            for only_rel in only_relations:
                for not_only_rel in not_only_relations:
                    if only_rel == not_only_rel:
                        contradiction = self._create_contradiction(
                            concept,
                            [{"concept": concept, "relation": "only_in", "value": only_rel},
                             {"concept": concept, "relation": "not_only_in", "value": not_only_rel}],
                            0.85,  # Высокий уровень расхождения
                            relation_type="exclusivity"
                        )
                        contradictions.append(contradiction)
        
        return contradictions
    
    def detect_temporal_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает временные противоречия (факты, которые верны в разные периоды времени).
        
        Returns:
            List: Список обнаруженных временных противоречий
        """
        contradictions = []
        
        # Получаем все концепты
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            facts = self.knowledge_graph.get_facts_by_concept(concept)
            
            # Группируем факты по временным периодам
            facts_by_period = defaultdict(list)
            for fact in facts:
                period = fact.get("temporal_context", "current")
                facts_by_period[period].append(fact)
            
            # Сравниваем факты из разных периодов
            periods = list(facts_by_period.keys())
            for i in range(len(periods)):
                for j in range(i + 1, len(periods)):
                    period1 = periods[i]
                    period2 = periods[j]
                    
                    for fact1 in facts_by_period[period1]:
                        for fact2 in facts_by_period[period2]:
                            # Пропускаем идентичные факты
                            if self._are_facts_equivalent(fact1, fact2):
                                continue
                            
                            # Вычисляем уровень расхождения
                            divergence = self._calculate_divergence(fact1, fact2)
                            
                            # Проверяем, превышает ли расхождение порог
                            if divergence > self.detection_threshold:
                                contradiction = self._create_contradiction(
                                    concept,
                                    [fact1, fact2],
                                    divergence,
                                    relation_type="temporal",
                                    temporal_context=f"{period1} vs {period2}"
                                )
                                contradictions.append(contradiction)
        
        return contradictions
    
    def detect_contextual_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает контекстуальные противоречия (факты, которые верны в разных контекстах).
        
        Returns:
            List: Список обнаруженных контекстуальных противоречий
        """
        contradictions = []
        
        # Получаем все концепты
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            facts = self.knowledge_graph.get_facts_by_concept(concept)
            
            # Группируем факты по контексту
            facts_by_context = defaultdict(list)
            for fact in facts:
                context = fact.get("context", "general")
                facts_by_context[context].append(fact)
            
            # Сравниваем факты из разных контекстов
            contexts = list(facts_by_context.keys())
            for i in range(len(contexts)):
                for j in range(i + 1, len(contexts)):
                    context1 = contexts[i]
                    context2 = contexts[j]
                    
                    for fact1 in facts_by_context[context1]:
                        for fact2 in facts_by_context[context2]:
                            # Пропускаем идентичные факты
                            if self._are_facts_equivalent(fact1, fact2):
                                continue
                            
                            # Вычисляем уровень расхождения
                            divergence = self._calculate_divergence(fact1, fact2)
                            
                            # Проверяем, превышает ли расхождение порог
                            if divergence > self.detection_threshold:
                                contradiction = self._create_contradiction(
                                    concept,
                                    [fact1, fact2],
                                    divergence,
                                    relation_type="contextual",
                                    context=f"{context1} vs {context2}"
                                )
                                contradictions.append(contradiction)
        
        return contradictions
    
    def detect_all_specialized_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает все специализированные типы противоречий.
        
        Returns:
            List: Список всех обнаруженных специализированных противоречий
        """
        all_contradictions = []
        
        # Иерархические противоречия
        hierarchy_contradictions = self.detect_hierarchy_contradictions()
        all_contradictions.extend(hierarchy_contradictions)
        
        # Противоречия эксклюзивности
        exclusivity_contradictions = self.detect_exclusivity_contradictions()
        all_contradictions.extend(exclusivity_contradictions)
        
        # Временные противоречия
        temporal_contradictions = self.detect_temporal_contradictions()
        all_contradictions.extend(temporal_contradictions)
        
        # Контекстуальные противоречия
        contextual_contradictions = self.detect_contextual_contradictions()
        all_contradictions.extend(contextual_contradictions)
        
        # Добавляем в основной список
        self.detected_contradictions.extend(all_contradictions)
        
        return all_contradictions

# Singleton для sentence-transformers моделей
try:
    from eva.mlearning.sentence_transformers_cache import get_sentence_transformer
except ImportError:
    get_sentence_transformer = None

logger = logging.getLogger("eva.contradiction.detection")

