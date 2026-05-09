"""Main contradiction detection class and core detection loop."""
import os
import logging
import time
import json
import re
import threading
import random
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
import numpy as np

from eva_ai.nlp_fallbacks import (
    compute_semantic_similarity,
    get_sentiment_analyzer,
    polarity_scores,
    tokenize,
    get_stopwords,
)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
except ImportError:
    get_sentence_transformer = None

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

from .detect_semantic import SemanticDetectionMixin
from .detect_logical import LogicalDetectionMixin
from .detect_temporal import TemporalDetectionMixin

logger = logging.getLogger("eva_ai.contradiction.detection")


class ContradictionDetector(SemanticDetectionMixin, LogicalDetectionMixin, TemporalDetectionMixin):
    """
    CD: Класс для обнаружения противоречий в знаниях системы.
    
    Features:
    - Семантическая детекция через embeddings
    - Логическая детекция (negation, antonymy)
    - Временная детекция (история изменений)
    - CD-1: Улучшенный алгоритм с контекстным анализом
    - CD-2: Интеграция с GNN для анализа контекста
    """
    
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
        self.source_reputation_system = source_reputation_system
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.stop_words = get_stopwords(("english", "russian"))
        self.detected_contradictions = []
        self.last_detection_time = 0
        self.detection_history = []
        
        # CD-2: GNN integration
        self.gnn_encoder = None
        self.gnn_context_window = 5
        
        # CD-1: Enhanced detection metrics
        self.context_cache: Dict[str, List] = {}
        self.confidence_calibration: Dict[str, float] = {}
        
        logger.info("Детектор противоречий инициализирован")
    
    def _init_nlp_model(self):
        """Инициализирует NLP-модель для анализа противоречий из локального синглтона."""
        logger.info("Инициализация NLP-модели для обнаружения противоречий...")
        
        if get_sentence_transformer is not None:
            model = get_sentence_transformer(device='auto')
            if model is not None:
                logger.info("NLP-модель загружена через локальный singleton")
                return model
        
        logger.error("Не удалось загрузить NLP-модель через локальный синглтон!")
        return None
    
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
        if not force and (time.time() - self.last_detection_time < 60):
            logger.info("Обнаружение противоречий пропущено - недавно выполнено")
            return self.detected_contradictions
        
        logger.info(f"Начало обнаружения противоречий для {'всех концептов' if concept is None else concept}")
        self.detected_contradictions = []
        
        try:
            if concept:
                contradictions = self._detect_contradictions_for_concept(concept)
                self.detected_contradictions.extend(contradictions)
            else:
                concepts = self.knowledge_graph.get_all_concepts()
                for i, c in enumerate(concepts):
                    contradictions = self._detect_contradictions_for_concept(c)
                    self.detected_contradictions.extend(contradictions)
                    if (i + 1) % 100 == 0:
                        logger.info(f"Проанализировано {i + 1}/{len(concepts)} концептов")
            
            self.last_detection_time = time.time()
            self.detection_history.append({
                "timestamp": self.last_detection_time,
                "concept": concept,
                "contradictions_found": len(self.detected_contradictions),
                "duration": time.time() - start_time
            })
            if len(self.detection_history) > 100:
                self.detection_history = self.detection_history[-100:]
            
            logger.info(f"Обнаружено {len(self.detected_contradictions)} противоречий")
            return self.detected_contradictions
        except Exception as e:
            logger.error(f"Ошибка обнаружения противоречий: {e}")
            return []
    
    def _detect_contradictions_for_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        CD-1: Обнаруживает противоречия для конкретного концепта с контекстным анализом.
        """
        facts = self.knowledge_graph.get_facts_by_concept(concept)
        if len(facts) < 2:
            return []
        
        facts_by_relation = defaultdict(list)
        for fact in facts:
            relation_type = fact.get("relation_type", "related_to")
            facts_by_relation[relation_type].append(fact)
        
        contradictions = []
        for relation_type, fact_group in facts_by_relation.items():
            if len(fact_group) < 2:
                continue
            
            potential_contradictions = self._find_potential_contradictions(fact_group)
            
            for i, j in potential_contradictions:
                fact1 = fact_group[i]
                fact2 = fact_group[j]
                
                divergence = self._calculate_divergence(fact1, fact2)
                
                context_score = self._get_context_boost(fact1, fact2)
                adjusted_divergence = min(1.0, divergence * (1.0 + context_score))
                
                if adjusted_divergence > self.detection_threshold:
                    contradiction = self._create_contradiction(
                        concept, [fact1, fact2], adjusted_divergence, relation_type=relation_type
                    )
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _get_context_boost(self, fact1: Dict, fact2: Dict) -> float:
        """
        CD-1: Вычисляет контекстный буст для расхождения.
        
        Учитывает:
        - Временную близость фактов
        - Источники (репутация)
        - Эмоциональную окраску
        """
        boost = 0.0
        
        t1 = fact1.get("timestamp", 0)
        t2 = fact2.get("timestamp", 0)
        if t1 > 0 and t2 > 0:
            time_diff = abs(t1 - t2) / 86400
            if time_diff < 1:
                boost += 0.1
        
        if self.source_reputation_system:
            rep1 = self.source_reputation_system.get_reputation(fact1.get("source", ""))
            rep2 = self.source_reputation_system.get_reputation(fact2.get("source", ""))
            if abs(rep1 - rep2) > 0.3:
                boost += 0.15
        
        sentiment1 = fact1.get("sentiment", 0)
        sentiment2 = fact2.get("sentiment", 0)
        if sentiment1 * sentiment2 < 0:
            boost += 0.1
        
        return boost
    
    def _find_potential_contradictions(self, facts: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Находит потенциальные пары противоречивых фактов."""
        potential_pairs = []
        
        numeric_facts = []
        for i, fact in enumerate(facts):
            if isinstance(fact.get("value"), (int, float)):
                numeric_facts.append((i, fact["value"]))
        
        if numeric_facts:
            values = [v for _, v in numeric_facts]
            mean = np.mean(values)
            std = np.std(values)
            for i, (idx, value) in enumerate(numeric_facts):
                if std > 0 and abs(value - mean) > 2 * std:
                    for j, (other_idx, _) in enumerate(numeric_facts):
                        if i != j:
                            potential_pairs.append((idx, other_idx))
        
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
        
        text_facts = []
        for i, fact in enumerate(facts):
            if isinstance(fact.get("value"), str) and "response" in fact.get("relation_type", ""):
                text_facts.append((i, fact["value"]))
        
        if len(text_facts) > 1:
            texts = [text for _, text in text_facts]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    try:
                        sim = float(compute_semantic_similarity([texts[i], texts[j]], self.nlp_model))
                    except Exception:
                        sim = 0.5
                    if 1.0 - sim > self.detection_threshold:
                        potential_pairs.append((text_facts[i][0], text_facts[j][0]))
        
        unique_pairs = list(set(potential_pairs))
        if len(unique_pairs) > self.max_conflicting_facts:
            unique_pairs = random.sample(unique_pairs, self.max_conflicting_facts)
        
        return unique_pairs
    
    def _calculate_divergence(self, fact1: Dict[str, Any], 
                            fact2: Dict[str, Any]) -> float:
        """Вычисляет уровень расхождения между двумя фактами."""
        if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
            v1, v2 = fact1["value"], fact2["value"]
            if v1 == 0 and v2 == 0:
                return 0.0
            elif v1 == 0 or v2 == 0:
                return 1.0
            else:
                return abs(v1 - v2) / max(abs(v1), abs(v2))
        
        if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
            return 1.0 if fact1["value"] != fact2["value"] else 0.0
        
        if isinstance(fact1.get("value"), str) and isinstance(fact2.get("value"), str):
            return self._calculate_text_divergence(fact1["value"], fact2["value"])
        
        return self._calculate_lexical_divergence(str(fact1.get("value", "")), 
                                               str(fact2.get("value", "")))
    
    def _create_contradiction(self, concept: str, conflicting_facts: List[Dict[str, Any]], 
                            divergence_level: float, **kwargs) -> Dict[str, Any]:
        """Создает объект противоречия."""
        timestamp = int(time.time() * 1000)
        hash_str = f"{concept}{divergence_level}{timestamp}"
        contradiction_id = f"contradiction_{hashlib.md5(hash_str.encode()).hexdigest()[:8]}"
        
        metadata = {
            "relation_type": kwargs.get("relation_type", "related_to"),
            "detection_method": "automatic",
            "timestamp": timestamp
        }
        
        if self.source_reputation_system:
            for i, fact in enumerate(conflicting_facts):
                source = fact.get("source")
                if source:
                    reputation = self.source_reputation_system.get_source_reputation(source)
                    metadata[f"source_reputation_{i}"] = reputation
        
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
        """Обнаруживает противоречия для нового факта."""
        concept = new_fact.get("concept")
        if not concept:
            return []
        
        existing_facts = self.knowledge_graph.get_facts_by_concept(concept)
        contradictions = []
        
        for fact in existing_facts:
            if self._are_facts_equivalent(new_fact, fact):
                continue
            divergence = self._calculate_divergence(new_fact, fact)
            if divergence > self.detection_threshold:
                contradiction = self._create_contradiction(
                    concept, [new_fact, fact], divergence,
                    relation_type=new_fact.get("relation_type", "related_to")
                )
                contradictions.append(contradiction)
        
        return contradictions
    
    def _are_facts_equivalent(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> bool:
        """Проверяет, являются ли два факта эквивалентными."""
        if fact1.get("concept") != fact2.get("concept"):
            return False
        if fact1.get("relation_type") != fact2.get("relation_type"):
            return False
        if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
            return abs(fact1["value"] - fact2["value"]) < 0.01 * max(abs(fact1["value"]), abs(fact2["value"]), 1)
        return str(fact1.get("value", "")) == str(fact2.get("value", ""))
    
    def analyze_fact_consistency(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Анализирует согласованность факта с существующими знаниями."""
        concept = fact.get("concept")
        if not concept:
            return {
                "consistent": False, "reason": "Concept is missing",
                "divergence": 1.0, "conflicting_facts": []
            }
        
        existing_facts = self.knowledge_graph.get_facts_by_concept(concept)
        if not existing_facts:
            return {
                "consistent": True, "reason": "No existing facts to compare with",
                "divergence": 0.0, "conflicting_facts": []
            }
        
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
        """Возвращает статистику обнаружения противоречий."""
        total_contradictions = len(self.detected_contradictions)
        contradictions_by_concept = defaultdict(int)
        for contradiction in self.detected_contradictions:
            contradictions_by_concept[contradiction["concept"]] += 1
        
        top_concepts = sorted(
            contradictions_by_concept.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        contradiction_types = defaultdict(int)
        for contradiction in self.detected_contradictions:
            if len(contradiction["conflicting_facts"]) == 2:
                fact1 = contradiction["conflicting_facts"][0]
                fact2 = contradiction["conflicting_facts"][1]
                if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                    contradiction_types["numeric"] += 1
                elif isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                    contradiction_types["boolean"] += 1
                elif "response" in fact1.get("relation_type", "") and "response" in fact2.get("relation_type", ""):
                    contradiction_types["response"] += 1
            
            if "relation_type" in contradiction["metadata"]:
                relation_type = contradiction["metadata"]["relation_type"]
                if relation_type.startswith("only_") or relation_type.startswith("not_only_"):
                    contradiction_types["exclusivity"] += 1
                if relation_type in ["is_a", "part_of", "member_of"]:
                    contradiction_types["hierarchy"] += 1
        
        severity_counts = {"low": 0, "medium": 0, "high": 0}
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
            "detection_history": self.detection_history[-5:]
        }
    
    def start_background_detection(self, interval: int = 3600):
        """Запускает фоновое обнаружение противоречий."""
        if not self.active:
            return
        
        def detection_loop():
            while self.active:
                try:
                    self.detect_contradictions()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Ошибка в фоновом обнаружении противоречий: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=detection_loop, daemon=True)
        thread.start()
        logger.info(f"Фоновое обнаружение противоречий запущено (интервал: {interval} секунд)")
    
    def stop_background_detection(self):
        """Останавливает фоновое обнаружение противоречий."""
        self.active = False
        logger.info("Фоновое обнаружение противоречий остановлено")
    
    def get_recent_contradictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает недавно обнаруженные противоречия."""
        return self.detected_contradictions[-limit:]
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает противоречие по ID."""
        for contradiction in self.detected_contradictions:
            if contradiction["contradiction_id"] == contradiction_id:
                return contradiction
        return None
    
    def mark_contradiction_resolved(self, contradiction_id: str, resolution: Dict[str, Any]):
        """Отмечает противоречие как разрешенное."""
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
        """Возвращает высокоприоритетные противоречия."""
        def get_priority(contradiction):
            severity_priority = 0.3 if contradiction["divergence_level"] < 0.4 else 0.6 if contradiction["divergence_level"] < 0.7 else 0.9
            age = time.time() - contradiction["timestamp"]
            age_priority = min(0.4, age / 86400 * 0.1)
            return severity_priority * 0.7 + age_priority * 0.3
        
        unresolved = [c for c in self.detected_contradictions if c["status"] != "resolved"]
        sorted_contradictions = sorted(unresolved, key=get_priority, reverse=True)
        return sorted_contradictions[:limit]
    
    def generate_contradiction_report(self) -> str:
        """Генерирует отчет об обнаруженных противоречиях."""
        stats = self.get_detection_statistics()
        report = "ОТЧЕТ ОБ ОБНАРУЖЕННЫХ ПРОТИВОРЕЧИЯХ\n"
        report += "=" * 50 + "\n\n"
        report += f"Всего обнаружено противоречий: {stats['total_contradictions']}\n"
        report += f"Последнее обнаружение: {datetime.fromtimestamp(stats['last_detection_time']).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "ТОП-10 КОНЦЕПТОВ С ПРОТИВОРЕЧИЯМИ:\n"
        for i, item in enumerate(stats["top_concepts"], 1):
            report += f"{i}. {item['concept']} - {item['count']} противоречий\n"
        report += "\n"
        report += "РАСПРЕДЕЛЕНИЕ ПО ТИПАМ:\n"
        for ctype, count in stats["contradiction_types"].items():
            report += f"- {ctype}: {count}\n"
        report += "\n"
        report += "РАСПРЕДЕЛЕНИЕ ПО СЕРЬЕЗНОСТИ:\n"
        report += f"- Высокая: {stats['severity_distribution']['high']}\n"
        report += f"- Средняя: {stats['severity_distribution']['medium']}\n"
        report += f"- Низкая: {stats['severity_distribution']['low']}\n"
        report += "\n"
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
        """Интегрирует решение противоречия в граф знаний."""
        contradiction = self.get_contradiction_by_id(contradiction_id)
        if not contradiction:
            logger.warning(f"Противоречие {contradiction_id} не найдено")
            return
        
        concept = contradiction["concept"]
        for fact in removed_facts:
            self.knowledge_graph.remove_fact(fact)
        
        if resolved_fact not in self.knowledge_graph.get_facts_by_concept(concept):
            self.knowledge_graph.add_fact(resolved_fact)
        
        resolution = {
            "resolved_fact": resolved_fact,
            "removed_facts": removed_facts,
            "timestamp": time.time(),
            "confidence": 0.9
        }
        self.mark_contradiction_resolved(contradiction_id, resolution)
        logger.info(f"Противоречие {contradiction_id} интегрировано в граф знаний")
    
    def detect_all_specialized_contradictions(self) -> List[Dict[str, Any]]:
        """Обнаруживает все специализированные типы противоречий."""
        all_contradictions = []
        hierarchy_contradictions = self.detect_hierarchy_contradictions()
        all_contradictions.extend(hierarchy_contradictions)
        exclusivity_contradictions = self.detect_exclusivity_contradictions()
        all_contradictions.extend(exclusivity_contradictions)
        temporal_contradictions = self.detect_temporal_contradictions()
        all_contradictions.extend(temporal_contradictions)
        contextual_contradictions = self.detect_contextual_contradictions()
        all_contradictions.extend(contextual_contradictions)
        self.detected_contradictions.extend(all_contradictions)
        return all_contradictions
    
    # === CD-2: GNN Integration ===
    
    def set_gnn_encoder(self, gnn_encoder):
        """
        CD-2: Установить GNN encoder для анализа контекста.
        
        Args:
            gnn_encoder: Объект GNN encoder с методом encode(nodes, edges)
        """
        self.gnn_encoder = gnn_encoder
        logger.info("GNN encoder установлен для ContradictionDetector")
    
    def detect_with_gnn_context(self, concept: str) -> List[Dict]:
        """
        CD-2: Детекция противоречий с использованием GNN контекста.
        
        Использует GNN для:
        - Анализа связей между фактами
        - Выявления паттернов противоречий в графе
        - Усиления обнаружения через topological features
        """
        if self.gnn_encoder is None:
            return self._detect_contradictions_for_concept(concept)
        
        facts = self.knowledge_graph.get_facts_by_concept(concept)
        if len(facts) < 2:
            return []
        
        try:
            nodes = [f.get("id", f.get("content", f"")) for f in facts]
            edges = self._build_fact_graph(facts)
            
            graph_vectors = self.gnn_encoder.encode(nodes, edges)
            
            gnn_enhanced_contradictions = []
            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    divergence = self._calculate_divergence(facts[i], facts[j])
                    
                    if i < len(graph_vectors) and j < len(graph_vectors):
                        gnn_sim = float(np.dot(graph_vectors[i], graph_vectors[j]) / (
                            np.linalg.norm(graph_vectors[i]) * np.linalg.norm(graph_vectors[j]) + 1e-8
                        ))
                        gnn_boost = 1.0 - gnn_sim
                        divergence = min(1.0, divergence + gnn_boost * 0.3)
                    
                    if divergence > self.detection_threshold:
                        contradiction = self._create_contradiction(
                            concept, [facts[i], facts[j]], divergence,
                            relation_type=facts[i].get("relation_type", "related_to"),
                            metadata={"gnn_enhanced": True}
                        )
                        gnn_enhanced_contradictions.append(contradiction)
            
            return gnn_enhanced_contradictions
            
        except Exception as e:
            logger.error(f"GNN context detection error: {e}")
            return self._detect_contradictions_for_concept(concept)
    
    def _build_fact_graph(self, facts: List[Dict]) -> List[Tuple[str, str]]:
        """CD-2: Построить граф связей между фактами."""
        edges = []
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts[i+1:], i+1):
                shared_keys = set(fact1.keys()) & set(fact2.keys())
                common_attrs = sum(1 for k in shared_keys if k not in ["id", "content", "value", "timestamp"])
                
                if common_attrs > 1:
                    node1 = fact1.get("id", f"fact_{i}")
                    node2 = fact2.get("id", f"fact_{j}")
                    edges.append((node1, node2))
        
        return edges
    
    def get_gnn_contradiction_analysis(self, concept: str) -> Dict:
        """
        CD-2: Получить анализ противоречий с GNN метриками.
        
        Returns:
            {
                "contradiction_count": int,
                "gnn_coherence_score": float,
                "conflict_regions": List[Dict],
                "recommended_resolution": str
            }
        """
        contradictions = self.detect_with_gnn_context(concept)
        
        if not contradictions:
            return {
                "contradiction_count": 0,
                "gnn_coherence_score": 1.0,
                "conflict_regions": [],
                "recommended_resolution": "no_conflict"
            }
        
        total_divergence = sum(c.get("divergence_level", 0) for c in contradictions)
        avg_divergence = total_divergence / len(contradictions)
        coherence_score = 1.0 - avg_divergence
        
        conflict_regions = []
        for c in contradictions:
            conflict_regions.append({
                "facts": [f.get("content", "") for f in c.get("conflicting_facts", [])],
                "divergence": c.get("divergence_level", 0),
                "severity": "high" if c.get("divergence_level", 0) > 0.7 else "medium"
            })
        
        recommended = "deep_analysis" if coherence_score < 0.5 else "minor_review"
        
        return {
            "contradiction_count": len(contradictions),
            "gnn_coherence_score": coherence_score,
            "conflict_regions": conflict_regions,
            "recommended_resolution": recommended
        }
