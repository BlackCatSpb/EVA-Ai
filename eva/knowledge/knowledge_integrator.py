"""Модуль интеграции знаний для ЕВА - улучшение согласованности знаний"""
import os
import logging
import time
import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from .knowledge_analyzer import KnowledgeAnalyzer

logger = logging.getLogger("eva.knowledge_integrator")

class KnowledgeIntegrator:
    """Модуль интеграции знаний для ЕВА - улучшение согласованности знаний."""
    
    def __init__(self, brain=None, knowledge_graph=None, knowledge_analyzer=None):
        """
        Инициализирует интегратор знаний.
        
        Args:
            brain: Ссылка на ядро ЕВА (опционально)
            knowledge_graph: Ссылка на граф знаний (опционально)
            knowledge_analyzer: Ссылка на анализатор знаний (опционально)
        """
        self.brain = brain
        self.knowledge_graph = knowledge_graph or KnowledgeGraph(brain=brain)
        self.knowledge_analyzer = knowledge_analyzer or KnowledgeAnalyzer(self.knowledge_graph, brain)
        self.knowledge_expander = None
        
        # Надежность источников
        self.source_reliability = defaultdict(float)
        
        # Инициализируем надежность источников по умолчанию
        self._init_source_reliability()
        
        # Загружаем историю для динамического обновления
        self._load_history_for_dynamic_updates()
        
        logger.info("KnowledgeIntegrator инициализирован")
    
    def _init_source_reliability(self):
        """Инициализирует базовые значения надежности источников."""
        # Официальные источники с высокой надежностью
        official_sources = [
            "wikipedia.org", "nih.gov", "nasa.gov", "who.int", "un.org",
            "science.gov", "nature.com", "springer.com", "ieee.org"
        ]
        for source in official_sources:
            self.source_reliability[source] = 0.95
            
        # Академические источники
        academic_sources = [
            "edu", "ac.uk", "edu.au", "scholar.google", "jstor.org"
        ]
        for source in academic_sources:
            self.source_reliability[source] = 0.85
            
        # Новостные источники
        news_sources = [
            "bbc.com", "reuters.com", "apnews.com", "nytimes.com", "theguardian.com"
        ]
        for source in news_sources:
            self.source_reliability[source] = 0.75
            
        # Социальные сети и блоги
        social_sources = [
            "blogspot.com", "wordpress.com", "medium.com", "twitter.com", "facebook.com"
        ]
        for source in social_sources:
            self.source_reliability[source] = 0.45
    
    def get_source_reliability(self, source: str) -> float:
        """
        Возвращает надежность источника.
        
        Args:
            source: Источник информации
            
        Returns:
            float: Надежность источника (0.0-1.0)
        """
        if not source:
            return 0.5  # Средняя надежность для неизвестных источников
        
        # Проверяем точное совпадение
        if source in self.source_reliability:
            return self.source_reliability[source]
        
        # Проверяем домены верхнего уровня
        for domain, reliability in self.source_reliability.items():
            if domain in source:
                return reliability
        
        # Проверяем TLD (домены верхнего уровня)
        tld = source.split('.')[-1]
        if tld in ['gov', 'edu', 'ac.uk']:
            return 0.8
        elif tld in ['org', 'net']:
            return 0.65
        else:
            return 0.5
    
    def update_source_reliability(self, source: str, reliability_change: float):
        """
        Обновляет надежность источника на основе новых данных.
        
        Args:
            source: Источник информации
            reliability_change: Изменение надежности (-1.0 to 1.0)
        """
        current_reliability = self.get_source_reliability(source)
        new_reliability = current_reliability + reliability_change * 0.1  # Меньше влияние на надежность
        new_reliability = max(0.1, min(0.99, new_reliability))  # Ограничиваем диапазон
        self.source_reliability[source] = new_reliability
        
        logger.debug(f"Надежность источника '{source}' обновлена: {current_reliability:.2f} -> {new_reliability:.2f}")
    
    def _load_history_for_dynamic_updates(self):
        """Загружает историю для динамического обновления надежности источников и доменов."""
        try:
            # Некоторые реализации графа знаний могут не иметь истории изменений
            history = []
            kg = getattr(self, "knowledge_graph", None)
            if kg is not None and hasattr(kg, "get_recent_changes"):
                history = kg.get_recent_changes(limit=100) or []
            
            # Анализируем историю для обновления надежности
            for event in history:
                if isinstance(event, dict) and event.get("action") in ["node_update", "node_creation"]:
                    # Обновляем надежность источника на основе успешных обновлений
                    if event.get("user_id"):
                        self.update_source_reliability(f"user_{event['user_id']}", 0.1)
            
            logger.debug("История загружена для динамического обновления надежности")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки истории для динамических обновлений: {e}")
    
    def integrate_knowledge(self, concept: str, depth: int = 1) -> bool:
        """
        Интегрирует знания, заполняя пробелы и разрешая противоречия.
        
        Args:
            concept: Концепт для интеграции
            depth: Глубина интеграции
            
        Returns:
            bool: Успешно ли выполнено
        """
        try:
            logger.info(f"Интеграция знаний по концепту '{concept}' (глубина: {depth})")
            
            # Анализируем пробелы
            gaps = self.knowledge_analyzer.analyze_knowledge_gaps(domain=concept, num_samples=5)
            
            if not gaps:
                gaps = []
            
            # Заполняем пробелы
            filled_gaps = 0
            for gap in gaps:
                if not isinstance(gap, dict):
                    continue
                if gap.get("gap_type") == "incomplete":
                    gap_concept = gap.get("concept")
                    if not gap_concept:
                        continue
                    # Пытаемся найти дополнительные связи
                    if self.brain and hasattr(self.brain, 'text_processor'):
                        related = self.brain.text_processor.analyze_connection_pattern(
                            gap_concept, 
                            [gap_concept], 
                            "related_to"
                        )
                        
                        # Добавляем новые связи
                        if related and related.get("most_related"):
                            for concept_name in related["most_related"]:
                                # Определяем тип связи
                                relation_type = "related_to"  # Простое отношение
                                # Находим узлы по именам, чтобы получить их ID
                                src_nodes = self.knowledge_graph.search_nodes(gap["concept"], limit=1)
                                dst_nodes = self.knowledge_graph.search_nodes(concept_name, limit=1)
                                if src_nodes and dst_nodes and len(src_nodes) > 0 and len(dst_nodes) > 0:
                                    self.knowledge_graph.add_edge(
                                        source_id=src_nodes[0].id,
                                        target_id=dst_nodes[0].id,
                                        relation_type=relation_type,
                                        strength=related.get("connection_strength", 0.5)
                                    )
                            filled_gaps += 1
                
                elif gap.get("gap_type") == "outdated":
                    gap_concept = gap.get("concept")
                    if not gap_concept:
                        continue
                    # Пытаемся обновить информацию
                    if self.brain and hasattr(self.brain, 'web_search_engine'):
                        knowledge = self.brain.web_search_engine.web_search_and_learn(
                            gap_concept, 
                            num_results=1
                        )
                        
                        if knowledge and isinstance(knowledge, list) and len(knowledge) > 0:
                            # Обновляем информацию в KnowledgeGraph
                            nodes = self.knowledge_graph.search_nodes(gap_concept, limit=1)
                            if nodes and len(nodes) > 0:
                                knowledge_item = knowledge[0]
                                if isinstance(knowledge_item, dict):
                                    content = knowledge_item.get("content", "")
                                    source = knowledge_item.get("source")
                                    if content:
                                        self.knowledge_graph.update_node(
                                            nodes[0].id,
                                            content,
                                            source=source
                                        )
                            filled_gaps += 1
            
            # Анализируем противоречия
            contradictions = self.knowledge_analyzer.detect_contradictions()
            
            if not contradictions:
                contradictions = []
            
            # Разрешаем противоречия
            resolved_contradictions = 0
            for contradiction in contradictions:
                if self._resolve_contradiction(contradiction):
                    resolved_contradictions += 1
            
            logger.info(f"Интеграция знаний завершена. Заполнено {filled_gaps} пробелов, разрешено {resolved_contradictions} противоречий")
            return filled_gaps > 0 or resolved_contradictions > 0
            
        except Exception as e:
            logger.error(f"Ошибка интеграции знаний по концепту '{concept}': {e}")
            
            # Уведомляем SelfAnalyzer об ошибке
            if self.brain and hasattr(self.brain, 'self_analyzer'):
                self.brain.self_analyzer.add_learning_opportunity(
                    concept=f"knowledge_integration_{concept}",
                    opportunity_type="integration",
                    priority=0.9,
                    domain="system",
                    evidence=[f"Ошибка интеграции знаний: {str(e)}"],
                    suggested_actions=[
                        "Проверить целостность графа знаний",
                        "Анализировать выявленные противоречия"
                    ]
                )
            
            return False
    
    def _resolve_contradiction(self, contradiction: Dict[str, Any]) -> bool:
        """
        Полноценно разрешает противоречие в знаниях с использованием 
        нескольких методов оценки и улучшения согласованности.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли разрешено противоречие
        """
        try:
            logger.info(f"Разрешение противоречия: {contradiction}")
            
            # Определяем тип противоречия
            contradiction_type = self._determine_contradiction_type(contradiction)
            
            # В зависимости от типа применяем соответствующую стратегию
            if contradiction_type == "opposite_relations":
                return self._resolve_opposite_relations(contradiction)
            elif contradiction_type == "conflicting_definitions":
                return self._resolve_conflicting_definitions(contradiction)
            elif contradiction_type == "cyclic_dependency":
                return self._resolve_cyclic_dependency(contradiction)
            elif contradiction_type == "domain_conflict":
                return self._resolve_domain_conflict(contradiction)
            else:
                # Общий метод разрешения для неизвестных типов
                return self._resolve_general_contradiction(contradiction)
                
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречия: {e}")
            return False
    
    def _determine_contradiction_type(self, contradiction: Dict[str, Any]) -> str:
        """
        Определяет тип противоречия на основе его характеристик.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            str: Тип противоречия
        """
        # Проверяем наличие ключевых признаков для каждого типа
        
        # Противоположные отношения (например, A противоположно B и B противоположно A)
        if "concept1" in contradiction and "concept2" in contradiction and "common_target" in contradiction:
            return "opposite_relations"
        
        # Конфликтующие определения (один концепт с разными определениями)
        if "concept" in contradiction and "domains" in contradiction and len(contradiction["domains"]) > 1:
            return "conflicting_definitions"
        
        # Циклические зависимости (A -> B -> C -> A)
        if "cycle" in contradiction:
            return "cyclic_dependency"
        
        # Конфликт доменов (разные домены дают противоречивую информацию)
        if "domains" in contradiction and "evidence" in contradiction and len(contradiction["domains"]) > 1:
            return "domain_conflict"
        
        # Общий тип противоречия
        return "general"
    
    def _resolve_opposite_relations(self, contradiction: Dict[str, Any]) -> bool:
        """
        Разрешает противоречия, вызванные противоположными отношениями.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли разрешено
        """
        try:
            concept1 = contradiction.get("concept1")
            concept2 = contradiction.get("concept2")
            common_target = contradiction.get("common_target")
            
            if not all([concept1, concept2, common_target]):
                return False

            # Получаем узлы
            node1_list = self.knowledge_graph.search_nodes(concept1, limit=1)
            node2_list = self.knowledge_graph.search_nodes(concept2, limit=1)
            target_list = self.knowledge_graph.search_nodes(common_target, limit=1)

            if not node1_list or not node2_list or not target_list:
                return False

            node1 = node1_list[0]
            node2 = node2_list[0]
            target_node = target_list[0]

            # Анализируем надежность источников для каждого утверждения
            edge1 = self.knowledge_graph.get_edges(node1.id)
            edge2 = self.knowledge_graph.get_edges(node2.id)
            
            # Оцениваем силу каждого утверждения
            strength1 = self._evaluate_statement_strength(node1.id, target_node.id, "opposite_of")
            strength2 = self._evaluate_statement_strength(node2.id, target_node.id, "opposite_of")
            
            # Если силы близки, возможно, это не противоречие, а разные аспекты
            if abs(strength1 - strength2) < 0.2:
                # Создаем гипотезу для объединения
                return self._create_hypothesis_for_opposite_relations(contradiction)
            
            # Усиливаем более сильное утверждение
            if strength1 > strength2:
                # Уменьшаем силу второго утверждения
                self._weaken_edge(node2.id, target_node.id, "opposite_of", 0.2)
                # Увеличиваем надежность источника первого утверждения
                self._update_source_reliability_from_node(node1)
            else:
                # Уменьшаем силу первого утверждения
                self._weaken_edge(node1.id, target_node.id, "opposite_of", 0.2)
                # Увеличиваем надежность источника второго утверждения
                self._update_source_reliability_from_node(node2)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка разрешения противоположных отношений: {e}")
            return False
    
    def _evaluate_statement_strength(self, source_id: str, target_id: str, relation: str) -> float:
        """
        Оценивает силу утверждения на основе нескольких факторов.
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation: Тип отношения
            
        Returns:
            float: Сила утверждения (0.0-1.0)
        """
        # Получаем связь
        edge = None
        edges = self.knowledge_graph.get_edges(source_id)
        for e in edges:
            if e.target_id == target_id and e.relation_type == relation:
                edge = e
                break
        
        if not edge:
            return 0.0
        
        # Начальная сила - это сила связи
        strength = edge.strength
        
        # Учитываем надежность источника
        node = self.knowledge_graph.get_node(source_id)
        if node:
            src_meta = getattr(node, "meta", {}) or {}
            primary_source = None
            if isinstance(src_meta, dict):
                sources_list = src_meta.get("sources")
                if isinstance(sources_list, list) and sources_list:
                    primary_source = sources_list[0]
                else:
                    primary_source = src_meta.get("source")
            if primary_source:
                source_reliability = self.get_source_reliability(primary_source)
            else:
                source_reliability = 1.0
            strength *= source_reliability
        
        # Учитываем актуальность информации
        current_time = time.time()
        node_age = getattr(node, 'last_updated', current_time)
        age_days = (current_time - node_age) / 86400
        freshness_factor = max(0.2, 1.0 - (age_days / 365))  # Уменьшается линейно в течение года
        strength *= freshness_factor
        
        # Учитываем авторитетность домена
        domain_authority = 0.8  # По умолчанию
        strength *= domain_authority
        
        # Учитываем количество подтверждений
        confirmation_count = self._count_confirmations(source_id, target_id, relation)
        confirmation_factor = min(1.0, 0.5 + confirmation_count * 0.1)
        strength *= confirmation_factor
        
        return min(1.0, max(0.0, strength))
    
    def _count_confirmations(self, source_id: str, target_id: str, relation: str) -> int:
        """
        Подсчитывает количество подтверждений утверждения.
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation: Тип отношения
            
        Returns:
            int: Количество подтверждений
        """
        # Получаем исходный узел
        source_node = self.knowledge_graph.get_node(source_id)
        if not source_node:
            return 0
        
        # Создаем векторное представление исходного узла
        source_vector = self._get_node_vector(source_node)
        if source_vector is None:
            # Если нет векторного представления, используем текстовый анализ
            return self._count_text_confirmations(source_node, target_id, relation)
        
        # Получаем все узлы, которые могут быть похожи
        similar_nodes = self._find_similar_nodes_by_vector(source_node, source_vector)
        
        # Подсчитываем подтверждения
        confirmations = 0
        for node, similarity in similar_nodes:
            # Проверяем, есть ли связь с целевым узлом и тем же отношением
            edges = self.knowledge_graph.get_edges(node.id)
            for edge in edges:
                if edge.target_id == target_id and edge.relation_type == relation:
                    # Учитываем силу подтверждения (чем выше сходство, тем сильнее подтверждение)
                    confirmations += min(1, int(similarity * 10))
                    break
        
        return confirmations
    
    def _get_node_vector(self, node: KnowledgeNode) -> Optional[np.ndarray]:
        """
        Получает векторное представление узла.
        
        Args:
            node: Узел знаний
            
        Returns:
            Optional[np.ndarray]: Векторное представление или None
        """
        # Если доступен ML, используем его для получения вектора
        if self.brain and hasattr(self.brain, 'text_processor'):
            try:
                return self.brain.text_processor.get_text_embedding(node.meta.get("description", node.content))
            except Exception as e:
                logger.warning(f"Ошибка получения векторного представления: {e}")
        
        return None
    
    def _find_similar_nodes_by_vector(self, node: KnowledgeNode, node_vector: np.ndarray, 
                                    threshold: float = 0.6, limit: int = 10) -> List[Tuple[KnowledgeNode, float]]:
        """
        Находит похожие узлы на основе векторного представления.
        
        Args:
            node: Базовый узел
            node_vector: Векторное представление базового узла
            threshold: Порог сходства
            limit: Максимальное количество результатов
            
        Returns:
            List[Tuple[KnowledgeNode, float]]: Список похожих узлов и их сходства
        """
        # Получаем все узлы
        all_nodes = self.knowledge_graph.get_all_nodes()
        all_nodes = all_nodes[:1000]
        
        # Вычисляем сходство
        similarities = []
        for other_node in all_nodes:
            if other_node.id == node.id:
                continue
            
            # Получаем вектор другого узла
            other_vector = self._get_node_vector(other_node)
            if other_vector is None:
                continue
            
            # Вычисляем косинусное сходство
            similarity = cosine_similarity([node_vector], [other_vector])[0][0]
            
            if similarity >= threshold:
                similarities.append((other_node, similarity))
        
        # Сортируем по сходству
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-N
        return similarities[:limit]
    
    def _count_text_confirmations(self, node: KnowledgeNode, target_id: str, relation: str) -> int:
        """
        Подсчитывает количество текстовых подтверждений утверждения.
        
        Args:
            node: Узел
            target_id: ID целевого узла
            relation: Тип отношения
            
        Returns:
            int: Количество подтверждений
        """
        # Получаем описание узла
        description = getattr(node, "description", "") or ""
        if not description:
            return 0
        
        # Получаем целевой узел
        target_node = self.knowledge_graph.get_node(target_id)
        if not target_node:
            return 0
        
        # Анализируем ключевые слова
        relation_keywords = {
            "cause": ["причина", "вызывает", "приводит к", "является причиной"],
            "effect": ["следствие", "результат", "является результатом", "вызывает"],
            "part_of": ["часть", "составная часть", "входит в"],
            "is_a": ["является", "тип", "разновидность", "подкатегория"],
            "related_to": ["связан с", "относится к", "имеет отношение к"],
            "opposite_of": ["противоположность", "противоположный", "антоним"]
        }.get(relation, [])
        
        # Считаем совпадения в описании
        description_lower = description.lower()
        target_content_lower = (getattr(target_node, "description", "") or "").lower()
        
        confirmations = 0
        for keyword in relation_keywords:
            # Проверяем, содержит ли описание ключевое слово и упоминание целевого узла
            if keyword in description_lower and target_content_lower in description_lower:
                confirmations += 1
        
        return confirmations
    
    def _weaken_edge(self, source_id: str, target_id: str, relation: str, amount: float):
        """
        Ослабляет силу связи между узлами.
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation: Тип отношения
            amount: На сколько ослабить (0.0-1.0)
        """
        try:
            # Получаем текущую связь
            edges = self.knowledge_graph.get_edges(source_id)
            for edge in edges:
                if edge.target_id == target_id and edge.relation_type == relation:
                    # Уменьшаем силу
                    new_strength = max(0.1, edge.strength - amount)
                    
                    # Обновляем связь через KnowledgeGraph
                    try:
                        edge.strength = new_strength
                        edge.last_updated = time.time()
                        if hasattr(self.knowledge_graph, "_update_edge_in_db"):
                            self.knowledge_graph._update_edge_in_db(edge)
                        logger.debug(f"Связь {source_id}-{relation}->{target_id} ослаблена: {edge.strength:.2f} -> {new_strength:.2f}")
                    except Exception as ex:
                        logger.error(f"Ошибка обновления силы связи: {ex}")
                    return
                    
        except Exception as e:
            logger.error(f"Ошибка ослабления связи: {e}")
    
    def _update_source_reliability_from_node(self, node: KnowledgeNode):
        """
        Обновляет надежность источника на основе узла.
        
        Args:
            node: Узел знаний
        """
        if node:
            src_meta = getattr(node, "meta", {}) or {}
            primary_source = None
            if isinstance(src_meta, dict):
                sources_list = src_meta.get("sources")
                if isinstance(sources_list, list) and sources_list:
                    primary_source = sources_list[0]
                else:
                    primary_source = src_meta.get("source")
            if primary_source:
                self.update_source_reliability(primary_source, 0.2)  # Увеличиваем надежность
    
    def _create_hypothesis_for_opposite_relations(self, contradiction: Dict[str, Any]) -> bool:
        """
        Создает гипотезу для объяснения противоположных отношений.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли создана гипотеза
        """
        try:
            concept1 = contradiction["concept1"]
            concept2 = contradiction["concept2"]
            common_target = contradiction["common_target"]
            
            # Формируем гипотезу
            hypothesis = (
                f"Концепты '{concept1}' и '{concept2}' могут быть противоположными по отношению к '{common_target}', "
                "но в разных контекстах или с разных точек зрения. Возможно, это отражает сложность или многогранность "
                "концепта, а не истинное противоречие."
            )
            
            # Создаем узел для гипотезы
            hypothesis_id = self.knowledge_graph.add_node(
                name=f"Hypothesis_{concept1}_{concept2}",
                description=hypothesis,
                node_type="hypothesis",
                domain="metaknowledge",
                strength=0.6,
                meta={"sources": ["system"]}
            )
            
            # Связываем гипотезу с концептами
            node1 = self.knowledge_graph.search_nodes(concept1, limit=1)
            node2 = self.knowledge_graph.search_nodes(concept2, limit=1)
            target_node = self.knowledge_graph.search_nodes(common_target, limit=1)
            
            if node1:
                self.knowledge_graph.add_edge(
                    source_id=hypothesis_id,
                    target_id=node1[0].id,
                    relation_type="explains",
                    strength=0.7
                )
            if node2:
                self.knowledge_graph.add_edge(
                    source_id=hypothesis_id,
                    target_id=node2[0].id,
                    relation_type="explains",
                    strength=0.7
                )
            if target_node:
                self.knowledge_graph.add_edge(
                    source_id=hypothesis_id,
                    target_id=target_node[0].id,
                    relation_type="applies_to",
                    strength=0.7
                )
            
            logger.info(f"Создана гипотеза для объяснения противоречия между {concept1} и {concept2}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания гипотезы: {e}")
            return False
    
    def _resolve_conflicting_definitions(self, contradiction: Dict[str, Any]) -> bool:
        """
        Разрешает противоречия, вызванные конфликтующими определениями.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли разрешено
        """
        try:
            concept = contradiction["concept"]
            domains = contradiction["domains"]
            
            # Получаем узлы в разных доменах
            nodes_by_domain = {}
            for domain in domains:
                nodes = self.knowledge_graph.search_nodes(concept, domains=[domain], limit=1)
                if nodes:
                    nodes_by_domain[domain] = nodes[0]
            
            if not nodes_by_domain:
                return False
            
            # Оцениваем каждое определение
            domain_scores = {}
            for domain, node in nodes_by_domain.items():
                # Оцениваем по нескольким критериям
                score = 0.0
                
                # Надежность источника
                if getattr(node, 'source', None):
                    score += self.get_source_reliability(node.source) * 0.4
                
                # Актуальность
                age_days = (time.time() - node.last_updated) / 86400
                score += max(0.0, 1.0 - (age_days / 365)) * 0.3
                
                # Авторитетность домена
                if self.knowledge_expander and hasattr(self.knowledge_expander, 'get_domain_authority'):
                    score += self.knowledge_expander.get_domain_authority(domain) * 0.3
                
                domain_scores[domain] = score
            
            # Находим домен с наивысшей оценкой
            best_domain = max(domain_scores, key=domain_scores.get)
            best_node = nodes_by_domain[best_domain]
            
            # Создаем обобщенное определение, если различия небольшие
            if max(domain_scores.values()) - min(domain_scores.values()) < 0.3:
                return self._create_generalized_definition(contradiction, nodes_by_domain)
            
            # Усиливаем лучшее определение
            best_desc = best_node.meta.get("description", "") if isinstance(best_node.meta, dict) else ""
            self.knowledge_graph.update_node(
                best_node.id,
                best_desc,
                strength=min(1.0, best_node.strength + 0.2)
            )
            
            # Ослабляем менее авторитетные определения
            for domain, node in nodes_by_domain.items():
                if domain != best_domain:
                    node_desc = node.meta.get("description", "") if isinstance(node.meta, dict) else ""
                    self.knowledge_graph.update_node(
                        node.id,
                        node_desc,
                        strength=max(0.3, node.strength - 0.1)
                    )
            
            # Добавляем информацию о том, что определение зависит от домена
            context_node_id = self.knowledge_graph.add_concept(
                f"Context_{concept}",
                f"Концепт '{concept}' имеет разные определения в разных доменах. "
                f"Основное определение используется в домене '{best_domain}'.",
                domain="metaknowledge",
                strength=0.7
            )
            
            # Связываем контекст с концептом
            concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if concept_nodes:
                self.knowledge_graph.add_edge(
                    context_node_id,
                    concept_nodes[0].id,
                    "provides_context_for",
                    strength=0.8
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка разрешения конфликтующих определений: {e}")
            return False
    
    def _create_generalized_definition(self, contradiction: Dict[str, Any], 
                                     nodes_by_domain: Dict[str, KnowledgeNode]) -> bool:
        """
        Создает обобщенное определение, объединяющее разные определения.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            nodes_by_domain: Узлы по доменам
            
        Returns:
            bool: Успешно ли создано обобщенное определение
        """
        concept = contradiction["concept"]
        
        # Получаем все описания
        descriptions = [node.meta.get("description", "") for node in nodes_by_domain.values()]
        
        # Если доступен ML, используем его для генерации обобщенного определения
        if self.brain and hasattr(self.brain, 'text_processor'):
            try:
                generalized = self.brain.text_processor.generate_generalized_definition(
                    concept,
                    descriptions
                )
                
                if generalized:
                    # Создаем новый узел с обобщенным определением
                    generalized_id = self.knowledge_graph.add_concept(
                        f"Generalized_{concept}",
                        generalized,
                        domain="metaknowledge",
                        strength=0.8,
                        source="system"
                    )
                    
                    # Связываем обобщенное определение с оригинальными
                    for domain, node in nodes_by_domain.items():
                        self.knowledge_graph.add_edge(
                            generalized_id,
                            node.id,
                            "generalizes",
                            strength=0.7
                        )
                    
                    # Создаем связь с основным концептом
                    concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
                    if concept_nodes:
                        self.knowledge_graph.add_edge(
                            concept_nodes[0].id,
                            generalized_id,
                            "has_generalized_definition",
                            strength=0.9
                        )
                    
                    return True
            except Exception as e:
                logger.warning(f"Ошибка генерации обобщенного определения: {e}")
        
        # Если ML недоступен, пытаемся создать простое обобщение
        try:
            # Объединяем ключевые элементы из всех определений
            all_keywords = []
            for node in nodes_by_domain.values():
                desc = node.meta.get("description", "")
                # Извлекаем ключевые слова
                words = re.findall(r'\b\w+\b', desc.lower())
                stop_words = {'и', 'в', 'на', 'с', 'к', 'от', 'по', 'для', 'о', 'об', 'про'}
                keywords = [word for word in words if word not in stop_words and len(word) > 3]
                all_keywords.extend(keywords)
            
            # Находим самые частые ключевые слова
            common_keywords = [word for word, _ in Counter(all_keywords).most_common(5)]
            
            # Формируем обобщенное определение
            generalized = (
                f"{concept} - это концепт, который может быть описан как "
                f"{', '.join(common_keywords[:3])} в различных контекстах. "
                "Он имеет разные интерпретации в зависимости от домена знаний."
            )
            
            # Создаем новый узел с обобщенным определением
            generalized_id = self.knowledge_graph.add_concept(
                f"Generalized_{concept}",
                generalized,
                domain="metaknowledge",
                strength=0.7,
                source="system"
            )
            
            # Связываем обобщенное определение с оригинальными
            for domain, node in nodes_by_domain.items():
                self.knowledge_graph.add_edge(
                    generalized_id,
                    node.id,
                    "generalizes",
                    strength=0.6
                )
            
            # Создаем связь с основным концептом
            concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if concept_nodes:
                self.knowledge_graph.add_edge(
                    concept_nodes[0].id,
                    generalized_id,
                    "has_generalized_definition",
                    strength=0.8
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания простого обобщенного определения: {e}")
            return False
    
    def _resolve_cyclic_dependency(self, contradiction: Dict[str, Any]) -> bool:
        """
        Разрешает противоречия, вызванные циклическими зависимостями.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли разрешено
        """
        try:
            cycle = contradiction["cycle"]
            parts = cycle.split(" -> ")
            
            if len(parts) < 4:
                return False
            
            # Извлекаем концепты и отношения
            concepts = [parts[i] for i in range(0, len(parts), 2)]
            relations = [parts[i] for i in range(1, len(parts), 2)]
            
            # Проверяем, является ли цикл истинным противоречием
            # или просто отражает сложные отношения
            if self._is_benign_cycle(concepts, relations):
                return self._handle_benign_cycle(contradiction, concepts, relations)
            
            # Оцениваем силу каждого отношения в цикле
            edge_strengths = []
            for i in range(len(concepts)):
                source = concepts[i]
                target = concepts[(i + 1) % len(concepts)]
                relation = relations[i]
                
                strength = self._evaluate_cycle_edge_strength(source, target, relation)
                edge_strengths.append((i, strength))
            
            # Находим самое слабое звено
            weakest_index, _ = min(edge_strengths, key=lambda x: x[1])
            
            # Ослабляем самое слабое звено
            source = concepts[weakest_index]
            target = concepts[(weakest_index + 1) % len(concepts)]
            relation = relations[weakest_index]
            
            source_node = self.knowledge_graph.search_nodes(source, limit=1)
            target_node = self.knowledge_graph.search_nodes(target, limit=1)
            
            if source_node and target_node:
                self._weaken_edge(source_node[0].id, target_node[0].id, relation, 0.3)
                
                # Создаем узел с объяснением
                explanation = (
                    f"Циклическая зависимость между '{source}', '{target}' и другими концептами "
                    "была разрешена путем ослабления связи '{relation}' из-за ее низкой надежности."
                )
                
                explanation_id = self.knowledge_graph.add_concept(
                    f"CycleResolution_{source}_{target}",
                    explanation,
                    domain="metaknowledge",
                    strength=0.7
                )
                
                # Связываем объяснение с концептами
                self.knowledge_graph.add_edge(
                    explanation_id,
                    source_node[0].id,
                    "explains_resolution_for",
                    strength=0.8
                )
                self.knowledge_graph.add_edge(
                    explanation_id,
                    target_node[0].id,
                    "explains_resolution_for",
                    strength=0.8
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка разрешения циклической зависимости: {e}")
            return False
    
    def _is_benign_cycle(self, concepts: List[str], relations: List[str]) -> bool:
        """
        Проверяет, является ли цикл безобидным (не противоречивым).
        
        Args:
            concepts: Список концептов в цикле
            relations: Список отношений в цикле
            
        Returns:
            bool: Является ли цикл безобидным
        """
        # Проверяем, содержит ли цикл отношения, которые могут образовывать
        # естественные циклы (например, в биологических или социальных системах)
        benign_relations = {"influences", "affects", "related_to", "connected_to", "depends_on"}
        
        return all(relation in benign_relations for relation in relations)
    
    def _handle_benign_cycle(self, contradiction: Dict[str, Any], 
                           concepts: List[str], relations: List[str]) -> bool:
        """
        Обрабатывает безобидный цикл, добавляя информацию о его природе.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            concepts: Список концептов в цикле
            relations: Список отношений в цикле
            
        Returns:
            bool: Успешно ли обработано
        """
        cycle_str = " -> ".join([f"{concepts[i]} ({relations[i]})" for i in range(len(relations))]) + f" -> {concepts[0]}"
        
        explanation = (
            f"Цикл '{cycle_str}' представляет собой естественную петлю взаимодействия, "
            "а не истинное противоречие. Такие циклы часто встречаются в сложных системах "
            "и отражают взаимное влияние элементов."
        )
        
        # Создаем узел с объяснением
        explanation_id = self.knowledge_graph.add_concept(
            f"CycleExplanation_{'_'.join(concepts[:2])}",
            explanation,
            domain="metaknowledge",
            strength=0.7
        )
        
        # Связываем объяснение с концептами
        for concept in concepts[:3]:  # Ограничиваем количество связей
            node = self.knowledge_graph.search_nodes(concept, limit=1)
            if node:
                self.knowledge_graph.add_edge(
                    explanation_id,
                    node[0].id,
                    "explains_nature_of",
                    strength=0.8
                )
        
        return True
    
    def _evaluate_cycle_edge_strength(self, source: str, target: str, relation: str) -> float:
        """
        Оценивает силу ребра в цикле.
        
        Args:
            source: Исходный концепт
            target: Целевой концепт
            relation: Тип отношения
            
        Returns:
            float: Сила ребра
        """
        source_node = self.knowledge_graph.search_nodes(source, limit=1)
        if not source_node:
            return 0.3
        
        edges = self.knowledge_graph.get_edges(source_node[0].id)
        for edge in edges:
            edge_target = getattr(edge, 'target_id', getattr(edge, 'target', None))
            target_node = self.knowledge_graph.get_node(edge_target)
            if target_node and target_node.content == target and edge.relation_type == relation:
                return self._evaluate_statement_strength(source_node[0].id, edge_target, relation)
        
        return 0.3
    
    def _resolve_domain_conflict(self, contradiction: Dict[str, Any]) -> bool:
        """
        Разрешает противоречия, вызванные конфликтом доменов.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли разрешено
        """
        try:
            # Получаем информацию о конфликте
            concept = contradiction.get("concept", "unknown_concept")
            domains = contradiction.get("domains", [])
            evidence = contradiction.get("evidence", [])
            
            # Оцениваем авторитетность каждого домена
            domain_authorities = {}
            if self.knowledge_expander and hasattr(self.knowledge_expander, 'get_domain_authority'):
                domain_authorities = {domain: self.knowledge_expander.get_domain_authority(domain) for domain in domains}
            else:
                domain_authorities = {domain: 0.5 for domain in domains}
            
            # Находим наиболее авторитетный домен
            main_domain = max(domain_authorities, key=domain_authorities.get)
            
            # Создаем узел для разрешения конфликта
            resolution_id = self.knowledge_graph.add_concept(
                f"DomainConflict_{concept}",
                f"Конфликт доменов для концепта '{concept}' разрешен в пользу домена '{main_domain}'.",
                domain="metaknowledge",
                strength=0.8
            )
            
            # Связываем разрешение с концептом
            concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if concept_nodes:
                self.knowledge_graph.add_edge(
                    resolution_id,
                    concept_nodes[0].id,
                    "resolves_conflict_for",
                    strength=0.9
                )
            
            # Добавляем информацию о том, что определение зависит от домена
            context = (
                f"Концепт '{concept}' может иметь разные значения или интерпретации "
                f"в зависимости от домена знаний. В контексте '{main_domain}' он определяется как основной."
            )
            
            context_id = self.knowledge_graph.add_concept(
                f"Context_{concept}",
                context,
                domain="metaknowledge",
                strength=0.7
            )
            
            # Связываем контекст с концептом
            if concept_nodes:
                self.knowledge_graph.add_edge(
                    context_id,
                    concept_nodes[0].id,
                    "provides_context",
                    strength=0.8
                )
            
            # Если доступен ML, пытаемся создать междоменную интеграцию
            if self.brain and hasattr(self.brain, 'text_processor'):
                try:
                    integration = self.brain.text_processor.integrate_domain_knowledge(
                        concept,
                        domains
                    )
                    
                    if integration:
                        integration_id = self.knowledge_graph.add_concept(
                            f"DomainIntegration_{concept}",
                            integration,
                            domain="metaknowledge",
                            strength=0.85
                        )
                        
                        # Связываем интеграцию с концептом
                        if concept_nodes:
                            self.knowledge_graph.add_edge(
                                integration_id,
                                concept_nodes[0].id,
                                "integrates_knowledge",
                                strength=0.9
                            )
                except Exception as e:
                    logger.warning(f"Ошибка интеграции доменных знаний: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка разрешения конфликта доменов: {e}")
            return False
    
    def _resolve_general_contradiction(self, contradiction: Dict[str, Any]) -> bool:
        """
        Разрешает общие противоречия с использованием комплексного подхода.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли разрешено
        """
        try:
            # Создаем задачу для более глубокого анализа
            if self.brain and hasattr(self.brain, 'self_analyzer'):
                concept = contradiction.get("concept", "unknown")
                self.brain.self_analyzer.add_learning_opportunity(
                    concept=concept,
                    opportunity_type="integration",
                    priority=0.85,
                    domain="knowledge_consistency",
                    evidence=contradiction.get("evidence", ["Обнаружено противоречие"]),
                    suggested_actions=[
                        "Провести глубокий анализ противоречия",
                        "Интегрировать информацию из разных источников",
                        "Создать гипотезу для разрешения противоречия"
                    ]
                )
            
            # Пытаемся определить тип противоречия на основе доступной информации
            if "concept" in contradiction and "domains" in contradiction:
                return self._resolve_domain_conflict(contradiction)
            elif "cycle" in contradiction:
                return self._resolve_cyclic_dependency(contradiction)
            elif len(contradiction.get("evidence", [])) > 1:
                return self._attempt_hypothesis_based_resolution(contradiction)
            
            # Если ничего не помогло, создаем узел с информацией о противоречии
            contradiction_id = self.knowledge_graph.add_concept(
                f"Contradiction_{contradiction.get('concept', 'unknown')}",
                "Обнаружено противоречие в знаниях. Требуется дополнительный анализ.",
                domain="metaknowledge",
                strength=0.5
            )
            
            # Связываем узел противоречия с концептом
            concept = contradiction.get("concept")
            if concept:
                concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
                if concept_nodes:
                    self.knowledge_graph.add_edge(
                        contradiction_id,
                        concept_nodes[0].id,
                        "concerns",
                        strength=0.7
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка разрешения общего противоречия: {e}")
            return False
    
    def _attempt_hypothesis_based_resolution(self, contradiction: Dict[str, Any]) -> bool:
        """
        Пытается разрешить противоречие с помощью гипотез.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли разрешено
        """
        try:
            concept = contradiction.get("concept", "unknown_concept")
            
            # Генерируем гипотезы
            hypotheses = self._generate_hypotheses(contradiction)
            
            if not hypotheses:
                return False
            
            # Оцениваем гипотезы
            evaluated_hypotheses = []
            for hypothesis in hypotheses:
                score = self._evaluate_hypothesis(hypothesis, contradiction)
                evaluated_hypotheses.append((hypothesis, score))
            
            # Сортируем по оценке
            evaluated_hypotheses.sort(key=lambda x: x[1], reverse=True)
            
            # Выбираем лучшую гипотезу
            best_hypothesis, best_score = evaluated_hypotheses[0]
            
            # Если гипотеза достаточно сильная, применяем ее
            if best_score > 0.6:
                return self._apply_hypothesis(best_hypothesis, contradiction)
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка гипотезного разрешения противоречия: {e}")
            return False
    
    def _generate_hypotheses(self, contradiction: Dict[str, Any]) -> List[str]:
        """
        Генерирует гипотезы для разрешения противоречия.
        
        Args:
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            List[str]: Список гипотез
        """
        hypotheses = []
        
        # Если доступен ML, используем его для генерации гипотез
        if self.brain and hasattr(self.brain, 'text_processor'):
            try:
                concept = contradiction.get("concept", "unknown")
                evidence = contradiction.get("evidence", [])
                
                generated = self.brain.text_processor.generate_contradiction_hypotheses(
                    concept,
                    evidence
                )
                
                if generated:
                    hypotheses.extend(generated)
            except Exception as e:
                logger.warning(f"Ошибка генерации гипотез: {e}")
        
        # Добавляем базовые гипотезы
        concept = contradiction.get("concept", "этот концепт")
        hypotheses.append(
            f"Противоречие в знаниях о {concept} может быть связано с различными контекстами "
            "или условиями применения, которые не были явно указаны."
        )
        hypotheses.append(
            f"Противоречие в знаниях о {concept} может отражать эволюцию понимания этого концепта "
            "со временем, где более новые данные заменяют устаревшие."
        )
        hypotheses.append(
            f"Противоречие в знаниях о {concept} может быть результатом различий в методологиях "
            "исследования или интерпретации данных в разных источниках."
        )
        
        return hypotheses
    
    def _evaluate_hypothesis(self, hypothesis: str, contradiction: Dict[str, Any]) -> float:
        """
        Оценивает гипотезу для разрешения противоречия.
        
        Args:
            hypothesis: Гипотеза
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            float: Оценка гипотезы (0.0-1.0)
        """
        score = 0.5  # Базовая оценка
        
        # Проверяем, содержит ли гипотеза ключевые слова из доказательств
        evidence = contradiction.get("evidence", [])
        if evidence:
            evidence_text = " ".join(evidence).lower()
            hypothesis_text = hypothesis.lower()
            
            # Подсчитываем совпадения
            matches = 0
            for word in evidence_text.split():
                if len(word) > 4 and word in hypothesis_text:  # Игнорируем короткие слова
                    matches += 1
            
            # Нормализуем оценку
            if matches > 0:
                score += min(0.3, matches * 0.05)
        
        # Проверяем структуру гипотезы
        if "может быть связано" in hypothesis_text or "возможно, это" in hypothesis_text:
            score += 0.1  # Хорошая структура гипотезы
        
        # Проверяем, содержит ли гипотеза объяснение
        if "потому что" in hypothesis_text or "так как" in hypothesis_text or "поскольку" in hypothesis_text:
            score += 0.15
        
        # Проверяем, содержит ли гипотеза упоминание о контексте
        if "контекст" in hypothesis_text or "условия" in hypothesis_text or "ситуация" in hypothesis_text:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _apply_hypothesis(self, hypothesis: str, contradiction: Dict[str, Any]) -> bool:
        """
        Применяет гипотезу для разрешения противоречия.
        
        Args:
            hypothesis: Гипотеза
            contradiction: Словарь с информацией о противоречии
            
        Returns:
            bool: Успешно ли применено
        """
        try:
            concept = contradiction.get("concept", "unknown_concept")
            
            # Создаем узел для гипотезы
            hypothesis_id = self.knowledge_graph.add_concept(
                f"Hypothesis_{concept}",
                hypothesis,
                domain="metaknowledge",
                strength=0.75,
                source="system"
            )
            
            # Связываем гипотезу с концептом
            concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if concept_nodes:
                self.knowledge_graph.add_edge(
                    hypothesis_id,
                    concept_nodes[0].id,
                    "explains",
                    strength=0.8
                )
            
            # Если гипотеза включает упоминание о контексте, создаем узел контекста
            hypothesis_lower = hypothesis.lower()
            if "контекст" in hypothesis_lower or "условия" in hypothesis_lower:
                context = (
                    "Этот концепт имеет разные интерпретации или применения в зависимости от контекста. "
                    "Гипотеза разрешает противоречие путем учета этих различий."
                )
                
                context_id = self.knowledge_graph.add_concept(
                    f"Context_{concept}",
                    context,
                    domain="metaknowledge",
                    strength=0.7
                )
                
                # Связываем контекст с концептом и гипотезой
                if concept_nodes:
                    self.knowledge_graph.add_edge(
                        context_id,
                        concept_nodes[0].id,
                        "provides_context",
                        strength=0.8
                    )
                self.knowledge_graph.add_edge(
                    context_id,
                    hypothesis_id,
                    "supports",
                    strength=0.9
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка применения гипотезы: {e}")
            return False
    
    def learn_from_user_feedback(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """
        Учится на основе пользовательского фидбэка.
        
        Args:
            feedback: Данные фидбэка
            user_id: ID пользователя, предоставившего фидбэк
            
        Returns:
            bool: Успешно ли выполнено
        """
        try:
            logger.info("Обучение на основе пользовательского фидбэка...")
            
            # Проверяем тип фидбэка
            feedback_type = feedback.get("feedback_type")
            if feedback_type == "correction":
                # Обрабатываем коррекцию
                return self._handle_correction(feedback, user_id)

            elif feedback_type == "suggestion":
                # Обрабатываем предложение
                return self._handle_suggestion(feedback, user_id)

            elif feedback_type == "rating":
                # Обрабатываем оценку
                return self._handle_rating(feedback, user_id)

            elif feedback_type == "contradiction_report":
                # Обрабатываем сообщение о противоречии
                return self._handle_contradiction_report(feedback, user_id)
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обучения на основе фидбэка: {e}")
            return False
    
    def _handle_contradiction_report(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """
        Обрабатывает сообщение пользователя о противоречии в знаниях.
        
        Args:
            feedback: Данные фидбэка
            user_id: ID пользователя
            
        Returns:
            bool: Успешно ли обработано
        """
        try:
            concept = feedback["concept"]
            description = feedback["description"]
            evidence = feedback.get("evidence", [])
            
            logger.info(f"Обработка сообщения о противоречии для концепта '{concept}': {description}")
            
            # Создаем структуру противоречия для анализа
            contradiction = {
                "concept": concept,
                "description": description,
                "evidence": evidence,
                "user_reported": True,
                "user_id": user_id,
                "timestamp": time.time()
            }
            
            # Пытаемся разрешить противоречие
            resolved = self._resolve_contradiction(contradiction)
            
            # Обновляем надежность источника (пользователя)
            if user_id:
                # Увеличиваем надежность, если противоречие было реальным
                reliability_change = 0.2 if resolved else -0.1
                self.update_source_reliability(f"user_{user_id}", reliability_change)
            
            # Добавляем в историю (если поддерживается)
            if resolved:
                kg = getattr(self, "knowledge_graph", None)
                if kg is not None and hasattr(kg, "record_history"):
                    try:
                        kg.record_history(
                            f"contradiction_{concept}",
                            "contradiction_resolved",
                            None,
                            {"concept": concept, "description": description},
                            user_id
                        )
                    except Exception as ex:
                        logger.debug(f"Не удалось записать историю в KnowledgeGraph: {ex}")
            
            logger.info(f"Сообщение о противоречии для концепта '{concept}' {'разрешено' if resolved else 'не разрешено'}")
            return resolved
            
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения о противоречии: {e}")
            return False
    
    def _handle_correction(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Обрабатывает коррекцию от пользователя."""
        try:
            concept = feedback["concept"]
            correction_type = feedback["correction_type"]
            correction = feedback["correction"]
            
            # Получаем текущий узел
            nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if not nodes:
                # Если концепт не существует, добавляем новый
                self.knowledge_graph.add_concept(
                    concept,
                    correction,
                    domain=feedback.get("domain", "general"),
                    strength=0.9,
                    user_id=user_id
                )
                return True
            
            node = nodes[0]
            
            # Обновляем информацию в зависимости от типа коррекции
            if correction_type == "inaccuracy":
                # Корректируем описание
                updated = self.knowledge_graph.update_concept(
                    node.id,
                    new_description=correction,
                    strength=min(1.0, node.strength + 0.1),
                    user_id=user_id
                )
                return updated
            
            elif correction_type == "missing_info":
                # Добавляем недостающую информацию к существующему описанию
                if not node.meta or not isinstance(node.meta, dict):
                    return False
                new_description = f"{node.meta['description']}\n\nДополнение: {correction}"
                updated = self.knowledge_graph.update_concept(
                    node.id,
                    new_description=new_description,
                    strength=min(1.0, node.strength + 0.05),
                    user_id=user_id
                )
                return updated
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обработки коррекции для концепта '{feedback['concept']}': {e}")
            return False
    
    def _handle_suggestion(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Обрабатывает предложение от пользователя."""
        try:
            suggestion_type = feedback["suggestion_type"]
            
            if suggestion_type == "new_concept":
                # Добавляем новый концепт
                self.knowledge_graph.add_concept(
                    feedback["concept"],
                    feedback["description"],
                    domain=feedback.get("domain", "general"),
                    strength=0.7,
                    user_id=user_id
                )
                return True
            
            elif suggestion_type == "new_connection":
                # Добавляем новую связь
                source_node = self.knowledge_graph.search_nodes(feedback["source_concept"], limit=1)
                target_node = self.knowledge_graph.search_nodes(feedback["target_concept"], limit=1)
                
                if source_node and target_node:
                    self.knowledge_graph.add_edge(
                        source_node[0].id,
                        target_node[0].id,
                        feedback["relation"],
                        strength=0.6,
                        user_id=user_id
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обработки предложения: {e}")
            return False
    
    def _handle_rating(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Обрабатывает оценку от пользователя."""
        try:
            concept = feedback["concept"]
            rating = feedback["rating"]  # 1-5
            
            # Получаем текущий узел
            nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if not nodes:
                return False
            
            node = nodes[0]
            
            # Корректируем силу знания на основе оценки
            # 5 -> +0.1, 4 -> +0.05, 3 -> 0, 2 -> -0.05, 1 -> -0.1
            strength_change = (rating - 3) * 0.05
            new_strength = max(0.1, min(1.0, node.strength + strength_change))
            
            if not node.meta or not isinstance(node.meta, dict) or "description" not in node.meta:
                return False
            return self.knowledge_graph.update_concept(
                node.id,
                new_description=node.meta.get("description", ""),
                strength=new_strength,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Ошибка обработки оценки для концепта '{feedback['concept']}': {e}")
            return False
    
    def auto_integrate_knowledge(self):
        """Автоматически интегрирует знания для улучшения согласованности."""
        try:
            logger.info("Автоматическая интеграция знаний...")
            
            # Анализируем пробелы и заполняем их
            gaps = self.knowledge_analyzer.analyze_knowledge_gaps(num_samples=5)
            for gap in gaps[:3]:  # Только топ-3 пробела
                self.integrate_knowledge(gap["concept"], depth=1)
            
            # Анализируем противоречия и разрешаем их
            contradictions = self.knowledge_analyzer.analyze_contradictions()
            for contradiction in contradictions[:2]:  # Только топ-2 противоречия
                self._resolve_contradiction(contradiction)
            
            # Проводим консолидацию для улучшения согласованности
            self._consolidate_knowledge()
            
            logger.info("Автоматическая интеграция знаний завершена")
            
        except Exception as e:
            logger.error(f"Ошибка автоматической интеграции знаний: {e}")
    
    def _consolidate_knowledge(self):
        """Проводит консолидацию знаний для улучшения согласованности."""
        try:
            logger.info("Консолидация знаний для улучшения согласованности...")
            
            # Анализируем структуру графа
            structure = self.knowledge_graph.analyze_structure()
            
            # Если низкая согласованность, укрепляем связи между похожими узлами
            if structure["coherence"] < 0.4:
                self._strengthen_connections_between_similar_nodes()
            
            # Если много изолированных узлов, пытаемся найти связи
            if structure["isolated_nodes"] > structure["total_nodes"] * 0.1:
                self._connect_isolated_nodes()
            
            logger.info("Консолидация знаний завершена")
            
        except Exception as e:
            logger.error(f"Ошибка консолидации знаний: {e}")
    
    def _strengthen_connections_between_similar_nodes(self, similarity_threshold: float = 0.7):
        """
        Укрепляет связи между похожими узлами для улучшения согласованности.
        
        Args:
            similarity_threshold: Порог сходства для создания связей
        """
        try:
            logger.info(f"Укрепление связей между похожими узлами (порог: {similarity_threshold})")
            
            # Получаем все узлы
            nodes = self.knowledge_graph.get_all_nodes(limit=500)
            
            # Если у нас мало узлов, пропускаем
            if len(nodes) < 10:
                return
            
            # Создаем матрицу сходства
            similarity_matrix = np.zeros((len(nodes), len(nodes)))
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    similarity = self._calculate_node_similarity(nodes[i], nodes[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            # Используем DBSCAN для кластеризации
            clustering = DBSCAN(eps=similarity_threshold, min_samples=2, metric='precomputed').fit(1 - similarity_matrix)
            
            # Для каждого кластера укрепляем связи
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:  # Шум
                    continue
                
                # Получаем узлы в кластере
                cluster_nodes = [nodes[i] for i, label in enumerate(clustering.labels_) if label == cluster_id]
                
                for i in range(len(cluster_nodes)):
                    for j in range(i + 1, len(cluster_nodes)):
                        node1, node2 = cluster_nodes[i], cluster_nodes[j]
                        
                        # Проверяем, есть ли уже связь
                        existing_edge = None
                        for edge in self.knowledge_graph.get_edges(node1.id):
                            if edge.target_id == node2.id:
                                existing_edge = edge
                                break
                        
                        if existing_edge:
                            # Увеличиваем силу существующей связи
                            new_strength = min(1.0, existing_edge.strength + 0.2)
                            self._update_edge_strength(existing_edge.id, new_strength)
                        else:
                            # Создаем новую связь
                            if self.knowledge_expander and hasattr(self.knowledge_expander, '_determine_relation_type'):
                                relation_type = self.knowledge_expander._determine_relation_type(node1.content, node2.content)
                            else:
                                relation_type = "related_to"
                            self.knowledge_graph.add_edge(
                                node1.id,
                                node2.id,
                                relation_type,
                                strength=similarity_matrix[nodes.index(node1), nodes.index(node2)],
                                user_id="system"
                            )
            
            logger.info(f"Укреплены связи между похожими узлами (порог: {similarity_threshold})")
            
        except Exception as e:
            logger.error(f"Ошибка укрепления связей между похожими узлами: {e}")
    
    def _calculate_node_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        """
        Вычисляет сходство между двумя узлами.
        
        Args:
            node1: Первый узел
            node2: Второй узел
            
        Returns:
            float: Сходство (0.0-1.0)
        """
        # Если доступен ML, используем его для вычисления сходства
        if self.brain and hasattr(self.brain, 'text_processor'):
            try:
                # Получаем векторные представления
                vec1 = self.brain.text_processor.get_text_embedding(node1.metadata.get("description", node1.content))
                vec2 = self.brain.text_processor.get_text_embedding(node2.metadata.get("description", node2.content))
                
                # Вычисляем косинусное сходство
                similarity = cosine_similarity([vec1], [vec2])[0][0]
                
                # Нормализуем и ограничиваем диапазон
                return max(0.0, min(1.0, similarity))
            except Exception as e:
                logger.warning(f"Ошибка вычисления сходства с помощью ML: {e}")
        
        # Текстовый анализ как fallback
        node1_meta = getattr(node1, 'metadata', None) or {}
        node2_meta = getattr(node2, 'metadata', None) or {}
        desc1 = node1_meta.get("description", node1.content).lower()
        desc2 = node2_meta.get("description", node2.content).lower()
        
        # Удаляем стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'к', 'от', 'по', 'для', 'о', 'об', 'про',
                     'и', 'в', 'во', 'не', 'что', 'он', 'на', 'с', 'со', 'как', 'а', 'то', 'все',
                     'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только',
                     'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь',
                     'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был',
                     'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя',
                     'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'нее', 'сейчас',
                     'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об',
                     'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего',
                     'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'свою', 'этой', 'перед', 'иногда',
                     'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю',
                     'между', 'уже', 'расскажи', 'напиши', 'покажи', 'какой', 'какая', 'какое', 'какие'}
        
        words1 = set(re.findall(r'\b\w+\b', desc1)) - stop_words
        words2 = set(re.findall(r'\b\w+\b', desc2)) - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        return len(intersection) / len(union)
    
    def _update_edge_strength(self, edge_id: str, new_strength: float):
        """
        Обновляет силу связи.
        
        Args:
            edge_id: ID связи
            new_strength: Новая сила
        """
        try:
            # Используем подключение БД KnowledgeGraph
            conn = getattr(self, "knowledge_graph", None)
            conn = getattr(conn, "db", None)
            if conn is None:
                raise RuntimeError("KnowledgeGraph.db is not initialized")
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE edges 
            SET strength = ? 
            WHERE id = ?
            ''', (new_strength, edge_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Ошибка обновления силы связи {edge_id}: {e}")
    
    def _connect_isolated_nodes(self):
        """Пытается найти связи для изолированных узлов."""
        try:
            logger.info("Поиск связей для изолированных узлов")
            
            # Получаем изолированные узлы
            isolated_nodes = self.knowledge_analyzer._find_isolated_nodes()
            
            for node in isolated_nodes[:10]:  # Ограничиваем 10 узлами
                # Пытаемся найти похожие узлы
                similar_nodes = self._find_similar_nodes(node, limit=3)
                
                for similar_node in similar_nodes:
                    # Определяем тип связи
                    if self.knowledge_expander and hasattr(self.knowledge_expander, '_determine_relation_type'):
                        relation_type = self.knowledge_expander._determine_relation_type(node.content, similar_node.content)
                    else:
                        relation_type = "related_to"
                    
                    # Создаем связь
                    self.knowledge_graph.add_edge(
                        node.id,
                        similar_node.id,
                        relation_type,
                        strength=0.6,
                        user_id="system"
                    )
            
            logger.info(f"Попытка соединить {len(isolated_nodes)} изолированных узлов")
            
        except Exception as e:
            logger.error(f"Ошибка соединения изолированных узлов: {e}")
    
    def _find_similar_nodes(self, node: KnowledgeNode, limit: int = 5) -> List[KnowledgeNode]:
        """
        Находит похожие узлы для данного узла.
        
        Args:
            node: Узел для сравнения
            limit: Максимальное количество результатов
            
        Returns:
            List[KnowledgeNode]: Список похожих узлов
        """
        all_nodes = self.knowledge_graph.get_all_nodes(limit=500)
        similarities = []
        
        for other_node in all_nodes:
            if other_node.id == node.id:
                continue
            
            similarity = self._calculate_node_similarity(node, other_node)
            if similarity > 0.5:  # Минимальный порог сходства
                similarities.append((other_node, similarity))
        
        # Сортируем по сходству
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-N
        return [node for node, _ in similarities[:limit]]