"""Conflict resolution, merging logic, consistency checks for KnowledgeIntegrator."""
import logging
import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_graph import KnowledgeNode

logger = logging.getLogger("eva.knowledge_integrator")


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

        contradiction_type = self._determine_contradiction_type(contradiction)

        if contradiction_type == "opposite_relations":
            return self._resolve_opposite_relations(contradiction)
        elif contradiction_type == "conflicting_definitions":
            return self._resolve_conflicting_definitions(contradiction)
        elif contradiction_type == "cyclic_dependency":
            return self._resolve_cyclic_dependency(contradiction)
        elif contradiction_type == "domain_conflict":
            return self._resolve_domain_conflict(contradiction)
        else:
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
    if "concept1" in contradiction and "concept2" in contradiction and "common_target" in contradiction:
        return "opposite_relations"

    if "concept" in contradiction and "domains" in contradiction and len(contradiction["domains"]) > 1:
        return "conflicting_definitions"

    if "cycle" in contradiction:
        return "cyclic_dependency"

    if "domains" in contradiction and "evidence" in contradiction and len(contradiction["domains"]) > 1:
        return "domain_conflict"

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

        node1_list = self.knowledge_graph.search_nodes(concept1, limit=1)
        node2_list = self.knowledge_graph.search_nodes(concept2, limit=1)
        target_list = self.knowledge_graph.search_nodes(common_target, limit=1)

        if not node1_list or not node2_list or not target_list:
            return False

        node1 = node1_list[0]
        node2 = node2_list[0]
        target_node = target_list[0]

        self.knowledge_graph.get_edges(node1.id)
        self.knowledge_graph.get_edges(node2.id)

        strength1 = self._evaluate_statement_strength(node1.id, target_node.id, "opposite_of")
        strength2 = self._evaluate_statement_strength(node2.id, target_node.id, "opposite_of")

        if abs(strength1 - strength2) < 0.2:
            return self._create_hypothesis_for_opposite_relations(contradiction)

        if strength1 > strength2:
            self._weaken_edge(node2.id, target_node.id, "opposite_of", 0.2)
            self._update_source_reliability_from_node(node1)
        else:
            self._weaken_edge(node1.id, target_node.id, "opposite_of", 0.2)
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
    edge = None
    edges = self.knowledge_graph.get_edges(source_id)
    for e in edges:
        if e.target_id == target_id and e.relation_type == relation:
            edge = e
            break

    if not edge:
        return 0.0

    strength = edge.strength

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

    current_time = time.time()
    node_age = getattr(node, 'last_updated', current_time)
    age_days = (current_time - node_age) / 86400
    freshness_factor = max(0.2, 1.0 - (age_days / 365))
    strength *= freshness_factor

    domain_authority = 0.8
    strength *= domain_authority

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
    source_node = self.knowledge_graph.get_node(source_id)
    if not source_node:
        return 0

    source_vector = self._get_node_vector(source_node)
    if source_vector is None:
        return self._count_text_confirmations(source_node, target_id, relation)

    similar_nodes = self._find_similar_nodes_by_vector(source_node, source_vector)

    confirmations = 0
    for node, similarity in similar_nodes:
        edges = self.knowledge_graph.get_edges(node.id)
        for edge in edges:
            if edge.target_id == target_id and edge.relation_type == relation:
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
    all_nodes = self.knowledge_graph.get_all_nodes()
    all_nodes = all_nodes[:1000]

    similarities = []
    for other_node in all_nodes:
        if other_node.id == node.id:
            continue

        other_vector = self._get_node_vector(other_node)
        if other_vector is None:
            continue

        similarity = cosine_similarity([node_vector], [other_vector])[0][0]

        if similarity >= threshold:
            similarities.append((other_node, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

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
    description = getattr(node, "description", "") or ""
    if not description:
        return 0

    target_node = self.knowledge_graph.get_node(target_id)
    if not target_node:
        return 0

    relation_keywords = {
        "cause": ["причина", "вызывает", "приводит к", "является причиной"],
        "effect": ["следствие", "результат", "является результатом", "вызывает"],
        "part_of": ["часть", "составная часть", "входит в"],
        "is_a": ["является", "тип", "разновидность", "подкатегория"],
        "related_to": ["связан с", "относится к", "имеет отношение к"],
        "opposite_of": ["противоположность", "противоположный", "антоним"]
    }.get(relation, [])

    description_lower = description.lower()
    target_content_lower = (getattr(target_node, "description", "") or "").lower()

    confirmations = 0
    for keyword in relation_keywords:
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
        edges = self.knowledge_graph.get_edges(source_id)
        for edge in edges:
            if edge.target_id == target_id and edge.relation_type == relation:
                new_strength = max(0.1, edge.strength - amount)

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
            self.update_source_reliability(primary_source, 0.2)


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

        hypothesis = (
            f"Концепты '{concept1}' и '{concept2}' могут быть противоположными по отношению к '{common_target}', "
            "но в разных контекстах или с разных точек зрения. Возможно, это отражает сложность или многогранность "
            "концепта, а не истинное противоречие."
        )

        hypothesis_id = self.knowledge_graph.add_node(
            name=f"Hypothesis_{concept1}_{concept2}",
            description=hypothesis,
            node_type="hypothesis",
            domain="metaknowledge",
            strength=0.6,
            meta={"sources": ["system"]}
        )

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

        nodes_by_domain = {}
        for domain in domains:
            nodes = self.knowledge_graph.search_nodes(concept, domains=[domain], limit=1)
            if nodes:
                nodes_by_domain[domain] = nodes[0]

        if not nodes_by_domain:
            return False

        domain_scores = {}
        for domain, node in nodes_by_domain.items():
            score = 0.0

            if getattr(node, 'source', None):
                score += self.get_source_reliability(node.source) * 0.4

            age_days = (time.time() - node.last_updated) / 86400
            score += max(0.0, 1.0 - (age_days / 365)) * 0.3

            if self.knowledge_expander and hasattr(self.knowledge_expander, 'get_domain_authority'):
                score += self.knowledge_expander.get_domain_authority(domain) * 0.3

            domain_scores[domain] = score

        best_domain = max(domain_scores, key=domain_scores.get)
        best_node = nodes_by_domain[best_domain]

        if max(domain_scores.values()) - min(domain_scores.values()) < 0.3:
            return self._create_generalized_definition(contradiction, nodes_by_domain)

        best_desc = best_node.meta.get("description", "") if isinstance(best_node.meta, dict) else ""
        self.knowledge_graph.update_node(
            best_node.id,
            best_desc,
            strength=min(1.0, best_node.strength + 0.2)
        )

        for domain, node in nodes_by_domain.items():
            if domain != best_domain:
                node_desc = node.meta.get("description", "") if isinstance(node.meta, dict) else ""
                self.knowledge_graph.update_node(
                    node.id,
                    node_desc,
                    strength=max(0.3, node.strength - 0.1)
                )

        context_node_id = self.knowledge_graph.add_concept(
            f"Context_{concept}",
            f"Концепт '{concept}' имеет разные определения в разных доменах. "
            f"Основное определение используется в домене '{best_domain}'.",
            domain="metaknowledge",
            strength=0.7
        )

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

    descriptions = [node.meta.get("description", "") for node in nodes_by_domain.values()]

    if self.brain and hasattr(self.brain, 'text_processor'):
        try:
            generalized = self.brain.text_processor.generate_generalized_definition(
                concept,
                descriptions
            )

            if generalized:
                generalized_id = self.knowledge_graph.add_concept(
                    f"Generalized_{concept}",
                    generalized,
                    domain="metaknowledge",
                    strength=0.8,
                    source="system"
                )

                for domain, node in nodes_by_domain.items():
                    self.knowledge_graph.add_edge(
                        generalized_id,
                        node.id,
                        "generalizes",
                        strength=0.7
                    )

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

    try:
        all_keywords = []
        for node in nodes_by_domain.values():
            desc = node.meta.get("description", "")
            words = re.findall(r'\b\w+\b', desc.lower())
            stop_words = {'и', 'в', 'на', 'с', 'к', 'от', 'по', 'для', 'о', 'об', 'про'}
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            all_keywords.extend(keywords)

        common_keywords = [word for word, _ in Counter(all_keywords).most_common(5)]

        generalized = (
            f"{concept} - это концепт, который может быть описан как "
            f"{', '.join(common_keywords[:3])} в различных контекстах. "
            "Он имеет разные интерпретации в зависимости от домена знаний."
        )

        generalized_id = self.knowledge_graph.add_concept(
            f"Generalized_{concept}",
            generalized,
            domain="metaknowledge",
            strength=0.7,
            source="system"
        )

        for domain, node in nodes_by_domain.items():
            self.knowledge_graph.add_edge(
                generalized_id,
                node.id,
                "generalizes",
                strength=0.6
            )

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

        concepts = [parts[i] for i in range(0, len(parts), 2)]
        relations = [parts[i] for i in range(1, len(parts), 2)]

        if self._is_benign_cycle(concepts, relations):
            return self._handle_benign_cycle(contradiction, concepts, relations)

        edge_strengths = []
        for i in range(len(concepts)):
            source = concepts[i]
            target = concepts[(i + 1) % len(concepts)]
            relation = relations[i]

            strength = self._evaluate_cycle_edge_strength(source, target, relation)
            edge_strengths.append((i, strength))

        weakest_index, _ = min(edge_strengths, key=lambda x: x[1])

        source = concepts[weakest_index]
        target = concepts[(weakest_index + 1) % len(concepts)]
        relation = relations[weakest_index]

        source_node = self.knowledge_graph.search_nodes(source, limit=1)
        target_node = self.knowledge_graph.search_nodes(target, limit=1)

        if source_node and target_node:
            self._weaken_edge(source_node[0].id, target_node[0].id, relation, 0.3)

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

    explanation_id = self.knowledge_graph.add_concept(
        f"CycleExplanation_{'_'.join(concepts[:2])}",
        explanation,
        domain="metaknowledge",
        strength=0.7
    )

    for concept in concepts[:3]:
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
        concept = contradiction.get("concept", "unknown_concept")
        domains = contradiction.get("domains", [])
        evidence = contradiction.get("evidence", [])

        domain_authorities = {}
        if self.knowledge_expander and hasattr(self.knowledge_expander, 'get_domain_authority'):
            domain_authorities = {domain: self.knowledge_expander.get_domain_authority(domain) for domain in domains}
        else:
            domain_authorities = {domain: 0.5 for domain in domains}

        main_domain = max(domain_authorities, key=domain_authorities.get)

        resolution_id = self.knowledge_graph.add_concept(
            f"DomainConflict_{concept}",
            f"Конфликт доменов для концепта '{concept}' разрешен в пользу домена '{main_domain}'.",
            domain="metaknowledge",
            strength=0.8
        )

        concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
        if concept_nodes:
            self.knowledge_graph.add_edge(
                resolution_id,
                concept_nodes[0].id,
                "resolves_conflict_for",
                strength=0.9
            )

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

        if concept_nodes:
            self.knowledge_graph.add_edge(
                context_id,
                concept_nodes[0].id,
                "provides_context",
                strength=0.8
            )

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

        if "concept" in contradiction and "domains" in contradiction:
            return self._resolve_domain_conflict(contradiction)
        elif "cycle" in contradiction:
            return self._resolve_cyclic_dependency(contradiction)
        elif len(contradiction.get("evidence", [])) > 1:
            return self._attempt_hypothesis_based_resolution(contradiction)

        contradiction_id = self.knowledge_graph.add_concept(
            f"Contradiction_{contradiction.get('concept', 'unknown')}",
            "Обнаружено противоречие в знаниях. Требуется дополнительный анализ.",
            domain="metaknowledge",
            strength=0.5
        )

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

        hypotheses = self._generate_hypotheses(contradiction)

        if not hypotheses:
            return False

        evaluated_hypotheses = []
        for hypothesis in hypotheses:
            score = self._evaluate_hypothesis(hypothesis, contradiction)
            evaluated_hypotheses.append((hypothesis, score))

        evaluated_hypotheses.sort(key=lambda x: x[1], reverse=True)

        best_hypothesis, best_score = evaluated_hypotheses[0]

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
    score = 0.5

    evidence = contradiction.get("evidence", [])
    if evidence:
        evidence_text = " ".join(evidence).lower()
        hypothesis_text = hypothesis.lower()

        matches = 0
        for word in evidence_text.split():
            if len(word) > 4 and word in hypothesis_text:
                matches += 1

        if matches > 0:
            score += min(0.3, matches * 0.05)

    if "может быть связано" in hypothesis_text or "возможно, это" in hypothesis_text:
        score += 0.1

    if "потому что" in hypothesis_text or "так как" in hypothesis_text or "поскольку" in hypothesis_text:
        score += 0.15

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

        hypothesis_id = self.knowledge_graph.add_concept(
            f"Hypothesis_{concept}",
            hypothesis,
            domain="metaknowledge",
            strength=0.75,
            source="system"
        )

        concept_nodes = self.knowledge_graph.search_nodes(concept, limit=1)
        if concept_nodes:
            self.knowledge_graph.add_edge(
                hypothesis_id,
                concept_nodes[0].id,
                "explains",
                strength=0.8
            )

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
