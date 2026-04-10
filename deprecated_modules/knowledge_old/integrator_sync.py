"""Synchronization, updates, background sync tasks for KnowledgeIntegrator."""
import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_graph import KnowledgeNode

logger = logging.getLogger("eva_ai.knowledge_integrator")


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

        gaps = self.knowledge_analyzer.analyze_knowledge_gaps(domain=concept, num_samples=5)

        if not gaps:
            gaps = []

        filled_gaps = 0
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            if gap.get("gap_type") == "incomplete":
                gap_concept = gap.get("concept")
                if not gap_concept:
                    continue
                if self.brain and hasattr(self.brain, 'text_processor'):
                    related = self.brain.text_processor.analyze_connection_pattern(
                        gap_concept,
                        [gap_concept],
                        "related_to"
                    )

                    if related and related.get("most_related"):
                        for concept_name in related["most_related"]:
                            relation_type = "related_to"
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
                if self.brain and hasattr(self.brain, 'web_search_engine'):
                    knowledge = self.brain.web_search_engine.web_search_and_learn(
                        gap_concept,
                        num_results=1
                    )

                    if knowledge and isinstance(knowledge, list) and len(knowledge) > 0:
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

        contradictions = self.knowledge_analyzer.detect_contradictions()

        if not contradictions:
            contradictions = []

        resolved_contradictions = 0
        for contradiction in contradictions:
            if self._resolve_contradiction(contradiction):
                resolved_contradictions += 1

        logger.info(f"Интеграция знаний завершена. Заполнено {filled_gaps} пробелов, разрешено {resolved_contradictions} противоречий")
        return filled_gaps > 0 or resolved_contradictions > 0

    except Exception as e:
        logger.error(f"Ошибка интеграции знаний по концепту '{concept}': {e}")

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


def auto_integrate_knowledge(self):
    """Автоматически интегрирует знания для улучшения согласованности."""
    try:
        logger.info("Автоматическая интеграция знаний...")

        gaps = self.knowledge_analyzer.analyze_knowledge_gaps(num_samples=5)
        for gap in gaps[:3]:
            self.integrate_knowledge(gap["concept"], depth=1)

        contradictions = self.knowledge_analyzer.analyze_contradictions()
        for contradiction in contradictions[:2]:
            self._resolve_contradiction(contradiction)

        self._consolidate_knowledge()

        logger.info("Автоматическая интеграция знаний завершена")

    except Exception as e:
        logger.error(f"Ошибка автоматической интеграции знаний: {e}")


def _consolidate_knowledge(self):
    """Проводит консолидацию знаний для улучшения согласованности."""
    try:
        logger.info("Консолидация знаний для улучшения согласованности...")

        structure = self.knowledge_graph.analyze_structure()

        if structure["coherence"] < 0.4:
            self._strengthen_connections_between_similar_nodes()

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

        nodes = self.knowledge_graph.get_all_nodes(limit=500)

        if len(nodes) < 10:
            return

        similarity_matrix = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = self._calculate_node_similarity(nodes[i], nodes[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        clustering = DBSCAN(eps=similarity_threshold, min_samples=2, metric='precomputed').fit(1 - similarity_matrix)

        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue

            cluster_nodes = [nodes[i] for i, label in enumerate(clustering.labels_) if label == cluster_id]

            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    node1, node2 = cluster_nodes[i], cluster_nodes[j]

                    existing_edge = None
                    for edge in self.knowledge_graph.get_edges(node1.id):
                        if edge.target_id == node2.id:
                            existing_edge = edge
                            break

                    if existing_edge:
                        new_strength = min(1.0, existing_edge.strength + 0.2)
                        self._update_edge_strength(existing_edge.id, new_strength)
                    else:
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
    if self.brain and hasattr(self.brain, 'text_processor'):
        try:
            vec1 = self.brain.text_processor.get_text_embedding(node1.metadata.get("description", node1.content))
            vec2 = self.brain.text_processor.get_text_embedding(node2.metadata.get("description", node2.content))

            similarity = cosine_similarity([vec1], [vec2])[0][0]

            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"Ошибка вычисления сходства с помощью ML: {e}")

    node1_meta = getattr(node1, 'metadata', None) or {}
    node2_meta = getattr(node2, 'metadata', None) or {}
    desc1 = node1_meta.get("description", node1.content).lower()
    desc2 = node2_meta.get("description", node2.content).lower()

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

        isolated_nodes = self.knowledge_analyzer._find_isolated_nodes()

        for node in isolated_nodes[:10]:
            similar_nodes = self._find_similar_nodes(node, limit=3)

            for similar_node in similar_nodes:
                if self.knowledge_expander and hasattr(self.knowledge_expander, '_determine_relation_type'):
                    relation_type = self.knowledge_expander._determine_relation_type(node.content, similar_node.content)
                else:
                    relation_type = "related_to"

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
        if similarity > 0.5:
            similarities.append((other_node, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return [node for node, _ in similarities[:limit]]
