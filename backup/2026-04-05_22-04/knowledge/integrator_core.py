"""Main class KnowledgeIntegrator, initialization, lifecycle, main integration loop."""
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from .knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from .knowledge_analyzer import KnowledgeAnalyzer
from .integrator_sources import (
    _init_source_reliability,
    get_source_reliability,
    update_source_reliability,
    _load_history_for_dynamic_updates,
    learn_from_user_feedback,
    _handle_contradiction_report,
    _handle_correction,
    _handle_suggestion,
    _handle_rating,
)
from .integrator_conflicts import (
    _resolve_contradiction,
    _determine_contradiction_type,
    _resolve_opposite_relations,
    _evaluate_statement_strength,
    _count_confirmations,
    _get_node_vector,
    _find_similar_nodes_by_vector,
    _count_text_confirmations,
    _weaken_edge,
    _update_source_reliability_from_node,
    _create_hypothesis_for_opposite_relations,
    _resolve_conflicting_definitions,
    _create_generalized_definition,
    _resolve_cyclic_dependency,
    _is_benign_cycle,
    _handle_benign_cycle,
    _evaluate_cycle_edge_strength,
    _resolve_domain_conflict,
    _resolve_general_contradiction,
    _attempt_hypothesis_based_resolution,
    _generate_hypotheses,
    _evaluate_hypothesis,
    _apply_hypothesis,
)
from .integrator_sync import (
    integrate_knowledge,
    auto_integrate_knowledge,
    _consolidate_knowledge,
    _strengthen_connections_between_similar_nodes,
    _calculate_node_similarity,
    _update_edge_strength,
    _connect_isolated_nodes,
    _find_similar_nodes,
)

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

        self.source_reliability = defaultdict(float)

        self._init_source_reliability()

        self._load_history_for_dynamic_updates()

        logger.info("KnowledgeIntegrator инициализирован")

    _init_source_reliability = _init_source_reliability
    get_source_reliability = get_source_reliability
    update_source_reliability = update_source_reliability
    _load_history_for_dynamic_updates = _load_history_for_dynamic_updates

    integrate_knowledge = integrate_knowledge
    _resolve_contradiction = _resolve_contradiction
    _determine_contradiction_type = _determine_contradiction_type
    _resolve_opposite_relations = _resolve_opposite_relations
    _evaluate_statement_strength = _evaluate_statement_strength
    _count_confirmations = _count_confirmations
    _get_node_vector = _get_node_vector
    _find_similar_nodes_by_vector = _find_similar_nodes_by_vector
    _count_text_confirmations = _count_text_confirmations
    _weaken_edge = _weaken_edge
    _update_source_reliability_from_node = _update_source_reliability_from_node
    _create_hypothesis_for_opposite_relations = _create_hypothesis_for_opposite_relations
    _resolve_conflicting_definitions = _resolve_conflicting_definitions
    _create_generalized_definition = _create_generalized_definition
    _resolve_cyclic_dependency = _resolve_cyclic_dependency
    _is_benign_cycle = _is_benign_cycle
    _handle_benign_cycle = _handle_benign_cycle
    _evaluate_cycle_edge_strength = _evaluate_cycle_edge_strength
    _resolve_domain_conflict = _resolve_domain_conflict
    _resolve_general_contradiction = _resolve_general_contradiction
    _attempt_hypothesis_based_resolution = _attempt_hypothesis_based_resolution
    _generate_hypotheses = _generate_hypotheses
    _evaluate_hypothesis = _evaluate_hypothesis
    _apply_hypothesis = _apply_hypothesis

    learn_from_user_feedback = learn_from_user_feedback
    _handle_contradiction_report = _handle_contradiction_report
    _handle_correction = _handle_correction
    _handle_suggestion = _handle_suggestion
    _handle_rating = _handle_rating

    auto_integrate_knowledge = auto_integrate_knowledge
    _consolidate_knowledge = _consolidate_knowledge
    _strengthen_connections_between_similar_nodes = _strengthen_connections_between_similar_nodes
    _calculate_node_similarity = _calculate_node_similarity
    _update_edge_strength = _update_edge_strength
    _connect_isolated_nodes = _connect_isolated_nodes
    _find_similar_nodes = _find_similar_nodes


def create_knowledge_integrator(brain=None, knowledge_graph=None, knowledge_analyzer=None) -> KnowledgeIntegrator:
    """
    Factory function для создания KnowledgeIntegrator.

    Args:
        brain: Ссылка на ядро ЕВА (опционально)
        knowledge_graph: Ссылка на граф знаний (опционально)
        knowledge_analyzer: Ссылка на анализатор знаний (опционально)

    Returns:
        KnowledgeIntegrator: Новый экземпляр интегратора знаний
    """
    return KnowledgeIntegrator(
        brain=brain,
        knowledge_graph=knowledge_graph,
        knowledge_analyzer=knowledge_analyzer
    )
