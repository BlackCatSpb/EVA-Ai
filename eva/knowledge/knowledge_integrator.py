"""Модуль интеграции знаний для ЕВА - улучшение согласованности знаний.

This module has been refactored into smaller, focused modules:
- integrator_core.py: Main class, initialization, lifecycle
- integrator_sources.py: Source management, fetching data, external APIs
- integrator_conflicts.py: Conflict resolution, merging logic, consistency checks
- integrator_sync.py: Synchronization, updates, background sync tasks

All functionality is preserved. This module re-exports the main classes and
factory functions for backward compatibility.
"""
from .integrator_core import KnowledgeIntegrator, create_knowledge_integrator
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

__all__ = [
    "KnowledgeIntegrator",
    "create_knowledge_integrator",
]
