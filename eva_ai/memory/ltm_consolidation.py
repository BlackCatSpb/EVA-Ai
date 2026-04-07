"""Consolidation, strengthening, forgetting mechanisms for LongTermMemory."""
import logging
import time
from typing import Dict, List, Any
from collections import defaultdict

from .memory_core import MemoryNeuron, MemoryField

logger = logging.getLogger("eva_ai.memory.long_term")


def _consolidate_from_working_impl(sm) -> None:
    if not sm.working_memory:
        return

    candidates = sm.working_memory.get_consolidation_candidates()

    for neuron in candidates:
        if not sm._is_duplicate(neuron):
            long_term_neuron = MemoryNeuron(
                id=f"semantic_{neuron.id}",
                content=neuron.content,
                content_type=neuron.content_type,
                strength=neuron.strength,
                importance=neuron.importance,
                timestamp=neuron.timestamp,
                metadata={
                    **neuron.metadata,
                    "memory_type": "semantic",
                    "source_working_id": neuron.id
                }
            )

            sm.neurons[long_term_neuron.id] = long_term_neuron
            sm._update_knowledge_graph(long_term_neuron)

            sm.db.save_neuron(long_term_neuron)

            field_name = neuron.metadata.get("field", "general")
            if field_name not in sm.fields:
                sm.fields[field_name] = MemoryField(
                    name=field_name,
                    description=f"Поле семантической памяти: {field_name}",
                    capacity=sm.capacity // 10,
                    metadata={"memory_type": "semantic"}
                )

            sm.fields[field_name].current_size += 1
            sm.db.save_field(sm.fields[field_name])

    sm.last_consolidation = time.time()


def _is_duplicate_impl(sm, neuron: MemoryNeuron) -> bool:
    if neuron.content_type == "fact" and isinstance(neuron.content, dict):
        subject = neuron.content.get("subject")
        predicate = neuron.content.get("predicate")
        obj = neuron.content.get("object")

        if subject and predicate and obj:
            for n in sm.neurons.values():
                if n.content_type != "fact" or not isinstance(n.content, dict):
                    continue

                n_subject = n.content.get("subject")
                n_predicate = n.content.get("predicate")
                n_obj = n.content.get("object")

                if subject == n_subject and predicate == n_predicate and obj == n_obj:
                    return True

    elif neuron.content_type == "text":
        for n in sm.neurons.values():
            if n.content_type != "text":
                continue

            similarity = sm._calculate_text_similarity(str(neuron.content), str(n.content))
            if similarity > 0.8:
                return True

    return False


def _generate_health_recommendations_impl(sm, stats: Dict[str, Any]) -> List[str]:
    recommendations = []

    if stats["usage"] > 0.9:
        recommendations.append(
            "Семантическая память переполнена. Рассмотрите увеличение емкости "
            "или улучшение фильтрации знаний."
        )
    elif stats["usage"] < 0.3:
        recommendations.append(
            "Семантическая память недостаточно заполнена. Проверьте настройки "
            "порога консолидации."
        )

    if stats["avg_importance"] < 0.4:
        recommendations.append(
            "Низкая средняя важность знаний. Проверьте критерии отбора "
            "информации для долгосрочного хранения."
        )

    if stats["knowledge_graph_size"] < len(sm.neurons) * 0.3:
        recommendations.append(
            "Низкая плотность графа знаний. Рассмотрите улучшение процесса "
            "связывания новых знаний с существующими."
        )

    return recommendations


def _consolidate_memory_impl(sm, contradiction_manager=None) -> None:
    logger.info("Начало глубокой консолидации семантической памяти")

    if contradiction_manager:
        sm._check_for_contradictions(contradiction_manager)

    sm._optimize_knowledge_graph()

    logger.info("Глубокая консолидация семантической памяти завершена")


def _check_for_contradictions_impl(sm, contradiction_manager) -> None:
    facts_by_concept = defaultdict(list)
    for neuron_id, neuron in sm.neurons.items():
        if neuron.content_type == "fact" and isinstance(neuron.content, dict):
            concept = neuron.content.get("subject")
            if concept:
                facts_by_concept[concept].append({
                    "id": neuron_id,
                    "fact": neuron.content,
                    "strength": neuron.strength,
                    "importance": neuron.importance
                })

    for concept, facts in facts_by_concept.items():
        if len(facts) < 2:
            continue

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                fact1 = facts[i]["fact"]
                fact2 = facts[j]["fact"]

                if fact1 == fact2:
                    continue

                contradiction = contradiction_manager.detector._create_contradiction(
                    concept,
                    [fact1, fact2],
                    0.5,
                    relation_type=fact1.get("predicate", "related_to")
                )

                contradiction_manager.add_contradiction(contradiction)


def _optimize_knowledge_graph_impl(sm) -> None:
    for concept, connections in list(sm.knowledge_graph.items()):
        strong_connections = [
            conn for conn in connections
            if sm.neurons.get(conn[2], MemoryNeuron("", "", "")).strength > 0.3
        ]
        if strong_connections:
            sm.knowledge_graph[concept] = strong_connections
        else:
            del sm.knowledge_graph[concept]

    sm._merge_similar_concepts()


def _merge_similar_concepts_impl(sm) -> None:
    concepts = list(sm.knowledge_graph.keys())
    merged = set()

    for i in range(len(concepts)):
        if concepts[i] in merged:
            continue

        for j in range(i + 1, len(concepts)):
            if concepts[j] in merged:
                continue

            similarity = sm._calculate_concept_similarity(concepts[i], concepts[j])
            if similarity > 0.8:
                sm._merge_concepts(concepts[i], concepts[j])
                merged.add(concepts[j])


def _merge_concepts_impl(sm, target: str, source: str) -> None:
    if source in sm.knowledge_graph:
        for connection in sm.knowledge_graph[source]:
            sm.knowledge_graph[target].append(connection)
        del sm.knowledge_graph[source]

    for neuron in sm.neurons.values():
        if neuron.content_type == "fact" and isinstance(neuron.content, dict):
            if neuron.content.get("subject") == source:
                neuron.content["subject"] = target
            if neuron.content.get("object") == source:
                neuron.content["object"] = target


def _cleanup_old_episodes_impl(em) -> None:
    current_time = time.time()
    cutoff_time = current_time - em.retention_period

    to_remove = [
        episode_id for episode_id, episode in em.episodes.items()
        if episode.timestamp < cutoff_time
    ]

    for episode_id in to_remove:
        user_id = em.episodes[episode_id].metadata.get("user_id", "system")
        if user_id in em.user_episodes:
            em.user_episodes[user_id] = [
                eid for eid in em.user_episodes[user_id] if eid != episode_id
            ]
            if not em.user_episodes[user_id]:
                del em.user_episodes[user_id]

        if episode_id in dict(em.temporal_index):
            em.temporal_index = [
                (ts, eid) for ts, eid in em.temporal_index if eid != episode_id
            ]

        del em.episodes[episode_id]

    logger.info(f"Удалено {len(to_remove)} устаревших эпизодов")
    em.last_cleanup = current_time


def _generate_episodic_health_recommendations_impl(em, stats: Dict[str, Any]) -> List[str]:
    recommendations = []

    if stats["usage"] > 0.9:
        recommendations.append(
            "Эпизодическая память переполнена. Рассмотрите увеличение емкости "
            "или сокращение периода хранения."
        )
    elif stats["usage"] < 0.2:
        recommendations.append(
            "Эпизодическая память недостаточно используется. Проверьте настройки "
            "периода хранения."
        )

    if stats["avg_history_length"] < 2:
        recommendations.append(
            "Слишком короткая история взаимодействий. Рассмотрите увеличение "
            "периода хранения эпизодов."
        )

    return recommendations


def _episodic_consolidate_from_working_impl(em, user_id: str = "system") -> None:
    if not em.working_memory:
        return

    episodes = em.working_memory.retrieve(
        {"content_type": "episode"},
        nlp_model=None,
        top_k=100
    )

    for episode in episodes:
        if episode.id not in em.episodes:
            episodic_neuron = MemoryNeuron(
                id=f"episodic_{episode.id}",
                content=episode.content,
                content_type="episode",
                strength=episode.strength,
                importance=episode.importance,
                timestamp=episode.timestamp,
                metadata={
                    **episode.metadata,
                    "memory_type": "episodic",
                    "source_working_id": episode.id,
                    "user_id": user_id
                }
            )

            em.store_episode(
                content=episodic_neuron.content,
                user_id=user_id,
                context=episodic_neuron.metadata.get("context")
            )
