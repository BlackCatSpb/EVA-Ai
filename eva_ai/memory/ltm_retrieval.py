"""Retrieval algorithms, search, recall for LongTermMemory."""
import logging
import time
from typing import Dict, List, Optional, Any

from .memory_core import MemoryNeuron

logger = logging.getLogger("eva_ai.memory.long_term")


def _retrieve_by_concept_impl(sm, concept: str, nlp_model=None,
                              max_distance: int = 2, top_k: int = 10) -> List[MemoryNeuron]:
    related_neurons = set()
    queue = [(concept, 0)]
    visited = {concept}

    while queue and len(related_neurons) < top_k * 2:
        current_concept, distance = queue.pop(0)

        if distance > max_distance:
            continue

        for neuron_id in sm.knowledge_graph.get(current_concept, []):
            related_neurons.add(neuron_id[2])

        for neighbor, _, _ in sm.knowledge_graph.get(current_concept, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    results = []
    for neuron_id in list(related_neurons)[:top_k]:
        if neuron_id in sm.neurons:
            results.append(sm.neurons[neuron_id])

    return results


def _retrieve_by_similarity_impl(sm, query: Any, nlp_model,
                                  threshold: float = 0.6, top_k: int = 5) -> List[MemoryNeuron]:
    results = []

    for neuron in sm.neurons.values():
        similarity = neuron.get_similarity(MemoryNeuron(
            id="query",
            content=query,
            content_type="text" if isinstance(query, str) else "fact"
        ), nlp_model)

        if similarity >= threshold:
            results.append((similarity, neuron))

    results.sort(key=lambda x: x[0], reverse=True)

    return [neuron for _, neuron in results[:top_k]]


def _retrieve_by_time_impl(em, start_time: Optional[float] = None,
                            end_time: Optional[float] = None,
                            user_id: Optional[str] = None,
                            top_k: int = 10) -> List[MemoryNeuron]:
    start_time = start_time or (time.time() - 86400)
    end_time = end_time or time.time()

    episodes = [
        em.episodes[episode_id] for ts, episode_id in em.temporal_index
        if start_time <= ts <= end_time
    ]

    if user_id:
        episodes = [
            e for e in episodes
            if e.metadata.get("user_id") == user_id
        ]

    return episodes[:top_k]


def _retrieve_episodic_by_similarity_impl(em, query: Any, user_id: Optional[str] = None,
                                           nlp_model=None, threshold: float = 0.6,
                                           top_k: int = 5) -> List[MemoryNeuron]:
    candidates = []

    for episode in em.episodes.values():
        if user_id and episode.metadata.get("user_id") != user_id:
            continue

        similarity = episode.get_similarity(MemoryNeuron(
            id="query",
            content=query,
            content_type="text" if isinstance(query, str) else "fact"
        ), nlp_model)

        if similarity >= threshold:
            candidates.append((similarity, episode))

    candidates.sort(key=lambda x: x[0], reverse=True)

    return [episode for _, episode in candidates[:top_k]]


def _get_user_history_impl(em, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
    cutoff_time = time.time() - (days * 86400)

    history = []
    for episode_id in em.user_episodes.get(user_id, []):
        episode = em.episodes.get(episode_id)
        if episode and episode.timestamp >= cutoff_time:
            history.append({
                "timestamp": episode.timestamp,
                "content": episode.content,
                "context": episode.metadata.get("context", {}),
                "importance": episode.importance
            })

    history.sort(key=lambda x: x["timestamp"], reverse=True)

    return history
