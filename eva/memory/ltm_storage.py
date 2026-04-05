"""Storage operations, database/file persistence for LongTermMemory."""
import logging
import time
from typing import Dict, Optional, Any

from .memory_core import MemoryNeuron, MemoryField

logger = logging.getLogger("eva.memory.long_term")


def _load_semantic_from_db(sm) -> None:
    try:
        cursor = sm.db.conn.cursor()
        cursor.execute("SELECT name FROM memory_fields")
        for (field_name,) in cursor.fetchall():
            field = sm.db.load_field(field_name)
            if field and field.metadata.get("memory_type") == "semantic":
                sm.fields[field_name] = field

        cursor.execute("SELECT id FROM memory_neurons")
        for (neuron_id,) in cursor.fetchall():
            neuron = sm.db.load_neuron(neuron_id)
            if neuron and neuron.metadata.get("memory_type") == "semantic":
                sm.neurons[neuron_id] = neuron
                sm._update_knowledge_graph(neuron)

        logger.info(f"Загружено {len(sm.neurons)} нейронов в семантическую память")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных семантической памяти: {e}")


def _update_knowledge_graph_impl(sm, neuron: MemoryNeuron) -> None:
    if neuron.content_type == "fact" and isinstance(neuron.content, dict):
        subject = neuron.content.get("subject")
        predicate = neuron.content.get("predicate")
        obj = neuron.content.get("object")

        if subject and obj:
            sm.knowledge_graph[subject].append((obj, predicate, neuron.id))
            sm.knowledge_graph[obj].append((subject, f"reverse_{predicate}", neuron.id))


def _load_episodic_from_db(em) -> None:
    try:
        cursor = em.db.conn.cursor()
        cursor.execute("SELECT id FROM memory_neurons WHERE content_type = 'episode'")
        for (episode_id,) in cursor.fetchall():
            neuron = em.db.load_neuron(episode_id)
            if neuron:
                em.episodes[episode_id] = neuron
                em.temporal_index.append((neuron.timestamp, episode_id))

                user_id = neuron.metadata.get("user_id", "system")
                em.user_episodes[user_id].append(episode_id)

        em.temporal_index.sort(key=lambda x: x[0])

        logger.info(f"Загружено {len(em.episodes)} эпизодов в эпизодическую память")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных эпизодической памяти: {e}")


def _store_episode_impl(em, content: Any, user_id: str = "system",
                        context: Optional[Dict[str, Any]] = None) -> str:
    timestamp = int(time.time() * 1000)
    content_hash = hash(str(content)) % 1000000
    episode_id = f"episode_{content_hash}_{timestamp}"

    neuron = MemoryNeuron(
        id=episode_id,
        content=content,
        content_type="episode",
        metadata={
            "user_id": user_id,
            "context": context or {},
            "memory_type": "episodic"
        }
    )

    em.episodes[episode_id] = neuron
    em.user_episodes[user_id].append(episode_id)
    em.temporal_index.append((neuron.timestamp, episode_id))
    em.temporal_index.sort(key=lambda x: x[0])

    em.db.save_neuron(neuron)

    return episode_id
