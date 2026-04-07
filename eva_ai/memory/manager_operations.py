"""CRUD operations and storage for MemoryManager."""
import os
import logging
import json
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva_ai.memory.manager")


def _load_working_memory(manager) -> None:
    try:
        if os.path.exists(manager.working_memory_file):
            with open(manager.working_memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    manager.working_memory = {item["id"]: item for item in data if "id" in item}
                else:
                    manager.working_memory = data
    except Exception as e:
        logger.error(f"Ошибка загрузки рабочей памяти: {e}")
        manager.working_memory = {}


def _save_working_memory(manager) -> None:
    try:
        if isinstance(manager.working_memory, dict):
            data = list(manager.working_memory.values())
        else:
            data = manager.working_memory
        with open(manager.working_memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения рабочей памяти: {e}")


def _load_semantic_memory(manager) -> None:
    try:
        if os.path.exists(manager.semantic_memory_file):
            with open(manager.semantic_memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    manager.semantic_memory = {item["id"]: item for item in data if "id" in item}
                else:
                    manager.semantic_memory = data
    except Exception as e:
        logger.error(f"Ошибка загрузки семантической памяти: {e}")
        manager.semantic_memory = {}


def _save_semantic_memory(manager) -> None:
    try:
        if isinstance(manager.semantic_memory, dict):
            data = list(manager.semantic_memory.values())
        else:
            data = manager.semantic_memory
        with open(manager.semantic_memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения семантической памяти: {e}")


def _load_episodic_memory(manager) -> None:
    try:
        if os.path.exists(manager.episodic_memory_file):
            with open(manager.episodic_memory_file, 'r', encoding='utf-8') as f:
                manager.episodic_memory = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки эпизодической памяти: {e}")
        manager.episodic_memory = []


def _save_episodic_memory(manager) -> None:
    try:
        with open(manager.episodic_memory_file, 'w', encoding='utf-8') as f:
            json.dump(manager.episodic_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения эпизодической памяти: {e}")


def _load_user_profiles(manager) -> None:
    try:
        if os.path.exists(manager.user_profiles_file):
            with open(manager.user_profiles_file, 'r', encoding='utf-8') as f:
                manager.user_profiles = json.load(f)
        else:
            manager.user_profiles = {}
    except Exception as e:
        logger.error(f"Ошибка загрузки профилей пользователей: {e}")
        manager.user_profiles = {}


def _save_user_profiles(manager) -> None:
    try:
        with open(manager.user_profiles_file, 'w', encoding='utf-8') as f:
            json.dump(manager.user_profiles, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения профилей пользователей: {e}")


def _save_memory(manager, memory_type: str) -> None:
    if memory_type == "working":
        _save_working_memory(manager)
    elif memory_type == "semantic":
        _save_semantic_memory(manager)
    elif memory_type == "episodic":
        _save_episodic_memory(manager)


def add_memory(manager, memory_type: str, content: Any, metadata: Optional[Dict] = None, user_id: Optional[str] = None) -> str:
    if memory_type not in manager.memory_locks:
        raise ValueError(f"Неизвестный тип памяти: {memory_type}")

    if content is None:
        raise ValueError("Content cannot be None")

    if isinstance(content, str):
        if len(content) > 100000:
            raise ValueError(f"Content too large: {len(content)} chars (max 100000)")
        if not content.strip():
            raise ValueError("Content cannot be empty string")
    elif isinstance(content, (dict, list)):
        try:
            content_str = json.dumps(content)
            if len(content_str) > 100000:
                raise ValueError(f"Content too large: {len(content_str)} chars (max 100000)")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Content cannot be serialized to JSON: {e}")
    elif not isinstance(content, (int, float, bool)):
        raise ValueError(f"Unsupported content type: {type(content).__name__}")

    memory_id = f"mem_{int(time.time())}_{os.urandom(4).hex()}"
    timestamp = time.time()

    memory_entry = {
        "id": memory_id,
        "content": content,
        "timestamp": timestamp,
        "metadata": metadata or {},
        "user_id": user_id
    }

    with manager.memory_locks[memory_type]:
        if memory_type == "working":
            if len(manager.working_memory) >= manager.max_working_memory:
                oldest_key = min(
                    (k for k, v in manager.working_memory.items() if isinstance(v, dict)),
                    key=lambda k: manager.working_memory[k].get('timestamp', 0),
                    default=None
                )
                if oldest_key is not None:
                    del manager.working_memory[oldest_key]
                    logger.debug(f"Removed oldest working memory entry: {oldest_key}")
            manager.working_memory[memory_id] = memory_entry
        elif memory_type == "semantic":
            if len(manager.semantic_memory) >= manager.max_semantic_memory:
                oldest_key = min(
                    (k for k, v in manager.semantic_memory.items() if isinstance(v, dict)),
                    key=lambda k: manager.semantic_memory[k].get('timestamp', 0),
                    default=None
                )
                if oldest_key is not None:
                    del manager.semantic_memory[oldest_key]
                    logger.debug(f"Removed oldest semantic memory entry: {oldest_key}")
            manager.semantic_memory[memory_id] = memory_entry
        elif memory_type == "episodic":
            if len(manager.episodic_memory) >= manager.max_episodic_memory:
                manager.episodic_memory.sort(key=lambda x: x.get('timestamp', 0) if isinstance(x, dict) else 0)
                manager.episodic_memory.pop(0)
                logger.debug("Removed oldest episodic memory entry")
            manager.episodic_memory.append(memory_entry)

        _save_memory(manager, memory_type)

    logger.debug(f"Добавлена информация в {memory_type} память: {memory_id}")
    return memory_id


def get_memory(manager, memory_id: str) -> Optional[Dict]:
    for memory_type in ["working", "semantic", "episodic"]:
        with manager.memory_locks[memory_type]:
            memory_obj = getattr(manager, f"{memory_type}_memory", None)
            if memory_obj is None:
                continue
            if isinstance(memory_obj, dict):
                entry = memory_obj.get(memory_id)
                if isinstance(entry, dict):
                    return entry.copy()
            elif isinstance(memory_obj, list):
                for entry in memory_obj:
                    if isinstance(entry, dict) and entry.get("id") == memory_id:
                        return entry.copy()
    return None


def delete_memory(manager, memory_id: str, memory_type: Optional[str] = None) -> bool:
    if memory_type:
        with manager.memory_locks[memory_type]:
            memory_obj = getattr(manager, f"{memory_type}_memory")
            if isinstance(memory_obj, dict):
                if memory_id in memory_obj:
                    del memory_obj[memory_id]
                    _save_memory(manager, memory_type)
                    logger.debug(f"Удалена информация из {memory_type} памяти: {memory_id}")
                    return True
            elif isinstance(memory_obj, list):
                for i, entry in enumerate(memory_obj):
                    if isinstance(entry, dict) and entry.get("id") == memory_id:
                        memory_obj.pop(i)
                        _save_memory(manager, memory_type)
                        logger.debug(f"Удалена информация из {memory_type} памяти: {memory_id}")
                        return True
            return False
    else:
        for memory_type in ["working", "semantic", "episodic"]:
            if delete_memory(manager, memory_id, memory_type):
                return True
        return False


def get_user_profile(manager, user_id: str) -> Dict:
    with manager.memory_locks["user_profiles"]:
        if user_id not in manager.user_profiles:
            manager.user_profiles[user_id] = {
                "id": user_id,
                "preferences": {},
                "interaction_history": [],
                "created_at": time.time(),
                "last_active": time.time()
            }
        return manager.user_profiles[user_id].copy()


def update_user_profile(manager, user_id: str, updates: Dict) -> bool:
    with manager.memory_locks["user_profiles"]:
        if user_id not in manager.user_profiles and len(manager.user_profiles) >= manager.max_user_profiles:
            oldest_user = min(
                (k for k, v in manager.user_profiles.items() if isinstance(v, dict)),
                key=lambda k: manager.user_profiles[k].get('last_active', 0),
                default=None
            )
            if oldest_user is not None:
                del manager.user_profiles[oldest_user]
                logger.debug(f"Removed oldest user profile: {oldest_user}")

        if user_id not in manager.user_profiles:
            manager.user_profiles[user_id] = {
                "id": user_id,
                "preferences": {},
                "interaction_history": [],
                "created_at": time.time(),
                "last_active": time.time()
            }

        for key, value in updates.items():
            if key == "preferences":
                manager.user_profiles[user_id]["preferences"].update(value)
            elif key == "interaction_history":
                manager.user_profiles[user_id]["interaction_history"].extend(value)
            else:
                manager.user_profiles[user_id][key] = value

        max_history = 100
        if len(manager.user_profiles[user_id].get("interaction_history", [])) > max_history:
            manager.user_profiles[user_id]["interaction_history"] = manager.user_profiles[user_id]["interaction_history"][-max_history:]

        manager.user_profiles[user_id]["last_active"] = time.time()
        _save_user_profiles(manager)
        logger.debug(f"Профиль пользователя {user_id} обновлен")
        return True


def add_interaction(manager, user_id: str, query: str, response: str, context: Optional[Dict] = None) -> str:
    interaction_id = f"inter_{int(time.time())}_{os.urandom(4).hex()}"

    interaction = {
        "id": interaction_id,
        "user_id": user_id,
        "query": query,
        "response": response,
        "timestamp": time.time(),
        "context": context or {}
    }

    add_memory(manager, "working", interaction, {"type": "interaction"}, user_id)

    with manager.memory_locks["user_profiles"]:
        if user_id not in manager.user_profiles:
            manager.user_profiles[user_id] = {
                "id": user_id,
                "preferences": {},
                "interaction_history": [],
                "created_at": time.time(),
                "last_active": time.time()
            }

        manager.user_profiles[user_id]["interaction_history"].append(interaction)
        manager.user_profiles[user_id]["last_active"] = time.time()

    _save_user_profiles(manager)
    logger.debug(f"Взаимодействие добавлено: {interaction_id}")
    return interaction_id


def update_interaction_response(manager, interaction_id: str, response: str) -> bool:
    for memory_type in ["working", "semantic", "episodic"]:
        with manager.memory_locks[memory_type]:
            memory_obj = getattr(manager, f"{memory_type}_memory", None)
            if memory_obj is None:
                continue
            if isinstance(memory_obj, dict):
                entry = memory_obj.get(interaction_id)
                if isinstance(entry, dict):
                    content = entry.get("content")
                    if isinstance(content, dict) and entry.get("metadata", {}).get("type") == "interaction":
                        content["response"] = response
                        _save_memory(manager, memory_type)
                        logger.debug(f"Ответ в истории обновлен: {interaction_id}")
                        return True
            elif isinstance(memory_obj, list):
                for entry in memory_obj:
                    if not isinstance(entry, dict):
                        continue
                    content = entry.get("content")
                    if not isinstance(content, dict):
                        continue
                    if entry.get("id") == interaction_id and entry.get("metadata", {}).get("type") == "interaction":
                        content["response"] = response
                        _save_memory(manager, memory_type)
                        logger.debug(f"Ответ в истории обновлен: {interaction_id}")
                        return True

    with manager.memory_locks["user_profiles"]:
        for user_id, profile in manager.user_profiles.items():
            for i, interaction in enumerate(profile["interaction_history"]):
                if interaction["id"] == interaction_id:
                    profile["interaction_history"][i]["response"] = response
                    _save_user_profiles(manager)
                    logger.debug(f"Ответ в профиле пользователя обновлен: {interaction_id}")
                    return True

    return False


def get_recent_actions(manager, limit: int = 100) -> List[Dict[str, Any]]:
    actions = []

    with manager.memory_locks["working"]:
        for entry in manager.working_memory.values():
            if "type" in entry.get("metadata", {}) and entry["metadata"]["type"] == "action":
                actions.append({
                    "id": entry["id"],
                    "type": entry["metadata"].get("action_type", "unknown"),
                    "description": entry["content"],
                    "timestamp": entry["timestamp"],
                    "system": True
                })

    with manager.memory_locks["user_profiles"]:
        for user_id, profile in manager.user_profiles.items():
            for interaction in profile["interaction_history"]:
                actions.append({
                    "id": interaction["id"],
                    "type": "user_interaction",
                    "description": f"Пользователь {user_id}: {interaction['query']}",
                    "timestamp": interaction["timestamp"],
                    "user_id": user_id
                })

    actions.sort(key=lambda x: x["timestamp"], reverse=True)
    return actions[:limit]


def get_recent_interactions(manager, limit: int = 50) -> List[Dict[str, Any]]:
    interactions = []

    with manager.memory_locks["user_profiles"]:
        for user_id, profile in manager.user_profiles.items():
            interactions.extend(profile["interaction_history"])

    interactions.sort(key=lambda x: x["timestamp"], reverse=True)
    return interactions[:limit]


def get_conversation_history(manager, user_id: str = "default_user", limit: int = 10) -> List[Dict[str, Any]]:
    interactions = get_recent_interactions(manager, limit)

    conversation_history = []
    for interaction in interactions:
        if isinstance(interaction, dict):
            if user_id and interaction.get("user_id") != user_id:
                continue
            conversation_history.append({
                "query": interaction.get("query", ""),
                "response": interaction.get("response", ""),
                "timestamp": interaction.get("timestamp", 0)
            })

    return conversation_history


def get_all_users(manager) -> List[Dict[str, Any]]:
    with manager.memory_locks["user_profiles"]:
        return [{"id": user_id, "last_active": profile["last_active"]}
                for user_id, profile in manager.user_profiles.items()]


def add_entity_extraction(manager, memory_id: str, entities: List[Dict]) -> None:
    if memory_id in manager.working_memory:
        manager.working_memory[memory_id]["extracted_entities"] = entities
        _save_memory(manager, "working")

    if memory_id in manager.semantic_memory:
        manager.semantic_memory[memory_id]["extracted_entities"] = entities
        _save_memory(manager, "semantic")


def search_memories_by_entity(manager, entity_term: str) -> List[Dict]:
    results = []

    for mem_list in [manager.working_memory, manager.semantic_memory]:
        for mem in mem_list.values():
            entities = mem.get("extracted_entities", [])
            for entity in entities:
                if entity_term.lower() in str(entity.get("term", "")).lower():
                    results.append(mem)
                    break

    return results


def extract_entities_from_text(manager, text: str) -> List[Dict]:
    if not manager.entity_extractor:
        return []

    entities = manager.entity_extractor.extract_ambiguous_terms(text)
    return [
        {
            "term": e.text,
            "type": e.ambiguity_type.value,
            "context": e.context,
            "confidence": e.confidence
        }
        for e in entities
    ]


def get_session_context(manager, session_id: str) -> Dict:
    if manager.episodic_memory:
        context_messages = []
        for entry in manager.episodic_memory:
            if isinstance(entry, dict) and entry.get('session_id') == session_id:
                context_messages.append(entry.get('content', ''))
        return {'context': '\n'.join(context_messages[-10:])}
    return {}
