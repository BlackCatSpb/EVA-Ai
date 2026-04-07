"""MemoryManager - реэкспорт из модульных компонентов."""
from .manager_core import MemoryManager
from .manager_operations import (
    add_memory, get_memory, delete_memory,
    get_user_profile, update_user_profile,
    add_interaction, update_interaction_response,
    get_recent_actions, get_recent_interactions,
    get_conversation_history, get_all_users,
    add_entity_extraction, search_memories_by_entity,
    extract_entities_from_text, get_session_context,
    _load_working_memory, _save_working_memory,
    _load_semantic_memory, _save_semantic_memory,
    _load_episodic_memory, _save_episodic_memory,
    _load_user_profiles, _save_user_profiles,
    _save_memory
)
from .manager_cache import (
    get_memory_statistics, analyze_memory_usage,
    set_cache_size, clear_cache, optimize_cache,
    clear_inactive_caches, compress_data
)
from .manager_gc import (
    get_all_nodes, get_all_edges, get_node, remove_node,
    export_memory_graph, import_memory_graph,
    save_memory_graph_manifest, load_memory_graph_manifest,
    get_graph_data, _MemoryNodeShim
)

__all__ = [
    'MemoryManager',
    'add_memory', 'get_memory', 'delete_memory',
    'get_user_profile', 'update_user_profile',
    'add_interaction', 'update_interaction_response',
    'get_recent_actions', 'get_recent_interactions',
    'get_conversation_history', 'get_all_users',
    'add_entity_extraction', 'search_memories_by_entity',
    'extract_entities_from_text', 'get_session_context',
    'get_memory_statistics', 'analyze_memory_usage',
    'set_cache_size', 'clear_cache', 'optimize_cache',
    'clear_inactive_caches', 'compress_data',
    'get_all_nodes', 'get_all_edges', 'get_node', 'remove_node',
    'export_memory_graph', 'import_memory_graph',
    'save_memory_graph_manifest', 'load_memory_graph_manifest',
    'get_graph_data',
]
