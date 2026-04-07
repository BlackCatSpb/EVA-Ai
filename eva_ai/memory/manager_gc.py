"""Garbage collection, graph export/import, and optimization for MemoryManager."""
import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Iterable
from pathlib import Path

logger = logging.getLogger("eva_ai.memory.manager")


class _MemoryNodeShim:
    """Простой адаптер для представления записи памяти как узла для GUI."""
    def __init__(self, entry: Dict[str, Any]):
        self._e = entry
        self.id = entry.get("id")
        self.content = entry.get("content")
        self.node_type = entry.get("metadata", {}).get("type", "fact")
        self.domain = entry.get("metadata", {}).get("domain", "unknown")
        ts = entry.get("timestamp", time.time())
        self.created_at = ts
        self.last_updated = ts
        self.timestamp = ts
        self.meta = entry.get("metadata", {})
        self.edges: list = []

    def get_strength_factor(self) -> float:
        strength = self._e.get("metadata", {}).get("strength") if isinstance(self._e, dict) else None
        try:
            return float(strength) if strength is not None else 1.0
        except Exception:
            return 1.0


def get_all_nodes(manager) -> List[Any]:
    nodes: List[_MemoryNodeShim] = []
    try:
        for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
            mem = getattr(manager, mem_type, None)
            if isinstance(mem, dict):
                for entry in mem.values():
                    if isinstance(entry, dict) and "id" in entry:
                        nodes.append(_MemoryNodeShim(entry))
            elif isinstance(mem, list):
                for entry in mem:
                    if isinstance(entry, dict) and "id" in entry:
                        nodes.append(_MemoryNodeShim(entry))
        return nodes
    except Exception as e:
        logger.error(f"Ошибка получения узлов памяти: {e}")
        return []


def get_all_edges(manager) -> List[Any]:
    return []


def get_node(manager, node_id: str) -> Optional[Any]:
    try:
        for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
            mem = getattr(manager, mem_type, None)
            if isinstance(mem, dict):
                for entry in mem.values():
                    if isinstance(entry, dict) and entry.get("id") == node_id:
                        return _MemoryNodeShim(entry)
            elif isinstance(mem, list):
                for entry in mem:
                    if isinstance(entry, dict) and entry.get("id") == node_id:
                        return _MemoryNodeShim(entry)
        return None
    except Exception as e:
        logger.error(f"Ошибка получения узла {node_id}: {e}")
        return None


def remove_node(manager, node_id: str) -> bool:
    removed = False
    try:
        for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
            mem = getattr(manager, mem_type, None)
            if isinstance(mem, dict):
                key_to_remove = None
                for key, entry in mem.items():
                    if isinstance(entry, dict) and entry.get("id") == node_id:
                        key_to_remove = key
                        break
                if key_to_remove is not None:
                    del mem[key_to_remove]
                    removed = True
            elif isinstance(mem, list):
                indices_to_remove = []
                for i, entry in enumerate(mem):
                    if isinstance(entry, dict) and entry.get("id") == node_id:
                        indices_to_remove.append(i)
                for i in sorted(indices_to_remove, reverse=True):
                    del mem[i]
                if indices_to_remove:
                    removed = True
            if removed:
                from .manager_operations import _save_memory
                _save_memory(manager, mem_type.replace("_memory", ""))
                break
        return removed
    except Exception as e:
        logger.error(f"Ошибка удаления узла {node_id}: {e}")
        return False


def export_memory_graph(manager) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    now_ts = time.time()

    def add_node(entry: Dict[str, Any], kind: str) -> None:
        records.append({
            "type": "node",
            "id": str(entry.get("id")),
            "kind": kind,
            "ts": float(entry.get("timestamp", now_ts)),
            "attrs": {
                "content": entry.get("content"),
                "metadata": entry.get("metadata", {}),
                "user_id": entry.get("user_id"),
            },
        })

    try:
        for entry in manager.working_memory.values():
            if isinstance(entry, dict) and entry.get("id"):
                add_node(entry, "working")
    except Exception:
        logger.debug("Ошибка экспорта working_memory", exc_info=True)

    try:
        for entry in manager.semantic_memory.values():
            if isinstance(entry, dict) and entry.get("id"):
                add_node(entry, "semantic")
    except Exception:
        logger.debug("Ошибка экспорта semantic_memory", exc_info=True)

    try:
        for entry in manager.episodic_memory:
            if isinstance(entry, dict) and entry.get("id"):
                add_node(entry, "episodic")
    except Exception:
        logger.debug("Ошибка экспорта episodic_memory", exc_info=True)

    try:
        for user_id, profile in manager.user_profiles.items():
            user_node_id = f"user:{user_id}"
            records.append({
                "type": "node",
                "id": user_node_id,
                "kind": "user_profile",
                "ts": float(profile.get("last_active", now_ts)),
                "attrs": profile,
            })
            for interaction in profile.get("interaction_history", []):
                if not isinstance(interaction, dict) or "id" not in interaction:
                    continue
                inter_id = str(interaction["id"])
                records.append({
                    "type": "node",
                    "id": inter_id,
                    "kind": "interaction",
                    "ts": float(interaction.get("timestamp", now_ts)),
                    "attrs": interaction,
                })
                records.append({
                    "type": "edge",
                    "src": user_node_id,
                    "dst": inter_id,
                    "label": "performed",
                    "ts": float(interaction.get("timestamp", now_ts)),
                    "attrs": {},
                })
    except Exception:
        logger.debug("Ошибка экспорта user_profiles", exc_info=True)

    return records


def import_memory_graph(manager, records: Iterable[Dict[str, Any]]) -> Tuple[int, int]:
    nodes_count = 0
    edges_count = 0
    try:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if rec.get("type") != "node":
                continue
            kind = rec.get("kind")
            nid = rec.get("id")
            attrs = rec.get("attrs", {}) or {}
            ts = float(rec.get("ts", time.time()))
            if not nid or not kind:
                continue
            if kind in ("working", "semantic", "episodic"):
                entry = {
                    "id": nid,
                    "content": attrs.get("content"),
                    "timestamp": ts,
                    "metadata": attrs.get("metadata", {}),
                    "user_id": attrs.get("user_id"),
                }
                if kind == "working":
                    with manager.memory_locks["working"]:
                        manager.working_memory[nid] = entry
                        from .manager_operations import _save_working_memory
                        _save_working_memory(manager)
                elif kind == "semantic":
                    with manager.memory_locks["semantic"]:
                        manager.semantic_memory[nid] = entry
                        from .manager_operations import _save_semantic_memory
                        _save_semantic_memory(manager)
                elif kind == "episodic":
                    with manager.memory_locks["episodic"]:
                        manager.episodic_memory.append(entry)
                        from .manager_operations import _save_episodic_memory
                        _save_episodic_memory(manager)
                nodes_count += 1
            elif kind == "user_profile":
                profile = dict(attrs)
                uid = profile.get("id") or str(nid).split(":", 1)[-1]
                with manager.memory_locks["user_profiles"]:
                    manager.user_profiles[uid] = profile
                    from .manager_operations import _save_user_profiles
                    _save_user_profiles(manager)
                nodes_count += 1
            elif kind == "interaction":
                interaction = dict(attrs)
                inter_id = interaction.get("id", nid)
                from .manager_operations import add_memory
                add_memory(manager, "working", interaction, {"type": "interaction"}, interaction.get("user_id"))
                nodes_count += 1

        for rec in records:
            if isinstance(rec, dict) and rec.get("type") == "edge":
                edges_count += 1
    except Exception:
        logger.error("Ошибка импорта графа памяти", exc_info=True)
    return nodes_count, edges_count


def save_memory_graph_manifest(
    manager,
    manifest_dir: str,
    records: Iterable[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    manifest_filename: str = "manifest.jsonl",
    meta_filename: str = "manifest_meta.json",
) -> Tuple[str, str]:
    base = Path(manifest_dir)
    base.mkdir(parents=True, exist_ok=True)

    manifest_path = base / manifest_filename
    meta_path = base / meta_filename
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

    with open(tmp_manifest, "w", encoding="utf-8") as f:
        for rec in records:
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.debug("Failed to write record to manifest file: %s", e)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception as e:
            logger.debug("Failed to flush/fsync manifest file: %s", e)
    os.replace(tmp_manifest, manifest_path)

    meta_obj = meta or {}
    meta_obj.setdefault("version", 1)
    meta_obj.setdefault("created_ts", time.time())
    with open(tmp_meta, "w", encoding="utf-8") as mf:
        json.dump(meta_obj, mf, ensure_ascii=False, indent=2)
        try:
            mf.flush()
            os.fsync(mf.fileno())
        except Exception as e:
            logger.debug("Failed to flush/fsync meta file: %s", e)
    os.replace(tmp_meta, meta_path)

    return str(manifest_path), str(meta_path)


def load_memory_graph_manifest(
    manager,
    manifest_dir: str,
    manifest_filename: str = "manifest.jsonl",
    meta_filename: str = "manifest_meta.json",
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base = Path(manifest_dir)
    manifest_path = base / manifest_filename
    meta_path = base / meta_filename

    records: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    try:
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as mf:
                meta = json.load(mf)
    except Exception:
        logger.debug("Не удалось прочитать manifest_meta.json", exc_info=True)

    if not manifest_path.exists():
        return records, meta

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= int(limit):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        logger.debug("Не удалось прочитать manifest.jsonl", exc_info=True)
    return records, meta


def get_graph_data(manager) -> Dict:
    nodes = []
    edges = []
    node_ids = set()

    if isinstance(manager.working_memory, dict):
        for key, value in manager.working_memory.items():
            if isinstance(value, dict):
                label = str(value.get("content", ""))[:50]
                nodes.append({'id': key, 'label': label, 'type': 'working'})
                node_ids.add(key)

    if isinstance(manager.semantic_memory, dict):
        for key, value in manager.semantic_memory.items():
            if isinstance(value, dict):
                label = str(value.get("content", ""))[:50]
                nodes.append({'id': key, 'label': label, 'type': 'semantic'})
                node_ids.add(key)

    if isinstance(manager.episodic_memory, list):
        for entry in manager.episodic_memory:
            if isinstance(entry, dict) and entry.get("id"):
                eid = entry["id"]
                label = str(entry.get("content", ""))[:50]
                session_id = entry.get("session_id")
                nodes.append({'id': eid, 'label': label, 'type': 'episodic'})
                node_ids.add(eid)
                if session_id:
                    edges.append({'source': eid, 'target': f"session:{session_id}", 'label': 'belongs_to'})

    if isinstance(manager.user_profiles, dict):
        for user_id, profile in manager.user_profiles.items():
            uid = f"user:{user_id}"
            nodes.append({'id': uid, 'label': user_id, 'type': 'user_profile'})
            node_ids.add(uid)
            if isinstance(profile, dict):
                for interaction in profile.get("interaction_history", []):
                    if isinstance(interaction, dict) and interaction.get("id"):
                        iid = interaction["id"]
                        if iid in node_ids:
                            edges.append({'source': uid, 'target': iid, 'label': 'performed'})

    return {'nodes': nodes, 'edges': edges, 'stats': {'total_nodes': len(nodes), 'total_edges': len(edges)}}
