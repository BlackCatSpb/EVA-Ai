"""
Типы и классы для графа знаний ЕВА
Часть модуля knowledge_graph.py (разделение на логические компоненты)
"""
import json
import time
import copy
from typing import Dict, List, Optional, Any, Union
from enum import Enum


def safe_json_loads(value):
    """Безопасная загрузка JSON с обработкой ошибок."""
    if not value:
        return {}
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, (bytes, bytearray)):
            return json.loads(value.decode('utf-8'))
        else:
            return {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}


class NodeType(Enum):
    CONCEPT = "concept"
    ENTITY = "entity"
    FACT = "fact"
    EVENT = "event"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    PROCESS = "process"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    OTHER = "other"


class RelationType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    CAUSES = "causes"
    SUPPORTS = "supports"
    USED_FOR = "used_for"
    LOCATED_AT = "located_at"
    OCCURS_DURING = "occurs_during"
    PRECEDES = "precedes"
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    OTHER = "other"


class KnowledgeNode:
    """Представляет узел в графе знаний с поддержкой версионирования и метаданных."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        node_type: str = "fact",
        domain: str = "general",
        strength: float = 0.5,
        timestamp: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
        version: int = 1,
        spatial_info: Optional[Dict[str, Any]] = None,
        temporal_info: Optional[Dict[str, Any]] = None
    ) -> None:
        self.id: str = id
        self.name: str = name
        self.description: str = description
        self.content: str = description
        self.node_type: str = node_type
        self.domain: str = domain
        self.strength: float = strength
        self.timestamp: float = timestamp or time.time()
        self.last_updated: float = self.timestamp
        self.meta: Dict[str, Any] = meta or {}
        self.version: int = version
        self.spatial_info: Dict[str, Any] = spatial_info or {}
        self.temporal_info: Dict[str, Any] = temporal_info or {}
        self.embedding: Optional[List[float]] = None
        self.history: List[Dict[str, Any]] = []
        if 'sources' not in self.meta:
            self.meta['sources'] = []
        self.contradictions: List[Dict[str, Any]] = []
        self.keyword_index: List[str] = []
        self.concept_index: List[str] = []
    
    def update(self, new_description: str, strength: Optional[float] = None,
              source: Optional[str] = None, user_id: Optional[str] = None,
              spatial_info: Optional[Dict[str, Any]] = None,
              temporal_info: Optional[Dict[str, Any]] = None) -> None:
        old_state = {
            "description": self.description,
            "strength": self.strength,
            "spatial_info": copy.deepcopy(self.spatial_info),
            "temporal_info": copy.deepcopy(self.temporal_info),
            "timestamp": self.last_updated,
            "version": self.version
        }
        self.description = new_description
        self.content = new_description
        self.last_updated = time.time()
        self.version += 1
        if strength is not None:
            self.strength = strength
        if spatial_info is not None:
            self.spatial_info = spatial_info
        if temporal_info is not None:
            self.temporal_info = temporal_info
        if source:
            self.meta['sources'].append({
                'source': source,
                'timestamp': time.time(),
                'user_id': user_id,
                'version': self.version
            })
        self.history.append({
            "timestamp": time.time(),
            "version": self.version,
            "changes": {
                "old": old_state,
                "new": {
                    "description": new_description,
                    "strength": self.strength,
                    "spatial_info": self.spatial_info,
                    "temporal_info": self.temporal_info
                },
                "source": source,
                "user_id": user_id
            }
        })
    
    def add_contradiction(self, contradictory_node_id: str, evidence: str, 
                         resolution: Optional[str] = None) -> None:
        self.contradictions.append({
            "node_id": contradictory_node_id,
            "evidence": evidence,
            "timestamp": time.time(),
            "resolved": resolution is not None,
            "resolution": resolution
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "node_type": self.node_type,
            "domain": self.domain,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated,
            "version": self.version,
            "meta": self.meta,
            "spatial_info": self.spatial_info,
            "temporal_info": self.temporal_info,
            "history": self.history,
            "contradictions": self.contradictions,
            "keyword_index": self.keyword_index,
            "concept_index": self.concept_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        node = cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            node_type=data.get("node_type", "fact"),
            domain=data.get("domain", "general"),
            strength=data.get("strength", 0.5),
            timestamp=data.get("timestamp"),
            meta=data.get("meta"),
            version=data.get("version", 1),
            spatial_info=data.get("spatial_info"),
            temporal_info=data.get("temporal_info")
        )
        node.last_updated = data.get("last_updated", node.timestamp)
        node.history = data.get("history", [])
        node.contradictions = data.get("contradictions", [])
        node.keyword_index = data.get("keyword_index", [])
        node.concept_index = data.get("concept_index", [])
        return node
    
    def __repr__(self) -> str:
        return f"KnowledgeNode(id={self.id}, name='{self.name}', type={self.node_type}, domain={self.domain})"


class KnowledgeEdge:
    """Представляет связь между узлами в графе знаний."""
    
    def __init__(
        self,
        id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        strength: float = 0.5,
        timestamp: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
        version: int = 1,
        spatial_info: Optional[Dict[str, Any]] = None,
        temporal_info: Optional[Dict[str, Any]] = None
    ) -> None:
        self.id: str = id
        self.source_id: str = source_id
        self.target_id: str = target_id
        self.relation_type: str = relation_type
        self.strength: float = strength
        self.timestamp: float = timestamp or time.time()
        self.last_updated: float = self.timestamp
        self.meta: Dict[str, Any] = meta or {}
        self.version: int = version
        self.spatial_info: Dict[str, Any] = spatial_info or {}
        self.temporal_info: Dict[str, Any] = temporal_info or {}
        self.embedding: Optional[List[float]] = None
        self.history: List[Dict[str, Any]] = []
        if 'sources' not in self.meta:
            self.meta['sources'] = []
    
    def update(self, new_strength: Optional[float] = None,
              source: Optional[str] = None, user_id: Optional[str] = None,
              spatial_info: Optional[Dict[str, Any]] = None,
              temporal_info: Optional[Dict[str, Any]] = None) -> None:
        old_state = {
            "strength": self.strength,
            "spatial_info": copy.deepcopy(self.spatial_info),
            "temporal_info": copy.deepcopy(self.temporal_info),
            "timestamp": self.last_updated,
            "version": self.version
        }
        self.last_updated = time.time()
        self.version += 1
        if new_strength is not None:
            self.strength = new_strength
        if spatial_info is not None:
            self.spatial_info = spatial_info
        if temporal_info is not None:
            self.temporal_info = temporal_info
        if source:
            self.meta['sources'].append({
                'source': source,
                'timestamp': time.time(),
                'user_id': user_id,
                'version': self.version
            })
        self.history.append({
            "timestamp": time.time(),
            "version": self.version,
            "changes": {
                "old": old_state,
                "new": {
                    "strength": self.strength,
                    "spatial_info": self.spatial_info,
                    "temporal_info": self.temporal_info
                },
                "source": source,
                "user_id": user_id
            }
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated,
            "version": self.version,
            "meta": self.meta,
            "spatial_info": self.spatial_info,
            "temporal_info": self.temporal_info,
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        edge = cls(
            id=data.get("id", ""),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=data.get("relation_type", "related_to"),
            strength=data.get("strength", 0.5),
            timestamp=data.get("timestamp"),
            meta=data.get("meta"),
            version=data.get("version", 1),
            spatial_info=data.get("spatial_info"),
            temporal_info=data.get("temporal_info")
        )
        edge.last_updated = data.get("last_updated", edge.timestamp)
        edge.history = data.get("history", [])
        return edge
    
    def __repr__(self) -> str:
        return f"KnowledgeEdge(id={self.id}, {self.source_id} -> {self.target_id}, type={self.relation_type})"
