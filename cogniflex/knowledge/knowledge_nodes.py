"""
Модуль узлов и связей графа знаний для CogniFlex
Содержит базовые классы для представления знаний
"""
import time
import copy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


class KnowledgeNode:
    """Представляет узел в графе знаний с поддержкой версионирования и метаданных."""
    
    def __init__(self, id: str, name: str, description: str, 
                 node_type: str = "fact", domain: str = "general",
                 strength: float = 0.5, timestamp: Optional[float] = None,
                 meta: Optional[Dict] = None, version: int = 1,
                 spatial_info: Optional[Dict[str, Any]] = None,
                 temporal_info: Optional[Dict[str, Any]] = None):
        """
        Инициализирует узел знаний.
        
        Args:
            id: Уникальный идентификатор узла
            name: Название узла
            description: Описание узла
            node_type: Тип узла
            domain: Домен знаний
            strength: Сила знания (0.0-1.0)
            timestamp: Временная метка создания
            meta: Дополнительные метаданные
            version: Версия узла
            spatial_info: Пространственная информация
            temporal_info: Временная информация
        """
        self.id = id
        self.name = name
        self.description = description
        self.content = description  # Для совместимости с другими модулями
        self.node_type = node_type
        self.domain = domain
        self.strength = strength
        self.timestamp = timestamp or time.time()
        self.last_updated = self.timestamp
        self.meta = meta or {}
        self.version = version
        self.spatial_info = spatial_info or {}
        self.temporal_info = temporal_info or {}
        self.embedding = None  # Векторное представление узла
        
        # Для отслеживания истории изменений
        self.history = []
        if self.meta is None:
            self.meta = {}
        if 'sources' not in self.meta:
            self.meta['sources'] = []
        
        # Для отслеживания противоречий
        self.contradictions = []
        
        # Для поддержки семантического поиска
        self.keyword_index = []
        self.concept_index = []
    
    def update(self, new_description: str, strength: Optional[float] = None,
              source: Optional[str] = None, user_id: Optional[str] = None,
              spatial_info: Optional[Dict[str, Any]] = None,
              temporal_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Обновляет узел знаний, создавая новую версию.
        
        Args:
            new_description: Новое описание
            strength: Новая сила связи
            source: Источник информации
            user_id: ID пользователя
            spatial_info: Новая пространственная информация
            temporal_info: Новая временная информация
        """
        # Сохраняем старое состояние для истории
        old_state = {
            "description": self.description,
            "strength": self.strength,
            "spatial_info": copy.deepcopy(self.spatial_info),
            "temporal_info": copy.deepcopy(self.temporal_info),
            "timestamp": self.last_updated,
            "version": self.version
        }
        
        # Обновляем узел
        self.description = new_description
        self.content = new_description  # Для совместимости
        self.last_updated = time.time()
        self.version += 1
        
        if strength is not None:
            self.strength = strength
        
        if spatial_info is not None:
            self.spatial_info = spatial_info
        
        if temporal_info is not None:
            self.temporal_info = temporal_info
        
        # Добавляем источник
        if source:
            if self.meta is None:
                self.meta = {}
            if 'sources' not in self.meta:
                self.meta['sources'] = []
            self.meta['sources'].append({
                'source': source,
                'timestamp': time.time(),
                'user_id': user_id,
                'version': self.version
            })
        
        # Записываем в историю
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
        """
        Добавляет информацию о противоречии.
        
        Args:
            contradictory_node_id: ID противоречивого узла
            evidence: Доказательства противоречия
            resolution: Предложенное разрешение
        """
        self.contradictions.append({
            "node_id": contradictory_node_id,
            "evidence": evidence,
            "timestamp": time.time(),
            "resolved": resolution is not None,
            "resolution": resolution
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует узел в словарь для сериализации."""
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
        """Создает узел из словаря."""
        node = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            node_type=data["node_type"],
            domain=data["domain"],
            strength=data["strength"],
            timestamp=data["timestamp"],
            meta=data["meta"],
            version=data["version"],
            spatial_info=data["spatial_info"],
            temporal_info=data["temporal_info"]
        )
        
        node.last_updated = data["last_updated"]
        node.history = data["history"]
        node.contradictions = data["contradictions"]
        node.keyword_index = data["keyword_index"]
        node.concept_index = data["concept_index"]
        
        return node
    
    def __repr__(self) -> str:
        return f"KnowledgeNode(id={self.id}, name='{self.name}', type={self.node_type}, domain={self.domain})"


class KnowledgeEdge:
    """Представляет связь между узлами в графе знаний."""
    
    def __init__(self, id: str, source_id: str, target_id: str, 
                 relation_type: str, strength: float = 0.5,
                 timestamp: Optional[float] = None, meta: Optional[Dict] = None,
                 version: int = 1, spatial_info: Optional[Dict[str, Any]] = None,
                 temporal_info: Optional[Dict[str, Any]] = None):
        """
        Инициализирует связь между узлами.
        
        Args:
            id: Уникальный идентификатор связи
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation_type: Тип связи
            strength: Сила связи (0.0-1.0)
            timestamp: Временная метка создания
            meta: Дополнительные метаданные
            version: Версия связи
            spatial_info: Пространственная информация
            temporal_info: Временная информация
        """
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.strength = strength
        self.timestamp = timestamp or time.time()
        self.last_updated = self.timestamp
        self.meta = meta or {}
        self.version = version
        self.spatial_info = spatial_info or {}
        self.temporal_info = temporal_info or {}
        self.embedding = None  # Векторное представление связи
        
        # История изменений
        self.history = []
        
        # Для отслеживания противоречий
        self.contradictions = []
    
    def update(self, new_relation_type: Optional[str] = None, 
              strength: Optional[float] = None, source: Optional[str] = None,
              user_id: Optional[str] = None) -> None:
        """
        Обновляет связь, создавая новую версию.
        
        Args:
            new_relation_type: Новый тип связи
            strength: Новая сила связи
            source: Источник информации
            user_id: ID пользователя
        """
        # Сохраняем старое состояние
        old_state = {
            "relation_type": self.relation_type,
            "strength": self.strength,
            "timestamp": self.last_updated,
            "version": self.version
        }
        
        # Обновляем связь
        if new_relation_type is not None:
            self.relation_type = new_relation_type
        
        if strength is not None:
            self.strength = strength
        
        self.last_updated = time.time()
        self.version += 1
        
        # Добавляем источник
        if source:
            if self.meta is None:
                self.meta = {}
            if 'sources' not in self.meta:
                self.meta['sources'] = []
            self.meta['sources'].append({
                'source': source,
                'timestamp': time.time(),
                'user_id': user_id,
                'version': self.version
            })
        
        # Записываем в историю
        self.history.append({
            "timestamp": time.time(),
            "version": self.version,
            "changes": {
                "old": old_state,
                "new": {
                    "relation_type": self.relation_type,
                    "strength": self.strength
                },
                "source": source,
                "user_id": user_id
            }
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует связь в словарь для сериализации."""
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
            "history": self.history,
            "contradictions": self.contradictions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        """Создает связь из словаря."""
        edge = cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            strength=data["strength"],
            timestamp=data["timestamp"],
            meta=data["meta"],
            version=data["version"],
            spatial_info=data["spatial_info"],
            temporal_info=data["temporal_info"]
        )
        
        edge.last_updated = data["last_updated"]
        edge.history = data["history"]
        edge.contradictions = data["contradictions"]
        
        return edge
    
    def __repr__(self) -> str:
        return f"KnowledgeEdge(id={self.id}, {self.source_id}->{self.target_id}, type={self.relation_type})"


# Функции для работы с узлами и связями
def create_node_id(name: str, domain: str = "general") -> str:
    """Создает уникальный ID для узла на основе имени и домена."""
    import hashlib
    base = f"{domain}:{name}"
    return hashlib.md5(base.encode()).hexdigest()[:16]


def create_edge_id(source_id: str, target_id: str, relation_type: str) -> str:
    """Создает уникальный ID для связи."""
    import hashlib
    base = f"{source_id}->{target_id}:{relation_type}"
    return hashlib.md5(base.encode()).hexdigest()[:16]


def validate_node(node: KnowledgeNode) -> bool:
    """Проверяет корректность узла."""
    if not node.id or not node.name:
        return False
    if not (0.0 <= node.strength <= 1.0):
        return False
    if node.version < 1:
        return False
    return True


def validate_edge(edge: KnowledgeEdge) -> bool:
    """Проверяет корректность связи."""
    if not edge.id or not edge.source_id or not edge.target_id:
        return False
    if not edge.relation_type:
        return False
    if not (0.0 <= edge.strength <= 1.0):
        return False
    if edge.version < 1:
        return False
    return True
