"""Ядро системы знаний ЕВА с поддержкой анализа противоречий"""
import os
import logging
import time
import sqlite3
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from collections import defaultdict, deque
from enum import Enum
from .knowledge_hybrid_index import KnowledgeHybridIndex
from .core_storage import KnowledgeStorageManager
from .core_operations import KnowledgeOperations
from .core_search import KnowledgeSearch
from .core_analytics import KnowledgeAnalytics

logger = logging.getLogger("eva.knowledge.core")

class RelationType(Enum):
    """Типы отношений в графе знаний."""
    IS_A = "is_a"
    PART_OF = "part_of" 
    MEMBER_OF = "member_of"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"

class KnowledgeNode:
    """Представляет узел в графе знаний."""
    
    def __init__(self, id: str, content: Any, node_type: str = "fact", 
                 domain: str = "general", strength: float = 0.5, 
                 timestamp: Optional[float] = None, meta: Optional[Dict] = None):
        """
        Инициализирует узел знаний.
        
        Args:
            id: Уникальный идентификатор узла
            content: Содержимое узла (текст, число, словарь и т.д.)
            node_type: Тип узла (fact, concept, belief и т.д.)
            domain: Домен знаний
            strength: Сила узла (0.0-1.0)
            timestamp: Временная метка создания
            meta: Метаданные узла
        """
        self.id = id
        self.content = content
        self.node_type = node_type
        self.domain = domain
        self.strength = max(0.0, min(1.0, strength))
        self.timestamp = timestamp or time.time()
        self.meta = meta or {}
        self.context = self.meta.get('context', '')
        
        self.get_strength_factor = self.get_strength_factor
        self.update_strength = self.update_strength
    
    def get_strength_factor(self) -> float:
        """Возвращает фактор силы узла с учетом времени и подтверждений."""
        time_factor = 0.95 ** ((time.time() - self.timestamp) / 86400)
        verification_factor = self.meta.get('verification_factor', 1.0)
        source_reputation = self.meta.get('source_reputation', 0.5)
        strength = self.strength * time_factor * verification_factor * (0.5 + source_reputation * 0.5)
        return max(0.1, min(1.0, strength))
    
    def update_strength(self, verification_factor: float = 1.0, 
                        time_factor: float = 1.0, source_reputation: float = None):
        """Обновляет силу узла на основе новых данных."""
        base_strength = self.strength
        if source_reputation is not None:
            if self.meta is None:
                self.meta = {}
            self.meta['source_reputation'] = source_reputation
            source_factor = 0.5 + source_reputation * 0.5
        else:
            source_factor = 1.0
        self.strength = min(1.0, base_strength * verification_factor * time_factor * source_factor)
        self.timestamp = time.time()
        if self.meta is None:
            self.meta = {}
        self.meta['verification_factor'] = verification_factor
        self.meta['last_update'] = self.timestamp
    
    def add_context(self, context: str):
        """Добавляет контекст к узлу."""
        self.context = context
        if self.meta is None:
            self.meta = {}
        self.meta['context'] = context
    
    def get_content_summary(self, max_length: int = 100) -> str:
        """Возвращает краткое описание содержимого узла."""
        if isinstance(self.content, str):
            return self.content[:max_length] + "..." if len(self.content) > max_length else self.content
        elif isinstance(self.content, (int, float)):
            return str(self.content)
        elif isinstance(self.content, dict):
            for field in ['description', 'text', 'content', 'value']:
                if field in self.content and isinstance(self.content[field], str):
                    return self.content[field][:max_length] + "..." if len(self.content[field]) > max_length else self.content[field]
            return str(self.content)[:max_length] + "..."
        else:
            return str(self.content)[:max_length] + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует узел в словарь для сериализации."""
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type,
            "domain": self.domain,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Создает узел из словаря."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            node_type=data.get("node_type", "fact"),
            domain=data.get("domain", "general"),
            strength=data.get("strength", 0.5),
            timestamp=data.get("timestamp"),
            meta=data.get("meta")
        )

class KnowledgeEdge:
    """Представляет связь между узлами в графе знаний."""
    
    def __init__(self, id: str, source: str, target: str, relation: str,
                 strength: float = 0.5, timestamp: Optional[float] = None, 
                 meta: Optional[Dict] = None):
        """
        Инициализирует связь между узлами.
        
        Args:
            id: Уникальный идентификатор связи
            source: ID исходного узла
            target: ID целевого узла
            relation: Тип отношения
            strength: Сила связи (0.0-1.0)
            timestamp: Временная метка создания
            meta: Метаданные связи
        """
        self.id = id
        self.source = source
        self.target = target
        self.relation = relation
        self.strength = max(0.0, min(1.0, strength))
        self.timestamp = timestamp or time.time()
        self.meta = meta or {}
        self.get_priority = self.get_priority
    
    def get_priority(self) -> float:
        """Возвращает приоритет связи с учетом силы и времени."""
        time_factor = 0.95 ** ((time.time() - self.timestamp) / 86400)
        return self.strength * time_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует связь в словарь для сериализации."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        """Создает связь из словаря."""
        return cls(
            id=data.get("id", ""),
            source=data.get("source", ""),
            target=data.get("target", ""),
            relation=data.get("relation", "related_to"),
            strength=data.get("strength", 0.5),
            timestamp=data.get("timestamp"),
            meta=data.get("meta")
        )

class KnowledgeGraph(KnowledgeStorageManager, KnowledgeOperations, KnowledgeSearch, KnowledgeAnalytics):
    """Граф знаний с поддержкой анализа противоречий."""
    
    def __init__(self, storage_path: Optional[str] = None, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует граф знаний.
        
        Args:
            storage_path: Путь к файлу хранения
            brain: Ссылка на ядро системы (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir
        storage_path = storage_path or os.path.join(cache_dir or os.getcwd(), "eva_knowledge.db")
        KnowledgeStorageManager.__init__(self, storage_path)
        self.storage_path = storage_path
        self.initialized = False
        self.nodes = {}
        self.edges = {}
        self.node_edges = defaultdict(list)
        self.domains = defaultdict(list)
        self.contexts = defaultdict(list)
        
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "domains": set(),
            "last_update": time.time()
        }
        
        try:
            base_cache_dir = (
                getattr(self.brain, 'cache_dir', None)
                or self.cache_dir
                or os.path.join(os.getcwd(), 'eva_cache')
            )
            self.hybrid_index = KnowledgeHybridIndex(base_cache_dir=base_cache_dir, namespace="kg_index")
        except Exception as e:
            logger.warning(f"Ошибка инициализации гибридного индекса: {e}")
            self.hybrid_index = None

        logger.info("Граф знаний инициализирован")
    
    def is_ready(self) -> bool:
        """Проверяет готовность графа знаний к работе."""
        return self.initialized and len(self.nodes) >= 0
    
    def initialize(self) -> bool:
        """Инициализирует граф знаний и загружает данные из хранилища."""
        if self.initialized:
            return True
        
        try:
            self.db = self.init_database()
            self.load_nodes(self.nodes, self.domains, self.contexts)
            self.load_edges(self.edges, self.node_edges)
            self._update_stats()
            self.initialized = True
            logger.info("Граф знаний успешно инициализирован")
            return True
            
        except Exception as e:
            logger.critical(f"Критическая ошибка при инициализации графа знаний: {e}")
            return False
    
    def close(self):
        """Закрывает соединение с базой данных."""
        KnowledgeStorageManager.close(self)
    
    def __del__(self):
        """Деструктор для закрытия соединения с базой данных."""
        self.close()
