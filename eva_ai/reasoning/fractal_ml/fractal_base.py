"""
Фрактальное хранилище - базовые классы
Иерархическая структура хранения с рекурсивной адресацией
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class FractalNodeType(Enum):
    """Типы узлов во фрактальном хранилище"""
    ROOT = "root"           # L0 - максимально общий
    CATEGORY = "category"   # L1 - категория
    ENTITY = "entity"       # L2 - конкретный объект
    DETAIL = "detail"       # L3 - детальное описание
    REASONING_STEP = "reasoning_step"  # Шаг рассуждения
    CLARIFICATION = "clarification"    # Вопрос уточнения
    QUERY = "query"         # Запрос пользователя
    RESPONSE = "response"   # Ответ системы


class FractalRelationType(Enum):
    """Типы связей между узлами"""
    PARENT_OF = "parent_of"         # L0 → L1
    CONTAINS = "contains"           # L1 → L2
    SPECIFIES = "specifies"         # L2 → L3
    FOLLOWS_FROM = "follows_from"   # Один шаг рассуждения → другой
    CLARIFIES = "clarifies"         # Вопрос уточняет
    BASED_ON = "based_on"           # Основан на
    RELATED_TO = "related_to"       # Связан с


@dataclass
class FractalNode:
    """Узел фрактального хранилища"""
    id: str
    content: str
    node_type: str
    level: int  # 0-3 (L0-L3)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    # Метаданные
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    
    # Контекст
    context: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    
    # Связи
    relations: Dict[str, List[str]] = field(default_factory=dict)  # relation_type -> [node_ids]
    
    def add_child(self, child_id: str):
        """Добавить дочерний узел"""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def add_relation(self, relation_type: str, node_id: str):
        """Добавить связь"""
        if relation_type not in self.relations:
            self.relations[relation_type] = []
        if node_id not in self.relations[relation_type]:
            self.relations[relation_type].append(node_id)
    
    def to_dict(self) -> Dict:
        """Сериализация"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FractalNode':
        """Десериализация"""
        return cls(**data)


@dataclass
class FractalEdge:
    """Связь между узлами"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    
    # Веса и метаданные
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FractalAddress:
    """
    Рекурсивная адресация между уровнями
    L0 → L1 → L2 → L3
    """
    
    # Параметры структуры
    MAX_LEVELS = 4
    BRANCHING_FACTOR = 16
    
    def __init__(self, root_id: str):
        self.root_id = root_id
        self._address_cache: Dict[str, Tuple[int, str]] = {}  # content_hash -> (level, node_id)
    
    @staticmethod
    def compute_address(content: str, level: int) -> str:
        """
        Вычислить адрес для контента на определённом уровне
        Адрес = хэш(content + level)
        """
        addr_input = f"{content}:{level}"
        return hashlib.sha256(addr_input.encode()).hexdigest()[:16]
    
    @staticmethod
    def get_parent_address(child_address: str) -> str:
        """
        Получить адрес родительского узла (уровень - 1)
        """
        # Обратная операция: зная child address, вычисляем parent
        # Это приблизительная операция
        return child_address[:8]  # Берём первые 8 символов
    
    def resolve_address(self, content: str, level: int) -> str:
        """Получить адрес для контента на уровне"""
        addr = self.compute_address(content, level)
        self._address_cache[addr] = (level, addr)
        return addr
    
    def get_level_path(self, node_id: str) -> List[str]:
        """
        Получить полный путь от корня до узла
        Например: [root_id, L0_id, L1_id, L2_id, L3_id]
        """
        path = [self.root_id]
        # Упрощённая реализация - в реальности нужно хранить дерево
        return path


class FractalIndex:
    """Индекс для быстрого поиска по хранилищу"""
    
    def __init__(self):
        # content_hash -> node_ids (инверсированный индекс)
        self._content_index: Dict[str, List[str]] = defaultdict(list)
        
        # level -> node_ids
        self._level_index: Dict[int, List[str]] = defaultdict(list)
        
        # node_type -> node_ids  
        self._type_index: Dict[str, List[str]] = defaultdict(list)
        
        # Текстовый поиск: слово -> node_ids
        self._text_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node: FractalNode):
        """Добавить узел в индекс"""
        node_id = node.id
        
        # Уровневый индекс
        self._level_index[node.level].append(node_id)
        
        # Типовый индекс
        self._type_index[node.node_type].append(node_id)
        
        # Текстовый индекс
        words = node.content.lower().split()
        for word in words:
            if len(word) > 2:
                self._text_index[word].add(node_id)
    
    def remove_node(self, node: FractalNode):
        """Удалить узел из индекса"""
        node_id = node.id
        
        # Уровневый индекс
        if node_id in self._level_index[node.level]:
            self._level_index[node.level].remove(node_id)
        
        # Типовый индекс
        if node_id in self._type_index[node.node_type]:
            self._type_index[node.node_type].remove(node_id)
        
        # Текстовый индекс
        words = node.content.lower().split()
        for word in words:
            if len(word) > 2 and node_id in self._text_index[word]:
                self._text_index[word].discard(node_id)
    
    def search_by_text(self, query: str, limit: int = 10) -> List[str]:
        """Поиск по тексту"""
        words = query.lower().split()
        result_sets = []
        
        for word in words:
            if len(word) > 2 and word in self._text_index:
                result_sets.append(self._text_index[word])
        
        if not result_sets:
            return []
        
        # Пересечение результатов
        result = set.intersection(*result_sets) if len(result_sets) > 1 else result_sets[0]
        return list(result)[:limit]
    
    def search_by_level(self, level: int) -> List[str]:
        """Поиск по уровню"""
        return self._level_index.get(level, [])
    
    def search_by_type(self, node_type: str) -> List[str]:
        """Поиск по типу"""
        return self._type_index.get(node_type, [])


class FractalMetadata:
    """Метаданные всего хранилища"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.total_nodes = 0
        self.nodes_by_level: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_edges = 0
        self.created_at = time.time()
        self.last_updated = time.time()
        self.version = "1.0"
    
    def to_dict(self) -> Dict:
        return {
            "storage_path": self.storage_path,
            "total_nodes": self.total_nodes,
            "nodes_by_level": self.nodes_by_level,
            "total_edges": self.total_edges,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "version": self.version
        }
    
    def save(self, path: str):
        """Сохранить метаданные"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'FractalMetadata':
        """Загрузить метаданные"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        meta = cls(data.get("storage_path", ""))
        meta.total_nodes = data.get("total_nodes", 0)
        meta.nodes_by_level = data.get("nodes_by_level", {0: 0, 1: 0, 2: 0, 3: 0})
        meta.total_edges = data.get("total_edges", 0)
        meta.created_at = data.get("created_at", time.time())
        meta.last_updated = data.get("last_updated", time.time())
        meta.version = data.get("version", "1.0")
        return meta


def create_fractal_id(content: str, node_type: str, level: int) -> str:
    """Создать уникальный ID для узла"""
    timestamp = str(time.time())
    id_input = f"{content}:{node_type}:{level}:{timestamp}"
    return hashlib.sha256(id_input.encode()).hexdigest()[:24]
