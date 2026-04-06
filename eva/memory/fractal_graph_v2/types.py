"""
Fractal Graph V2 - Фрактальная иерархическая семантическая сеть

Архитектура:
- MySQL-подобная структура (реализовано на SQLite с расширениями)
- Векторные эмбеддинги для семантического поиска
- Фрактальные уровни (L0-LN) с семантическими группами
- Автоматическая кластеризация и детекция противоречий

Структура БД:
  nodes - атомарные единицы знаний
  edges - семантические связи
  semantic_groups - кластеры-образы верхнего уровня
  node_embeddings - векторы узлов
"""

import os
import json
import time
import uuid
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

import numpy as np

logger = logging.getLogger("eva.fractal_graph_v2")


class NodeType(Enum):
    """Типы узлов фрактального графа."""
    # Фрактальные уровни (L0-L3)
    ROOT = "root"           # L0 - корень графа
    CONCEPT = "concept"     # L1 - концепты
    FACT = "fact"           # L2 - факты
    DETAIL = "detail"       # L3 - детали
    
    # Атрибуты
    ATTRIBUTE = "attribute"
    CONTEXT = "context"
    EMOTION = "emotion"
    SENSORY = "sensory"
    
    # Объекты
    OBJECT = "object"
    ENTITY = "entity"
    
    # Связи
    CAUSAL = "causal"
    RELATION = "relation"
    
    # Модели (статичные)
    MODEL_A = "model_a"
    MODEL_B = "model_b"
    MODEL_C = "model_c"
    
    # Семантическая группа (образ)
    SEMANTIC_GROUP = "semantic_group"


class RelationType(Enum):
    """Типы связей между узлами."""
    ATTRIBUTE_OF = "attribute_of"
    CONTEXT_OF = "context_of"
    OBJECT_OF = "object_of"
    EMOTIONAL = "emotional"
    SENSORY = "sensory"
    CAUSAL = "causal"
    IS_A = "is_a"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"


class MemoryTier(Enum):
    """Уровни хранения памяти."""
    HOT = "hot"     # В RAM
    WARM = "warm"   # Сжатые в RAM
    COLD = "cold"   # На диске


@dataclass
class FractalNode:
    """Узел фрактальной памяти - атомарная единица знания."""
    id: str
    content: str                     # Текстовое содержание
    node_type: str                   # Тип узла
    level: int = 0                   # Фрактальный уровень (0 - самый глубокий)
    
    # Иерархия
    parent_group_id: Optional[str] = None  # Семантическая группа
    
    # Вектор (вычисляется отдельно)
    embedding: Optional[List[float]] = None
    
    # Уверенность (0-1), накапливается из подтверждений
    confidence: float = 0.5
    
    # Временные метки
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Статистика
    access_count: int = 0
    version: int = 1
    
    # Флаги
    is_static: bool = False          # Статичные узлы (модели) не удаляются
    is_contradiction: bool = False   # Помечен как противоречие
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FractalNode':
        return cls(**data)


@dataclass
class FractalEdge:
    """Связь между узлами в графе."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    
    # Вес связи (частота подтверждения, семантическая близость)
    weight: float = 0.5
    
    # Временные метки
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Флаги
    contradiction_flag: bool = False  # Связь была предметом противоречия
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FractalEdge':
        return cls(**data)


@dataclass
class SemanticGroup:
    """Семантическая группа (образ) - кластер узлов верхнего уровня."""
    id: str
    name: str                        # Название образа (например, "снег")
    node_type: str = "semantic_group"
    level: int = 2                   # Уровень группы (обычно 2+)
    
    # Вектор группы (центроид или агрегация векторов членов)
    embedding: Optional[List[float]] = None
    
    # Статистика
    member_count: int = 0
    avg_confidence: float = 0.5
    
    # Иерархия
    parent_group_id: Optional[str] = None  # Группа более высокого уровня
    
    # Временные метки
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Кластеризация
    cluster_coherence: float = 0.0    # Связность кластера (0-1)
    needs_recluster: bool = False     # Флаг перекластеризации
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticGroup':
        return cls(**data)


def create_node_id(content: str, node_type: str) -> str:
    """Создать ID узла на основе хеша контента."""
    hash_input = f"{content}:{node_type}"
    return f"node_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"


def create_edge_id(source_id: str, target_id: str, relation: str) -> str:
    """Создать ID связи."""
    hash_input = f"{source_id}:{target_id}:{relation}"
    return f"edge_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"


def create_group_id(name: str) -> str:
    """Создать ID группы."""
    hash_input = f"{name}:{time.time()}"
    return f"group_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"