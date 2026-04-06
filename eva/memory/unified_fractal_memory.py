"""
UnifiedFractalMemory — единая фрактальная память на SSD

Архитектура:
- Модели A, B, C статично встроены как постоянные узлы графа
- Знания ЕВЫ хранятся фрактально (L0-L3) вокруг моделей
- Hot/Cold tiering: активные узлы в RAM, остальные на SSD
- Модели загружаются через llama.cpp и живут как Llama instances в графе

SSD Structure:
  unified_fractal_memory/
  ├── nodes.json          — все узлы графа
  ├── edges.json          — все связи
  ├── models/             — GGUF файлы моделей
  │   ├── model_a.gguf
  │   ├── model_b.gguf
  │   └── model_c.gguf
  └── hot_cache/          — горячие узлы в RAM (сериализованные)
"""

import os
import json
import time
import hashlib
import logging
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    HOT = "hot"       # В RAM, мгновенный доступ
    WARM = "warm"     # Сжатые в RAM
    COLD = "cold"     # На SSD, подгрузка по запросу


class NodeType(Enum):
    # Статичные узлы моделей (НИКОГДА не удаляются)
    MODEL_A = "model_a"           # Qwen 2.5 3B - логика
    MODEL_B = "model_b"           # Qwen 2.5 3B - развитие
    MODEL_C = "model_c"           # Qwen 2.5 Coder 1.5B - код
    
    # Фрактальные уровни знаний
    ROOT = "root"                 # L0 - корень
    CONCEPT = "concept"           # L1 - концепты
    FACT = "fact"                 # L2 - факты
    DETAIL = "detail"             # L3 - детали
    
    # Динамические узлы
    QUERY = "query"
    RESPONSE = "response"
    REASONING = "reasoning"
    CONTEXT = "context"


class EventTypes:
    """Типы событий для миграции памяти"""
    MEMORY_TIER_MIGRATED = "memory.tier_migrated"


@dataclass
class MemoryNode:
    """Узел фрактальной памяти"""
    id: str
    node_type: str
    level: int  # 0-3
    content: str
    tier: str = "cold"  # hot, warm, cold
    
    # Для модельных узлов
    model_path: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Метаданные
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    usage_count: int = 0  # Для миграции - отслеживание использования
    last_accessed: float = field(default_factory=time.time)
    version: int = 1
    
    # Контекст и связи
    context: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    
    # Флаги
    is_static: bool = False  # Статичные узлы (модели) не удаляются
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryNode':
        return cls(**data)


@dataclass
class MemoryEdge:
    """Связь между узлами"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class UnifiedFractalMemory:
    """
    Единая фрактальная память на SSD со статично встроенными моделями.
    
    Модели A, B, C — постоянные узлы графа, никогда не удаляются.
    Знания хранятся фрактально L0-L3 вокруг моделей.
    """
    
    MAX_LEVELS = 4
    BRANCHING_FACTOR = 16
    HOT_NODE_LIMIT = 1000      # Макс узлов в RAM
    WARM_NODE_LIMIT = 5000     # Макс сжатых узлов в RAM
    AUTO_SAVE_INTERVAL = 10    # Сохранять каждые N операций
    
    def __init__(self, storage_dir: str, config: Dict[str, Any] = None):
        self.storage_dir = storage_dir
        self.config = config or {}
        
        # Поддиректории
        self.nodes_file = os.path.join(storage_dir, "nodes.json")
        self.edges_file = os.path.join(storage_dir, "edges.json")
        self.models_dir = os.path.join(storage_dir, "models")
        self.hot_cache_dir = os.path.join(storage_dir, "hot_cache")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.hot_cache_dir, exist_ok=True)
        
        # Граф
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        
        # Llama instances (живые модели)
        self.model_instances: Dict[str, Any] = {}  # model_type -> Llama instance
        
        # Hot cache tracking
        self.hot_nodes: set = set()
        self.warm_nodes: set = set()
        
        # Lock для потокобезопасности
        self._lock = threading.RLock()
        self._dirty = False
        self._op_count = 0
        
        # EventBus для публикации событий миграции
        self._event_bus = None
        try:
            from eva.core.event_bus import Event, EventPriority
            self._event_types = EventTypes
            self._Event = Event
            self._EventPriority = EventPriority
        except ImportError:
            pass
        
        # Загрузка
        self._load()
        
        # Создание статичных узлов моделей (если ещё нет)
        self._ensure_model_nodes()
        
        # Graph Learning — обучение через граф опыта
        self.context_builder = None
        self.learning_loop = None
        self.snapshot_manager = None
        self._init_graph_learning()
        
        logger.info(f"UnifiedFractalMemory: {len(self.nodes)} узлов, {len(self.edges)} связей, "
                    f"моделей: {len(self.model_instances)}")
    
    def _init_graph_learning(self):
        """Инициализирует систему обучения через граф"""
        try:
            from eva.memory.graph_learning import DynamicContextBuilder, GraphLearningLoop, SnapshotManager
            
            self.coordinator = DynamicContextBuilder(self, max_experiences=5, max_concepts=3)
            self.context_builder = self.coordinator
            self.learning_loop = GraphLearningLoop(self, self.coordinator, min_quality=0.7, cluster_interval=300)
            self.snapshot_manager = SnapshotManager(self, self.coordinator)
            
            self.learning_loop.start()
            logger.info("Graph Learning инициализирован")
        except Exception as e:
            logger.warning(f"Graph Learning не инициализирован: {e}")
            self.coordinator = None
            self.context_builder = None
            self.learning_loop = None
            self.snapshot_manager = None

    def save_experience(self, query: str, response: str, model_used: str, quality_score: float) -> str:
        """Сохраняет опыт Q&A для обучения"""
        if self.learning_loop:
            return self.learning_loop.add_experience(query, response, model_used, quality_score)
        return ""

    def get_context_for_query(self, query: str) -> str:
        """Получает контекст из графа для запроса"""
        if self.coordinator:
            return self.coordinator.build_context(query)
        return ""
    
    def _publish_migration_event(self, from_tier: str, to_tier: str, node_ids: List[str]):
        """Публикует событие миграции в EventBus"""
        if self._event_bus is None:
            try:
                from eva.core.event_bus import get_event_bus
                self._event_bus = get_event_bus()
            except Exception:
                pass
        
        if self._event_bus and hasattr(self, '_Event'):
            event = self._Event(
                event_type=EventTypes.MEMORY_TIER_MIGRATED,
                source="unified_fractal_memory",
                data={
                    "from_tier": from_tier,
                    "to_tier": to_tier,
                    "node_ids": node_ids,
                    "count": len(node_ids)
                },
                timestamp=time.time(),
                priority=self._EventPriority.NORMAL
            )
            self._event_bus.publish(event)
    
    def _migrate_to_warm(self):
        """Перенос малоиспользуемых узлов из hot в warm"""
        nodes_to_migrate = []
        for node_id in self.hot_nodes:
            node = self.nodes.get(node_id)
            if node and not node.is_static and node.access_count < 3:
                nodes_to_migrate.append(node_id)
        
        migrated_count = 0
        for node_id in nodes_to_migrate[:100]:
            if node_id in self.hot_nodes:
                self.hot_nodes.discard(node_id)
                self.warm_nodes.add(node_id)
                if node_id in self.nodes:
                    self.nodes[node_id].tier = MemoryTier.WARM.value
                migrated_count += 1
        
        if migrated_count > 0:
            self._dirty = True
            self._publish_migration_event("hot", "warm", nodes_to_migrate[:migrated_count])
            logger.info(f"Мигрировано {migrated_count} узлов из hot в warm")
    
    def _migrate_to_cold(self):
        """Перенос неактивных узлов из warm в cold"""
        nodes_to_migrate = []
        current_time = time.time()
        
        for node_id in self.warm_nodes:
            node = self.nodes.get(node_id)
            if node and not node.is_static and (current_time - node.last_accessed) > 3600:
                nodes_to_migrate.append(node_id)
        
        migrated_count = 0
        for node_id in nodes_to_migrate[:50]:
            if node_id in self.warm_nodes:
                self.warm_nodes.discard(node_id)
                if node_id in self.nodes:
                    self.nodes[node_id].tier = MemoryTier.COLD.value
                migrated_count += 1
        
        if migrated_count > 0:
            self._dirty = True
            self._publish_migration_event("warm", "cold", nodes_to_migrate[:migrated_count])
            logger.info(f"Мигрировано {migrated_count} узлов из warm в cold")
    
    def _evict_hot_nodes(self, count: int = 10):
        """Вытеснение least recently used hot nodes"""
        sorted_nodes = sorted(
            [n for n in self.hot_nodes if n in self.nodes and not self.nodes[n].is_static],
            key=lambda x: self.nodes[x].last_accessed
        )
        
        evicted = []
        for node_id in sorted_nodes[:count]:
            self.hot_nodes.discard(node_id)
            self.warm_nodes.add(node_id)
            if node_id in self.nodes:
                self.nodes[node_id].tier = MemoryTier.WARM.value
            evicted.append(node_id)
        
        if evicted:
            self._dirty = True
            self._publish_migration_event("hot", "warm", evicted)
            logger.info(f"Вытеснено {len(evicted)} узлов из hot в warm (LRU)")
    
    def _check_and_migrate(self):
        """Проверяет лимиты и запускает миграцию при необходимости"""
        if len(self.hot_nodes) > self.HOT_NODE_LIMIT:
            excess = len(self.hot_nodes) - self.HOT_NODE_LIMIT
            self._evict_hot_nodes(max(10, excess // 10))
        
        if len(self.hot_nodes) > self.HOT_NODE_LIMIT * 0.9:
            self._migrate_to_warm()
        
        if len(self.warm_nodes) > self.WARM_NODE_LIMIT:
            self._migrate_to_cold()

    def _load(self):
        """Загрузка графа с SSD"""
        if os.path.exists(self.nodes_file):
            try:
                with open(self.nodes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node_data in data.values():
                        node = MemoryNode.from_dict(node_data)
                        self.nodes[node.id] = node
                        if node.tier == "hot":
                            self.hot_nodes.add(node.id)
                        elif node.tier == "warm":
                            self.warm_nodes.add(node.id)
                logger.info(f"Загружено {len(self.nodes)} узлов")
            except Exception as e:
                logger.error(f"Ошибка загрузки узлов: {e}")
        
        if os.path.exists(self.edges_file):
            try:
                with open(self.edges_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for edge_data in data.values():
                        edge = MemoryEdge(**edge_data)
                        self.edges[edge.id] = edge
                logger.info(f"Загружено {len(self.edges)} связей")
            except Exception as e:
                logger.error(f"Ошибка загрузки связей: {e}")
    
    def _save(self):
        """Сохранение графа на SSD"""
        with self._lock:
            if not self._dirty and self._op_count == 0:
                return
            
            try:
                nodes_data = {nid: node.to_dict() for nid, node in self.nodes.items()}
                with open(self.nodes_file, 'w', encoding='utf-8') as f:
                    json.dump(nodes_data, f, indent=2, ensure_ascii=False)
                
                edges_data = {eid: edge.to_dict() for eid, edge in self.edges.items()}
                with open(self.edges_file, 'w', encoding='utf-8') as f:
                    json.dump(edges_data, f, indent=2, ensure_ascii=False)
                
                self._dirty = False
                self._op_count = 0
                
                logger.debug(f"Сохранено {len(self.nodes)} узлов, {len(self.edges)} связей")
            except Exception as e:
                logger.error(f"Ошибка сохранения: {e}")
    
    def _ensure_model_nodes(self):
        """Создаёт статичные узлы моделей (если ещё не существуют)"""
        model_configs = [
            {
                'type': NodeType.MODEL_A.value,
                'name': 'Qwen 2.5 3B Instruct',
                'role': 'logic',
                'level': 0,
                'content': 'Модель A: Qwen 2.5 3B Instruct — логическое ядро. Отвечает кратко по теме вопроса.',
                'config': {
                    'temperature': 0.2,
                    'max_tokens': 256,
                    'top_p': 0.85,
                    'repeat_penalty': 1.5
                }
            },
            {
                'type': NodeType.MODEL_B.value,
                'name': 'Qwen 2.5 3B Instruct',
                'role': 'concept_expansion',
                'level': 0,
                'content': 'Модель B: Qwen 2.5 3B Instruct — развитие мысли. Проверяет, развивает, добавляет детали.',
                'config': {
                    'temperature': 0.6,
                    'max_tokens': 512,
                    'top_p': 0.85,
                    'repeat_penalty': 1.5
                }
            },
            {
                'type': NodeType.MODEL_C.value,
                'name': 'Qwen 2.5 Coder 1.5B Instruct',
                'role': 'code_generation',
                'level': 0,
                'content': 'Модель C: Qwen 2.5 Coder 1.5B Instruct — генерация кода. Пишет чистый рабочий код.',
                'config': {
                    'temperature': 0.1,
                    'max_tokens': 512,
                    'top_p': 0.9,
                    'repeat_penalty': 1.3
                }
            }
        ]
        
        for mc in model_configs:
            node_id = f"model::{mc['type']}"
            if node_id not in self.nodes:
                node = MemoryNode(
                    id=node_id,
                    node_type=mc['type'],
                    level=mc['level'],
                    content=mc['content'],
                    tier="hot",  # Модели всегда в hot
                    model_config=mc['config'],
                    is_static=True,
                    context={'name': mc['name'], 'role': mc['role']}
                )
                self.nodes[node_id] = node
                self.hot_nodes.add(node_id)
                self._dirty = True
                logger.info(f"Создан статичный узел модели: {node_id}")
        
        # Связи между моделями
        self._add_edge_if_missing("model::model_a", "model::model_b", "passes_to")
        self._add_edge_if_missing("model::model_b", "model::model_c", "can_delegate_code")
        
        self._save()
    
    def _add_edge_if_missing(self, source_id: str, target_id: str, relation: str):
        """Добавляет связь если ещё не существует"""
        edge_id = f"{source_id}__{target_id}__{relation}"
        if edge_id not in self.edges:
            self.edges[edge_id] = MemoryEdge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation
            )
            self._dirty = True
    
    def register_model_instance(self, model_type: str, llama_instance):
        """Регистрирует живой Llama instance в графе"""
        with self._lock:
            self.model_instances[model_type] = llama_instance
            
            # Обновляем узел модели
            node_id = f"model::{model_type}"
            if node_id in self.nodes:
                self.nodes[node_id].context['instance_loaded'] = True
                self.nodes[node_id].context['loaded_at'] = time.time()
                self.nodes[node_id].updated_at = time.time()
                self._dirty = True
            
            logger.info(f"Зарегистрирована модель в графе: {model_type}")
    
    def export_model_to_graph(self, model_type: str, gguf_path: str) -> Dict[str, int]:
        """Экспортирует структуру GGUF модели в фрактальный граф"""
        try:
            from eva.memory.gguf_fractal_exporter import GGUFFractalExporter
            exporter = GGUFFractalExporter(gguf_path)
            stats = exporter.export_to_fractal_memory(self, model_type)
            logger.info(f"Модель {model_type} экспортирована в граф: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Ошибка экспорта модели {model_type} в граф: {e}")
            return {'nodes_created': 0, 'edges_created': 0}
    
    def get_model_instance(self, model_type: str):
        """Получает Llama instance из графа"""
        return self.model_instances.get(model_type)
    
    def add_model_node(self, node_id: str, content: str, level: int, parent_id: str = None,
                       context: Dict[str, Any] = None, node_type: str = "model_component") -> str:
        """Добавляет узел модели с кастомным ID"""
        with self._lock:
            if node_id in self.nodes:
                # Обновляем существующий
                existing = self.nodes[node_id]
                existing.content = content
                existing.level = level
                if context:
                    existing.context.update(context)
                existing.updated_at = time.time()
                self._dirty = True
            else:
                node = MemoryNode(
                    id=node_id,
                    node_type=node_type,
                    level=level,
                    content=content,
                    tier="hot" if level <= 1 else "cold",
                    context=context or {},
                    is_static=True
                )
                self.nodes[node_id] = node
                self._dirty = True
            
            # Связь с родителем
            if parent_id and parent_id in self.nodes:
                edge_id = f"{parent_id}__{node_id}__contains"
                if edge_id not in self.edges:
                    self.edges[edge_id] = MemoryEdge(
                        id=edge_id,
                        source_id=parent_id,
                        target_id=node_id,
                        relation_type="contains"
                    )
                    self._dirty = True
            
            return node_id

    def add_knowledge(self, content: str, level: int = 2, parent_id: str = None, 
                      context: Dict[str, Any] = None) -> str:
        """Добавляет знание в фрактальный граф"""
        with self._lock:
            node_id = f"knowledge::{hashlib.sha256(content.encode()).hexdigest()[:16]}"
            
            if node_id in self.nodes:
                # Обновляем существующий
                self.nodes[node_id].content = content
                self.nodes[node_id].updated_at = time.time()
                self.nodes[node_id].access_count += 1
            else:
                node = MemoryNode(
                    id=node_id,
                    node_type=NodeType.FACT.value if level == 2 else NodeType.CONCEPT.value,
                    level=level,
                    content=content,
                    tier="cold",
                    parent_id=parent_id,
                    context=context or {}
                )
                self.nodes[node_id] = node
                
                if parent_id and parent_id in self.nodes:
                    self.nodes[parent_id].child_ids.append(node_id)
                    self._add_edge_if_missing(parent_id, node_id, "contains")
            
            self._dirty = True
            self._op_count += 1
            
            if self._op_count >= self.AUTO_SAVE_INTERVAL:
                self._save()
            
            self._check_and_migrate()
            
            return node_id
    
    def retrieve_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск знаний в графе по запросу"""
        with self._lock:
            query_lower = query.lower()
            results = []
            
            for node_id, node in self.nodes.items():
                if node.is_static:
                    continue  # Пропускаем узлы моделей
                
                # Простой поиск по совпадению ключевых слов
                content_lower = node.content.lower()
                score = 0
                words = query_lower.split()
                for word in words:
                    if len(word) > 2 and word in content_lower:
                        score += 1
                
                if score > 0:
                    node.access_count += 1
                    node.last_accessed = time.time()
                    results.append({
                        'id': node_id,
                        'content': node.content,
                        'score': score,
                        'level': node.level,
                        'type': node.node_type,
                        'context': node.context
                    })
            
            # Сортируем по релевантности
            results.sort(key=lambda x: x['score'], reverse=True)
            self._check_and_migrate()
            return results[:top_k]
    
    def get_model_context(self, model_type: str) -> Dict[str, Any]:
        """Получает контекст модели + связанные знания из графа"""
        with self._lock:
            node_id = f"model::{model_type}"
            if node_id not in self.nodes:
                return {}
            
            model_node = self.nodes[node_id]
            
            # Собираем связанные знания
            related_knowledge = []
            for edge in self.edges.values():
                if edge.source_id == node_id or edge.target_id == node_id:
                    other_id = edge.target_id if edge.source_id == node_id else edge.source_id
                    if other_id in self.nodes:
                        other = self.nodes[other_id]
                        if not other.is_static:
                            related_knowledge.append({
                                'content': other.content,
                                'relation': edge.relation_type,
                                'strength': edge.strength
                            })
            
            return {
                'model': model_node.context.get('name', ''),
                'role': model_node.context.get('role', ''),
                'config': model_node.model_config,
                'instance_loaded': model_node.context.get('instance_loaded', False),
                'related_knowledge': related_knowledge
            }
    
    def get_static_models(self) -> List[Dict[str, Any]]:
        """Возвращает все статичные модели в графе"""
        models = []
        for node_id, node in self.nodes.items():
            if node.is_static and node.node_type.startswith('model_'):
                models.append({
                    'id': node_id,
                    'type': node.node_type,
                    'name': node.context.get('name', ''),
                    'role': node.context.get('role', ''),
                    'config': node.model_config,
                    'instance_loaded': node.context.get('instance_loaded', False),
                    'tier': node.tier
                })
        return models
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика фрактальной памяти"""
        with self._lock:
            type_counts = {}
            level_counts = {}
            tier_counts = {}
            
            for node in self.nodes.values():
                type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
                level_counts[node.level] = level_counts.get(node.level, 0) + 1
                tier_counts[node.tier] = tier_counts.get(node.tier, 0) + 1
            
            return {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'hot_nodes': len(self.hot_nodes),
                'warm_nodes': len(self.warm_nodes),
                'model_instances': len(self.model_instances),
                'by_type': type_counts,
                'by_level': level_counts,
                'by_tier': tier_counts,
                'static_models': sum(1 for n in self.nodes.values() if n.is_static)
            }
    
    def flush(self):
        """Принудительное сохранение"""
        self._save()
    
    def close(self):
        """Закрытие: сохраняем граф"""
        self._save()
        logger.info("UnifiedFractalMemory закрыт, граф сохранён на SSD")
