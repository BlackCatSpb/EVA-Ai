"""
Интегрированный граф знаний для ЕВА с событийной поддержкой

Объединяет функциональность knowledge_graph.py и knowledge_graph_new.py
с интеграцией в EventBus и BaseComponent.
"""
import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime

from ..core.base_component import BaseComponent, ComponentState
from ..core.event_bus import EventBus, Event, EventTypes

# Импорты подмодулей
from .knowledge_nodes import KnowledgeNode, KnowledgeEdge, create_node_id, create_edge_id
from .knowledge_storage import KnowledgeStorage
from .knowledge_search import KnowledgeSearch
from .knowledge_analytics import KnowledgeAnalytics
from .knowledge_visualization import KnowledgeVisualization

# Импорты для интеграции с другими модулями
try:
    from eva.memory.hybrid_token_cache import HybridTokenCache
    HybridTokenCache = HybridTokenCache
except ImportError:
    HybridTokenCache = None

try:
    from eva.mlearning.unified_text_processor import UnifiedTextProcessor
    UnifiedTextProcessor = UnifiedTextProcessor
except ImportError:
    UnifiedTextProcessor = None

try:
    from eva.mlearning.ml_unit import MLUnit
    MLUnit = MLUnit
except ImportError:
    MLUnit = None

logger = logging.getLogger("eva.knowledge_graph")

@dataclass
class KnowledgeGraphStats:
    """Статистика графа знаний"""
    total_nodes: int = 0
    total_edges: int = 0
    node_creations: int = 0
    node_updates: int = 0
    edge_creations: int = 0
    edge_updates: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_processing_time: float = 0.0
    last_update: float = 0.0

class IntegratedKnowledgeGraph(BaseComponent):
    """
    Интегрированный граф знаний с событийной поддержкой.
    
    Объединяет функциональность всех подмодулей графа знаний с
    интеграцией в событийную систему ЕВА.
    """
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None, 
                 event_bus: Optional[EventBus] = None, name: str = "knowledge_graph"):
        """
        Инициализирует интегрированный граф знаний.
        
        Args:
            brain: Ссылка на CoreBrain
            cache_dir: Директория для кэша
            event_bus: Шина событий
            name: Имя компонента
        """
        super().__init__(brain=brain, event_bus=event_bus, name=name)
        
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "eva_knowledge_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "knowledge_graph.db")
        
        # Структуры данных
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        
        # Статистика
        self.stats = KnowledgeGraphStats()
        
        # Инициализируем подмодули
        self._init_submodules()
        
        # Компоненты для интеграции
        self.hybrid_cache = None
        self.text_processor = None
        self.ml_unit = None
        
        # Состояние загрузки
        self._data_loaded = False
        
        logger.info(f"IntegratedKnowledgeGraph {self.name} инициализирован")
    
    def _init_submodules(self):
        """Инициализирует подмодули графа знаний"""
        try:
            self.storage = KnowledgeStorage(self.db_path)
            self.search = KnowledgeSearch(self.db_path)
            self.analytics = KnowledgeAnalytics(self.brain)
            # Visualization инициализируется позже с данными
            self.visualization = None
            logger.debug("Подмодули графа знаний инициализированы")
        except Exception as e:
            logger.error(f"Ошибка инициализации подмодулей: {e}")
            raise
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация графа знаний...")
            
            # Инициализируем компоненты интеграции
            self._initialize_integration_components()
            
            # Загружаем данные из хранилища
            self._load_data()
            
            # Инициализируем визуализацию с загруженными данными
            self.visualization = KnowledgeVisualization(self.nodes, self.edges)
            
            # Оптимизируем индексы поиска
            if hasattr(self.search, 'optimize_search_indexes'):
                self.search.optimize_search_indexes()
            
            self._data_loaded = True
            self.initialized = True
            
            # Обновляем статистику
            self._update_stats()
            
            # Публикуем событие инициализации
            self._emit_event("knowledge_graph.initialized", {
                'nodes_count': len(self.nodes),
                'edges_count': len(self.edges),
                'cache_dir': self.cache_dir
            })
            
            logger.info(f"Граф знаний инициализирован: {len(self.nodes)} узлов, {len(self.edges)} связей")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации графа знаний: {e}", exc_info=True)
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск графа знаний...")
            
            # Запускаем фоновые процессы
            self._start_background_processes()
            
            # Публикуем событие запуска
            self._emit_event("knowledge_graph.started", {
                'nodes_count': len(self.nodes),
                'edges_count': len(self.edges)
            })
            
            logger.info("Граф знаний запущен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска графа знаний: {e}", exc_info=True)
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка графа знаний...")
            
            # Останавливаем фоновые процессы
            self._stop_background_processes()
            
            # Сохраняем данные
            self._save_data()
            
            # Публикуем событие остановки
            self._emit_event("knowledge_graph.stopped", {
                'final_nodes_count': len(self.nodes),
                'final_edges_count': len(self.edges)
            })
            
            logger.info("Граф знаний остановлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки графа знаний: {e}", exc_info=True)
            return False
    
    def _initialize_integration_components(self):
        """Инициализирует компоненты для интеграции"""
        # HybridCache
        if HybridTokenCache and self.brain:
            try:
                if hasattr(self.brain, 'hybrid_cache') and self.brain.hybrid_cache:
                    self.hybrid_cache = self.brain.hybrid_cache
                    logger.debug("HybridCache подключен из brain")
            except Exception as e:
                logger.warning(f"Ошибка подключения HybridCache: {e}")
        
        # TextProcessor
        if UnifiedTextProcessor and self.brain:
            try:
                if hasattr(self.brain, 'text_processor') and self.brain.text_processor:
                    self.text_processor = self.brain.text_processor
                    logger.debug("TextProcessor подключен из brain")
            except Exception as e:
                logger.warning(f"Ошибка подключения TextProcessor: {e}")
        
        # MLUnit
        if MLUnit and self.brain:
            try:
                if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
                    self.ml_unit = self.brain.ml_unit
                    logger.debug("MLUnit подключен из brain")
            except Exception as e:
                logger.warning(f"Ошибка подключения MLUnit: {e}")
    
    def _load_data(self):
        """Загружает данные из хранилища"""
        try:
            # Загружаем узлы
            self.nodes = self.storage.load_nodes()
            
            # Загружаем связи
            self.edges = self.storage.load_edges()
            
            logger.info(f"Загружено: {len(self.nodes)} узлов, {len(self.edges)} связей")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}", exc_info=True)
            # Начинаем с пустыми структурами
            self.nodes = {}
            self.edges = {}
    
    def _save_data(self):
        """Сохраняет данные в хранилище"""
        try:
            # Сохраняем узлы
            nodes_success = self.storage.save_nodes(self.nodes)
            
            # Сохраняем связи
            edges_success = self.storage.save_edges(self.edges)
            
            logger.debug(f"Сохранено: {len(self.nodes)} узлов, {len(self.edges)} связей")
            
            return nodes_success and edges_success
            
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}", exc_info=True)
            return False
    
    def _start_background_processes(self):
        """Запускает фоновые процессы"""
        # Здесь можно запустить фоновые задачи:
        # - очистка кэша
        # - оптимизация индексов
        # - обновление статистики
        pass
    
    def _stop_background_processes(self):
        """Останавливает фоновые процессы"""
        # Останавливаем фоновые задачи
        pass
    
    def _update_stats(self):
        """Обновляет статистику"""
        self.stats.total_nodes = len(self.nodes)
        self.stats.total_edges = len(self.edges)
        self.stats.last_update = time.time()
    
    # === Основные методы работы с графом знаний ===
    
    def add_node(self, name: str, description: str, node_type: str = "fact", 
                domain: str = "general", strength: float = 0.5,
                meta: Optional[Dict] = None, spatial_info: Optional[Dict] = None,
                temporal_info: Optional[Dict] = None, user_id: Optional[str] = None,
                source: Optional[str] = None) -> str:
        """
        Добавляет новый узел в граф знаний.
        
        Args:
            name: Название узла
            description: Описание узла
            node_type: Тип узла
            domain: Домен знаний
            strength: Сила знания
            meta: Дополнительные метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            
        Returns:
            str: ID добавленного узла
        """
        start_time = time.time()
        
        try:
            # Создаем узел
            node_id = create_node_id(name)
            node = KnowledgeNode(
                id=node_id,
                name=name,
                description=description,
                node_type=node_type,
                domain=domain,
                strength=strength,
                meta=meta or {},
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Добавляем user_id и source в метаданные
            if user_id:
                if node.meta is None:
                    node.meta = {}
                node.meta['user_id'] = user_id
            if source:
                if node.meta is None:
                    node.meta = {}
                node.meta['source'] = source
            
            # Добавляем в память
            self.nodes[node_id] = node
            
            # Сохраняем в хранилище
            self.storage.save_node(node)
            
            # Обновляем статистику
            self.stats.node_creations += 1
            self._update_stats()
            
            # Публикуем событие
            self._emit_event("knowledge_graph.node_added", {
                'node_id': node_id,
                'name': name,
                'node_type': node_type,
                'domain': domain,
                'user_id': user_id
            })
            
            processing_time = time.time() - start_time
            logger.debug(f"Узел добавлен: {node_id} за {processing_time:.3f}s")
            
            return node_id
            
        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(f"Ошибка добавления узла: {e}", exc_info=True)
            return ""
    
    def add_edge(self, source_id: str, target_id: str, relation_type: str,
                strength: float = 0.5, meta: Optional[Dict] = None,
                spatial_info: Optional[Dict] = None, temporal_info: Optional[Dict] = None,
                user_id: Optional[str] = None, source: Optional[str] = None) -> str:
        """
        Добавляет связь между узлами в графе знаний.
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation_type: Тип связи
            strength: Сила связи
            meta: Дополнительные метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            
        Returns:
            str: ID добавленной связи
        """
        start_time = time.time()
        
        try:
            # Проверяем существование узлов
            if source_id not in self.nodes or target_id not in self.nodes:
                logger.warning(f"Один из узлов не найден: {source_id} -> {target_id}")
                return ""
            
            # Создаем связь
            edge_id = create_edge_id(source_id, target_id, relation_type)
            edge = KnowledgeEdge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                meta=meta or {},
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Добавляем user_id и source в метаданные
            if user_id:
                if edge.meta is None:
                    edge.meta = {}
                edge.meta['user_id'] = user_id
            if source:
                if edge.meta is None:
                    edge.meta = {}
                edge.meta['source'] = source
            
            # Добавляем в память
            self.edges[edge_id] = edge
            
            # Сохраняем в хранилище
            self.storage.save_edge(edge)
            
            # Обновляем статистику
            self.stats.edge_creations += 1
            self._update_stats()
            
            # Публикуем событие
            self._emit_event("knowledge_graph.edge_added", {
                'edge_id': edge_id,
                'source_id': source_id,
                'target_id': target_id,
                'relation_type': relation_type,
                'user_id': user_id
            })
            
            processing_time = time.time() - start_time
            logger.debug(f"Связь добавлена: {edge_id} за {processing_time:.3f}s")
            
            return edge_id
            
        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(f"Ошибка добавления связи: {e}", exc_info=True)
            return ""
    
    def search_nodes(self, query: str, limit: int = 10, 
                     domains: Optional[List[str]] = None,
                     node_types: Optional[List[str]] = None,
                     min_strength: float = 0.0) -> List[KnowledgeNode]:
        """
        Ищет узлы по запросу.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            domains: Фильтр по доменам
            node_types: Фильтр по типам узлов
            min_strength: Минимальная сила знания
            
        Returns:
            List[KnowledgeNode]: Найденные узлы
        """
        start_time = time.time()
        self.stats.total_queries += 1
        
        try:
            # Используем поисковый модуль
            results = self.search.search_nodes(query, limit, domains, node_types, min_strength)
            
            # Обновляем статистику
            self.stats.successful_queries += 1
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            
            # Публикуем событие
            self._emit_event("knowledge_graph.search_performed", {
                'query': query,
                'results_count': len(results),
                'processing_time': processing_time
            })
            
            logger.debug(f"Поиск выполнен: '{query}' -> {len(results)} результатов за {processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(f"Ошибка поиска: {e}", exc_info=True)
            return []
    
    def search_by_concept(self, concept: str, limit: int = 5) -> List[KnowledgeNode]:
        """
        Ищет узлы по концепту с использованием NLP.
        
        Args:
            concept: Концепт для поиска
            limit: Максимальное количество результатов
            
        Returns:
            List[KnowledgeNode]: Найденные узлы
        """
        try:
            if hasattr(self.search, 'search_by_concept'):
                return self.search.search_by_concept(concept, limit)
            else:
                # Запасной вариант - обычный поиск
                return self.search_nodes(concept, limit)
        except Exception as e:
            logger.error(f"Ошибка поиска по концепту: {e}", exc_info=True)
            return []
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Получает узел по ID.
        
        Args:
            node_id: ID узла
            
        Returns:
            Optional[KnowledgeNode]: Узел или None
        """
        return self.nodes.get(node_id)
    
    def get_edges(self, node_id: str, direction: str = "both") -> List[KnowledgeEdge]:
        """
        Получает связи узла.
        
        Args:
            node_id: ID узла
            direction: Направление (in/out/both)
            
        Returns:
            List[KnowledgeEdge]: Связи узла
        """
        edges = []
        
        if direction in ["out", "both"]:
            edges.extend([edge for edge in self.edges.values() if edge.source_id == node_id])
        
        if direction in ["in", "both"]:
            edges.extend([edge for edge in self.edges.values() if edge.target_id == node_id])
        
        return edges
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику графа знаний.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        self._update_stats()
        
        return {
            'total_nodes': self.stats.total_nodes,
            'total_edges': self.stats.total_edges,
            'node_creations': self.stats.node_creations,
            'node_updates': self.stats.node_updates,
            'edge_creations': self.stats.edge_creations,
            'edge_updates': self.stats.edge_updates,
            'total_queries': self.stats.total_queries,
            'successful_queries': self.stats.successful_queries,
            'failed_queries': self.stats.failed_queries,
            'total_processing_time': self.stats.total_processing_time,
            'average_processing_time': (
                self.stats.total_processing_time / max(self.stats.total_queries, 1)
            ),
            'last_update': self.stats.last_update,
            'cache_dir': self.cache_dir,
            'initialized': self.initialized,
            'data_loaded': self._data_loaded
        }
    
    def get_all_nodes(self) -> List[KnowledgeNode]:
        """Возвращает все узлы графа знаний."""
        return list(self.nodes.values())
    
    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """
        Возвращает все концепты в формате для MemoryGraphML.
        
        Returns:
            List[Dict]: Список концептов с id, type, description
        """
        concepts = []
        for node in self.nodes.values():
            concepts.append({
                'id': node.id,
                'type': node.node_type,
                'description': node.description or node.meta.get('description', ''),
                'domain': node.domain,
                'properties': node.meta
            })
        
        for edge in self.edges.values():
            concepts.append({
                'id': edge.id,
                'type': 'relation',
                'description': f"{edge.source_id} -> {edge.target_id}: {edge.relation_type}",
                'domain': 'general',
                'properties': edge.meta
            })
        
        return concepts
    
    def get_all_edges(self) -> List[KnowledgeEdge]:
        """Возвращает все связи графа знаний."""
        return list(self.edges.values())
    
    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия в графе знаний.
        
        Returns:
            List[Dict[str, Any]]: Найденные противоречия
        """
        try:
            contradictions = self.analytics.detect_contradictions(self.nodes, self.edges)
            
            # Публикуем событие
            self._emit_event("knowledge_graph.contradictions_detected", {
                'contradictions_count': len(contradictions)
            })
            
            return contradictions
        except Exception as e:
            logger.error(f"Ошибка обнаружения противоречий: {e}", exc_info=True)
            return []
    
    def get_domain_statistics(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Возвращает статистику по домену.
        
        Args:
            domain: Домен (None для всех доменов)
            
        Returns:
            Dict[str, Any]: Статистика по домену
        """
        try:
            if domain:
                # Фильтруем узлы по домену
                domain_nodes = [node for node in self.nodes.values() if node.domain == domain]
                domain_edges = []
                
                for edge in self.edges.values():
                    source_node = self.nodes.get(edge.source_id)
                    target_node = self.nodes.get(edge.target_id)
                    if source_node and source_node.domain == domain:
                        domain_edges.append(edge)
                    elif target_node and target_node.domain == domain:
                        domain_edges.append(edge)
                
                return {
                    'domain': domain,
                    'nodes_count': len(domain_nodes),
                    'edges_count': len(domain_edges),
                    'average_strength': sum(n.strength for n in domain_nodes) / max(len(domain_nodes), 1)
                }
            else:
                # Статистика по всем доменам
                domain_stats = {}
                for node in self.nodes.values():
                    if node.domain not in domain_stats:
                        domain_stats[node.domain] = {'nodes': 0, 'total_strength': 0.0}
                    domain_stats[node.domain]['nodes'] += 1
                    domain_stats[node.domain]['total_strength'] += node.strength
                
                # Преобразуем в итоговый формат
                result = {}
                for domain, stats in domain_stats.items():
                    result[domain] = {
                        'nodes_count': stats['nodes'],
                        'average_strength': stats['total_strength'] / stats['nodes']
                    }
                
                return result
                
        except Exception as e:
            logger.error(f"Ошибка получения статистики по домену: {e}", exc_info=True)
            return {}
    
    def cleanup(self):
        """Очищает ресурсы графа знаний"""
        try:
            logger.info("Очистка ресурсов графа знаний...")
            
            # Сохраняем данные
            if self._data_loaded:
                self._save_data()
            
            # Очищаем структуры
            self.nodes.clear()
            self.edges.clear()
            
            # Сбрасываем статистику
            self.stats = KnowledgeGraphStats()
            
            logger.info("Ресурсы графа знаний очищены")
            
        except Exception as e:
            logger.error(f"Ошибка очистки ресурсов: {e}", exc_info=True)
    
    def __str__(self) -> str:
        return f"IntegratedKnowledgeGraph(name='{self.name}', nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    def __repr__(self) -> str:
        return self.__str__()
