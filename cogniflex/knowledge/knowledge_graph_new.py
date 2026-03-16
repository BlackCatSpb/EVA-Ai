"""
Основной модуль графа знаний для CogniFlex
Координирует работу всех подмодулей
"""
import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any

logger = logging.getLogger("cogniflex.knowledge_graph")

# Импорты подмодулей
from .knowledge_nodes import KnowledgeNode, KnowledgeEdge, create_node_id, create_edge_id
from .knowledge_storage import KnowledgeStorage
from .knowledge_search import KnowledgeSearch
from .knowledge_analytics import KnowledgeAnalytics
from .knowledge_visualization import KnowledgeVisualization

# Импорты для интеграции с другими модулями
try:
    from cogniflex.memory.hybrid_token_cache import HybridTokenCache
    HybridTokenCache = HybridTokenCache
except ImportError:
    HybridTokenCache = None

try:
    from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
    UnifiedTextProcessor = UnifiedTextProcessor
except ImportError:
    UnifiedTextProcessor = None

try:
    from cogniflex.mlearning.ml_unit import MLUnit
    MLUnit = MLUnit
except ImportError:
    MLUnit = None


class KnowledgeGraph:
    """
    Основной класс графа знаний - координирует работу всех подмодулей.
    Упрощенная версия для предотвращения зависаний при инициализации.
    """
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует граф знаний.
        
        Args:
            brain: Ссылка на CoreBrain
            cache_dir: Директория для кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "knowledge_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "knowledge_graph.db")
        
        # Инициализируем подмодули
        self.storage = KnowledgeStorage(self.db_path)
        self.search = KnowledgeSearch(self.db_path)
        self.analytics = KnowledgeAnalytics(brain)
        
        # Загружаем данные при инициализации
        self.nodes = {}
        self.edges = {}
        self.initialized = False
        
        # Компоненты для интеграции
        self.hybrid_cache = None
        self.text_processor = None
        self.ml_unit = None
        
        # Флаги состояния
        self.running = False
        self.background_thread = None
        
        logger.info("KnowledgeGraph инициализирован")
    
    def initialize(self) -> bool:
        """
        Инициализирует граф знаний с загрузкой данных.
        
        Returns:
            bool: Успешность инициализации
        """
        try:
            logger.info("Начало инициализации графа знаний...")
            
            # Инициализируем компоненты интеграции
            self._initialize_integration_components()
            
            # Загружаем данные из хранилища
            self._load_data()
            
            # Оптимизируем индексы поиска
            self.search.optimize_search_indexes()
            
            self.initialized = True
            self.running = True
            
            logger.info(f"Граф знаний инициализирован: {len(self.nodes)} узлов, {len(self.edges)} связей")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации графа знаний: {e}", exc_info=True)
            return False
    
    def _initialize_integration_components(self):
        """Инициализирует компоненты для интеграции с другими модулями."""
        try:
            # Инициализируем гибридный кэш
            if HybridTokenCache and self.brain:
                self.hybrid_cache = HybridTokenCache(self.brain)
                self.search.hybrid_cache = self.hybrid_cache
                logger.debug("HybridTokenCache инициализирован")
            
            # Инициализируем текстовый процессор
            if UnifiedTextProcessor:
                self.text_processor = UnifiedTextProcessor()
                self.search.text_processor = self.text_processor
                logger.debug("UnifiedTextProcessor инициализирован")
            
            # Получаем MLUnit из brain
            if self.brain and hasattr(self.brain, 'components'):
                self.ml_unit = self.brain.components.get('ml_unit')
                if self.ml_unit:
                    logger.debug("MLUnit подключен к KnowledgeGraph")
            
        except Exception as e:
            logger.warning(f"Ошибка инициализации компонентов интеграции: {e}")
    
    def _load_data(self):
        """Загружает данные из хранилища."""
        try:
            self.nodes = self.storage.load_all_nodes()
            self.edges = self.storage.load_all_edges()
            logger.debug(f"Загружено {len(self.nodes)} узлов и {len(self.edges)} связей")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            self.nodes = {}
            self.edges = {}
    
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
            meta: Метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            
        Returns:
            str: ID созданного узла
        """
        try:
            # Создаем ID узла
            node_id = create_node_id(name, domain)
            
            # Создаем узел
            node = KnowledgeNode(
                id=node_id,
                name=name,
                description=description,
                node_type=node_type,
                domain=domain,
                strength=strength,
                meta=meta,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Добавляем источник
            if source:
                node.meta['sources'] = [{
                    'source': source,
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'version': 1
                }]
            
            # Сохраняем в хранилище
            if self.storage.save_node(node):
                self.nodes[node_id] = node
                logger.debug(f"Добавлен узел: {node_id}")
                return node_id
            else:
                logger.error(f"Ошибка сохранения узла: {node_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Ошибка добавления узла: {e}", exc_info=True)
            return ""
    
    def add_edge(self, source_id: str, target_id: str, relation_type: str,
                strength: float = 0.5, meta: Optional[Dict] = None,
                spatial_info: Optional[Dict] = None, temporal_info: Optional[Dict] = None,
                user_id: Optional[str] = None, source: Optional[str] = None) -> str:
        """
        Добавляет новую связь в граф знаний.
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation_type: Тип связи
            strength: Сила связи
            meta: Метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            
        Returns:
            str: ID созданной связи
        """
        try:
            # Проверяем существование узлов
            if source_id not in self.nodes or target_id not in self.nodes:
                logger.error(f"Узлы не найдены: {source_id}, {target_id}")
                return ""
            
            # Создаем ID связи
            edge_id = create_edge_id(source_id, target_id, relation_type)
            
            # Создаем связь
            edge = KnowledgeEdge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                meta=meta,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Добавляем источник
            if source:
                edge.meta['sources'] = [{
                    'source': source,
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'version': 1
                }]
            
            # Сохраняем в хранилище
            if self.storage.save_edge(edge):
                self.edges[edge_id] = edge
                logger.debug(f"Добавлена связь: {edge_id}")
                return edge_id
            else:
                logger.error(f"Ошибка сохранения связи: {edge_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Ошибка добавления связи: {e}", exc_info=True)
            return ""
    
    def search_nodes(self, query: str, limit: int = 10, 
                     domains: Optional[List[str]] = None,
                     node_types: Optional[List[str]] = None,
                     min_strength: float = 0.0) -> List[KnowledgeNode]:
        """
        Ищет узлы в графе знаний.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            domains: Фильтр по доменам
            node_types: Фильтр по типам узлов
            min_strength: Минимальная сила знания
            
        Returns:
            List[KnowledgeNode]: Список найденных узлов
        """
        return self.search.search_nodes(query, limit, domains, node_types, min_strength)
    
    def search_by_concept(self, concept: str, limit: int = 5) -> List[KnowledgeNode]:
        """
        Ищет узлы по концепту с использованием NLP.
        
        Args:
            concept: Концепт для поиска
            limit: Максимальное количество результатов
            
        Returns:
            List[KnowledgeNode]: Список найденных узлов
        """
        return self.search.search_by_concept(concept, limit)
    
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
            direction: Направление (source, target, both)
            
        Returns:
            List[KnowledgeEdge]: Список связей
        """
        edges = []
        
        for edge in self.edges.values():
            if direction == "source" and edge.source_id == node_id:
                edges.append(edge)
            elif direction == "target" and edge.target_id == node_id:
                edges.append(edge)
            elif direction == "both" and (edge.source_id == node_id or edge.target_id == node_id):
                edges.append(edge)
        
        return edges
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику графа знаний.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        storage_stats = self.storage.get_statistics()
        search_stats = self.search.get_search_statistics()
        
        # Добавляем статистику по доменам
        domains = {}
        for node in self.nodes.values():
            domains[node.domain] = domains.get(node.domain, 0) + 1
        
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "domains": domains,
            "storage": storage_stats,
            "search": search_stats,
            "initialized": self.initialized,
            "running": self.running
        }
    
    def export_graph(self, format: str = "json") -> Any:
        """
        Экспортирует граф знаний.
        
        Args:
            format: Формат экспорта
            
        Returns:
            Any: Экспортированные данные
        """
        visualization = KnowledgeVisualization(self.nodes, self.edges)
        return visualization.export_graph(format)
    
    def visualize_concept(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Визуализирует концепт и связанные с ним знания.
        
        Args:
            concept: Концепт для визуализации
            depth: Глубина графа
            
        Returns:
            Dict[str, Any]: Данные для визуализации
        """
        visualization = KnowledgeVisualization(self.nodes, self.edges)
        return visualization.generate_knowledge_graph(concept, depth)
    
    def analyze_contradictions(self) -> List[Dict[str, Any]]:
        """
        Анализирует противоречия в графе знаний.
        
        Returns:
            List[Dict[str, Any]]: Список противоречий
        """
        return self.analytics.detect_contradictions(self.nodes, self.edges)
    
    def get_domain_statistics(self, domain: Optional[str] = None) -> Any:
        """
        Возвращает статистику по домену.
        
        Args:
            domain: Домен знаний (опционально)
            
        Returns:
            Dict[str, Any] или List[Dict]: Статистика домена
        """
        if domain:
            return self.analytics.get_domain_statistics(self.nodes, domain)
        else:
            # Возвращаем статистику по всем доменам для GUI
            domains = {}
            for node in self.nodes.values():
                domains[node.domain] = domains.get(node.domain, 0) + 1
            
            # Формируем полную статистику по доменам
            domain_stats = {}
            for domain_name, count in domains.items():
                domain_stats[domain_name] = self.analytics.get_domain_statistics(self.nodes, domain_name)
            
            return domain_stats
    
    def get_all_nodes(self) -> List[KnowledgeNode]:
        """
        Возвращает все узлы графа знаний.
        
        Returns:
            List[KnowledgeNode]: Список всех узлов
        """
        return list(self.nodes.values())
    
    def get_all_edges(self) -> List[KnowledgeEdge]:
        """
        Возвращает все связи графа знаний.
        
        Returns:
            List[KnowledgeEdge]: Список всех связей
        """
        return list(self.edges.values())
    
    def cleanup(self):
        """Очищает ресурсы графа знаний."""
        try:
            self.running = False
            
            if self.background_thread and self.background_thread.is_alive():
                self.background_thread.join(timeout=5)
            
            # Очищаем кэш
            if self.search:
                self.search.clear_cache()
            
            if logger:
                logger.info("Граф знаний очищен")
            
        except Exception as e:
            if logger:
                logger.error(f"Ошибка очистки графа знаний: {e}", exc_info=True)
    
    def __del__(self):
        """Деструктор."""
        self.cleanup()
