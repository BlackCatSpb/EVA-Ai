"""
UnifiedCacheBridge - объединённый кэш модели и графа знаний.

Связывает HybridTokenCache (модель) с KnowledgeGraphCore (граф знаний),
предзагружая релевантные узлы графа в быстрый кэш токенов перед генерацией.

Архитектура:
  Query -> Semantic Match -> Graph Nodes -> Token Cache (VRAM/RAM) -> Generation
"""

import os
import time
import json
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


class UnifiedCacheBridge:
    """
    Объединённый кэш-мост между моделью и графом знаний.
    
    При запросе:
    1. Семантически матчит запрос с узлами графа знаний
    2. Предзагружает релевантные узлы в быстрый токен-кэш
    3. Обогащает промпт контекстом из графа
    4. Кэширует результат генерации для повторного использования
    """
    
    def __init__(
        self,
        token_cache: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        max_graph_nodes_cache: int = 100,
        semantic_threshold: float = 0.3,
        cache_dir: str = None
    ):
        self.token_cache = token_cache  # HybridTokenCache
        self.knowledge_graph = knowledge_graph  # KnowledgeGraphCore
        
        self.max_graph_nodes_cache = max_graph_nodes_cache
        self.semantic_threshold = semantic_threshold
        
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'unified_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Кэш соответствия запросов -> узлы графа
        self._query_graph_index: Dict[str, List[str]] = {}
        
        # Кэш обогащённых промптов
        self._enriched_prompt_cache: OrderedDict[str, str] = OrderedDict()
        self._max_enriched_cache = 500
        
        # Статистика
        self.stats = {
            'graph_hits': 0,
            'graph_misses': 0,
            'enriched_prompt_hits': 0,
            'enriched_prompt_misses': 0,
            'tokens_preloaded': 0,
            'total_queries': 0,
            'generation_cache_hits': 0,
            'generation_cache_misses': 0,
        }
        
        self._lock = threading.RLock()
        
        self._load_state()
        
        logger.info(f"UnifiedCacheBridge инициализирован (dir={self.cache_dir})")
    
    # === ИНИЦИАЛИЗАЦИЯ ===
    
    def set_token_cache(self, cache: Any):
        """Установить токен-кэш модели."""
        self.token_cache = cache
        logger.info("Токен-кэш модели подключён")
    
    def set_knowledge_graph(self, graph: Any):
        """Установить граф знаний."""
        self.knowledge_graph = graph
        logger.info("Граф знаний подключён")
    
    # === СЕМАНТИЧЕСКИЙ ПОИСК В ГРАФЕ ===
    
    def find_relevant_graph_nodes(self, query: str, max_nodes: int = 10) -> List[Dict]:
        """
        Найти релевантные узлы в графе знаний по запросу.
        
        Returns:
            List[Dict]: Список узлов с метаданными и score релевантности
        """
        if not self.knowledge_graph:
            return []
        
        self.stats['total_queries'] += 1
        query_hash = self._hash(query)
        
        # Проверяем кэш запросов
        if query_hash in self._query_graph_index:
            node_ids = self._query_graph_index[query_hash]
            nodes = self._get_nodes_by_ids(node_ids)
            if nodes:
                self.stats['graph_hits'] += 1
                return nodes
        
        # Ищем в графе
        nodes = self._search_graph(query, max_nodes)
        
        if nodes:
            node_ids = [n['id'] for n in nodes]
            self._query_graph_index[query_hash] = node_ids
            self.stats['graph_hits'] += 1
        else:
            self.stats['graph_misses'] += 1
        
        return nodes
    
    def _search_graph(self, query: str, max_nodes: int) -> List[Dict]:
        """Поиск узлов в графе знаний по ключевым словам."""
        if not self.knowledge_graph:
            return []
        
        query_words = set(query.lower().split())
        scored_nodes = []
        
        # Получаем узлы из графа
        graph_nodes = getattr(self.knowledge_graph, 'nodes', {})
        
        for node_id, node in graph_nodes.items():
            # Собираем текст для сравнения
            texts = []
            if hasattr(node, 'name'):
                texts.append(node.name.lower())
            elif isinstance(node, dict):
                texts.append(node.get('name', '').lower())
                texts.append(node.get('description', '').lower())
            
            if hasattr(node, 'description'):
                texts.append(node.description.lower())
            
            node_text = ' '.join(texts)
            if not node_text.strip():
                continue
            
            node_words = set(node_text.split())
            
            # Считаем пересечение
            intersection = query_words & node_words
            if not intersection:
                continue
            
            # TF-IDF упрощённый
            score = len(intersection) / max(len(query_words), 1)
            
            # Бонус за точное совпадение имени
            if any(q in node_text for q in query_words):
                score *= 1.5
            
            if score >= self.semantic_threshold:
                node_data = {
                    'id': node_id,
                    'score': score,
                }
                if isinstance(node, dict):
                    node_data.update(node)
                else:
                    node_data['name'] = getattr(node, 'name', '')
                    node_data['description'] = getattr(node, 'description', '')
                    node_data['node_type'] = getattr(node, 'node_type', '')
                
                scored_nodes.append(node_data)
        
        # Также ищем по рёбрам
        graph_edges = getattr(self.knowledge_graph, 'edges', {})
        for edge_id, edge in graph_edges.items():
            edge_texts = []
            if isinstance(edge, dict):
                edge_texts.append(edge.get('source', '').lower())
                edge_texts.append(edge.get('target', '').lower())
                edge_texts.append(edge.get('relation', '').lower())
            else:
                edge_texts.append(getattr(edge, 'source', '').lower())
                edge_texts.append(getattr(edge, 'target', '').lower())
                edge_texts.append(getattr(edge, 'relation', '').lower())
            
            edge_text = ' '.join(edge_texts)
            edge_words = set(edge_text.split())
            intersection = query_words & edge_words
            
            if intersection:
                score = len(intersection) / max(len(query_words), 1) * 0.8
                if score >= self.semantic_threshold:
                    edge_data = {
                        'id': edge_id,
                        'score': score,
                        'is_edge': True,
                    }
                    if isinstance(edge, dict):
                        edge_data.update(edge)
                    scored_nodes.append(edge_data)
        
        # Сортируем по score
        scored_nodes.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_nodes[:max_nodes]
    
    def _get_nodes_by_ids(self, node_ids: List[str]) -> List[Dict]:
        """Получить узлы по ID из кэша."""
        if not self.knowledge_graph:
            return []
        
        nodes = []
        graph_nodes = getattr(self.knowledge_graph, 'nodes', {})
        
        for nid in node_ids:
            if nid in graph_nodes:
                node = graph_nodes[nid]
                node_data = {'id': nid}
                if isinstance(node, dict):
                    node_data.update(node)
                else:
                    node_data['name'] = getattr(node, 'name', '')
                    node_data['description'] = getattr(node, 'description', '')
                nodes.append(node_data)
        
        return nodes
    
    # === ПРЕДЗАГРУЗКА В ТОКЕН-КЭШ ===
    
    def preload_graph_context(self, query: str) -> int:
        """
        Предзагрузить релевантный контекст графа в токен-кэш.
        
        Returns:
            int: количество предзагруженных токенов
        """
        if not self.token_cache:
            return 0
        
        nodes = self.find_relevant_graph_nodes(query)
        if not nodes:
            return 0
        
        preloaded = 0
        
        for node in nodes[:self.max_graph_nodes_cache]:
            # Формируем ключ для кэша
            cache_key = f"graph_node:{node['id']}"
            
            # Проверяем, есть ли уже в кэше
            if self.token_cache.get(cache_key) is not None:
                continue
            
            # Сериализуем узел в строку
            node_text = self._node_to_text(node)
            
            # Кэшируем в токен-кэш
            try:
                self.token_cache.put(cache_key, {
                    'text': node_text,
                    'node_id': node['id'],
                    'score': node.get('score', 0),
                    'timestamp': time.time(),
                })
                preloaded += 1
            except Exception as e:
                logger.warning(f"Ошибка предзагрузки узла {node['id']}: {e}")
        
        if preloaded > 0:
            self.stats['tokens_preloaded'] += preloaded
            logger.debug(f"Предзагружено {preloaded} узлов графа в токен-кэш")
        
        return preloaded
    
    def _node_to_text(self, node: Dict) -> str:
        """Преобразовать узел графа в текстовое представление."""
        parts = []
        
        name = node.get('name', '')
        if name:
            parts.append(f"Название: {name}")
        
        description = node.get('description', '')
        if description:
            parts.append(f"Описание: {description}")
        
        node_type = node.get('node_type', '')
        if node_type:
            parts.append(f"Тип: {node_type}")
        
        domain = node.get('domain', '')
        if domain:
            parts.append(f"Домен: {domain}")
        
        return ' | '.join(parts) if parts else str(node)
    
    # === ОБОГАЩЕНИЕ ПРОМПТА ===
    
    def build_enriched_prompt(self, query: str) -> str:
        """
        Построить обогащённый промпт с контекстом из графа знаний.
        
        Returns:
            str: обогащённый промпт или оригинальный запрос
        """
        query_hash = self._hash(query)
        
        # Проверяем кэш обогащённых промптов
        if query_hash in self._enriched_prompt_cache:
            self.stats['enriched_prompt_hits'] += 1
            self._enriched_prompt_cache.move_to_end(query_hash)
            return self._enriched_prompt_cache[query_hash]
        
        self.stats['enriched_prompt_misses'] += 1
        
        # Ищем релевантные узлы
        nodes = self.find_relevant_graph_nodes(query)
        
        if not nodes:
            return query
        
        # Строим контекст
        context_parts = []
        for node in nodes[:5]:  # Максимум 5 узлов для контекста
            text = self._node_to_text(node)
            if text:
                context_parts.append(text)
        
        if not context_parts:
            return query
        
        context = '\n'.join(context_parts)
        
        enriched = f"""Контекст из базы знаний:
{context}

Вопрос пользователя: {query}

Ответь на основе контекста, если он релевантен."""
        
        # Кэшируем
        if len(self._enriched_prompt_cache) >= self._max_enriched_cache:
            self._enriched_prompt_cache.popitem(last=False)
        self._enriched_prompt_cache[query_hash] = enriched
        
        return enriched
    
    # === КЭШИРОВАНИЕ ГЕНЕРАЦИИ ===
    
    def cache_generation_result(self, query: str, response: str, metadata: Dict = None):
        """Кэшировать результат генерации."""
        if not self.token_cache:
            return
        
        cache_key = f"gen:{self._hash(query)}"
        
        try:
            self.token_cache.put(cache_key, {
                'response': response,
                'query': query,
                'metadata': metadata or {},
                'timestamp': time.time(),
            })
        except Exception as e:
            logger.warning(f"Ошибка кэширования генерации: {e}")
    
    def get_cached_generation(self, query: str) -> Optional[str]:
        """Получить кэшированный результат генерации."""
        if not self.token_cache:
            return None
        
        cache_key = f"gen:{self._hash(query)}"
        
        try:
            cached = self.token_cache.get(cache_key)
            if cached and isinstance(cached, dict):
                # Проверяем TTL (1 час)
                if time.time() - cached.get('timestamp', 0) < 3600:
                    self.stats['generation_cache_hits'] += 1
                    return cached.get('response')
        except Exception:
            pass
        
        self.stats['generation_cache_misses'] += 1
        return None
    
    # === ПОЛНЫЙ ЦИКЛ ===
    
    def prepare_for_generation(self, query: str) -> Dict[str, Any]:
        """
        Полный цикл подготовки к генерации:
        1. Проверяем кэш генерации
        2. Предзагружаем граф в токен-кэш
        3. Обогащаем промпт
        
        Returns:
            Dict с полями:
            - cached_response: если есть готовый ответ
            - prompt: промпт для генерации
            - graph_nodes: найденные узлы графа
            - preloaded_count: количество предзагруженных узлов
        """
        result = {
            'cached_response': None,
            'prompt': query,
            'graph_nodes': [],
            'preloaded_count': 0,
        }
        
        # 1. Проверяем кэш генерации
        cached = self.get_cached_generation(query)
        if cached:
            result['cached_response'] = cached
            return result
        
        # 2. Предзагружаем граф
        preloaded = self.preload_graph_context(query)
        result['preloaded_count'] = preloaded
        
        # 3. Находим узлы для метаданных
        nodes = self.find_relevant_graph_nodes(query)
        result['graph_nodes'] = nodes
        
        # 4. Обогащаем промпт
        result['prompt'] = self.build_enriched_prompt(query)
        
        return result
    
    # === СТАТИСТИКА ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику моста."""
        total_graph = self.stats['graph_hits'] + self.stats['graph_misses']
        total_enriched = self.stats['enriched_prompt_hits'] + self.stats['enriched_prompt_misses']
        total_gen = self.stats['generation_cache_hits'] + self.stats['generation_cache_misses']
        
        token_cache_stats = {}
        if self.token_cache and hasattr(self.token_cache, 'get_cache_stats'):
            try:
                token_cache_stats = self.token_cache.get_cache_stats()
            except Exception:
                pass
        
        graph_stats = {}
        if self.knowledge_graph and hasattr(self.knowledge_graph, 'stats'):
            try:
                graph_stats = {
                    'graph_total_nodes': self.knowledge_graph.stats.get('total_nodes', 0),
                    'graph_total_edges': self.knowledge_graph.stats.get('total_edges', 0),
                }
            except Exception:
                pass
        
        return {
            **self.stats,
            'graph_hit_rate': self.stats['graph_hits'] / total_graph * 100 if total_graph > 0 else 0,
            'enriched_hit_rate': self.stats['enriched_prompt_hits'] / total_enriched * 100 if total_enriched > 0 else 0,
            'generation_cache_hit_rate': self.stats['generation_cache_hits'] / total_gen * 100 if total_gen > 0 else 0,
            'enriched_prompt_cache_size': len(self._enriched_prompt_cache),
            'token_cache': token_cache_stats,
            'knowledge_graph': graph_stats,
        }
    
    # === СЕРИАЛИЗАЦИЯ ===
    
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _load_state(self):
        """Загрузить состояние."""
        state_file = os.path.join(self.cache_dir, 'unified_bridge_state.json')
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stats.update(data.get('stats', {}))
                    self._query_graph_index.update(data.get('query_index', {}))
                logger.info("Состояние UnifiedCacheBridge загружено")
            except Exception as e:
                logger.warning(f"Не загружено состояние: {e}")
    
    def save_state(self):
        """Сохранить состояние."""
        state_file = os.path.join(self.cache_dir, 'unified_bridge_state.json')
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'stats': self.stats,
                    'query_index': self._query_graph_index,
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Не сохранено состояние: {e}")
    
    def cleanup(self):
        """Очистка."""
        self.save_state()
        logger.info("UnifiedCacheBridge сохранён")


def create_unified_bridge(
    token_cache: Optional[Any] = None,
    knowledge_graph: Optional[Any] = None,
    cache_dir: str = None
) -> UnifiedCacheBridge:
    """Фабричная функция."""
    return UnifiedCacheBridge(
        token_cache=token_cache,
        knowledge_graph=knowledge_graph,
        cache_dir=cache_dir
    )
