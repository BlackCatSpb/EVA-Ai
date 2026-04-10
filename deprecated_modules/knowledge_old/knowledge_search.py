"""
Модуль поиска в графе знаний для ЕВА
Содержит функции поиска и кэширования
"""
import time
import hashlib
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva_ai.knowledge_search")

# Импорты для интеграции с другими модулями
try:
    from eva_ai.memory.hybrid_token_cache import HybridTokenCache
    HybridTokenCache = HybridTokenCache
except ImportError:
    HybridTokenCache = None

try:
    from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
    UnifiedTextProcessor = UnifiedTextProcessor
except ImportError:
    UnifiedTextProcessor = None

from .knowledge_nodes import KnowledgeNode


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


class KnowledgeSearch:
    """Класс для поиска в графе знаний с поддержкой кэширования."""
    
    def __init__(self, db_path: str, hybrid_cache: Optional[HybridTokenCache] = None,
                 text_processor: Optional[UnifiedTextProcessor] = None):
        """
        Инициализирует поисковую систему.
        
        Args:
            db_path: Путь к базе данных SQLite
            hybrid_cache: Гибридный кэш токенов
            text_processor: Текстовый процессор для NLP
        """
        self.db_path = db_path
        self.hybrid_cache = hybrid_cache
        self.text_processor = text_processor
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time": 0.0
        }
    
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
        start_time = time.time()
        
        # Формируем ключ кэша
        cache_key = self._generate_cache_key(query, domains, node_types, min_strength, limit)
        
        # Проверяем гибридный кэш
        if self.hybrid_cache:
            cached_result = self.hybrid_cache.get_token(cache_key)
            if cached_result:
                logger.debug(f"Найдены кэшированные результаты поиска для '{query}'")
                self._update_statistics(start_time, True)
                return cached_result
        
        # Выполняем поиск
        results = self._search_nodes_internal(query, limit, domains, node_types, min_strength)
        
        # Сохраняем в кэш
        if self.hybrid_cache and results:
            self.hybrid_cache.add_token(cache_key, results)
        
        self._update_statistics(start_time, False)
        return results
    
    def _generate_cache_key(self, query: str, domains: Optional[List[str]], 
                           node_types: Optional[List[str]], min_strength: float, 
                           limit: int) -> str:
        """
        Генерирует ключ кэша для поискового запроса.
        
        Args:
            query: Поисковый запрос
            domains: Домены
            node_types: Типы узлов
            min_strength: Минимальная сила
            limit: Лимит результатов
            
        Returns:
            str: Ключ кэша
        """
        config = {
            "query": query,
            "domains": sorted(domains) if domains else None,
            "node_types": sorted(node_types) if node_types else None,
            "min_strength": min_strength,
            "limit": limit
        }
        config_str = json.dumps(config, sort_keys=True)
        return f"search:{hashlib.md5(config_str.encode()).hexdigest()}"
    
    def _search_nodes_internal(self, query: str, limit: int = 10, 
                              domains: Optional[List[str]] = None,
                              node_types: Optional[List[str]] = None,
                              min_strength: float = 0.0) -> List[KnowledgeNode]:
        """
        Выполняет внутренний поиск узлов.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            domains: Фильтр по доменам
            node_types: Фильтр по типам узлов
            min_strength: Минимальная сила знания
            
        Returns:
            List[KnowledgeNode]: Список найденных узлов
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Подготавливаем запрос
            query_lower = query.lower()
            params = [f"%{query_lower}%", f"%{query_lower}%", min_strength]
            
            sql = """
            SELECT id, name, description, node_type, domain, strength, timestamp, last_updated, 
                   version, meta, spatial_info, temporal_info, history, contradictions, 
                   keyword_index, concept_index
            FROM nodes
            WHERE (LOWER(name) LIKE ? OR LOWER(description) LIKE ?)
            AND strength >= ?
            """
            
            # Добавляем фильтр по доменам, если указан
            if domains and len(domains) > 0:
                placeholders = ",".join(["?" for _ in domains])
                sql += f" AND domain IN ({placeholders})"
                params.extend(domains)
            
            # Добавляем фильтр по типам узлов, если указан
            if node_types and len(node_types) > 0:
                placeholders = ",".join(["?" for _ in node_types])
                sql += f" AND node_type IN ({placeholders})"
                params.extend(node_types)
            
            # Добавляем лимит
            sql += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                node = KnowledgeNode(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    node_type=row[3],
                    domain=row[4],
                    strength=row[5],
                    timestamp=row[6],
                    meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                    version=row[8],
                    spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},
                    temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {}
                )
                node.last_updated = row[7]
                node.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                node.contradictions = safe_json_loads(row[13]) if len(row) > 13 and row[13] else []
                node.keyword_index = safe_json_loads(row[14]) if len(row) > 14 and row[14] else []
                node.concept_index = safe_json_loads(row[15]) if len(row) > 15 and row[15] else []
                
                results.append(node)
            
            logger.debug(f"Найдено {len(results)} узлов по запросу '{query}'")
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска узлов в графе знаний: {e}", exc_info=True)
            return []
    
    def search_by_concept(self, concept: str, limit: int = 5) -> List[KnowledgeNode]:
        """
        Ищет узлы по концепту с использованием NLP.
        
        Args:
            concept: Концепт для поиска
            limit: Максимальное количество результатов
            
        Returns:
            List[KnowledgeNode]: Список найденных узлов
        """
        if not self.text_processor:
            return self.search_nodes(concept, limit)
        
        try:
            # Анализируем концепт
            analysis = self.text_processor.process_text(concept)
            
            # Используем ключевые слова для поиска
            keywords = [kw["word"] for kw in analysis.keywords[:3]] if analysis.keywords else [concept]
            
            # Выполняем поиск по каждому ключевому слову
            all_results = []
            for keyword in keywords:
                results = self.search_nodes(keyword, limit=limit)
                all_results.extend(results)
            
            # Уникализируем результаты
            seen = set()
            unique_results = []
            for result in all_results:
                if result.id not in seen:
                    seen.add(result.id)
                    unique_results.append(result)
            
            # Сортируем по релевантности
            unique_results.sort(key=lambda x: x.strength, reverse=True)
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка поиска по концепту: {e}", exc_info=True)
            return self.search_nodes(concept, limit)
    
    def search_temporal(self, start_time: float, end_time: float, 
                       entity: Optional[str] = None, 
                       event_type: Optional[str] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Выполняет временной поиск в графе знаний.
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
            entity: Сущность для фильтрации
            event_type: Тип события для фильтрации
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: Результаты временного поиска
        """
        results = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            params = [start_time, end_time]
            sql = """
            SELECT id, name, description, node_type, domain, timestamp, temporal_info
            FROM nodes
            WHERE timestamp >= ? AND timestamp <= ?
            """
            
            if entity:
                sql += " AND (LOWER(name) LIKE ? OR LOWER(description) LIKE ?)"
                entity_param = f"%{entity.lower()}%"
                params.extend([entity_param, entity_param])
            
            if event_type:
                sql += " AND json_extract(temporal_info, '$.event_type') = ?"
                params.append(event_type)
            
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            
            for row in cursor.fetchall():
                temporal_info = safe_json_loads(row[6]) if row[6] else {}
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "node_type": row[3],
                    "domain": row[4],
                    "timestamp": row[5],
                    "temporal_info": temporal_info
                })
            
            conn.close()
            logger.debug(f"Найдено {len(results)} временных записей")
            
        except Exception as e:
            logger.error(f"Ошибка временного поиска: {e}", exc_info=True)
        
        return results
    
    def search_similar(self, node_id: str, limit: int = 5) -> List[KnowledgeNode]:
        """
        Ищет похожие узлы на основе эмбеддингов.
        
        Args:
            node_id: ID эталонного узла
            limit: Максимальное количество результатов
            
        Returns:
            List[KnowledgeNode]: Список похожих узлов
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получаем эталонный узел
            cursor.execute("SELECT * FROM nodes WHERE id = ?", [node_id])
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return []
            
            reference_node = KnowledgeNode(
                id=row[0],
                name=row[1],
                description=row[2],
                node_type=row[3],
                domain=row[4],
                strength=row[5],
                timestamp=row[6],
                meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                version=row[8],
                spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},
                temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {}
            )
            
            # Ищем похожие узлы по домену и ключевым словам
            cursor.execute("""
                SELECT id, name, description, node_type, domain, strength, timestamp, last_updated, 
                       version, meta, spatial_info, temporal_info, history, contradictions, 
                       keyword_index, concept_index
                FROM nodes 
                WHERE domain = ? AND id != ? 
                ORDER BY strength DESC 
                LIMIT ?
            """, [reference_node.domain, node_id, limit])
            
            results = []
            for row in cursor.fetchall():
                node = KnowledgeNode(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    node_type=row[3],
                    domain=row[4],
                    strength=row[5],
                    timestamp=row[6],
                    meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                    version=row[8],
                    spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},
                    temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {}
                )
                node.last_updated = row[7]
                node.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                node.contradictions = safe_json_loads(row[13]) if len(row) > 13 and row[13] else []
                node.keyword_index = safe_json_loads(row[14]) if len(row) > 14 and row[14] else []
                node.concept_index = safe_json_loads(row[15]) if len(row) > 15 and row[15] else []
                
                results.append(node)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска похожих узлов: {e}", exc_info=True)
            return []
    
    def _update_statistics(self, start_time: float, cache_hit: bool):
        """Обновляет статистику поиска."""
        search_time = time.time() - start_time
        self.search_stats["total_searches"] += 1
        
        if cache_hit:
            self.search_stats["cache_hits"] += 1
        
        # Обновляем среднее время поиска
        total_time = self.search_stats["avg_search_time"] * (self.search_stats["total_searches"] - 1)
        self.search_stats["avg_search_time"] = (total_time + search_time) / self.search_stats["total_searches"]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику поиска."""
        stats = self.search_stats.copy()
        if stats["total_searches"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_searches"]
        else:
            stats["cache_hit_rate"] = 0.0
        return stats
    
    def clear_cache(self):
        """Очищает кэш поиска."""
        if self.hybrid_cache:
            # Очищаем только ключи поиска
            # В зависимости от реализации кэша
            logger.info("Кэш поиска очищен")
    
    def optimize_search_indexes(self):
        """Оптимизирует индексы для поиска."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Создаем индексы для ускорения поиска
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_description ON nodes(description)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_strength ON nodes(strength)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON nodes(timestamp)")
            
            conn.commit()
            conn.close()
            
            logger.info("Индексы поиска оптимизированы")
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации индексов: {e}", exc_info=True)
