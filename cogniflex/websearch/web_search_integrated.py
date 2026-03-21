"""
Интегрированный поисковый движок CogniFlex
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("cogniflex.websearch")

from cogniflex.core.base_component import BaseComponent, ComponentState
from cogniflex.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный поисковый движок
try:
    from cogniflex.websearch.web_search_engine import WebSearchEngine
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный WebSearchEngine недоступен")


class IntegratedWebSearchEngine(BaseComponent):
    """Интегрированный поисковый движок с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("web_search_engine", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'websearch_cache')
        
        # Инициализируем оригинальный движок если доступен
        self._original_engine = None
        if ORIGINAL_AVAILABLE:
            try:
                self._original_engine = WebSearchEngine()
                logger.info("Оригинальный WebSearchEngine инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального движка: {e}")
        
        # Статистика поиска
        self.stats = {
            "searches_performed": 0,
            "results_found": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        # Кэш результатов поиска
        self.search_cache = {}
        
        logger.info(f"IntegratedWebSearchEngine {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация поискового движка...")
            
            # Инициализируем оригинальный движок
            if self._original_engine and hasattr(self._original_engine, 'initialize'):
                self._original_engine.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Загружаем кэш поиска
            self._load_search_cache()
            
            # Публикуем событие инициализации
            self._emit_event("web_search_engine.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir,
                'cache_size': len(self.search_cache)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации поискового движка: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск поискового движка...")
            
            # Запускаем оригинальный движок
            if self._original_engine and hasattr(self._original_engine, 'start'):
                self._original_engine.start()
            
            # Публикуем событие запуска
            self._emit_event("web_search_engine.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска поискового движка: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка поискового движка...")
            
            # Останавливаем оригинальный движок
            if self._original_engine and hasattr(self._original_engine, 'stop'):
                self._original_engine.stop()
            
            # Сохраняем кэш поиска
            self._save_search_cache()
            
            # Публикуем событие остановки
            self._emit_event("web_search_engine.stopped", {
                'component': self.name,
                'stats': self.stats,
                'cache_size': len(self.search_cache)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки поискового движка: {e}")
            return False
    
    def search(self, query: str, search_config: Optional[Dict] = None, max_results: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Выполняет поиск в интернете"""
        start_time = time.time()
        
        try:
            # Обрабатываем max_results параметр для обратной совместимости
            if max_results is not None:
                if search_config is None:
                    search_config = {}
                search_config["max_results"] = max_results
            elif search_config is None:
                search_config = {}
            
            # Проверяем кэш
            cache_key = self._generate_cache_key(query, search_config)
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                self.stats["cache_hits"] += 1
                
                # Публикуем событие кэш-хита
                self._emit_event("web_search_engine.cache_hit", {
                    'query_length': len(query),
                    'cache_key': cache_key
                })
                
                return cached_result
            
            # Используем оригинальный движок если доступен
            if self._original_engine and hasattr(self._original_engine, 'search'):
                # Передаем параметры в оригинальный движок
                if max_results is not None:
                    result = self._original_engine.search(query, max_results=max_results)
                else:
                    result = self._original_engine.search(query, search_config)
            else:
                # Базовый поиск
                result = self._basic_web_search(query, search_config)
            
            # Обновляем статистику
            self.stats["searches_performed"] += 1
            
            if result.get("status") == "completed":
                results = result.get("results", [])
                results = self._filter_results(results)
                result["results"] = results
                self.stats["results_found"] += len(results)
                
                # Сохраняем в кэш
                self.search_cache[cache_key] = result
            
            # Публикуем событие поиска
            self._emit_event("web_search_engine.search_performed", {
                'query_length': len(query),
                'success': result.get("status") == "completed",
                'results_count': len(result.get("results", [])),
                'processing_time': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}
    
    def _basic_web_search(self, query: str, search_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Базовый веб-поиск (симуляция)"""
        # Симуляция результатов поиска
        max_results = search_config.get("max_results", 10) if search_config else 10
        
        # Генерируем фейковые результаты
        results = []
        for i in range(min(max_results, 5)):  # Ограничиваем для симуляции
            result = {
                "title": f"Результат поиска #{i+1} для: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"Это фрагмент текста о {query}. Здесь содержится релевантная информация...",
                "relevance_score": 0.9 - (i * 0.1),
                "source": "simulated_search",
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
        
        results = self._filter_results(results)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_time": time.time(),
            "source": "integrated_web_search"
        }
    
    def search_with_filters(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Поиск с фильтрами"""
        try:
            # Используем оригинальный движок если доступен
            if self._original_engine and hasattr(self._original_engine, 'search_with_filters'):
                result = self._original_engine.search_with_filters(query, filters)
            else:
                # Базовый поиск с фильтрами
                search_config = {"filters": filters}
                result = self._basic_web_search(query, search_config)
            
            # Применяем фильтры к результатам
            if result.get("success", False):
                filtered_results = self._apply_filters(result.get("results", []), filters)
                result["results"] = filtered_results
                result["total_results"] = len(filtered_results)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка поиска с фильтрами: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Применяет фильтры к результатам поиска"""
        filtered_results = results.copy()
        
        # Фильтрация по источнику
        if "source" in filters:
            source_filter = filters["source"]
            filtered_results = [r for r in filtered_results if r.get("source") == source_filter]
        
        # Фильтрация по релевантности
        if "min_relevance" in filters:
            min_relevance = filters["min_relevance"]
            filtered_results = [r for r in filtered_results if r.get("relevance_score", 0) >= min_relevance]
        
        # Фильтрация по домену
        if "domain" in filters:
            domain_filter = filters["domain"]
            filtered_results = [r for r in filtered_results if domain_filter in r.get("url", "")]
        
        return filtered_results
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Возвращает предложения для автодополнения поиска"""
        try:
            # Используем оригинальный движок если доступен
            if self._original_engine and hasattr(self._original_engine, 'get_search_suggestions'):
                return self._original_engine.get_search_suggestions(partial_query)
            else:
                # Базовые предложения
                suggestions = [
                    f"{partial_query} tutorial",
                    f"{partial_query} guide",
                    f"{partial_query} examples",
                    f"{partial_query} best practices",
                    f"how to {partial_query}"
                ]
                return suggestions[:5]  # Ограничиваем количество
                
        except Exception as e:
            logger.error(f"Ошибка получения предложений: {e}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику поиска"""
        stats = self.stats.copy()
        
        # Добавляем детальную статистику
        stats.update({
            "cache_size": len(self.search_cache),
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["searches_performed"]),
            "average_results_per_search": self.stats["results_found"] / max(1, self.stats["searches_performed"]),
            "most_common_queries": self._get_most_common_queries()
        })
        
        # Добавляем статистику из оригинального движка
        if self._original_engine and hasattr(self._original_engine, 'get_statistics'):
            original_stats = self._original_engine.get_statistics()
            stats.update(original_stats)
        
        return stats
    
    def _get_most_common_queries(self) -> List[str]:
        """Возвращает наиболее частые запросы"""
        # Анализируем кэш для поиска частых запросов
        query_counts = {}
        for cache_key, result in self.search_cache.items():
            query = result.get("query", "")
            if query:
                query_counts[query] = query_counts.get(query, 0) + 1
        
        # Сортируем по частоте
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [q[0] for q in sorted_queries[:5]]
    
    def _generate_cache_key(self, query: str, search_config: Optional[Dict] = None) -> str:
        """Генерирует ключ для кэша"""
        import hashlib
        key_data = query + str(search_config or {})
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_search_cache(self):
        """Загружает кэш поиска"""
        try:
            cache_file = os.path.join(self.cache_dir, 'search_cache.json')
            if os.path.exists(cache_file):
                import json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
                logger.info(f"Загружено {len(self.search_cache)} записей в кэше поиска")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша поиска: {e}")
            self.search_cache = {}
    
    def _save_search_cache(self):
        """Сохраняет кэш поиска"""
        try:
            cache_file = os.path.join(self.cache_dir, 'search_cache.json')
            import json
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено {len(self.search_cache)} записей в кэше поиска")
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша поиска: {e}")
    
    def _filter_search_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove URLs, HTML artifacts, limit length from search result"""
        filtered = result.copy()
        
        # Remove URL from result to prevent loop
        if 'url' in filtered:
            del filtered['url']
        
        # Clean snippet from HTML artifacts
        if 'snippet' in filtered:
            snippet = filtered['snippet']
            # Remove HTML tags
            snippet = re.sub(r'<[^>]+>', '', snippet)
            # Remove URLs
            snippet = re.sub(r'https?://\S+', '', snippet)
            # Remove extra whitespace
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            # Limit length
            if len(snippet) > 200:
                snippet = snippet[:200] + '...'
            filtered['snippet'] = snippet
        
        # Clean title
        if 'title' in filtered:
            title = filtered['title']
            title = re.sub(r'<[^>]+>', '', title)
            if len(title) > 100:
                title = title[:100] + '...'
            filtered['title'] = title
        
        return filtered
    
    def _filter_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter all results to remove URLs and HTML artifacts"""
        return [self._filter_search_result(r) for r in results]
