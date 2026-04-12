"""
Интегрированный поисковый движок ЕВА
Поддерживает BaseComponent и EventBus
Оптимизирован: асинхронные запросы, connection pooling
"""

import logging
import time
import os
import re
import json
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from functools import wraps

logger = logging.getLogger("eva_ai.websearch")

from eva_ai.core.base_component import BaseComponent, ComponentState
from eva_ai.core.event_bus import get_event_bus, Event, EventTypes

try:
    from eva_ai.websearch.web_search_engine import WebSearchEngine
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный WebSearchEngine недоступен")


def load_brain_config() -> Dict:
    """Загружает конфигурацию brain"""
    # Пробуем несколько путей для поиска конфига
    possible_paths = [
        os.path.join(os.getcwd(), 'brain_config.json'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'brain_config.json'),
        'C:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\brain_config.json'
    ]
    
    for config_path in possible_paths:
        if os.path.exists(config_path):
            logger.info(f"Found brain_config.json at: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    logger.warning(f"brain_config.json not found in any of: {possible_paths}")
    return {}


def tavily_search(query: str, api_key: str = None, max_results: int = 5) -> Dict[str, Any]:
    """Выполняет поиск через Tavily API (синхронная версия)"""
    if not api_key:
        config = load_brain_config()
        api_key = config.get('tavily_api_key') or os.environ.get('TAVILY_API_KEY')
        logger.info(f"Loaded config, checking for tavily_api_key: found={bool(api_key)}")
    
    if not api_key:
        logger.error(f"Tavily API key не найден. Config keys: {config.keys() if config else 'empty'}")
        logger.warning("Tavily API key не найден")
        return {"error": "API key не найден", "results": []}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {"query": query, "max_results": max_results}
    
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Tavily API error: {response.status_code}")
            return {"error": f"API error: {response.status_code}", "results": []}
            
    except requests.exceptions.Timeout:
        logger.error("Tavily API timeout")
        return {"error": "API timeout", "results": []}
    except Exception as e:
        logger.error(f"Tavily API exception: {e}")
        return {"error": str(e), "results": []}


async def tavily_search_async(
    query: str, 
    api_key: str = None, 
    max_results: int = 5,
    session: Optional[aiohttp.ClientSession] = None
) -> Dict[str, Any]:
    """
    Асинхронный поиск через Tavily API с connection pooling.
    
    Args:
        query: Поисковый запрос
        api_key: API ключ Tavily
        max_results: Максимальное количество результатов
        session: aiohttp сессия (опционально, для reuse)
        
    Returns:
        Результаты поиска
    """
    if not api_key:
        config = load_brain_config()
        api_key = config.get('tavily_api_key') or os.environ.get('TAVILY_API_KEY')
        logger.info(f"[ASYNC] Loaded config, checking for tavily_api_key: found={bool(api_key)}")
    
    if not api_key:
        logger.error(f"[ASYNC] Tavily API key не найден")
        logger.warning("Tavily API key не найден")
        return {"error": "API key не найден", "results": []}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {"query": query, "max_results": max_results}
    
    # Создаем сессию если не передана
    close_session = False
    if session is None:
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5),
            timeout=aiohttp.ClientTimeout(total=10)
        )
        close_session = True
    
    try:
        async with session.post(
            "https://api.tavily.com/search",
            headers=headers,
            json=data
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Tavily API error: {response.status}")
                return {"error": f"API error: {response.status}", "results": []}
                
    except asyncio.TimeoutError:
        logger.error("Tavily API timeout")
        return {"error": "API timeout", "results": []}
    except Exception as e:
        logger.error(f"Tavily API exception: {e}")
        return {"error": str(e), "results": []}
    finally:
        if close_session:
            await session.close()


class AsyncWebSearchClient:
    """
    Асинхронный клиент для веб-поиска с connection pooling.
    Оптимизирован для высокой производительности.
    """
    
    def __init__(self, max_connections: int = 10, max_connections_per_host: int = 5):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Получить или создать aiohttp сессию."""
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(
                        limit=self.max_connections,
                        limit_per_host=self.max_connections_per_host,
                        enable_cleanup_closed=True,
                        force_close=False,
                    ),
                    timeout=aiohttp.ClientTimeout(total=15, connect=5),
                    headers={"User-Agent": "EVA-AI/1.0"}
                )
                logger.debug(f"AsyncWebSearchClient: создана новая сессия")
            return self._session
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Выполнить асинхронный поиск.
        
        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            api_key: API ключ Tavily
            
        Returns:
            Результаты поиска
        """
        session = await self._get_session()
        return await tavily_search_async(
            query=query,
            api_key=api_key,
            max_results=max_results,
            session=session
        )
    
    async def search_batch(
        self, 
        queries: List[str], 
        max_results: int = 5,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Выполнить batch поиск для нескольких запросов параллельно.
        
        Args:
            queries: Список поисковых запросов
            max_results: Максимальное количество результатов на запрос
            api_key: API ключ Tavily
            
        Returns:
            Список результатов поиска
        """
        session = await self._get_session()
        
        tasks = [
            tavily_search_async(
                query=q,
                api_key=api_key,
                max_results=max_results,
                session=session
            )
            for q in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обработка ошибок
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search error for query '{queries[i]}': {result}")
                processed_results.append({"error": str(result), "results": []})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def close(self):
        """Закрыть сессию и освободить ресурсы."""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
                logger.debug("AsyncWebSearchClient: сессия закрыта")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class IntegratedWebSearchEngine(BaseComponent):
    """Интегрированный поисковый движок с поддержкой событий"""
    
    _instance = None
    
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
            "errors": 0,
            "tavily_requests": 0,
            "tavily_responses": 0,
            "tavily_errors": 0,
            "active_requests": 0
        }
        
        # Кэш результатов поиска
        self.search_cache = {}
        
        # Асинхронный клиент для оптимизации производительности
        self._async_client: Optional[AsyncWebSearchClient] = None
        
        logger.info(f"IntegratedWebSearchEngine {self.name} инициализирован")
    
    @classmethod
    def get_instance(cls, event_bus=None, brain=None) -> 'IntegratedWebSearchEngine':
        """Получить экземпляр движка (singleton)"""
        if cls._instance is None:
            cls._instance = cls(event_bus=event_bus, brain=brain)
        return cls._instance
    
    async def _get_async_client(self) -> AsyncWebSearchClient:
        """Получить или создать асинхронный клиент."""
        if self._async_client is None:
            self._async_client = AsyncWebSearchClient(
                max_connections=10,
                max_connections_per_host=5
            )
        return self._async_client
    
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
            
            # Закрываем асинхронный клиент
            if self._async_client:
                try:
                    # Создаем новый event loop если нужно
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    loop.run_until_complete(self._async_client.close())
                    logger.info("Асинхронный клиент поиска закрыт")
                except Exception as e:
                    logger.warning(f"Ошибка закрытия асинхронного клиента: {e}")
            
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
            
            # Используем Tavily API с приоритетом
            self.stats["active_requests"] += 1
            self.stats["tavily_requests"] += 1
            logger.info(f"Tavily request started: query={query[:50]}...")
            
            tavily_result = tavily_search(query, max_results=max_results or 5)
            
            self.stats["active_requests"] = max(0, self.stats["active_requests"] - 1)
            
            if "error" not in tavily_result and tavily_result.get("results"):
                self.stats["tavily_responses"] += 1
                result = {
                    "status": "completed",
                    "results": tavily_result.get("results", []),
                    "source": "tavily"
                }
            else:
                if "error" in tavily_result:
                    self.stats["tavily_errors"] += 1
                    logger.error(f"Tavily search failed: {tavily_result.get('error')}")
                # БЕЗ FALLBACK - только Tavily
                result = {"status": "error", "error": tavily_result.get('error', 'Tavily failed'), "results": []}
            
            # Обновляем статистику
            self.stats["searches_performed"] += 1
            
            if result.get("status") == "completed":
                results = result.get("results", [])
                results = self._filter_results(results, query)
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
        """Веб-поиск через Tavily API с fallback"""
        max_results = search_config.get("max_results", 10) if search_config else 10
        
        # Пробуем Tavily API
        tavily_result = tavily_search(query, max_results=max_results)
        
        if "error" not in tavily_result or not tavily_result.get("results"):
            # Успешный ответ от Tavily
            results = tavily_result.get("results", [])
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", r.get("snippet", "")),
                    "relevance_score": r.get("score", 0.9),
                    "source": "tavily",
                    "timestamp": datetime.now().isoformat()
                })
            
            formatted_results = self._filter_results(formatted_results)
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "search_time": time.time(),
                "source": "tavily"
            }
        
        # Fallback: симулированные результаты при ошибке API
        logger.warning(f"Tavily API недоступен, используем fallback: {tavily_result.get('error')}")
        results = []
        for i in range(min(max_results, 5)):
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
            "source": "integrated_web_search_fallback"
        }
    
    async def search_async(
        self, 
        query: str, 
        search_config: Optional[Dict] = None, 
        max_results: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Асинхронный поиск в интернете с оптимизацией производительности.
        
        Использует connection pooling и не блокирует event loop.
        
        Args:
            query: Поисковый запрос
            search_config: Конфигурация поиска
            max_results: Максимальное количество результатов
            
        Returns:
            Результаты поиска
        """
        start_time = time.time()
        
        try:
            # Обрабатываем max_results параметр
            if max_results is not None:
                if search_config is None:
                    search_config = {}
                search_config["max_results"] = max_results
            elif search_config is None:
                search_config = {}
            
            # Проверяем кэш
            cache_key = self._generate_cache_key(query, search_config)
            if cache_key in self.search_cache:
                self.stats["cache_hits"] += 1
                cached_result = self.search_cache[cache_key].copy()
                cached_result["from_cache"] = True
                cached_result["search_time_ms"] = (time.time() - start_time) * 1000
                logger.debug(f"Cache hit для запроса: {query[:50]}...")
                return cached_result
            
            self.stats["active_requests"] += 1
            
            # Выполняем асинхронный поиск через Tavily
            max_results_val = search_config.get("max_results", 10)
            
            try:
                client = await self._get_async_client()
                tavily_result = await client.search(
                    query=query,
                    max_results=max_results_val
                )
                
                self.stats["tavily_requests"] += 1
                
                if "error" not in tavily_result:
                    self.stats["tavily_responses"] += 1
                    results = tavily_result.get("results", [])
                    
                    formatted_results = []
                    for r in results:
                        formatted_results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("content", r.get("snippet", "")),
                            "relevance_score": r.get("score", 0.9),
                            "source": "tavily",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    formatted_results = self._filter_results(formatted_results)
                    
                    result = {
                        "success": True,
                        "query": query,
                        "results": formatted_results,
                        "total_results": len(formatted_results),
                        "search_time": time.time(),
                        "search_time_ms": (time.time() - start_time) * 1000,
                        "source": "tavily_async",
                        "from_cache": False
                    }
                    
                    # Сохраняем в кэш
                    self.search_cache[cache_key] = result.copy()
                    self._save_search_cache_async()
                    
                    return result
                    
                else:
                    self.stats["tavily_errors"] += 1
                    logger.warning(f"Tavily error: {tavily_result.get('error')}")
                    # Fallback на синхронный метод
                    return self._basic_web_search(query, search_config)
                    
            except Exception as e:
                logger.error(f"Ошибка асинхронного поиска: {e}")
                # Fallback на синхронный метод
                return self._basic_web_search(query, search_config)
            
        except Exception as e:
            logger.error(f"Ошибка в search_async: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}
        finally:
            self.stats["active_requests"] -= 1
    
    async def search_batch_async(
        self, 
        queries: List[str], 
        search_config: Optional[Dict] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Параллельный поиск для нескольких запросов.
        
        Args:
            queries: Список поисковых запросов
            search_config: Конфигурация поиска
            max_results: Максимальное количество результатов на запрос
            
        Returns:
            Список результатов поиска
        """
        tasks = [
            self.search_async(q, search_config, max_results)
            for q in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обработка ошибок
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch search error for query '{queries[i]}': {result}")
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "query": queries[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _save_search_cache_async(self):
        """Асинхронное сохранение кэша (non-blocking)."""
        try:
            # Сохраняем только последние 100 результатов
            if len(self.search_cache) > 100:
                # Удаляем старые записи
                sorted_cache = sorted(
                    self.search_cache.items(),
                    key=lambda x: x[1].get("search_time", 0),
                    reverse=True
                )
                self.search_cache = dict(sorted_cache[:100])
            
            # Сохранение в файл в отдельном потоке
            import threading
            def save():
                try:
                    cache_file = os.path.join(self.cache_dir, "search_cache.json")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.debug(f"Ошибка сохранения кэша: {e}")
            
            threading.Thread(target=save, daemon=True).start()
            
        except Exception as e:
            logger.debug(f"Ошибка подготовки кэша: {e}")
    
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
        query_counts = {}
        for cache_key, result in self.search_cache.items():
            query = result.get("query", "")
            if query:
                query_counts[query] = query_counts.get(query, 0) + 1
        
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [q[0] for q in sorted_queries[:5]]
    
    def web_search_and_learn(self, concept: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Выполняет веб-поиск по концепту и преобразует выдачу в формат знаний.
        
        Args:
            concept: Концепт или запрос для поиска
            num_results: Число результатов
            
        Returns:
            List[Dict]: Список знаний в формате для обучения
        """
        try:
            response = self.search(concept, max_results=num_results)
            
            knowledge = []
            results = response.get("results", []) if isinstance(response, dict) else []
            
            for r in results[:num_results]:
                if isinstance(r, dict):
                    knowledge.append({
                        "concept": concept,
                        "content": r.get("snippet", r.get("title", "")),
                        "domain": "general",
                        "source": f"web:{r.get('source', 'tavily')}",
                        "relevance": 1.0,
                        "metadata": {
                            "url": r.get("url", ""),
                            "engine": r.get("source", "tavily"),
                            "timestamp": time.time()
                        }
                    })
                else:
                    knowledge.append({
                        "concept": concept,
                        "content": str(r),
                        "domain": "general",
                        "source": "web:tavily",
                        "relevance": 1.0,
                        "metadata": {"timestamp": time.time()}
                    })
            
            return knowledge
        except Exception as e:
            logger.error(f"Ошибка web_search_and_learn('{concept}'): {e}")
            return []
    
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
    
    def _filter_results(self, results: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """
        Filter results:
        1. Remove video platforms (YouTube, etc.)
        2. Apply semantic relevance scoring
        3. Keep only most relevant results
        """
        # Блокируемые домены (видео-платформы)
        blocked_domains = [
            'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
            'tiktok.com', 'instagram.com/reel', 'facebook.com/watch',
            'twitch.tv', 'netflix.com', 'rutube.ru', 'vk.com/video'
        ]
        
        filtered = []
        for result in results:
            url = result.get('url', '').lower()
            
            # Проверяем, не из заблокированного домена
            if any(domain in url for domain in blocked_domains):
                logger.debug(f"Filtered out video result: {url[:60]}")
                continue
            
            # Базовая фильтрация контента
            cleaned = self._filter_search_result(result)
            
            # Добавляем URL обратно для семантического анализа
            cleaned['url'] = result.get('url', '')
            
            filtered.append(cleaned)
        
        # Если есть запрос - применяем семантическое ранжирование
        if query and len(filtered) > 1:
            filtered = self._rank_by_relevance(filtered, query)
        
        # Удаляем URL перед возвратом
        for r in filtered:
            r.pop('url', None)
        
        return filtered
    
    def _rank_by_relevance(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Ранжирует результаты по семантической релевантности запросу.
        Простая реализация на основе ключевых слов + длины контента.
        """
        import re
        
        query_words = set(re.findall(r'\w+', query.lower()))
        scored_results = []
        
        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # Считаем совпадения слов
            title_matches = sum(1 for word in query_words if word in title)
            snippet_matches = sum(1 for word in query_words if word in snippet)
            
            # Оценка релевантности (0-1)
            relevance = (title_matches * 0.6 + snippet_matches * 0.4) / max(len(query_words), 1)
            
            # Бонус за длину и качество контента
            content_length = len(snippet)
            quality_bonus = min(content_length / 500, 0.2)  # Макс +0.2 за длинный контент
            
            # Штраф за короткий/бесполезный контент
            if content_length < 50:
                quality_bonus = -0.3
            
            final_score = relevance + quality_bonus
            
            result['_relevance_score'] = final_score
            scored_results.append(result)
        
        # Сортируем по релевантности
        scored_results.sort(key=lambda x: x['_relevance_score'], reverse=True)
        
        # Удаляем вспомогательные поля
        for r in scored_results:
            r.pop('_relevance_score', None)
        
        return scored_results
    
    def enrich_with_context(
        self,
        query: str,
        response: str = "",
        max_results: int = 3
    ) -> Dict[str, Any]:
        """
        Обогащает контекст для запроса и ответа через веб-поиск
        
        Args:
            query: Запрос пользователя
            response: Текущий ответ Qwen
            max_results: Максимальное количество результатов для контекста
            
        Returns:
            Dict с обогащённым контекстом
        """
        logger.info(f"Обогащение контекста для запроса: {query[:50]}...")
        
        try:
            # Формируем поисковый запрос на основе query и response
            search_query = query
            if response:
                # Добавляем ключевые слова из ответа если он короткий
                if len(response) < 200:
                    search_query = f"{query} {response[:100]}"
            
            # Выполняем поиск
            result = self.search(search_query, max_results=max_results)
            
            if not result.get('success'):
                return {
                    'success': False,
                    'context': '',
                    'enrichment_available': False
                }
            
            # Формируем контекст из результатов
            results = result.get('results', [])
            context_parts = []
            
            for r in results:
                title = r.get('title', '')
                snippet = r.get('snippet', r.get('text', ''))
                if snippet:
                    context_parts.append(f"- {title}: {snippet[:150]}")
            
            context = "\n".join(context_parts)
            
            return {
                'success': True,
                'context': context,
                'enrichment_available': len(results) > 0,
                'results_count': len(results),
                'search_query': search_query
            }
            
        except Exception as e:
            logger.warning(f"Ошибка обогащения контекста: {e}")
            return {
                'success': False,
                'context': '',
                'enrichment_available': False,
                'error': str(e)
            }
    
    def format_enrichment_prompt(self, enrichment_result: Dict[str, Any]) -> str:
        """
        Форматирует результат обогащения для промпта Qwen
        
        Args:
            enrichment_result: Результат enrich_with_context()
            
        Returns:
            str: Форматированный контекст для промпта
        """
        if not enrichment_result.get('success') or not enrichment_result.get('context'):
            return ""
        
        return """Вот результаты веб-поиска:
""" + enrichment_result.get('context', '') + """

Используя эти результаты, дай развёрнутый и информативный ответ пользователю."""


# Функция для получения экземпляра движка
def get_web_search_engine() -> Optional[IntegratedWebSearchEngine]:
    """Получить экземпляр IntegratedWebSearchEngine"""
    return IntegratedWebSearchEngine.get_instance()
