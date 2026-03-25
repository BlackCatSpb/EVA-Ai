"""Модуль веб-поиска для CogniFlex - обеспечивает поиск информации в интернете"""
import os
import json
import sqlite3
import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .search_models import SearchResult, SearchQuery
from .database_manager import DatabaseManager
from .search_engines import SearchEngines
from .cache_manager import CacheManager

logger = logging.getLogger("cogniflex.web_search")

class WebSearchEngine:
    """Модуль для выполнения веб-поиска и анализа результатов."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует поисковый движок.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            cache_dir: Путь к директории кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cogniflex_web_search_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Используем thread-local storage для SQLite соединений
        self._local = threading.local()
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "search_history.db")
        
        # Настройки поиска
        self.search_settings = {
            "max_results": 10,
            "timeout": 15.0,
            "max_concurrent_requests": 3,
            "use_cache": True,
            "cache_ttl": 86400,  # 24 часа в секундах
            "active_engines": ["google", "yandex"]
        }
        
        # Активные поисковые системы
        self.active_search_engines = {
            "google": True,
            "yandex": True,
            "bing": False,
            "wikipedia": True
        }
        
        # Кэш поисковых запросов
        self.search_cache = {}
        self._load_cache()
        
        # Очередь поисковых запросов
        self.search_queue = queue.Queue()
        self.search_tasks = {}
        
        # Фоновый процесс
        self.running = False
        self.search_thread = None
        
        # Инициализация базы данных для хранения истории поиска
        try:
            self.db = self._init_database()
            # Обновляем статистику из базы данных
            self._update_query_stats()
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            self.db = None
        
        # Статистика
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_processing_time": 0.0,
            "last_query": "",
            "last_update": time.time()
        }
        
        # Задачи для анализа проблем
        self.analysis_tasks = []
        
        logger.info("WebSearchEngine инициализирован")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает соединение с базой данных для текущего потока."""
        # Проверяем, есть ли уже соединение для этого потока
        if not hasattr(self._local, "connection"):
            # Создаем новое соединение для этого потока
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Разрешаем использование в разных потоках
            )
        
        return self._local.connection
    
    def _init_database(self) -> sqlite3.Connection:
        """Инициализирует базу данных для хранения истории поиска."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Создаем таблицу запросов
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_time REAL,
                success BOOLEAN,
                result_count INTEGER,
                cached BOOLEAN DEFAULT 0
            )
            """)
            
            # Таблица результатов
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                title TEXT,
                url TEXT NOT NULL,
                snippet TEXT,
                source TEXT,
                relevance_score REAL DEFAULT 1.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES search_queries(id)
            )
            """)
            
            # Таблица статистики
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_stats (
                id INTEGER PRIMARY KEY,
                total_queries INTEGER DEFAULT 0,
                successful_queries INTEGER DEFAULT 0,
                failed_queries INTEGER DEFAULT 0,
                avg_processing_time REAL DEFAULT 0.0,
                last_update DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Инициализируем статистику, если таблица пуста
            cursor.execute("SELECT COUNT(*) FROM search_stats")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                INSERT INTO search_stats (total_queries, successful_queries, failed_queries, avg_processing_time)
                VALUES (0, 0, 0, 0.0)
                """)
            
            conn.commit()
            return conn
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}", exc_info=True)
            raise
    
    def _update_query_stats(self):
        """Обновляет статистику запросов из базы данных."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем общую статистику
            cursor.execute("SELECT * FROM search_stats ORDER BY id DESC LIMIT 1")
            stats = cursor.fetchone()
            
            if stats:
                self.stats = {
                    "total_queries": stats[1],
                    "successful_queries": stats[2],
                    "failed_queries": stats[3],
                    "avg_processing_time": stats[4],
                    "last_update": stats[5]
                }
            
            # Получаем последний запрос
            cursor.execute("SELECT query FROM search_queries ORDER BY timestamp DESC LIMIT 1")
            last_query = cursor.fetchone()
            if last_query:
                self.stats["last_query"] = last_query[0]
                
        except Exception as e:
            logger.error(f"Ошибка обновления статистики запросов: {e}", exc_info=True)
    
    def _load_cache(self):
        """Загружает кэш поисковых запросов."""
        try:
            cache_path = os.path.join(self.cache_dir, "search_cache.json")
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
                logger.debug(f"Загружено {len(self.search_cache)} кэшированных запросов")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша поиска: {e}")
    
    def _save_cache(self):
        """Сохраняет кэш поисковых запросов."""
        try:
            cache_path = os.path.join(self.cache_dir, "search_cache.json")
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша поиска: {e}")
    
    def set_search_engines(self, use_google: bool = True, use_yandex: bool = True, use_bing: bool = True):
        """
        Устанавливает активные поисковые системы.
        
        Args:
            use_google: Использовать Google
            use_yandex: Использовать Yandex
            use_bing: Использовать Bing
        """
        # Проверяем, действительно ли состояние меняется
        new_settings = {
            "google": use_google,
            "yandex": use_yandex,
            "bing": use_bing
        }
        if self.active_search_engines != new_settings:
            self.active_search_engines = new_settings
            logger.info(f"Поисковые системы обновлены: Google={use_google}, Yandex={use_yandex}, Bing={use_bing}")
        else:
            logger.debug("Настройки поисковых систем не изменились")
    
    def get_active_search_engines(self) -> List[str]:
        """
        Возвращает список активных поисковых систем.
        
        Returns:
            List[str]: Список активных поисковых систем
        """
        return [engine for engine, active in self.active_search_engines.items() if active]
    
    def start(self):
        """Запускает фоновый процесс обработки поисковых запросов."""
        if not self.running:
            self.running = True
            self.search_thread = threading.Thread(target=self._search_worker, daemon=True)
            self.search_thread.start()
            logger.info("WebSearchEngine запущен")
    
    def stop(self):
        """Останавливает фоновый процесс."""
        if self.running:
            self.running = False
            if self.search_thread:
                self.search_thread.join(timeout=5)
            logger.info("WebSearchEngine остановлен")
    
    def _search_worker(self):
        """Фоновый процесс обработки поисковых запросов."""
        while self.running:
            try:
                # Получаем задачу из очереди с таймаутом
                task = self.search_queue.get(timeout=1)
                if task:
                    self._process_search_task(task)
                    self.search_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в фоновом процессе поиска: {e}", exc_info=True)
    
    def _process_search_task(self, task: Dict[str, Any]):
        """Обрабатывает поисковую задачу."""
        try:
            query = task["query"]
            task_id = task["task_id"]
            
            # Выполняем поиск
            results = self._perform_search(query)
            
            # Сохраняем результат
            self.search_tasks[task_id] = {
                "status": "completed",
                "results": results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки поисковой задачи: {e}", exc_info=True)
            self.search_tasks[task["task_id"]] = {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def search(self, query: str, max_results: int = None, use_cache: bool = None) -> Dict[str, Any]:
        """Выполняет поиск по запросу."""
        start_time = time.time()
        
        if max_results is None:
            max_results = self.search_settings["max_results"]
        if use_cache is None:
            use_cache = self.search_settings["use_cache"]
        
        try:
            # Проверяем кэш
            if use_cache:
                cache_manager = CacheManager(self.cache_dir, self.search_settings["cache_ttl"])
                cached_results = cache_manager.get_cached_results(query)
                if cached_results:
                    processing_time = time.time() - start_time
                    self._update_stats(query, "completed", cached_results, processing_time, cached=True)
                    return {
                        "status": "completed",
                        "query": query,
                        "results": cached_results,
                        "message": f"Найдено {len(cached_results)} результатов (из кэша)",
                        "processing_time": processing_time,
                        "cached": True
                    }
            
            # Выполняем поиск
            results = self._perform_search(query, max_results)
            processing_time = time.time() - start_time
            
            # Сохраняем в кэш
            if use_cache and results:
                cache_manager = CacheManager(self.cache_dir, self.search_settings["cache_ttl"])
                cache_manager.save_to_cache(query, results)
            
            # Обновляем статистику
            self._update_stats(query, "completed", results, processing_time)
            
            return {
                "status": "completed",
                "query": query,
                "results": results,
                "message": f"Найдено {len(results)} результатов",
                "processing_time": processing_time,
                "cached": False
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка поиска: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Обновляем статистику
            self._update_stats(query, "failed", [], processing_time)
            
            return {
                "status": "failed",
                "query": query,
                "results": [],
                "message": error_msg,
                "processing_time": processing_time,
                "cached": False
            }
    
    def _perform_search(self, query: str, max_results: int = None) -> List[SearchResult]:
        """Выполняет поиск через активные поисковые системы."""
        if max_results is None:
            max_results = self.search_settings["max_results"]
        
        all_results = []
        search_engines = SearchEngines()
        active_engines = self.get_active_search_engines()
        
        # Выполняем поиск через каждую активную поисковую систему
        with ThreadPoolExecutor(max_workers=self.search_settings["max_concurrent_requests"]) as executor:
            futures = []
            
            for engine in active_engines:
                if engine == "google":
                    future = executor.submit(search_engines.search_google, query, max_results)
                elif engine == "yandex":
                    future = executor.submit(search_engines.search_yandex, query, max_results)
                elif engine == "bing":
                    future = executor.submit(search_engines.search_bing, query, max_results)
                elif engine == "wikipedia":
                    future = executor.submit(search_engines.search_wikipedia, query, max_results)
                else:
                    continue
                
                futures.append((engine, future))
            
            # Собираем результаты
            for engine, future in futures:
                try:
                    results = future.result(timeout=self.search_settings["timeout"])
                    all_results.extend(results)
                    logger.debug(f"Получено {len(results)} результатов от {engine}")
                except Exception as e:
                    logger.error(f"Ошибка поиска через {engine}: {e}")
        
        # Сортируем по релевантности и ограничиваем количество
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_results[:max_results]
    
    def _update_stats(self, query: str, status: str, results: List[SearchResult], 
                     processing_time: float, cached: bool = False):
        """Обновляет статистику поиска."""
        try:
            self.stats["total_queries"] += 1
            if status == "completed":
                self.stats["successful_queries"] += 1
            else:
                self.stats["failed_queries"] += 1
            
            # Обновляем среднее время обработки
            total_time = self.stats["avg_processing_time"] * (self.stats["total_queries"] - 1)
            self.stats["avg_processing_time"] = (total_time + processing_time) / self.stats["total_queries"]
            
            self.stats["last_query"] = query
            self.stats["last_update"] = time.time()
            
            # Сохраняем в базу данных
            if hasattr(self, 'db') and self.db:
                db_manager = DatabaseManager(self.cache_dir)
                db_manager.save_query(query, status, results, 
                                    f"Processed {len(results)} results", processing_time)
                db_manager.update_stats(self.stats)
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику поиска."""
        return self.stats.copy()
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает недавние поисковые запросы."""
        try:
            db_manager = DatabaseManager(self.cache_dir)
            return db_manager.get_recent_queries(limit)
        except Exception as e:
            logger.error(f"Ошибка получения недавних запросов: {e}")
            return []
    
    def clear_cache(self):
        """Очищает кэш поиска."""
        try:
            cache_manager = CacheManager(self.cache_dir)
            cache_manager.clear_cache()
            logger.info("Кэш поиска очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Возвращает информацию о кэше."""
        try:
            cache_manager = CacheManager(self.cache_dir)
            return {
                "cache_size": cache_manager.get_cache_size(),
                "cache_dir": self.cache_dir,
                "cache_ttl": self.search_settings["cache_ttl"]
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о кэше: {e}")
            return {}
    
    def search_async(self, query: str, max_results: int = None) -> str:
        """Запускает асинхронный поиск и возвращает ID задачи."""
        task_id = f"search_{int(time.time() * 1000)}"
        
        task = {
            "task_id": task_id,
            "query": query,
            "max_results": max_results or self.search_settings["max_results"],
            "timestamp": time.time()
        }
        
        self.search_queue.put(task)
        self.search_tasks[task_id] = {
            "status": "pending",
            "timestamp": time.time()
        }
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Возвращает статус асинхронной задачи."""
        return self.search_tasks.get(task_id, {"status": "not_found"})
    
    def configure_settings(self, **kwargs):
        """Настраивает параметры поиска."""
        for key, value in kwargs.items():
            if key in self.search_settings:
                self.search_settings[key] = value
                logger.info(f"Настройка {key} изменена на {value}")
    
    def get_settings(self) -> Dict[str, Any]:
        """Возвращает текущие настройки поиска."""
        return self.search_settings.copy()
    
    def web_search_and_learn(self, concept: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Выполняет веб-поиск по концепту и преобразует выдачу в формат знаний,
        совместимый с модулями расширения/интеграции знаний.

        Args:
            concept: Концепт или запрос для поиска
            num_results: Число результатов, которое следует вернуть

        Returns:
            List[Dict[str, Any]]: Список знаний в формате
                {
                    "concept": str,
                    "content": str,              # краткое содержимое/сниппет
                    "domain": str,               # общий домен (без ML-определения)
                    "source": str,               # источник (web:engine)
                    "relevance": float,          # оценка релевантности
                    "metadata": {                # доп. сведения
                        "url": str,
                        "engine": str,
                        "timestamp": float
                    }
                }
        """
        try:
            # Выполняем поиск c учётом кэша
            response = self.search(concept, max_results=num_results, use_cache=self.search_settings.get("use_cache", True))

            # Унифицируем результаты
            results: List[SearchResult] = []
            if isinstance(response, dict):
                # ожидаемый путь: dict со списком SearchResult
                res = response.get("results") or []
                # элементы могут быть SearchResult или dict (при чтении из кэша)
                for item in res:
                    if isinstance(item, SearchResult):
                        results.append(item)
                    elif isinstance(item, dict):
                        try:
                            results.append(SearchResult(**item))
                        except Exception:
                            # пропускаем некорректные элементы
                            continue
            elif isinstance(response, list):
                # редко: некоторые места могли ожидать list
                for item in response:
                    if isinstance(item, SearchResult):
                        results.append(item)
                    elif isinstance(item, dict):
                        try:
                            results.append(SearchResult(**item))
                        except Exception:
                            continue

            knowledge: List[Dict[str, Any]] = []
            for r in results[:num_results]:
                # Формируем запись знания. Домен определяем как "general" здесь,
                # чтобы не тянуть зависимости от KnowledgeExpander.
                knowledge.append({
                    "concept": r.query or concept,
                    "content": r.snippet or (r.title or ""),
                    "domain": "general",
                    "source": f"web:{r.source}",
                    "relevance": float(getattr(r, "relevance_score", 1.0) or 1.0),
                    "metadata": {
                        "url": r.url,
                        "engine": r.source,
                        "timestamp": getattr(r, "timestamp", time.time())
                    }
                })

            return knowledge
        except Exception as e:
            logger.error(f"Ошибка web_search_and_learn('{concept}'): {e}", exc_info=True)
            return []
    
    def __del__(self):
        """Деструктор - останавливает фоновые процессы."""
        try:
            self.stop()
            if hasattr(self, 'db') and self.db:
                db_manager = DatabaseManager(self.cache_dir)
                db_manager.close()
        except Exception as e:
            logger.error(f"Ошибка в деструкторе WebSearchEngine: {e}")