"""
Модуль менеджера моделей для CogniFlex
Управляет загрузкой, кэшированием и использованием моделей машинного обучения
"""

import os
import logging
import time
import asyncio
import sqlite3
import json
import threading
import re
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from cogniflex.mlearning.async_text_generator import AsyncTextGenerator
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("cogniflex.model_manager")

# Импорты для интеграции с другими модулями
try:
    from cogniflex.mlearning.ml_core import ModelHealth
    logger.debug("ModelHealth импортирован из cogniflex.mlearning.ml_core")
except ImportError:
    logger.warning("ModelHealth недоступен, используем локальную реализацию")
    
    @dataclass
    class ModelHealth:
        """Представляет состояние модели машинного обучения."""
        model_name: str
        model_type: str = "transformer"
        status: str = "healthy"  # healthy, warning, critical
        health_score: float = 0.9  # 0.0-1.0
        usage_count: int = 0
        error_count: int = 0
        response_time: float = 0.0
        last_used: float = field(default_factory=time.time)
        memory_usage: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetadata:
    """Метаданные модели машинного обучения."""
    id: str
    name: str
    model_path: str
    model_type: str
    priority: int
    tags: List[str]
    description: str = ""
    domain: str = "general"
    strength: float = 0.7
    timestamp: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

class ModelStatus(Enum):
    """Статус модели машинного обучения."""
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADED = "unloaded"
    PENDING = "pending"

@dataclass
class ModelInstance:
    """Экземпляр загруженной модели."""
    model: Any
    tokenizer: Any
    metadata: ModelMetadata
    status: ModelStatus = ModelStatus.PENDING
    health: ModelHealth = field(init=False)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        self.health = ModelHealth(
            model_name=self.metadata.name,
            model_type=self.metadata.model_type
        )

class ModelManager:
    """Менеджер моделей для CogniFlex - управляет загрузкой и использованием моделей ML."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None, 
                 model_dir: Optional[str] = None, use_gpu: bool = True,
                 max_workers: int = 4, hybrid_cache_size: int = 50000,
                 autoload: bool = True):
        """
        Инициализирует менеджер моделей.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            cache_dir: Путь к директории кэша
            model_dir: Путь к директории моделей
            use_gpu: Использовать GPU если доступен
            max_workers: Максимальное количество рабочих потоков
            hybrid_cache_size: Размер гибридного кэша
            autoload: Запускать ли фоновые службы и автозагрузку моделей
        """
        self.brain = brain
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.hybrid_cache_size = hybrid_cache_size
        self.autoload = autoload
        
        # Подписываемся на событие готовности text_processor
        if brain and hasattr(brain, 'events'):
            brain.events.subscribe('text_processor_ready', self._on_text_processor_ready)
        
        # Определяем директории
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = model_dir or os.path.join(base_dir, "cogniflex_models")
        self.cache_dir = cache_dir or os.path.join(base_dir, "core", "cogniflex_cache", "models")
        
        # Создаем директории если не существуют
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Директория моделей установлена: {self.model_dir}")
        logger.info(f"Директория кэша установлена: {self.cache_dir}")
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "models.db")
        
        # Инициализируем базу данных
        self._init_db()
        
        # Словари для хранения моделей
        self.models = {}  # Активные модели
        self.model_metadata = {}  # Метаданные всех моделей
        self.model_futures = {}  # Асинхронные задачи загрузки
        
        # Блокировки для многопоточного доступа
        self.model_lock = threading.RLock()
        self.db_lock = threading.RLock()
        
        # Пул потоков для асинхронной загрузки
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Инициализируем компоненты
        self._init_components()
        
        # Загружаем данные из БД
        self._load_data()
        # Миграция: приводим устаревшие записи к актуальным стандартным моделям
        try:
            self._migrate_default_models()
        except Exception as e:
            logger.warning(f"Миграция стандартных моделей пропущена: {e}")
        
        # Если нет моделей, добавляем базовые
        if not self.model_metadata:
            self._add_default_models()
        
        # Сканируем директорию моделей на старте, чтобы обнаружить локальные модели
        try:
            discovered = self.scan_models_directory()
            if discovered:
                logger.info(f"На старте обнаружено и зарегистрировано локальных моделей: {discovered}")
        except Exception as e:
            logger.error(f"Не удалось просканировать директорию моделей при инициализации: {e}", exc_info=True)
        
        # Состояние
        self.initialized = True
        self.running = False
        self.stop_event = threading.Event()
        
        # Запускаем фоновые службы (опционально)
        if self.autoload:
            self._start_background_services()
        
        logger.info(f"ModelManager инициализирован с {len(self.model_metadata)} моделями и {self.max_workers} рабочими потоками")

    def _migrate_default_models(self):
        """
        Мигрирует/выравнивает стандартную модель 'default_text_gen' на QWEN по умолчанию.
        - Если в БД есть запись с id='default_text_gen' и она указывает на gpt2/ruGPT3/иное не-QWEN,
          обновляем на Qwen/Qwen2.5-7B-Instruct, тип 'qwen', высокий приоритет и теги.
        - Если записи нет, создаём alias 'default_text_gen' на QWEN.
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT id, model_path, model_type, name, priority, tags FROM models WHERE id = ?", ("default_text_gen",))
                row = cursor.fetchone()
                if row:
                    _id, _path, _type, _name, _priority, _tags = row
                    # Нормализуем путь для проверки
                    norm_path = _path.strip()
                    norm_lower = norm_path.lower()
                    should_update = False
                    # Любая не-QWEN цель требует миграции
                    if norm_lower in ("gpt2", "distilgpt2", "gpt2-medium"):
                        should_update = True
                    # Устаревшие/вариативные ruGPT3
                    if norm_lower in ("sberbank-ai/rugpt3small", "sberbank-ai/rugpt3small_based_on_gpt2",
                                      "ai-forever/rugpt3small_based_on_gpt2", "ai-forever/rugpt3medium_based_on_gpt2",
                                      "ai-forever/rugpt3large_based_on_gpt2"):
                        should_update = True
                    # Если тип/путь уже указывает на qwen — не обновляем
                    if "qwen" in norm_lower:
                        should_update = False
                    if should_update:
                        new_path = "Qwen/Qwen2.5-7B-Instruct"
                        new_type = "qwen"
                        new_name = _name or "QWEN2.5 7B Instruct (default)"
                        new_priority = 100 if (_priority is None or _priority < 100) else _priority
                        try:
                            import json as _json
                            tags_list = []
                            if _tags:
                                try:
                                    tags_list = _json.loads(_tags) if isinstance(_tags, str) else list(_tags)
                                except Exception:
                                    pass
                            for t in ["default", "qwen", "multilingual", "russian", "instruct"]:
                                if t not in tags_list:
                                    tags_list.append(t)
                            tags_json = _json.dumps(tags_list)
                        except Exception:
                            tags_json = "[\"default\", \"qwen\"]"
                        cursor.execute(
                            """
                            UPDATE models
                            SET model_path = ?, model_type = ?, name = ?, priority = ?, tags = ?, last_updated = ?
                            WHERE id = ?
                            """,
                            (new_path, new_type, new_name, new_priority, tags_json, time.time(), "default_text_gen")
                        )
                        conn.commit()
                        logger.info("Миграция default_text_gen -> Qwen/Qwen2.5-7B-Instruct выполнена")
                else:
                    # Создаём alias default_text_gen на QWEN, если отсутствует
                    conn.close()
                    # Используем штатную регистрацию для корректной синхронизации с кэшем/БД
                    self.register_model(
                        "default_text_gen",
                        "Qwen/Qwen2.5-7B-Instruct",
                        "qwen",
                        priority=100,
                        name="QWEN2.5 7B Instruct (default)",
                        tags=["default", "qwen", "multilingual", "russian", "instruct"]
                    )
                # Безопасно закрываем соединение, если всё ещё открыто
                try:
                    conn.close()
                except Exception:
                    pass

            # Обновляем кэш метаданных
            self.model_metadata.clear()
            self._load_data()
        except Exception as e:
            logger.error(f"Ошибка миграции стандартной модели: {e}", exc_info=True)
    
    def _init_db(self):
        """Инициализирует базу данных для хранения метаданных моделей."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица моделей
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                model_type TEXT NOT NULL,
                priority INTEGER NOT NULL,
                tags TEXT DEFAULT '[]',
                description TEXT DEFAULT '',
                domain TEXT DEFAULT 'general',
                strength REAL DEFAULT 0.7,
                timestamp REAL NOT NULL,
                last_updated REAL NOT NULL
            )
            """)
            
            # Индексы для ускорения поиска
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_priority ON models(priority)")
            
            conn.commit()
            conn.close()
            logger.debug("База данных моделей инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных моделей: {e}", exc_info=True)
    
    def _init_components(self):
        """Инициализирует компоненты для интеграции с другими модулями."""
        # ModelManager НЕ должен инициализировать гибридный кэш
        # Он должен использовать уже инициализированный из MemoryManager через MLUnit
        
        # Используем текстовый процессор из MLUnit
        self.text_processor = None
        if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
            # Проверяем, инициализирован ли текстовый процессор
            if hasattr(self.brain.ml_unit, 'text_processor') and self.brain.ml_unit.text_processor:
                self.text_processor = self.brain.ml_unit.text_processor
                logger.debug("Используем UnifiedTextProcessor для токенизации")
        
        # Если text_processor недоступен, он будет установлен через событийную систему
        
        # ModelManager НЕ должен использовать StreamTokenizer
        # Удаляем все упоминания StreamTokenizer
        self.token_streamer = self.text_processor  # Используем напрямую UnifiedTextProcessor
        
        # Определение устройства
        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"ModelManager использует {self.device} для вычислений")

        # Async text generator adapter
        try:
            self.async_gen = AsyncTextGenerator(device=self.device)
        except Exception:
            self.async_gen = AsyncTextGenerator(device="cpu")

    async def generate_stream(
        self,
        model_id: str,
        prompt: str,
        sampling: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Асинхронная потоковая генерация текста для заданной модели.

        Не использует model.generate; управляет декодированием вручную через AsyncTextGenerator.
        """
        # ensure model is loaded or being loaded
        if model_id not in self.models:
            self.load_model(model_id)
            # Wait briefly for background load if a future exists
            fut = self.model_futures.get(model_id)
            if fut is not None:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, lambda: fut.result(timeout=120))
                except Exception as e:
                    logger.error(f"Не удалось загрузить модель {model_id} для stream: {e}")
                    return
        model_instance = self.models.get(model_id)
        if not model_instance:
            logger.error(f"Модель {model_id} недоступна для stream")
            return

        model = model_instance.model
        tokenizer = model_instance.tokenizer
        text_processor = self.text_processor

        async for chunk in self.async_gen.generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            text_processor=text_processor,
            sampling=sampling,
            cache_opts=cache,
            cancel_event=cancel_event,
        ):
            yield chunk
    
    def _on_text_processor_ready(self, text_processor):
        """Вызывается, когда текстовый процессор становится доступным."""
        self.text_processor = text_processor
        self.token_streamer = text_processor
        logger.info("UnifiedTextProcessor стал доступен для ModelManager через событийную систему")
    
    def _is_offline(self) -> bool:
        """Определяет, находится ли Transformers/HF в офлайн-режиме."""
        try:
            return (
                str(os.environ.get("TRANSFORMERS_OFFLINE", "")).strip() == "1" or
                str(os.environ.get("HF_HUB_OFFLINE", "")).strip() == "1"
            )
        except Exception:
            return False
    
    def _start_background_services(self):
        """Запускает фоновые службы менеджера моделей."""
        # Останавливаем существующие потоки, если они есть
        self.stop()
        
        # Запускаем фоновый мониторинг
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring,
            name="ModelManagerMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Запускаем фоновую загрузку
        self.loading_thread = threading.Thread(
            target=self._background_loading,
            name="ModelManagerLoading",
            daemon=True
        )
        self.loading_thread.start()
        
        self.running = True
        logger.info("Фоновые службы ModelManager запущены")
    
    def _background_monitoring(self):
        """Фоновый мониторинг состояния моделей."""
        while not self.stop_event.is_set():
            try:
                # Проверяем здоровье каждые 5 минут
                self.stop_event.wait(300)
                self._check_models_health()
                
                # Проверяем целостность
                self._check_integrity()
                
            except Exception as e:
                logger.error(f"Ошибка в фоновом мониторинге ModelManager: {e}", exc_info=True)
    
    def _background_loading(self):
        """Фоновая загрузка моделей."""
        while not self.stop_event.is_set():
            try:
                # Загружаем модели каждые 10 секунд
                self.stop_event.wait(10)
                self._load_pending_models()
                
            except Exception as e:
                logger.error(f"Ошибка в фоновой загрузке ModelManager: {e}", exc_info=True)

    # ----------------------------
    # Low-impact loading helpers
    # ----------------------------
    def _emit_model_load_event(self, kind: str, payload: Dict[str, Any]) -> None:
        try:
            brain = getattr(self, "brain", None)
            data = {"event": kind, **payload}
            if brain is None:
                logger.debug(f"model_load event: {data}")
                return
            # Prefer unified event bus
            if hasattr(brain, "events") and brain.events:
                try:
                    brain.events.trigger('model_load', data)
                except Exception:
                    pass
            # Optional UI callbacks
            cbs = getattr(brain, 'on_model_load', None)
            if cbs:
                for cb in cbs:
                    try:
                        cb(data)
                    except Exception:
                        pass
        except Exception:
            pass

    def _emit_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        try:
            brain = getattr(self, "brain", None)
            if not brain:
                return
            if hasattr(brain, "events") and brain.events:
                try:
                    brain.events.trigger('metrics', metrics)
                except Exception:
                    pass
            if hasattr(brain, "emit_metrics"):
                try:
                    brain.emit_metrics(metrics)
                except Exception:
                    pass
        except Exception:
            pass

    def load_model_low_impact(self, model_id: str, throttle_sleep: float = 0.05, max_threads: int = 1) -> bool:
        """
        Запускает щадящую (низкое влияние) фоновую загрузку модели с троттлингом CPU/IO и событиями прогресса.
        Возвращает True, если загрузка запущена (или уже идет/завершена).
        """
        if model_id in self.models or model_id in self.model_futures:
            return True
        metadata = self.model_metadata.get(model_id)
        if not metadata:
            logger.warning(f"Метаданные для модели {model_id} не найдены")
            return False

        def _task():
            start_ts = time.time()
            self._emit_model_load_event('model_load_start', {"model_id": model_id, "name": metadata.name})
            # Reduce thread usage
            try:
                torch.set_num_threads(max(1, int(max_threads)))
            except Exception:
                pass
            try:
                os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
                os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
            except Exception:
                pass
            try:
                # Step 1: tokenizer
                tok = self._load_tokenizer_from_path(metadata.model_path)
                if tok is None:
                    raise RuntimeError("tokenizer_load_failed")
                self._emit_model_load_event('model_load_progress', {"model_id": model_id, "progress": 20})
                time.sleep(max(0.0, throttle_sleep))

                # Step 2: model weights (heavy)
                mdl = self._load_model_from_path(metadata.model_path, metadata.model_type)
                if mdl is None:
                    raise RuntimeError("model_load_failed")
                self._emit_model_load_event('model_load_progress', {"model_id": model_id, "progress": 90})
                time.sleep(max(0.0, throttle_sleep))

                instance = ModelInstance(model=mdl, tokenizer=tok, metadata=metadata, status=ModelStatus.LOADED)
                with self.model_lock:
                    self.models[model_id] = instance
                    if model_id in self.model_futures:
                        del self.model_futures[model_id]

                # notify MLUnit links
                self._register_model_with_ml_unit(model_id)

                dur = time.time() - start_ts
                self._emit_model_load_event('model_load_complete', {"model_id": model_id, "duration_sec": round(dur, 3)})
                try:
                    self._emit_metrics([
                        {"name": "models.load_completed", "component": "model_manager", "type": "counter", "value": 1.0},
                        {"name": "models.load_duration_sec", "component": "model_manager", "type": "histogram", "value": float(dur)},
                    ])
                except Exception:
                    pass
            except Exception as e:
                self._emit_model_load_event('model_load_error', {"model_id": model_id, "error": str(e)})
                try:
                    self._emit_metrics([
                        {"name": "models.load_failed", "component": "model_manager", "type": "counter", "value": 1.0},
                    ])
                except Exception:
                    pass
                with self.model_lock:
                    if model_id in self.model_futures:
                        del self.model_futures[model_id]

        # Prefer scheduling via DeferredCommandSystem with low priority
        brain = getattr(self, 'brain', None)
        try:
            if brain and hasattr(brain, 'deferred_system') and brain.deferred_system:
                # Use a small wrapper to run inside DCS
                def _cmd():
                    _task()
                # Fallback to brain.add_deferred_command if available
                if hasattr(brain, 'add_deferred_command') and callable(brain.add_deferred_command):
                    brain.add_deferred_command(_cmd)
                else:
                    # Run via executor as background
                    fut = self.executor.submit(_cmd)
                    self.model_futures[model_id] = fut
            else:
                fut = self.executor.submit(_task)
                self.model_futures[model_id] = fut
        except Exception:
            # Last resort
            fut = self.executor.submit(_task)
            self.model_futures[model_id] = fut
        return True

    def schedule_low_impact_load(self, model_id: str) -> bool:
        """Планирует щадящую загрузку через отложенную систему, если она доступна."""
        return self.load_model_low_impact(model_id)
    
    def _check_models_health(self):
        """Проверяет здоровье загруженных моделей."""
        with self.model_lock:
            for model_id, model_instance in self.models.items():
                # Обновляем статистику
                model_instance.health.usage_count = model_instance.usage_count
                model_instance.health.error_count = model_instance.error_count
                model_instance.health.last_used = model_instance.last_used
                
                # Вычисляем оценку здоровья
                health_score = 0.9  # Базовая оценка
                
                # Уменьшаем оценку за ошибки
                if model_instance.error_count > 0:
                    health_score -= min(0.5, model_instance.error_count * 0.05)
                
                # Уменьшаем оценку за время бездействия
                idle_time = time.time() - model_instance.last_used
                if idle_time > 3600:  # Больше часа
                    health_score -= min(0.2, (idle_time - 3600) / 86400 * 0.2)
                
                model_instance.health.health_score = max(0.1, health_score)
                
                # Обновляем статус
                if health_score < 0.3:
                    model_instance.health.status = "critical"
                elif health_score < 0.6:
                    model_instance.health.status = "warning"
                else:
                    model_instance.health.status = "healthy"
    
    def _check_integrity(self):
        """Проверяет целостность системы моделей."""
        # Проверяем отсутствующие файлы
        missing_files = []
        for model_id, metadata in self.model_metadata.items():
            if not os.path.exists(metadata.model_path):
                missing_files.append((model_id, metadata.model_path))
        
        if missing_files:
            logger.warning(f"Обнаружено {len(missing_files)} отсутствующих файлов моделей")
            
            # Помечаем модели как недействительные
            for model_id, _ in missing_files:
                self.unload_model(model_id)
                with self.db_lock:
                    try:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute("UPDATE models SET priority = ? WHERE id = ?", (-1, model_id))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        logger.error(f"Ошибка обновления приоритета модели: {e}", exc_info=True)
    
    def _load_pending_models(self):
        """Загружает модели с высоким приоритетом, которые еще не загружены."""
        with self.model_lock:
            # Получаем список моделей для загрузки (отсортированных по приоритету)
            models_to_load = [
                (model_id, metadata) 
                for model_id, metadata in self.model_metadata.items()
                if model_id not in self.models 
                and model_id not in self.model_futures
                and metadata.priority >= 0  # Только действительные модели
            ]
            
            # Сортируем по приоритету (от высшего к низшему)
            models_to_load.sort(key=lambda x: (-x[1].priority, x[1].timestamp))
            
            # Загружаем первые доступные модели
            for model_id, metadata in models_to_load[:self.max_workers]:
                if len(self.models) < self.max_workers:
                    logger.info(f"Загрузка модели {metadata.name} (приоритет: {metadata.priority})")
                    self.load_model(model_id)
    
    def _load_data(self):
        """Загружает данные моделей из базы данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
            SELECT id, name, model_path, model_type, priority, tags, 
                   description, domain, strength, timestamp, last_updated
            FROM models
            """)
            
            for row in cursor.fetchall():
                try:
                    # Исправленная обработка тегов
                    tags = []
                    if row[5]:
                        try:
                            # Попробуем распарсить как JSON
                            tags = json.loads(row[5])
                        except json.JSONDecodeError:
                            # Если не JSON, возможно, это строка с тегами
                            if isinstance(row[5], str):
                                # Разделяем по запятым или пробелам
                                tags = [tag.strip() for tag in re.split(r'[,\s]+', row[5]) if tag.strip()]
                    
                    metadata = ModelMetadata(
                        id=row[0],
                        name=row[1],
                        model_path=row[2],
                        model_type=row[3],
                        priority=row[4],
                        tags=tags,
                        description=row[6],
                        domain=row[7],
                        strength=row[8],
                        timestamp=row[9],
                        last_updated=row[10]
                    )
                    self.model_metadata[row[0]] = metadata
                except Exception as e:
                    logger.error(f"Ошибка обработки метаданных модели {row[0]}: {str(e)}", exc_info=True)
            
            conn.close()
            logger.info(f"Загружено {len(self.model_metadata)} метаданных моделей")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}", exc_info=True)
    
    def scan_models_directory(self):
        """Сканирует директорию моделей и регистрирует новые модели."""
        start_time = time.time()
        logger.info(f"Начало сканирования директории моделей: {self.model_dir}")
        
        new_models = 0
        try:
            # Ищем все директории с моделями
            for root, dirs, files in os.walk(self.model_dir):
                # Проверяем, содержит ли директория файлы модели
                if "config.json" in files or "pytorch_model.bin" in files:
                    # Генерируем ID модели на основе хеша пути
                    model_id = hashlib.md5(root.encode()).hexdigest()
                    
                    # Проверяем, уже ли зарегистрирована эта модель
                    if model_id not in self.model_metadata:
                        # Определяем тип модели
                        model_type = self._detect_model_type(root)
                        
                        # Создаем метаданные
                        metadata = ModelMetadata(
                            id=model_id,
                            name=os.path.basename(root),
                            model_path=root,
                            model_type=model_type,
                            priority=5,  # Приоритет по умолчанию
                            tags=["local"],
                            domain="general",
                            strength=0.7
                        )
                        
                        # Сохраняем в базу данных
                        self._save_model_metadata(metadata)
                        self.model_metadata[model_id] = metadata
                        new_models += 1
            
            logger.info(f"Сканирование завершено за {time.time() - start_time:.2f} сек. Обнаружено моделей: {new_models}")
            return new_models
        except Exception as e:
            logger.error(f"Ошибка сканирования директории моделей: {e}", exc_info=True)
            return 0
    
    def _detect_model_type(self, model_path: str) -> str:
        """Определяет тип модели по пути."""
        path_lower = model_path.lower()
        if "qwen" in path_lower:
            return "qwen"
        elif "rugpt" in path_lower:
            return "rugpt3"
        elif "sberbank-ai" in path_lower and "rugpt" in path_lower:
            return "rugpt3"
        elif "gpt2" in path_lower:
            return "gpt"
        elif "gpt" in path_lower:
            return "gpt"
        elif "bart" in path_lower:
            return "bart"
        elif "t5" in path_lower:
            return "t5"
        elif "dialogpt" in path_lower:
            return "dialogpt"
        elif "bert" in path_lower:
            return "bert"
        elif "xlm" in path_lower:
            return "xlm"
        else:
            return "transformer"
    
    def _save_model_metadata(self, metadata: ModelMetadata):
        """Сохраняет метаданные модели в базу данных."""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Проверяем, существует ли модель
                cursor.execute("SELECT id FROM models WHERE id = ?", (metadata.id,))
                exists = cursor.fetchone()
                
                if exists:
                    # Обновляем существующую запись
                    cursor.execute("""
                    UPDATE models SET
                        name = ?, model_path = ?, model_type = ?, priority = ?,
                        tags = ?, description = ?, domain = ?, strength = ?, last_updated = ?
                    WHERE id = ?
                    """, (
                        metadata.name,
                        metadata.model_path,
                        metadata.model_type,
                        metadata.priority,
                        json.dumps(metadata.tags),
                        metadata.description,
                        metadata.domain,
                        metadata.strength,
                        time.time(),
                        metadata.id
                    ))
                else:
                    # Добавляем новую запись
                    cursor.execute("""
                    INSERT INTO models (
                        id, name, model_path, model_type, priority,
                        tags, description, domain, strength, timestamp, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metadata.id,
                        metadata.name,
                        metadata.model_path,
                        metadata.model_type,
                        metadata.priority,
                        json.dumps(metadata.tags),
                        metadata.description,
                        metadata.domain,
                        metadata.strength,
                        metadata.timestamp,
                        time.time()
                    ))
                
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных модели: {e}", exc_info=True)
    
    def register_model(
        self,
        model_id: str,
        model_path: str,
        model_type: str,
        priority: int = 5,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        domain: str = "general",
        strength: float = 0.7,
    ) -> ModelMetadata:
        """
        Регистрирует (или обновляет) модель в БД и локальном кэше метаданных без загрузки весов.
        Возвращает объект ModelMetadata.
        """
        try:
            name = name or (os.path.basename(model_path) if model_path else model_id)
            tags = tags or []
            metadata = ModelMetadata(
                id=model_id,
                name=name,
                model_path=model_path,
                model_type=model_type,
                priority=priority,
                tags=tags,
                description=description,
                domain=domain,
                strength=strength,
                timestamp=time.time(),
                last_updated=time.time(),
            )
            self._save_model_metadata(metadata)
            # Обновляем локальный кэш метаданных
            self.model_metadata[model_id] = metadata
            logger.info(f"Модель зарегистрирована/обновлена: {model_id} -> {model_path} ({model_type}, priority={priority})")
            return metadata
        except Exception as e:
            logger.error(f"Не удалось зарегистрировать модель {model_id}: {e}", exc_info=True)
            # Создаём минимальный объект для возврата, чтобы не падать
            return ModelMetadata(
                id=model_id,
                name=name or model_id,
                model_path=model_path,
                model_type=model_type,
                priority=max(0, priority),
                tags=tags or [],
            )

    def _add_default_models(self):
        """Добавляет базовые модели по умолчанию, если БД пуста или отсутствуют критичные alias."""
        try:
            # Обновляем модельные метаданные из БД на всякий случай
            if not self.model_metadata:
                self._load_data()

            # Убедимся, что существует alias default_text_gen -> QWEN
            if "default_text_gen" not in self.model_metadata:
                self.register_model(
                    "default_text_gen",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "qwen",
                    priority=100,
                    name="QWEN2.5 7B Instruct (default)",
                    tags=["default", "qwen", "multilingual", "russian", "instruct"],
                )

            logger.debug("Базовые модели по умолчанию проверены/добавлены")
        except Exception as e:
            logger.error(f"Ошибка добавления базовых моделей по умолчанию: {e}", exc_info=True)
    
    def load_model(self, model_id: str) -> bool:
        """
        Загружает модель асинхронно.
        
        Args:
            model_id: ID модели
            
        Returns:
            bool: Успешно ли запущена загрузка
        """
        if model_id in self.models:
            logger.debug(f"Модель {model_id} уже загружена")
            return True
        
        if model_id in self.model_futures:
            logger.debug(f"Загрузка модели {model_id} уже запущена")
            return True
        
        metadata = self.model_metadata.get(model_id)
        if not metadata:
            logger.warning(f"Метаданные для модели {model_id} не найдены")
            return False
        
        logger.info(f"Запущена асинхронная загрузка модели: {model_id}")
        
        # Запускаем загрузку в фоновом потоке
        future = self.executor.submit(self._load_model_internal, model_id)
        self.model_futures[model_id] = future
        
        return True
    
    def _load_model_internal(self, model_id: str) -> Optional[ModelInstance]:
        """Внутренняя реализация загрузки модели."""
        start_time = time.time()
        logger.info(f"Загрузка модели {model_id}")
        
        try:
            metadata = self.model_metadata.get(model_id)
            if not metadata:
                logger.error(f"Метаданные для модели {model_id} не найдены")
                return None
            
            # Загружаем токенизатор
            tokenizer = self._load_tokenizer_from_path(metadata.model_path)
            if tokenizer is None:
                logger.error(f"Не удалось загрузить токенизатор для модели {model_id}")
                return None
            
            # Загружаем модель
            model = self._load_model_from_path(metadata.model_path, metadata.model_type)
            if model is None:
                logger.error(f"Не удалось загрузить модель {model_id}")
                return None
            
            # Создаем экземпляр модели
            model_instance = ModelInstance(
                model=model,
                tokenizer=tokenizer,
                metadata=metadata
            )
            
            # Помечаем как загруженную
            with self.model_lock:
                self.models[model_id] = model_instance
                if model_id in self.model_futures:
                    del self.model_futures[model_id]
            
            # Уведомляем MLUnit о доступности модели
            self._register_model_with_ml_unit(model_id)
            
            logger.info(f"Модель {model_id} успешно загружена за {time.time() - start_time:.2f} сек")
            return model_instance
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_id}: {str(e)}", exc_info=True)
            with self.model_lock:
                if model_id in self.model_futures:
                    del self.model_futures[model_id]
            return None
    
    def _load_tokenizer_from_path(self, model_path: str):
        """Загружает токенизатор из локального пути или Hugging Face репозитория."""
        try:
            # Если это стандартная модель (например, gpt2), загружаем из HuggingFace
            if model_path in ["gpt2", "gpt2-medium", "distilgpt2"]:
                logger.info(f"Загрузка стандартного токенизатора {model_path} из HuggingFace Hub")
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained(
                    model_path,
                    local_files_only=self._is_offline()
                )
                # Добавляем pad_token если его нет
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
            
            # Попытка загрузки напрямую (HF repo id или локальный путь)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=self._is_offline()
                )
                return tokenizer
            except Exception as e:
                logger.debug(f"Не удалось загрузить токенизатор напрямую ({model_path}), пробуем по структуре: {e}")
            
            # Если прямой путь не сработал, пробуем как локальную структуру
            if not os.path.exists(model_path):
                logger.error(f"Путь к локальной модели не существует: {model_path}")
                return None
            
            # Проверяем наличие основных файлов
            config_path = os.path.join(model_path, "config.json")
            vocab_path = os.path.join(model_path, "vocab.json")
            merges_path = os.path.join(model_path, "merges.txt")
            
            if not os.path.exists(config_path):
                logger.warning(f"Файл config.json отсутствует в {model_path}")
            
            # Пытаемся определить тип модели по имени директории
            model_type = None
            if "gpt2" in model_path.lower():
                model_type = "gpt2"
            elif "bart" in model_path.lower():
                model_type = "bart"
            elif "dialogpt" in model_path.lower():
                model_type = "dialogpt"
            elif "t5" in model_path.lower():
                model_type = "t5"
            
            # Если модель из Hugging Face Hub, используем специфичную загрузку
            if model_type:
                try:
                    if model_type == "gpt2":
                        from transformers import GPT2Tokenizer
                        tokenizer = GPT2Tokenizer.from_pretrained(
                            model_path,
                            local_files_only=self._is_offline()
                        )
                    elif model_type == "bart":
                        from transformers import BartTokenizer
                        tokenizer = BartTokenizer.from_pretrained(
                            model_path,
                            local_files_only=self._is_offline()
                        )
                    elif model_type == "dialogpt":
                        from transformers import GPT2Tokenizer
                        tokenizer = GPT2Tokenizer.from_pretrained(
                            model_path,
                            local_files_only=self._is_offline()
                        )
                    elif model_type == "t5":
                        from transformers import T5Tokenizer
                        tokenizer = T5Tokenizer.from_pretrained(
                            model_path,
                            local_files_only=self._is_offline()
                        )
                    
                    logger.debug(f"Токенизатор {model_type} успешно загружен из {model_path}")
                    return tokenizer
                except Exception as e:
                    logger.warning(f"Ошибка загрузки специфичного токенизатора: {str(e)}")
            
            # Обычная загрузка как резерв
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=self._is_offline()
            )
            return tokenizer
            
        except Exception as e:
            logger.error(f"Ошибка загрузки токенизатора из {model_path}: {str(e)}")
            return None
    
    def _load_model_from_path(self, model_path: str, model_type: str):
        """Загружает модель из локального пути или Hugging Face репозитория."""
        try:
            # Если это стандартная модель (например, gpt2), загружаем из HuggingFace
            if model_path in ["gpt2", "gpt2-medium", "distilgpt2"]:
                logger.info(f"Загрузка стандартной модели {model_path} из HuggingFace Hub")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=self._is_offline()
                )
                model = model.to(self.device)
                return model
            
            # Попытка загрузки напрямую (HF repo id или локальный путь)
            try:
                if "gpt" in model_type.lower():
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=self._is_offline()
                    )
                elif "bart" in model_type.lower() or "t5" in model_type.lower():
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=self._is_offline()
                    )
                else:
                    # Пытаемся определить автоматически
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            local_files_only=self._is_offline()
                        )
                    except Exception:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            local_files_only=self._is_offline()
                        )
            except Exception as e:
                logger.debug(f"Не удалось загрузить модель напрямую ({model_path}), пробуем как локальный путь: {e}")
                if not os.path.exists(model_path):
                    logger.error(f"Путь к локальной модели не существует: {model_path}")
                    return None
                if "gpt" in model_type.lower():
                    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
                elif "bart" in model_type.lower() or "t5" in model_type.lower():
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
                else:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
                    except Exception:
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
            
            # Определяем тип модели
            # Перемещаем на устройство
            model = model.to(self.device)
            
            logger.info(f"Модель загружена с устройства: {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели из {model_path}: {str(e)}")
            return None
    
    def unload_model(self, model_id: str) -> bool:
        """
        Выгружает модель из памяти.
        
        Args:
            model_id: ID модели
            
        Returns:
            bool: Успешно ли выгружена
        """
        with self.model_lock:
            if model_id in self.models:
                # Удаляем модель
                del self.models[model_id]
                logger.info(f"Модель {model_id} выгружена из памяти")
                return True
            
            if model_id in self.model_futures:
                # Отменяем загрузку
                future = self.model_futures[model_id]
                future.cancel()
                del self.model_futures[model_id]
                logger.info(f"Загрузка модели {model_id} отменена")
                return True
            
            logger.warning(f"Модель {model_id} не найдена для выгрузки")
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelInstance]:
        """
        Возвращает экземпляр модели.
        
        Args:
            model_id: ID модели
            
        Returns:
            Optional[ModelInstance]: Экземпляр модели или None
        """
        with self.model_lock:
            return self.models.get(model_id)
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Возвращает метаданные модели.
        
        Args:
            model_id: ID модели
            
        Returns:
            Optional[ModelMetadata]: Метаданные модели или None
        """
        return self.model_metadata.get(model_id)
    
    def get_available_models(self, domain: Optional[str] = None, 
                            min_priority: int = 0) -> List[ModelMetadata]:
        """
        Возвращает доступные модели.
        
        Args:
            domain: Фильтр по домену
            min_priority: Минимальный приоритет
            
        Returns:
            List[ModelMetadata]: Список доступных моделей
        """
        models = [
            metadata for metadata in self.model_metadata.values()
            if metadata.priority >= min_priority
        ]
        
        if domain:
            models = [m for m in models if m.domain == domain]
        
        # Сортируем по приоритету
        models.sort(key=lambda m: (-m.priority, m.timestamp))
        
        return models
    
    def generate_response(self, prompt: str, model_id: Optional[str] = None, 
                        max_length: int = 100, temperature: float = 0.7,
                        top_p: float = 0.9, task: str = "text-generation") -> Dict[str, Any]:
        """
        Генерирует ответ с использованием модели.
        
        Args:
            prompt: Входной текст
            model_id: ID модели (опционально)
            max_length: Максимальная длина ответа
            temperature: Температура генерации
            top_p: Параметр top-p сэмплинга
            task: Тип задачи
            
        Returns:
            Dict[str, Any]: Сгенерированный ответ
        """
        # Проверяем готовность компонентов
        if not self.is_ready():
            return {"error": "model_manager_not_ready"}
        
        if self.token_streamer and hasattr(self.token_streamer, 'is_ready') and not self.token_streamer.is_ready():
            return {"error": "token_streamer_not_ready"}
        
        start_time = time.time()
        
        # Выбираем модель
        if model_id:
            model_instance = self.get_model(model_id)
            if not model_instance:
                logger.warning(f"Модель {model_id} не загружена, пытаемся загрузить")
                if not self.load_model(model_id):
                    return {"error": "model_not_available"}
                model_instance = self.get_model(model_id)
                if not model_instance:
                    return {"error": "model_load_failed"}
        else:
            # Выбираем лучшую доступную модель
            model_instance = self._select_best_model(task)
            if not model_instance:
                logger.warning(f"Модель для задачи '{task}' не найдена")
                # Попробуем использовать первую доступную модель как резерв
                if self.models:
                    model_instance = next(iter(self.models.values()))
                    logger.info(f"Используем резервную модель: {model_instance.metadata.name}")
                else:
                    return {"error": "no_model_available"}
        
        # Обновляем статистику
        model_instance.last_used = time.time()
        model_instance.usage_count += 1
        
        try:
            # Токенизация
            if self.token_streamer and hasattr(self.token_streamer, 'tokenize_async'):
                # Используем UnifiedTextProcessor для токенизации
                tokens = self.token_streamer.tokenize_async([prompt])[0]
                input_ids = tokens["input_ids"]
            else:
                # Проверяем, есть ли токенизатор у модели
                if not model_instance.tokenizer:
                    logger.error("Токенизатор не доступен для модели")
                    return {"error": "tokenizer_not_available"}
                
                # Резервная токенизация с правильным преобразованием
                inputs = model_instance.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Перемещаем на устройство модели
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            
            # Генерация
            generate_kwargs = {
                "input_ids": input_ids,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": model_instance.tokenizer.pad_token_id
            }
            
            # Добавляем attention_mask если он есть
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            outputs = model_instance.model.generate(**generate_kwargs)
            
            # Декодирование
            response = model_instance.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            logger.debug(f"Генерация ответа завершена за {processing_time:.2f} сек")
            
            return {
                "text": response,
                "model_id": model_instance.metadata.id,
                "model_name": model_instance.metadata.name,
                "processing_time": processing_time,
                "token_count": len(outputs[0])
            }
            
        except Exception as e:
            if 'model_instance' in locals():
                model_instance.error_count += 1
            logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _select_best_model(self, task: str) -> Optional[ModelInstance]:
        """Выбирает лучшую модель для задачи."""
        with self.model_lock:
            # Сначала ищем загруженные модели подходящего типа
            for model_id, model_instance in self.models.items():
                if self._is_model_suitable(model_instance, task):
                    logger.info(f"Найдена подходящая загруженная модель: {model_instance.metadata.name}")
                    return model_instance
            
            # Если есть любая загруженная модель, используем её
            if self.models:
                model_instance = next(iter(self.models.values()))
                logger.info(f"Используем любую доступную модель: {model_instance.metadata.name}")
                return model_instance
            
            # Если нет загруженных, ищем в метаданных
            suitable_models = []
            for model_id, metadata in self.model_metadata.items():
                if self._is_model_metadata_suitable(metadata, task):
                    suitable_models.append(metadata)
            
            # Сортируем по приоритету
            suitable_models.sort(key=lambda m: (-m.priority, m.timestamp))
            
            # Пытаемся загрузить лучшую модель
            if suitable_models:
                self.load_model(suitable_models[0].id)
                # Ожидаем завершения загрузки
                import time
                for _ in range(10):  # Максимум 10 секунд
                    model = self.get_model(suitable_models[0].id)
                    if model:
                        return model
                    time.sleep(1)
            
            return None
    
    def _is_model_suitable(self, model_instance: ModelInstance, task: str) -> bool:
        """Проверяет, подходит ли модель для задачи."""
        model_type = model_instance.metadata.model_type.lower()
        
        if task == "text-generation":
            # Проверяем на наличие ключевых слов в типе модели, приоритет QWEN и русским моделям
            return any(keyword in model_type for keyword in ["qwen", "rugpt", "gpt", "dialogpt", "text-generation"])
        elif task == "summarization":
            return any(keyword in model_type for keyword in ["bart", "t5", "summarization"])
        elif task == "translation":
            return any(keyword in model_type for keyword in ["t5", "translation"])
        else:
            return True
    
    def _is_model_metadata_suitable(self, metadata: ModelMetadata, task: str) -> bool:
        """Проверяет, подходит ли метаданные модели для задачи."""
        model_type = metadata.model_type.lower()
        
        if task == "text-generation":
            return "qwen" in model_type or "rugpt" in model_type or "gpt" in model_type or "dialogpt" in model_type
        elif task == "summarization":
            return "bart" in model_type or "t5" in model_type
        elif task == "translation":
            return "t5" in model_type
        else:
            return True
    
    def start(self):
        """Запускает фоновые процессы менеджера моделей."""
        if self.running:
            return
            
        self.stop_event.clear()
        self.running = True
        logger.info("ModelManager запущен")
    
    def stop(self):
        """Останавливает фоновые процессы менеджера моделей."""
        if not self.running:
            return
            
        self.stop_event.set()
        self.running = False
        
        # Дожидаемся завершения фоновых потоков
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        if hasattr(self, 'loading_thread') and self.loading_thread.is_alive():
            self.loading_thread.join(timeout=2.0)
        
        # Отменяем все асинхронные задачи
        for future in self.model_futures.values():
            future.cancel()
        
        # Выгружаем все модели
        for model_id in list(self.models.keys()):
            self.unload_model(model_id)
        
        # Закрываем пул потоков
        self.executor.shutdown(wait=True)
        
        logger.info("ModelManager остановлен")
    
    def close(self):
        """Закрывает менеджер моделей и освобождает ресурсы."""
        self.stop()
        
        # Освобождаем ресурсы
        self.models = {}
        self.model_metadata = {}
        self.model_futures = {}
        
        logger.info("ModelManager закрыт")
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирован ли менеджер моделей."""
        return self.initialized
    
    def is_running(self) -> bool:
        """Проверяет, запущен ли менеджер моделей."""
        return self.running
    
    def is_ready(self) -> bool:
        """Проверяет готовность менеджера моделей к работе."""
        return (self.initialized and 
                self.running and 
                (len(self.models) > 0 or len(self.model_metadata) > 0))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по моделям.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        with self.model_lock:
            loaded_count = len(self.models)
            total_count = len(self.model_metadata)
            pending_count = len(self.model_futures)
            
            # Собираем статистику по приоритетам
            priority_stats = defaultdict(int)
            for metadata in self.model_metadata.values():
                priority_stats[metadata.priority] += 1
            
            return {
                "loaded_models": loaded_count,
                "total_models": total_count,
                "pending_models": pending_count,
                "unloaded_models": total_count - loaded_count - pending_count,
                "priority_distribution": dict(priority_stats),
                "device": self.device,
                "gpu_available": torch.cuda.is_available()
            }
    
    def get_model_health(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о здоровье модели.
        
        Args:
            model_id: ID модели
            
        Returns:
            Optional[Dict[str, Any]]: Информация о здоровье или None
        """
        model_instance = self.get_model(model_id)
        if not model_instance:
            return None
        
        return {
            "status": model_instance.health.status,
            "health_score": model_instance.health.health_score,
            "usage_count": model_instance.usage_count,
            "error_count": model_instance.error_count,
            "last_used": model_instance.last_used,
            "memory_usage": model_instance.health.memory_usage
        }
    
    def update_model_priority(self, model_id: str, priority: int) -> bool:
        """
        Обновляет приоритет модели.
        
        Args:
            model_id: ID модели
            priority: Новый приоритет
            
        Returns:
            bool: Успешно ли обновлено
        """
        metadata = self.model_metadata.get(model_id)
        if not metadata:
            logger.warning(f"Метаданные для модели {model_id} не найдены")
            return False
        
        metadata.priority = priority
        
        # Сохраняем в базу данных
        self._save_model_metadata(metadata)
        
        logger.info(f"Приоритет модели {model_id} обновлен до {priority}")
        return True
    
    def get_model_for_task(self, task: str) -> Optional[Tuple[Any, Any, str]]:
        """
        Возвращает модель для конкретной задачи в формате (model, tokenizer, model_name).
        
        Args:
            task: Тип задачи
            
        Returns:
            Optional[Tuple[Any, Any, str]]: Кортеж (model, tokenizer, model_name) или None
        """
        model_instance = self._select_best_model(task)
        if model_instance:
            logger.info(f"Найдена модель для задачи '{task}': {model_instance.metadata.name}")
            return (
                model_instance.model,
                model_instance.tokenizer,
                model_instance.metadata.name
            )
        
        logger.warning(f"generate_response: модель для задачи '{task}' не найдена")
        return None
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Возвращает детальный отчет о состоянии моделей.
        
        Returns:
            Dict[str, Any]: Отчет о состоянии
        """
        report = {
            "timestamp": time.time(),
            "statistics": self.get_statistics(),
            "models": []
        }
        
        with self.model_lock:
            for model_id, model_instance in self.models.items():
                report["models"].append({
                    "id": model_id,
                    "name": model_instance.metadata.name,
                    "type": model_instance.metadata.model_type,
                    "priority": model_instance.metadata.priority,
                    "health": {
                        "status": model_instance.health.status,
                        "score": model_instance.health.health_score,
                        "usage": model_instance.usage_count,
                        "errors": model_instance.error_count
                    },
                    "last_used": model_instance.last_used
                })
        
        # Добавляем незагруженные модели
        for model_id, metadata in self.model_metadata.items():
            if model_id not in self.models and model_id not in self.model_futures:
                report["models"].append({
                    "id": model_id,
                    "name": metadata.name,
                    "type": metadata.model_type,
                    "priority": metadata.priority,
                    "status": "unloaded"
                })
        
        return report
    
    def register_model(self, model_id: str, model_path: str, model_type: str, priority: int = 5,
                       name: Optional[str] = None, tags: Optional[List[str]] = None,
                       description: str = "", domain: str = "general", strength: float = 0.7,
                       **kwargs):
        """Регистрирует новую модель в системе. Поддерживает как локальные пути, так и HF repo IDs.
        Дополнительные параметры в kwargs игнорируются, но допускаются для совместимости.
        """
        # Нормализуем тип модели
        normalized_type = model_type.lower()
        
        # Добавляем ключевые слова для распознавания
        if "gpt2" in normalized_type:
            normalized_type = "gpt"
        elif "bart" in normalized_type:
            normalized_type = "bart"
        elif "t5" in normalized_type:
            normalized_type = "t5"
        elif "dialogpt" in normalized_type:
            normalized_type = "dialogpt"
        
        # Имя и теги по умолчанию
        resolved_name = name or os.path.basename(model_path)
        resolved_tags = tags if tags is not None else (["hf"] if "/" in model_path else ["local"])
        
        # Создаем метаданные
        metadata = ModelMetadata(
            id=model_id,
            name=resolved_name,
            model_path=model_path,
            model_type=normalized_type,
            priority=priority,
            tags=resolved_tags,
            description=description,
            domain=domain,
            strength=strength
        )
        
        # Сохраняем в базу данных
        self._save_model_metadata(metadata)
        self.model_metadata[model_id] = metadata
        
        # Синхронизируем с MLUnit
        self._register_model_with_ml_unit(model_id)
        
        logger.info(f"Модель зарегистрирована: id='{model_id}', name='{resolved_name}', type='{normalized_type}', priority={priority}")
    
    def _register_model_with_ml_unit(self, model_id: str):
        """Регистрирует модель в MLUnit."""
        if hasattr(self.brain, 'ml_unit') and hasattr(self.brain.ml_unit, 'add_model'):
            model_instance = self.get_model(model_id)
            if model_instance:
                self.brain.ml_unit.add_model(model_id, model_instance)
                logger.debug(f"Модель {model_id} зарегистрирована в MLUnit")
            else:
                logger.warning(f"Не удалось получить экземпляр модели {model_id} для регистрации в MLUnit")
    
    def test_text_generation(self):
        """Тестирует генерацию текста с помощью модели."""
        try:
            # Проверяем, есть ли модель для генерации
            model_instance = self._select_best_model("text-generation")
            if not model_instance:
                logger.warning("Модель для генерации текста не найдена")
                return False
            
            # Тестируем генерацию
            response = self.generate_response(
                "Привет, как дела?",
                max_length=50,
                temperature=0.7,
                top_p=0.9
            )
            
            if "error" in response:
                logger.error(f"Ошибка генерации текста: {response['error']}")
                return False
            
            # Проверяем качество ответа
            text = response["text"]
            if len(text) < 10 or text.count("Привет") > 3:
                logger.warning("Сгенерированный текст некачественный")
                return False
            
            logger.info(f"Тест генерации текста успешен. Ответ: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка тестирования генерации текста: {e}", exc_info=True)
            return False
    
    def _add_default_models(self):
        """Добавляет базовые модели (Qwen + русскоязычные) с корректными HF repo ID и метаданными."""
        try:
            # Регистрируем QWEN модели с поддержкой русского языка (приоритет)
            
            # QWEN2.5 - самая мощная модель с отличной поддержкой русского
            self.register_model(
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "qwen",
                priority=98,
                name="QWEN2.5 7B Instruct",
                tags=["qwen", "multilingual", "russian", "32k_context", "instruct"]
            )
            
            # QWEN2.5 14B для более сложных задач
            self.register_model(
                "Qwen/Qwen2.5-14B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
                "qwen",
                priority=99,
                name="QWEN2.5 14B Instruct",
                tags=["qwen", "multilingual", "russian", "32k_context", "instruct", "large"]
            )
            
            # QWEN2.5 3B для быстрых ответов
            self.register_model(
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "qwen",
                priority=95,
                name="QWEN2.5 3B Instruct",
                tags=["qwen", "multilingual", "russian", "32k_context", "instruct", "fast"]
            )
            
            # QWEN2.5 Coder для программирования
            self.register_model(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "qwen",
                priority=97,
                name="QWEN2.5 Coder 7B",
                tags=["qwen", "multilingual", "russian", "32k_context", "coding"]
            )
            
            # Регистрируем мощные русскоязычные модели с поддержкой длинного контекста (8000+ токенов)
            
            # RuGPT-3 Large - самая мощная русская модель
            self.register_model(
                "ai-forever/rugpt3large_based_on_gpt2",
                "ai-forever/rugpt3large_based_on_gpt2",
                "rugpt3",
                priority=85,
                name="RuGPT-3 Large",
                tags=["russian", "large", "8k_context", "generation"]
            )
            
            # RuGPT-3 Medium
            self.register_model(
                "ai-forever/rugpt3medium_based_on_gpt2",
                "ai-forever/rugpt3medium_based_on_gpt2", 
                "rugpt3",
                priority=80,
                name="RuGPT-3 Medium",
                tags=["russian", "medium", "generation"]
            )
            
            # DialoGPT для диалогов на русском и английском
            self.register_model(
                "microsoft/DialoGPT-large",
                "microsoft/DialoGPT-large",
                "dialogpt",
                priority=75,
                name="DialoGPT Large Multilingual",
                tags=["multilingual", "dialogue", "8k_context"]
            )
            
            # Стандартная модель для генерации по умолчанию: QWEN2.5 7B Instruct
            # Используем legacy id 'default_text_gen' для совместимости и высокого приоритета
            self.register_model(
                "default_text_gen",
                "Qwen/Qwen2.5-7B-Instruct",
                "qwen",
                priority=100,
                name="QWEN2.5 7B Instruct (default)",
                tags=["default", "qwen", "multilingual", "russian", "32k_context", "instruct"]
            )

            # GPT-2 как запасная базовая модель (низкий приоритет)
            self.register_model(
                "gpt2_base",
                "gpt2",
                "gpt2",
                priority=40,
                name="GPT-2 Base",
                tags=["fallback", "generation"]
            )
            
            logger.info("Добавлены QWEN и русскоязычные модели с поддержкой длинного контекста")
        except Exception as e:
            logger.error(f"Ошибка добавления базовых моделей: {e}", exc_info=True)