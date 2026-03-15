"""
Training Orchestrator for CogniFlex
- Full pipeline to train/update KnowledgeGraph from documents via the core (brain) and MLUnit
- Uses chained tokenization with HybridTokenCache and UnifiedTextProcessor (async)
- Persists progress/checkpoints and auto-resumes from the last successful batch on failure
"""
from __future__ import annotations

import os
import json
import time
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
from ..core.deferred_command_system import CommandPriority
import logging

logger = logging.getLogger("cogniflex.training.orchestrator")


@dataclass
class TrainingProgress:
    document_id: str
    total_chunks: int
    processed_chunks: int
    last_batch_end: int
    last_success_ts: float
    model_id: Optional[str] = None
    pipeline_version: str = "v1"

    @staticmethod
    def load(path: str) -> Optional["TrainingProgress"]:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return TrainingProgress(**data)
        except Exception as e:
            logger.error(f"Ошибка приостановки процесса обучения: {e}")
            return None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


class TrainingOrchestrator:
    """
    Coordinates sequential knowledge graph training across the CogniFlex stack.
    - Integrates: MLUnit (model_manager, token_streamer), KnowledgeGraph, HybridTokenCache
    - Chained tokenization with hybrid cache persistence per-chunk
    - Batch application to KnowledgeGraph with checkpointing
    """

    def __init__(
        self,
        brain: Any,
        cache_dir: Optional[str] = None,
        batch_size: int = 16,
        overlap_tokens: int = 64,
        max_retries: int = 3,
        backoff_sec: float = 2.0,
        pipeline_version: str = "v1",
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_adapt: bool = False,
        mem_high_pct: float = 85.0,
        mem_critical_pct: float = 95.0,
        min_batch_size: int = 1,
        adapt_cooldown_sec: float = 10.0,
    ) -> None:
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cogniflex_cache", "training")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.batch_size = max(1, batch_size)
        self.overlap_tokens = max(0, overlap_tokens)
        self.max_retries = max(0, max_retries)
        self.backoff_sec = max(0.0, backoff_sec)
        self.pipeline_version = pipeline_version
        self.progress_cb = progress_cb
        self.auto_adapt = bool(auto_adapt)
        self.mem_high_pct = float(mem_high_pct)
        self.mem_critical_pct = float(mem_critical_pct)
        self.min_batch_size = max(1, int(min_batch_size))
        self.adapt_cooldown_sec = max(0.0, float(adapt_cooldown_sec))
        self._last_adapt_ts: float = 0.0
        self._last_progress_ts: float = time.time()
        self._paused: bool = False

        # Initialize deferred system reference
        self.deferred_system = getattr(brain, "deferred_system", None)
        
        # Initialize components with None
        self.ml_unit = None
        self.knowledge_graph = None
        self.fractal_tokenizer = None
        self.token_streamer = None
        self.hybrid_cache = None
        
        # Try to get components immediately
        self._try_init_components()
        
        # If we have a deferred system and any critical component is missing,
        # schedule a deferred initialization
        if self.deferred_system and not self._all_components_ready():
            try:
                # Try different methods for different deferred systems
                if hasattr(self.deferred_system, 'add_command'):
                    self.deferred_system.add_command(
                        command=self._deferred_init_components,
                        args=(),
                        kwargs={},
                        priority=CommandPriority.HIGH,
                        max_retries=3,
                        retry_delay=5.0,
                        command_id='training_orchestrator_init_components'
                    )
                elif hasattr(self.deferred_system, 'defer_command'):
                    self.deferred_system.defer_command(
                        self._deferred_init_components,
                        priority='high',
                        name='training_orchestrator_init_components',
                        retries=3,
                        delay=5.0
                    )
                logger.info("Запланирована отложенная инициализация компонентов TrainingOrchestrator")
            except Exception as e:
                logger.warning(f"Не удалось запланировать отложенную инициализацию: {e}")
                # Continue without deferred initialization

        # Optional psutil for memory checks
        try:
            import psutil  # type: ignore
            self._psutil = psutil
        except Exception:
            self._psutil = None

        # Register load shedding callbacks if DeferredCommandSystem is available
        try:
            dsys = self.deferred_system or getattr(self.brain, "deferred_system", None)
            if dsys and hasattr(dsys, "register_load_shed_callback"):
                # Reduce batch size when memory is high
                def _cond_high_mem() -> bool:
                    if not self._psutil:
                        return False
                    try:
                        return float(self._psutil.virtual_memory().percent) >= float(self.mem_high_pct)
                    except Exception:
                        return False

                def _action_reduce_batch() -> None:
                    try:
                        old = self.batch_size
                        new_bs = max(self.min_batch_size, max(1, self.batch_size // 2))
                        if new_bs != old:
                            self.batch_size = new_bs
                            logger.warning(f"Load-shed: batch_size {old} -> {new_bs} due to high memory")
                            self._emit("resource_adjustment", {
                                "old_batch_size": old,
                                "new_batch_size": new_bs,
                                "reason": "load_shed_high_mem",
                            })
                            try:
                                self._emit_metrics([
                                    {
                                        "name": "training.resource_adjustments",
                                        "component": "training_orchestrator",
                                        "type": "counter",
                                        "value": 1.0,
                                    },
                                    {
                                        "name": "training.batch_size",
                                        "component": "training_orchestrator",
                                        "type": "gauge",
                                        "value": float(self.batch_size),
                                        "labels": {"reason": "load_shed_high_mem"},
                                    },
                                ])
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"_action_reduce_batch error: {e}")

                dsys.register_load_shed_callback(
                    _cond_high_mem,
                    _action_reduce_batch,
                    name="training_reduce_batch_on_high_mem",
                    cooldown_sec=max(5.0, float(self.adapt_cooldown_sec)),
                )

                # Pause when memory is critical
                def _cond_critical_mem() -> bool:
                    if not self._psutil:
                        return False
                    try:
                        return float(self._psutil.virtual_memory().percent) >= float(self.mem_critical_pct)
                    except Exception:
                        return False

                def _action_pause_training() -> None:
                    try:
                        if not self._paused:
                            self.pause(reason="critical_memory")
                    except Exception as e:
                        logger.debug(f"_action_pause_training error: {e}")

                dsys.register_load_shed_callback(
                    _cond_critical_mem,
                    _action_pause_training,
                    name="training_pause_on_critical_mem",
                    cooldown_sec=5.0,
                )
        except Exception as e:
            logger.warning(f"Failed to register load shedding callbacks: {e}")
                
    def _try_init_components(self):
        """Пытается инициализировать компоненты синхронно."""
        try:
            # Получаем компоненты из brain, но не сбрасываем уже установленные
            if self.ml_unit is None:
                self.ml_unit = getattr(self.brain, "ml_unit", None)
            if self.knowledge_graph is None:
                self.knowledge_graph = getattr(self.brain, "knowledge_graph", None)
            
            # Дополнительная проверка для KnowledgeGraph через component_initializer
            if not self.knowledge_graph and hasattr(self.brain, 'component_initializer'):
                initializer = getattr(self.brain, 'component_initializer', None)
                if initializer and hasattr(initializer, 'get_component'):
                    self.knowledge_graph = initializer.get_component('knowledge_graph')
                    if self.knowledge_graph:
                        logger.debug("KnowledgeGraph найден через component_initializer")
            
            # Если все еще нет KnowledgeGraph, пробуем создать его напрямую
            if not self.knowledge_graph:
                try:
                    from cogniflex.knowledge.knowledge_graph import KnowledgeGraph
                    self.knowledge_graph = KnowledgeGraph(brain=self.brain)
                    logger.debug("KnowledgeGraph создан напрямую")
                except Exception as e:
                    logger.debug(f"Не удалось создать KnowledgeGraph напрямую: {e}")
            
            # Проверяем наличие фрактального токенизатора
            self.fractal_tokenizer = getattr(self.brain, "fractal_tokenizer", None)
            
            # Ищем токенизатор в разных местах
            if not self.fractal_tokenizer and self.ml_unit:
                # Проверяем token_streamer в ml_unit
                self.token_streamer = getattr(self.ml_unit, "token_streamer", None)
                
                # Проверяем text_processor и его tokenizer
                text_processor = getattr(self.ml_unit, "text_processor", None)
                if text_processor:
                    tokenizer = getattr(text_processor, "tokenizer", None)
                    if tokenizer:
                        self.token_streamer = tokenizer
                        logger.debug("Токенизатор найден в text_processor.tokenizer")
                    else:
                        # Пробуем использовать сам text_processor как токенизатор
                        self.token_streamer = text_processor
                        logger.debug("Используем text_processor как токенизатор")
                
                # Проверяем model_manager через brain (система отложенных команд)
                if hasattr(self.brain, 'components') and 'model_manager' in self.brain.components:
                    model_manager = self.brain.components['model_manager']
                    if model_manager:
                        # Пробуем свойство tokenizer
                        tokenizer = getattr(model_manager, 'tokenizer', None)
                        if tokenizer:
                            self.token_streamer = tokenizer
                            logger.debug("Токенизатор найден в brain.model_manager.tokenizer")
                        else:
                            # Пробуем словарь tokenizers
                            tokenizers = getattr(model_manager, 'tokenizers', None)
                            if tokenizers and isinstance(tokenizers, dict) and tokenizers:
                                first_tokenizer = next(iter(tokenizers.values()))
                                self.token_streamer = first_tokenizer
                                logger.debug("Токенизатор найден в brain.model_manager.tokenizers")
                
                # Дополнительная проверка через прямое обращение к brain.model_manager
                if not self.token_streamer and hasattr(self.brain, 'model_manager'):
                    model_manager = getattr(self.brain, 'model_manager', None)
                    if model_manager:
                        tokenizer = getattr(model_manager, 'tokenizer', None)
                        if tokenizer:
                            self.token_streamer = tokenizer
                            logger.debug("Токенизатор найден в brain.model_manager (прямое обращение)")
                        else:
                            tokenizers = getattr(model_manager, 'tokenizers', None)
                            if tokenizers and isinstance(tokenizers, dict) and tokenizers:
                                first_tokenizer = next(iter(tokenizers.values()))
                                self.token_streamer = first_tokenizer
                                logger.debug("Токенизатор найден в brain.model_manager.tokenizers (прямое обращение)")
            
        except Exception as e:
            logger.error(f"Ошибка в _try_init_components: {e}")
    
    def _find_tokenizer_dynamically(self):
        """Динамический поиск токенизатора - можно вызывать после инициализации других компонентов"""
        try:
            logger.debug("Динамический поиск токенизатора...")
            
            # Проверяем brain.components
            if hasattr(self.brain, 'components'):
                model_manager = self.brain.components.get('model_manager')
                if model_manager:
                    tokenizer = getattr(model_manager, 'tokenizer', None)
                    if tokenizer:
                        self.token_streamer = tokenizer
                        logger.info("Токенизатор найден динамически в brain.model_manager.tokenizer")
                        return True
                    
                    tokenizers = getattr(model_manager, 'tokenizers', None)
                    if tokenizers and isinstance(tokenizers, dict) and tokenizers:
                        first_tokenizer = next(iter(tokenizers.values()))
                        self.token_streamer = first_tokenizer
                        logger.info("Токенизатор найден динамически в brain.model_manager.tokenizers")
                        return True
            
            # Проверяем text_processor
            if hasattr(self.brain, 'components') and 'text_processor' in self.brain.components:
                text_processor = self.brain.components['text_processor']
                if text_processor:
                    tokenizer = getattr(text_processor, 'tokenizer', None)
                    if tokenizer:
                        self.token_streamer = tokenizer
                        logger.info("Токенизатор найден динамически в text_processor.tokenizer")
                        return True
                    else:
                        self.token_streamer = text_processor
                        logger.info("Используем text_processor как токенизатор (динамически)")
                        return True
            
            logger.debug("Токенизатор не найден при динамическом поиске")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка динамического поиска токенизатора: {e}")
            return False
    
    def _find_hybrid_cache_dynamically(self):
        """Динамический поиск гибридного кэша"""
        try:
            # Ищем гибридный кэш в разных местах
            if self.hybrid_cache is None and self.ml_unit:
                self.hybrid_cache = getattr(self.ml_unit, "hybrid_cache", None)
                if self.hybrid_cache:
                    logger.debug("Гибридный кэш найден в ml_unit")
            
            if self.hybrid_cache is None and hasattr(self.brain, 'memory_manager'):
                memory_manager = getattr(self.brain, 'memory_manager', None)
                if memory_manager:
                    self.hybrid_cache = getattr(memory_manager, 'hybrid_cache', None)
                    if self.hybrid_cache:
                        logger.debug("Гибридный кэш найден в memory_manager")
            
            # Дополнительные источники гибридного кэша
            if self.hybrid_cache is None and hasattr(self.brain, 'component_initializer'):
                initializer = getattr(self.brain, 'component_initializer', None)
                if initializer:
                    # Проверяем, есть ли у инициализатора доступ к кэшу
                    if hasattr(initializer, 'get_component'):
                        cache_component = initializer.get_component('hybrid_cache')
                        if cache_component is not None:
                            self.hybrid_cache = cache_component
                            logger.debug("Гибридный кэш найден через component_initializer")
            
            # Проверяем integrated_learning_manager для доступа к кэшу
            if self.hybrid_cache is None and hasattr(self.brain, 'integrated_learning_manager'):
                integrated_learning = getattr(self.brain, 'integrated_learning_manager', None)
                if integrated_learning:
                    self.hybrid_cache = getattr(integrated_learning, 'hybrid_cache', None)
                    if self.hybrid_cache:
                        logger.debug("Гибридный кэш найден в integrated_learning_manager")
            
            # Дополнительные источники токенизатора
            if not self.token_streamer and hasattr(self.brain, 'model_manager'):
                model_manager = getattr(self.brain, 'model_manager', None)
                if model_manager:
                    # Сначала пробуем свойство tokenizer для совместимости
                    tokenizer = getattr(model_manager, 'tokenizer', None)
                    if tokenizer:
                        self.token_streamer = tokenizer
                        logger.debug("Токенизатор найден в brain.model_manager.tokenizer")
                    else:
                        # Пробуем в словаре tokenizers
                        tokenizers = getattr(model_manager, 'tokenizers', None)
                        if tokenizers and isinstance(tokenizers, dict) and tokenizers:
                            # Берем первый доступный токенизатор
                            first_tokenizer = next(iter(tokenizers.values()))
                            self.token_streamer = first_tokenizer
                            logger.debug("Токенизатор найден в brain.model_manager.tokenizers")
            
            # Проверяем integrated_learning_manager для токенизатора
            if not self.token_streamer and hasattr(self.brain, 'integrated_learning_manager'):
                integrated_learning = getattr(self.brain, 'integrated_learning_manager', None)
                if integrated_learning:
                    # Ищем токенизатор через fractal_model_manager
                    fractal_manager = getattr(integrated_learning, 'fractal_model_manager', None)
                    if fractal_manager:
                        tokenizer = getattr(fractal_manager, 'tokenizer', None)
                        if tokenizer:
                            self.token_streamer = tokenizer
                            logger.debug("Токенизатор найден через integrated_learning_manager")
            
            # Логируем предупреждения, если компоненты недоступны
            has_tokenizer = bool(self.token_streamer or self.fractal_tokenizer)
            has_cache = self.hybrid_cache is not None
            
            if not has_tokenizer:
                logger.warning("Токенизатор недоступен. Обучение будет ограничено.")
                logger.info("Доступные источники токенизатора проверены, но не найдены")
            else:
                logger.info("Токенизатор доступен для обучения")
            
            if not has_cache:
                logger.warning("Гибридный кэш недоступен. Кэширование будет отключено.")
                logger.info("Работаем без кэширования эмбеддингов")
            else:
                logger.info("Гибридный кэш доступен для обучения")
            
            if not self.knowledge_graph:
                logger.debug("KnowledgeGraph недоступен. Обучение будет работать в режиме только извлечения.")
            else:
                logger.debug("KnowledgeGraph доступен для обучения")
            
            if has_tokenizer and has_cache:
                logger.info("Все основные компоненты для обучения доступны")
            elif has_tokenizer:
                logger.info("Базовые компоненты для обучения доступны (без кэша)")
                
        except Exception as e:
            logger.error(f"Ошибка при инициализации компонентов: {e}", exc_info=True)
    
    def _deferred_init_components(self):
        """Отложенная инициализация компонентов с повторными попытками."""
        try:
            self._try_init_components()
            
            if not self._all_components_ready():
                raise RuntimeError("Не все компоненты инициализированы")
                
            logger.info("Все компоненты TrainingOrchestrator инициализированы")
            return True
            
        except Exception as e:
            logger.warning(f"Ошибка при отложенной инициализации компонентов: {e}")
            
            # Планируем повторную попытку, если доступна система отложенных команд
            if self.deferred_system and hasattr(self.deferred_system, 'defer_command'):
                self.deferred_system.defer_command(
                    self._deferred_init_components,
                    priority='high',
                    name='retry_training_orchestrator_init',
                    delay=5.0,  # Задержка перед повторной попыткой
                    retries=2   # Максимальное количество повторных попыток
                )
                logger.info("Запланирована повторная попытка инициализации компонентов")
            
            return False
    
    def _all_components_ready(self):
        """Проверяет, все ли критические компоненты инициализированы."""
        # Проверяем наличие хотя бы одного токенизатора и гибридного кэша
        has_tokenizer = bool(self.fractal_tokenizer or 
                           (self.ml_unit and hasattr(self.ml_unit, 'token_streamer')))
                           
        has_hybrid_cache = self.hybrid_cache is not None or \
                              (self.ml_unit and hasattr(self.ml_unit, 'hybrid_cache'))
        
        return has_tokenizer and has_hybrid_cache

    # ----------------------------
    # Readiness checks
    # ----------------------------
    def _can_train_now(self) -> bool:
        """Returns True if ML model or fractal storage is loaded."""
        try:
            # Получаем актуальный brain (может измениться после инициализации)
            brain = getattr(self, 'brain', None)
            
            # Если brain None, пробуем получить из ml_unit
            if brain is None and self.ml_unit is not None:
                brain = getattr(self.ml_unit, 'brain', None)
                if brain is not None:
                    self.brain = brain  # Сохраняем для будущих вызовов
                    logger.info(f"[_can_train_now] Brain recovered from ml_unit: {id(brain)}")
            
            # Если всё еще None, пробуем через core_brain
            if brain is None:
                try:
                    from cogniflex.core.core_brain import CoreBrain
                    # Проверяем есть ли глобальный экземпляр
                    import cogniflex.core.core_brain as core_module
                    if hasattr(core_module, '_global_brain_instance'):
                        brain = core_module._global_brain_instance
                        logger.info(f"[_can_train_now] Brain recovered from global instance: {id(brain) if brain else None}")
                except Exception:
                    pass
            
            if brain is None:
                logger.warning("[_can_train_now] brain is None, cannot check training readiness")
                return False
            
            # Обновляем компоненты перед проверкой
            self._try_init_components()
            
            # Check models via brain flags (models_ready or fractal_ready) or model_manager contents
            models_ready_flag = bool(getattr(brain, 'models_ready', False))
            fractal_ready_flag = bool(getattr(brain, 'fractal_ready', False))
            
            # Проверяем наличие токенизатора как критически важного компонента
            has_tokenizer = bool(self.fractal_tokenizer or 
                               (self.ml_unit and hasattr(self.ml_unit, 'token_streamer')) or
                               (self.ml_unit and hasattr(self.ml_unit, 'text_processor') and 
                                self.ml_unit.text_processor and hasattr(self.ml_unit.text_processor, 'tokenizer')))
            
            # Проверяем гибридный кэш
            cache_ready = self.hybrid_cache is not None or \
                              (self.ml_unit and hasattr(self.ml_unit, 'hybrid_cache')) or \
                              (brain and hasattr(brain, 'memory_manager') and 
                               brain.memory_manager and hasattr(brain.memory_manager, 'hybrid_cache'))
            
            # Debug logging
            logger.info(f"[_can_train_now] models_ready={models_ready_flag}, fractal_ready={fractal_ready_flag}, has_tokenizer={has_tokenizer}, cache_ready={cache_ready}")
            
            # Если есть токенизатор и кэш, разрешаем обучение даже без загруженных моделей
            if has_tokenizer and cache_ready:
                logger.info(f"[_can_train_now] Training allowed via tokenizer+cache")
                return True
            
            # Проверяем model_manager как fallback
            if not (models_ready_flag or fractal_ready_flag):
                mm = None
                try:
                    mm = getattr(self.ml_unit, 'model_manager', None) if self.ml_unit else None
                    if mm is None:
                        mm = getattr(brain, 'model_manager', None)
                except Exception:
                    mm = None
                try:
                    if mm and hasattr(mm, 'models') and isinstance(mm.models, dict) and len(mm.models) > 0:
                        models_ready_flag = True
                        logger.info(f"[_can_train_now] Models detected via model_manager")
                    elif mm and hasattr(mm, 'get_available_models'):
                        models = mm.get_available_models()
                        if models and len(models) > 0:
                            models_ready_flag = True
                            logger.info(f"[_can_train_now] Models detected via get_available_models")
                except Exception:
                    pass
            
            result = bool(has_tokenizer and cache_ready and (models_ready_flag or fractal_ready_flag))
            logger.info(f"[_can_train_now] Result: {result} (tokenizer={has_tokenizer}, cache={cache_ready}, models={models_ready_flag}, fractal={fractal_ready_flag})")
            return result
        except Exception as e:
            logger.error(f"[_can_train_now] Exception: {e}", exc_info=True)
            return False

    # ----------------------------
    # Public API
    # ----------------------------
    
    def start_training_session(
        self,
        model_id: Optional[str] = None,
        dataset_path: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        use_fractal: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Запускает сессию обучения (метод для совместимости с GUI).
        
        Args:
            model_id: ID модели для обучения
            dataset_path: Путь к датасету
            epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Скорость обучения
            use_fractal: Использовать фрактальное обучение
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результат запуска обучения
        """
        try:
            logger.info(f"Запуск сессии обучения: model_id={model_id}, epochs={epochs}")
            
            # Создаем фиктивный документ для обучения
            class MockDocument:
                def __init__(self, path):
                    self.path = path
                    self.id = os.path.basename(path)
                
                def iter_segments(self):
                    # Если указан путь к датасету, читаем его
                    if dataset_path and os.path.exists(dataset_path):
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if line.strip():
                                    yield {
                                        'text': line.strip(),
                                        'metadata': {
                                            'source': 'training_dataset',
                                            'line_number': line_num
                                        }
                                    }
                    else:
                        # Иначе используем тестовые данные
                        yield {
                            'text': "Тестовый текст для обучения модели.",
                            'metadata': {'source': 'mock_data'}
                        }
                
                @property
                def metadata(self):
                    return {
                        'id': self.id,
                        'source': 'training_session',
                        'model_id': model_id
                    }
            
            # Создаем mock документ
            mock_doc = MockDocument(dataset_path or "mock_training.txt")
            
            # Запускаем обучение через существующий метод
            result = self.train_from_document(
                imported_doc=mock_doc,
                model_id=model_id,
                use_fractal=use_fractal,
                fractal_config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    **kwargs
                }
            )
            
            logger.info(f"Сессия обучения запущена: {result.get('status', 'unknown')}")
            return result
            
        except Exception as e:
            error_msg = f"Ошибка запуска сессии обучения: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                'status': 'error',
                'error': error_msg,
                'session_id': None
            }
    
    def train_from_document(
        self, 
        imported_doc: Any, 
        model_id: Optional[str] = None,
        use_fractal: bool = False,
        fractal_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trains the knowledge graph from a single imported document object.
        
        Args:
            imported_doc: Document object with iter_segments() method
            model_id: ID of the model to use for training
            use_fractal: Whether to use fractal transformer for training
            fractal_config: Configuration for fractal training
            
        Document object should provide:
        - id: stable id or filename
        - iter_segments(): yields normalized text segments
        - metadata: dict with document metadata
        """
        doc_id = self._infer_doc_id(imported_doc)
        self._emit("started", {
            "document_id": doc_id, 
            "model_id": model_id,
            "use_fractal": use_fractal
        })

        # Принудительно обновляем ссылку на brain перед проверкой
        # Это важно когда fractal_ready устанавливается после создания TrainingOrchestrator
        if self.brain is None:
            # Пробуем получить brain из ml_unit или из глобального контекста
            if self.ml_unit and hasattr(self.ml_unit, 'brain'):
                self.brain = self.ml_unit.brain
                logger.info(f"[train_from_document] Brain recovered from ml_unit: {id(self.brain) if self.brain else None}")
        
        # Gate training until ML models are loaded and hybrid cache is ready
        if not self._can_train_now():
            reason = "models_not_ready_or_cache_unavailable"
            logger.warning("Training deferred: models not ready or hybrid cache unavailable")
            self._emit("deferred", {"reason": reason, "model_id": model_id})
            # Metrics: training deferred
            try:
                self._emit_metrics([
                    {"name": "training.deferred", "component": "training_orchestrator", "type": "counter", "value": 1.0,
                     "labels": {"reason": reason}},
                ])
            except Exception:
                pass
            return {"status": "deferred", "reason": reason}

        # Clear contradictions before starting training to avoid stale states
        try:
            brain = getattr(self, "brain", None)
            if brain and hasattr(brain, "clear_all_contradictions"):
                report = brain.clear_all_contradictions()
                logger.info(
                    f"Training pre-clean: contradictions cleared ok={report.get('ok')}, cleared={report.get('cleared')}"
                )
                try:
                    self._emit_metrics([
                        {
                            "name": "training.contradictions_cleared_pre",
                            "component": "training_orchestrator",
                            "type": "counter",
                            "value": float(report.get("cleared", 0) or 0),
                        }
                    ])
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Training pre-clean: failed to clear contradictions: {e}")

        # Step 1: Initialize progress tracking
        progress_path = self._progress_path(doc_id)
        progress = TrainingProgress.load(progress_path)
        
        # Получаем сегменты документа
        segments = list(imported_doc.iter_segments())
        
        if progress is None or progress.pipeline_version != self.pipeline_version:
            # New document or pipeline version changed
            total_chunks = len(segments)
            progress = TrainingProgress(
                document_id=doc_id,
                total_chunks=total_chunks,
                processed_chunks=0,
                last_batch_end=0,
                last_success_ts=time.time(),
                model_id=model_id,
                pipeline_version=self.pipeline_version,
            )
            progress.save(progress_path)
            self._emit("progress_initialized", {
                "document_id": doc_id,
                "total_chunks": total_chunks,
                "model_id": model_id,
                "use_fractal": use_fractal,
            })
        else:
            # Resume existing progress
            total_chunks = len(segments)
            logger.info(f"Resuming training for document '{doc_id}' from chunk {progress.last_batch_end}")

        # Step 2: Check if we should use fractal training
        if use_fractal:
            return self._train_with_fractal(
                doc_id=doc_id,
                imported_doc=imported_doc,
                model_id=model_id,
                progress=progress,
                fractal_config=fractal_config or {}
            )
            
        # Step 3: Process in batches with checkpointing (legacy training)
        start_idx = progress.last_batch_end
        total_chunks = progress.total_chunks
        logger.info(f"Training document '{doc_id}' with {total_chunks} chunks")

        # Emit start event
        self._emit("start", {
            "document_id": doc_id,
            "model_id": model_id,
            "total_chunks": total_chunks,
            "resume_from": start_idx,
        })
        # Metrics: document started
        try:
            self._emit_metrics([
                {
                    "name": "training.documents_started",
                    "component": "training_orchestrator",
                    "type": "counter",
                    "value": 1.0,
                    "labels": {"pipeline": self.pipeline_version or "v1"},
                },
                {
                    "name": "training.total_chunks",
                    "component": "training_orchestrator",
                    "type": "gauge",
                    "value": float(total_chunks),
                },
                {
                    "name": "training.resume_from",
                    "component": "training_orchestrator",
                    "type": "gauge",
                    "value": float(start_idx),
                },
            ])
        except Exception:
            pass

        # Enter training mode to prevent heavy model loads in MLUnit
        self._enter_training_mode()
        try:
            # Main loop in batches with retries
            while start_idx < total_chunks:
                # Respect pause signal (e.g., from load shedding)
                if self._paused:
                    try:
                        self._emit("paused", {"document_id": doc_id, "reason": getattr(self, "_pause_reason", None)})
                    except Exception:
                        pass
                    time.sleep(0.25)
                    # allow resume
                    continue
                # Optionally adapt before building batch
                self._check_and_adapt_resources()
                end_idx = min(total_chunks, start_idx + self.batch_size)
                batch = segments[start_idx:end_idx]

                for attempt in range(self.max_retries + 1):
                    try:
                        self._emit("batch_start", {
                            "document_id": doc_id,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "attempt": attempt + 1,
                            "max_attempts": self.max_retries + 1,
                        })
                        # Metrics: batch start and current batch size
                        try:
                            self._emit_metrics([
                                {
                                    "name": "training.batches_started",
                                    "component": "training_orchestrator",
                                    "type": "counter",
                                    "value": 1.0,
                                    "labels": {"attempt": str(attempt + 1)},
                                },
                                {
                                    "name": "training.batch_size",
                                    "component": "training_orchestrator",
                                    "type": "gauge",
                                    "value": float(self.batch_size),
                                },
                            ])
                        except Exception:
                            pass
                        self._process_batch(doc_id, batch, start_idx, model_id)

                        # Checkpoint after successful batch apply
                        progress.processed_chunks = end_idx
                        progress.last_batch_end = end_idx
                        progress.last_success_ts = time.time()
                        self._last_progress_ts = progress.last_success_ts
                        progress.save(progress_path)
                        logger.info(f"Checkpoint saved at chunk {end_idx}/{total_chunks} for '{doc_id}'")
                        self._emit("batch_end", {
                            "document_id": doc_id,
                            "end_idx": end_idx,
                            "total_chunks": total_chunks,
                            "processed_chunks": progress.processed_chunks,
                        })
                        # Metrics: batch completed and chunks processed in this batch
                        try:
                            self._emit_metrics([
                                {
                                    "name": "training.batches_completed",
                                    "component": "training_orchestrator",
                                    "type": "counter",
                                    "value": 1.0,
                                },
                                {
                                    "name": "training.chunks_processed",
                                    "component": "training_orchestrator",
                                    "type": "counter",
                                    "value": float(end_idx - start_idx),
                                },
                            ])
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        logger.error(
                            f"Batch {start_idx}-{end_idx} failed (attempt {attempt+1}/{self.max_retries+1}): {e}",
                            exc_info=True,
                        )
                        if attempt < self.max_retries:
                            self._emit("batch_retry", {
                                "document_id": doc_id,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "attempt": attempt + 1,
                                "error": str(e),
                            })
                            # Metrics: batch retry
                            try:
                                self._emit_metrics([
                                    {
                                        "name": "training.batch_retries",
                                        "component": "training_orchestrator",
                                        "type": "counter",
                                        "value": 1.0,
                                        "labels": {"attempt": str(attempt + 1)},
                                    }
                                ])
                            except Exception:
                                pass
                            time.sleep(self.backoff_sec * (2 ** attempt))
                            continue
                        else:
                            # Fatal for this batch — stop and return status, can resume later
                            result = {
                                "status": "failed",
                                "document_id": doc_id,
                                "processed_chunks": progress.processed_chunks,
                                "error": str(e),
                            }
                            self._emit("failed", result)
                            # Metrics: document failed
                            try:
                                self._emit_metrics([
                                    {
                                        "name": "training.documents_failed",
                                        "component": "training_orchestrator",
                                        "type": "counter",
                                        "value": 1.0,
                                    },
                                    {
                                        "name": "training.processed_chunks",
                                        "component": "training_orchestrator",
                                        "type": "gauge",
                                        "value": float(progress.processed_chunks),
                                    },
                                ])
                            except Exception:
                                pass
                            return result

                start_idx = end_idx
        finally:
            # Always exit training mode
            self._exit_training_mode()

        result = {
            "status": "completed",
            "document_id": doc_id,
            "processed_chunks": total_chunks,
            "total_chunks": total_chunks,
        }
        self._emit("completed", result)
        # Metrics: document completed
        try:
            self._emit_metrics([
                {
                    "name": "training.documents_completed",
                    "component": "training_orchestrator",
                    "type": "counter",
                    "value": 1.0,
                },
                {
                    "name": "training.processed_chunks",
                    "component": "training_orchestrator",
                    "type": "gauge",
                    "value": float(total_chunks),
                },
            ])
        except Exception:
            pass
        return result

    def _train_with_fractal(
        self, 
        doc_id: str,
        imported_doc: Any,
        model_id: Optional[str],
        progress: TrainingProgress,
        fractal_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train using the FractalKnowledgeTrainer.
        
        Args:
            doc_id: Document ID
            imported_doc: Document object with iter_segments()
            model_id: Model ID to use
            progress: Training progress tracker
            fractal_config: Configuration for fractal training
            
        Returns:
            Training result dictionary
        """
        try:
            # Import here to avoid circular imports
            from .fractal_trainer import FractalKnowledgeTrainer
            from .fractal_transformer import FractalConfig
            
            # Get model and tokenizer
            model_manager = getattr(self.brain, 'model_manager', None)
            if model_manager is None:
                raise ValueError("ModelManager not available in brain")
                
            # Load or create fractal model
            model, tokenizer, _ = model_manager.get_model_for_task(
                'fractal-text-generation',
                model_id=model_id,
                **fractal_config.get('model_kwargs', {})
            )
            
            # Initialize trainer
            trainer = FractalKnowledgeTrainer(
                model=model,
                tokenizer=tokenizer,
                config=fractal_config.get('trainer_config', {})
            )
            
            # Prepare training data
            train_dataset = self._prepare_fractal_training_data(imported_doc)
            
            # Start training
            self._emit("fractal_training_started", {
                "document_id": doc_id,
                "model_id": model_id,
                "num_examples": len(train_dataset),
                **fractal_config
            })
            
            # Train the model
            metrics = trainer.train(train_dataset)
            
            # Save the trained model
            if fractal_config.get('save_model', True):
                output_dir = fractal_config.get('output_dir', f"./models/{model_id}")
                trainer.save_pretrained(output_dir)
                
                # Update model in the model manager
                if hasattr(model_manager, 'update_model'):
                    model_manager.update_model(
                        model_id=model_id,
                        model=model,
                        tokenizer=tokenizer,
                        config=fractal_config
                    )
            
            # Update progress
            progress.processed_chunks = progress.total_chunks
            progress.last_success_ts = time.time()
            progress.save(self._progress_path(doc_id))
            
            # Emit completion event
            result = {
                "status": "completed",
                "document_id": doc_id,
                "metrics": metrics,
                "use_fractal": True,
                **fractal_config
            }
            self._emit("fractal_training_completed", result)
            return result
            
        except Exception as e:
            error_msg = f"Fractal training failed: {str(e)}"
            logger.exception(error_msg)
            result = {
                "status": "failed",
                "document_id": doc_id,
                "error": error_msg,
                "use_fractal": True,
                **fractal_config
            }
            self._emit("fractal_training_failed", result)
            return result
    
    def _prepare_fractal_training_data(self, imported_doc: Any) -> List[Dict[str, Any]]:
        """
        Prepare training data for fractal training.
        
        Args:
            imported_doc: Document object with iter_segments()
            
        Returns:
            List of training examples
        """
        examples = []
        
        # Process each segment in the document
        for segment in imported_doc.iter_segments():
            # Here you would typically preprocess the segment and extract
            # input-output pairs for training. This is a simplified example.
            example = {
                'text': segment.text,
                'metadata': getattr(segment, 'metadata', {}),
                'fractal_path': getattr(segment, 'fractal_path', []),
            }
            examples.append(example)
            
            # Add augmented examples if needed
            if hasattr(imported_doc, 'augment_segment'):
                augmented = imported_doc.augment_segment(segment)
                if augmented:
                    examples.extend([{
                        'text': aug_text,
                        'metadata': getattr(segment, 'metadata', {}),
                        'fractal_path': getattr(segment, 'fractal_path', []),
                    } for aug_text in augmented])
        
        return examples
    
    # ----------------------------
    # Internals
    # ----------------------------
    def _process_batch(self, doc_id: str, batch: List[str], offset: int, model_id: Optional[str]) -> None:
        """
        Processes a batch of text segments:
        - Chained tokenization with hybrid cache
        - Entity/relation extraction via MLUnit/Core
        - Transactional apply to KnowledgeGraph
        """
        # Step 1: tokenize with caching (async if available)
        tokenized_list: List[Dict[str, Any]] = []
        for idx, text in enumerate(batch):
            cache_key = self._cache_key(doc_id, offset + idx, text, model_id)
            cached = self._cache_get(cache_key)
            if cached is not None:
                tokenized_list.append(cached)
                # Emit fine-grained progress when served from cache
                self._emit("batch_progress", {
                    "document_id": doc_id,
                    "processed_chunks": offset + idx + 1,
                    "total_chunks": None  # GUI will fallback to stored total
                })
                self._last_progress_ts = time.time()
                # Metrics: chunk progressed (cache hit)
                try:
                    self._emit_metrics([
                        {
                            "name": "training.chunks_processed",
                            "component": "training_orchestrator",
                            "type": "counter",
                            "value": 1.0,
                            "labels": {"source": "cache"},
                        }
                    ])
                except Exception:
                    pass
                continue

            if self.token_streamer and hasattr(self.token_streamer, "tokenize_async"):
                try:
                    # tokenize one-by-one to keep per-chunk cache granularity
                    result = self.token_streamer.tokenize_async([text])[0]
                except Exception as e:
                    logger.debug(f"tokenize_async failed, fallback to split: {e}")
                    result = {"tokens": text.split(), "token_count": len(text.split())}
            else:
                result = {"tokens": text.split(), "token_count": len(text.split())}

            # Save into hybrid cache immediately
            self._cache_put(cache_key, result)
            tokenized_list.append(result)
            # Emit fine-grained progress after processing each chunk
            self._emit("batch_progress", {
                "document_id": doc_id,
                "processed_chunks": offset + idx + 1,
                "total_chunks": None
            })
            self._last_progress_ts = time.time()
            # Metrics: chunk progressed (processed)
            try:
                self._emit_metrics([
                    {
                        "name": "training.chunks_processed",
                        "component": "training_orchestrator",
                        "type": "counter",
                        "value": 1.0,
                        "labels": {"source": "processed"},
                    }
                ])
            except Exception:
                pass

        # Step 2: extract entities/relations using ML pipeline
        extracted: List[Dict[str, Any]] = self._extract_knowledge(batch, tokenized_list, model_id)

        # Step 3: transactional apply to KnowledgeGraph
        self._apply_to_knowledge_graph(extracted, doc_id, offset)

    def _extract_knowledge(
        self, texts: List[str], tokenized_list: List[Dict[str, Any]], model_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Placeholder extraction method using MLUnit/text_processor if available.
        Output format per item:
        {
            "nodes": [{"name": str, "description": str, "node_type": str, "domain": str, "strength": float}],
            "edges": [{"source": str, "target": str, "type": str, "strength": float}]
        }
        """
        results: List[Dict[str, Any]] = []
        for text, tok in zip(texts, tokenized_list):
            try:
                # Prefer brain.ml_unit.text_processor if it provides advanced analysis
                tp = getattr(self.ml_unit, "text_processor", None) if self.ml_unit else None
                if tp and hasattr(tp, "extract_entities_relations"):
                    analysis = tp.extract_entities_relations(text, tokens=tok.get("tokens"))
                    results.append(analysis or {"nodes": [], "edges": []})
                else:
                    # Minimal heuristic fallback
                    top_words = [w for w in (tok.get("tokens") or []) if len(w) > 3][:5]
                    nodes = [
                        {"name": w, "description": f"Concept: {w}", "node_type": "concept", "domain": "general", "strength": 0.6}
                        for w in top_words
                    ]
                    results.append({"nodes": nodes, "edges": []})
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                results.append({"nodes": [], "edges": []})
        return results

    def _apply_to_knowledge_graph(self, extracted: List[Dict[str, Any]], doc_id: str, offset: int) -> None:
        if not self.knowledge_graph:
            logger.warning("KnowledgeGraph not available; skipping apply")
            return
        # Apply in a simple transactional manner — if any add fails, raise to trigger retry
        for i, item in enumerate(extracted):
            nodes = item.get("nodes", []) or []
            edges = item.get("edges", []) or []
            node_ids: List[str] = []
            for n in nodes:
                node_id = self.knowledge_graph.add_node(
                    name=n.get("name", ""),
                    description=n.get("description", ""),
                    node_type=n.get("node_type", "concept"),
                    domain=n.get("domain", "general"),
                    strength=float(n.get("strength", 0.5)),
                    meta={"doc_id": doc_id, "chunk_index": offset + i},
                )
                if node_id:
                    node_ids.append(node_id)
            for e in edges:
                src = e.get("source")
                dst = e.get("target")
                rel = e.get("type", "related_to")
                if not (src and dst):
                    continue
                self.knowledge_graph.add_edge(
                    src, dst, rel,
                    strength=float(e.get("strength", 0.5)),
                    meta={"doc_id": doc_id}
                )

    # ----------------------------
    # Utilities
    # ----------------------------
    def _progress_path(self, doc_id: str) -> str:
        safe = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"progress_{safe}.json")

    def _infer_doc_id(self, imported_doc: Any) -> str:
        name = getattr(imported_doc, "source_path", None) or getattr(imported_doc, "title", None) or "document"
        return str(name)

    def _cache_key(self, doc_id: str, chunk_idx: int, text: str, model_id: Optional[str]) -> str:
        key_raw = f"{doc_id}|{chunk_idx}|{model_id or 'auto'}|{self.pipeline_version}|{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        return hashlib.md5(key_raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            cache = self.hybrid_cache
            if not cache:
                return None
            return cache.disk_cache.get(key) or cache.get_token(key)
        except Exception:
            return None

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        try:
            cache = self.hybrid_cache
            if not cache:
                return
            # Store both in disk (stable) and memory where possible
            cache._save_token_to_disk(key, value)
            if len(cache.memory_cache) < cache.max_memory_tokens:
                cache.memory_cache.put(key, value)
        except Exception as e:
            logger.debug(f"Cache put failed: {e}")

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """
        Safely emit progress events to the provided callback for GUI/CLI integration.
        The callback receives a dict like: {"event": <event>, ...payload}
        """
        if not self.progress_cb:
            return
        try:
            payload = {"event": event}
            payload.update(data or {})
            self.progress_cb(payload)
        except Exception as e:
            logger.debug(f"Progress callback failed: {e}")

    def _emit_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Safely forwards normalized metrics via event bus ('metrics') and direct emit for compatibility."""
        try:
            brain = getattr(self, "brain", None)
            if not brain:
                return
            emitted = False
            # Try event bus first
            try:
                if hasattr(brain, "events") and getattr(brain, "events", None):
                    brain.events.trigger('metrics', metrics)
                    emitted = True
            except Exception:
                emitted = False
            # Fallback to direct emit only if event failed/unavailable
            if not emitted:
                try:
                    emit_fn = getattr(brain, "emit_metrics", None)
                    if callable(emit_fn):
                        emit_fn(metrics)
                except Exception:
                    pass
        except Exception:
            # Never raise from metrics
            pass

    def _enter_training_mode(self) -> None:
        """Set brain flag and env var to signal training mode to other subsystems (e.g., MLUnit)."""
        try:
            import os as _os
            # Track previous env to restore later
            self._prev_training_env = _os.environ.get("COGNIFLEX_TRAINING")
            _os.environ["COGNIFLEX_TRAINING"] = "1"
        except Exception:
            self._prev_training_env = None
        try:
            if self.brain is not None:
                # Preserve previous flags
                self._prev_brain_in_training = (
                    getattr(self.brain, "_in_training", None),
                    getattr(self.brain, "in_training", None),
                )
                setattr(self.brain, "_in_training", True)
                setattr(self.brain, "in_training", True)
        except Exception:
            self._prev_brain_in_training = (None, None)

    def _exit_training_mode(self) -> None:
        """Restore brain flag and env var after training completes."""
        try:
            import os as _os
            if getattr(self, "_prev_training_env", None) is None:
                # Was unset before
                _os.environ.pop("COGNIFLEX_TRAINING", None)
            else:
                _os.environ["COGNIFLEX_TRAINING"] = self._prev_training_env  # type: ignore
        except Exception:
            pass
        try:
            prev = getattr(self, "_prev_brain_in_training", (None, None))
            if self.brain is not None:
                if prev[0] is None:
                    # Was not set
                    if hasattr(self.brain, "_in_training"):
                        delattr(self.brain, "_in_training")
                else:
                    setattr(self.brain, "_in_training", prev[0])
                if prev[1] is None:
                    if hasattr(self.brain, "in_training"):
                        delattr(self.brain, "in_training")
                else:
                    setattr(self.brain, "in_training", prev[1])
        except Exception:
            pass

    def _check_and_adapt_resources(self) -> None:
        """
        If auto_adapt is enabled and psutil is available, reduce batch size when memory is high.
        - When memory >= mem_critical_pct: set batch_size to min_batch_size immediately
        - When memory >= mem_high_pct: decay batch_size by half (>= min)
        Uses cooldown to avoid flapping.
        """
        if not self.auto_adapt or not self._psutil:
            return
        now = time.time()
        if now - self._last_adapt_ts < self.adapt_cooldown_sec:
            return
        try:
            mem = self._psutil.virtual_memory()
            mem_pct = float(mem.percent)
        except Exception:
            return
        new_bs = self.batch_size
        reason = None
        if mem_pct >= self.mem_critical_pct:
            new_bs = self.min_batch_size
            reason = f"critical_mem={mem_pct:.1f}%"
        elif mem_pct >= self.mem_high_pct:
            new_bs = max(self.min_batch_size, max(1, self.batch_size // 2))
            reason = f"high_mem={mem_pct:.1f}%"
        if new_bs != self.batch_size and reason:
            old = self.batch_size
            self.batch_size = new_bs
            self._last_adapt_ts = now
            logger.warning(f"Auto-adapt: batch_size {old} -> {self.batch_size} due to {reason}")
            self._emit("resource_adjustment", {
                "old_batch_size": old,
                "new_batch_size": self.batch_size,
                "reason": reason,
            })
            # Metrics: resource adjustment
            try:
                self._emit_metrics([
                    {
                        "name": "training.resource_adjustments",
                        "component": "training_orchestrator",
                        "type": "counter",
                        "value": 1.0,
                    },
                    {
                        "name": "training.batch_size",
                        "component": "training_orchestrator",
                        "type": "gauge",
                        "value": float(self.batch_size),
                        "labels": {"reason": reason},
                    },
                ])
            except Exception:
                pass

    # ----------------------------
    # Pause/Resume Controls
    # ----------------------------
    def pause(self, reason: Optional[str] = None) -> None:
        try:
            self._paused = True
            self._pause_reason = reason or "manual"
            logger.warning(f"Training paused (reason={self._pause_reason})")
            self._emit("paused", {"reason": self._pause_reason})
            try:
                self._emit_metrics([
                    {
                        "name": "training.pauses",
                        "component": "training_orchestrator",
                        "type": "counter",
                        "value": 1.0,
                        "labels": {"reason": self._pause_reason},
                    }
                ])
            except Exception:
                pass
        except Exception:
            pass

    def resume(self) -> None:
        try:
            was_paused = self._paused
            self._paused = False
            self._pause_reason = None
            if was_paused:
                logger.warning("Training resumed")
                self._emit("resumed", {})
                try:
                    self._emit_metrics([
                        {
                            "name": "training.resumes",
                            "component": "training_orchestrator",
                            "type": "counter",
                            "value": 1.0,
                        }
                    ])
                except Exception:
                    pass
        except Exception:
            pass
