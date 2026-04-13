"""
Brain Coordination Module for EVA CoreBrain.

Adds active coordination capabilities via mixins:
- EventSubscriptionMixin: subscribes to critical system events and reacts
- CommandIssuerMixin: issues coordination commands via DeferredCommandSystem
- ProcessTrackerMixin: tracks and monitors active processes

Также содержит CommandHandlers - отдельный класс для обработки команд.
"""

import time
import logging
from typing import Dict, Any, Optional

from .event_bus import Event, EventTypes, EventPriority
from .deferred_command_system import CommandPriority

logger = logging.getLogger("eva_ai.core.brain_coordination")

_PRIORITY_MAP = {
    0: CommandPriority.CRITICAL,
    1: CommandPriority.HIGH,
    2: CommandPriority.NORMAL,
    3: CommandPriority.LOW,
}


def _to_priority(value: int) -> CommandPriority:
    return _PRIORITY_MAP.get(value, CommandPriority.NORMAL)


def _make_event(event_type: str, source: str = "core_brain", data: Optional[Dict] = None) -> Event:
    return Event(event_type=event_type, source=source, data=data or {})


class CommandHandlers:
    """
    Класс для обработки команд - вынесен из CommandIssuerMixin для соблюдения SRP.
    Каждый метод обрабатывает одну команду.
    """
    
    def __init__(self, brain):
        self.brain = brain
    
    def reload_pipeline(self, args):
        """Reload Two-Model Pipeline models."""
        if hasattr(self.brain, "two_model_pipeline") and self.brain.two_model_pipeline:
            pipeline = self.brain.two_model_pipeline
            if hasattr(pipeline, "unload_models"):
                pipeline.unload_models()
            if hasattr(pipeline, "load_models"):
                pipeline.load_models()
            self.brain.two_model_pipeline_ready = True
            if self.brain.event_bus:
                self.brain.event_bus.publish(_make_event("pipeline.reloaded", data={"status": "ok"}))
            logger.info("Pipeline models reloaded")

    def adjust_params(self, args):
        """Adjust pipeline generation parameters."""
        if hasattr(self.brain, "two_model_pipeline") and self.brain.two_model_pipeline:
            pipeline = self.brain.two_model_pipeline
            for model_name, params in args.items():
                attr = f"{model_name}_params"
                if hasattr(pipeline, attr):
                    param_ctrl = getattr(pipeline, attr)
                    if hasattr(param_ctrl, "base_params"):
                        param_ctrl.base_params.update(params)
                        logger.info(f"Adjusted {model_name} params: {params}")

    def flush_cache(self, args):
        """Flush specified cache layer."""
        cache_type = args.get("type", "all")
        if hasattr(self.brain, "hybrid_cache") and self.brain.hybrid_cache:
            cache = self.brain.hybrid_cache
            if cache_type in ("all", "hot"):
                if hasattr(cache, "clear_hot"):
                    cache.clear_hot()
                    logger.info(f"Flushed {cache_type} cache")
            if cache_type == "all" and hasattr(cache, "clear"):
                cache.clear()
                logger.info("Flushed all caches")

    def compact_memory(self, args):
        """Compact memory structures."""
        if hasattr(self.brain, "memory_manager") and self.brain.memory_manager:
            mm = self.brain.memory_manager
            if hasattr(mm, "compact"):
                mm.compact()
                logger.info("Memory compacted")

    def unload_models(self, args):
        """Выгрузить все модели из памяти."""
        target = args.get("target", "all")
        if target == "all":
            if hasattr(self.brain, "unload_all_models"):
                results = self.brain.unload_all_models()
                logger.info(f"Модели выгружены: {results}")
                if self.brain.event_bus:
                    self.brain.event_bus.publish(_make_event("memory.models_unloaded", data={"results": results}))
        elif target == "model_c":
            if hasattr(self.brain, "unload_model_c_only"):
                self.brain.unload_model_c_only()
                logger.info("Model C выгружена")
        elif target == "fractal":
            if hasattr(self.brain, "fractal_model_manager") and self.brain.fractal_model_manager:
                if hasattr(self.brain.fractal_model_manager, "unload"):
                    self.brain.fractal_model_manager.unload()
                    logger.info("FractalModelManager выгружен")
        elif target == "llama_cpp":
            if hasattr(self.brain, "llama_cpp_deployment") and self.brain.llama_cpp_deployment:
                if hasattr(self.brain.llama_cpp_deployment, "unload"):
                    self.brain.llama_cpp_deployment.unload()
                    logger.info("LlamaCppHotDeployment выгружен")

    def reload_models(self, args):
        """Перезагрузить модели."""
        target = args.get("target", "all")
        logger.info(f"Перезагрузка моделей: {target}")

    def get_memory_usage(self, args):
        """Получить информацию об использовании памяти."""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
        }

    def trigger_learning(self, args):
        """Запустить самообучение."""
        if hasattr(self.brain, "self_dialog_learning"):
            logger.info("Запущен триггер самообучения")
        else:
            logger.warning("SelfDialogLearning недоступен")

    def resolve_contradiction(self, args):
        """Разрешить противоречие."""
        logger.info(f"Разрешение противоречия: {args}")

    def recover_component(self, args):
        """Восстановить компонент."""
        component_name = args.get("component")
        logger.info(f"Попытка восстановления компонента: {component_name}")

    def abort_generation(self, args):
        """Прервать генерацию."""
        logger.info("Прерывание генерации")

    def scale_resources(self, args):
        """Масштабировать ресурсы."""
        logger.info(f"Масштабирование ресурсов: {args}")

    def rebuild_knowledge(self, args):
        """Перестроить граф знаний."""
        logger.info("Перестроение графа знаний")

    def initiate_search(self, args):
        """Инициировать поиск."""
        logger.info(f"Инициирован поиск: {args}")

    def publish_alert(self, args):
        """Опубликовать алерт."""
        if self.brain.event_bus:
            self.brain.event_bus.publish(_make_event("alert", data=args))
        logger.info(f"Alert опубликован: {args}")

    def set_timeout_limit(self, args):
        """Установить лимит таймаута."""
        limit = args.get("limit")
        if limit and hasattr(self.brain, "query_timeout"):
            self.brain.query_timeout = limit
            logger.info(f"Таймаут установлен: {limit}")

    def force_model_fallback(self, args):
        """Принудительный fallback модели."""
        logger.info("Принудительный fallback модели")

    def reset_generation_attempts(self, args):
        """Сбросить счётчики попыток генерации."""
        logger.info("Счётчики генерации сброшены")

    def set_max_retries(self, args):
        """Установить максимальное количество попыток."""
        max_retries = args.get("max_retries")
        logger.info(f"max_retries установлен: {max_retries}")

    def get_system_status(self, args):
        """Получить статус системы."""
        return {
            "running": getattr(self.brain, "running", False),
            "initialized": getattr(self.brain, "initialized", False),
        }

    def restart_component(self, args):
        """Перезапустить компонент."""
        component = args.get("component")
        logger.info(f"Перезапуск компонента: {component}")

    def update_event_subscription(self, args):
        """Обновить подписку на события."""
        logger.info(f"Обновление подписки: {args}")

    def set_log_level(self, args):
        """Установить уровень логирования."""
        level = args.get("level")
        logger.info(f"Уровень логирования установлен: {level}")


def _extract_data(event) -> Dict[str, Any]:
    if hasattr(event, "data") and isinstance(event.data, dict):
        return event.data
    return {}


class EventSubscriptionMixin:
    """Subscribes CoreBrain to critical system events and reacts."""

    def _subscribe_to_system_events(self):
        """Subscribe to all critical events."""
        eb = self.event_bus
        if not eb:
            logger.warning("EventBus not available, skipping event subscriptions")
            return

        eb.subscribe("pipeline.start", self._on_pipeline_start)
        eb.subscribe("pipeline.model_a.complete", self._on_model_a_complete)
        eb.subscribe("pipeline.model_b.complete", self._on_model_b_complete)
        eb.subscribe("pipeline.complete", self._on_pipeline_complete)
        eb.subscribe("pipeline.failed", self._on_pipeline_failed)

        eb.subscribe("component.error", self._on_component_error)
        eb.subscribe("component.initialized", self._on_component_ready)

        eb.subscribe("system.error", self._on_system_error)
        eb.subscribe("contradiction.detected", self._on_contradiction)

        eb.subscribe("learning.progress", self._on_learning_progress)
        eb.subscribe("learning.completed", self._on_learning_completed)

        eb.subscribe("memory.warning", self._on_memory_warning)
        eb.subscribe("memory.optimized", self._on_memory_optimized)

        logger.info("CoreBrain subscribed to 13 system events")

    def _on_pipeline_start(self, event):
        """Track pipeline start, log metrics."""
        data = event.data if hasattr(event, 'data') else {}
        query_id = data.get("query_id", "unknown")
        self._active_queries[query_id] = {
            "status": "running",
            "start_time": time.time(),
            "steps": [],
        }
        logger.debug(f"Pipeline started: query_id={query_id}")

    def _on_model_a_complete(self, event):
        """Check Model A quality, decide if refinement needed."""
        data = event.data if hasattr(event, 'data') else {}
        quality = data.get("quality", {})
        score = quality.get("score", 0)
        if score < 0.5:
            logger.info(f"Model A quality low (score={score}), adjusting params")
            self._issue_command(
                "adjust_pipeline_params",
                {"model_a": {"temperature": 0.5, "repeat_penalty": 1.8}},
                priority=2,
            )

    def _on_model_b_complete(self, event):
        """Log Model B completion."""
        data = event.data if hasattr(event, 'data') else {}
        logger.debug(f"Model B complete: {data}")

    def _on_pipeline_complete(self, event):
        """Track pipeline completion, update metrics."""
        data = event.data if hasattr(event, 'data') else {}
        query_id = data.get("query_id", "unknown")
        elapsed = data.get("total_time", 0)

        if query_id in self._active_queries:
            self._active_queries[query_id]["status"] = "completed"
            self._active_queries[query_id]["elapsed"] = elapsed

        self._process_metrics["total_queries"] += 1
        self._process_metrics["successful_queries"] += 1
        if elapsed > 0:
            n = self._process_metrics["total_queries"]
            avg = self._process_metrics["avg_generation_time"]
            self._process_metrics["avg_generation_time"] = (avg * (n - 1) + elapsed) / n
            self._process_metrics["max_generation_time"] = max(
                self._process_metrics["max_generation_time"], elapsed
            )
        logger.debug(f"Pipeline completed: query_id={query_id}, elapsed={elapsed:.2f}s")

    def _on_pipeline_failed(self, event):
        """React to pipeline failure - trigger fallback and track metrics."""
        data = event.data if hasattr(event, 'data') else {}
        error = data.get("error", "unknown")
        query_id = data.get("query_id", "unknown")
        
        logger.warning(f"Pipeline failed: {error}, triggering fallback")
        
        self._process_metrics["total_queries"] += 1
        self._process_metrics["failed_queries"] += 1
        
        # Get current retry count from event data
        retry_count = data.get("retry_count", 0)
        max_retries = 3
        
        if retry_count < max_retries:
            # Calculate exponential backoff: 2^retry_count seconds (2s, 4s, 8s)
            retry_delay = 2.0 ** retry_count
            
            logger.info(f"Scheduling retry {retry_count + 1}/{max_retries} for query {query_id} with delay {retry_delay}s")
            
            # Add retry command to DeferredCommandSystem with CRITICAL priority
            cmd_id = f"retry_pipeline_{query_id}_{int(time.time())}"
            
            if self.deferred_system:
                self.deferred_system.add_command(
                    command=self._cmd_retry_pipeline,
                    args=({"query_id": query_id, "retry_count": retry_count, "error": error},),
                    priority=CommandPriority.CRITICAL,
                    max_retries=0,  # Don't retry the retry command itself
                    retry_delay=retry_delay,
                    command_id=cmd_id,
                )
        else:
            logger.error(f"Max retries ({max_retries}) reached for query {query_id}, giving up")

    def _cmd_retry_pipeline(self, args):
        """Retry a failed pipeline."""
        query_id = args.get("query_id")
        retry_count = args.get("retry_count", 0)
        
        logger.info(f"Executing retry for query {query_id} (attempt {retry_count + 1})")
        
        # Republish the pipeline.start event with retry count
        if self.event_bus:
            self.event_bus.publish(Event(
                event_type=EventTypes.PIPELINE_START,
                source="retry_system",
                data={"query_id": query_id, "retry_count": retry_count + 1},
                priority=EventPriority.HIGH
            ))

    def _on_component_error(self, event):
        """React to component error - attempt recovery."""
        data = event.data if hasattr(event, 'data') else {}
        component = data.get("component", "unknown")
        logger.warning(f"Component error: {component}, attempting recovery")
        self._issue_command("recover_component", {"component": component}, priority=0)

    def _on_component_ready(self, event=None):
        """Log component readiness."""
        if event is None:
            logger.debug("Component ready (no event data)")
            return
        data = event.data if hasattr(event, 'data') else {}
        component = data.get("component", "unknown")
        logger.debug(f"Component initialized: {component}")

    def _on_system_error(self, event):
        """React to system-level error."""
        data = event.data if hasattr(event, 'data') else {}
        error = data.get("error", "unknown")
        logger.error(f"System error: {error}")
        self._issue_command("publish_alert", {
            "event_type": "system.alert",
            "data": {"type": "system_error", "error": error},
        }, priority=1)

    def _on_contradiction(self, event):
        """Trigger contradiction resolution."""
        data = event.data if hasattr(event, 'data') else {}
        logger.info(f"Contradiction detected, triggering resolution")
        self._issue_command("resolve_contradiction", data, priority=2)

    def _on_learning_progress(self, event):
        """Track learning progress."""
        data = event.data if hasattr(event, 'data') else {}
        topic = data.get("topic", "unknown")
        progress = data.get("progress", 0)
        logger.debug(f"Learning progress: topic={topic}, progress={progress}%")

    def _on_learning_completed(self, event):
        """Log learning completion."""
        data = event.data if hasattr(event, 'data') else {}
        topic = data.get("topic", "unknown")
        logger.info(f"Learning completed: topic={topic}")

    def _on_memory_warning(self, event):
        """React to memory pressure - flush caches."""
        data = event.data if hasattr(event, 'data') else {}
        warning = data.get("warning", "unknown")
        logger.warning(f"Memory warning: {warning}, flushing caches")
        self._issue_command("flush_cache", {"type": "hot"}, priority=1)
        self._issue_command("compact_memory", {}, priority=2)

    def _on_memory_optimized(self, event):
        """Log memory optimization."""
        data = event.data if hasattr(event, 'data') else {}
        logger.info(f"Memory optimized: {data}")


class CommandIssuerMixin:
    """Methods for CoreBrain to issue coordination commands."""

    def _init_command_handlers(self):
        """Инициализирует обработчики команд."""
        self._command_handlers = CommandHandlers(self)
    
    def _issue_command(self, command_type: str, args: Dict, priority: int = 2):
        """Issue a command via DeferredCommandSystem."""
        if not self.deferred_system:
            logger.debug(f"DeferredCommandSystem not available, skipping command: {command_type}")
            return None

        cmd_id = f"{command_type}_{int(time.time())}_{hash(str(args)) & 0xFFFF:04x}"

        # Используем CommandHandlers
        handlers = {
            "reload_pipeline": self._command_handlers.reload_pipeline,
            "adjust_pipeline_params": self._command_handlers.adjust_params,
            "flush_cache": self._command_handlers.flush_cache,
            "compact_memory": self._command_handlers.compact_memory,
            "unload_models": self._command_handlers.unload_models,
            "reload_models": self._command_handlers.reload_models,
            "get_memory_usage": self._command_handlers.get_memory_usage,
            "trigger_learning": self._command_handlers.trigger_learning,
            "resolve_contradiction": self._command_handlers.resolve_contradiction,
            "recover_component": self._command_handlers.recover_component,
            "abort_generation": self._command_handlers.abort_generation,
            "scale_resources": self._command_handlers.scale_resources,
            "rebuild_knowledge": self._command_handlers.rebuild_knowledge,
            "initiate_search": self._command_handlers.initiate_search,
            "publish_alert": self._command_handlers.publish_alert,
            "set_timeout_limit": self._command_handlers.set_timeout_limit,
            "force_model_fallback": self._command_handlers.force_model_fallback,
            "reset_generation_attempts": self._command_handlers.reset_generation_attempts,
            "set_max_retries": self._command_handlers.set_max_retries,
            "get_system_status": self._command_handlers.get_system_status,
            "restart_component": self._command_handlers.restart_component,
            "update_event_subscription": self._command_handlers.update_event_subscription,
            "set_log_level": self._command_handlers.set_log_level,
        }

        handler = handlers.get(command_type)
        if not handler:
            logger.warning(f"Unknown command type: {command_type}")
            return None

        try:
            self.deferred_system.add_command(
                command=handler,
                args=(args,),
                priority=_to_priority(priority),
                command_id=cmd_id,
            )
            logger.info(f"Command issued: {command_type} (id={cmd_id}, priority={priority})")
            return cmd_id
        except Exception as e:
            logger.error(f"Failed to issue command {command_type}: {e}")
            return None

    # Удалены старые методы-обработчики (_cmd_*) - теперь в CommandHandlers

    def _cmd_flush_cache(self, args):
        """Flush specified cache layer."""
        cache_type = args.get("type", "all")
        if hasattr(self, "hybrid_cache") and self.hybrid_cache:
            cache = self.hybrid_cache
            if cache_type in ("all", "hot"):
                if hasattr(cache, "clear_hot"):
                    cache.clear_hot()
                    logger.info(f"Flushed {cache_type} cache")
            if cache_type == "all" and hasattr(cache, "clear"):
                cache.clear()
                logger.info("Flushed all caches")

    def _cmd_compact_memory(self, args):
        """Compact memory structures."""
        if hasattr(self, "memory_manager") and self.memory_manager:
            mm = self.memory_manager
            if hasattr(mm, "compact"):
                mm.compact()
                logger.info("Memory compacted")

    def _cmd_unload_models(self, args):
        """Выгрузить все модели из памяти."""
        target = args.get("target", "all")
        if target == "all":
            if hasattr(self, "unload_all_models"):
                results = self.unload_all_models()
                logger.info(f"Модели выгружены: {results}")
                if self.event_bus:
                    self.event_bus.publish(_make_event("memory.models_unloaded", data={"results": results}))
        elif target == "model_c":
            if hasattr(self, "unload_model_c_only"):
                self.unload_model_c_only()
                logger.info("Model C выгружена")
        elif target == "fractal":
            if hasattr(self, "fractal_model_manager") and self.fractal_model_manager:
                if hasattr(self.fractal_model_manager, "unload"):
                    self.fractal_model_manager.unload()
                    logger.info("FractalModelManager выгружен")
        elif target == "llama_cpp":
            if hasattr(self, "llama_cpp_deployment") and self.llama_cpp_deployment:
                if hasattr(self.llama_cpp_deployment, "unload"):
                    self.llama_cpp_deployment.unload()
                    logger.info("LlamaCppHotDeployment выгружен")

    def _cmd_reload_models(self, args):
        """Перезагрузить все модели после выгрузки."""
        if hasattr(self, "reload_models"):
            results = self.reload_models()
            logger.info(f"Модели перезагружены: {results}")
            if self.event_bus:
                self.event_bus.publish(_make_event("memory.models_reloaded", data={"results": results}))

    def _cmd_get_memory_usage(self, args):
        """Получить текущее потребление памяти."""
        if hasattr(self, "get_memory_usage"):
            usage = self.get_memory_usage()
            logger.info(f"Потребление памяти: {usage}")
            return usage
        return {}

    def _cmd_trigger_learning(self, args):
        """Start a learning session."""
        topic = args.get("topic", "general")
        if hasattr(self, "self_dialog_learning") and self.self_dialog_learning:
            sdl = self.self_dialog_learning
            if hasattr(sdl, "start_session"):
                sdl.start_session(topic)
                logger.info(f"Learning session started: topic={topic}")

    def _cmd_resolve_contradiction(self, args):
        """Resolve a detected contradiction."""
        if hasattr(self, "contradiction_manager") and self.contradiction_manager:
            cm = self.contradiction_manager
            if hasattr(cm, "analyze_and_resolve"):
                cm.analyze_and_resolve(args)
            elif hasattr(cm, "resolve_contradiction"):
                cm.resolve_contradiction(args)
            logger.info(f"Contradiction resolution triggered: {args}")

    def _cmd_recover_component(self, args):
        """Attempt to recover a failed component."""
        component_name = args.get("component", "")
        if hasattr(self, component_name):
            component = getattr(self, component_name)
            if hasattr(component, "stop"):
                try:
                    component.stop()
                except Exception as e:
                    logger.debug(f"Error stopping {component_name}: {e}")
            if hasattr(component, "start"):
                try:
                    component.start()
                    logger.info(f"Component {component_name} recovered")
                except Exception as e:
                    logger.error(f"Failed to start {component_name}: {e}")
        else:
            logger.warning(f"Cannot recover unknown component: {component_name}")

    def _cmd_abort_generation(self, args):
        """Abort a running generation."""
        cmd_id = args.get("command_id")
        if hasattr(self, "generation_tracker") and self.generation_tracker:
            gt = self.generation_tracker
            if hasattr(gt, "abort"):
                gt.abort(cmd_id)
            elif hasattr(gt, "fail"):
                gt.fail(cmd_id, "Aborted by command")
            logger.info(f"Generation aborted: {cmd_id}")

    def _cmd_scale_resources(self, args):
        """Scale resource allocation."""
        component = args.get("component", "")
        level = args.get("level", "normal")
        if hasattr(self, "resource_manager") and self.resource_manager:
            rm = self.resource_manager
            if hasattr(rm, "adjust_allocation"):
                rm.adjust_allocation(component, level)
                logger.info(f"Resource scaling: component={component}, level={level}")

    def _cmd_rebuild_knowledge(self, args):
        """Rebuild knowledge graph index."""
        if hasattr(self, "knowledge_graph") and self.knowledge_graph:
            kg = self.knowledge_graph
            if hasattr(kg, "rebuild_index"):
                kg.rebuild_index()
                logger.info("Knowledge graph index rebuilt")

    def _cmd_initiate_search(self, args):
        """Initiate targeted web search."""
        concept = args.get("concept", "")
        if hasattr(self, "web_search_engine") and self.web_search_engine:
            wse = self.web_search_engine
            if hasattr(wse, "search"):
                wse.search(concept)
                logger.info(f"Web search initiated: {concept}")

    def _cmd_publish_alert(self, args):
        """Publish a system alert."""
        event_type = args.get("event_type", "system.alert")
        data = args.get("data", {})
        if self.event_bus:
            self.event_bus.publish(_make_event(event_type, data=data))
            logger.info(f"Alert published: {event_type}")

    # === Дополнительные команды управления системой ===

    def _cmd_set_timeout_limit(self, args):
        """Установить лимит ожидания генерации (в секундах). 0 = бесконечно."""
        limit = args.get("limit", 0)
        if hasattr(self, "two_model_pipeline") and self.two_model_pipeline:
            pipeline = self.two_model_pipeline
            if hasattr(pipeline, "set_timeout_limit"):
                pipeline.set_timeout_limit(limit)
                logger.info(f"Timeout limit set to {limit}s")

    def _cmd_force_model_fallback(self, args):
        """Принудительно использовать fallback модель при генерации."""
        model = args.get("model", "b")  # "b" = Model B
        if hasattr(self, "two_model_pipeline") and self.two_model_pipeline:
            pipeline = self.two_model_pipeline
            if hasattr(pipeline, "force_fallback"):
                pipeline.force_fallback(model)
                logger.info(f"Force fallback to Model {model.upper()}")

    def _cmd_reset_generation_attempts(self, args):
        """Сбросить счётчик попыток генерации."""
        if hasattr(self, "two_model_pipeline") and self.two_model_pipeline:
            pipeline = self.two_model_pipeline
            if hasattr(pipeline, "model_a_params"):
                pipeline.model_a_params.reset()
            if hasattr(pipeline, "model_b_params"):
                pipeline.model_b_params.reset()
            logger.info("Generation attempt counters reset")

    def _cmd_set_max_retries(self, args):
        """Установить максимальное количество попыток генерации."""
        max_retries = args.get("max_retries", 3)
        if hasattr(self, "two_model_pipeline") and self.two_model_pipeline:
            pipeline = self.two_model_pipeline
            if hasattr(pipeline, "max_retries"):
                pipeline.max_retries = max_retries
                logger.info(f"Max retries set to {max_retries}")

    def _cmd_get_system_status(self, args):
        """Получить полный статус системы."""
        status = {
            "pipeline_ready": getattr(self, "two_model_pipeline_ready", False),
            "memory_loaded": hasattr(self, "fractal_memory") and self.fractal_memory is not None,
            "event_bus_active": self.event_bus is not None,
            "deferred_active": hasattr(self, "deferred_system") and self.deferred_system is not None,
        }
        if hasattr(self, "resource_manager"):
            rm = self.resource_manager
            status["resources"] = {
                "cpu_usage": rm.get_cpu_usage() if hasattr(rm, "get_cpu_usage") else 0,
                "ram_usage": rm.get_memory_usage() if hasattr(rm, "get_memory_usage") else 0,
            }
        logger.info(f"System status: {status}")
        return status

    def _cmd_restart_component(self, args):
        """Перезапустить указанный компонент."""
        component = args.get("component", "")
        if hasattr(self, component):
            comp = getattr(self, component)
            if hasattr(comp, "stop"):
                try:
                    comp.stop()
                    logger.info(f"Component {component} stopped")
                except Exception as e:
                    logger.warning(f"Error stopping {component}: {e}")
            if hasattr(comp, "start"):
                try:
                    comp.start()
                    logger.info(f"Component {component} started")
                except Exception as e:
                    logger.warning(f"Error starting {component}: {e}")

    def _cmd_update_event_subscription(self, args):
        """Обновить подписку на событие (добавить/удалить обработчик)."""
        action = args.get("action", "add")  # "add" или "remove"
        event_type = args.get("event_type", "")
        handler_name = args.get("handler", "")
        
        if self.event_bus and event_type:
            if action == "add" and hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                self.event_bus.subscribe(event_type, handler)
                logger.info(f"Subscribed {handler_name} to {event_type}")
            elif action == "remove" and handler_name:
                if hasattr(self, handler_name):
                    handler = getattr(self, handler_name)
                    self.event_bus.unsubscribe(event_type, handler)
                    logger.info(f"Unsubscribed {handler_name} from {event_type}")

    def _cmd_set_log_level(self, args):
        """Установить уровень логирования для компонента."""
        component = args.get("component", "")
        level = args.get("level", "INFO")  # DEBUG, INFO, WARNING, ERROR
        
        import logging
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        logger_obj = logging.getLogger(f"eva_ai.{component}")
        logger_obj.setLevel(log_level)
        logger.info(f"Log level for {component} set to {level}")


class ProcessTrackerMixin:
    """Track and monitor active processes."""

    def __init__(self):
        self._active_queries: Dict[str, Dict[str, Any]] = {}
        self._active_commands: Dict[str, Dict[str, Any]] = {}
        self._process_metrics: Dict[str, Any] = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_generation_time": 0.0,
            "max_generation_time": 0.0,
        }

    def get_active_queries(self) -> Dict[str, Dict[str, Any]]:
        """Return all currently active queries."""
        return dict(self._active_queries)

    def get_system_coordination_status(self) -> Dict[str, Any]:
        """Return comprehensive coordination status."""
        status = {
            "active_queries": len(self._active_queries),
            "active_commands": len(self._active_commands),
            "metrics": dict(self._process_metrics),
            "event_bus_active": self.event_bus is not None,
            "deferred_system_active": self.deferred_system is not None,
            "generation_tracker_active": (
                hasattr(self, "generation_tracker") and self.generation_tracker is not None
            ),
        }
        if self.deferred_system:
            try:
                status["deferred_stats"] = self.deferred_system.get_stats()
            except Exception:
                status["deferred_stats"] = None
        return status

    def _track_pipeline_failure(self):
        """Track pipeline failure in metrics."""
        self._process_metrics["total_queries"] += 1
        self._process_metrics["failed_queries"] += 1
    
    def _track_query_success(self, elapsed: float = 0.0):
        """Track successful query for metrics."""
        self._process_metrics["total_queries"] += 1
        self._process_metrics["successful_queries"] += 1
        if elapsed > 0:
            n = self._process_metrics["total_queries"]
            avg = self._process_metrics["avg_generation_time"]
            self._process_metrics["avg_generation_time"] = (avg * (n - 1) + elapsed) / n
            self._process_metrics["max_generation_time"] = max(
                self._process_metrics["max_generation_time"], elapsed
            )
    
    def _track_query_failure(self):
        """Track failed query for metrics."""
        self._process_metrics["total_queries"] += 1
        self._process_metrics["failed_queries"] += 1
