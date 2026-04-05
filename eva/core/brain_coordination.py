"""
Brain Coordination Module for EVA CoreBrain.

Adds active coordination capabilities via mixins:
- EventSubscriptionMixin: subscribes to critical system events and reacts
- CommandIssuerMixin: issues coordination commands via DeferredCommandSystem
- ProcessTrackerMixin: tracks and monitors active processes
"""

import time
import logging
from typing import Dict, Any, Optional

from .event_bus import Event
from .deferred_command_system import CommandPriority

logger = logging.getLogger("eva.core.brain_coordination")

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
        data = _extract_data(event)
        query_id = data.get("query_id", "unknown")
        self._active_queries[query_id] = {
            "status": "running",
            "start_time": time.time(),
            "steps": [],
        }
        logger.debug(f"Pipeline started: query_id={query_id}")

    def _on_model_a_complete(self, event):
        """Check Model A quality, decide if refinement needed."""
        data = _extract_data(event)
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
        data = _extract_data(event)
        logger.debug(f"Model B complete: {data}")

    def _on_pipeline_complete(self, event):
        """Track pipeline completion, update metrics."""
        data = _extract_data(event)
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
        data = _extract_data(event)
        error = data.get("error", "unknown")
        logger.warning(f"Pipeline failed: {error}, triggering fallback")

        self._process_metrics["total_queries"] += 1
        self._process_metrics["failed_queries"] += 1

        if "model" in error.lower() or "load" in error.lower():
            self._issue_command("reload_pipeline", {}, priority=1)

    def _on_component_error(self, event):
        """React to component error - attempt recovery."""
        data = _extract_data(event)
        component = data.get("component", "unknown")
        logger.warning(f"Component error: {component}, attempting recovery")
        self._issue_command("recover_component", {"component": component}, priority=0)

    def _on_component_ready(self, event):
        """Log component readiness."""
        data = _extract_data(event)
        component = data.get("component", "unknown")
        logger.debug(f"Component initialized: {component}")

    def _on_system_error(self, event):
        """React to system-level error."""
        data = _extract_data(event)
        error = data.get("error", "unknown")
        logger.error(f"System error: {error}")
        self._issue_command("publish_alert", {
            "event_type": "system.alert",
            "data": {"type": "system_error", "error": error},
        }, priority=1)

    def _on_contradiction(self, event):
        """Trigger contradiction resolution."""
        data = _extract_data(event)
        logger.info(f"Contradiction detected, triggering resolution")
        self._issue_command("resolve_contradiction", data, priority=2)

    def _on_learning_progress(self, event):
        """Track learning progress."""
        data = _extract_data(event)
        topic = data.get("topic", "unknown")
        progress = data.get("progress", 0)
        logger.debug(f"Learning progress: topic={topic}, progress={progress}%")

    def _on_learning_completed(self, event):
        """Log learning completion."""
        data = _extract_data(event)
        topic = data.get("topic", "unknown")
        logger.info(f"Learning completed: topic={topic}")

    def _on_memory_warning(self, event):
        """React to memory pressure - flush caches."""
        data = _extract_data(event)
        warning = data.get("warning", "unknown")
        logger.warning(f"Memory warning: {warning}, flushing caches")
        self._issue_command("flush_cache", {"type": "hot"}, priority=1)
        self._issue_command("compact_memory", {}, priority=2)

    def _on_memory_optimized(self, event):
        """Log memory optimization."""
        data = _extract_data(event)
        logger.info(f"Memory optimized: {data}")


class CommandIssuerMixin:
    """Methods for CoreBrain to issue coordination commands."""

    def _issue_command(self, command_type: str, args: Dict, priority: int = 2):
        """Issue a command via DeferredCommandSystem."""
        if not self.deferred_system:
            logger.debug(f"DeferredCommandSystem not available, skipping command: {command_type}")
            return None

        cmd_id = f"{command_type}_{int(time.time())}_{hash(str(args)) & 0xFFFF:04x}"

        handlers = {
            "reload_pipeline": self._cmd_reload_pipeline,
            "adjust_pipeline_params": self._cmd_adjust_params,
            "flush_cache": self._cmd_flush_cache,
            "compact_memory": self._cmd_compact_memory,
            "trigger_learning": self._cmd_trigger_learning,
            "resolve_contradiction": self._cmd_resolve_contradiction,
            "recover_component": self._cmd_recover_component,
            "abort_generation": self._cmd_abort_generation,
            "scale_resources": self._cmd_scale_resources,
            "rebuild_knowledge": self._cmd_rebuild_knowledge,
            "initiate_search": self._cmd_initiate_search,
            "publish_alert": self._cmd_publish_alert,
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

    def _cmd_reload_pipeline(self, args):
        """Reload Two-Model Pipeline models."""
        if hasattr(self, "two_model_pipeline") and self.two_model_pipeline:
            pipeline = self.two_model_pipeline
            if hasattr(pipeline, "unload_models"):
                pipeline.unload_models()
            if hasattr(pipeline, "load_models"):
                pipeline.load_models()
            self.two_model_pipeline_ready = True
            if self.event_bus:
                self.event_bus.publish(_make_event("pipeline.reloaded", data={"status": "ok"}))
            logger.info("Pipeline models reloaded")

    def _cmd_adjust_params(self, args):
        """Adjust pipeline generation parameters."""
        if hasattr(self, "two_model_pipeline") and self.two_model_pipeline:
            pipeline = self.two_model_pipeline
            for model_name, params in args.items():
                attr = f"{model_name}_params"
                if hasattr(pipeline, attr):
                    param_ctrl = getattr(pipeline, attr)
                    if hasattr(param_ctrl, "base_params"):
                        param_ctrl.base_params.update(params)
                        logger.info(f"Adjusted {model_name} params: {params}")

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
