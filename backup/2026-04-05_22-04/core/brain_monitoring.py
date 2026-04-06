"""
Logging, metrics, health checks, and resource monitoring for CoreBrain.
Memory eviction methods moved to brain_memory.py.
"""
import time
import logging
from typing import Dict, Any, Optional

query_logger = logging.getLogger("eva.core_brain.query_processing")
logger = logging.getLogger("eva.core_brain")


class MonitoringMixin:
    """Mixin providing logging, metrics, health checks, and monitoring to CoreBrain."""

    def _log_throttled(self, logger_obj: logging.Logger, level: int, key: str, message: str) -> None:
        """Logs a message no more than once per self.log_throttle_seconds for the given key."""
        try:
            now = time.time()
            with self._log_throttle_lock:
                if len(self._log_throttle) > 500:
                    self._log_throttle = {k: v for k, v in list(self._log_throttle.items())[-250:]}
                last = self._log_throttle.get(key, 0.0)
                if (now - last) >= float(self.log_throttle_seconds):
                    self._log_throttle[key] = now
                    should_log = True
                else:
                    should_log = False
            if should_log:
                logger_obj.log(level, message)
        except Exception:
            logger_obj.log(level, message)

    def log_module_activity(self, module_name: str, activity: str, details: Dict[str, Any] = None):
        """Logs module activity."""
        try:
            with self.activity_lock:
                timestamp = time.time()
                if module_name not in self.module_activity:
                    self.module_activity[module_name] = {'first_access': timestamp, 'last_access': timestamp, 'access_count': 0, 'activities': []}
                self.module_activity[module_name]['last_access'] = timestamp
                self.module_activity[module_name]['access_count'] += 1
                activity_record = {'timestamp': timestamp, 'module': module_name, 'activity': activity, 'details': details or {}}
                self.module_activity[module_name]['activities'].append(activity_record)
                self.module_activity[module_name]['activities'] = self.module_activity[module_name]['activities'][-100:]
                self.module_access_log.append(activity_record)
                if len(self.module_access_log) > 1000:
                    self.module_access_log = self.module_access_log[-500:]
                query_logger.debug(f"Module activity {module_name}: {activity}")
        except Exception as e:
            query_logger.error(f"Error logging module activity {module_name}: {e}")

    def get_module_activity(self, module_name: str = None) -> Dict[str, Any]:
        """Returns module activity information."""
        with self.activity_lock:
            if module_name:
                return self.module_activity.get(module_name, {})
            return {'modules': dict(self.module_activity), 'total_accesses': len(self.module_access_log),
                    'active_modules': len([m for m, a in self.module_activity.items() if time.time() - a['last_access'] < 300]),
                    'recent_activities': self.module_access_log[-20:] if self.module_access_log else []}

    def get_system_health(self) -> Dict[str, Any]:
        """Returns system health status."""
        try:
            health_status = {"status": "healthy", "timestamp": time.time(), "components": {}, "warnings": [], "errors": [], "resources": {}}
            if self.state_manager:
                state = self.state_manager.get_state()
                health_status["system_state"] = state.value if hasattr(state, 'value') else str(state)
                if str(state) in ["ERROR", "FAILED"]:
                    health_status["status"] = "unhealthy"
                    health_status["errors"].append(f"System state: {state}")
            if self.resource_manager:
                try:
                    resource_info = self.resource_manager.get_system_info() or {}
                    memory_percent = resource_info.get("memory_percent", 0)
                    if memory_percent > 90:
                        health_status["warnings"].append(f"High memory usage: {memory_percent}%")
                        health_status["status"] = "degraded"
                    health_status["components"]["resources"] = "ok"
                    health_status["resources"] = resource_info
                except Exception as e:
                    health_status["errors"].append(f"Resource manager error: {e}")
                    health_status["status"] = "degraded"
            else:
                health_status.setdefault("resources", {})
            for component in ['ml_unit', 'memory_manager']:
                if component in self.components and self.components[component]:
                    health_status["components"][component] = "ok"
                else:
                    health_status["warnings"].append(f"Component {component} not available")
                    if health_status["status"] == "healthy":
                        health_status["status"] = "degraded"
            return health_status
        except Exception as e:
            query_logger.error(f"Error getting system health status: {e}", exc_info=True)
            return {"status": "error", "timestamp": time.time(), "error": str(e), "components": {}, "warnings": [], "errors": [str(e)]}

    def get_metrics(self) -> Dict[str, Any]:
        """Returns system metrics."""
        query_logger.debug("System metrics requested")
        return self.metrics_manager.get_metrics()

    def emit_metric(self, metric: Dict[str, Any]) -> bool:
        try:
            if hasattr(self.metrics_manager, "emit"):
                return bool(self.metrics_manager.emit(metric))
        except Exception:
            pass
        return False

    def emit_metrics(self, metrics) -> int:
        try:
            if hasattr(self.metrics_manager, "emit_many"):
                return int(self.metrics_manager.emit_many(metrics))
        except Exception:
            pass
        return 0

    def flush_emitted_metrics(self):
        try:
            if hasattr(self.metrics_manager, "flush"):
                return list(self.metrics_manager.flush())
        except Exception:
            pass
        return []

    def get_status(self) -> Dict[str, Any]:
        """Returns extended system status."""
        status = {"initialized": self.initialized, "running": self.running, "components": len(self.components),
                  "metrics": self.metrics_manager.get_metrics() if hasattr(self.metrics_manager, 'get_metrics') else {},
                  "two_model_pipeline": {"ready": self.two_model_pipeline_ready if hasattr(self, 'two_model_pipeline_ready') else False,
                                         "active": self.two_model_pipeline is not None if hasattr(self, 'two_model_pipeline') else False},
                  "llama_cpp": {"ready": self.llama_cpp_ready if hasattr(self, 'llama_cpp_ready') else False,
                                "active": self.llama_cpp_deployment is not None if hasattr(self, 'llama_cpp_deployment') else False}}
        if self.state_manager:
            status["system_state"] = self.state_manager.get_system_summary()
            status["health"] = {"status": self.state_manager.get_state().value}
        if self.resource_manager:
            status["resources"] = self.resource_manager.get_resource_summary()
        if self.config_manager:
            try:
                status["config_valid"] = self.config_manager.validate_config() if hasattr(self.config_manager, 'validate_config') else None
            except Exception:
                status["config_valid"] = None
        return status

    def get_system_metrics(self) -> Dict[str, Any]:
        """Returns system metrics including neuromorphic status."""
        base = self.system_metrics_manager.get_metrics() if self.system_metrics_manager else {}
        try:
            neu = self.components.get('neuromorphic_simulator') if hasattr(self, 'components') else None
            if neu is None and hasattr(self, 'neuromorphic_simulator'):
                neu = getattr(self, 'neuromorphic_simulator')
            if neu:
                neu_status = {"available": True, "running": bool(getattr(neu, 'running', False)), "use_nest": bool(getattr(neu, 'use_nest', False))}
                if hasattr(neu, 'get_system_health'):
                    health = neu.get_system_health()
                    neu_status.update({"health_status": health.get("status"), "health_score": health.get("health_score"),
                                       "interaction_strength": (health.get("analysis", {}) or {}).get("interaction_strength"),
                                       "total_activities": (health.get("analysis", {}) or {}).get("total_activities"),
                                       "timestamp": health.get("timestamp")})
                base["neuromorphic"] = neu_status
            else:
                base["neuromorphic"] = {"available": False}
        except Exception as e:
            try:
                base["neuromorphic_error"] = str(e)
            except Exception:
                pass
        return base

    def get_system_dashboard_data(self) -> Dict[str, Any]:
        """Returns data for the system dashboard."""
        self._log_throttled(query_logger, logging.INFO, "dashboard_request", "Dashboard data requested")
        dashboard_start = time.time()
        try:
            data = {"timestamp": time.time(), "metrics": self.system_metrics_manager.get_metrics() if self.system_metrics_manager else {},
                    "health": self.get_system_health(), "contradiction_stats": self.get_contradiction_statistics(),
                    "learning_opportunities": self._get_learning_opportunities(), "system_info": self._get_system_info()}
            dashboard_time = time.time() - dashboard_start
            self._log_throttled(query_logger, logging.INFO, "dashboard_ready", f"Dashboard data generated in {dashboard_time:.4f} sec")
            return data
        except Exception as e:
            query_logger.error(f"Error generating dashboard data in {time.time() - dashboard_start:.4f} sec: {e}", exc_info=True)
            return {"error": str(e), "timestamp": time.time(), "partial_data": {"metrics": {}, "health": self.get_system_health()}}

    def _get_learning_opportunities(self):
        """Returns learning opportunities."""
        try:
            opportunities = []
            if 'ml_unit' in self.components and self.components['ml_unit'] and hasattr(self.components['ml_unit'], 'get_learning_opportunities'):
                opportunities.extend(self.components['ml_unit'].get_learning_opportunities())
            if not opportunities:
                opportunities = [{"type": "pattern_analysis", "description": "Анализ паттернов в запросах пользователей", "priority": "medium", "timestamp": time.time()},
                                 {"type": "knowledge_expansion", "description": "Расширение базы знаний", "priority": "low", "timestamp": time.time()}]
            return opportunities
        except Exception as e:
            query_logger.error(f"Error getting learning opportunities: {e}", exc_info=True)
            return []

    def get_resource_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of resource usage."""
        try:
            if hasattr(self, 'resource_manager') and self.resource_manager:
                return {'cpu_usage': self.resource_manager.get_cpu_usage(), 'memory_usage': self.resource_manager.get_memory_usage(),
                        'disk_usage': self.resource_manager.get_disk_usage() if hasattr(self.resource_manager, 'get_disk_usage') else 0,
                        'timestamp': time.time(), 'io_tokens': getattr(self.resource_manager, 'io_tokens', 0)}
        except Exception as e:
            query_logger.warning(f"Error getting resource snapshot: {e}")
        return {}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Returns cache statistics."""
        try:
            cache_stats = {}
            if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
                hc_stats = getattr(self.hybrid_cache, 'get_stats', lambda: {})()
                if callable(hc_stats):
                    hc_stats = hc_stats()
                cache_stats['hit_rate'] = hc_stats.get('hit_rate', 0.0) if isinstance(hc_stats, dict) else 0.0
                cache_stats['cache_utilization_percent'] = hc_stats.get('utilization', 0.0) if isinstance(hc_stats, dict) else 0.0
                cache_stats['disk_stats'] = {'entries': hc_stats.get('disk_entries', 0) if isinstance(hc_stats, dict) else 0}
            return cache_stats
        except Exception as e:
            query_logger.warning(f"Error getting cache stats: {e}")
        return {}

    def get_cache_health_status(self) -> Dict[str, Any]:
        """Returns detailed cache health status."""
        try:
            if not hasattr(self, 'token_cache') or not self.token_cache:
                return {'status': 'unavailable', 'message': 'Token cache unavailable'}
            stats = self.token_cache.get_cache_stats()
            import psutil
            memory = psutil.virtual_memory()
            total_requests = stats.get('total_requests', 0)
            total_hits = stats.get('vram_hits', 0) + stats.get('ram_hits', 0) + stats.get('disk_hits', 0)
            hit_rate = total_hits / max(1, total_requests)
            status = 'excellent' if hit_rate > 0.8 else 'good' if hit_rate > 0.6 else 'fair' if hit_rate > 0.4 else 'poor'
            return {'status': status, 'hit_rate': hit_rate, 'memory_usage': memory.percent, 'cache_stats': stats,
                    'recommendations': self._get_cache_recommendations(stats, memory.percent)}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _get_cache_recommendations(self, stats: Dict, memory_percent: float):
        """Generates cache optimization recommendations."""
        recommendations = []
        try:
            total_requests = stats.get('total_requests', 0)
            total_hits = stats.get('vram_hits', 0) + stats.get('ram_hits', 0) + stats.get('disk_hits', 0)
            hit_rate = total_hits / max(1, total_requests)
            if hit_rate < 0.5:
                recommendations.append("Низкий hit rate кэша. Рассмотрите увеличение размера кэша.")
            if memory_percent > 85:
                recommendations.append("Высокое использование памяти. Рекомендуется агрессивное вытеснение.")
            vram_hits = stats.get('vram_hits', 0)
            ram_hits = stats.get('ram_hits', 0)
            disk_hits = stats.get('disk_hits', 0)
            try:
                import torch
            except ImportError:
                torch = None
            if vram_hits == 0 and torch is not None and torch.cuda.is_available():
                recommendations.append("VRAM кэш не используется. Проверьте настройки GPU.")
            if disk_hits > ram_hits * 2:
                recommendations.append("Частое обращение к SSD. Рассмотрите увеличение RAM кэша.")
        except Exception as e:
            recommendations.append(f"Ошибка анализа рекомендаций: {e}")
        return recommendations
