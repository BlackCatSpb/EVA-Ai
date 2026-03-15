"""
ModuleRecoveryJob: попытка восстановления критических модулей
"""
from __future__ import annotations
import logging
from typing import Any, Dict
from .base_job import BaseJob, CommandPriority

logger = logging.getLogger("cogniflex.core.autopilot.recovery")


class ModuleRecoveryJob(BaseJob):
    job_type = "ModuleRecoveryJob"
    resource_class = "CPU"
    default_priority = CommandPriority.HIGH

    def run(self, context: Dict[str, Any]) -> None:
        brain = self.brain
        initializer = getattr(brain, 'component_initializer', None)
        if not initializer:
            logger.info("ModuleRecoveryJob: ComponentInitializer недоступен")
            return
        targets = context.get('targets') or []
        for module_key in targets:
            try:
                method_name = {
                    'ml_unit': '_init_ml_unit',
                    'knowledge_graph': '_init_knowledge_graph',
                    'memory_manager': '_init_memory_manager',
                    'text_processor': '_init_text_processor',
                    'neuromorphic_simulator': '_init_neuromorphic_simulator',
                    'ethics_framework': '_init_ethics_framework',
                    'web_search_engine': '_init_web_search_engine',
                    'distributed_system': '_init_distributed_system',
                    'adaptation_manager': '_init_adaptation_manager',
                    'contradiction_resolver': '_init_contradiction_resolver',
                    'self_analyzer': '_init_self_analyzer',
                    'learning_scheduler': '_init_learning_scheduler',
                }.get(module_key)
                if method_name and hasattr(initializer, method_name):
                    getattr(initializer, method_name)()
                    logger.info(f"Восстановление {module_key}: успех")
                else:
                    logger.warning(f"ModuleRecoveryJob: метод для {module_key} не найден ({method_name})")
            except Exception as e:
                logger.error(f"Ошибка восстановления {module_key}: {e}", exc_info=True)
