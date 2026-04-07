"""
ModuleRecoveryDetector: выявляет проблемы модулей и инициирует восстановление
"""
from __future__ import annotations
from typing import List, Dict, Any
from .base_detector import BaseDetector


class ModuleRecoveryDetector(BaseDetector):
    name = "recovery"
    cooldown_sec = 30.0

    def _do_probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        brain = context.get('brain')
        if not brain:
            return []
        
        # Проверяем состояние модулей
        to_recover = []
        try:
            # Проверяем основные компоненты
            components = ['memory', 'reasoning', 'learning', 'adaptation']
            for comp in components:
                if hasattr(brain, comp):
                    module = getattr(brain, comp)
                    if hasattr(module, 'health_check') and not module.health_check():
                        to_recover.append(comp)
        except Exception:
            pass
        
        if not to_recover:
            return []
        return [{"job_type": "ModuleRecoveryJob", "params": {"targets": to_recover}}]
