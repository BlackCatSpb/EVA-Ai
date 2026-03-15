"""
ModuleRecoveryDetector: выявляет проблемы модулей и инициирует восстановление
"""
from __future__ import annotations
from typing import List, Dict, Any
from .base_detector import BaseDetector


class ModuleRecoveryDetector(BaseDetector):
    name = "module_recovery"
    cooldown_sec = 60.0

    def _do_probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        brain = context.get('brain')
        if not brain:
            return []
        critical = ['knowledge_graph', 'ml_unit', 'memory_manager']
        to_recover = []
        for key in critical + ['response_generator', 'text_processor']:
            comp = (brain.components or {}).get(key)
            if not comp:
                to_recover.append(key)
                continue
            # если есть health_check и он False — восстановление
            try:
                if hasattr(comp, 'health_check'):
                    status = comp.health_check()
                    ok = bool(status.get('healthy', False)) if isinstance(status, dict) else bool(status)
                    if not ok:
                        to_recover.append(key)
            except Exception:
                to_recover.append(key)
        if not to_recover:
            return []
        return [{"job_type": "ModuleRecoveryJob", "params": {"targets": to_recover}}]
