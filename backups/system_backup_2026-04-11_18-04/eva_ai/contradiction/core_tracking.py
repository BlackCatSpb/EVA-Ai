"""Tracking history, statistics, reporting."""
import logging
import time
import json as _json
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger("eva_ai.contradiction.core.tracking")


class TrackingMixin:
    """Mixin providing tracking history, statistics, and reporting."""
    
    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по противоречиям."""
        total = len(self.contradictions)
        resolved = sum(1 for c in self.contradictions.values() if c.is_resolved())
        active = total - resolved
        
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        by_domain = {}
        
        for c in self.contradictions.values():
            domain = c.metadata.get("domain", "general") if c.metadata else "general"
            if domain not in by_domain:
                by_domain[domain] = 0
            by_domain[domain] += 1
            severity = c.severity
            if severity in by_severity:
                by_severity[severity] += 1
        
        return {
            "total": total,
            "resolved": resolved,
            "active": active,
            "by_severity": by_severity,
            "by_domain": by_domain,
            "contradictions": [c.to_dict() for c in self.contradictions.values()]
        }
    
    def get_contradiction_summary(self) -> Dict[str, int]:
        """Возвращает краткую сводку по противоречиям."""
        total = len(self.contradictions)
        resolved = sum(1 for c in self.contradictions.values() if c.is_resolved())
        return {
            "total": total,
            "resolved": resolved,
            "active": total - resolved
        }
    
    def generate_report(self) -> str:
        """Генерирует текстовый отчет о противоречиях."""
        stats = self.get_contradiction_statistics()
        summary = self.get_contradiction_summary()
        
        report = "ОТЧЕТ О ПРОТИВОРЕЧИЯХ\n"
        report += "=" * 50 + "\n\n"
        report += f"Всего противоречий: {summary['total']}\n"
        report += f"Разрешено: {summary['resolved']}\n"
        report += f"Активных: {summary['active']}\n\n"
        
        report += "ПО СЕРЬЕЗНОСТИ:\n"
        for severity, count in stats["by_severity"].items():
            report += f"  {severity}: {count}\n"
        report += "\n"
        
        report += "ПО ДОМЕНАМ:\n"
        for domain, count in stats["by_domain"].items():
            report += f"  {domain}: {count}\n"
        report += "\n"
        
        high_priority = self.prioritize_contradictions(5) if hasattr(self, 'prioritize_contradictions') else []
        if high_priority:
            report += "ВЫСОКОПРИОРИТЕТНЫЕ:\n"
            for i, c in enumerate(high_priority, 1):
                report += f"  {i}. {c['concept']} (severity: {c.get('severity', 'unknown')})\n"
        
        return report
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Возвращает историю изменений противоречий."""
        history = []
        for c in self.contradictions.values():
            for entry in c.resolution_history:
                history.append({
                    "contradiction_id": c.contradiction_id,
                    "concept": c.concept,
                    "action": entry.get("resolver", "system"),
                    "timestamp": entry["timestamp"],
                    "confidence": entry.get("confidence", 0.0)
                })
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history[:limit]
    
    def export_contradictions(self, format: str = "json") -> str:
        """Экспортирует противоречия в указанном формате."""
        data = [c.to_dict() for c in self.contradictions.values()]
        if format == "json":
            return _json.dumps(data, ensure_ascii=False, indent=2)
        elif format == "csv":
            if not data:
                return ""
            headers = list(data[0].keys())
            lines = [",".join(headers)]
            for item in data:
                row = [str(item.get(h, "")) for h in headers]
                lines.append(",".join(row))
            return "\n".join(lines)
        return _json.dumps(data, ensure_ascii=False, indent=2)
