"""User feedback processing, correction application."""
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger("eva.contradiction.learning.feedback")


class FeedbackProcessingMixin:
    """Mixin providing user feedback processing and correction capabilities."""
    
    def update_status(self, new_status: str, progress: Optional[float] = None):
        """
        Обновляет статус возможности обучения.
        
        Args:
            new_status: Новый статус
            progress: Прогресс (опционально)
        """
        self.status = new_status
        self.last_updated = time.time()
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        self.learning_history.append({
            "status": new_status,
            "progress": self.progress,
            "timestamp": self.last_updated
        })
    
    def add_learning_task(self, task: Dict[str, Any]):
        """Добавляет задачу обучения."""
        self.learning_tasks.append(task)
        self.last_updated = time.time()
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Помечает задачу как выполненную."""
        for i, task in enumerate(self.learning_tasks):
            if task["task_id"] == task_id:
                completed_task = task.copy()
                completed_task["completed_at"] = time.time()
                completed_task["result"] = result
                self.completed_tasks.append(completed_task)
                del self.learning_tasks[i]
                total_tasks = len(self.completed_tasks) + len(self.learning_tasks)
                if total_tasks > 0:
                    self.progress = len(self.completed_tasks) / total_tasks
                self.last_updated = time.time()
                break
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Возвращает отчет о прогрессе."""
        return {
            "id": self.id,
            "concept": self.concept,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "total_tasks": len(self.learning_tasks) + len(self.completed_tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len(self.learning_tasks),
            "impact_assessment": self.impact_assessment,
            "last_updated": self.last_updated
        }
    
    def get_learning_recommendations(self) -> List[str]:
        """Генерирует рекомендации по обучению на основе противоречия."""
        recommendations = []
        
        if "numeric_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проведите дополнительный анализ для определения наиболее точного числового значения. "
                "Рассмотрите возможность усреднения показателей или выявления контекстных условий, "
                "при которых верно каждое значение."
            )
        elif "boolean_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проверьте условия, при которых каждое утверждение является верным. "
                "Возможно, противоречие возникает из-за различия в контексте или условиях."
            )
        elif "exclusivity_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проанализируйте, не являются ли утверждения 'только' и 'не только' "
                "применимыми в разных контекстах или подкатегориях."
            )
        elif "hierarchy_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Пересмотрите иерархию для устранения циклических зависимостей или "
                "взаимоисключающих классификаций. Возможно, некоторые связи должны "
                "быть заменены на другие типы отношений."
            )
        elif "response_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проанализируйте контекст использования каждого ответа. Возможно, "
                "разные ответы применимы в разных сценариях или для разных аудиторий."
            )
        
        severity = self.metadata.get("severity", "medium")
        if severity == "high":
            recommendations.append(
                "Это высокосерьезное противоречие требует немедленного внимания. "
                "Рассмотрите возможность привлечения экспертов для разрешения."
            )
        elif severity == "medium":
            recommendations.append(
                "Это среднесерьезное противоречие важно для точности системы. "
                "Планируйте его разрешение в ближайшее время."
            )
        else:
            recommendations.append(
                "Это низкосерьезное противоречие имеет минимальное влияние на систему. "
                "Его можно разрешить в рамках регулярного процесса обновления знаний."
            )
        
        recommendations.append(
            "Соберите дополнительные данные из авторитетных источников для подтверждения или "
            "опровержения конфликтующих утверждений."
        )
        recommendations.append(
            "Проведите анализ контекстных условий, при которых проявляется каждое утверждение."
        )
        return recommendations
    
    def is_high_priority(self) -> bool:
        """Проверяет, является ли возможность обучения высокоприоритетной."""
        return self.priority > 0.7
    
    def get_time_since_creation(self) -> float:
        """Возвращает время с момента создания возможности обучения."""
        return time.time() - self.created_at
    
    def requires_immediate_attention(self) -> bool:
        """Проверяет, требует ли возможность обучения немедленного внимания."""
        if self.priority > 0.8 and self.get_time_since_creation() > 86400:
            return True
        if self.priority > 0.6 and self.get_time_since_creation() > 3 * 86400:
            return True
        return False
