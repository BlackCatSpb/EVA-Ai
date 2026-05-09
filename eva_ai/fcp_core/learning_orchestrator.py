"""
LearningGraphManager - Управление сигналами обратной связи

Заимствовано из FCP/src/fcp_knowledge/learning_manager.py

Собирает статистику по слоям и доменам.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class LearningSignal:
    """Сигнал обратной связи для обучения."""
    query: str
    domain: str
    layer_id: int
    success: bool
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_quality: float = 0.0
    user_feedback: Optional[int] = None


@dataclass
class LayerSensitivity:
    """Чувствительность слоя к домену."""
    domain: str
    layer_id: int
    success_rate: float = 0.0
    num_queries: int = 0
    avg_confidence: float = 0.0
    needs_retraining: bool = False


class LearningGraphManager:
    """
    Менеджер сигналов обратной связи.

    Отслеживает:
    - Успешность по каждому слою и домену
    - Когда нужен переобучение
    - Статистику точности
    """

    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.signals: List[LearningSignal] = []
        self.layer_sensitivity: Dict[str, Dict[int, LayerSensitivity]] = {}

        self.domains = ["facts", "reasoning", "creative", "memory", "general"]

        for domain in self.domains:
            self.layer_sensitivity[domain] = {}
            for layer_id in range(num_layers):
                self.layer_sensitivity[domain][layer_id] = LayerSensitivity(
                    domain=domain,
                    layer_id=layer_id
                )

    def add_signal(
        self,
        query: str,
        domain: str,
        layer_id: int,
        success: bool,
        confidence: float,
        response_quality: float = 0.0,
        user_feedback: Optional[int] = None
    ):
        """Добавить сигнал."""
        signal = LearningSignal(
            query=query,
            domain=domain,
            layer_id=layer_id,
            success=success,
            confidence=confidence,
            response_quality=response_quality,
            user_feedback=user_feedback
        )

        self.signals.append(signal)
        self._update_sensitivity(signal)
        self._check_retraining_needed(domain, layer_id)

    def _update_sensitivity(self, signal: LearningSignal):
        """Обновить статистику чувствительности."""
        domain = signal.domain
        layer_id = signal.layer_id

        if domain not in self.layer_sensitivity:
            self.layer_sensitivity[domain] = {}

        if layer_id not in self.layer_sensitivity[domain]:
            self.layer_sensitivity[domain][layer_id] = LayerSensitivity(
                domain=domain,
                layer_id=layer_id
            )

        sens = self.layer_sensitivity[domain][layer_id]

        n = sens.num_queries + 1

        if signal.success:
            sens.success_rate = (sens.success_rate * (n - 1) + 1) / n
        else:
            sens.success_rate = sens.success_rate * (n - 1) / n

        sens.avg_confidence = (sens.avg_confidence * (n - 1) + signal.confidence) / n
        sens.num_queries = n

    def _check_retraining_needed(self, domain: str, layer_id: int):
        """Проверить нужно ли переобучение."""
        sens = self.layer_sensitivity.get(domain, {}).get(layer_id)

        if sens and sens.num_queries >= 10:
            if sens.success_rate < 0.6:
                sens.needs_retraining = True

    def get_layer_for_domain(self, domain: str) -> List[int]:
        """Получить список слоёв для домена, отсортированных по успешности."""
        if domain not in self.layer_sensitivity:
            return list(range(self.num_layers))

        layers = []
        for layer_id, sens in self.layer_sensitivity[domain].items():
            layers.append((layer_id, sens.success_rate))

        layers.sort(key=lambda x: x[1], reverse=True)

        return [l[0] for l in layers]

    def get_layers_needing_retraining(self, domain: str) -> List[int]:
        """Получить слои, которым нужно переобучение."""
        if domain not in self.layer_sensitivity:
            return []

        return [
            layer_id for layer_id, sens in self.layer_sensitivity[domain].items()
            if sens.needs_retraining
        ]

    def get_statistics(self, domain: str) -> Dict:
        """Получить статистику по домену."""
        if domain not in self.layer_sensitivity:
            return {}

        layers = self.layer_sensitivity[domain]

        total_queries = sum(s.num_queries for s in layers.values())
        avg_success = np.mean([s.success_rate for s in layers.values() if s.num_queries > 0])

        return {
            "domain": domain,
            "total_queries": total_queries,
            "avg_success_rate": avg_success,
            "layers_needing_retraining": len(self.get_layers_needing_retraining(domain))
        }

    def clear_signals(self):
        """Очистить старые сигналы."""
        self.signals.clear()


class LearningOrchestrator:
    """
    LO: Оркестратор обучения.
    
    Features:
    - Анализ успешности по доменам (LO-1)
    - Автоматическое формирование обучающих задач (LO-2)
    - Интеграция с LoRA manager
    """

    def __init__(
        self,
        learning_manager: LearningGraphManager,
        lora_manager
    ):
        self.learning_manager = learning_manager
        self.lora_manager = lora_manager
        self.retrain_threshold = 0.6
        self.min_queries_for_decision = 10
        
        # LO-1: Domain trends tracking
        self.domain_trends: Dict[str, List[float]] = {d: [] for d in learning_manager.domains}
        self.domain_history: Dict[str, List[Dict]] = {d: [] for d in learning_manager.domains}
    
    def should_retrain(self, domain: str) -> bool:
        """Определить нужно ли переобучение."""
        stats = self.learning_manager.get_statistics(domain)

        if stats["total_queries"] < self.min_queries_for_decision:
            return False

        return stats["avg_success_rate"] < self.retrain_threshold
    
    def analyze_domain_trend(self, domain: str) -> str:
        """
        LO-1: Analyze success rate trend for domain.
        
        Returns:
            "improving", "stable", "declining", or "unknown"
        """
        if domain not in self.domain_trends or len(self.domain_trends[domain]) < 3:
            return "unknown"
        
        recent = self.domain_trends[domain][-5:]
        if len(recent) < 2:
            return "unknown"
        
        slope = (recent[-1] - recent[0]) / len(recent)
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def get_domain_success_analysis(self, domain: str) -> Dict:
        """
        LO-1: Detailed domain success analysis.
        
        Returns:
            {
                "total_queries": int,
                "avg_success_rate": float,
                "trend": str,
                "confidence_evolution": List[float],
                "layers_needing_retraining": List[int],
                "recommended_action": str
            }
        """
        stats = self.learning_manager.get_statistics(domain)
        trend = self.analyze_domain_trend(domain)
        
        confidence_history = self.domain_trends.get(domain, [])
        
        layers_need_training = self.learning_manager.get_layers_needing_retraining(domain)
        
        success_rate = stats.get("avg_success_rate", 0.5)
        
        if trend == "declining" or success_rate < 0.5:
            recommended = "immediate_retraining"
        elif success_rate < self.retrain_threshold:
            recommended = "scheduled_retraining"
        elif trend == "improving":
            recommended = "monitor"
        else:
            recommended = "maintain"
        
        return {
            "domain": domain,
            "total_queries": stats.get("total_queries", 0),
            "avg_success_rate": success_rate,
            "trend": trend,
            "confidence_evolution": confidence_history[-10:],
            "layers_needing_retraining": layers_need_training,
            "recommended_action": recommended
        }
    
    def generate_training_tasks(self, domain: str, max_tasks: int = 10) -> List[Dict]:
        """
        LO-2: Automatically generate training tasks from failed queries.
        
        Returns:
            List of training tasks [{query, expected_response, priority}, ...]
        """
        failed_signals = [
            s for s in self.learning_manager.signals
            if s.domain == domain and not s.success
        ]
        
        tasks = []
        seen_queries = set()
        
        for signal in failed_signals:
            query_key = signal.query[:50]
            if query_key in seen_queries:
                continue
            
            seen_queries.add(query_key)
            
            priority = 1.0 - signal.confidence
            if signal.user_feedback is not None:
                if signal.user_feedback < 0:
                    priority = 1.0
                elif signal.user_feedback > 0:
                    priority = 0.5
            
            task = {
                "query": signal.query,
                "layer_id": signal.layer_id,
                "priority": priority,
                "quality_score": signal.response_quality,
                "feedback": signal.user_feedback
            }
            tasks.append(task)
            
            if len(tasks) >= max_tasks:
                break
        
        tasks.sort(key=lambda x: x["priority"], reverse=True)
        return tasks
    
    def get_retrain_plan(self, domain: str) -> Dict:
        """Получить план переобучения."""
        layers = self.learning_manager.get_layers_needing_retraining(domain)

        rank_map = {}
        for layer_id in layers:
            sens = self.learning_manager.layer_sensitivity[domain][layer_id]
            if sens.success_rate < 0.4:
                rank_map[layer_id] = 16
            elif sens.success_rate < 0.5:
                rank_map[layer_id] = 8
            else:
                rank_map[layer_id] = 4

        return {
            "domain": domain,
            "layers": layers,
            "ranks": rank_map,
            "reason": f"success_rate < {self.retrain_threshold}",
            "training_tasks": self.generate_training_tasks(domain, max_tasks=5)
        }
    
    def execute_retrain(self, domain: str) -> bool:
        """Выполнить переобучение."""
        if not self.should_retrain(domain):
            return False

        plan = self.get_retrain_plan(domain)
        
        if hasattr(self.lora_manager, 'create_training_job'):
            for layer_id, rank in plan["ranks"].items():
                job_params = {
                    "domain": domain,
                    "layer_id": layer_id,
                    "rank": rank,
                    "tasks": [t["query"] for t in plan.get("training_tasks", [])[:3]]
                }
                self.lora_manager.create_training_job(**job_params)
            return True
        
        return True
    
    def update_domain_trend(self, domain: str, success_rate: float):
        """Update domain trend tracking."""
        if domain not in self.domain_trends:
            self.domain_trends[domain] = []
        self.domain_trends[domain].append(success_rate)
        
        if len(self.domain_trends[domain]) > 100:
            self.domain_trends[domain] = self.domain_trends[domain][-50:]
    
    def get_all_domain_analysis(self) -> Dict[str, Dict]:
        """Get analysis for all domains."""
        return {
            domain: self.get_domain_success_analysis(domain)
            for domain in self.learning_manager.domains
        }
