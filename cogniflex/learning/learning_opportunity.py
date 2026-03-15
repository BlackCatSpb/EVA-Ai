# cogniflex/learning/learning_opportunity.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class LearningOpportunity:
    concept: str
    opportunity_type: str
    priority: int
    domain: str
    evidence: List[Any]
    suggested_actions: List[Any]
    created_at: float
    last_updated: float
    executed: bool
    execution: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
