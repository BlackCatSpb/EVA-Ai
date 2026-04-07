# eva/learning/learning_opportunity.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class LearningOpportunity:
    id: str
    concept: str
    opportunity_type: str
    priority: float
    domain: str
    evidence: List[Any]
    suggested_actions: List[Any]
    created_at: float
    last_updated: float
    executed: bool
    execution: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
