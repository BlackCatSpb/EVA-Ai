"""
Reasoning Node Types for Knowledge Graph
Extends KnowledgeNode with reasoning-specific fields
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time

class ReasoningNodeType(Enum):
    REASONING_STEP = "reasoning_step"
    CLARIFICATION_QUESTION = "clarification_question"
    REASONING_SESSION = "reasoning_session"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"

class ReasoningRelationType(Enum):
    FOLLOWS_FROM = "follows_from"
    CLARIFIES = "clarifies"
    ASSESSES = "assesses"
    GENERATED_FROM = "generated_from"
    ANALYZED_BY = "analyzed_by"

@dataclass
class ReasoningNode:
    """Node for storing reasoning steps in Knowledge Graph"""
    id: str
    name: str
    description: str
    reasoning_type: str = "reasoning_step"
    domain: str = "system"
    
    confidence_score: float = 1.0
    parent_step_id: Optional[str] = None
    session_id: Optional[str] = None
    
    query_context: Optional[str] = None
    response_context: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_knowledge_node_format(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "node_type": self.reasoning_type,
            "domain": self.domain,
            "strength": self.confidence_score,
            "meta": {
                "confidence_score": self.confidence_score,
                "parent_step_id": self.parent_step_id,
                "session_id": self.session_id,
                "query_context": self.query_context,
                "response_context": self.response_context,
                "analysis_data": self.analysis_data,
                "iteration": self.iteration,
                "timestamp": self.timestamp
            }
        }

@dataclass  
class ReasoningSession:
    """Root node for a complete reasoning session"""
    session_id: str
    original_query: str
    iterations: int
    final_confidence: float
    clarifications_asked: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_node(self) -> Dict[str, Any]:
        return {
            "id": self.session_id,
            "name": f"Reasoning Session",
            "description": f"Query: {self.original_query}",
            "node_type": "reasoning_session",
            "domain": "system",
            "strength": self.final_confidence,
            "meta": {
                "original_query": self.original_query,
                "iterations": self.iterations,
                "final_confidence": self.final_confidence,
                "clarifications_asked": self.clarifications_asked,
                "timestamp": self.timestamp
            }
        }
