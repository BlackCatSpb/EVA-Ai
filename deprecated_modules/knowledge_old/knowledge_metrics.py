"""
Метрики для Knowledge Analyzer
Часть модуля knowledge_analyzer.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class KnowledgeMetrics:
    """Метрики знаний."""
    total_nodes: int = 0
    total_edges: int = 0
    avg_connectivity: float = 0.0
    coverage_score: float = 0.0
    quality_score: float = 0.0
    consistency_score: float = 0.0
    freshness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "avg_connectivity": self.avg_connectivity,
            "coverage_score": self.coverage_score,
            "quality_score": self.quality_score,
            "consistency_score": self.consistency_score,
            "freshness_score": self.freshness_score
        }


@dataclass
class DomainAnalysis:
    """Анализ домена знаний."""
    domain: str
    node_count: int
    edge_count: int
    density: float
    avg_strength: float
    topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "avg_strength": self.avg_strength,
            "topics": self.topics
        }


@dataclass
class ConceptRelation:
    """Связь между концептами."""
    source: str
    target: str
    relation_type: str
    strength: float
    bidirectional: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "bidirectional": self.bidirectional
        }
