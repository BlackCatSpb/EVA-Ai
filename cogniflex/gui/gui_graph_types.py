"""
Типы для Knowledge Graph GUI Module
Часть модуля knowledge_graph_module.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class VisualizationType(Enum):
    """Типы визуализации."""
    GRAPH = "graph"
    TREE = "tree"
    NETWORK = "network"
    HIERARCHY = "hierarchy"


class NodeDisplayMode(Enum):
    """Режимы отображения узлов."""
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"
    HIDDEN = "hidden"


@dataclass
class GraphNode:
    """Узел для визуализации."""
    id: str
    label: str
    node_type: str
    x: float = 0.0
    y: float = 0.0
    size: float = 1.0
    color: str = "#000000"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "color": self.color,
            "metadata": self.metadata
        }


@dataclass
class GraphEdge:
    """Связь для визуализации."""
    source: str
    target: str
    label: str
    weight: float = 1.0
    color: str = "#000000"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "weight": self.weight,
            "color": self.color
        }


@dataclass
class GraphLayout:
    """Компоновка графа."""
    layout_type: VisualizationType
    width: int = 800
    height: int = 600
    zoom: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout_type": self.layout_type.value if isinstance(self.layout_type, VisualizationType) else self.layout_type,
            "width": self.width,
            "height": self.height,
            "zoom": self.zoom
        }
