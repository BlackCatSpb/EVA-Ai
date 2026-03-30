"""
Fractal Address - Enhanced FractalAddress class with hierarchical addressing
"""
import hashlib
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


MAX_LEVELS = 4
BRANCHING_FACTOR = 16
EMBEDDING_DIM = 384


class FractalAddress:
    """
    Enhanced fractal address for hierarchical storage (L0→L1→L2→L3)
    Uses 384-dimensional embeddings with fractal addressing scheme
    """
    
    def __init__(self, dimensions: Optional[List[float]] = None, level: int = 0):
        self.level = level
        if dimensions is None:
            self.dimensions = [0.0] * EMBEDDING_DIM
        else:
            self.dimensions = dimensions[:EMBEDDING_DIM] if len(dimensions) >= EMBEDDING_DIM else dimensions + [0.0] * (EMBEDDING_DIM - len(dimensions))
        self.address_hash = self._compute_hash()
        self._normalized = None
    
    def _compute_hash(self) -> str:
        coord_str = ",".join(f"{d:.6f}" for d in self.dimensions[:16])
        return hashlib.md5(coord_str.encode()).hexdigest()[:16]
    
    def distance_to(self, other: 'FractalAddress') -> float:
        if len(self.dimensions) != len(other.dimensions):
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(self.dimensions, other.dimensions)) ** 0.5
    
    def cosine_similarity(self, other: 'FractalAddress') -> float:
        dot = sum(a * b for a, b in zip(self.normalized, other.normalized))
        return dot
    
    @property
    def normalized(self) -> List[float]:
        if self._normalized is None:
            norm = sum(x**2 for x in self.dimensions) ** 0.5
            if norm > 0:
                self._normalized = [x / norm for x in self.dimensions]
            else:
                self._normalized = self.dimensions[:]
        return self._normalized
    
    def get_fractal_path(self) -> List[str]:
        """Get hierarchical path components based on branching factor"""
        path = []
        for i in range(MAX_LEVELS):
            idx = self._get_branch_index(i)
            path.append(f"L{i}:{idx:02X}")
        return path
    
    def _get_branch_index(self, level: int) -> int:
        start_idx = level * (EMBEDDING_DIM // MAX_LEVELS)
        end_idx = start_idx + (EMBEDDING_DIM // MAX_LEVELS)
        values = self.dimensions[start_idx:end_idx]
        avg = sum(values) / len(values) if values else 0.0
        return int((avg + 1) / 2 * (BRANCHING_FACTOR - 1))
    
    def to_vector(self) -> np.ndarray:
        return np.array(self.dimensions, dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, level: int = 0) -> 'FractalAddress':
        return cls(dimensions=vector.tolist(), level=level)
    
    @classmethod
    def create_root(cls) -> 'FractalAddress':
        return cls(dimensions=[0.0] * EMBEDDING_DIM, level=0)
    
    def __repr__(self):
        return f"FractalAddress(level={self.level}, hash={self.address_hash}, dims={len(self.dimensions)})"
    
    def __eq__(self, other):
        if not isinstance(other, FractalAddress):
            return False
        return self.address_hash == other.address_hash
    
    def __hash__(self):
        return hash(self.address_hash)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "level": self.level,
            "hash": self.address_hash,
            "fractal_path": self.get_fractal_path()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FractalAddress':
        return cls(dimensions=data.get("dimensions"), level=data.get("level", 0))