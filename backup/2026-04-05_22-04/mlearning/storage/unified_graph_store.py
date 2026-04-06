"""
Unified Memory Graph Store with Fractal Storage Integration
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, Tuple
import torch
import numpy as np
from torch import nn

from .fractal_weight_store import FractalWeightStore

logger = logging.getLogger(__name__)

class UnifiedMemoryGraph:
    """
    Unified memory graph that uses fractal storage as its core
    """
    def __init__(self,
                base_path: str,
                device: str = 'cpu',
                block_size: int = 64,
                fractal_levels: int = 3):
        """
        Initialize unified memory graph
        
        Args:
            base_path: Base path for storage
            device: Device to use
            block_size: Size of fractal blocks
            fractal_levels: Number of fractal levels
        """
        self.base_path = Path(base_path)
        self.device = device
        
        # Create storage directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.base_path / "nodes", exist_ok=True)
        os.makedirs(self.base_path / "edges", exist_ok=True)
        
        # Initialize fractal storage
        self.fractal_store = FractalWeightStore(
            block_size=block_size,
            fractal_levels=fractal_levels,
            device=device
        )
        
        # Memory structures
        self.nodes = {}
        self.edges = {}
        
    def store(self, key: str, value: Any) -> None:
        """Store data in the fractal storage"""
        self.fractal_store.store(key, value)
        
    def get(self, key: str) -> Any:
        """Get stored data"""
        return self.fractal_store.get(key)
        
    def add_node(self,
                content: Union[str, Dict],
                node_type: str = "concept",
                vector: Optional[torch.Tensor] = None,
                metadata: Optional[Dict] = None) -> str:
        """Add node to graph"""
        # Generate node ID
        node_id = f"node_{len(self.nodes)}"
        
        # Create node record
        node = {
            'id': node_id,
            'type': node_type,
            'content': content,
            **(metadata or {})
        }
        
        # Store node
        self.nodes[node_id] = node
        
        # Store vector if provided
        if vector is not None:
            vector = vector.to(self.device)
            self.node_vectors[node_id] = vector
            
            # Store in fractal storage
            self.fractal_store.store(
                node_id,
                vector,
                metadata=node
            )
            
        return node_id
        
    def add_edge(self,
                source: str,
                target: str,
                edge_type: str = "related",
                metadata: Optional[Dict] = None):
        """Add edge between nodes"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target node not found")
            
        edge = {
            'source': source,
            'target': target,
            'type': edge_type,
            **(metadata or {})
        }
        
        self.edges.append(edge)
        
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node by ID"""
        return self.nodes.get(node_id)
        
    def get_node_vector(self, node_id: str) -> Optional[torch.Tensor]:
        """Get node vector"""
        return self.node_vectors.get(node_id)
        
    def get_connected_nodes(self,
                          node_id: str,
                          edge_types: Optional[List[str]] = None) -> List[Dict]:
        """Get nodes connected to given node"""
        if node_id not in self.nodes:
            return []
            
        connected = []
        for edge in self.edges:
            if edge['source'] == node_id:
                if edge_types and edge['type'] not in edge_types:
                    continue
                    
                target_id = edge['target']
                if target_id in self.nodes:
                    connected.append({
                        **self.nodes[target_id],
                        'edge_type': edge['type']
                    })
                    
        return connected
        
    def find_similar_nodes(self,
                         vector: torch.Tensor,
                         limit: int = 10,
                         threshold: float = 0.5) -> List[Dict]:
        """Find nodes with similar vectors"""
        if not self.node_vectors:
            return []
            
        similarities = []
        query = vector.to(self.device)
        
        for node_id, node_vector in self.node_vectors.items():
            if node_vector.shape != query.shape:
                continue
                
            sim = torch.nn.functional.cosine_similarity(
                query.flatten(),
                node_vector.flatten(),
                dim=0
            )
            
            if sim >= threshold:
                similarities.append((node_id, sim.item()))
                
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        results = []
        for node_id, sim in similarities[:limit]:
            results.append({
                **self.nodes[node_id],
                'similarity': sim
            })
            
        return results
        
    def get_training_data(self,
                        node_types: Optional[List[str]] = None) -> List[Dict]:
        """Get training data from graph"""
        data = []
        nodes = self.nodes.items()
        
        if node_types:
            nodes = [
                (nid, node) for nid, node in nodes
                if node['type'] in node_types
            ]
            
        for node_id, node in nodes:
            if node_id not in self.node_vectors:
                continue
                
            vector = self.node_vectors[node_id]
            
            # Get connected nodes
            connected = self.get_connected_nodes(node_id)
            if not connected:
                continue
                
            # Get target vectors
            target_vectors = []
            for target in connected:
                target_id = target['id']
                if target_id in self.node_vectors:
                    target_vectors.append(
                        self.node_vectors[target_id]
                    )
                    
            if not target_vectors:
                continue
                
            target = torch.stack(target_vectors).mean(dim=0)
            
            data.append({
                'input': vector,
                'target': target,
                'metadata': node
            })
            
        return data
        
    def save_state(self):
        """Save graph state"""
        # Save nodes and edges
        torch.save({
            'nodes': self.nodes,
            'edges': self.edges
        }, self.base_path / "graph.pt")
        
        # Save vectors
        torch.save(
            self.node_vectors,
            self.base_path / "vectors.pt"
        )
        
        # Save fractal storage
        self.fractal_store.save_to_disk()
        
    def load_state(self):
        """Load graph state"""
        # Load graph structure
        if (self.base_path / "graph.pt").exists():
            data = torch.load(self.base_path / "graph.pt", weights_only=False)
            self.nodes = data['nodes']
            self.edges = data['edges']
            
        # Load vectors
        if (self.base_path / "vectors.pt").exists():
            self.node_vectors = torch.load(self.base_path / "vectors.pt", weights_only=False)
            
        # Load fractal storage
        self.fractal_store.load_from_disk()
