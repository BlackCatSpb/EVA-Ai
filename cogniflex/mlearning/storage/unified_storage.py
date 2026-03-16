"""
Unified Fractal Storage for CogniFlex
Combines knowledge storage, model weights, and learning capabilities
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union, List, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn

logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """Configuration for unified storage"""
    base_path: str
    device: str = 'cpu'
    block_size: int = 64
    fractal_levels: int = 3
    cache_size: int = 10000

class UnifiedFractalStorage:
    """
    Unified storage system that combines:
    - Knowledge graph functionality
    - Model weight storage
    - Vector representations
    - Learning capabilities
    """
    def __init__(self, config: StorageConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup storage paths
        self.base_path = Path(config.base_path)
        self.nodes_path = self.base_path / "nodes"
        self.edges_path = self.base_path / "edges"
        self.models_path = self.base_path / "models"
        
        for path in [self.nodes_path, self.edges_path, self.models_path]:
            os.makedirs(path, exist_ok=True)
            
        # Initialize storage components
        self.nodes: Dict[str, Dict] = {}  # Node metadata
        self.edges: List[Dict] = []  # Edge data
        self.node_vectors: Dict[str, torch.Tensor] = {}  # Node embeddings
        self.model_weights: Dict[str, Dict[str, torch.Tensor]] = {}  # Model parameters
        
        # ML components
        self.models: Dict[str, nn.Module] = {}  # Active models
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        
    def store_knowledge(self, content: Union[str, Dict],
                       node_type: str = "concept",
                       vector: Optional[torch.Tensor] = None,
                       metadata: Optional[Dict] = None) -> str:
        """Store knowledge in the unified storage"""
        # Generate node ID
        node_id = f"node_{len(self.nodes)}"
        
        # Create node record
        node = {
            'id': node_id,
            'type': node_type,
            'content': content,
            **(metadata or {})
        }
        
        # Store node data
        self.nodes[node_id] = node
        
        # Store vector if provided
        if vector is not None:
            self.node_vectors[node_id] = vector.to(self.device)
            
            # Save vector to disk
            torch.save(vector.cpu(), self.nodes_path / f"{node_id}.pt")
            
        # Save metadata
        with open(self.nodes_path / f"{node_id}.json", 'w') as f:
            json.dump(node, f)
            
        return node_id
        
    def connect_knowledge(self, source_id: str, target_id: str,
                         relation_type: str,
                         metadata: Optional[Dict] = None):
        """Create a connection between knowledge nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node not found")
            
        edge = {
            'source': source_id,
            'target': target_id,
            'type': relation_type,
            **(metadata or {})
        }
        
        self.edges.append(edge)
        
    def query_knowledge(self, query: Union[str, torch.Tensor],
                       limit: int = 10) -> List[Dict]:
        """Query knowledge using semantic similarity"""
        if isinstance(query, str):
            # TODO: Implement text vectorization
            return []
            
        query_vector = query.to(self.device)
        
        # Calculate similarities
        similarities = []
        for node_id, vector in self.node_vectors.items():
            sim = torch.nn.functional.cosine_similarity(
                query_vector.flatten(),
                vector.flatten(),
                dim=0
            )
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
        
    def register_model(self, model_id: str,
                      model: nn.Module,
                      optimizer: Optional[torch.optim.Optimizer] = None):
        """Register a model for training"""
        self.models[model_id] = model.to(self.device)
        
        if optimizer:
            self.optimizers[model_id] = optimizer
        else:
            self.optimizers[model_id] = torch.optim.Adam(
                model.parameters()
            )
            
    def save_model(self, model_id: str):
        """Save model state to storage"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.models[model_id]
        state_dict = model.state_dict()
        
        # Save model weights
        torch.save(state_dict,
                  self.models_path / f"{model_id}.pt")
                  
        # Save optimizer state
        if model_id in self.optimizers:
            torch.save(
                self.optimizers[model_id].state_dict(),
                self.models_path / f"{model_id}_optimizer.pt"
            )
            
    def load_model(self, model_id: str,
                  model: nn.Module,
                  load_optimizer: bool = True) -> nn.Module:
        """Load model state from storage"""
        model_path = self.models_path / f"{model_id}.pt"
        if not model_path.exists():
            raise ValueError(f"No saved state for model {model_id}")
            
        # Load weights
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        self.models[model_id] = model
        
        # Load optimizer if requested
        if load_optimizer:
            opt_path = self.models_path / f"{model_id}_optimizer.pt"
            if opt_path.exists():
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.load_state_dict(
                    torch.load(opt_path)
                )
                self.optimizers[model_id] = optimizer
                
        return model
        
    def train_step(self, model_id: str,
                  batch: Dict[str, torch.Tensor],
                  criterion: nn.Module) -> float:
        """Perform single training step"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.models[model_id]
        optimizer = self.optimizers[model_id]
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    def get_training_data(self,
                         node_types: Optional[List[str]] = None
                         ) -> List[Dict]:
        """Get training data from knowledge graph"""
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
                
            # Get node vector
            vector = self.node_vectors[node_id]
            
            # Get connected nodes
            connected = [
                e['target'] for e in self.edges
                if e['source'] == node_id
            ]
            
            if not connected:
                continue
                
            # Get target vectors
            target_vectors = []
            for target_id in connected:
                if target_id in self.node_vectors:
                    target_vectors.append(
                        self.node_vectors[target_id]
                    )
                    
            if not target_vectors:
                continue
                
            # Average connected vectors
            target = torch.stack(target_vectors).mean(dim=0)
            
            data.append({
                'input': vector,
                'target': target,
                'metadata': node
            })
            
        return data
        
    def save_state(self):
        """Save full storage state"""
        # Save nodes
        for node_id, node in self.nodes.items():
            with open(self.nodes_path / f"{node_id}.json", 'w') as f:
                json.dump(node, f)
                
        # Save edges
        with open(self.edges_path / "edges.json", 'w') as f:
            json.dump(self.edges, f)
            
        # Save vectors
        for node_id, vector in self.node_vectors.items():
            torch.save(vector.cpu(),
                      self.nodes_path / f"{node_id}.pt")
                      
        # Save models
        for model_id in self.models:
            self.save_model(model_id)
            
    def load_state(self):
        """Load full storage state"""
        # Load nodes
        for path in self.nodes_path.glob("*.json"):
            node_id = path.stem
            with open(path) as f:
                self.nodes[node_id] = json.load(f)
                
        # Load vectors
        for path in self.nodes_path.glob("*.pt"):
            node_id = path.stem
            vector = torch.load(path)
            self.node_vectors[node_id] = vector.to(self.device)
            
        # Load edges
        edge_path = self.edges_path / "edges.json"
        if edge_path.exists():
            with open(edge_path) as f:
                self.edges = json.load(f)
