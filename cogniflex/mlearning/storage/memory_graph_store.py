"""
Memory Graph Storage Module for CogniFlex
Implements graph-based memory storage with fractal compression
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from .fractal_store import FractalWeightStore

logger = logging.getLogger(__name__)

class MemoryGraphStore:
    """
    Implements unified graph-based memory storage with fractal compression and ML capabilities
    """
    
    def __init__(self, 
                 base_path: str,
                 block_size: int = 64,
                 fractal_levels: int = 3,
                 device: str = 'cpu'):
        """
        Initialize the memory graph store
        
        Args:
            base_path: Base path for storage
            block_size: Size of fractal blocks
            fractal_levels: Number of fractal compression levels
            device: Device to store tensors on
        """
        self.base_path = Path(base_path)
        self.device = device
        self.block_size = block_size
        self.fractal_levels = fractal_levels
        
        # Create storage directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.base_path / "nodes", exist_ok=True)
        os.makedirs(self.base_path / "edges", exist_ok=True)
        
        # Initialize fractal storages
        self.node_store = FractalWeightStore(
            block_size=block_size,
            fractal_levels=fractal_levels,
            device=device
        )
        
        self.edge_store = FractalWeightStore(
            block_size=block_size,
            fractal_levels=fractal_levels,
            device=device
        )
        
        self.fractal_store = self.node_store
        self.memory_graph: Dict[str, Any] = {}
        
        # Memory graph structure with vector storage
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.node_vectors: Dict[str, torch.Tensor] = {}
        
        # Stats
        self.stats = {
            'nodes': 0,
            'edges': 0,
            'total_size': 0
        }
        
    def add_node(self, content: Union[str, Dict],
                node_type: str,
                vector: Optional[torch.Tensor] = None,
                metadata: Optional[Dict] = None) -> str:
        """
        Add a node to the graph with optional vector representation
        
        Args:
            content: Node content (text or structured data)
            node_type: Type of node
            vector: Vector representation 
            metadata: Additional metadata
        """
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
            self.node_vectors[node_id] = vector.to(self.device)
            self.node_store.store_tensor(
                node_id,
                vector,
                metadata=node
            )
            
        self.stats['nodes'] += 1
        return node_id
        
    def add_edge(self, source: str, target: str,
                edge_type: str,
                metadata: Optional[Dict] = None):
        """Add an edge between nodes"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target node not found")
            
        edge = {
            'source': source,
            'target': target, 
            'type': edge_type,
            **(metadata or {})
        }
        
        self.edges.append(edge)
        self.stats['edges'] += 1
        
    def get_node_vector(self, node_id: str) -> Optional[torch.Tensor]:
        """Get vector representation of a node"""
        return self.node_vectors.get(node_id)
        
    def find_similar_nodes(self, vector: torch.Tensor,
                          limit: int = 10,
                          threshold: float = 0.5) -> List[Dict]:
        """Find nodes with similar vectors"""
        if not self.node_vectors:
            return []
            
        results = self.node_store.get_similar_tensors(
            vector,
            limit=limit
        )
        
        similar_nodes = []
        for res in results:
            if res['similarity'] >= threshold:
                node_id = res['id']
                similar_nodes.append({
                    **self.nodes[node_id],
                    'similarity': res['similarity']
                })
                
        return similar_nodes
        
    def get_connected_nodes(self, node_id: str,
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
        
    def get_training_data(self, node_types: Optional[List[str]] = None) -> List[Dict]:
        """Get training data from graph structure"""
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
            
            # Get connected nodes as targets
            connected = self.get_connected_nodes(node_id)
            if not connected:
                continue
                
            # Average connected node vectors as target
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
                'id': node_id,
                'vector': vector,
                'target': target,
                'metadata': node
            })
            
        return data
        
    def store_tensor(self, 
                    tensor_id: str,
                    tensor: Tensor,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a tensor in the memory graph with fractal compression
        
        Args:
            tensor_id: Unique ID for the tensor
            tensor: The tensor to store
            metadata: Optional metadata about the tensor
            
        Returns:
            bool: Success status
        """
        try:
            # Create minimal wrapper model to use fractal store
            wrapper = torch.nn.Module()
            wrapper.weight = torch.nn.Parameter(tensor)
            
            # Pack into fractal store
            success = self.fractal_store.pack_model_weights(
                wrapper,
                model_id=tensor_id
            )
            
            if success:
                # Add to memory graph
                self.memory_graph[tensor_id] = {
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "metadata": metadata or {},
                    "compressed": True
                }
                
                # Update stats
                self.stats["total_tensors"] += 1
                self.stats["original_size"] += tensor.numel() * tensor.element_size()
                self.stats["compressed_size"] += self._get_compressed_size(tensor_id)
                self._update_compression_ratio()
                
                logger.debug(f"Stored tensor {tensor_id} in memory graph")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error storing tensor {tensor_id}: {str(e)}")
            return False
            
    def load_tensor(self, tensor_id: str) -> Optional[Tensor]:
        """
        Load a tensor from the memory graph
        
        Args:
            tensor_id: ID of tensor to load
            
        Returns:
            Optional[Tensor]: The loaded tensor or None if not found
        """
        try:
            if tensor_id not in self.memory_graph:
                logger.warning(f"Tensor {tensor_id} not found in memory graph")
                return None
                
            # Get metadata
            meta = self.memory_graph[tensor_id]
            
            # Create wrapper model to use fractal store 
            wrapper = torch.nn.Module()
            wrapper.weight = torch.nn.Parameter(
                torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"].split(".")[-1]))
            )
            
            # Unpack from fractal store
            success = self.fractal_store.unpack_model_weights(
                wrapper,
                model_id=tensor_id
            )
            
            if success:
                return wrapper.weight.data
                
            return None
            
        except Exception as e:
            logger.error(f"Error loading tensor {tensor_id}: {str(e)}")
            return None
            
    def remove_tensor(self, tensor_id: str) -> bool:
        """
        Remove a tensor from the memory graph
        
        Args:
            tensor_id: ID of tensor to remove
            
        Returns:
            bool: Success status
        """
        try:
            if tensor_id not in self.memory_graph:
                return False
                
            # Remove from fractal store
            self.fractal_store.remove_model(tensor_id)
            
            # Update stats
            meta = self.memory_graph[tensor_id]
            shape = meta["shape"]
            dtype = getattr(torch, meta["dtype"].split(".")[-1])
            numel = np.prod(shape)
            self.stats["total_tensors"] -= 1
            self.stats["original_size"] -= numel * torch._utils._element_size(dtype)
            self.stats["compressed_size"] -= self._get_compressed_size(tensor_id)
            self._update_compression_ratio()
            
            # Remove from graph
            del self.memory_graph[tensor_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing tensor {tensor_id}: {str(e)}")
            return False
            
    def get_tensor_metadata(self, tensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a tensor
        
        Args:
            tensor_id: ID of tensor
            
        Returns:
            Optional[Dict]: Tensor metadata or None if not found
        """
        return self.memory_graph.get(tensor_id)
            
    def save_to_disk(self, compress: bool = True) -> bool:
        """
        Save the entire memory graph to disk
        
        Args:
            compress: Whether to use compression
            
        Returns:
            bool: Success status
        """
        try:
            # Save fractal store
            self.fractal_store.save_to_disk_sharded(
                str(self.base_path / "fractal_store"),
                shard_size=50,
                by_level=True,
                compress=compress
            )
            
            # Save memory graph structure
            torch.save(
                self.memory_graph,
                self.base_path / "memory_graph.pt"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to disk: {str(e)}")
            return False
            
    def load_from_disk(self) -> bool:
        """
        Load the memory graph from disk
        
        Returns:
            bool: Success status 
        """
        try:
            # Load memory graph structure
            graph_path = self.base_path / "memory_graph.pt"
            if not graph_path.exists():
                return False
                
            self.memory_graph = torch.load(graph_path)
            
            # Load fractal store
            self.fractal_store.load_from_disk_sharded(
                str(self.base_path / "fractal_store")
            )
            
            # Recalculate stats
            self._recalculate_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading from disk: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self.stats.copy()
        
    def _get_compressed_size(self, tensor_id: str) -> int:
        """Get compressed size of a tensor"""
        # Get from fractal store stats
        return self.fractal_store.get_model_stats(tensor_id).get("compressed_size", 0)
        
    def _update_compression_ratio(self):
        """Update compression ratio stat"""
        if self.stats["original_size"] > 0:
            self.stats["compression_ratio"] = (
                self.stats["compressed_size"] / self.stats["original_size"]
            )
            
    def _recalculate_stats(self):
        """Recalculate all stats"""
        self.stats = {
            "total_tensors": len(self.memory_graph),
            "compressed_size": sum(
                self._get_compressed_size(tid) for tid in self.memory_graph
            ),
            "original_size": sum(
                np.prod(meta["shape"]) * torch._utils._element_size(
                    getattr(torch, meta["dtype"].split(".")[-1])
                )
                for meta in self.memory_graph.values()
            ),
            "compression_ratio": 0.0
        }
        self._update_compression_ratio()
