"""
Unified Fractal Storage for ЕВА
Combines knowledge graph, model storage, and learning capabilities
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List, Union
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn

from .fractal_store import FractalWeightStore, KnowledgeGraphProxy
from .memory_graph_store import MemoryGraphStore
from .model_storage_config import ModelStorageConfig

logger = logging.getLogger(__name__)

@dataclass
class UnifiedStorageConfig:
    """Configuration for unified fractal storage"""
    base_path: str
    device: str = 'cpu'
    block_size: int = 64
    fractal_levels: int = 3
    cache_size: int = 10000
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 1000

class UnifiedFractalStore:
    """
    Unified storage combining knowledge graph, model weights, and learning capabilities
    """
    def __init__(self, config: UnifiedStorageConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Ensure base directories exist
        self.base_path = Path(config.base_path)
        self.store_path = self.base_path / "unified_store"
        os.makedirs(self.store_path, exist_ok=True)
        
        # Initialize components
        self.weight_store = FractalWeightStore(
            base_path=str(self.store_path),
            block_size=config.block_size,
            device=config.device
        )
        
        self.graph_store = MemoryGraphStore(
            base_path=str(self.store_path / "graph"),
            block_size=config.block_size,
            fractal_levels=config.fractal_levels,
            device=config.device
        )
        
        # Training setup
        self.optimizer = None
        self.current_model = None
        
    def _vectorize_text(self, text: str) -> np.ndarray:
        """Векторизация текста с использованием TF-IDF или случайных векторов"""
        try:
            # Пробуем использовать TF-IDF
            if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer:
                vector = self.tfidf_vectorizer.transform([text]).toarray()[0]
                # Дополняем до нужного размера
                if len(vector) < self.config.block_size:
                    vector = np.pad(vector, (0, self.config.block_size - len(vector)))
                return vector
        except Exception:
            pass
        
        # Fallback: хеширование текста + случайный шум
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        vector = np.random.randn(self.config.block_size) * 0.1
        return vector
    
    def store_knowledge(self, content: Union[str, Dict], 
                       node_type: str = "concept",
                       metadata: Optional[Dict] = None) -> str:
        """Store knowledge in the graph with fractal encoding"""
        # Convert content to vector representation
        if isinstance(content, str):
            vector = self._vectorize_text(content)
        else:
            vector = np.array(list(content.values()))
            
        # Store in graph
        node_id = self.graph_store.add_node(
            content=content,
            node_type=node_type,
            vector=vector,
            metadata=metadata or {}
        )
        
        return node_id
        
    def query_knowledge(self, query: Union[str, Dict], 
                       limit: int = 10) -> List[Dict]:
        """Query knowledge using semantic similarity"""
        if isinstance(query, str):
            vector = self._vectorize_text(query)
        else:
            vector = np.array(list(query.values()))
            
        results = self.graph_store.search_nodes(
            vector=vector,
            limit=limit
        )
        
        return results
        
    def train_on_knowledge(self, 
                          model: nn.Module,
                          criterion: nn.Module,
                          node_types: Optional[List[str]] = None):
        """Train model on knowledge graph data"""
        self.current_model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Get training data from graph
        train_data = self.graph_store.get_training_data(
            node_types=node_types
        )
        
        for epoch in range(self.config.max_epochs):
            total_loss = 0
            batches = 0
            
            for batch in self._get_batches(train_data):
                loss = self._train_step(batch, criterion)
                total_loss += loss
                batches += 1
                
            avg_loss = total_loss / batches
            logger.info(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")
    
    def _get_batches(self, data: List[Dict]):
        """Generate training batches"""
        batch_size = self.config.batch_size
        data_len = len(data)
        for i in range(0, data_len, batch_size):
            batch = data[i:min(i + batch_size, data_len)]
            yield [item for item in batch if isinstance(item.get("content"), (dict, list)) or "vector" in item]
            
    def _train_step(self, batch: List[Dict], 
                   criterion: nn.Module) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Prepare batch data
        inputs = []
        targets = []
        for item in batch:
            if not isinstance(item.get("content"), (dict, list)):
                continue
            if "vector" not in item or "target" not in item:
                continue
            try:
                x = torch.tensor(item["vector"], 
                               device=self.device)
                y = torch.tensor(item["target"], 
                               device=self.device)
                inputs.append(x)
                targets.append(y)
            except Exception as e:
                logger.warning(f"Skipping invalid training item: {e}")
                
        if not inputs:
            return 0.0
            
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        # Forward pass
        outputs = self.current_model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def save_model_state(self, model_id: str):
        """Save model state to fractal storage"""
        if self.current_model is None:
            raise ValueError("No model currently loaded")
            
        state_dict = self.current_model.state_dict()
        self.weight_store.save_weights(
            model_id=model_id,
            weights=state_dict
        )
        
    def load_model_state(self, model_id: str):
        """Load model state from fractal storage"""
        if self.current_model is None:
            raise ValueError("No model currently loaded")
            
        state_dict = self.weight_store.load_weights(model_id)
        self.current_model.load_state_dict(state_dict)
