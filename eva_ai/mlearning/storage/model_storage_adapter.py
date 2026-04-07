"""
Integration module between ModelManager and MemoryGraphStore
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch import nn
from .memory_graph_store import MemoryGraphStore

logger = logging.getLogger(__name__)

class ModelStorageAdapter:
    """Adapter for integrating MemoryGraphStore with ModelManager"""
    
    def __init__(self, base_path: str, device: str = 'cpu'):
        self.storage = MemoryGraphStore(
            base_path=str(Path(base_path) / "model_storage"),
            block_size=64,
            fractal_levels=3,
            device=device
        )
        self.model_cache: Dict[str, nn.Module] = {}
        
    def store_model(self, model_id: str, model: nn.Module, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a model's state in the graph storage"""
        try:
            state = model.state_dict()
            success = True
            
            for key, tensor in state.items():
                tensor_id = f"{model_id}.{key}"
                if not self.storage.store_tensor(tensor_id, tensor, metadata=metadata):
                    success = False
                    break
                    
            if success:
                self.model_cache[model_id] = model
                
            return success
            
        except Exception as e:
            logger.error(f"Error storing model {model_id}: {str(e)}")
            return False
            
    def load_model(self, 
                  model_id: str,
                  model_class: Optional[type] = None,
                  model_args: Optional[Dict[str, Any]] = None) -> Optional[nn.Module]:
        """Load a model from graph storage"""
        try:
            if model_id in self.model_cache:
                return self.model_cache[model_id]
                
            # Load state dict
            state = {}
            for key in self.storage.memory_graph.get(model_id, {}).get("state_keys", []):
                tensor = self.storage.load_tensor(f"{model_id}.{key}")
                if tensor is not None:
                    state[key] = tensor
                    
            if not state:
                return None
                
            if model_class:
                model = model_class(**(model_args or {}))
                model.load_state_dict(state)
                self.model_cache[model_id] = model
                return model
                
            return None
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return None
            
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from storage"""
        try:
            if model_id in self.storage.memory_graph:
                for key in self.storage.memory_graph[model_id]["state_keys"]:
                    self.storage.remove_tensor(f"{model_id}.{key}")
                    
            if model_id in self.model_cache:
                del self.model_cache[model_id]
                
            return True
            
        except Exception as e:
            logger.error(f"Error removing model {model_id}: {str(e)}")
            return False
            
    def save_to_disk(self) -> bool:
        """Save all models to disk"""
        return self.storage.save_to_disk()
        
    def load_from_disk(self) -> bool:
        """Load all models from disk"""
        return self.storage.load_from_disk()
        
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata"""
        return self.storage.get_tensor_metadata(model_id)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            "cached_models": len(self.model_cache),
            "storage": self.storage.get_stats()
        }
