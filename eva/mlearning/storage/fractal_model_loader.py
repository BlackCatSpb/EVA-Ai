"""
Model loader that uses fractal storage for ЕВА
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import torch
from torch import nn
from .fractal_store import FractalWeightStore
from .memory_graph_store import MemoryGraphStore
from .model_storage_config import ModelStorageConfig

logger = logging.getLogger(__name__)

class FractalModelLoader:
    """Handles loading of models using fractal storage"""
    
    def __init__(self, config: ModelStorageConfig):
        self.config = config
        self.device = config.device
        
        # Ensure base directories exist
        store_path = Path(config.base_path) / config.fractal_store_dir
        os.makedirs(store_path, exist_ok=True)
        os.makedirs(store_path / "fractal_store", exist_ok=True)
        
        self.store = MemoryGraphStore(
            base_path=str(store_path),
            block_size=config.block_size,
            fractal_levels=config.fractal_levels,
            device=config.device
        )
        
    def has_model(self, model_id: str) -> bool:
        """Check if a model exists in storage"""
        try:
            # Check if base directories exist
            store_path = Path(self.config.base_path) / self.config.fractal_store_dir
            if not store_path.exists():
                os.makedirs(store_path, exist_ok=True)
            
            # Initialize fractal store structure if needed
            if not (store_path / "fractal_store").exists():
                os.makedirs(store_path / "fractal_store", exist_ok=True)
                logger.info(f"Created fractal store structure at {store_path}")
                return False  # New store, no models yet
            
            # Check for model index file
            index_path = store_path / f"{model_id}.index"
            if not index_path.exists():
                return False
                
            # Load index
            with open(index_path) as f:
                tensor_ids = f.read().splitlines()
                
            # Check all tensors exist
            for tensor_id in tensor_ids:
                if not self.store.has_tensor(tensor_id):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking model {model_id}: {str(e)}")
            return False

    def list_models(self) -> List[str]:
        """List all models in the fractal storage."""
        try:
            store_path = Path(self.config.base_path) / self.config.fractal_store_dir
            if not store_path.exists():
                return []
            
            model_ids = []
            for f in store_path.glob("*.index"):
                model_ids.append(f.stem)
            return model_ids
        except Exception as e:
            logger.error(f"Error listing models in fractal storage: {str(e)}")
            return []

    def load_model(self, model_id: str, model_class: Any = None) -> Optional[nn.Module]:
        """Load a model from fractal storage"""
        try:
            # Ensure store is initialized
            store_path = Path(self.config.base_path) / self.config.fractal_store_dir
            if not store_path.exists():
                logger.error(f"Fractal store path {store_path} does not exist")
                return None
                
            # Check if model exists
            if not self.has_model(model_id):
                logger.info(f"Model {model_id} not found in fractal storage")
                return None
                
            # Get model-specific settings
            model_config = self.config.model_map.get(model_id, {})
            
            # Create empty model instance
            if model_class is None:
                model = self._create_default_model()
            else:
                model = model_class()
                
            # Load state dict from fractal storage
            state_dict = {}
            success = True
            
            for name in model.state_dict().keys():
                tensor_id = f"{model_id}.{name}"
                tensor = self.store.load_tensor(tensor_id)
                if tensor is None:
                    success = False
                    break
                state_dict[name] = tensor
                
            if not success:
                return None
                
            # Load state dict into model
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return None
            
    def save_model(self, model_id: str, model: nn.Module, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a model to fractal storage"""
        try:
            success = True
            
            # Get state dict
            state_dict = model.state_dict()
            
            # Save each tensor
            for name, tensor in state_dict.items():
                tensor_id = f"{model_id}.{name}"
                if not self.store.store_tensor(tensor_id, tensor, metadata=metadata):
                    success = False
                    break
                    
            return success
            
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {str(e)}")
            return False
            
    def _create_default_model(self) -> nn.Module:
        """Creates a default empty model if no class specified"""
        return nn.Sequential(nn.Linear(1, 1))
