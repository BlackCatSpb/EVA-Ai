"""
Configuration for model storage system
"""

import os
import logging
import torch
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class ModelStorageConfig:
    """Configuration for model storage using fractal storage system"""
    base_path: str
    block_size: int = 64
    fractal_levels: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    compression_level: int = 2
    cache_size_gb: float = 4.0
    prefetch_enabled: bool = True
    cleanup_threshold_gb: float = 8.0
    
    # Storage paths configuration
    fractal_store_dir: str = "fractal_store"  # Only fractal storage is used
    
    # Model-specific settings
    model_map: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        # Initialize with empty model map - all models will be loaded from fractal storage
        if self.model_map is None:
            self.model_map = {}
            
        # Ensure base directory exists
        os.makedirs(self.base_path, exist_ok=True)
        
        # Initialize fractal store directory
        os.makedirs(os.path.join(self.base_path, self.fractal_store_dir), exist_ok=True)
        
        # Set default device
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.device = 'cpu'
            logger.warning("CUDA not available, falling back to CPU")
