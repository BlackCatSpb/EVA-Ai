"""Fractal Weight Store Interface"""

import time
from typing import Any, Dict, Optional, Union, List
import torch
import numpy as np
from collections import OrderedDict
from .fractal_store import FractalContainer

class FractalWeightStore:
    """Extended FractalWeightStore with additional storage methods"""
    
    def __init__(self, block_size: int = 64, fractal_levels: int = 5, 
                 containers_per_group: int = 4, device: str = "cpu"):
        """Initialize fractal weight store"""
        self.block_size = block_size
        self.fractal_levels = fractal_levels
        self.containers_per_group = containers_per_group
        self.containers: Dict[str, FractalContainer] = {}
        self.fractal_tree: Dict[int, List[str]] = {}
        self.hot_window: OrderedDict[str, float] = OrderedDict()
        self.metadata: Dict[str, Any] = {}
        try:
            config_device = self.config.get('device', 'cpu') if hasattr(self, 'config') else device
            use_cuda = (config_device != "cpu") if isinstance(config_device, str) else False
            if use_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except Exception:
            self.device = "cpu"
        self.hot_window_size: int = 500 * 1024 * 1024
        self.total_memory: int = 0
    
    def store(self, key: str, value: Any) -> None:
        """Store a value in the fractal storage"""
        if isinstance(value, dict):
            self.metadata[key] = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    tensor_key = f"{key}.{k}"
                    self.store_tensor(tensor_key, v)
                    self.metadata[key][k] = {'type': 'tensor', 'key': tensor_key}
                else:
                    self.metadata[key][k] = v
        elif isinstance(value, torch.Tensor):
            self.store_tensor(key, value)
        else:
            self.metadata[key] = value
            
    def save_to_disk(self, output_path: str, knowledge_graph: Optional[Dict[str, Any]] = None) -> bool:
        """Сохраняет фрактальное хранилище на диск"""
        try:
            import pickle
            import json
            from pathlib import Path

            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)

            containers_file = out_dir / "containers.pkl"
            with open(containers_file, 'wb') as f:
                pickle.dump(self.containers, f)

            metadata_file = out_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            config_file = out_dir / "config.json"
            config = {
                "block_size": self.block_size,
                "fractal_levels": self.fractal_levels,
                "containers_per_group": self.containers_per_group,
                "device": self.device,
                "total_containers": len(self.containers)
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

            return True

        except Exception as e:
            print(f"Ошибка сохранения: {e}")
            return False

    def update_hot_window(self, key: str) -> None:
        """Update hot window with new access"""
        self.hot_window[key] = time.time()
        if len(self.hot_window) > 100:
            for _ in range(10):
                self.hot_window.popitem(last=False)

    def get(self, key: str) -> Any:
        """Get a stored value"""
        if key in self.metadata:
            value = self.metadata[key]
            if isinstance(value, dict):
                result = {}
                for k, v in value.items():
                    if isinstance(v, dict) and v.get('type') == 'tensor':
                        result[k] = self.get_tensor_by_key(v['key'])
                    else:
                        result[k] = v
                return result
            return value
        elif key in self.containers:
            return self.get_tensor_by_key(key)
        return None

    def get_tensor_by_key(self, key: str) -> Optional[torch.Tensor]:
        """Get a stored tensor"""
        if key not in self.containers:
            return None
        container = self.containers[key]
        data = torch.from_numpy(container.data)
        if self.device == 'cuda':
            data = data.cuda()
        return data

    def store_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Store a tensor value"""
        data = tensor.detach().cpu().numpy()
        container = FractalContainer(
            id=key,
            level=0,
            position=(0,) * len(data.shape),
            data=data,
            shape=data.shape,
            dtype=str(data.dtype)
        )
        self.containers[key] = container
        self.update_hot_window(key)

        
    def update_hot_window(self, key: str) -> None:
        """Update hot window with new access"""
        self.hot_window[key] = time.time()
        if len(self.hot_window) > 100:
            for _ in range(10):
                self.hot_window.popitem(last=False)
                
    def get(self, key: str) -> Any:
        """Get a stored value"""
        if key in self.metadata:
            value = self.metadata[key]
            if isinstance(value, dict):
                result = {}
                for k, v in value.items():
                    if isinstance(v, dict) and v.get('type') == 'tensor':
                        result[k] = self.get_tensor(v['key'])
                    else:
                        result[k] = v
                return result
            return value
        elif key in self.containers:
            return self.get_tensor(key)
        return None
