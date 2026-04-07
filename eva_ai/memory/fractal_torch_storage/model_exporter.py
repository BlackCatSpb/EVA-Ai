"""
Model Exporter - универсальный модуль экспорта моделей в фрактальное хранилище.
Поддерживает Qwen и другие модели на базе transformers.
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
from .base_storage import FractalWeightStorage
from .weight_index import WeightIndex

logger = logging.getLogger("eva_ai.memory.fractal_torch_storage.exporter")


class ModelExporter:
    """
    Универсальный экспортер моделей в фрактальное хранилище.
    
    Workflow:
    1. Загружает модель из стандартного пути
    2. Извлекает все веса и метаданные
    3. Сохраняет в фрактальное хранилище
    4. Экспортирует токенизатор и конфигурацию
    
    Поддерживаемые форматы:
    - HuggingFace Transformers
    - Qwen (Qwen2, Qwen2.5)
    - Llama
    """
    
    def __init__(self, export_dir: Optional[str] = None):
        """
        Args:
            export_dir: Директория для экспорта
        """
        self.export_dir = export_dir or os.path.join(
            os.path.dirname(__file__), "exported_models"
        )
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Инициализируем хранилище
        self.storage = FractalWeightStorage(
            storage_dir=self.export_dir,
            max_cache_size_gb=50.0  # Для больших моделей
        )
        
        logger.info(f"ModelExporter инициализирован: export_dir={self.export_dir}")
    
    def export_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        quantization: Optional[str] = None,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        Экспортирует модель в фрактальное хранилище.
        
        Args:
            model_path: Путь к модели (HuggingFace или локальный)
            model_name: Имя модели (для логов)
            quantization: Тип квантизации (None, "int8", "int4")
            device: Устройство загрузки
            
        Returns:
            Dict: Результат экспорта
        """
        start_time = time.time()
        
        if model_name is None:
            model_name = os.path.basename(model_path)
        
        logger.info(f"Экспорт модели: {model_name} из {model_path}")
        
        result = {
            "model_name": model_name,
            "model_path": model_path,
            "status": "started",
            "layers_exported": 0,
            "total_weights": 0,
            "total_bytes": 0,
            "config": {},
            "tokenizer": {},
            "errors": []
        }
        
        try:
            # 1. Загружаем конфигурацию
            config = self._load_config(model_path)
            result["config"] = config
            logger.info(f"Конфигурация загружена: {len(config)} параметров")
            
            # 2. Загружаем модель
            model = self._load_model(model_path, quantization, device)
            if model is None:
                result["status"] = "error"
                result["errors"].append("Не удалось загрузить модель")
                return result
            
            # 3. Экспортируем веса
            weights_result = self._export_weights(model, model_name)
            result["layers_exported"] = weights_result["layers"]
            result["total_weights"] = weights_result["weights"]
            result["total_bytes"] = weights_result["bytes"]
            
            # 4. Экспортируем токенизатор
            tokenizer = self._load_tokenizer(model_path)
            result["tokenizer"] = tokenizer
            
            # 5. Сохраняем метаданные
            metadata = {
                "model_name": model_name,
                "model_path": model_path,
                "quantization": quantization,
                "device": device,
                "export_time": time.time() - start_time,
                "layers": result["layers_exported"],
                "weights": result["total_weights"],
                "bytes": result["total_bytes"],
                "config": config,
                "tokenizer": tokenizer
            }
            
            metadata_path = os.path.join(self.export_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 6. Сохраняем хранилище
            self.storage.save_to_disk(f"{model_name}_weights.bin")
            
            result["status"] = "success"
            result["export_time"] = time.time() - start_time
            result["metadata_path"] = metadata_path
            
            logger.info(
                f"Экспорт завершён: {result['layers_exported']} слоёв, "
                f"{result['total_weights']} весов, "
                f"{result['total_bytes'] / (1024**3):.2f} GB, "
                f"{result['export_time']:.1f}s"
            )
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Ошибка экспорта: {e}", exc_info=True)
        
        return result
    
    def _load_config(self, model_path: str) -> Dict:
        """Загружает конфигурацию модели."""
        config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json"
        ]
        
        config = {}
        
        for config_file in config_files:
            config_path = os.path.join(model_path, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        config[config_file.replace('.json', '')] = data
                except Exception as e:
                    logger.warning(f"Ошибка загрузки {config_file}: {e}")
        
        return config
    
    def _load_model(self, model_path: str, quantization: Optional[str], device: str):
        """Загружает модель."""
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            
            # Определяем устройство
            if device == "auto":
                if torch.cuda.is_available():
                    device_map = "auto"
                else:
                    device_map = "cpu"
            else:
                device_map = device
            
            load_kwargs["device_map"] = device_map
            
            # Квантизация через BitsAndBytesConfig
            if quantization == "int8":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                load_kwargs["quantization_config"] = bnb_config
            elif quantization == "int4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                load_kwargs["quantization_config"] = bnb_config
            
            logger.info(f"Загрузка модели с device_map={device_map}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return None
    
    def _export_weights(self, model, model_name: str) -> Dict:
        """Экспортирует веса модели."""
        layers = 0
        weights = 0
        bytes_total = 0
        
        for name, param in model.named_parameters():
            try:
                # Определяем слой и тензор
                parts = name.split(".")
                if len(parts) >= 2:
                    layer_name = ".".join(parts[:-1])
                    tensor_name = parts[-1]
                else:
                    layer_name = "model"
                    tensor_name = name
                
                # Конвертируем в numpy
                tensor_data = param.detach().cpu()
                
                # Сохраняем в хранилище
                key = self.storage.store_weight(
                    layer_name=f"{model_name}.{layer_name}",
                    tensor_name=tensor_name,
                    data=tensor_data.numpy().tobytes(),
                    shape=tuple(tensor_data.shape),
                    dtype=str(tensor_data.dtype)
                )
                
                weights += 1
                bytes_total += tensor_data.numel() * tensor_data.element_size()
                
                # Считаем уникальные слои
                if tensor_name == "weight":
                    layers += 1
                
            except Exception as e:
                logger.warning(f"Ошибка экспорта {name}: {e}")
        
        return {
            "layers": layers,
            "weights": weights,
            "bytes": bytes_total
        }
    
    def _load_tokenizer(self, model_path: str) -> Dict:
        """Загружает токенизатор."""
        tokenizer = {}
        
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"
        ]
        
        for file_name in tokenizer_files:
            file_path = os.path.join(model_path, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tokenizer[file_name] = f.read()
                except Exception as e:
                    logger.warning(f"Ошибка загрузки {file_name}: {e}")
        
        return tokenizer
    
    def import_model(self, model_name: str) -> Optional[Dict]:
        """
        Импортирует модель из фрактального хранилища.
        
        Args:
            model_name: Имя модели
            
        Returns:
            Optional[Dict]: Модель и токенизатор или None
        """
        try:
            # Загружаем метаданные
            metadata_path = os.path.join(self.export_dir, f"{model_name}_metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.error(f"Метаданные не найдены: {metadata_path}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Загружаем хранилище
            self.storage.load_from_disk(f"{model_name}_weights.bin")
            
            logger.info(f"Модель {model_name} импортирована из фрактального хранилища")
            
            return {
                "metadata": metadata,
                "storage": self.storage
            }
            
        except Exception as e:
            logger.error(f"Ошибка импорта: {e}")
            return None
    
    def list_exported_models(self) -> List[str]:
        """Список экспортированных моделей."""
        models = []
        
        for file_name in os.listdir(self.export_dir):
            if file_name.endswith("_metadata.json"):
                model_name = file_name.replace("_metadata.json", "")
                models.append(model_name)
        
        return models
    
    def get_export_stats(self, model_name: str) -> Optional[Dict]:
        """Статистика экспортированной модели."""
        metadata_path = os.path.join(self.export_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        storage_stats = self.storage.get_stats()
        
        return {
            "model_name": model_name,
            "layers": metadata.get("layers", 0),
            "weights": metadata.get("weights", 0),
            "bytes": metadata.get("bytes", 0),
            "gb": metadata.get("bytes", 0) / (1024**3),
            "quantization": metadata.get("quantization"),
            "storage_stats": storage_stats
        }
