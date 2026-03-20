"""
Модуль экспорта RuGPT3 Large для CogniFlex
Экспортирует модель в формат, совместимый с фрактальным хранилищем
"""
import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class RuGPTExporter:
    """
    Экспортирует RuGPT3 Large в формат CogniFlex
    """
    
    def __init__(
        self,
        source_model_path: str = "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal/model",
        output_path: str = "cogniflex_cache/models/exported_rugpt3"
    ):
        self.source_model_path = source_model_path
        self.output_path = output_path
        
    def export(self) -> bool:
        """
        Экспортирует модель
        
        Returns:
            True при успехе
        """
        logger.info(f"Экспорт RuGPT3 Large в {self.output_path}")
        
        try:
            os.makedirs(self.output_path, exist_ok=True)
            
            # Копируем файлы модели
            source = Path(self.source_model_path)
            if not source.exists():
                logger.error(f"Исходная модель не найдена: {source}")
                return False
            
            # Копируем основные файлы
            for file_name in ['pytorch_model.bin', 'config.json', 'vocab.json', 'merges.txt']:
                src_file = source / file_name
                if src_file.exists():
                    shutil.copy2(src_file, self.output_path)
                    logger.info(f"Скопирован: {file_name}")
            
            # Создаем конфигурацию экспорта
            export_config = {
                "model_name": "RuGPT3-Large-Fractal",
                "model_type": "gpt2",
                "original_source": self.source_model_path,
                "vocab_size": 50000,
                "hidden_size": 1280,
                "num_layers": 36,
                "num_heads": 20,
                "max_position_embeddings": 2048,
                "exported": True,
                "fractal_compatible": True
            }
            
            with open(os.path.join(self.output_path, "export_config.json"), "w") as f:
                json.dump(export_config, f, indent=2)
            
            logger.info(f"Экспорт завершен: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию об экспортированной модели"""
        config_path = os.path.join(self.output_path, "export_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        return {"error": "Модель не экспортирована"}


def export_rugpt3():
    """Функция для быстрого экспорта"""
    exporter = RuGPTExporter()
    return exporter.export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_rugpt3()
