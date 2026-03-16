#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Экспортирует данные из локальной директории модели в фрактальное хранилище и граф памяти.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("export_to_fractal.log")
    ]
)
logger = logging.getLogger(__name__)

class FractalLevel(Enum):
    """Уровни фрактальной структуры."""
    TOKENS = 0
    IMAGES = 1
    CONCEPTS = 2
    DOMAINS = 3

class FractalContainer:
    """Контейнер фрактального хранилища."""
    
    def __init__(
        self,
        id: str,
        level: FractalLevel,
        position: int,
        data_path: str,
        shape: Tuple[int, ...],
        dtype: str,
        metadata: Optional[Dict[str, Any]] = None,
        children: Optional[List[str]] = None
    ):
        self.id = id
        self.level = level
        self.position = position
        self.data_path = data_path
        self.shape = shape
        self.dtype = dtype
        self.metadata = metadata or {}
        self.children = children or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует контейнер в словарь."""
        return {
            "id": self.id,
            "level": self.level.value,
            "position": self.position,
            "data_path": self.data_path,
            "shape": self.shape,
            "dtype": self.dtype,
            "metadata": self.metadata,
            "children": self.children
        }

class FractalWeightStore:
    """Хранилище весов в фрактальной структуре."""
    
    def __init__(self, output_dir: str, block_size: int = 64, fractal_levels: int = 4):
        self.output_dir = Path(output_dir)
        self.block_size = block_size
        self.fractal_levels = fractal_levels
        self.containers: Dict[str, FractalContainer] = {}
        self.fractal_tree: Dict[int, List[str]] = {level: [] for level in range(fractal_levels)}
        self.hot_window: List[str] = []
        self.total_memory = 0
        self.model_id = ""
        
        # Создаем необходимые директории
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_container_id(self, layer_name: str, fragment_idx: int, level: int) -> str:
        """Генерирует уникальный ID для контейнера."""
        clean_layer_name = re.sub(r'[^a-zA-Z0-9_]', '_', layer_name)
        return f"{clean_layer_name}_L{level}_F{fragment_idx}"
    
    def _save_container_index(self) -> bool:
        """Сохраняет индекс контейнеров в файл."""
        try:
            index_path = self.output_dir / "containers.jsonl"
            with open(index_path, "w", encoding="utf-8") as f:
                for container in self.containers.values():
                    f.write(json.dumps(container.to_dict()) + "\n")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса контейнеров: {e}")
            return False

def export_to_fractal(
    model_dir: str,
    output_dir: str,
    model_id: str,
    task: str = "text-generation",
    device: str = "cpu",
    fractal_levels: int = 4,
    block_size: int = 64,
    local_files_only: bool = True,
) -> bool:
    """
    Экспортирует модель в фрактальное хранилище.
    
    Args:
        model_dir: Путь к директории модели
        output_dir: Путь к выходной директории
        model_id: Идентификатор модели
        task: Тип задачи
        
    Returns:
        bool: True если экспорт успешен, иначе False
    """
    try:
        from cogniflex.mlearning.storage.fractal_store import export_hf_model_to_fractal

        out_task_dir = os.path.join(output_dir, task)
        os.makedirs(out_task_dir, exist_ok=True)

        logger.info(f"Экспорт HF-модели из '{model_dir}' в фрактальное хранилище: '{out_task_dir}'")

        ok = export_hf_model_to_fractal(
            hf_model_dir_or_id=model_dir,
            output_path=out_task_dir,
            model_id=model_id,
            device=device,
            fractal_levels=int(fractal_levels),
            block_size=int(block_size),
            local_files_only=bool(local_files_only),
        )

        if ok:
            logger.info(f"Экспорт модели {model_id} успешно завершен")
        else:
            logger.error(f"Экспорт модели {model_id} завершился ошибкой")
        return bool(ok)
        
    except Exception as e:
        logger.error(f"Ошибка при экспорте модели: {e}", exc_info=True)
        return False

def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description="Экспорт модели в фрактальное хранилище"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Путь к директории модели"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Путь к выходной директории"
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Идентификатор модели"
    )
    parser.add_argument(
        "--task",
        default="text-generation",
        help="Тип задачи (по умолчанию: text-generation)"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство для экспорта (cpu или cuda). По умолчанию cpu"
    )

    parser.add_argument(
        "--fractal-levels",
        type=int,
        default=4,
        help="Количество уровней фрактала (по умолчанию: 4)"
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Размер блока (по умолчанию: 64)"
    )

    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Разрешает загрузку из сети (по умолчанию используются только локальные файлы)."
    )
    
    args = parser.parse_args()
    
    if not export_to_fractal(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        task=args.task,
        device=args.device,
        fractal_levels=args.fractal_levels,
        block_size=args.block_size,
        local_files_only=(not bool(args.allow_download))
    ):
        sys.exit(1)

if __name__ == "__main__":
    main()
