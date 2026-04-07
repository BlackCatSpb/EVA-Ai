"""
Расширенный токенизатор с поддержкой фрактальных метаданных.
"""
import os
import json
import logging
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from transformers import PreTrainedTokenizerFast, AutoTokenizer

logger = logging.getLogger("eva_ai.tokenizer")

class ExtendedFractalTokenizer(PreTrainedTokenizerFast):
    """
    Расширенный токенизатор с поддержкой фрактальных метаданных.
    Добавляет специальные токены для работы с фрактальной структурой.
    """
    
    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        special_tokens: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Инициализация расширенного токенизатора.
        
        Args:
            tokenizer_file: Путь к файлу токенизатора
            tokenizer_name: Имя предобученного токенизатора из HuggingFace
            special_tokens: Дополнительные специальные токены
            **kwargs: Дополнительные аргументы для родительского класса
        """
        # Базовые специальные токены для фрактальной структуры
        self.fractal_tokens = {
            'fractal_start': '<|fractal|>',
            'fractal_end': '</fractal>',
            'metadata_start': '<|meta|>',
            'metadata_end': '</meta>',
            'level_sep': '|',
            'fractal_sep': ':',
        }
        
        # Обновляем специальные токены, если переданы
        if special_tokens:
            self.fractal_tokens.update(special_tokens)
        
        # Загружаем базовый токенизатор
        if tokenizer_file and os.path.isfile(tokenizer_file):
            super().__init__(tokenizer_file=tokenizer_file, **kwargs)
        elif tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
            super().__init__(
                tokenizer_file=tokenizer.vocab_files_names['vocab_file'],
                **kwargs
            )
        else:
            raise ValueError(
                "Необходимо указать либо tokenizer_file, либо tokenizer_name"
            )
        
        # Добавляем специальные токены
        self.add_special_tokens({
            'additional_special_tokens': list(self.fractal_tokens.values())
        })
        
        # Кэш для хранения закодированных фрактальных путей
        self._fractal_path_cache = {}
    
    def encode_fractal_path(self, path: List[Union[str, int]]) -> str:
        """
        Кодирует путь во фрактальной структуре в строку.
        
        Args:
            path: Список идентификаторов узлов в пути
            
        Returns:
            Строковое представление пути
        """
        cache_key = tuple(path)
        if cache_key in self._fractal_path_cache:
            return self._fractal_path_cache[cache_key]
            
        path_str = self.fractal_tokens['fractal_start']
        path_str += self.fractal_tokens['level_sep'].join(map(str, path))
        path_str += self.fractal_tokens['fractal_end']
        
        self._fractal_path_cache[cache_key] = path_str
        return path_str
    
    def encode_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Кодирует метаданные в строку.
        
        Args:
            metadata: Словарь метаданных
            
        Returns:
            Строковое представление метаданных
        """
        if not metadata:
            return ""
            
        meta_str = self.fractal_tokens['metadata_start']
        meta_items = []
        
        for key, value in metadata.items()():
            if isinstance(value, (str, int, float, bool)):
                meta_items.append(f"{key}={str(value).lower()}")
            elif isinstance(value, (list, tuple)):
                meta_items.append(f"{key}={','.join(map(str, value))}")
            else:
                logger.warning(f"Неподдерживаемый тип метаданных: {type(value)}")
        
        meta_str += self.fractal_tokens['fractal_sep'].join(meta_items)
        meta_str += self.fractal_tokens['metadata_end']
        
        return meta_str
    
    def prepare_fractal_input(
        self,
        text: str,
        fractal_path: Optional[List[Union[str, int]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Подготавливает входные данные с фрактальными метаданными.
        
        Args:
            text: Исходный текст
            fractal_path: Путь во фрактальной структуре
            metadata: Метаданные
            
        Returns:
            Текст с добавленными фрактальными метаданными
        """
        result = []
        
        # Добавляем путь во фрактальной структуре, если указан
        if fractal_path:
            result.append(self.encode_fractal_path(fractal_path))
        
        # Добавляем метаданные, если они есть
        if metadata:
            result.append(self.encode_metadata(metadata))
        
        # Добавляем исходный текст
        result.append(text)
        
        return " ".join(filter(None, result))
    
    def batch_encode_plus_fractal(
        self,
        texts: List[str],
        fractal_paths: Optional[List[List[Union[str, int]]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Пакетное кодирование текстов с фрактальными метаданными.
        
        Args:
            texts: Список текстов для кодирования
            fractal_paths: Список путей во фрактальной структуре
            metadatas: Список словарей метаданных
            **kwargs: Дополнительные аргументы для batch_encode_plus
            
        Returns:
            Словарь с закодированными входами
        """
        if fractal_paths is None:
            fractal_paths = [None] * len(texts)
        if metadatas is None:
            metadatas = [{}] * len(texts)
            
        # Подготавливаем входные данные
        prepared_texts = [
            self.prepare_fractal_input(text, path, meta)
            for text, path, meta in zip(texts, fractal_paths, metadatas)
        ]
        
        # Кодируем с помощью родительского метода
        return super().batch_encode_plus(
            prepared_texts,
            **kwargs
        )
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Сохраняет токенизатор в указанную директорию.
        
        Args:
            save_directory: Директория для сохранения
            **kwargs: Дополнительные аргументы
        """
        # Создаем директорию, если не существует
        os.makedirs(save_directory, exist_ok=True)
        
        # Сохраняем конфигурацию токенизатора
        config = {
            'fractal_tokens': self.fractal_tokens,
            'model_max_length': self.model_max_length,
            'padding_side': self.padding_side,
            'truncation_side': self.truncation_side,
        }
        
        config_path = os.path.join(save_directory, 'tokenizer_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Сохраняем словарь токенизатора
        super().save_pretrained(save_directory, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        Загружает предобученный токенизатор.
        
        Args:
            pretrained_model_name_or_path: Путь к директории с сохраненным токенизатором
                или имя модели из HuggingFace Hub
            **kwargs: Дополнительные аргументы
            
        Returns:
            Загруженный экземпляр ExtendedFractalTokenizer
        """
        # Сначала загружаем базовый токенизатор
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Загружаем конфигурацию, если она есть
        config_path = os.path.join(pretrained_model_name_or_path, 'tokenizer_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Обновляем атрибуты токенизатора
            if 'fractal_tokens' in config:
                tokenizer.fractal_tokens = config['fractal_tokens']
            
            for attr in ['model_max_length', 'padding_side', 'truncation_side']:
                if attr in config:
                    setattr(tokenizer, attr, config[attr])
        
        return tokenizer
