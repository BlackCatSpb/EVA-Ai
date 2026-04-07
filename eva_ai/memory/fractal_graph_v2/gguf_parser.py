"""
GGUF Model Parser - Парсинг GGUF моделей для извлечения знаний в граф памяти

Извлекает из GGUF:
1. Токенизатор (vocab) - слова/подслова -> token IDs
2. Конфигурацию модели (архитектура, слои, размерности)
3. Архитектурные паттерны (типы слоёв, attention heads, FFN)
4. Системный промпт / chat template
"""

import os
import json
import logging
import struct
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.fractal_graph_v2.gguf_parser")


@dataclass
class GGUFModelInfo:
    """Информация о GGUF модели."""
    model_path: str
    model_type: str = ""                    # qwen2, llama, etc.
    architecture: str = ""                   # Архитектура модели
    vocab_size: int = 0                      # Размер словаря
    hidden_size: int = 0                     # Размер скрытого слоя
    num_layers: int = 0                     # Количество слоёв
    num_attention_heads: int = 0            # Количество attention голов
    num_key_value_heads: int = 0            # KV heads (GQA)
    max_position_embeddings: int = 0        # Максимальная длина контекста
    intermediate_size: int = 0              # Размер FFN
    rope_theta: float = 0                    # RoPE base frequency
    rms_norm_eps: float = 0                 # RMS Norm epsilon
    
    # Токенизатор
    tokenizer_type: str = ""
    chat_template: str = ""
    vocab: Dict[str, int] = field(default_factory=dict)
    
    # Метаданные
    quantization_version: str = ""
    file_size: int = 0


class GGUFModelParser:
    """
    Парсер GGUF моделей.
    
    Позволяет извлечь структуру и знания модели для сохранения
    в фрактальном графе памяти.
    """
    
    # GGUF magic number
    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    
    # Типы данных GGUF
    GGUF_DATA_TYPES = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        4: "Q5_0",
        5: "Q5_1",
        6: "Q8_0",
        7: "Q8_1",
        8: "Q2_K",
        9: "Q3_K",
        10: "Q4_K",
        11: "Q5_K",
        12: "Q6_K",
        13: "Q8_K",
        14: "I8",
        15: "I16",
        16: "I32",
        17: "I64",
        18: "F64",
        19: "BF16",
    }
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    
    def parse(self) -> GGUFModelInfo:
        """Парсить GGUF файл и извлечь информацию."""
        logger.info(f"Начинаем парсинг GGUF: {self.model_path}")
        
        info = GGUFModelInfo(model_path=self.model_path)
        
        try:
            with open(self.model_path, 'rb') as f:
                # Читаем magic number
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != self.GGUF_MAGIC:
                    logger.warning(f"Неверный GGUF magic: {hex(magic)}")
                    return info
                
                # Читаем версию
                version = struct.unpack('<I', f.read(4))[0]
                info.quantization_version = str(version)
                logger.info(f"GGUF version: {version}")
                
                # Читаем количество tensor'ов
                n_tensors = struct.unpack('<Q', f.read(8))[0]
                logger.info(f"Количество tensors: {n_tensors}")
                
                # Читаем метаданные (key-value pairs)
                metadata = self._parse_metadata(f)
                
                # Извлекаем конфигурацию
                info.model_type = metadata.get('general.architecture', 'unknown')
                info.architecture = metadata.get('general.architecture', 'unknown')
                info.vocab_size = metadata.get(f'{info.model_type}.vocab_size', 0)
                info.hidden_size = metadata.get(f'{info.model_type}.hidden_size', 0)
                info.num_layers = metadata.get(f'{info.model_type}.num_hidden_layers', 0)
                info.num_attention_heads = metadata.get(f'{info.model_type}.num_attention_heads', 0)
                info.num_key_value_heads = metadata.get(f'{info.model_type}.num_key_value_heads', 0) or info.num_attention_heads
                info.max_position_embeddings = metadata.get(f'{info.model_type}.max_position_embeddings', 0)
                info.intermediate_size = metadata.get(f'{info.model_type}.intermediate_size', 0)
                info.rope_theta = metadata.get(f'{info.model_type}.rope_theta', 10000.0)
                info.rms_norm_eps = metadata.get(f'{info.model_type}.rms_norm_eps', 1e-6)
                
                # Токенизатор
                info.tokenizer_type = metadata.get('tokenizer.type', 'unknown')
                info.chat_template = metadata.get('tokenizer.chat_template', '')
                
                logger.info(f"Модель: {info.architecture}, слоёв: {info.num_layers}, vocab: {info.vocab_size}")
                
                # Парсим vocab (может быть большой)
                info.vocab = self._parse_vocab(f, metadata, info.vocab_size)
                
        except Exception as e:
            logger.error(f"Ошибка парсинга GGUF: {e}")
        
        return info
    
    def _parse_metadata(self, f) -> Dict[str, Any]:
        """Парсинг метаданных (key-value pairs)."""
        metadata = {}
        
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        for _ in range(n_kv):
            # Читаем ключ
            key_len = struct.unpack('<I', f.read(4))[0]
            key = f.read(key_len).decode('utf-8')
            
            # Читаем тип значения
            value_type = struct.unpack('<I', f.read(4))[0]
            
            # Читаем значение в зависимости от типа
            if value_type == 0:  # UINT8
                value = struct.unpack('<I', f.read(4))[0]
            elif value_type == 1:  # INT8
                value = struct.unpack('<i', f.read(4))[0]
            elif value_type == 2:  # FLOAT32
                value = struct.unpack('<f', f.read(4))[0]
            elif value_type == 3:  # BOOL
                value = bool(struct.unpack('<I', f.read(4))[0])
            elif value_type == 4:  # STRING
                str_len = struct.unpack('<I', f.read(4))[0]
                value = f.read(str_len).decode('utf-8')
            elif value_type == 5:  # ARRAY
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                value = f"Array[{arr_type}, {arr_len}]"
            elif value_type == 6:  # UINT64
                value = struct.unpack('<Q', f.read(8))[0]
            elif value_type == 7:  # INT64
                value = struct.unpack('<q', f.read(8))[0]
            elif value_type == 8:  # FLOAT64
                value = struct.unpack('<d', f.read(8))[0]
            else:
                value = f"unknown_type_{value_type}"
            
            metadata[key] = value
        
        return metadata
    
    def _parse_vocab(self, f, metadata: Dict, vocab_size: int) -> Dict[str, int]:
        """Парсинг словаря токенизатора."""
        vocab = {}
        
        # Пытаемся найти vocab в metadata
        tokenizer_model = metadata.get('tokenizer.model', '')
        
        if tokenizer_model == 'BPE':
            # BPE токенизатор - обычно в отдельном файле или в GGUF
            # Пробуем извлечь из metadata
            pass
        
        # Пробуем прочитать из model file
        try:
            # Переходим к концу файла или ищем special tensors
            f.seek(0, 2)
            file_size = f.tell()
            
            # Для больших vocab - читаем только начало
            max_vocab_to_read = min(vocab_size, 10000)
            
            logger.info(f"Попытка извлечь {max_vocab_to_read} токенов из словаря")
            
            # Пока просто возвращаем размер - реальное извлечение vocab
            # требует знания формата конкретного токенизатора
            vocab = {f"token_{i}": i for i in range(vocab_size)}
            
        except Exception as e:
            logger.warning(f"Не удалось извлечь vocab: {e}")
        
        return vocab
    
    def extract_knowledge_for_graph(self) -> List[Dict[str, Any]]:
        """
        Извлечь знания из модели для сохранения в графе.
        
        Возвращает список узлов для добавления в граф.
        """
        info = self.parse()
        
        nodes = []
        
        # 1. Узел модели (статичный)
        nodes.append({
            "type": "MODEL_A" if "3b" in self.model_path.lower() else "MODEL_UNKNOWN",
            "level": -1,  # Системный уровень
            "content": f"GGUF модель: {info.architecture}",
            "metadata": {
                "model_path": self.model_path,
                "architecture": info.architecture,
                "vocab_size": info.vocab_size,
                "hidden_size": info.hidden_size,
                "num_layers": info.num_layers,
                "file_size": self.file_size
            }
        })
        
        # 2. Архитектурные компоненты как узлы
        if info.num_layers > 0:
            nodes.append({
                "type": "CONCEPT",
                "level": 1,
                "content": f"Трансформер с {info.num_layers} слоями",
                "metadata": {
                    "architecture": info.architecture,
                    "num_layers": info.num_layers,
                    "hidden_size": info.hidden_size,
                    "attention_heads": info.num_attention_heads
                }
            })
        
        if info.vocab_size > 0:
            nodes.append({
                "type": "CONCEPT",
                "level": 1,
                "content": f"Словарь на {info.vocab_size} токенов",
                "metadata": {
                    "vocab_size": info.vocab_size,
                    "tokenizer_type": info.tokenizer_type
                }
            })
        
        if info.max_position_embeddings > 0:
            nodes.append({
                "type": "CONCEPT",
                "level": 1,
                "content": f"Максимальная длина контекста: {info.max_position_embeddings}",
                "metadata": {
                    "max_context": info.max_position_embeddings,
                    "rope_theta": info.rope_theta
                }
            })
        
        # 3. Паттерны внимания как знания
        if info.num_attention_heads > 0:
            nodes.append({
                "type": "FACT",
                "level": 2,
                "content": f"Multi-head attention с {info.num_attention_heads} головами",
                "metadata": {
                    "attention_heads": info.num_attention_heads,
                    "kv_heads": info.num_key_value_heads
                }
            })
        
        # 4. FFN как знание
        if info.intermediate_size > 0:
            nodes.append({
                "type": "FACT",
                "level": 2,
                "content": f"Feed-forward network размером {info.intermediate_size}",
                "metadata": {
                    "intermediate_size": info.intermediate_size
                }
            })
        
        # 5. ROPE как знание
        if info.rope_theta > 0:
            nodes.append({
                "type": "FACT",
                "level": 2,
                "content": f"RoPE позиционное кодирование с base={info.rope_theta}",
                "metadata": {
                    "rope_theta": info.rope_theta
                }
            })
        
        return nodes


def parse_gguf_model(model_path: str) -> GGUFModelInfo:
    """Фабричная функция для парсинга GGUF."""
    parser = GGUFModelParser(model_path)
    return parser.parse()


def extract_to_graph(model_path: str, graph) -> Dict[str, Any]:
    """
    Извлечь знания из GGUF модели и добавить в граф.
    
    Returns:
        Результат добавления узлов
    """
    parser = GGUFModelParser(model_path)
    knowledge_nodes = parser.extract_knowledge_for_graph()
    
    added_count = 0
    for node_data in knowledge_nodes:
        try:
            graph.add_node(
                content=node_data["content"],
                node_type=node_data["type"],
                level=node_data["level"],
                metadata=node_data.get("metadata", {})
            )
            added_count += 1
        except Exception as e:
            logger.warning(f"Не удалось добавить узел: {e}")
    
    return {
        "model_path": model_path,
        "nodes_extracted": len(knowledge_nodes),
        "nodes_added": added_count
    }