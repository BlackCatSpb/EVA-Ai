"""
GGUF Parser — парсинг бинарного формата GGUF

Извлекает:
- Метаданные модели (архитектура, гиперпараметры, токенизатор)
- Структуру тензоров (имя, форма, тип данных, квантование, смещение)
- Иерархию слоёв для фрактального экспорта

GGUF Format:
  Header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
  KV pairs: key(str) + value(type + data)
  Tensor info: name(str) + n_dims(4) + shape[n_dims](8*n) + dtype(4) + offset(8)
  Alignment padding to 32 bytes
  Tensor data at offsets
"""

import struct
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# GGUF constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# Data types
GGUF_TYPES = {
    0: 'uint8', 1: 'int8', 2: 'uint16', 3: 'int16',
    4: 'uint32', 5: 'int32', 6: 'float32', 7: 'bool',
    8: 'string', 9: 'array', 10: 'uint64', 11: 'int64',
    12: 'float64'
}

# Tensor dtypes
GGUF_TENSOR_TYPES = {
    0: 'float32', 1: 'float16', 2: 'q4_0', 3: 'q4_1',
    4: 'q5_0', 5: 'q5_1', 6: 'q8_0', 7: 'q8_1',
    8: 'q2_k', 9: 'q3_k', 10: 'q4_k', 11: 'q5_k',
    12: 'q6_k', 13: 'q8_k', 14: 'iq2_xxs', 15: 'iq2_xs',
    16: 'iq3_xxs', 17: 'iq1_s', 18: 'iq4_nl', 19: 'iq3_s',
    20: 'iq2_s', 21: 'iq4_xs', 22: 'i8', 23: 'i16',
    24: 'i32', 25: 'i64', 26: 'f64', 27: 'iq1_m',
    28: 'bf16'
}

# Bytes per element for each tensor type
TENSOR_TYPE_BYTES = {
    'float32': 4, 'float16': 2, 'bf16': 2,
    'q4_0': 0.5, 'q4_1': 0.625,
    'q5_0': 0.625, 'q5_1': 0.75,
    'q8_0': 1.0625, 'q8_1': 1.125,
    'q2_k': 0.3125, 'q3_k': 0.375,
    'q4_k': 0.53125, 'q5_k': 0.625,
    'q6_k': 0.75, 'q8_k': 1.0,
    'iq2_xxs': 0.25, 'iq2_xs': 0.28125,
    'iq3_xxs': 0.375, 'iq1_s': 0.1875,
    'iq4_nl': 0.5, 'iq3_s': 0.40625,
    'iq2_s': 0.28125, 'iq4_xs': 0.5,
    'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
    'f64': 8, 'iq1_m': 0.21875,
}


@dataclass
class TensorInfo:
    """Информация о тензоре"""
    name: str
    shape: List[int]
    dtype: str
    dtype_id: int
    offset: int
    n_elements: int = 0
    size_bytes: int = 0
    layer: int = -1
    component: str = ""
    
    def __post_init__(self):
        self.n_elements = 1
        for s in self.shape:
            self.n_elements *= s
        bpe = TENSOR_TYPE_BYTES.get(self.dtype, 1)
        self.size_bytes = int(self.n_elements * bpe)
        
        # Parse layer number from name
        parts = self.name.split('.')
        for p in parts:
            if p.isdigit():
                self.layer = int(p)
                break
        
        # Parse component type
        if 'attn' in self.name or 'self_attn' in self.name:
            if 'q_proj' in self.name: self.component = 'q_proj'
            elif 'k_proj' in self.name: self.component = 'k_proj'
            elif 'v_proj' in self.name: self.component = 'v_proj'
            elif 'o_proj' in self.name: self.component = 'o_proj'
            elif 'attn_qkv' in self.name: self.component = 'qkv'
            elif 'attn_out' in self.name: self.component = 'attn_out'
            else: self.component = 'attention'
        elif 'ffn' in self.name or 'mlp' in self.name:
            if 'gate' in self.name or 'down' in self.name: self.component = 'ffn_down'
            elif 'up' in self.name: self.component = 'ffn_up'
            else: self.component = 'ffn'
        elif 'token_embd' in self.name: self.component = 'token_embd'
        elif 'output' in self.name: self.component = 'output'
        elif 'attn_norm' in self.name or 'layer_norm' in self.name: self.component = 'norm_attn'
        elif 'ffn_norm' in self.name: self.component = 'norm_ffn'
        else: self.component = 'other'


@dataclass
class GGUFMetadata:
    """Метаданные GGUF модели"""
    general_architecture: str = ""
    general_name: str = ""
    general_file_type: str = ""
    general_total_params: int = 0
    general_quantization_version: int = 0
    
    # Architecture-specific
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    feed_forward_length: int = 0
    attention_head_count: int = 0
    attention_head_count_kv: int = 0
    rms_norm_eps: float = 0.0
    rope_freq_base: float = 0.0
    
    # Tokenizer
    tokenizer_model: str = ""
    tokenizer_ggml_tokens: List[str] = field(default_factory=list)
    
    # Raw
    raw_kv: Dict[str, Any] = field(default_factory=dict)


class GGUFParser:
    """Парсер GGUF файлов"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata = GGUFMetadata()
        self.tensors: List[TensorInfo] = []
        self.tensor_map: Dict[str, TensorInfo] = {}
        self.file_size = 0
        self.data_offset = 0
    
    def parse(self) -> Dict[str, Any]:
        """Полный парсинг GGUF файла"""
        logger.info(f"Парсинг GGUF: {self.file_path}")
        
        with open(self.file_path, 'rb') as f:
            # Header
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {magic}")
            
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]
            
            logger.info(f"GGUF v{version}: {n_tensors} тензоров, {n_kv} KV пар")
            
            # KV pairs
            for i in range(n_kv):
                key = self._read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata.raw_kv[key] = value
                self._extract_metadata(key, value)
            
            # Alignment
            alignment = 32
            pos = f.tell()
            padding = (alignment - (pos % alignment)) % alignment
            f.seek(pos + padding)
            self.data_offset = f.tell()
            
            # Tensor info
            for i in range(n_tensors):
                tensor = self._read_tensor_info(f)
                self.tensors.append(tensor)
                self.tensor_map[tensor.name] = tensor
            
            # File size
            f.seek(0, 2)
            self.file_size = f.tell()
        
        logger.info(f"Распарсено: {len(self.tensors)} тензоров, {len(self.metadata.raw_kv)} KV пар")
        logger.info(f"Архитектура: {self.metadata.general_architecture}, "
                    f"слоёв: {self.metadata.block_count}, "
                    f"params: {self.metadata.general_total_params:,}")
        
        return self._build_summary()
    
    def _read_string(self, f) -> str:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            raise ValueError(f"Unexpected end of file while reading string length at pos {f.tell()}")
        length = struct.unpack('<Q', length_bytes)[0]
        # Sanity check: string length shouldn't be absurdly large
        if length > 10_000_000:
            raise ValueError(f"String length {length} seems too large at pos {f.tell() - 8}")
        return f.read(length).decode('utf-8', errors='replace')
    
    def _read_value(self, f, value_type: int) -> Any:
        if value_type == 0: return struct.unpack('<B', f.read(1))[0]
        elif value_type == 1: return struct.unpack('<b', f.read(1))[0]
        elif value_type == 2: return struct.unpack('<H', f.read(2))[0]
        elif value_type == 3: return struct.unpack('<h', f.read(2))[0]
        elif value_type == 4: return struct.unpack('<I', f.read(4))[0]
        elif value_type == 5: return struct.unpack('<i', f.read(4))[0]
        elif value_type == 6: return struct.unpack('<f', f.read(4))[0]
        elif value_type == 7: return struct.unpack('<B', f.read(1))[0] != 0
        elif value_type == 8: return self._read_string(f)
        elif value_type == 10: return struct.unpack('<Q', f.read(8))[0]
        elif value_type == 11: return struct.unpack('<q', f.read(8))[0]
        elif value_type == 12: return struct.unpack('<d', f.read(8))[0]
        elif value_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len_bytes = f.read(8)
            if len(arr_len_bytes) < 8:
                return None
            arr_len = struct.unpack('<Q', arr_len_bytes)[0]
            # Skip huge arrays (tokenizer tokens can be 151k+ strings)
            if arr_len > 100000:
                # For string arrays, we need to skip each string individually
                if arr_type == 8:  # string
                    for _ in range(arr_len):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.seek(slen, 1)
                elif arr_type in (0, 1):  # uint8/int8
                    f.seek(arr_len, 1)
                elif arr_type in (2, 3):  # uint16/int16
                    f.seek(arr_len * 2, 1)
                elif arr_type in (4, 5, 6):  # uint32/int32/float32
                    f.seek(arr_len * 4, 1)
                elif arr_type in (10, 11, 12):  # uint64/int64/float64
                    f.seek(arr_len * 8, 1)
                return f"[array of {arr_len} items, skipped]"
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        else:
            return None
    
    def _read_tensor_info(self, f) -> TensorInfo:
        name = self._read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack('<Q', f.read(8))[0])
        dtype_id = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        
        dtype = GGUF_TENSOR_TYPES.get(dtype_id, f'unknown_{dtype_id}')
        
        return TensorInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            dtype_id=dtype_id,
            offset=offset
        )
    
    def _extract_metadata(self, key: str, value: Any):
        """Извлекает ключевые метаданные"""
        if key == 'general.architecture':
            self.metadata.general_architecture = value
        elif key == 'general.name':
            self.metadata.general_name = value
        elif key == 'general.file_type':
            self.metadata.general_file_type = value
        elif key == 'general.parameter_count':
            self.metadata.general_total_params = value
        elif key == 'general.quantization_version':
            self.metadata.general_quantization_version = value
        elif key == f'{self.metadata.general_architecture or "llama"}.context_length':
            self.metadata.context_length = value
        elif key.endswith('.embedding_length'):
            self.metadata.embedding_length = value
        elif key.endswith('.block_count'):
            self.metadata.block_count = value
        elif key.endswith('.feed_forward_length'):
            self.metadata.feed_forward_length = value
        elif key.endswith('.attention.head_count'):
            self.metadata.attention_head_count = value
        elif key.endswith('.attention.head_count_kv'):
            self.metadata.attention_head_count_kv = value
        elif key.endswith('.attention.layer_norm_rms_epsilon'):
            self.metadata.rms_norm_eps = value
        elif key.endswith('.rope.freq_base'):
            self.metadata.rope_freq_base = value
        elif key == 'tokenizer.ggml.model':
            self.metadata.tokenizer_model = value
        elif key == 'tokenizer.ggml.tokens':
            self.metadata.tokenizer_ggml_tokens = value
    
    def _build_summary(self) -> Dict[str, Any]:
        """Строит сводку структуры модели"""
        # Группировка по слоям
        layers = {}
        components = {}
        
        for t in self.tensors:
            layer_key = f"layer_{t.layer}" if t.layer >= 0 else "shared"
            if layer_key not in layers:
                layers[layer_key] = []
            layers[layer_key].append(t.name)
            
            if t.component not in components:
                components[t.component] = 0
            components[t.component] += 1
        
        total_params = sum(t.n_elements for t in self.tensors)
        total_bytes = sum(t.size_bytes for t in self.tensors)
        
        return {
            'file_path': self.file_path,
            'file_size': self.file_size,
            'architecture': self.metadata.general_architecture,
            'name': self.metadata.general_name,
            'total_params': total_params,
            'total_tensor_bytes': total_bytes,
            'n_tensors': len(self.tensors),
            'n_layers': self.metadata.block_count,
            'embedding_dim': self.metadata.embedding_length,
            'context_length': self.metadata.context_length,
            'quantization': self._get_quantization_summary(),
            'layers': {k: len(v) for k, v in layers.items()},
            'components': components,
            'metadata': {
                'context_length': self.metadata.context_length,
                'embedding_length': self.metadata.embedding_length,
                'block_count': self.metadata.block_count,
                'feed_forward_length': self.metadata.feed_forward_length,
                'attention_heads': self.metadata.attention_head_count,
                'tokenizer': self.metadata.tokenizer_model
            }
        }
    
    def _get_quantization_summary(self) -> Dict[str, int]:
        """Сводка по типам квантования"""
        qt = {}
        for t in self.tensors:
            qt[t.dtype] = qt.get(t.dtype, 0) + 1
        return qt
    
    def get_layer_tensors(self, layer_num: int) -> List[TensorInfo]:
        """Все тензоры конкретного слоя"""
        return [t for t in self.tensors if t.layer == layer_num]
    
    def get_component_tensors(self, component: str) -> List[TensorInfo]:
        """Все тензоры компонента"""
        return [t for t in self.tensors if t.component == component]
    
    def get_fractal_hierarchy(self) -> Dict[str, Any]:
        """
        Строит фрактальную иерархию для экспорта в граф:
        L0: Модель целиком
        L1: Компонентные группы (embeddings, layers, output)
        L2: Отдельные слои (layer_0, layer_1, ...)
        L3: Тензоры внутри слоя (q_proj, k_proj, v_proj, ...)
        """
        hierarchy = {
            'level': 0,
            'type': 'model',
            'name': self.metadata.general_name or self.metadata.general_architecture,
            'summary': self._build_summary(),
            'children': []
        }
        
        # L1: Компонентные группы
        layer_tensors = {}  # layer_num -> [tensors]
        shared_tensors = []
        
        for t in self.tensors:
            if t.layer >= 0:
                if t.layer not in layer_tensors:
                    layer_tensors[t.layer] = []
                layer_tensors[t.layer].append(t)
            else:
                shared_tensors.append(t)
        
        # Shared components (L1)
        shared_node = {
            'level': 1,
            'type': 'shared_components',
            'name': 'shared',
            'children': []
        }
        
        component_groups = {}
        for t in shared_tensors:
            if t.component not in component_groups:
                component_groups[t.component] = []
            component_groups[t.component].append(t)
        
        for comp, tensors in component_groups.items():
            shared_node['children'].append({
                'level': 2,
                'type': 'component',
                'name': comp,
                'tensor_count': len(tensors),
                'total_params': sum(t.n_elements for t in tensors),
                'children': [
                    {
                        'level': 3,
                        'type': 'tensor',
                        'name': t.name,
                        'shape': t.shape,
                        'dtype': t.dtype,
                        'n_elements': t.n_elements,
                        'size_bytes': t.size_bytes,
                        'offset': t.offset
                    }
                    for t in tensors
                ]
            })
        
        if shared_node['children']:
            hierarchy['children'].append(shared_node)
        
        # Layer groups (L1)
        layers_node = {
            'level': 1,
            'type': 'layer_stack',
            'name': f'layers_0_to_{self.metadata.block_count - 1}',
            'block_count': self.metadata.block_count,
            'children': []
        }
        
        # Group layers into chunks for L2
        chunk_size = max(1, self.metadata.block_count // 8)  # ~8 chunks
        chunks = {}
        for layer_num in sorted(layer_tensors.keys()):
            chunk_id = layer_num // chunk_size
            if chunk_id not in chunks:
                chunks[chunk_id] = {}
            chunks[chunk_id][layer_num] = layer_tensors[layer_num]
        
        for chunk_id in sorted(chunks.keys()):
            chunk_layers = chunks[chunk_id]
            layer_range = sorted(chunk_layers.keys())
            chunk_node = {
                'level': 2,
                'type': 'layer_chunk',
                'name': f'layers_{layer_range[0]}-{layer_range[-1]}',
                'layer_range': [layer_range[0], layer_range[-1]],
                'children': []
            }
            
            for ln in layer_range:
                layer_node = {
                    'level': 3,
                    'type': 'layer',
                    'name': f'layer_{ln}',
                    'layer_num': ln,
                    'tensor_count': len(chunk_layers[ln]),
                    'total_params': sum(t.n_elements for t in chunk_layers[ln]),
                    'components': list(set(t.component for t in chunk_layers[ln]))
                }
                chunk_node['children'].append(layer_node)
            
            layers_node['children'].append(chunk_node)
        
        if layers_node['children']:
            hierarchy['children'].append(layers_node)
        
        return hierarchy
