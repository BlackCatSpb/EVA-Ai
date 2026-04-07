"""
GGUF Fractal Exporter — экспорт структуры GGUF модели в фрактальный граф

Использует llama_cpp для извлечения метаданных (без ручного парсинга бинарного формата),
затем строит фрактальную иерархию L0→L1→L2→L3 и сохраняет в UnifiedFractalMemory.

Архитектура фрактального графа модели:
  L0: Модель целиком (общая информация)
  L1: Компонентные группы (embeddings, layer_stack, output, tokenizer)
  L2: Чанки слоёв (layers_0-4, layers_5-9, ...) + компоненты
  L3: Отдельные слои и тензоры
"""

import os
import json
import logging
from typing import Dict, Any, List
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class GGUFFractalExporter:
    """Экспорт GGUF модели в фрактальный граф"""
    
    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self.metadata = {}
        self.file_size = 0
        self._loaded = False
    
    def load(self):
        """Загружает метаданные модели через llama_cpp (vocab_only=True)"""
        if self._loaded:
            return
        
        logger.info(f"Загрузка метаданных GGUF: {self.gguf_path}")
        model = Llama(
            model_path=self.gguf_path,
            vocab_only=True,
            verbose=False
        )
        
        self.metadata = dict(model.metadata)
        self.file_size = os.path.getsize(self.gguf_path)
        self._loaded = True
        
        logger.info(f"Метаданные загружены: {len(self.metadata)} ключей, файл={self.file_size} байт")
    
    def get_architecture(self) -> str:
        return self.metadata.get('general.architecture', 'unknown')
    
    def get_block_count(self) -> int:
        arch = self.get_architecture()
        return int(self.metadata.get(f'{arch}.block_count', 0))
    
    def get_embedding_length(self) -> int:
        arch = self.get_architecture()
        return int(self.metadata.get(f'{arch}.embedding_length', 0))
    
    def get_context_length(self) -> int:
        arch = self.get_architecture()
        return int(self.metadata.get(f'{arch}.context_length', 0))
    
    def get_feed_forward_length(self) -> int:
        arch = self.get_architecture()
        return int(self.metadata.get(f'{arch}.feed_forward_length', 0))
    
    def get_attention_heads(self) -> int:
        arch = self.get_architecture()
        return int(self.metadata.get(f'{arch}.attention.head_count', 0))
    
    def get_attention_heads_kv(self) -> int:
        arch = self.get_architecture()
        return int(self.metadata.get(f'{arch}.attention.head_count_kv', 0))
    
    def get_total_params(self) -> int:
        """Вычисляет примерное количество параметров"""
        block_count = self.get_block_count()
        embed_dim = self.get_embedding_length()
        ffn_dim = self.get_feed_forward_length()
        heads = self.get_attention_heads()
        
        # Qwen 2.5 approximate params calculation
        # Attention: Q/K/V + O = 4 * embed_dim^2
        # FFN: gate + up + down = 3 * embed_dim * ffn_dim
        # Embedding + output = 2 * vocab_size * embed_dim (approx)
        vocab_size = 151936  # approximate for Qwen
        
        attn_params = 4 * (embed_dim * embed_dim) * block_count
        ffn_params = 3 * (embed_dim * ffn_dim) * block_count
        embedding_params = vocab_size * embed_dim
        norm_params = 2 * embed_dim * block_count
        
        total = attn_params + ffn_params + embedding_params + norm_params
        return total
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Сводка модели"""
        self.load()
        arch = self.get_architecture()
        return {
            'name': self.metadata.get('general.name', ''),
            'architecture': arch,
            'type': self.metadata.get('general.type', ''),
            'finetune': self.metadata.get('general.finetune', ''),
            'version': self.metadata.get('general.version', ''),
            'size_label': self.metadata.get('general.size_label', ''),
            'quantization_version': self.metadata.get('general.quantization_version', 0),
            'file_type': self.metadata.get('general.file_type', 0),
            'block_count': self.get_block_count(),
            'embedding_length': self.get_embedding_length(),
            'context_length': self.get_context_length(),
            'feed_forward_length': self.get_feed_forward_length(),
            'attention_heads': self.get_attention_heads(),
            'attention_heads_kv': self.get_attention_heads_kv(),
            'rope_freq_base': float(self.metadata.get(f'{arch}.rope.freq_base', 0)),
            'rms_norm_eps': float(self.metadata.get(f'{arch}.attention.layer_norm_rms_epsilon', 0)),
            'tokenizer_model': self.metadata.get('tokenizer.ggml.model', ''),
            'tokenizer_pre': self.metadata.get('tokenizer.ggml.pre', ''),
            'bos_token_id': self.metadata.get('tokenizer.ggml.bos_token_id', None),
            'eos_token_id': self.metadata.get('tokenizer.ggml.eos_token_id', None),
            'file_size_bytes': self.file_size,
            'file_size_gb': round(self.file_size / (1024**3), 2),
            'total_params': self.get_total_params()
        }
    
    def build_fractal_hierarchy(self) -> Dict[str, Any]:
        """
        Строит фрактальную иерархию модели:
        L0: Модель целиком
        L1: Компонентные группы
        L2: Чанки слоёв
        L3: Отдельные слои
        """
        self.load()
        summary = self.get_model_summary()
        block_count = self.get_block_count()
        arch = self.get_architecture()
        
        hierarchy = {
            'level': 0,
            'type': 'model',
            'name': summary['name'],
            'address': 'L0::model',
            'content': json.dumps(summary, ensure_ascii=False),
            'children': []
        }
        
        # L1: Компонентные группы
        l1_nodes = []
        
        # 1. Token embeddings
        emb_node = {
            'level': 1,
            'type': 'component_group',
            'name': 'token_embeddings',
            'address': 'L1::token_embeddings',
            'content': f'Token embedding layer: dim={summary["embedding_length"]}, vocab size from tokenizer',
            'children': []
        }
        l1_nodes.append(emb_node)
        
        # 2. Layer stack (основная часть модели)
        chunk_size = max(1, block_count // 8)
        layers_node = {
            'level': 1,
            'type': 'component_group',
            'name': 'layer_stack',
            'address': 'L1::layer_stack',
            'content': f'Transformer layers: {block_count} blocks, each with attention + FFN',
            'children': []
        }
        
        for chunk_start in range(0, block_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, block_count) - 1
            chunk_node = {
                'level': 2,
                'type': 'layer_chunk',
                'name': f'layers_{chunk_start}_to_{chunk_end}',
                'address': f'L2::layers_{chunk_start}_{chunk_end}',
                'content': f'Layer chunk {chunk_start}-{chunk_end}: attention (Q/K/V/O) + FFN (gate/up/down) + norms',
                'children': []
            }
            
            for layer_num in range(chunk_start, chunk_end + 1):
                layer_node = {
                    'level': 3,
                    'type': 'layer',
                    'name': f'{arch}_blk_{layer_num}',
                    'address': f'L3::{arch}_blk_{layer_num}',
                    'content': self._describe_layer(layer_num, arch, summary),
                    'children': self._layer_components(layer_num, arch, summary)
                }
                chunk_node['children'].append(layer_node)
            
            layers_node['children'].append(chunk_node)
        
        l1_nodes.append(layers_node)
        
        # 3. Output layer
        out_node = {
            'level': 1,
            'type': 'component_group',
            'name': 'output',
            'address': 'L1::output',
            'content': f'Output layer: projects embeddings to vocabulary logits',
            'children': []
        }
        l1_nodes.append(out_node)
        
        # 4. Tokenizer
        tok_node = {
            'level': 1,
            'type': 'component_group',
            'name': 'tokenizer',
            'address': 'L1::tokenizer',
            'content': f'Tokenizer: {summary["tokenizer_model"]} (pre={summary["tokenizer_pre"]}), BOS={summary["bos_token_id"]}, EOS={summary["eos_token_id"]}',
            'children': []
        }
        l1_nodes.append(tok_node)
        
        hierarchy['children'] = l1_nodes
        return hierarchy
    
    def _describe_layer(self, layer_num: int, arch: str, summary: Dict) -> str:
        """Описание слоя"""
        return (
            f"Layer {layer_num}: "
            f"self_attn (Q:{summary['embedding_length']}x{summary['attention_heads']}, "
            f"K/V:{summary['embedding_length']}x{summary['attention_heads_kv']}, "
            f"O:{summary['embedding_length']}x{summary['attention_heads']}) + "
            f"ffn (gate/up:{summary['feed_forward_length']}x{summary['embedding_length']}, "
            f"down:{summary['embedding_length']}x{summary['feed_forward_length']}) + "
            f"attn_norm + ffn_norm (rms_eps={summary.get('rope_freq_base', 1e-6)})"
        )
    
    def _layer_components(self, layer_num: int, arch: str, summary: Dict) -> List[Dict]:
        """Компоненты слоя (L3 -> L4)"""
        prefix = f'{arch}.blk.{layer_num}'
        components = [
            {'name': 'attn_q', 'desc': f'Query projection: {summary["embedding_length"]} -> {summary["embedding_length"]}'},
            {'name': 'attn_k', 'desc': f'Key projection: {summary["embedding_length"]} -> {summary["embedding_length"] // (summary["attention_heads"] // summary["attention_heads_kv"])}'},
            {'name': 'attn_v', 'desc': f'Value projection: {summary["embedding_length"]} -> {summary["embedding_length"] // (summary["attention_heads"] // summary["attention_heads_kv"])}'},
            {'name': 'attn_output', 'desc': f'Output projection: {summary["embedding_length"]} -> {summary["embedding_length"]}'},
            {'name': 'ffn_gate', 'desc': f'FFN gate (SiLU): {summary["embedding_length"]} -> {summary["feed_forward_length"]}'},
            {'name': 'ffn_up', 'desc': f'FFN up: {summary["embedding_length"]} -> {summary["feed_forward_length"]}'},
            {'name': 'ffn_down', 'desc': f'FFN down: {summary["feed_forward_length"]} -> {summary["embedding_length"]}'},
            {'name': 'attn_norm', 'desc': f'Attention RMS norm (eps={summary.get("rope_freq_base", 1e-6)})'},
            {'name': 'ffn_norm', 'desc': f'FFN RMS norm (eps={summary.get("rope_freq_base", 1e-6)})'},
        ]
        
        return [
            {
                'level': 4,
                'type': 'tensor_group',
                'name': f'{prefix}.{c["name"]}',
                'address': f'L4::{prefix}.{c["name"]}',
                'content': c['desc']
            }
            for c in components
        ]
    
    def export_to_fractal_memory(self, fractal_memory, model_type: str = 'model_a') -> Dict[str, Any]:
        """Экспортирует модель в UnifiedFractalMemory с кастомными ID"""
        self.load()
        summary = self.get_model_summary()
        arch = summary['architecture']
        block_count = summary['block_count']
        
        nodes_created = 0
        edges_created = 0
        
        # L0: Root
        root_content = json.dumps({
            'model_type': model_type,
            'gguf_path': self.gguf_path,
            **summary
        }, ensure_ascii=False)
        
        root_id = f'model::{model_type}::L0'
        fractal_memory.add_model_node(
            node_id=root_id,
            content=root_content,
            level=0,
            context={'model_type': model_type, 'node_type': 'model_root'},
            node_type='model_root'
        )
        nodes_created += 1
        
        # L1: token_embeddings
        emb_id = f'model::{model_type}::L1::token_embeddings'
        fractal_memory.add_model_node(
            node_id=emb_id,
            content=f'Token embedding: dim={summary["embedding_length"]}',
            level=1,
            parent_id=root_id,
            context={'model_type': model_type, 'node_type': 'component'},
            node_type='component'
        )
        nodes_created += 1
        
        # L1: layer_stack
        stack_id = f'model::{model_type}::L1::layer_stack'
        fractal_memory.add_model_node(
            node_id=stack_id,
            content=f'Transformer layers: {block_count} blocks',
            level=1,
            parent_id=root_id,
            context={'model_type': model_type, 'node_type': 'component'},
            node_type='component'
        )
        nodes_created += 1
        
        # L2: chunks -> L3: layers -> L4: tensors
        chunk_size = max(1, block_count // 8)
        for chunk_start in range(0, block_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, block_count) - 1
            chunk_id = f'model::{model_type}::L2::layers_{chunk_start}_{chunk_end}'
            fractal_memory.add_model_node(
                node_id=chunk_id,
                content=f'Layers {chunk_start}-{chunk_end}',
                level=2,
                parent_id=stack_id,
                context={'model_type': model_type, 'node_type': 'chunk'},
                node_type='chunk'
            )
            nodes_created += 1
            
            for layer_num in range(chunk_start, chunk_end + 1):
                layer_id = f'model::{model_type}::L3::{arch}_blk_{layer_num}'
                fractal_memory.add_model_node(
                    node_id=layer_id,
                    content=f'Layer {layer_num}: self_attn + ffn, dim={summary["embedding_length"]}',
                    level=3,
                    parent_id=chunk_id,
                    context={'model_type': model_type, 'node_type': 'layer'},
                    node_type='layer'
                )
                nodes_created += 1
                
                for t in ['attn_q', 'attn_k', 'attn_v', 'attn_o', 'ffn_gate', 'ffn_up', 'ffn_down', 'attn_norm', 'ffn_norm']:
                    tensor_id = f'model::{model_type}::L4::{arch}.blk.{layer_num}.{t}'
                    fractal_memory.add_model_node(
                        node_id=tensor_id,
                        content=f'{t}: dim={summary["embedding_length"]}',
                        level=4,
                        parent_id=layer_id,
                        context={'model_type': model_type, 'node_type': 'tensor'},
                        node_type='tensor'
                    )
                    nodes_created += 1
        
        # L1: output
        out_id = f'model::{model_type}::L1::output'
        fractal_memory.add_model_node(
            node_id=out_id,
            content='Output projection',
            level=1,
            parent_id=root_id,
            context={'model_type': model_type, 'node_type': 'component'},
            node_type='component'
        )
        nodes_created += 1
        
        # L1: tokenizer
        tok_id = f'model::{model_type}::L1::tokenizer'
        fractal_memory.add_model_node(
            node_id=tok_id,
            content=f'Tokenizer: {summary["tokenizer_model"]}',
            level=1,
            parent_id=root_id,
            context={'model_type': model_type, 'node_type': 'component'},
            node_type='component'
        )
        nodes_created += 1
        
        fractal_memory.flush()
        logger.info(f"Экспортировано {model_type}: {nodes_created} узлов")
        return {'model_type': model_type, 'nodes_created': nodes_created, 'blocks': block_count}
