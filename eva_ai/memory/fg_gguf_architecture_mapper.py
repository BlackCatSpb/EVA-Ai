"""
GGUF Model Architecture → FractalGraph Mapping
================================================

Структура модели Qwen2.5-3B в FG:

GGUF LAYERS (28-32) → FG LEVELS:
================================
Layer 0-1 (embeddings):     Level 0 (system/static) - минимальная память
Layer 2-8 (early):          Level 1 (short-term hot) - часто используемые
Layer 9-20 (middle):        Level 2 (context window) - основной рабочий слой
Layer 21-31 (late):        Level 3 (reasoning) - самые абстрактные

GGUF COMPONENTS → FG NODES:
===========================
1. VOCAB (~150K tokens) → Semantic Groups, не все узлы в RAM
2. EMBEDDINGS → FG embedding vectors, lazy load
3. ATTENTION HEADS → Node relations (sparse, not dense matrix)
4. FFN WEIGHTS → Metadata в узлах, not full tensors
5. LAYER NORM → Confidence decay parameters

HOT WINDOW STRATEGY:
===================
- Level 1-2: активные в RAM (max 500 узлов)
- Level 0: системные (metadata, config)
- Level 3: on-demand загрузка

ACI (ConceptMiner) INTEGRATION:
================================
- Динамическое создание концептов из контекста
- semantic_search находит релевантные узлы
- add_knowledge связывает новые концепты
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.fg.gguf_architecture_mapper")

@dataclass
class LayerMapping:
    """Маппинг слоя GGUF на FG уровень."""
    gguf_layers: str          # "0-8" или "attention_early"
    fg_level: int            # 0-3
    hot_window: bool         # в RAM
    max_nodes: int           # max узлов в окне
    node_type: str           # тип узлов для этого слоя
    
# Маппинг архитектуры GGUF -> FG
LAYER_MAPPING = {
    # System level - config, vocab metadata
    "config": LayerMapping(
        gguf_layers="config",
        fg_level=0,
        hot_window=True,
        max_nodes=50,
        node_type="model_config"
    ),
    "vocab": LayerMapping(
        gguf_layers="vocab",
        fg_level=0,
        hot_window=False,  # Только metadata, не полный vocab
        max_nodes=1000,
        node_type="vocab_entry"
    ),
    
    # Embedding layer - начало
    "embedding": LayerMapping(
        gguf_layers="0-1",
        fg_level=0,
        hot_window=True,
        max_nodes=100,
        node_type="embedding"
    ),
    
    # Early layers - часто активируются
    "early_attention": LayerMapping(
        gguf_layers="2-8",
        fg_level=1,
        hot_window=True,
        max_nodes=200,
        node_type="attention_early"
    ),
    "early_ffn": LayerMapping(
        gguf_layers="2-8",
        fg_level=1,
        hot_window=True,
        max_nodes=200,
        node_type="ffn_early"
    ),
    
    # Middle layers - основной контекст
    "middle_attention": LayerMapping(
        gguf_layers="9-20",
        fg_level=2,
        hot_window=True,
        max_nodes=300,
        node_type="attention_middle"
    ),
    "middle_ffn": LayerMapping(
        gguf_layers="9-20",
        fg_level=2,
        hot_window=True,
        max_nodes=300,
        node_type="ffn_middle"
    ),
    
    # Late layers - абстрактное reasoning
    "late_attention": LayerMapping(
        gguf_layers="21-28",
        fg_level=3,
        hot_window=False,
        max_nodes=100,
        node_type="attention_late"
    ),
    "late_ffn": LayerMapping(
        gguf_layers="21-28",
        fg_level=3,
        hot_window=False,
        max_nodes=100,
        node_type="ffn_late"
    ),
}


class GGUFToFGArchitectureMapper:
    """
    Маппер архитектуры GGUF модели на FG структуру.
    
    Особенности:
    - Минимизирует RAM (только hot window в памяти)
    - Использует FG для холодного хранения весов metadata
    - ACI создаёт концепты динамически при генерации
    """
    
    def __init__(self, fractal_graph=None, config: Dict = None):
        self.fg = fractal_graph
        self.config = config or {}
        
        # Hot window настройки
        self.hot_window_max = self.config.get('hot_window_max', 500)
        
        # Текущее состояние hot window
        self._hot_window: Dict[str, Any] = {}
        
    def load_model_architecture(self, model_path: str) -> bool:
        """
        Загрузить архитектуру модели в FG.
        
        Создаёт:
        - Model config (level 0)
        - Vocab metadata (level 0)
        - Layer structures (levels 1-3)
        """
        if not self.fg:
            return False
            
        try:
            from eva_ai.memory.fractal_graph_v2.gguf_parser import GGUFModelParser
            
            parser = GGUFModelParser()
            model_info = parser.parse(model_path)
            
            # 1. Сохраняем config модели
            self._create_model_config_node(model_info)
            
            # 2. Сохраняем metadata vocab (не полный vocab!)
            self._create_vocab_metadata_node(model_info)
            
            # 3. Создаём структуру слоёв
            self._create_layer_structure(model_info)
            
            logger.info(f"Model architecture loaded: {model_info.model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Architecture load error: {e}")
            return False
    
    def _create_model_config_node(self, model_info):
        """Создать узел конфигурации модели (level 0)."""
        self.fg.add_node(
            content=f"Model: {model_info.model_type}, Hidden: {model_info.hidden_size}, Layers: {model_info.num_layers}",
            node_type="model_config",
            level=0,
            confidence=1.0,
            metadata={
                'source': 'gguf_architecture',
                'architecture': model_info.architecture,
                'hidden_size': model_info.hidden_size,
                'num_layers': model_info.num_layers,
                'num_attention_heads': model_info.num_attention_heads,
                'intermediate_size': model_info.intermediate_size,
                'max_position_embeddings': model_info.max_position_embeddings,
                'vocab_size': model_info.vocab_size,
                'is_static': True  # Не удалять
            }
        )
    
    def _create_vocab_metadata_node(self, model_info):
        """Создать metadata узлы для vocab (без полного vocab!)."""
        # Сохраняем только metadata, не все 150K токенов
        self.fg.add_node(
            content=f"Vocab size: {model_info.vocab_size}, Type: {model_info.tokenizer_type}",
            node_type="vocab_metadata",
            level=0,
            confidence=1.0,
            metadata={
                'source': 'gguf_vocab',
                'vocab_size': model_info.vocab_size,
                'tokenizer_type': model_info.tokenizer_type,
                'chat_template': model_info.chat_template[:500] if model_info.chat_template else "",
                'is_static': True
            }
        )
    
    def _create_layer_structure(self, model_info):
        """Создать структуру слоёв в FG."""
        num_layers = model_info.num_layers or 28
        
        # Early layers (2-8) -> level 1
        for layer_id in range(2, min(9, num_layers)):
            self.fg.add_node(
                content=f"Layer {layer_id}: early attention + ffn",
                node_type="layer_early",
                level=1,
                confidence=0.9,
                metadata={
                    'layer_id': layer_id,
                    'layer_type': 'early',
                    'source': 'gguf_layer'
                }
            )
        
        # Middle layers (9-20) -> level 2
        for layer_id in range(9, min(21, num_layers)):
            self.fg.add_node(
                content=f"Layer {layer_id}: middle attention + ffn",
                node_type="layer_middle",
                level=2,
                confidence=0.9,
                metadata={
                    'layer_id': layer_id,
                    'layer_type': 'middle',
                    'source': 'gguf_layer'
                }
            )
        
        # Late layers (21+) -> level 3
        for layer_id in range(21, num_layers):
            self.fg.add_node(
                content=f"Layer {layer_id}: late reasoning attention + ffn",
                node_type="layer_late",
                level=3,
                confidence=0.8,
                metadata={
                    'layer_id': layer_id,
                    'layer_type': 'late',
                    'source': 'gguf_layer'
                }
            )
        
        logger.info(f"Created layer structure: {num_layers} layers mapped to levels 1-3")
    
    def create_aci_concept_from_context(self, context: str, query: str) -> Optional[str]:
        """
        Использовать ACI (ConceptMiner) для создания концепта из контекста.
        
        Это ключевая интеграция:
        1. semantic_search находит релевантные узлы
        2. ACI создаёт новый концепт
        3. add_knowledge связывает с существующими
        """
        if not self.fg:
            return None
            
        try:
            # 1. Поиск релевантных узлов
            relevant = self.fg.semantic_search(query, top_k=5, min_similarity=0.5)
            
            if not relevant:
                # Нет релевантных - создаём новый концепт с нуля
                concept_id = self._create_new_concept(context, query)
                return concept_id
            
            # 2. ACI: создаём концепт на основе найденных
            concept_id = self._create_aci_concept(relevant, context, query)
            
            # 3. Связываем через add_knowledge
            for r in relevant[:3]:
                node_id = r.get('id', '')
                if node_id:
                    self.fg.add_knowledge(
                        subject=query[:50],
                        relation="related_to",
                        object_=node_id
                    )
            
            return concept_id
            
        except Exception as e:
            logger.debug(f"ACI concept creation error: {e}")
            return None
    
    def _create_new_concept(self, context: str, query: str) -> str:
        """Создать новый концепт."""
        node = self.fg.add_node(
            content=f"Concept: {query[:100]}",
            node_type="aci_concept",
            level=2,
            confidence=0.7,
            metadata={
                'source': 'aci_generation',
                'context': context[:200],
                'is_dynamic': True
            }
        )
        return node.id if node else ""
    
    def _create_aci_concept(self, relevant: List[Dict], context: str, query: str) -> str:
        """Создать концепт на основе релевантных узлов."""
        # Объединяем контекст из релевантных
        combined_context = context
        for r in relevant:
            if r.get('content'):
                combined_context += " " + r['content'][:100]
        
        node = self.fg.add_node(
            content=f"ACI Concept: {query[:80]} - {len(relevant)} related",
            node_type="aci_concept",
            level=2,
            confidence=0.8,
            metadata={
                'source': 'aci_generation',
                'related_count': len(relevant),
                'context': combined_context[:300],
                'is_dynamic': True
            }
        )
        return node.id if node else ""
    
    def get_hot_window_stats(self) -> Dict[str, Any]:
        """Получить статистику hot window."""
        if not self.fg:
            return {"error": "No FG"}
            
        stats = self.fg.get_stats() if hasattr(self.fg, 'get_stats') else {}
        
        # Подсчитаем по уровням
        level_counts = {}
        for node_id, node in self.fg.storage.nodes.items():
            level = getattr(node, 'level', 1)
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            "total_nodes": stats.get('total_nodes', 0),
            "by_level": level_counts,
            "hot_window_max": self.hot_window_max
        }


def create_architecture_mapper(fg, config: Dict = None) -> GGUFToFGArchitectureMapper:
    """Создать маппер."""
    return GGUFToFGArchitectureMapper(fg, config)