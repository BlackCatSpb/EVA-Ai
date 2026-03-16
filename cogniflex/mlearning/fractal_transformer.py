"""
Fractal Transformer - архитектура трансформера с нативной поддержкой фрактального хранилища.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from transformers import PreTrainedModel, PretrainedConfig
from pathlib import Path
import logging
import math

logger = logging.getLogger("cogniflex.fractal_transformer")

class FractalConfig(PretrainedConfig):
    """Конфигурация для Fractal Transformer."""
    
    def __init__(
        self,
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        fractal_levels=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.fractal_levels = fractal_levels


class FractalAttention(nn.Module):
    """Механизм внимания с поддержкой фрактальной структуры."""
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.fractal_attention = nn.Linear(config.hidden_size, config.num_attention_heads)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None, fractal_metadata=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Расчет внимания с учетом фрактальных метаданных
        if fractal_metadata is not None:
            fractal_attention = self.fractal_attention(fractal_metadata)
            fractal_attention = fractal_attention.unsqueeze(-1).expand_as(query_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores + fractal_attention
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer


class FractalLayer(nn.Module):
    """Один слой фрактального трансформера."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = FractalAttention(config)
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None, fractal_metadata=None):
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            fractal_metadata=fractal_metadata
        )
        hidden_states = self.layernorm1(hidden_states + attention_output)
        
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.layernorm2(hidden_states + intermediate_output)
        
        return layer_output


class FractalTransformer(PreTrainedModel):
    """
    Основной класс Fractal Transformer с нативной поддержкой фрактального хранилища.
    """
    
    config_class = FractalConfig
    base_model_prefix = "fractal_transformer"
    
    def __init__(self, config, brain=None):
        super().__init__(config)
        self.config = config
        self.brain = brain
        
        # Инициализация эмбеддингов
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)  # Для сегментов
        
        # Инициализация слоев
        self.layers = nn.ModuleList([FractalLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Инициализация весов
        self.init_weights()
        
    def init_weights(self):
        """Инициализация весов."""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Инициализация весов для различных типов слоев."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        fractal_metadata=None,
        **kwargs
    ):
        """
        Прямой проход через модель.
        
        Args:
            input_ids: Токены входной последовательности
            attention_mask: Маска внимания
            token_type_ids: Идентификаторы сегментов
            position_ids: Позиционные индексы
            fractal_metadata: Метаданные фрактальной структуры
            
        Returns:
            Выходные скрытые состояния
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("Необходимо указать input_ids")
        
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        try:
            # Получаем эмбеддинги
            words_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            
            # Суммируем эмбеддинги
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
            embeddings = self.dropout(embeddings)
            
            # Проходим через слои
            hidden_states = embeddings
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    fractal_metadata=fractal_metadata
                )
            
            return hidden_states
        except Exception as e:
            logger.error(f"Error in FractalTransformer forward: {e}", exc_info=True)
            raise
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Загрузка предобученной модели."""
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
    
    def save_pretrained(self, save_directory):
        """Сохранение модели."""
        super().save_pretrained(save_directory)
        
    def get_input_embeddings(self):
        return self.word_embeddings
    
    def set_input_embeddings(self, value):
        self.word_embeddings = value
