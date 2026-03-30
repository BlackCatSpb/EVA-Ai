"""
Нейроморфный слой памяти для интеграции с Fractal Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import logging

logger = logging.getLogger("eva.neuromorphic_memory")

class NeuromorphicMemoryLayer(nn.Module):
    """
    Нейроморфный слой памяти, вдохновленный биологическими нейронными сетями.
    Обеспечивает механизмы для работы с разными типами памяти:
    - Рабочая память (краткосрочная)
    - Семантическая память (долгосрочная)
    - Эпизодическая память (контекстуальная)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        memory_slots: int = 32,
        memory_size: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Инициализация нейроморфного слоя памяти.
        
        Args:
            hidden_size: Размер скрытого состояния
            memory_slots: Количество слотов памяти
            memory_size: Размерность каждого слота памяти
            num_heads: Количество голов внимания
            dropout: Вероятность дропаута
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = memory_size // num_heads
        
        # Проверка размерностей
        assert (
            self.head_dim * num_heads == memory_size
        ), "memory_size должен быть кратен num_heads"
        
        # Инициализация памяти
        self.memory = nn.Parameter(torch.zeros(1, memory_slots, memory_size))
        nn.init.xavier_uniform_(self.memory)
        
        # Проекции для запросов, ключей и значений
        self.query_proj = nn.Linear(hidden_size, memory_size)
        self.key_proj = nn.Linear(hidden_size, memory_size)
        self.value_proj = nn.Linear(hidden_size, memory_size)
        
        # Выходные проекции
        self.output_proj = nn.Linear(memory_size, hidden_size)
        
        # Нормализация и дропаут
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Инициализация весов проекций."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        if self.query_proj.bias is not None:
            nn.init.constant_(self.query_proj.bias, 0.)
        if self.key_proj.bias is not None:
            nn.init.constant_(self.key_proj.bias, 0.)
        if self.value_proj.bias is not None:
            nn.init.constant_(self.value_proj.bias, 0.)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Разделение тензора на головы внимания."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Объединение голов внимания."""
        x = x.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len, _, _ = x.size()
        return x.view(batch_size, seq_len, -1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_memory: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Прямой проход через нейроморфный слой памяти.
        
        Args:
            hidden_states: Входные скрытые состояния (batch_size, seq_len, hidden_size)
            attention_mask: Маска внимания (batch_size, seq_len)
            past_memory: Предыдущее состояние памяти (batch_size, memory_slots, memory_size)
            use_memory: Использовать ли механизм памяти
            
        Returns:
            Кортеж (output_states, memory_outputs), где:
            - output_states: Выходные скрытые состояния (batch_size, seq_len, hidden_size)
            - memory_outputs: Словарь с дополнительными выходами памяти
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Инициализируем память, если не передана
        if past_memory is None:
            past_memory = self.memory.expand(batch_size, -1, -1)
        
        # Проекции для запросов, ключей и значений
        queries = self.query_proj(hidden_states)  # (batch_size, seq_len, memory_size)
        keys = self.key_proj(hidden_states)       # (batch_size, seq_len, memory_size)
        values = self.value_proj(hidden_states)   # (batch_size, seq_len, memory_size)
        
        # Разделение на головы
        queries = self._split_heads(queries)  # (batch_size, num_heads, seq_len, head_dim)
        keys = self._split_heads(keys)        # (batch_size, num_heads, seq_len, head_dim)
        values = self._split_heads(values)    # (batch_size, num_heads, seq_len, head_dim)
        
        # Вычисление внимания между входом и памятью
        memory_keys = self._split_heads(self.key_proj(past_memory))  # (batch_size, num_heads, memory_slots, head_dim)
        memory_values = self._split_heads(self.value_proj(past_memory))  # (batch_size, num_heads, memory_slots, head_dim)
        
        # Вычисление весов внимания между входом и памятью
        attention_scores = torch.matmul(
            queries, memory_keys.transpose(-1, -2)  # (batch_size, num_heads, seq_len, memory_slots)
        ) / math.sqrt(self.head_dim)
        
        # Применение маски внимания, если есть
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )
        
        # Применение softmax для получения весов внимания
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Взвешенная сумма значений памяти
        context = torch.matmul(attention_probs, memory_values)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Объединение голов
        context = self._combine_heads(context)  # (batch_size, seq_len, memory_size)
        
        # Выходной слой
        output = self.output_proj(context)  # (batch_size, seq_len, hidden_size)
        
        # Добавление остаточного соединения и нормализация
        output = self.layer_norm(hidden_states + self.dropout(output))
        
        # Подготовка выходных данных памяти
        memory_outputs = {
            'memory': past_memory,
            'attention_probs': attention_probs,
            'memory_updated': self._update_memory(
                keys, values, attention_probs, past_memory
            )
        }
        
        return output, memory_outputs
    
    def _update_memory(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_probs: torch.Tensor,
        current_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Обновление состояния памяти на основе текущего входа.
        
        Args:
            keys: Ключи внимания (batch_size, num_heads, seq_len, head_dim)
            values: Значения внимания (batch_size, num_heads, seq_len, head_dim)
            attention_probs: Веса внимания (batch_size, num_heads, seq_len, memory_slots)
            current_memory: Текущее состояние памяти (batch_size, memory_slots, memory_size)
            
        Returns:
            Обновленное состояние памяти (batch_size, memory_slots, memory_size)
        """
        batch_size, num_heads, seq_len, head_dim = keys.size()
        
        # Объединяем головы для ключей и значений
        keys = self._combine_heads(keys)  # (batch_size, seq_len, memory_size)
        values = self._combine_heads(values)  # (batch_size, seq_len, memory_size)
        
        # Вычисляем веса обновления для каждого слота памяти
        update_weights = attention_probs.sum(dim=2)  # (batch_size, num_heads, memory_slots)
        update_weights = update_weights.mean(dim=1)  # (batch_size, memory_slots)
        update_weights = update_weights.unsqueeze(-1)  # (batch_size, memory_slots, 1)
        
        # Вычисляем обновления для памяти
        memory_updates = torch.bmm(
            attention_probs.mean(dim=1).transpose(1, 2),  # (batch_size, memory_slots, seq_len)
            values  # (batch_size, seq_len, memory_size)
        )  # (batch_size, memory_slots, memory_size)
        
        # Применяем веса обновления
        memory_updates = memory_updates * update_weights
        
        # Обновляем память
        updated_memory = current_memory + memory_updates
        
        # Нормализуем обновленную память
        updated_memory = F.layer_norm(
            updated_memory, (self.memory_size,)
        )
        
        return updated_memory
    
    def get_initial_memory(self, batch_size: int = 1) -> torch.Tensor:
        """
        Возвращает начальное состояние памяти.
        
        Args:
            batch_size: Размер батча
            
        Returns:
            Начальное состояние памяти (batch_size, memory_slots, memory_size)
        """
        return self.memory.expand(batch_size, -1, -1)
