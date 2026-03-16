#!/usr/bin/env python3
"""
Создание базовой модели эмбеддингов для CogniFlex
"""

import torch
import torch.nn as nn
from pathlib import Path
import json

def create_basic_embedding_model():
    """Создает базовую модель эмбеддингов"""
    
    class BasicEmbeddingModel(nn.Module):
        def __init__(self, vocab_size=50000, embedding_dim=768, max_length=512):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.max_length = max_length
            self.vocab_size = vocab_size
            
            # Основной слой эмбеддингов
            self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.position_embeddings = nn.Embedding(max_length, embedding_dim)
            
            # Нормализация
            self.layer_norm = nn.LayerNorm(embedding_dim)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, input_ids):
            seq_length = input_ids.size(1)
            
            # Создаем позиционные ID
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
            # Получаем эмбеддинги
            token_embeddings = self.token_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            
            # Комбинируем и нормализуем
            embeddings = self.layer_norm(token_embeddings + position_embeddings)
            embeddings = self.dropout(embeddings)
            
            return embeddings
    
    # Создаем модель
    model = BasicEmbeddingModel()
    
    # Инициализируем веса
    nn.init.normal_(model.token_embeddings.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.position_embeddings.weight, mean=0.0, std=0.02)
    
    return model

def save_embedding_model():
    """Сохраняет модель эмбеддингов"""
    
    # Пути
    embeddings_dir = Path("cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем модель
    model = create_basic_embedding_model()
    
    # Сохраняем в safetensors формате
    model_path = embeddings_dir / "embedding_model.safetensors"
    
    try:
        from safetensors.torch import save_file
        # Сохраняем только состояние модели
        state_dict = model.state_dict()
        save_file(state_dict, str(model_path))
        print(f"✅ Модель эмбеддингов сохранена: {model_path}")
    except ImportError:
        # Fallback: сохраняем в формате .pt
        model_path = embeddings_dir / "embedding_model.pt"
        torch.save(model.state_dict(), str(model_path))
        print(f"⚠️ Модель сохранена в .pt формате: {model_path}")
    
    # Создаем метаданные
    metadata = {
        "model_type": "basic_embedding",
        "vocab_size": 50000,
        "embedding_dim": 768,
        "max_length": 512,
        "architecture": "BasicEmbeddingModel",
        "created_at": "2025-06-17T00:00:00Z",
        "languages": ["ru", "en"],
        "file_name": model_path.name
    }
    
    metadata_path = embeddings_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Метаданные сохранены: {metadata_path}")
    return model_path

if __name__ == "__main__":
    save_embedding_model()
