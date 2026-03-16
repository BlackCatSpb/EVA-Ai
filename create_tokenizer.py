#!/usr/bin/env python3
"""
Создание токенизатора для ruGPT3 модели
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, GPT2Tokenizer

def create_tokenizer():
    """Создает и сохраняет токенизатор для модели"""
    
    # Пути
    model_dir = Path("cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/text-generation")
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Пробуем загрузить ruGPT3 токенизатор
        tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
        
        # Сохраняем токенизатор локально
        tokenizer.save_pretrained(str(tokenizer_dir))
        print(f"✅ Токенизатор ruGPT3 сохранен: {tokenizer_dir}")
        
    except Exception as e:
        print(f"⚠️ Не удалось загрузить ruGPT3 токенизатор: {e}")
        
        # Fallback: создаем базовый GPT2 токенизатор
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            # Добавляем pad_token если его нет
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            tokenizer.save_pretrained(str(tokenizer_dir))
            print(f"✅ Базовый токенизатор GPT2 сохранен: {tokenizer_dir}")
            
        except Exception as e2:
            print(f"❌ Не удалось создать токенизатор: {e2}")
            return False
    
    # Создаем конфигурацию токенизатора
    tokenizer_config = {
        "vocab_size": getattr(tokenizer, 'vocab_size', 50257),
        "model_max_length": getattr(tokenizer, 'model_max_length', 1024),
        "padding_side": "right",
        "truncation_side": "right",
        "use_fast": True,
        "auto_map": {
            "AutoTokenizer": ["tokenization_gpt2.GPT2Tokenizer", "tokenization_gpt2_fast.GPT2TokenizerFast"]
        }
    }
    
    import json
    config_path = tokenizer_dir / "tokenizer_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"✅ Конфигурация токенизатора сохранена: {config_path}")
    return True

if __name__ == "__main__":
    create_tokenizer()
