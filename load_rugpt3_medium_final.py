#!/usr/bin/env python3
"""
Загрузка ruGPT-3 Medium с фрактальным хранилищем
"""

import os
import sys
import logging
import time
import torch
from pathlib import Path

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniflex.core.utils import setup_logging
from cogniflex.mlearning.fractal_rugpt3_manager import FractalRuGPT3Manager
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_rugpt3_medium():
    """Загружает ruGPT-3 Medium"""
    setup_logging(log_dir='logs')
    
    print("🚀 Загрузка ruGPT-3 Medium...")
    
    model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🎯 Устройство: {device}")
    print(f"📦 Модель: {model_name}")
    
    try:
        # Загрузка токенизатора
        print("📝 Загрузка токенизатора...")
        tokenizer_start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer_time = time.time() - tokenizer_start
        print(f"✅ Токенизатор загружен за {tokenizer_time:.2f} сек")
        print(f"📝 Словарь: {tokenizer.vocab_size} токенов")
        
        # Загрузка модели
        print("🧠 Загрузка модели...")
        model_start = time.time()
        
        # Оптимизированные параметры для загрузки
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"  # Загружаем на CPU для стабильности
        )
        
        model_time = time.time() - model_start
        print(f"✅ Модель загружена за {model_time:.2f} сек")
        
        # Информация о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🔢 Всего параметров: {total_params:,}")
        print(f"🎯 Обучаемых: {trainable_params:,}")
        print(f"💾 Размер в памяти: {total_params * 4 / (1024**2):.1f} MB")
        
        # Конфигурация
        config = model.config
        print(f"🧠 Тип: {config.model_type}")
        if hasattr(config, 'n_embd'):
            print(f"📐 Размерность: {config.n_embd}")
        if hasattr(config, 'n_layer'):
            print(f"📚 Слои: {config.n_layer}")
        if hasattr(config, 'n_head'):
            print(f"👥 Головы: {config.n_head}")
        
        # Тест токенизации
        print("\n🧪 Тест токенизации...")
        test_texts = [
            "Привет! Как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о машинном обучении и нейронных сетях",
            "Преимущества фрактального хранилища для моделей"
        ]
        
        for text in test_texts:
            start_time = time.time()
            tokens = tokenizer.encode(text, return_tensors="pt")
            token_time = time.time() - start_time
            
            print(f"📝 '{text[:40]}...' -> {tokens.numel()} токенов ({token_time:.4f} сек)")
        
        # Тест генерации
        print("\n🎯 Тест генерации...")
        for query in test_texts[:3]:  # Только первые 3
            print(f"\n📝 Запрос: {query}")
            
            start_time = time.time()
            
            with torch.no_grad():
                inputs = tokenizer.encode(query, return_tensors="pt")
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            gen_time = time.time() - start_time
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Очистка от запроса
            if response.startswith(query):
                response = response[len(query):].strip()
            
            print(f"💬 Ответ: {response[:150]}...")
            print(f"⏱️ Время: {gen_time:.2f} сек")
        
        # Сохранение во фрактальное хранилище
        print(f"\n💾 Сохранение во фрактальное хранилище...")
        
        # Создаем фрактальный менеджер
        fractal_manager = FractalRuGPT3Manager(
            brain=None,
            model_name="rugpt3medium",
            storage_path="./fractal_storage"
        )
        
        # Сохраняем модель
        try:
            fractal_manager.model = model
            fractal_manager.tokenizer = tokenizer
            fractal_manager.initialized = True
            
            # Экспорт модели
            export_path = "./fractal_exports/rugpt3medium"
            if fractal_manager.export_model(export_path):
                print(f"✅ Модель экспортирована во фрактальное хранилище: {export_path}")
                
                # Проверка размера
                total_size = 0
                for root, dirs, files in os.walk(export_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                
                print(f"💾 Размер фрактального экспорта: {total_size / (1024**2):.1f} MB")
                
            else:
                print(f"❌ Не удалось экспортировать модель")
                
        except Exception as e:
            print(f"⚠️ Ошибка фрактального экспорта: {e}")
        
        # Сохранение стандартной модели
        print(f"\n💾 Сохранение стандартной модели...")
        save_path = "./saved_models/rugpt3medium"
        os.makedirs(save_path, exist_ok=True)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"✅ Модель сохранена в {save_path}")
        
        # Проверка размера
        total_size = 0
        for root, dirs, files in os.walk(save_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        print(f"💾 Размер стандартной модели: {total_size / (1024**2):.1f} MB")
        
        # Создаем метаданные
        metadata = {
            "model_name": model_name,
            "description": "ruGPT-3 Medium",
            "total_params": total_params,
            "vocab_size": tokenizer.vocab_size,
            "device": device,
            "load_time": {
                "tokenizer": tokenizer_time,
                "model": model_time,
                "total": tokenizer_time + model_time
            },
            "save_size_mb": total_size / (1024**2),
            "fractal_export": os.path.exists("./fractal_exports/rugpt3medium"),
            "timestamp": time.time()
        }
        
        import json
        with open(os.path.join(save_path, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"📋 Метаданные сохранены")
        
        # Очистка
        print(f"\n🧹 Очистка...")
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ ruGPT-3 Medium успешно загружен и сохранен!")
        
        # Итоговая статистика
        print(f"\n📊 Итоговая статистика:")
        print(f"  🤖 Модель: ruGPT-3 Medium")
        print(f"  🔢 Параметры: {total_params:,}")
        print(f"  💾 Размер: {total_size / (1024**2):.1f} MB")
        print(f"  ⏱️ Время загрузки: {tokenizer_time + model_time:.2f} сек")
        print(f"  🎯 Устройство: {device}")
        print(f"  💾 Фрактальное хранилище: {'✅' if os.path.exists('./fractal_exports/rugpt3medium') else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    load_rugpt3_medium()
