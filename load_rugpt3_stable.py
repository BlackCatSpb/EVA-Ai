#!/usr/bin/env python3
"""
Загрузка ruGPT-3 моделей с проверкой доступности и устойчивой загрузкой
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
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_available_models():
    """Проверяет доступные русскоязычные модели"""
    print("🔍 Проверка доступных русскоязычных моделей...")
    
    models_to_check = [
        {
            "name": "sberbank-ai/rugpt3small_based_on_gpt2",
            "description": "ruGPT-3 Small",
            "size": "~600MB"
        },
        {
            "name": "sberbank-ai/rugpt3large_based_on_gpt2", 
            "description": "ruGPT-3 Medium",
            "size": "~1.5GB"
        },
        {
            "name": "sberbank-ai/rugpt2small",
            "description": "ruGPT-2 Small", 
            "size": "~500MB"
        },
        {
            "name": "ai-forever/ruGPT-3.5-13B",
            "description": "ruGPT-3.5 13B",
            "size": "~26GB"
        }
    ]
    
    for model_info in models_to_check:
        print(f"\n🤖 {model_info['description']} ({model_info['size']})")
        print(f"   ID: {model_info['name']}")
        
        try:
            # Проверяем токенизатор
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_info['name'], trust_remote_code=True)
            tokenizer_time = time.time() - start_time
            print(f"   ✅ Токенизатор загружен за {tokenizer_time:.2f} сек")
            print(f"   📝 Словарь: {tokenizer.vocab_size} токенов")
            
            # Проверяем конфигурацию модели
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_info['name'], trust_remote_code=True)
                print(f"   🧠 Конфигурация: {config.model_type}")
                if hasattr(config, 'n_embd'):
                    print(f"   📐 Размерность: {config.n_embd}")
                if hasattr(config, 'n_layer'):
                    print(f"   📚 Слои: {config.n_layer}")
                if hasattr(config, 'n_head'):
                    print(f"   👥 Головы: {config.n_head}")
                    
            except Exception as e:
                print(f"   ⚠️ Ошибка конфигурации: {e}")
            
            # Пробуем загрузить модель (только конфигурацию)
            try:
                # Только проверка доступности без загрузки весов
                model = AutoModelForCausalLM.from_pretrained(
                    model_info['name'],
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="cpu"  # Загружаем только на CPU для проверки
                )
                
                total_params = sum(p.numel() for p in model.parameters())
                print(f"   🔢 Параметры: {total_params:,}")
                print(f"   💾 Память: {total_params * 4 / (1024**2):.1f} MB")
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ❌ Ошибка загрузки модели: {e}")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

def load_rugpt3_small():
    """Загружает ruGPT-3 Small как более надежную опцию"""
    setup_logging(log_dir='logs')
    
    print("🚀 Загрузка ruGPT-3 Small...")
    
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
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
        
        # Отключаем problematic переменные окружения
        os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
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
            "Расскажи о машинном обучении и нейронных сетях"
        ]
        
        for text in test_texts:
            start_time = time.time()
            tokens = tokenizer.encode(text, return_tensors="pt")
            token_time = time.time() - start_time
            
            print(f"📝 '{text[:30]}...' -> {tokens.numel()} токенов ({token_time:.4f} сек)")
        
        # Тест генерации
        print("\n🎯 Тест генерации...")
        for query in test_texts[:2]:  # Только первые 2
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
        
        # Сохранение модели
        print(f"\n💾 Сохранение модели...")
        save_path = "./saved_models/rugpt3small"
        os.makedirs(save_path, exist_ok=True)
        
        # Сохраняем модель
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"✅ Модель сохранена в {save_path}")
        
        # Проверка размера
        total_size = 0
        for root, dirs, files in os.walk(save_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        print(f"💾 Размер сохраненной модели: {total_size / (1024**2):.1f} MB")
        
        # Создаем метаданные
        metadata = {
            "model_name": model_name,
            "description": "ruGPT-3 Small",
            "total_params": total_params,
            "vocab_size": tokenizer.vocab_size,
            "device": device,
            "load_time": {
                "tokenizer": tokenizer_time,
                "model": model_time,
                "total": tokenizer_time + model_time
            },
            "save_size_mb": total_size / (1024**2),
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
        
        print(f"✅ ruGPT-3 Small успешно загружена и сохранена!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_performance():
    """Тестирует производительность на GPU"""
    print("🚀 Тест производительности на GPU...")
    
    if not torch.cuda.is_available():
        print("❌ GPU не доступен")
        return
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"✅ GPU: {gpu_name}")
    print(f"💾 Память: {gpu_memory:.1f} GB")
    
    # Пробуем загрузить ruGPT-3 Small на GPU
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    
    try:
        print(f"\n📦 Загрузка {model_name} на GPU...")
        
        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Модель на GPU с оптимизацией
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # float16 для экономии памяти
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Модель загружена на GPU")
        
        # Память
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"💾 Использование GPU: {memory_used:.1f}/{gpu_memory:.1f} GB ({memory_used/gpu_memory*100:.1f}%)")
        
        # Тест токенизации
        test_text = "Привет! Как дела? Расскажи о машинном обучении."
        
        # CPU токенизация
        start_time = time.time()
        inputs_cpu = tokenizer.encode(test_text, return_tensors="pt")
        cpu_time = time.time() - start_time
        print(f"⏱️ CPU токенизация: {cpu_time:.4f} сек")
        
        # GPU токенизация (перемещение на GPU)
        start_time = time.time()
        inputs_gpu = inputs_cpu.to("cuda")
        gpu_time = time.time() - start_time
        print(f"⚡ GPU токенизация: {gpu_time:.4f} сек")
        
        # Тест генерации
        print(f"\n🎯 Тест генерации на GPU...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs_gpu,
                max_length=inputs_gpu.shape[1] + 60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"💬 Ответ: {response[:150]}...")
        print(f"⏱️ Время генерации: {gen_time:.2f} сек")
        
        # Сравнение производительности
        print(f"\n📊 Сравнение производительности:")
        if cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"🚀 Ускорение токенизации: {speedup:.1f}x")
        
        # Очистка
        del model, inputs_cpu, inputs_gpu, outputs
        torch.cuda.empty_cache()
        print(f"✅ GPU память очищена")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Загрузка ruGPT-3 моделей")
    parser.add_argument("--check", action="store_true", help="Проверить доступные модели")
    parser.add_argument("--load-small", action="store_true", help="Загрузить ruGPT-3 Small")
    parser.add_argument("--gpu-test", action="store_true", help="Тест GPU производительности")
    
    args = parser.parse_args()
    
    if args.check:
        check_available_models()
    elif args.load_small:
        load_rugpt3_small()
    elif args.gpu_test:
        test_gpu_performance()
    else:
        print("Используйте: --check, --load-small или --gpu-test")
        check_available_models()
