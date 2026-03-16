#!/usr/bin/env python3
"""
Упрощенная загрузка ruGPT-3 Medium с фрактальным хранилищем
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
from cogniflex.mlearning.fractal_rugpt3_manager import FractalRuGPT3Manager, RUSSIAN_MODELS
from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig

def load_rugpt3_medium_simple():
    """Простая загрузка ruGPT-3 Medium"""
    setup_logging(log_dir='logs')
    
    print("🚀 Загрузка ruGPT-3 Medium...")
    
    # Проверяем GPU
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"
    print(f"🎯 Устройство: {device}")
    
    # Создаем конфигурацию
    config = ModelStorageConfig(
        base_path="./fractal_storage",
        device=device,
        block_size=1024,
        fractal_levels=4
    )
    
    # Создаем менеджер
    print("📦 Создание FractalRuGPT3Manager...")
    manager = FractalRuGPT3Manager(
        brain=None,
        model_name="rugpt3medium",
        storage_path="./fractal_storage"
    )
    
    # Инициализация
    print("🔧 Инициализация...")
    init_start = time.time()
    
    try:
        if manager.initialize():
            init_time = time.time() - init_start
            print(f"✅ Менеджер инициализирован за {init_time:.2f} сек")
            
            # Информация о модели
            model_info = RUSSIAN_MODELS.get("rugpt3medium", {})
            print(f"📊 Модель: {model_info.get('description', 'Unknown')}")
            print(f"💾 Размер: {model_info.get('size_mb', 'Unknown')} MB")
            print(f"⭐ Качество: {model_info.get('quality', 'Unknown')}/10")
            
            # Токенизатор
            if manager.tokenizer:
                vocab_size = manager.tokenizer.vocab_size
                print(f"📝 Размер словаря: {vocab_size} токенов")
                
                # Тест токенизации
                test_text = "Привет! Как дела?"
                tokens = manager.tokenizer.encode(test_text)
                print(f"🧪 Токенизация: '{test_text}' -> {len(tokens)} токенов")
                print(f"   Токены: {tokens[:10]}...")
            
            # Модель
            if manager.model:
                if hasattr(manager.model, 'config'):
                    config = manager.model.config
                    if hasattr(config, 'n_embd'):
                        print(f"🧠 Размерность: {config.n_embd}")
                    if hasattr(config, 'n_head'):
                        print(f"👥 Голов внимания: {config.n_head}")
                    if hasattr(config, 'n_layer'):
                        print(f"📚 Слои: {config.n_layer}")
                
                # Подсчет параметров
                total_params = sum(p.numel() for p in manager.model.parameters())
                trainable_params = sum(p.numel() for p in manager.model.parameters() if p.requires_grad)
                
                print(f"🔢 Всего параметров: {total_params:,}")
                print(f"🎯 Обучаемых параметров: {trainable_params:,}")
                print(f"💾 Размер модели в памяти: {total_params * 4 / (1024**2):.1f} MB")
            
            # Тест генерации
            print("\n🎯 Тест генерации...")
            test_queries = [
                "Привет! Как дела?",
                "Что такое искусственный интеллект?",
                "Расскажи о машинном обучении"
            ]
            
            for query in test_queries:
                print(f"\n📝 Запрос: {query}")
                start_time = time.time()
                
                try:
                    response = manager.generate_response(
                        query,
                        max_tokens=80,
                        temperature=0.7,
                        do_sample=True
                    )
                    
                    gen_time = time.time() - start_time
                    print(f"💬 Ответ: {response[:150]}...")
                    print(f"⏱️ Время: {gen_time:.2f} сек")
                    
                except Exception as e:
                    print(f"❌ Ошибка генерации: {e}")
            
            # Экспорт
            print(f"\n📦 Экспорт модели...")
            export_path = "./fractal_exports/rugpt3medium"
            
            if manager.export_model(export_path):
                print(f"✅ Модель экспортирована в {export_path}")
                
                # Проверка размера
                export_size = 0
                for root, dirs, files in os.walk(export_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        export_size += os.path.getsize(file_path)
                
                export_size_mb = export_size / (1024 * 1024)
                print(f"💾 Размер экспорта: {export_size_mb:.1f} MB")
                
                # Содержимое
                print(f"\n📁 Содержимое экспорта:")
                for root, dirs, files in os.walk(export_path):
                    level = root.replace(export_path, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                        print(f"{subindent}{file} ({file_size:.1f} MB)")
            else:
                print(f"❌ Не удалось экспортировать модель")
            
            # Сохранение метаданных
            metadata = {
                "model_name": "rugpt3medium",
                "model_info": model_info,
                "device": device,
                "gpu_available": gpu_available,
                "load_time": init_time,
                "vocab_size": manager.tokenizer.vocab_size if manager.tokenizer else None,
                "total_params": total_params if 'total_params' in locals() else None,
                "export_path": export_path,
                "timestamp": time.time()
            }
            
            metadata_file = Path(export_path) / "load_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"📋 Метаданные сохранены в {metadata_file}")
            
        else:
            print("❌ Не удалось инициализировать менеджер")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Очистка
    print(f"\n🧹 Очистка...")
    try:
        del manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"✅ Очистка завершена")
    except Exception as e:
        print(f"⚠️ Ошибка очистки: {e}")
    
    print(f"\n🎉 Загрузка ruGPT-3 Medium завершена!")
    return True

def test_gpu_tokenization():
    """Тест токенизации на GPU"""
    print("🚀 Тест токенизации на GPU...")
    
    if not torch.cuda.is_available():
        print("❌ GPU не доступен")
        return
    
    print(f"✅ GPU доступен: {torch.cuda.get_device_name()}")
    print(f"💾 Видеопамять: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Пробуем загрузить модель на GPU
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "sberbank-ai/rugpt3small_based_on_gpt2"  # Начнем с smaller модели
        
        print(f"📦 Загрузка {model_name} на GPU...")
        
        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Модель на GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Используем float16 для экономии памяти
            device_map="auto"
        )
        
        print(f"✅ Модель загружена на GPU")
        
        # Тест токенизации
        test_text = "Привет! Как дела? Расскажи о машинном обучении."
        
        # CPU токенизация
        start_time = time.time()
        inputs_cpu = tokenizer.encode(test_text, return_tensors="pt")
        cpu_time = time.time() - start_time
        print(f"⏱️ CPU токенизация: {cpu_time:.4f} сек, токенов: {inputs_cpu.numel()}")
        
        # GPU токенизация (если возможно)
        try:
            start_time = time.time()
            inputs_gpu = tokenizer.encode(test_text, return_tensors="pt").to("cuda")
            gpu_time = time.time() - start_time
            print(f"⚡ GPU токенизация: {gpu_time:.4f} сек, токенов: {inputs_gpu.numel()}")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"🚀 Ускорение: {speedup:.1f}x")
        except Exception as e:
            print(f"⚠️ GPU токенизация не удалась: {e}")
        
        # Тест генерации на GPU
        print(f"\n🎯 Тест генерации на GPU...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs_gpu,
                max_length=inputs_gpu.shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"💬 Ответ: {response[:150]}...")
        print(f"⏱️ Время генерации: {gen_time:.2f} сек")
        
        # Память GPU
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"💾 Использование GPU: {memory_used:.1f}/{memory_total:.1f} GB")
        
        # Очистка
        del model, inputs_cpu, inputs_gpu, outputs
        torch.cuda.empty_cache()
        print(f"✅ GPU память очищена")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки на GPU: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Загрузка ruGPT-3 Medium")
    parser.add_argument("--load", action="store_true", help="Загрузить ruGPT-3 Medium")
    parser.add_argument("--gpu-test", action="store_true", help="Тест GPU токенизации")
    
    args = parser.parse_args()
    
    if args.load:
        load_rugpt3_medium_simple()
    elif args.gpu_test:
        test_gpu_tokenization()
    else:
        print("Используйте: --load или --gpu-test")
        load_rugpt3_medium_simple()
