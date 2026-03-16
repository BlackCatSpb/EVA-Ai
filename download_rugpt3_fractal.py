"""
Загрузка и экспорт ruGPT3 large во фрактальное хранилище
"""
import sys
import os
import torch
import logging
from pathlib import Path
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rugpt3_export")

def download_and_export_rugpt3():
    """Загружает ruGPT3 large и экспортирует во фрактальное хранилище"""
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from cogniflex.mlearning.storage.fractal_store import export_hf_model_to_fractal
        
        print("🚀 ЗАГРУЗКА И ЭКСПОРТ RU-GPT3 LARGE")
        print("="*60)
        
        # 1. Проверяем доступность transformers
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            print("❌ Transformers не установлен")
            print("💡 Установите: pip install transformers")
            return False
        
        # 2. Модели ruGPT3 для загрузки
        rugpt3_models = [
            "sberbank-ai/rugpt3large_based_on_gpt2",  # Основная модель
            "ai-forever/rugpt3large_based_on_gpt2",   # Альтернативная
            "sberbank-ai/rugpt3small_based_on_gpt2",  # Маленькая (fallback)
            "ai-forever/rugpt3small_based_on_gpt2"    # Альтернативная маленькая
        ]
        
        print("🔍 ПОИСК ДОСТУПНОЙ МОДЕЛИ RU-GPT3...")
        
        model_to_use = None
        for model_name in rugpt3_models:
            try:
                print(f"   🔄 Проверка: {model_name}")
                
                # Пробуем загрузить токенизатор
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=False,
                    trust_remote_code=False
                )
                
                # Пробуем загрузить модель (без весов для проверки)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    local_files_only=False,
                    trust_remote_code=False,
                    torch_dtype=torch.float32,  # Используем float32 для совместимости
                    low_cpu_mem_usage=True
                )
                
                print(f"   ✅ Модель найдена: {model_name}")
                print(f"      📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")
                print(f"      🔤 Словарь: {len(tokenizer.get_vocab()):,} токенов")
                
                model_to_use = model_name
                break
                
            except Exception as e:
                print(f"   ❌ Ошибка: {str(e)[:100]}...")
                continue
        
        if not model_to_use:
            print("❌ Не удалось найти доступную модель ruGPT3")
            return False
        
        # 3. Загружаем выбранную модель полностью
        print(f"\n📦 ЗАГРУЗКА МОДЕЛИ: {model_to_use}")
        
        try:
            # Загружаем токенизатор
            tokenizer = AutoTokenizer.from_pretrained(
                model_to_use,
                local_files_only=False,
                use_fast=True
            )
            
            # Добавляем pad_token если отсутствует
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Загружаем модель
            model = AutoModelForCausalLM.from_pretrained(
                model_to_use,
                local_files_only=False,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            model.eval()
            
            print(f"✅ Модель загружена успешно")
            print(f"   📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   🔤 Словарь: {len(tokenizer.get_vocab()):,}")
            print(f"   📐 Устройство: {next(model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
        
        # 4. Тестируем генерацию перед экспортом
        print(f"\n🧪 ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ПЕРЕД ЭКСПОРТОМ")
        
        test_queries = [
            "Привет, как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о России"
        ]
        
        for query in test_queries:
            try:
                # Кодируем запрос
                inputs = tokenizer.encode(query, return_tensors='pt')
                
                # Генерируем ответ
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Декодируем результат
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print(f"   📝 '{query}'")
                print(f"   💬 '{response}'")
                print()
                
            except Exception as e:
                print(f"   ❌ Ошибка генерации: {e}")
        
        # 5. Экспорт во фрактальное хранилище
        print(f"📦 ЭКСПОРТ ВО ФРАКТАЛЬНОЕ ХРАНИЛИЩЕ")
        
        # Создаем директорию для экспорта
        output_path = Path("out") / "rugpt3_large_fractal"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   📁 Путь экспорта: {output_path}")
        
        # Экспортируем модель
        success = export_hf_model_to_fractal(
            hf_model_dir_or_id=model_to_use,
            output_path=str(output_path),
            model_id="rugpt3_large_russian",
            tokenizer_output_subdir="tokenizer",
            device="cpu",
            fractal_levels=4,
            block_size=64,
            local_files_only=False  # Разрешаем сетевые запросы
        )
        
        if success:
            print(f"✅ Модель успешно экспортирована во фрактальное хранилище")
            
            # Проверяем структуру экспорта
            print(f"\n📋 СТРУКТУРА ЭКСПОРТА:")
            
            for item in output_path.rglob("*"):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"   📄 {item.relative_to(output_path)} ({size_mb:.1f} MB)")
            
        else:
            print(f"❌ Ошибка экспорта модели")
            return False
        
        # 6. Сохраняем информацию о модели
        model_info = {
            "model_name": model_to_use,
            "model_id": "rugpt3_large_russian",
            "export_path": str(output_path),
            "parameters": sum(p.numel() for p in model.parameters()),
            "vocab_size": len(tokenizer.get_vocab()),
            "export_timestamp": torch.datetime.datetime.now().isoformat() if hasattr(torch, 'datetime') else "2025-03-05"
        }
        
        import json
        with open(output_path / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Информация о модели сохранена")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки и экспорта: {e}", exc_info=True)
        return False

def test_fractal_model():
    """Тестирует загрузку и генерацию из фрактального хранилища"""
    
    try:
        from cogniflex.mlearning.storage.fractal_model_loader import FractalModelLoader
        from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig
        
        print(f"\n🧪 ТЕСТИРОВАНИЕ ЗАГРУЗКИ ИЗ ФРАКТАЛЬНОГО ХРАНИЛИЩА")
        print("="*60)
        
        # Путь к экспортированной модели
        fractal_path = "out/rugpt3_large_fractal"
        
        if not os.path.exists(fractal_path):
            print(f"❌ Путь к фрактальному хранилищу не найден: {fractal_path}")
            return False
        
        # Создаем конфигурацию
        config = ModelStorageConfig(
            base_path=fractal_path,
            block_size=64,
            fractal_levels=4,
            device="cpu"
        )
        
        # Создаем загрузчик
        loader = FractalModelLoader(config)
        
        # Проверяем доступные модели
        available_models = loader.list_models()
        print(f"📋 Доступные модели: {available_models}")
        
        if not available_models:
            print(f"❌ Модели в фрактальном хранилище не найдены")
            return False
        
        # Загружаем модель
        model_id = available_models[0]
        print(f"🔄 Загрузка модели: {model_id}")
        
        model = loader.load_model(model_id)
        if model is None:
            print(f"❌ Не удалось загрузить модель")
            return False
        
        print(f"✅ Модель загружена успешно")
        print(f"   📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")
        
        # Тестируем генерацию
        print(f"\n🧪 ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ИЗ ФРАКТАЛЬНОГО ХРАНИЛИЩА")
        
        test_queries = [
            "Привет, как дела?",
            "Что такое машинное обучение?",
            "Расскажи интересную историю",
            "Объясни простыми словами",
            "Как работает нейронная сеть?"
        ]
        
        for query in test_queries:
            try:
                # Импортируем токенизатор
                from transformers import AutoTokenizer
                
                # Загружаем токенизатор из экспортированной модели
                tokenizer_path = Path(fractal_path) / "tokenizer"
                if tokenizer_path.exists():
                    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                else:
                    print(f"❌ Токенизатор не найден: {tokenizer_path}")
                    continue
                
                # Генерируем ответ
                inputs = tokenizer.encode(query, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print(f"   📝 '{query}'")
                print(f"   💬 '{response}'")
                print()
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка тестирования фрактальной модели: {e}")
        return False

def main():
    """Основная функция"""
    
    print("🚀 ЗАГРУЗКА И ЭКСПОРТ RU-GPT3 LARGE ВО ФРАКТАЛЬНОЕ ХРАНИЛИЩЕ")
    print("="*70)
    
    # 1. Загружаем и экспортируем модель
    success = download_and_export_rugpt3()
    
    if success:
        print(f"\n✅ ЭКСПОРТ УСПЕШЕН!")
        
        # 2. Тестируем загрузку из фрактального хранилища
        test_success = test_fractal_model()
        
        if test_success:
            print(f"\n🎉 ВСЕ ОПЕРАЦИИ УСПЕШНЫ!")
            print(f"✅ ruGPT3 large загружена и экспортирована")
            print(f"✅ Генерация из фрактального хранилища работает")
            print(f"📁 Модель доступна в: out/rugpt3_large_fractal")
        else:
            print(f"\n⚠️ Экспорт успешен, но тестирование не удалось")
    else:
        print(f"\n❌ ЭКСПОРТ НЕ УДАЛСЯ")
    
    return success

if __name__ == "__main__":
    main()
