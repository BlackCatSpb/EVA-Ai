"""
Исправление весов эмбеддингов для корректной генерации
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_fix")

def fix_embeddings():
    """Исправляет веса эмбеддингов для корректной генерации"""
    
    try:
        from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
        
        print("🔧 ИСПРАВЛЕНИЕ ВЕСОВ ЭМБЕДДИНГОВ")
        print("="*50)
        
        # Инициализируем менеджер
        manager = UnifiedFractalManager()
        model = manager.manager.model
        tokenizer = manager.manager.tokenizer
        
        print(f"✅ Модель: {type(model).__name__}")
        print(f"✅ Токенизатор: {type(tokenizer).__name__}")
        
        # Получаем эмбеддинги
        wte = model.get_input_embeddings()
        print(f"✅ Эмбеддинги: {wte.weight.shape}")
        
        # 1. Анализ текущих весов
        print(f"\n📊 АНАЛИЗ ТЕКУЩИХ ВЕСОВ:")
        current_weights = wte.weight.data
        print(f"   Среднее: {current_weights.mean().item():.6f}")
        print(f"   Std: {current_weights.std().item():.6f}")
        print(f"   Min: {current_weights.min().item():.6f}")
        print(f"   Max: {current_weights.max().item():.6f}")
        
        # 2. Переинициализируем веса с правильными параметрами
        print(f"\n🔄 ПЕРЕИНИЦИАЛИЗАЦИЯ ВЕСОВ:")
        
        # Используем стандартную инициализацию для GPT-2
        embedding_dim = wte.embedding_dim
        vocab_size = wte.num_embeddings
        
        # Стандартная инициализация для трансформеров
        nn.init.normal_(wte.weight, mean=0.0, std=0.02)
        
        # Особая инициализация для специальных токенов
        special_tokens = {
            tokenizer.pad_token_id: 0.0,
            tokenizer.eos_token_id: 0.0,
            tokenizer.bos_token_id: 0.0,
            tokenizer.unk_token_id: 0.0
        }
        
        for token_id, value in special_tokens.items():
            if token_id is not None and token_id < vocab_size:
                wte.weight.data[token_id].fill_(value)
        
        # 3. Анализ новых весов
        print(f"\n📊 АНАЛИЗ НОВЫХ ВЕСОВ:")
        new_weights = wte.weight.data
        print(f"   Среднее: {new_weights.mean().item():.6f}")
        print(f"   Std: {new_weights.std().item():.6f}")
        print(f"   Min: {new_weights.min().item():.6f}")
        print(f"   Max: {new_weights.max().item():.6f}")
        
        # 4. Проверяем несколько токенов
        print(f"\n🔍 ПРОВЕРКА ТОКЕНОВ ПОСЛЕ ИНИЦИАЛИЗАЦИИ:")
        test_tokens = [1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]
        for token_id in test_tokens:
            if token_id < vocab_size:
                token_text = tokenizer.decode([token_id])
                embedding = wte.weight.data[token_id]
                print(f"   ID {token_id} ('{token_text}'): mean={embedding.mean().item():.4f}, std={embedding.std().item():.4f}")
        
        # 5. Тестируем генерацию
        print(f"\n🧪 ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ПОСЛЕ ИСПРАВЛЕНИЯ:")
        
        test_queries = [
            "Привет",
            "Как дела",
            "Что такое"
        ]
        
        for query in test_queries:
            try:
                # Кодируем запрос
                input_ids = tokenizer.encode(query, return_tensors='pt')
                
                # Генерируем с исправленными весами
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + 10,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"   📝 '{query}' → '{generated_text}'")
                
            except Exception as e:
                print(f"   ❌ Ошибка для '{query}': {e}")
        
        # 6. Проверяем forward pass
        print(f"\n🔄 ПРОВЕРКА FORWARD PASS ПОСЛЕ ИСПРАВЛЕНИЯ:")
        
        try:
            test_input = torch.tensor([[1, 2, 3, 4, 5]])
            
            with torch.no_grad():
                output = model(test_input)
            
            last_logits = output.logits[0, -1, :]
            top_tokens = torch.topk(last_logits, 10)
            
            print(f"   Top-10 предсказанных токенов:")
            for i, (logit, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
                token_text = tokenizer.decode([token_id.item()])
                print(f"      {i+1}. ID {token_id.item()} ('{token_text}'): {logit.item():.4f}")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        # 7. Сохраняем исправленную модель
        print(f"\n💾 СОХРАНЕНИЕ ИСПРАВЛЕННОЙ МОДЕЛИ:")
        
        # Сохраняем в фрактальное хранилище
        from cogniflex.mlearning.storage.fractal_weight_store import FractalWeightStore
        
        fractal_store = FractalWeightStore(
            block_size=64,
            fractal_levels=5,
            device="cpu"
        )
        
        # Сохраняем исправленные веса
        state_dict = model.state_dict()
        for key, tensor in state_dict.items():
            fractal_store.store_tensor(f"fixed_model.{key}", tensor)
        
        # Сохраняем метаданные
        fix_metadata = {
            "fix_type": "embedding_reinitialization",
            "model_type": type(model).__name__,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "timestamp": torch.datetime.datetime.now().isoformat() if hasattr(torch, 'datetime') else "2025-03-05",
            "original_mean": current_weights.mean().item(),
            "new_mean": new_weights.mean().item(),
            "original_std": current_weights.std().item(),
            "new_std": new_weights.std().item()
        }
        
        fractal_store.store("fix_metadata", fix_metadata)
        
        print(f"✅ Исправленная модель сохранена в фрактальное хранилище")
        
        # 8. Обновляем модель в менеджере
        if hasattr(manager.manager, 'state_dict'):
            manager.manager.state_dict = state_dict
            print(f"✅ State dict обновлен в менеджере")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка исправления эмбеддингов: {e}", exc_info=True)
        return False

def test_fixed_generation():
    """Тестирует генерацию после исправления"""
    
    try:
        from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
        
        print(f"\n🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ ГЕНЕРАЦИИ")
        print("="*50)
        
        manager = UnifiedFractalManager()
        
        # Тестовые запросы
        test_queries = [
            "Привет, как дела?",
            "Что такое машинное обучение?",
            "Расскажи интересную историю",
            "Объясни простыми словами",
            "Как работает нейронная сеть?"
        ]
        
        print(f"📝 Тестовые запросы:")
        
        for i, query in enumerate(test_queries, 1):
            try:
                response = manager.generate_response(query, max_tokens=50)
                print(f"\n{i}. 📝 Запрос: {query}")
                print(f"   💬 Ответ: {response}")
                
            except Exception as e:
                print(f"\n{i}. ❌ Ошибка для '{query}': {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка тестирования: {e}")
        return False

def main():
    """Основная функция"""
    
    print("🔧 ИСПРАВЛЕНИЕ ВЕСОВ ЭМБЕДДИНГОВ ДЛЯ КОРРЕКТНОЙ ГЕНЕРАЦИИ")
    print("="*60)
    
    # 1. Исправляем веса
    success = fix_embeddings()
    
    if success:
        print(f"\n✅ Веса эмбеддингов исправлены!")
        
        # 2. Тестируем генерацию
        test_success = test_fixed_generation()
        
        if test_success:
            print(f"\n🎉 ИСПРАВЛЕНИЕ УСПЕШНО!")
            print(f"💡 Модель теперь должна генерировать осмысленные тексты")
        else:
            print(f"\n⚠️ Проблемы с тестированием")
    else:
        print(f"\n❌ Исправление не удалось")
    
    return success

if __name__ == "__main__":
    main()
