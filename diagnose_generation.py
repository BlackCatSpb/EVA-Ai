"""
Диагностика проблемы генерации текста
"""
import sys
import os
import torch
import logging
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generation_diagnostic")

def diagnose_generation():
    """Диагностирует проблему генерации текста"""
    
    try:
        from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
        
        print("🔍 ДИАГНОСТИКА ПРОБЛЕМЫ ГЕНЕРАЦИИ ТЕКСТА")
        print("="*60)
        
        # Инициализируем менеджер
        manager = UnifiedFractalManager()
        print(f"✅ Менеджер: {type(manager.manager).__name__}")
        
        # Проверяем модель
        model = manager.manager.model
        tokenizer = manager.manager.tokenizer
        
        print(f"✅ Модель: {type(model).__name__}")
        print(f"✅ Токенизатор: {type(tokenizer).__name__}")
        
        # 1. Проверяем словарь токенизатора
        print(f"\n📊 СЛОВАРЬ ТОКЕНИЗАТОРА:")
        print(f"   Размер: {len(tokenizer.get_vocab())}")
        print(f"   Специальные токены: {tokenizer.all_special_tokens}")
        
        # 2. Проверяем эмбеддинги
        print(f"\n🔤 ЭМБЕДДИНГИ МОДЕЛИ:")
        wte = model.get_input_embeddings()
        print(f"   Размер эмбеддингов: {wte.weight.shape}")
        print(f"   Тип данных: {wte.weight.dtype}")
        print(f"   Устройство: {wte.weight.device}")
        
        # 3. Проверяем несколько токенов
        test_tokens = [1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]
        print(f"\n🔍 ПРОВЕРКА ТОКЕНОВ:")
        for token_id in test_tokens:
            if token_id < len(tokenizer.get_vocab()):
                token_text = tokenizer.decode([token_id])
                print(f"   ID {token_id}: '{token_text}'")
        
        # 4. Проверяем кодировку/декодировку
        test_text = "Привет мир"
        print(f"\n🔄 ТЕСТ КОДИРОВКИ/ДЕКОДИРОВКИ:")
        print(f"   Исходный текст: '{test_text}'")
        
        # Кодируем
        encoded = tokenizer.encode(test_text, return_tensors='pt')
        print(f"   Закодировано: {encoded.tolist()}")
        
        # Декодируем
        decoded = tokenizer.decode(encoded[0])
        print(f"   Декодировано: '{decoded}'")
        
        # 5. Проверяем генерацию с разными параметрами
        print(f"\n🎓 ТЕСТ ГЕНЕРАЦИИ С РАЗНЫМИ ПАРАМЕТРАМИ:")
        
        test_params = [
            {"max_tokens": 10, "temperature": 0.1, "do_sample": False},
            {"max_tokens": 10, "temperature": 0.7, "do_sample": True},
            {"max_tokens": 10, "temperature": 1.0, "do_sample": True},
            {"max_tokens": 20, "temperature": 0.1, "do_sample": False, "repetition_penalty": 1.0},
        ]
        
        for i, params in enumerate(test_params):
            print(f"\n   Параметры {i+1}: {params}")
            
            try:
                # Генерируем с токенами
                input_ids = tokenizer.encode("Привет", return_tensors='pt')
                
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + params["max_tokens"],
                        temperature=params.get("temperature", 0.7),
                        do_sample=params.get("do_sample", True),
                        repetition_penalty=params.get("repetition_penalty", 1.0),
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"   Результат: '{generated_text}'")
                
            except Exception as e:
                print(f"   Ошибка: {e}")
        
        # 6. Проверяем веса модели
        print(f"\n📊 ПРОВЕРКА ВЕСОВ МОДЕЛИ:")
        
        # Проверяем первые несколько весов
        wte_weight = wte.weight.data
        print(f"   Минимальное значение: {wte_weight.min().item():.6f}")
        print(f"   Максимальное значение: {wte_weight.max().item():.6f}")
        print(f"   Среднее значение: {wte_weight.mean().item():.6f}")
        print(f"   Стандартное отклонение: {wte_weight.std().item():.6f}")
        
        # Проверяем первые 10 токенов в эмбеддингах
        print(f"\n🔤 ПЕРВЫЕ 10 ТОКЕНОВ В ЭМБЕДДИНГАХ:")
        for i in range(min(10, wte_weight.shape[0])):
            embedding = wte_weight[i]
            token_text = tokenizer.decode([i])
            print(f"   ID {i} ('{token_text}'): min={embedding.min().item():.4f}, max={embedding.max().item():.4f}")
        
        # 7. Проверяем forward pass
        print(f"\n🔄 ПРОВЕРКА FORWARD PASS:")
        
        try:
            # Создаем тестовый вход
            test_input = torch.tensor([[1, 2, 3, 4, 5]])
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"   Форма logits: {output.logits.shape}")
            print(f"   Тип logits: {output.logits.dtype}")
            
            # Проверяем предсказания
            last_logits = output.logits[0, -1, :]
            predicted_token = torch.argmax(last_logits).item()
            predicted_text = tokenizer.decode([predicted_token])
            
            print(f"   Предсказанный токен: {predicted_token} ('{predicted_text}')")
            print(f"   Top-5 токенов:")
            
            top5 = torch.topk(last_logits, 5)
            for i, (logit, token_id) in enumerate(zip(top5.values, top5.indices)):
                token_text = tokenizer.decode([token_id.item()])
                print(f"      {i+1}. ID {token_id.item()} ('{token_text}'): {logit.item():.4f}")
            
        except Exception as e:
            print(f"   Ошибка forward pass: {e}")
        
        # 8. Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        
        # Анализируем проблемы
        issues = []
        
        # Проверяем словарь
        if len(tokenizer.get_vocab()) < 1000:
            issues.append("Слишком маленький словарь токенизатора")
        
        # Проверяем веса
        if abs(wte_weight.mean().item()) < 0.001:
            issues.append("Веса эмбеддингов близки к нулю")
        
        # Проверяем стандартное отклонение
        if wte_weight.std().item() < 0.01:
            issues.append("Веса эмбеддингов имеют низкую вариативность")
        
        if issues:
            print(f"   ❌ Обнаруженные проблемы:")
            for issue in issues:
                print(f"      • {issue}")
        else:
            print(f"   ✅ Критических проблем не обнаружено")
        
        print(f"\n🔧 ВОЗМОЖНЫЕ РЕШЕНИЯ:")
        print(f"   1. Переинициализировать веса эмбеддингов")
        print(f"   2. Обновить токенизатор с поддержкой русского языка")
        print(f"   3. Дообучить модель на русских текстах")
        print(f"   4. Проверить параметры генерации")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка диагностики: {e}", exc_info=True)
        return False

def main():
    """Основная функция"""
    
    success = diagnose_generation()
    
    if success:
        print(f"\n🎉 ДИАГНОСТИКА ЗАВЕРШЕНА")
        print(f"📋 Проверьте рекомендации выше для исправления генерации")
    else:
        print(f"\n❌ Ошибка диагностики")
    
    return success

if __name__ == "__main__":
    main()
