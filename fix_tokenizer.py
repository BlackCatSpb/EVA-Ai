"""
Исправление токенизатора для корректной генерации текста
"""
import sys
import torch
import logging
from pathlib import Path
sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tokenizer_fix")

def test_tokenizer():
    """Тестируем и исправляем токенизатор"""
    logger.info("🔧 ТЕСТИРОВАНИЕ И ИСПРАВЛЕНИЕ ТОКЕНИЗАТОРА")
    logger.info("="*60)
    
    try:
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        # Создаем менеджер
        manager = OptimizedFractalModelManager()
        
        if not hasattr(manager, 'tokenizer') or manager.tokenizer is None:
            logger.error("❌ Токенизатор не доступен")
            return False
        
        logger.info(f"✅ Токенизатор загружен: {type(manager.tokenizer).__name__}")
        logger.info(f"📊 Vocab size: {len(manager.tokenizer.get_vocab()):,}")
        
        # Тестируем токенизацию
        test_texts = [
            "Привет",
            "Как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о России"
        ]
        
        logger.info("🧪 ТЕСТИРОВАНИЕ ТОКЕНИЗАЦИИ:")
        
        for i, text in enumerate(test_texts, 1):
            try:
                logger.info(f"{i}. 📝 '{text}'")
                
                # Кодируем
                inputs = manager.tokenizer(text, return_tensors='pt')
                input_ids = inputs['input_ids']
                
                logger.info(f"   📊 Input IDs: {input_ids.shape} -> {input_ids.flatten()[:10].tolist()}...")
                
                # Декодируем обратно
                decoded = manager.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                logger.info(f"   💬 Декодировано: '{decoded}'")
                
                # Проверяем на проблемы
                if text != decoded.strip():
                    logger.warning(f"   ⚠️ Несоответствие: '{text}' != '{decoded.strip()}'")
                else:
                    logger.info(f"   ✅ Совпадение: OK")
                
                # Проверяем специальные токены
                special_tokens = manager.tokenizer.all_special_tokens
                logger.info(f"   🎯 Специальные токены: {special_tokens}")
                
                # Проверяем параметры
                logger.info(f"   📋 Pad token: {manager.tokenizer.pad_token}")
                logger.info(f"   📋 EOS token: {manager.tokenizer.eos_token}")
                logger.info(f"   📋 BOS token: {manager.tokenizer.bos_token}")
                logger.info(f"   📋 UNK token: {manager.tokenizer.unk_token}")
                
            except Exception as e:
                logger.error(f"   ❌ Ошибка: {e}")
        
        # Тестируем генерацию с разными параметрами
        logger.info("\n🧪 ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ:")
        
        test_query = "Привет, как дела?"
        
        # 1. Без специальных токенов
        try:
            inputs = manager.tokenizer(test_query, return_tensors='pt')
            with torch.no_grad():
                outputs = manager.model.generate(
                    inputs['input_ids'].to(manager.device),
                    max_length=30,
                    do_sample=False,
                    pad_token_id=manager.tokenizer.eos_token_id
                )
            
            decoded = manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"1️⃣ Без сэмплинга: '{decoded}'")
            
        except Exception as e:
            logger.error(f"   ❌ Ошибка: {e}")
        
        # 2. С правильными параметрами
        try:
            inputs = manager.tokenizer(test_query, return_tensors='pt')
            with torch.no_grad():
                outputs = manager.model.generate(
                    inputs['input_ids'].to(manager.device),
                    max_length=30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=manager.tokenizer.eos_token_id,
                    eos_token_id=manager.tokenizer.eos_token_id
                )
            
            decoded = manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"2️⃣ С сэмплингом: '{decoded}'")
            
        except Exception as e:
            logger.error(f"   ❌ Ошибка: {e}")
        
        # 3. С очисткой
        try:
            inputs = manager.tokenizer(test_query, return_tensors='pt')
            with torch.no_grad():
                outputs = manager.model.generate(
                    inputs['input_ids'].to(manager.device),
                    max_length=30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=manager.tokenizer.eos_token_id,
                    eos_token_id=manager.tokenizer.eos_token_id,
                    clean_up_tokenization_spaces=True
                )
            
            decoded = manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"3️⃣ С очисткой: '{decoded}'")
            
        except Exception as e:
            logger.error(f"   ❌ Ошибка: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def fix_tokenizer():
    """Исправляет токенизатор"""
    logger.info("🔧 ИСПРАВЛЕНИЕ ТОКЕНИЗАТОРА")
    
    try:
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        manager = OptimizedFractalModelManager()
        
        if hasattr(manager, 'tokenizer') and manager.tokenizer:
            # Проверяем и устанавливаем специальные токены
            if manager.tokenizer.pad_token is None:
                manager.tokenizer.pad_token = manager.tokenizer.eos_token
                logger.info("✅ Pad token установлен")
            
            if manager.tokenizer.bos_token is None:
                manager.tokenizer.bos_token = "<|startoftext|>"
                logger.info("✅ BOS token установлен")
            
            # Проверяем vocab_size
            model_device = next(manager.model.parameters()).device
            model_vocab_size = manager.model.config.vocab_size if hasattr(manager.model, 'config') else 50257
            
            if len(manager.tokenizer) != model_vocab_size:
                logger.warning(f"Размер словаря токенизатора: {len(manager.tokenizer)}, модели: {model_vocab_size}")
                
                # Добавляем недостающие токены
                for i in range(len(manager.tokenizer), model_vocab_size):
                    manager.tokenizer.add_tokens([f"<extra_token_{i}>"])
                
                logger.info(f"✅ Добавлено {model_vocab_size - len(manager.tokenizer)} токенов")
            
            logger.info("✅ Токенизатор исправлен")
            return True
            
        else:
            logger.error("❌ Токенизатор недоступен")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка исправления токенизатора: {e}")
        return False

def main():
    """Основная функция"""
    logger.info("🚀 ЗАПУСК ИСПРАВЛЕНИЯ ТОКЕНИЗАТОРА")
    
    # Сначала тестируем
    if test_tokenizer():
        logger.info("\n🔧 НАЧИНАЕМ ИСПРАВЛЕНИЕ...")
        
        if fix_tokenizer():
            logger.info("\n✅ ТОКЕНИЗАТОР УСПЕШНО ИСПРАВЛЕН")
            
            # Финальный тест
            logger.info("\n🧪 ФИНАЛЬНЫЙ ТЕСТ ПОСЛЕ ИСПРАВЛЕНИЯ:")
            
            from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
            
            manager = OptimizedFractalModelManager()
            response = manager.generate_text("Привет, как дела?", max_length=50)
            
            logger.info(f"💬 Ответ: '{response}'")
            
            if response and not response.startswith("---"):
                logger.info("🎉 ГЕНЕРАЦИЯ РАБОТАЕТ КОРРЕКТНО!")
            else:
                logger.warning("⚠️ Проблема с генерацией все ещё присутствует")
            
            return True
        else:
            logger.error("❌ ИСПРАВЛЕНИЕ ТОКЕНИЗАТОРА НЕ УДАЛОСЬ")
            return False
    else:
        logger.error("❌ ТЕСТИРОВАНИЕ НЕ УДАЛОСЬ")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
