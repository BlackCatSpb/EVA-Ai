"""
Минимальный тест токенизации для поиска проблемы
"""
import sys
import torch
import logging
sys.path.append('.')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_tokenization")

def test_tokenization():
    """Тестируем токенизацию пошагово"""
    logger.info("🧪 ТЕСТИРОВАНИЕ ТОКЕНИЗАЦИИ")
    logger.info("="*50)
    
    try:
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        # Создаем менеджер
        manager = OptimizedFractalModelManager()
        
        logger.info(f"📊 Устройство менеджера: {manager.device}")
        logger.info(f"📊 Устройство модели: {next(manager.model.parameters()).device}")
        
        # Тестируем токенизацию напрямую
        test_text = "Привет"
        
        logger.info("1️⃣ ТЕСТ ПРЯМОЙ ТОКЕНИЗАЦИИ:")
        
        # 1. Токенизатор напрямую
        direct_inputs = manager.tokenizer(test_text, return_tensors='pt')
        logger.info(f"   📊 Прямая токенизация:")
        logger.info(f"      input_ids.device: {direct_inputs['input_ids'].device}")
        logger.info(f"      attention_mask.device: {direct_inputs['attention_mask'].device}")
        
        # 2. Через optimized_tokenize
        logger.info("2️⃣ ТЕСТ ЧЕРЕЗ OPTIMIZED_TOKENIZE:")
        optimized_results = manager.optimized_tokenize([test_text])
        if optimized_results:
            result = optimized_results[0]
            logger.info(f"   📊 Optimized tokenize:")
            logger.info(f"      input_ids.device: {result['input_ids'].device}")
            logger.info(f"      attention_mask.device: {result['attention_mask'].device}")
            
            # 3. Проверяем перемещение
            logger.info("3️⃣ ПРОВЕРКА ПЕРЕМЕЩЕНИЯ:")
            if result['input_ids'].device != manager.device:
                logger.error(f"   ❌ input_ids на неверном устройстве!")
                moved_input = result['input_ids'].to(manager.device)
                logger.info(f"   ✅ После перемещения: {moved_input.device}")
            else:
                logger.info(f"   ✅ input_ids уже на правильном устройстве")
            
            if result['attention_mask'].device != manager.device:
                logger.error(f"   ❌ attention_mask на неверном устройстве!")
                moved_mask = result['attention_mask'].to(manager.device)
                logger.info(f"   ✅ После перемещения: {moved_mask.device}")
            else:
                logger.info(f"   ✅ attention_mask уже на правильном устройстве")
        
        # 4. Тест генерации с исправленными тензорами
        logger.info("4️⃣ ТЕСТ ГЕНЕРАЦИИ:")
        
        if optimized_results:
            result = optimized_results[0]
            
            # Принудительно перемещаем
            input_ids = result['input_ids'].to(manager.device)
            attention_mask = result['attention_mask'].to(manager.device)
            
            logger.info(f"   📊 Перед генерацией:")
            logger.info(f"      input_ids.device: {input_ids.device}")
            logger.info(f"      attention_mask.device: {attention_mask.device}")
            logger.info(f"      model.device: {next(manager.model.parameters()).device}")
            
            try:
                with torch.no_grad():
                    output = manager.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=20,
                        do_sample=False,
                        pad_token_id=manager.tokenizer.eos_token_id
                    )
                
                logger.info(f"   ✅ Генерация успешна!")
                response = manager.tokenizer.decode(output[0], skip_special_tokens=True)
                logger.info(f"   💬 Ответ: '{response}'")
                
            except Exception as e:
                logger.error(f"   ❌ Ошибка генерации: {e}")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_tokenization()
