"""
Анализ и исправление ошибок OptimizedFractalModelManager
"""
import sys
import os
import torch
import logging
from pathlib import Path
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_manager")

def analyze_errors():
    """Анализирует ошибки в OptimizedFractalModelManager"""
    logger.info("🔍 АНАЛИЗ ОШИБОК OptimizedFractalModelManager")
    logger.info("="*60)
    
    try:
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        # 1. Проверяем инициализацию
        logger.info("1️⃣ ПРОВЕРКА ИНИЦИАЛИЗАЦИИ")
        
        manager = OptimizedFractalModelManager()
        
        logger.info(f"   ✅ Менеджер создан")
        logger.info(f"   🔧 Устройство: {getattr(manager, 'device', 'N/A')}")
        logger.info(f"   💾 Память токенов: {getattr(manager, 'max_memory_tokens', 'N/A')}")
        
        # 2. Проверяем токенизатор
        logger.info("\n2️⃣ ПРОВЕРКА ТОКЕНИЗАТОРА")
        
        if hasattr(manager, 'tokenizer') and manager.tokenizer:
            logger.info(f"   ✅ Токенизатор загружен")
            logger.info(f"   📚 Vocab size: {len(manager.tokenizer.get_vocab()):,}")
            
            # Тестируем токенизацию
            test_text = "Привет"
            try:
                # Проверяем, где создаются тензоры
                inputs = manager.tokenizer(test_text, return_tensors='pt')
                
                for key, tensor in inputs.items():
                    logger.info(f"   📊 {key}: device={tensor.device}, shape={tensor.shape}")
                
                # Проверяем метод optimized_tokenize
                if hasattr(manager, 'optimized_tokenize'):
                    logger.info("   🔧 Тестируем optimized_tokenize...")
                    tokenized = manager.optimized_tokenize([test_text])
                    
                    for i, result in enumerate(tokenized):
                        input_ids = result.get('input_ids')
                        attention_mask = result.get('attention_mask')
                        
                        if input_ids is not None:
                            logger.info(f"      📊 Результат {i}: input_ids.device={input_ids.device}")
                        if attention_mask is not None:
                            logger.info(f"      📊 Результат {i}: attention_mask.device={attention_mask.device}")
                
            except Exception as e:
                logger.error(f"   ❌ Ошибка токенизации: {e}")
        
        # 3. Проверяем модель
        logger.info("\n3️⃣ ПРОВЕРКА МОДЕЛИ")
        
        if hasattr(manager, 'model') and manager.model is not None:
            logger.info(f"   ✅ Модель загружена")
            logger.info(f"   📊 Параметров: {sum(p.numel() for p in manager.model.parameters()):,}")
            logger.info(f"   🔧 Устройство модели: {next(manager.model.parameters()).device}")
            
            # Проверяем соответствие устройств
            model_device = next(manager.model.parameters()).device
            manager_device = getattr(manager, 'device', 'cpu')
            
            logger.info(f"   📊 Устройство менеджера: {manager_device}")
            logger.info(f"   📊 Устройство модели: {model_device}")
            logger.info(f"   ✅ Соответствие: {model_device == manager_device}")
        
        # 4. Тестируем генерацию с отладкой
        logger.info("\n4️⃣ ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ С ОТЛАДКОЙ")
        
        if hasattr(manager, 'generate_text'):
            try:
                # Отключаем CUDA для теста
                logger.info("   🔧 Тестируем на CPU...")
                
                # Создаем менеджер с CPU
                cpu_manager = OptimizedFractalModelManager()
                
                # Принудительно устанавливаем CPU
                if hasattr(cpu_manager, 'device'):
                    cpu_manager.device = torch.device('cpu')
                    logger.info(f"   🔧 Установлено устройство: {cpu_manager.device}")
                
                # Перезагружаем модель на CPU
                if hasattr(cpu_manager, 'model') and cpu_manager.model is not None:
                    cpu_manager.model = cpu_manager.model.to('cpu')
                    logger.info(f"   🔧 Модель перенесена на CPU")
                
                # Тестируем генерацию
                response = cpu_manager.generate_text('Привет', max_length=30)
                logger.info(f"   ✅ Генерация на CPU: '{response}'")
                
            except Exception as e:
                logger.error(f"   ❌ Ошибка генерации: {e}")
                import traceback
                logger.error(f"   📋 Traceback: {traceback.format_exc()}")
        
        # 5. Анализируем конкретные места ошибок
        logger.info("\n5️⃣ АНАЛИЗ КОНКРЕТНЫХ МЕСТ ОШИБОК")
        
        # Проверяем строку 533 в generate_response_optimized
        import inspect
        source_lines = inspect.getsourcelines(manager.generate_response_optimized)
        
        if source_lines:
            for i, line in enumerate(source_lines[530:540], start=531):
                logger.info(f"   📝 Строка {i}: {line.strip()}")
        
        # 6. Рекомендации по исправлению
        logger.info("\n6️⃣ РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ")
        
        recommendations = [
            "1. Убедиться, что все тензоры создаются на правильном устройстве",
            "2. Добавить проверку device в методе optimized_tokenize",
            "3. Исправить перенос тензоров в generate_response_optimized",
            "4. Добавить fallback на CPU при ошибках устройства",
            "5. Улучшить обработку ошибок токенизации"
        ]
        
        for rec in recommendations:
            logger.info(f"   💡 {rec}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка анализа: {e}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return False

def create_fixed_version():
    """Создает исправленную версию менеджера"""
    logger.info("\n🔧 СОЗДАНИЕ ИСПРАВЛЕННОЙ ВЕРСИИ")
    logger.info("="*50)
    
    try:
        # Читаем текущий код
        manager_file = Path("cogniflex/mlearning/optimized_fractal_model_manager.py")
        
        if not manager_file.exists():
            logger.error("❌ Файл менеджера не найден")
            return False
        
        with open(manager_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Создаем исправленную версию
        fixed_content = content.replace(
            "inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)",
            "inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)"
        )
        
        # Сохраняем исправленную версию
        fixed_file = Path("cogniflex/mlearning/optimized_fractal_model_manager_fixed.py")
        
        with open(fixed_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        logger.info(f"✅ Исправленная версия сохранена: {fixed_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания исправленной версии: {e}")
        return False

def main():
    """Основная функция"""
    success = analyze_errors()
    
    if success:
        logger.info("\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
        logger.info("✅ Проблемы идентифицированы")
        logger.info("✅ Рекомендации подготовлены")
        
        # Создаем исправленную версию
        create_fixed_version()
    else:
        logger.error("\n❌ АНАЛИЗ НЕ УДАЛСЯ")
    
    return success

if __name__ == "__main__":
    main()
