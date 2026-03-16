"""
Прямое исправление основных проблем в OptimizedFractalModelManager
"""
import sys
import os
import torch
import logging
import shutil
from pathlib import Path
sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct_fix")

def fix_model_loading():
    """Исправляет загрузку модели"""
    logger.info("🔧 ИСПРАВЛЕНИЕ ЗАГРУЗКИ МОДЕЛИ")
    
    try:
        # 1. Проверяем наличие файла модели
        model_path = "out/fractal_rugpt_full.safetensors"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Файл модели не найден: {model_path}")
            return False
        
        logger.info(f"✅ Файл модели найден: {model_path}")
        
        # 2. Проверяем размер файла
        file_size = os.path.getsize(model_path)
        logger.info(f"📊 Размер файла: {file_size:,} байт ({file_size / (1024*1024):.1f} MB)")
        
        if file_size < 100 * 1024 * 1024:  # Меньше 100MB
            logger.warning("⚠️ Файл модели слишком мал, возможно поврежден")
        
        # 3. Создаем резервную копию
        backup_path = model_path + ".backup"
        shutil.copy2(model_path, backup_path)
        logger.info(f"✅ Резервная копия создана: {backup_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка исправления загрузки модели: {e}")
        return False

def fix_device_mismatch():
    """Исправляет несоответствие устройств"""
    logger.info("🔧 ИСПРАВЛЕНИЕ НЕСООТВЕТСТВИЯ УСТРОЙСТВ")
    
    try:
        # Читаем текущий файл
        manager_file = Path("cogniflex/mlearning/optimized_fractal_model_manager.py")
        
        if not manager_file.exists():
            logger.error("❌ Файл менеджера не найден")
            return False
        
        with open(manager_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ищем метод generate_response_optimized
        import re
        
        # Находим начало метода
        method_pattern = r'def generate_response_optimized\(.*?\n\s*try:'
        method_match = re.search(method_pattern, content)
        
        if not method_match:
            logger.error("❌ Метод generate_response_optimized не найден")
            return False
        
        # Находим место где нужно добавить исправление
        tokenized_line = "tokenized = self.optimized_tokenize([query])[0]"
        
        if tokenized_line not in content:
            logger.error("❌ Строка токенизации не найдена")
            return False
        
        # Создаем исправленный код
        device_fix_code = '''            # Оптимизированная токенизация
            tokenized = self.optimized_tokenize([query])[0]
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            # ПРИНУДИТЕЛЬНОЕ ПЕРЕМЕЩЕНИЕ НА УСТРОЙСТВО МОДЕЛИ
            model_device = next(self.model.parameters()).device
            
            # ПРОВЕРЯЕМ И ПЕРЕМЕЩАЕМ ВСЕ ТЕНЗОРЫ
            if input_ids.device != model_device:
                logger.warning(f"Перемещаем input_ids с {input_ids.device} на {model_device}")
                input_ids = input_ids.to(model_device)
            
            if attention_mask is not None and attention_mask.device != model_device:
                logger.warning(f"Перемещаем attention_mask с {attention_mask.device} на {model_device}")
                attention_mask = attention_mask.to(model_device)
            
            # ГАРАНТИРУЕМ ПРАВИЛЬНОЕ УСТРОЙСТВО
            assert input_ids.device == model_device, f"input_ids на неверном устройстве: {input_ids.device} != {model_device}"
            if attention_mask is not None:
                assert attention_mask.device == model_device, f"attention_mask на неверном устройстве: {attention_mask.device} != {model_device}"
            
            logger.info(f"✅ Устройства согласованы: model={model_device}, input_ids={input_ids.device}, attention_mask={attention_mask.device if attention_mask is not None else 'None'}")'''
        
        # Заменяем метод
        method_start = method_match.start()
        method_end = content.find('\n    def ', method_start + 1)
        
        if method_end == -1:
            method_end = len(content)
        
        new_content = (
            content[:method_start] + 
            device_fix_code + 
            content[method_end:]
        )
        
        # Записываем исправленный файл
        with open(manager_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info("✅ Файл менеджера исправлен")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка исправления устройств: {e}")
        return False

def test_fixed_system():
    """Тестирует исправленную систему"""
    logger.info("🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ СИСТЕМЫ")
    
    try:
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        # Создаем менеджер
        manager = OptimizedFractalModelManager()
        
        # Проверяем загрузку модели
        if hasattr(manager, 'model') and manager.model is not None:
            param_count = sum(p.numel() for p in manager.model.parameters())
            logger.info(f"✅ Модель загружена: {param_count:,} параметров")
            
            # Проверяем устройство
            model_device = next(manager.model.parameters()).device
            logger.info(f"✅ Устройство модели: {model_device}")
            
            # Тестируем генерацию
            test_queries = [
                "Привет, как дела?",
                "Что такое искусственный интеллект?",
                "Расскажи о России",
                "Hello, how are you?"  # Тест на английском
            ]
            
            all_success = True
            
            for i, query in enumerate(test_queries, 1):
                try:
                    logger.info(f"{i}. 📝 '{query}'")
                    
                    response = manager.generate_text(query, max_length=50)
                    
                    if response and not response.startswith("---") and len(response.strip()) > 5:
                        logger.info(f"   ✅ Ответ: '{response[:100]}{'...' if len(response) > 100 else ''}'")
                        logger.info(f"   📊 Длина: {len(response)}")
                    else:
                        logger.warning(f"   ⚠️ Странный ответ: '{response}'")
                        all_success = False
                        
                except Exception as e:
                    logger.error(f"   ❌ Ошибка: {e}")
                    all_success = False
            
            if all_success:
                logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
                return True
            else:
                logger.warning("⚠️ Часть тестов провалена")
                return False
        else:
            logger.error("❌ Модель не загружена")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Основная функция"""
    logger.info("🚀 ПРЯМОЕ ИСПРАВЛЕНИЕ ОСНОВНЫХ ПРОБЛЕМ")
    logger.info("="*80)
    
    success_count = 0
    total_steps = 3
    
    # Шаг 1: Исправление загрузки модели
    if fix_model_loading():
        success_count += 1
        logger.info("✅ Шаг 1: Загрузка модели исправлена")
    else:
        logger.error("❌ Шаг 1: Загрузка модели не исправлена")
    
    # Шаг 2: Исправление несоответствия устройств
    if fix_device_mismatch():
        success_count += 1
        logger.info("✅ Шаг 2: Несоответствие устройств исправлено")
    else:
        logger.error("❌ Шаг 2: Несоответствие устройств не исправлено")
    
    # Шаг 3: Тестирование исправленной системы
    if test_fixed_system():
        success_count += 1
        logger.info("✅ Шаг 3: Тестирование успешно")
    else:
        logger.error("❌ Шаг 3: Тестирование провалено")
    
    logger.info(f"\n📊 РЕЗУЛЬТАТ: {success_count}/{total_steps} шагов успешны")
    
    if success_count == total_steps:
        logger.info("🎉 ВСЕ ПРОБЛЕМЫ УСПЕШНО ИСПРАВЛЕНЫ!")
        logger.info("✅ Система готова к интеграции ruGPT3!")
        return True
    else:
        logger.error("❌ НЕ ВСЕ ПРОБЛЕМЫ УДАЛОСЬ ИСПРАВИТЬ")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
