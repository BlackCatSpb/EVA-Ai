#!/usr/bin/env python3
"""
Проверка токенизации и генерации в CogniFlex
"""

import os
import sys

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniflex.core.core_brain import CoreBrain
import logging

def check_tokenization_and_generation():
    """Проверяет токенизацию и генерацию"""
    logging.basicConfig(level=logging.INFO)
    
    print('🔍 Проверка токенизации и генерации...')
    try:
        brain = CoreBrain()
        
        # Проверяем текстовый процессор
        print('🔧 Текстовый процессор:')
        text_processor = brain.get_component('text_processor')
        if text_processor:
            print(f'  ✅ Найден: {type(text_processor).__name__}')
            
            # Тестируем токенизацию
            test_text = 'Привет! Как дела?'
            try:
                if hasattr(text_processor, 'tokenize'):
                    tokens = text_processor.tokenize(test_text)
                    print(f'  ✅ Токенизация: {str(tokens)[:50]}...')
                else:
                    print(f'  ⚠️  Метод tokenize не найден')
                    
                if hasattr(text_processor, 'get_embeddings'):
                    embeddings = text_processor.get_embeddings([test_text])
                    shape_info = ''
                    if hasattr(embeddings, 'shape'):
                        shape_info = f' {embeddings.shape}'
                    print(f'  ✅ Эмбеддинги: {type(embeddings)}{shape_info}')
                else:
                    print(f'  ⚠️  Метод get_embeddings не найден')
            except Exception as e:
                print(f'  ❌ Ошибка токенизации: {e}')
        else:
            print(f'  ❌ Текстовый процессор не найден')
        
        # Проверяем модель
        print('\n🔧 Модель:')
        if hasattr(brain, 'fractal_model_manager'):
            model_manager = brain.fractal_model_manager
            print(f'  ✅ FractalModelManager найден')
            
            try:
                # Тестируем генерацию
                response = model_manager.generate('Привет!', max_length=50, temperature=0.7)
                print(f'  ✅ Генерация: {response[:100]}...')
            except Exception as e:
                print(f'  ❌ Ошибка генерации: {e}')
        else:
            print(f'  ❌ FractalModelManager не найден')
            
        # Проверяем кэш токенов
        print('\n🔧 Гибридный кэш:')
        hybrid_cache = brain.get_component('hybrid_cache')
        if hybrid_cache:
            print(f'  ✅ Найден: {type(hybrid_cache).__name__}')
            
            try:
                stats = hybrid_cache.get_stats()
                print(f'  📊 Статистика кэша: {stats}')
            except Exception as e:
                print(f'  ❌ Ошибка статистики: {e}')
        else:
            print(f'  ❌ Гибридный кэш не найден')
            
        # Проверяем обработку запросов
        print('\n🔧 Обработка запросов:')
        try:
            response = brain.process_query('Привет! Расскажи о себе кратко.')
            print(f'  ✅ Ответ: {response[:200]}...')
        except Exception as e:
            print(f'  ❌ Ошибка обработки: {e}')
            
    except Exception as e:
        print(f'❌ Общая ошибка: {e}')
        import traceback
        traceback.print_exc()

def main():
    """Основная функция"""
    check_tokenization_and_generation()

if __name__ == "__main__":
    main()
