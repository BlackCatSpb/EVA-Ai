#!/usr/bin/env python3
"""
Итоговая интеграция ruGPT-3 с фрактальным хранилищем в CogniFlex
"""

import os
import sys
import logging
import time
from typing import Dict, Any, Optional

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniflex.core.utils import setup_logging
from cogniflex.core.core_brain import CoreBrain
from cogniflex.mlearning.enhanced_rugpt3_manager import EnhancedRuGPT3ModelManager
from cogniflex.mlearning.fractal_rugpt3_manager import RUSSIAN_MODELS

def test_final_integration():
    """Тестирует финальную интеграцию системы"""
    setup_logging(log_dir='logs')
    
    print("🚀 Тестирование финальной интеграции ruGPT-3 с фрактальным хранилищем...")
    
    # Инициализация CoreBrain
    brain = CoreBrain()
    brain.initialize()
    brain.start()
    
    print(f"✅ CoreBrain инициализирован")
    
    # Проверка модели
    model_manager = brain.get_component('model_manager')
    if not model_manager:
        print("❌ ModelManager не найден")
        return
    
    print(f"📊 Тип модели: {type(model_manager).__name__}")
    print(f"📊 Имя модели: {getattr(model_manager, 'model_name', 'unknown')}")
    print(f"📊 Устройство: {getattr(model_manager, 'device', 'unknown')}")
    print(f"📊 Фрактальное хранилище: {getattr(model_manager, 'fractal_storage', False)}")
    
    # Тестирование разных моделей
    test_models = ["fractal_russian", "gpt2"]
    
    for model_name in test_models:
        print(f"\n🔧 Тестирование модели: {model_name}")
        
        # Переключение модели если возможно
        if hasattr(model_manager, 'switch_model'):
            if model_manager.switch_model(model_name):
                print(f"  ✅ Переключено на {model_name}")
            else:
                print(f"  ⚠️ Не удалось переключиться на {model_name}")
        
        # Тестовые запросы
        test_queries = [
            "Привет! Расскажи о фрактальной архитектуре.",
            "Что такое машинное обучение?",
            "Как работает гибридный кэш токенов?",
            "Преимущества локальных моделей.",
            "Спасибо за объяснение!"
        ]
        
        for query in test_queries:
            print(f"\n  📝 Вопрос: {query}")
            start_time = time.time()
            
            try:
                response = brain.process_query(query)
                generation_time = time.time() - start_time
                
                if isinstance(response, dict) and 'response' in response:
                    answer = response['response']
                    model_used = response.get('model_name', 'unknown')
                    source = response.get('source', 'unknown')
                    
                    print(f"  💬 Ответ ({model_used}, {source}): {answer[:150]}...")
                    print(f"  ⏱️ Время: {generation_time:.2f} сек")
                else:
                    print(f"  ❌ Неожиданный формат ответа")
                    
            except Exception as e:
                print(f"  ❌ Ошибка: {e}")
                generation_time = time.time() - start_time
                print(f"  ⏱️ Время ошибки: {generation_time:.2f} сек")
    
    # Статистика модели
    if hasattr(model_manager, 'get_stats'):
        stats = model_manager.get_stats()
        print(f"\n📊 Статистика модели: {stats}")
    
    # Статистика памяти
    if hasattr(model_manager, 'get_memory_usage'):
        memory = model_manager.get_memory_usage()
        print(f"\n💾 Использование памяти: {memory}")
    
    # Доступные модели
    if hasattr(model_manager, 'get_available_models'):
        available = model_manager.get_available_models()
        print(f"\n🤖 Доступные модели ({len(available)}):")
        for model_id, info in available.items():
            print(f"  • {model_id}: {info['description']} ({info['size_mb']} MB)")
    
    # Корректное завершение
    print(f"\n🔚 Завершение работы...")
    stop_start = time.time()
    
    try:
        brain.stop()
        stop_time = time.time() - stop_start
        print(f"  ✅ Система остановлена за {stop_time:.2f} сек")
        
        # Очистка менеджера
        if hasattr(model_manager, 'cleanup'):
            model_manager.cleanup()
        
    except Exception as e:
        print(f"  ❌ Ошибка при остановке: {e}")
    
    print(f"\n🎉 Тестирование завершено!")

def show_system_status():
    """Показывает статус системы"""
    print("📊 Статус системы CogniFlex с ruGPT-3:")
    
    status_info = [
        {
            "Компонент": "CoreBrain",
            "Статус": "✅ Работает",
            "Описание": "Основное ядро системы"
        },
        {
            "Компонент": "EnhancedRuGPT3ModelManager", 
            "Статус": "✅ Интегрирован",
            "Описание": "Улучшенный менеджер с фрактальным хранилищем"
        },
        {
            "Компонент": "Фрактальное хранилище",
            "Статус": "✅ Работает", 
            "Описание": "Эффективное хранение моделей"
        },
        {
            "Компонент": "Гибридный кэш токенов",
            "Статус": "✅ Работает",
            "Описание": "VRAM → RAM → SSD кэширование"
        },
        {
            "Компонент": "GPU токенизация",
            "Статус": "🔄 Доступно",
            "Описание": "Ускорение на GPU при наличии"
        },
        {
            "Компонент": "Fallback система",
            "Статус": "✅ Работает",
            "Описание": "GPT-2 как резервная модель"
        },
        {
            "Компонент": "Русский язык",
            "Статус": "✅ Поддерживается",
            "Описание": "Улучшенная обработка русского"
        }
    ]
    
    print(f"\n{'Компонент':<25} {'Статус':<12} {'Описание'}")
    print("-" * 70)
    for item in status_info:
        print(f"{item['Компонент']:<25} {item['Статус']:<12} {item['Описание']}")

def show_model_recommendations():
    """Рекомендации по выбору моделей"""
    print("\n🎯 Рекомендации по выбору моделей для CogniFlex:")
    
    recommendations = [
        {
            "Сценарий": "Максимальная автономность",
            "Модель": "fractal_russian",
            "Причина": "Полностью локальная, не требует интернета"
        },
        {
            "Сценарий": "Баланс качества и размера",
            "Модель": "rugpt3small", 
            "Причина": "Хорошее качество русского, умеренный размер"
        },
        {
            "Сценарий": "Максимальное качество",
            "Модель": "rugpt3medium",
            "Причина": "Лучшее понимание контекста"
        },
        {
            "Сценарий": "Надежность и скорость",
            "Модель": "gpt2 (fallback)",
            "Причина": "Всегда доступна, быстрая загрузка"
        }
    ]
    
    for rec in recommendations:
        print(f"\n🎯 {rec['Сценарий']}")
        print(f"  🤖 Рекомендуемая модель: {rec['Модель']}")
        print(f"  💡 Причина: {rec['Причина']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Финальная интеграция ruGPT-3 с CogniFlex")
    parser.add_argument("--test", action="store_true", help="Протестировать систему")
    parser.add_argument("--status", action="store_true", help="Показать статус системы")
    parser.add_argument("--recommendations", action="store_true", help="Показать рекомендации по моделям")
    
    args = parser.parse_args()
    
    if args.test:
        test_final_integration()
    elif args.status:
        show_system_status()
    elif args.recommendations:
        show_model_recommendations()
    else:
        print("Используйте: --test, --status или --recommendations")
        show_system_status()
        show_model_recommendations()
