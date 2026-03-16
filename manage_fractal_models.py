#!/usr/bin/env python3
"""
Скрипт для экспорта моделей во фрактальное хранилище CogniFlex
"""

import os
import sys
import logging
import json
from pathlib import Path

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniflex.core.utils import setup_logging
from cogniflex.mlearning.enhanced_rugpt3_manager import EnhancedRuGPT3ModelManager, RUSSIAN_MODELS

def export_models_to_fractal_storage():
    """Экспортирует доступные модели во фрактальное хранилище"""
    setup_logging(log_dir='logs')
    
    print("🚀 Экспорт моделей во фрактальное хранилище CogniFlex...")
    
    # Директория для экспорта
    export_dir = Path("./fractal_exports")
    export_dir.mkdir(exist_ok=True)
    
    # Создаем менеджер
    manager = EnhancedRuGPT3ModelManager(
        brain=None,
        model_name="fractal_russian",
        cache_dir="./cache"
    )
    
    if not manager.initialized:
        print("❌ Не удалось инициализировать менеджер")
        return
    
    # Получаем доступные модели
    available_models = manager.get_available_models()
    
    print(f"\n📊 Доступные модели ({len(available_models)}):")
    for model_id, model_info in available_models.items():
        print(f"  • {model_id}: {model_info['description']}")
        print(f"    Размер: {model_info['size_mb']} MB")
        print(f"    Качество: {model_info['quality']}/10")
        print(f"    Локальная: {'Да' if not model_info['requires_download'] else 'Нет'}")
        print()
    
    # Экспортируем текущую модель
    current_model = manager.model_name
    export_path = export_dir / current_model
    
    print(f"📦 Экспорт модели {current_model}...")
    
    if manager.export_model(str(export_path)):
        print(f"✅ Модель успешно экспортирована в {export_path}")
        
        # Показываем содержимое
        print(f"\n📁 Содержимое экспорта:")
        for item in export_path.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  📄 {item.relative_to(export_path)} ({size_mb:.1f} MB)")
        
        # Показываем метаданные
        metadata_file = export_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"\n📊 Метаданные:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    else:
        print(f"❌ Не удалось экспортировать модель {current_model}")
    
    # Создаем отчет о моделях
    report = {
        "export_timestamp": str(Path().cwd()),
        "current_model": current_model,
        "available_models": available_models,
        "fractal_storage": True,
        "export_path": str(export_path)
    }
    
    report_file = export_dir / "model_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 Отчет сохранен в {report_file}")
    print(f"\n🎉 Экспорт завершен!")

def test_fractal_models():
    """Тестирует работу с фрактальными моделями"""
    setup_logging(log_dir='logs')
    
    print("🧪 Тестирование фрактальных моделей...")
    
    # Тестируем разные модели
    test_models = ["fractal_russian", "gpt2"]
    
    for model_name in test_models:
        print(f"\n🔧 Тест модели: {model_name}")
        
        try:
            manager = EnhancedRuGPT3ModelManager(
                brain=None,
                model_name=model_name,
                cache_dir="./cache"
            )
            
            if manager.initialized:
                print(f"  ✅ Модель {model_name} инициализирована")
                
                # Тест генерации
                test_queries = [
                    "Привет! Как дела?",
                    "Что такое искусственный интеллект?",
                    "Расскажи о машинном обучении"
                ]
                
                for query in test_queries:
                    response = manager.generate_response(query)
                    print(f"  📝 Вопрос: {query}")
                    print(f"  💬 Ответ: {response[:100]}...")
                    print()
                
                # Статистика
                stats = manager.get_stats()
                print(f"  📊 Статистика: {stats}")
                
                # Очистка
                manager.cleanup()
                
            else:
                print(f"  ❌ Не удалось инициализировать модель {model_name}")
                
        except Exception as e:
            print(f"  ❌ Ошибка с моделью {model_name}: {e}")
    
    print(f"\n🎉 Тестирование завершено!")

def show_model_comparison():
    """Показывает сравнение моделей"""
    print("📊 Сравнение русскоязычных моделей для CogniFlex:")
    
    models_info = [
        {
            "name": "Фрактальная модель",
            "description": "Локальная модель с фрактальной архитектурой",
            "size": "300 MB",
            "quality": "5/10",
            "internet": "Не требуется",
            "features": ["Фрактальное хранилище", "Уникальный токенизатор", "Локальная работа"]
        },
        {
            "name": "ruGPT-3 Small",
            "description": "Модель от Сбера (125M параметров)",
            "size": "600 MB", 
            "quality": "8/10",
            "internet": "Требуется для загрузки",
            "features": ["Хорошее качество русского", "Оптимизированный размер", "Поддержка Сбера"]
        },
        {
            "name": "ruGPT-3 Medium",
            "description": "Модель от Сбера (355M параметров)",
            "size": "1.5 GB",
            "quality": "9/10", 
            "internet": "Требуется для загрузки",
            "features": ["Высокое качество", "Больший размер", "Лучшее понимание"]
        },
        {
            "name": "GPT-2 (fallback)",
            "description": "Стандартная модель OpenAI",
            "size": "500 MB",
            "quality": "6/10 (с переводом)",
            "internet": "Не требуется",
            "features": ["Надежность", "Быстрая загрузка", "Fallback система"]
        }
    ]
    
    for model in models_info:
        print(f"\n🤖 {model['name']}")
        print(f"  📝 {model['description']}")
        print(f"  💾 Размер: {model['size']}")
        print(f"  ⭐ Качество: {model['quality']}")
        print(f"  🌐 Интернет: {model['internet']}")
        print(f"  🚀 Особенности: {', '.join(model['features'])}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Управление фрактальными моделями CogniFlex")
    parser.add_argument("--export", action="store_true", help="Экспортировать модели во фрактальное хранилище")
    parser.add_argument("--test", action="store_true", help="Протестировать фрактальные модели")
    parser.add_argument("--compare", action="store_true", help="Показать сравнение моделей")
    
    args = parser.parse_args()
    
    if args.export:
        export_models_to_fractal_storage()
    elif args.test:
        test_fractal_models()
    elif args.compare:
        show_model_comparison()
    else:
        print("Используйте: --export, --test или --compare")
        show_model_comparison()
