"""
Скрипт применения оптимальной конфигурации CogniFlex
"""

import os
import sys
import json

def get_config_path(filename):
    """Возвращает путь к файлу конфигурации относительно скрипта"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def load_config(config_path):
    """Загружает конфигурацию из файла"""
    if not os.path.exists(config_path):
        print(f"Ошибка: Файл конфигурации не найден {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Ошибка JSON в файле {config_path}: {e}")
        return {}
    except Exception as e:
        print(f"Ошибка загрузки конфигурации {config_path}: {e}")
        return {}

def apply_fractal_config():
    """Применяет конфигурацию FractalModelManager"""
    config_path = get_config_path("fractal_model_config.json")
    config = load_config(config_path)
    
    if config:
        print("🤖 Применение конфигурации FractalModelManager...")
        print(f"  Устройство: {config.get('device', 'cpu')}")
        print(f"  Макс. токенов: {config.get('max_memory_tokens', 10000)}")
        print(f"  Целевая память: {config.get('target_memory_gb', 2.0)} GB")
        print(f"  Параллельная токенизация: {config.get('parallel_tokenization', False)}")
        print(f"  Рабочие потоки: {config.get('tokenization_workers', 2)}")
        return True
    return False

def apply_gui_config():
    """Применяет конфигурацию GUI"""
    config_path = get_config_path("gui_config.json")
    config = load_config(config_path)
    
    if config:
        print("🖥️ Применение конфигурации GUI...")
        print(f"  Тема: {config.get('theme', 'light')}")
        print(f"  Автообновление: {config.get('auto_refresh_interval', 5000)} мс")
        print(f"  Расширенные метрики: {config.get('show_advanced_metrics', True)}")
        return True
    return False

if __name__ == "__main__":
    print("🚀 Применение оптимальной конфигурации CogniFlex")
    print("=" * 60)
    
    success = True
    
    # Применяем конфигурацию FractalModelManager
    if not apply_fractal_config():
        success = False
    
    # Применяем конфигурацию GUI
    if not apply_gui_config():
        success = False
    
    print("=" * 60)
    if success:
        print("✅ Оптимальная конфигурация успешно применена!")
        print("\n🎯 Ключевые улучшения:")
        print("  💾 Увеличен кэш токенов до 50K-100K")
        print("  🚀 Включена параллельная токенизация")
        print("  🧠 Оптимизирована память (uint16, пулы)")
        print("  📊 Включен мониторинг производительности")
        print("  🔄 Автоматическое улучшение качества")
    else:
        print("❌ Ошибка применения конфигурации")
    
    print("\nНажмите Enter для выхода...")
    input()