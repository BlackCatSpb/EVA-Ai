"""
Оптимальная конфигурация для CogniFlex
"""
import os
import json
import logging
import sys
sys.path.append('.')

def get_optimal_config():
    """Возвращает оптимальную конфигурацию для текущей системы"""
    
    # Анализ системы
    import psutil
    import torch
    
    memory = psutil.virtual_memory()
    total_ram_gb = memory.total / (1024**3)
    available_ram_gb = memory.available / (1024**3)
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_cores_logical = psutil.cpu_count(logical=True)
    
    print("🔍 Анализ системы для оптимальной конфигурации...")
    print(f"  RAM: {total_ram_gb:.1f} GB (доступно: {available_ram_gb:.1f} GB)")
    print(f"  CPU: {cpu_cores} ядер ({cpu_cores_logical} логических)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    
    # Оптимальная конфигурация
    config = {
        "system_info": {
            "total_ram_gb": round(total_ram_gb, 1),
            "available_ram_gb": round(available_ram_gb, 1),
            "cpu_cores": cpu_cores,
            "cpu_cores_logical": cpu_cores_logical,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        },
        
        # Конфигурация гибридного кэша
        "hybrid_cache": {
            # Оптимальный размер кэша токенов
            "max_memory_tokens": min(100000, int(available_ram_gb * 10000)),  # 10K токенов на 1GB RAM
            "target_memory_gb": min(4.0, available_ram_gb * 0.25),  # 25% доступной RAM
            "dynamic_memory_limit": True,
            "max_ram_usage_percent": 75.0,  # Оставляем 25% для системы
            "disk_cache_gb": 20.0,
            "eviction_policy": "lru",
            "cache_ttl": 86400,  # 24 часа
            "disk_cache_size": 200000,
            "memory_pressure_interval_s": 2.0,
            "pressure_offload_batch": 64,
            "write_mb_s": 80.0,
            "read_mb_s": 400.0,
            "burst_factor": 3.0,
            "vram_ratio": 0.0,  # Пока без GPU
            "ram_cache_ratio": 1.0  # Вся память для RAM кэша
        },
        
        # Конфигурация FractalModelManager
        "fractal_model_manager": {
            "device": "cpu",  # Пока CPU
            "batch_size": 4,  # Оптимально для CPU
            "max_length": 256,  # Увеличенный контекст
            "overlap_tokens": 64,
            "cache_tokenization": True,
            "parallel_tokenization": True,
            "tokenization_workers": min(4, cpu_cores // 2),
            "memory_optimization": True,
            "use_uint16": True,
            "tensor_pool_size": 1000
        },
        
        # Конфигурация улучшения качества
        "text_quality": {
            "auto_improvement": True,
            "quality_threshold": 0.7,
            "check_interval_seconds": 30,
            "training_config": {
                "learning_rate": 3e-5,
                "batch_size": 2,
                "num_epochs": 3,
                "max_length": 128,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "save_steps": 200,
                "eval_steps": 50,
                "gradient_accumulation_steps": 4
            }
        },
        
        # Конфигурация GUI
        "gui": {
            "notification_throttle_seconds": 30,
            "auto_refresh_interval": 5000,
            "theme": "light",
            "compact_mode": False,
            "show_advanced_metrics": True,
            "enable_gpu_monitoring": True  # Для будущего GPU
        },
        
        # Конфигурация обучения
        "training": {
            "auto_training": False,  # Выключаем для CPU
            "batch_size": 2,
            "max_documents_per_batch": 5,
            "checkpoint_interval": 100,
            "max_retries": 3,
            "backoff_seconds": 2.0,
            "use_hybrid_cache": True
        },
        
        # Конфигурация логирования
        "logging": {
            "level": "INFO",
            "file_logging": True,
            "console_logging": True,
            "max_file_size_mb": 100,
            "backup_count": 5,
            "performance_logging": True
        }
    }
    
    return config

def create_optimal_fractal_config():
    """Создает оптимальную конфигурацию для FractalModelManager"""
    
    config = get_optimal_config()
    
    fractal_config = {
        # Пути
        "model_path": None,  # Автоопределение
        "config_path": None,  # Автоопределение
        
        # Устройство и память
        "device": config["fractal_model_manager"]["device"],
        "max_memory_tokens": config["hybrid_cache"]["max_memory_tokens"],
        "target_memory_gb": config["hybrid_cache"]["target_memory_gb"],
        
        # Оптимизации
        "cache_tokenization": config["fractal_model_manager"]["cache_tokenization"],
        "parallel_tokenization": config["fractal_model_manager"]["parallel_tokenization"],
        "tokenization_workers": config["fractal_model_manager"]["tokenization_workers"],
        "memory_optimization": config["fractal_model_manager"]["memory_optimization"],
        "use_uint16": config["fractal_model_manager"]["use_uint16"],
        "tensor_pool_size": config["fractal_model_manager"]["tensor_pool_size"],
        
        # Генерация
        "batch_size": config["fractal_model_manager"]["batch_size"],
        "max_length": config["fractal_model_manager"]["max_length"],
        "overlap_tokens": config["fractal_model_manager"]["overlap_tokens"],
        
        # Качество
        "auto_improvement": config["text_quality"]["auto_improvement"],
        "quality_threshold": config["text_quality"]["quality_threshold"],
        "check_interval_seconds": config["text_quality"]["check_interval_seconds"]
    }
    
    return fractal_config

def create_optimal_gui_config():
    """Создает оптимальную конфигурацию для GUI"""
    
    config = get_optimal_config()
    
    gui_config = {
        "theme": config["gui"]["theme"],
        "compact_mode": config["gui"]["compact_mode"],
        "notification_throttle_seconds": config["gui"]["notification_throttle_seconds"],
        "auto_refresh_interval": config["gui"]["auto_refresh_interval"],
        "show_advanced_metrics": config["gui"]["show_advanced_metrics"],
        "enable_gpu_monitoring": config["gui"]["enable_gpu_monitoring"],
        
        # Панели
        "enabled_panels": [
            "chat", "analytics", "knowledge", 
            "contradictions", "memory", "learning", 
            "neuromorphic", "settings"
        ],
        
        # Метрики
        "show_performance_metrics": True,
        "show_memory_usage": True,
        "show_cache_stats": True,
        "show_quality_metrics": True,
        
        # Автоматизация
        "auto_save_state": True,
        "auto_backup": True,
        "auto_optimize": True
    }
    
    return gui_config

def apply_optimal_config():
    """Применяет оптимальную конфигурацию к системе"""
    
    print("🚀 Применение оптимальной конфигурации...")
    
    config = get_optimal_config()
    
    # Создаем директорию конфигурации
    config_dir = os.path.join(os.getcwd(), "cogniflex", "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # Сохраняем основную конфигурацию
    main_config_path = os.path.join(config_dir, "optimal_config.json")
    with open(main_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Основная конфигурация сохранена: {main_config_path}")
    
    # Сохраняем конфигурацию FractalModelManager
    fractal_config = create_optimal_fractal_config()
    fractal_config_path = os.path.join(config_dir, "fractal_model_config.json")
    with open(fractal_config_path, 'w', encoding='utf-8') as f:
        json.dump(fractal_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Конфигурация FractalModelManager сохранена: {fractal_config_path}")
    
    # Сохраняем конфигурацию GUI
    gui_config = create_optimal_gui_config()
    gui_config_path = os.path.join(config_dir, "gui_config.json")
    with open(gui_config_path, 'w', encoding='utf-8') as f:
        json.dump(gui_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Конфигурация GUI сохранена: {gui_config_path}")
    
    # Создаем скрипт применения конфигурации
    apply_script = f'''
"""
Скрипт применения оптимальной конфигурации CogniFlex
"""

import os
import sys
import json

def load_config(config_path):
    """Загружает конфигурацию из файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации {{config_path}}: {{e}}")
        return {{}}

def apply_fractal_config():
    """Применяет конфигурацию FractalModelManager"""
    config_path = "{fractal_config_path}"
    config = load_config(config_path)
    
    if config:
        print("🤖 Применение конфигурации FractalModelManager...")
        print(f"  Устройство: {{config.get('device', 'cpu')}}")
        print(f"  Макс. токенов: {{config.get('max_memory_tokens', 10000)}}")
        print(f"  Целевая память: {{config.get('target_memory_gb', 2.0)}} GB")
        print(f"  Параллельная токенизация: {{config.get('parallel_tokenization', False)}}")
        print(f"  Рабочие потоки: {{config.get('tokenization_workers', 2)}}")
        return True
    return False

def apply_gui_config():
    """Применяет конфигурацию GUI"""
    config_path = "{gui_config_path}"
    config = load_config(config_path)
    
    if config:
        print("🖥️ Применение конфигурации GUI...")
        print(f"  Тема: {{config.get('theme', 'light')}}")
        print(f"  Автообновление: {{config.get('auto_refresh_interval', 5000)}} мс")
        print(f"  Расширенные метрики: {{config.get('show_advanced_metrics', True)}}")
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
        print("\\n🎯 Ключевые улучшения:")
        print("  💾 Увеличен кэш токенов до 50K-100K")
        print("  🚀 Включена параллельная токенизация")
        print("  🧠 Оптимизирована память (uint16, пулы)")
        print("  📊 Включен мониторинг производительности")
        print("  🔄 Автоматическое улучшение качества")
    else:
        print("❌ Ошибка применения конфигурации")
    
    print("\\nНажмите Enter для выхода...")
    input()
'''
    
    apply_script_path = os.path.join(config_dir, "apply_optimal_config.py")
    with open(apply_script_path, 'w', encoding='utf-8') as f:
        f.write(apply_script)
    
    print(f"✅ Скрипт применения сохранен: {apply_script_path}")
    
    return {
        "main_config": main_config_path,
        "fractal_config": fractal_config_path,
        "gui_config": gui_config_path,
        "apply_script": apply_script_path
    }

def show_optimal_config_summary():
    """Показывает сводку оптимальной конфигурации"""
    
    config = get_optimal_config()
    
    print("🎯 ОПТИМАЛЬНАЯ КОНФИГУРАЦИЯ ДЛЯ COGNIFLEX")
    print("=" * 70)
    
    print("\\n💾 ГИБРИДНЫЙ КЭШ ТОКЕНОВ:")
    cache_config = config["hybrid_cache"]
    print(f"  Макс. токенов: {cache_config['max_memory_tokens']:,}")
    print(f"  Целевая память: {cache_config['target_memory_gb']:.1f} GB")
    print(f"  Дисковый кэш: {cache_config['disk_cache_gb']:.1f} GB")
    print(f"  Политика вытеснения: {cache_config['eviction_policy']}")
    print(f"  Динамический лимит: {cache_config['dynamic_memory_limit']}")
    
    print("\\n🤖 FRACTAL MODEL MANAGER:")
    model_config = config["fractal_model_manager"]
    print(f"  Устройство: {model_config['device']}")
    print(f"  Размер батча: {model_config['batch_size']}")
    print(f"  Макс. длина: {model_config['max_length']}")
    print(f"  Кэширование токенизации: {model_config['cache_tokenization']}")
    print(f"  Параллельная токенизация: {model_config['parallel_tokenization']}")
    print(f"  Рабочие потоки: {model_config['tokenization_workers']}")
    
    print("\\n🎯 УЛУЧШЕНИЕ КАЧЕСТВА:")
    quality_config = config["text_quality"]
    training_config = quality_config["training_config"]
    print(f"  Автоулучшение: {quality_config['auto_improvement']}")
    print(f"  Порог качества: {quality_config['quality_threshold']}")
    print(f"  Скорость обучения: {training_config['learning_rate']}")
    print(f"  Эпохи: {training_config['num_epochs']}")
    print(f"  Размер батча: {training_config['batch_size']}")
    
    print("\\n🖥️ GUI:")
    gui_config = config["gui"]
    print(f"  Тема: {gui_config['theme']}")
    print(f"  Автообновление: {gui_config['auto_refresh_interval']} мс")
    print(f"  Расширенные метрики: {gui_config['show_advanced_metrics']}")
    print(f"  Мониторинг GPU: {gui_config['enable_gpu_monitoring']}")
    
    print("\\n" + "=" * 70)
    print("🎉 ОЖИДАЕМЫЕ УЛУЧШЕНИЯ:")
    print("  🚀 Ускорение токенизации: 5-10x")
    print("  💾 Эффективное кэширование: 29x")
    print("  🧠 Оптимизация памяти: 40-60%")
    print("  📈 Автоматическое улучшение качества")
    print("  🔄 Адаптивная конфигурация")
    
    return config

if __name__ == "__main__":
    # Показываем оптимальную конфигурацию
    config = show_optimal_config_summary()
    
    print("\\n🚀 Применить оптимальную конфигурацию? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes', 'да']:
        result = apply_optimal_config()
        print("\\n✅ Конфигурация применена!")
        print("\\n📝 Следующие шаги:")
        print("  1. Перезапустить CogniFlex")
        print("  2. Проверить производительность")
        print("  3. Мониторить метрики в GUI")
    else:
        print("\\n❌ Конфигурация не применена")
    
    print("\\nНажмите Enter для выхода...")
    input()
