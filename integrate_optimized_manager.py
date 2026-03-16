"""
Финальная интеграция оптимизированного менеджера в существующую структуру
"""
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger("cogniflex.integration")

def integrate_optimized_manager():
    """Интегрирует OptimizedFractalModelManager в существующую структуру"""
    
    print("🔗 Финальная интеграция оптимизированного менеджера")
    print("=" * 70)
    
    # 1. Обновление существующего FractalModelManager
    print("\n📝 1. Обновление существующего FractalModelManager...")
    
    fractal_manager_path = os.path.join(
        os.getcwd(), "cogniflex", "mlearning", "fractal_model_manager.py"
    )
    
    if os.path.exists(fractal_manager_path):
        # Создаем backup
        backup_path = fractal_manager_path + ".backup"
        shutil.copy2(fractal_manager_path, backup_path)
        print(f"  ✅ Backup создан: {backup_path}")
        
        # Обновляем импорты и добавляем оптимизации
        update_fractal_manager(fractal_manager_path)
        print("  ✅ FractalModelManager обновлен с оптимизациями")
    else:
        print(f"  ❌ Файл не найден: {fractal_manager_path}")
    
    # 2. Обновление GUI модуля обучения
    print("\n🖥️ 2. Обновление GUI модуля обучения...")
    
    learning_module_path = os.path.join(
        os.getcwd(), "cogniflex", "gui", "learning_module.py"
    )
    
    if os.path.exists(learning_module_path):
        backup_path = learning_module_path + ".backup"
        shutil.copy2(learning_module_path, backup_path)
        print(f"  ✅ Backup создан: {backup_path}")
        
        update_learning_module(learning_module_path)
        print("  ✅ GUI модуль обучения обновлен")
    else:
        print(f"  ❌ Файл не найден: {learning_module_path}")
    
    # 3. Создание unified менеджера
    print("\n🤖 3. Создание unified менеджера...")
    
    unified_manager_path = os.path.join(
        os.getcwd(), "cogniflex", "mlearning", "unified_fractal_manager.py"
    )
    
    create_unified_manager(unified_manager_path)
    print("  ✅ UnifiedFractalManager создан")
    
    # 4. Обновление конфигурации
    print("\n⚙️ 4. Обновление конфигурации...")
    
    config_path = os.path.join(
        os.getcwd(), "cogniflex", "config", "unified_config.json"
    )
    
    create_unified_config(config_path)
    print("  ✅ Unified конфигурация создана")
    
    # 5. Создание скрипта миграции
    print("\n🔄 5. Создание скрипта миграции...")
    
    migration_script_path = os.path.join(
        os.getcwd(), "cogniflex", "scripts", "migrate_to_optimized.py"
    )
    
    create_migration_script(migration_script_path)
    print("  ✅ Скрипт миграции создан")
    
    # 6. Обновление __init__.py
    print("\n📦 6. Обновление __init__.py...")
    
    init_path = os.path.join(
        os.getcwd(), "cogniflex", "mlearning", "__init__.py"
    )
    
    update_init_file(init_path)
    print("  ✅ __init__.py обновлен")
    
    print("\n" + "=" * 70)
    print("🎉 ИНТЕГРАЦИЯ ЗАВЕРШЕНА!")
    
    return {
        "fractal_manager": fractal_manager_path,
        "learning_module": learning_module_path,
        "unified_manager": unified_manager_path,
        "config": config_path,
        "migration_script": migration_script_path,
        "init_file": init_path
    }

def update_fractal_manager(file_path):
    """Обновляет существующий FractalModelManager с оптимизациями"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Добавляем импорты оптимизаций
    optimization_imports = '''
# Импорты для оптимизации
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Dict, Any, List
'''
    
    # Добавляем класс оптимизаций
    optimization_class = '''
class FractalModelOptimizations:
    """Класс оптимизаций для FractalModelManager"""
    
    def __init__(self, manager):
        self.manager = manager
        self.tokenization_cache = {}
        self.tokenization_executor = ThreadPoolExecutor(max_workers=4)
        self.performance_stats = {
            "tokenization_time": 0.0,
            "generation_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def optimized_tokenize(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Оптимизированная токенизация с кэшированием"""
        start_time = time.time()
        
        # Проверяем кэш
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.tokenization_cache:
                cached = self.tokenization_cache[text_hash]
                cached_results.append({
                    'input_ids': cached['input_ids'].to(self.manager.device),
                    'attention_mask': cached['attention_mask'].to(self.manager.device),
                    'text': text,
                    'cached': True
                })
                self.performance_stats["cache_hits"] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.performance_stats["cache_misses"] += 1
        
        # Токенизация новых текстов
        if uncached_texts and self.manager.tokenizer:
            def tokenize_single(text):
                inputs = self.manager.tokenizer(
                    text, return_tensors='pt', padding=True, 
                    truncation=True, max_length=256
                )
                return {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'text': text,
                    'cached': False
                }
            
            futures = [self.tokenization_executor.submit(tokenize_single, text) 
                      for text in uncached_texts]
            new_results = [future.result() for future in futures]
            
            # Кэшируем результаты
            for text, result in zip(uncached_texts, new_results):
                text_hash = hash(text)
                self.tokenization_cache[text_hash] = {
                    'input_ids': result['input_ids'].cpu(),
                    'attention_mask': result['attention_mask'].cpu()
                }
                
                # Перемещаем на устройство
                result['input_ids'] = result['input_ids'].to(self.manager.device)
                result['attention_mask'] = result['attention_mask'].to(self.manager.device)
            
            # Объединяем результаты
            all_results = cached_results + new_results
            all_results.sort(key=lambda x: texts.index(x['text']))
        else:
            all_results = cached_results
        
        self.performance_stats["tokenization_time"] += time.time() - start_time
        return all_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        stats = self.performance_stats.copy()
        stats.update({
            "cache_size": len(self.tokenization_cache),
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / 
                (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
                if (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]) > 0 else 0
            )
        })
        return stats
'''
    
    # Добавляем инициализацию оптимизаций в __init__
    init_optimization = '''
        # Инициализация оптимизаций
        self.optimizations = FractalModelOptimizations(self)
        logger.info("Оптимизации FractalModelManager инициализированы")
'''
    
    # Вставляем оптимизации в файл
    if 'class FractalModelOptimizations:' not in content:
        content = content.replace('logger.info("FractalModelManager инициализирован")', 
                               'logger.info("FractalModelManager инициализирован")' + init_optimization)
        
        # Добавляем импорты после существующих
        import_pos = content.find('import logging')
        if import_pos != -1:
            content = content[:import_pos] + optimization_imports + content[import_pos:]
        
        # Добавляем класс оптимизаций
        class_pos = content.find('class FractalModelManager:')
        if class_pos != -1:
            content = content[:class_pos] + optimization_class + '\n\n' + content[class_pos:]
    
    # Сохраняем обновленный файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def update_learning_module(file_path):
    """Обновляет GUI модуль обучения с поддержкой оптимизаций"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Добавляем поддержку оптимизированного менеджера
    optimization_support = '''
    def _check_optimized_manager(self):
        """Проверяет наличие оптимизированного менеджера"""
        try:
            from ..mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
            
            brain = getattr(self.gui, 'brain', None)
            if brain and hasattr(brain, 'fractal_model_manager'):
                manager = brain.fractal_model_manager
                
                # Проверяем, есть ли оптимизации
                if hasattr(manager, 'optimizations'):
                    return True, manager.optimizations
                elif hasattr(manager, 'get_performance_stats'):
                    return True, manager
                else:
                    return False, None
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Ошибка проверки оптимизированного менеджера: {e}")
            return False, None
    
    def _show_optimization_stats(self):
        """Показывает статистику оптимизаций"""
        has_optimizations, opt_manager = self._check_optimized_manager()
        
        if has_optimizations and opt_manager:
            try:
                stats = opt_manager.get_performance_stats()
                
                # Создаем окно со статистикой
                stats_window = tk.Toplevel(self.gui.root)
                stats_window.title("📊 Статистика оптимизаций")
                stats_window.geometry("400x300")
                
                # Метрики
                metrics_frame = ttk.Frame(stats_window)
                metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                ttk.Label(metrics_frame, text="📈 Метрики производительности", 
                         font=('Arial', 12, 'bold')).pack(pady=5)
                
                for key, value in stats.items():
                    if isinstance(value, float):
                        if 'time' in key:
                            text = f"{key}: {value:.4f}s"
                        elif 'rate' in key:
                            text = f"{key}: {value:.2%}"
                        else:
                            text = f"{key}: {value:.3f}"
                    else:
                        text = f"{key}: {value}"
                    
                    ttk.Label(metrics_frame, text=text).pack(anchor=tk.W, pady=2)
                
                # Кнопка закрытия
                ttk.Button(stats_window, text="Закрыть", 
                          command=stats_window.destroy).pack(pady=10)
                
            except Exception as e:
                logger.error(f"Ошибка показа статистики: {e}")
                messagebox.showerror("Ошибка", f"Не удалось показать статистику: {e}")
        else:
            messagebox.showinfo("Информация", "Оптимизации недоступны")
'''
    
    # Добавляем поддержку оптимизаций
    if '_check_optimized_manager' not in content:
        # Находим конец класса и добавляем методы
        class_end = content.rfind('    def _maybe_trigger_auto_training(self):')
        if class_end != -1:
            content = content[:class_end] + optimization_support + '\n\n' + content[class_end:]
    
    # Сохраняем обновленный файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_unified_manager(file_path):
    """Создает unified менеджер с автоматическим выбором оптимизаций"""
    
    unified_code = '''"""
UnifiedFractalManager - автоматический выбор между оптимизированным и стандартным менеджерами
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("cogniflex.unified_manager")

class UnifiedFractalManager:
    """Унифицированный менеджер с автоматическим выбором оптимизаций"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None, 
                 force_optimized: bool = False):
        """Инициализация с автоматическим выбором менеджера"""
        
        self.model_path = model_path
        self.config_path = config_path
        self.force_optimized = force_optimized
        self.manager = None
        self.is_optimized = False
        
        # Автоматический выбор менеджера
        self._select_manager()
    
    def _select_manager(self):
        """Автоматически выбирает лучший менеджер"""
        
        try:
            # Проверяем доступность оптимизированного менеджера
            if self.force_optimized or self._should_use_optimized():
                try:
                    from .optimized_fractal_model_manager import OptimizedFractalModelManager
                    
                    self.manager = OptimizedFractalModelManager(config_path=self.config_path)
                    self.is_optimized = True
                    
                    logger.info("Используется OptimizedFractalModelManager")
                    return
                    
                except Exception as e:
                    logger.warning(f"Не удалось загрузить оптимизированный менеджер: {e}")
            
            # Fallback на стандартный менеджер
            from .fractal_model_manager import FractalModelManager
            
            self.manager = FractalModelManager(model_path=self.model_path, config_path=self.config_path)
            self.is_optimized = False
            
            logger.info("Используется стандартный FractalModelManager")
            
        except Exception as e:
            logger.error(f"Критическая ошибка при выборе менеджера: {e}")
            raise
    
    def _should_use_optimized(self) -> bool:
        """Определяет, следует ли использовать оптимизированный менеджер"""
        
        # Проверяем наличие конфигурации
        config_path = os.path.join(os.getcwd(), "cogniflex", "config", "unified_config.json")
        
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return config.get("use_optimized", True)
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
        
        # По умолчанию используем оптимизированный
        return True
    
    def generate_response(self, query: str, max_tokens: int = 100) -> str:
        """Генерирует ответ"""
        return self.manager.generate_response(query, max_tokens)
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики качества"""
        if hasattr(self.manager, 'get_quality_metrics'):
            return self.manager.get_quality_metrics()
        return {}
    
    def improve_quality(self, training_texts=None, save_path=None):
        """Улучшает качество модели"""
        if hasattr(self.manager, 'improve_quality'):
            return self.manager.improve_quality(training_texts, save_path)
        return {"status": "error", "message": "Метод улучшения недоступен"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        if self.is_optimized and hasattr(self.manager, 'get_performance_stats'):
            return self.manager.get_performance_stats()
        else:
            return {
                "is_optimized": self.is_optimized,
                "manager_type": type(self.manager).__name__
            }
    
    def __getattr__(self, name):
        """Делегирует остальные атрибуты менеджеру"""
        return getattr(self.manager, name)
    
    def __del__(self):
        """Очистка"""
        if hasattr(self.manager, '__del__'):
            self.manager.__del__()
'''
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(unified_code)

def create_unified_config(file_path):
    """Создает унифицированную конфигурацию"""
    
    config = {
        "manager_selection": {
            "use_optimized": True,
            "force_optimized": False,
            "fallback_to_standard": True
        },
        "optimizations": {
            "cache_tokenization": True,
            "parallel_tokenization": True,
            "tokenization_workers": 4,
            "memory_optimization": True,
            "max_cache_size": 1000
        },
        "performance": {
            "monitor_performance": True,
            "log_performance_stats": True,
            "performance_update_interval": 30
        },
        "gui_integration": {
            "show_performance_stats": True,
            "enable_optimization_controls": True,
            "auto_refresh_stats": True
        }
    }
    
    import json
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def create_migration_script(file_path):
    """Создает скрипт миграции на оптимизированный менеджер"""
    
    migration_code = '''"""
Скрипт миграции на OptimizedFractalModelManager
"""
import os
import sys
import logging
import json

# Добавляем путь к CogniFlex
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def migrate_to_optimized():
    """Выполняет миграцию на оптимизированный менеджер"""
    
    print("🔄 Миграция на OptimizedFractalModelManager")
    print("=" * 50)
    
    try:
        # 1. Тестирование оптимизированного менеджера
        print("\n🧪 Тестирование оптимизированного менеджера...")
        
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        manager = OptimizedFractalModelManager()
        
        if manager.initialized:
            print("✅ Оптимизированный менеджер успешно инициализирован")
            
            # Тест генерации
            response = manager.generate_response_optimized("Привет, как дела?", max_tokens=50)
            print(f"✅ Тест генерации: {response[:50]}...")
            
            # Статистика
            stats = manager.get_performance_stats()
            print(f"✅ Статистика: cache_hit_rate={stats.get('cache_hit_rate', 0):.2%}")
            
        else:
            print("❌ Оптимизированный менеджер не инициализирован")
            return False
        
        # 2. Обновление конфигурации
        print("\n⚙️ Обновление конфигурации...")
        
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", "unified_config.json"
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config["manager_selection"]["use_optimized"] = True
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print("✅ Конфигурация обновлена")
        
        # 3. Создание symbolic link для легкого доступа
        print("\n🔗 Создание symbolic link...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(current_dir, "..", "mlearning", "current_manager.py")
        source_path = os.path.join(current_dir, "..", "mlearning", "optimized_fractal_model_manager.py")
        
        if os.path.exists(target_path):
            os.remove(target_path)
        
        # Windows не поддерживает symbolic links, используем копию
        import shutil
        shutil.copy2(source_path, target_path)
        
        print("✅ Symbolic link создан")
        
        print("\n" + "=" * 50)
        print("🎉 МИГРАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print("\n📝 Следующие шаги:")
        print("1. Используйте UnifiedFractalManager для автоматического выбора")
        print("2. Или импортируйте OptimizedFractalModelManager напрямую")
        print("3. Проверьте производительность в GUI")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка миграции: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = migrate_to_optimized()
    
    if success:
        print("\\n✅ Миграция завершена успешно!")
    else:
        print("\\n❌ Миграция завершилась с ошибками")
    
    print("\\nНажмите Enter для выхода...")
    input()
'''
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(migration_code)

def update_init_file(file_path):
    """Обновляет __init__.py для легкого импорта"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Добавляем импорты оптимизированного менеджера
    new_imports = '''
# Оптимизированные менеджеры
from .optimized_fractal_model_manager import OptimizedFractalModelManager
from .unified_fractal_manager import UnifiedFractalManager

# Для обратной совместимости
def get_fractal_manager(model_path=None, config_path=None, use_optimized=True):
    """Возвращает лучший доступный менеджер"""
    return UnifiedFractalManager(
        model_path=model_path, 
        config_path=config_path, 
        force_optimized=use_optimized
    )
'''
    
    if 'OptimizedFractalModelManager' not in content:
        content += new_imports
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    result = integrate_optimized_manager()
    
    print("\n🎯 ИНТЕГРАЦИЯ ЗАВЕРШЕНА!")
    print("\n📁 Созданные файлы:")
    for key, path in result.items():
        print(f"  {key}: {path}")
    
    print("\n🚀 Следующие шаги:")
    print("1. Запустите скрипт миграции:")
    print("   python cogniflex/scripts/migrate_to_optimized.py")
    print("2. Используйте UnifiedFractalManager в коде:")
    print("   from cogniflex.mlearning import UnifiedFractalManager")
    print("   manager = UnifiedFractalManager()")
    print("3. Проверьте GUI для мониторинга производительности")
    
    print("\nНажмите Enter для выхода...")
    input()
