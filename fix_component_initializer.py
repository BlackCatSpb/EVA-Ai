#!/usr/bin/env python3
"""
Исправление ComponentInitializer для использования интегрированных модулей
"""

import os
import sys

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_component_initializer():
    """Исправляет ComponentInitializer для использования интегрированных версий"""
    print("🔧 Исправление ComponentInitializer...")
    
    # Читаем текущий файл
    initializer_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "core", 
        "component_initializer.py"
    )
    
    try:
        with open(initializer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Заменяем локальные функции на вызовы методов класса
        
        # 1. Заменяем analytics_manager
        analytics_pattern = '''        # Создаем директорию для кэша, если её нет
        cache_dir = os.path.join(getattr(self.core_brain, 'cache_dir', './cache'), 'analytics')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Создаем экземпляр менеджера аналитики
        analytics_manager = AnalyticsManager(
            brain=self.core_brain,
            cache_dir=cache_dir
        )
                os.makedirs(cache_dir, exist_ok=True)
                
                # Создаем экземпляр менеджера аналитики
                analytics_manager = AnalyticsManager(
                    brain=self.core_brain,
                    cache_dir=cache_dir
                )
                
                self.logger.info("AnalyticsManager успешно создан")
                
                # Сохраняем ссылку для обратной совместимости
                self.analytics_manager = analytics_manager
                self.core_brain.analytics_manager = analytics_manager
                
                # Запускаем мониторинг
                analytics_manager.start_monitoring()
                
                return analytics_manager
                
            except Exception as e:
                self.logger.error(f"Ошибка при создании analytics_manager: {e}", exc_info=True)
                return None'''
        
        analytics_replacement = '''        # Analytics Manager - используем интегрированную версию
        self.component_factories['analytics_manager'] = self.create_analytics_manager'''
        
        content = content.replace(analytics_pattern, analytics_replacement)
        
        # 2. Заменяем learning_manager
        learning_pattern = '''        def create_learning_manager():
            try:
                from cogniflex.learning.learning_manager import LearningManager
                
                # Получаем зависимости
                memory_manager = self.get_component('memory_manager')
                ml_unit = self.get_component('ml_unit')
                
                if memory_manager is None or ml_unit is None:
                    self.logger.error(
                        "Не удалось инициализировать LearningManager: "
                        f"memory_manager={memory_manager is not None}, "
                        f"ml_unit={ml_unit is not None}"
                    )
                    return None
                
                # Создаем экземпляр менеджера обучения
                learning_manager = LearningManager(
                    brain=self.core_brain,
                    config=getattr(self.core_brain, 'config', {}).get('learning', {})
                )
                
                # Инициализируем, если требуется
                if hasattr(learning_manager, 'initialize') and callable(learning_manager.initialize):
                    learning_manager.initialize()
                
                # Регистрируем компонент
                if not self.register_component('learning_manager', learning_manager, 
                                            ['memory_manager', 'ml_unit']):
                    self.logger.warning("Не удалось зарегистрировать learning_manager как компонент")
                
                # Сохраняем ссылку для обратной совместимости
                self.learning_manager = learning_manager
                self.core_brain.learning_manager = learning_manager
                
                return learning_manager
                
            except Exception as e:
                self.logger.error(f"Ошибка при создании learning_manager: {e}", exc_info=True)
                return None'''
        
        learning_replacement = '''        # Learning Manager - используем интегрированную версию
        self.component_factories['learning_manager'] = self.create_learning_manager'''
        
        content = content.replace(learning_pattern, learning_replacement)
        
        # 3. Заменяем web_search_engine
        web_search_pattern = '''        # Web Search Engine
        def create_web_search_engine():
            try:
                from cogniflex.websearch.web_search_engine import WebSearchEngine
                
                # Создаем директорию для кэша, если её нет
                cache_dir = os.path.join(getattr(self.core_brain, 'cache_dir', './cache'), 'web_search')
                os.makedirs(cache_dir, exist_ok=True)
                
                # Получаем зависимости
                memory_manager = self.get_component('memory_manager')
                
                if memory_manager is None:
                    self.logger.error("Не удалось инициализировать WebSearchEngine: memory_manager не найден")
                    return None
                
                # Создаем экземпляр поискового движка
                web_search_engine = WebSearchEngine(
                    brain=self.core_brain,
                    cache_dir=cache_dir
                )
                
                # Инициализируем, если требуется
                if hasattr(web_search_engine, 'initialize') and callable(web_search_engine.initialize):
                    web_search_engine.initialize()
                
                # Регистрируем компонент
                if not self.register_component('web_search_engine', web_search_engine, 
                                            ['memory_manager']):
                    self.logger.warning("Не удалось зарегистрировать web_search_engine как компонент")
                
                # Сохраняем ссылку для обратной совместимости
                self.web_search_engine = web_search_engine
                self.core_brain.web_search_engine = web_search_engine
                
                return web_search_engine
                
            except Exception as e:
                self.logger.error(f"Ошибка при создании web_search_engine: {e}", exc_info=True)
                return None'''
        
        web_search_replacement = '''        # Web Search Engine - используем интегрированную версию
        self.component_factories['web_search_engine'] = self.create_web_search_engine'''
        
        content = content.replace(web_search_pattern, web_search_replacement)
        
        # 4. Удаляем дублированные регистрации в component_factories
        # Находим и удаляем старые регистрации
        old_registrations = [
            "'contradiction_manager': create_contradiction_manager,",
            "'learning_manager': create_learning_manager,",
            "'web_search_engine': create_web_search_engine,",
            "'analytics_manager': create_analytics_manager,",
            "'ethics_framework': create_ethics_framework,",
            "'adaptation_manager': create_adaptation_manager,"
        ]
        
        for registration in old_registrations:
            content = content.replace(registration, f"# {registration} # заменено на метод класса")
        
        # Сохраняем исправленный файл
        with open(initializer_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"   ✅ ComponentInitializer исправлен")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка исправления ComponentInitializer: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 Исправление ComponentInitializer для интегрированных модулей")
    print("=" * 60)
    
    success = fix_component_initializer()
    
    if success:
        print("\n🎉 ComponentInitializer успешно исправлен!")
        print("📋 Теперь используются интегрированные версии всех модулей")
    else:
        print("\n❌ Ошибка при исправлении ComponentInitializer")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
