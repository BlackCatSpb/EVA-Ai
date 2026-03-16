#!/usr/bin/env python3
"""
Скрипт интеграции модулей CogniFlex
Создает интегрированные версии модулей с поддержкой BaseComponent и EventBus
"""

import os
import sys
import shutil
from datetime import datetime

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_integrated_adaptation():
    """Создает интегрированную версию адаптационного модуля"""
    print("🔧 Создание интегрированного адаптационного модуля...")
    
    template = '''"""
Интегрированный адаптационный менеджер CogniFlex
Поддерживает BaseComponent и EventBus
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("cogniflex.adaptation")

from cogniflex.core.base_component import BaseComponent, ComponentState
from cogniflex.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный AdaptationManager
try:
    from cogniflex.adaptation.adaptation_core import AdaptationManager
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный AdaptationManager недоступен")


class IntegratedAdaptationManager(BaseComponent):
    """Интегрированный менеджер адаптации с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("adaptation_manager", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'adaptation_cache')
        
        # Инициализируем оригинальный менеджер если доступен
        self._original_manager = None
        if ORIGINAL_AVAILABLE:
            try:
                self._original_manager = AdaptationManager(brain, cache_dir)
                logger.info("Оригинальный AdaptationManager инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального менеджера: {e}")
        
        # Статистика
        self.stats = {
            "adaptations_performed": 0,
            "profiles_created": 0,
            "feedback_processed": 0,
            "errors": 0
        }
        
        logger.info(f"IntegratedAdaptationManager {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация адаптационного менеджера...")
            
            # Инициализируем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'initialize'):
                self._original_manager.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Публикуем событие инициализации
            self._emit_event("adaptation_manager.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации адаптационного менеджера: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск адаптационного менеджера...")
            
            # Запускаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'start'):
                self._original_manager.start()
            
            # Публикуем событие запуска
            self._emit_event("adaptation_manager.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска адаптационного менеджера: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка адаптационного менеджера...")
            
            # Останавливаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'stop'):
                self._original_manager.stop()
            
            # Публикуем событие остановки
            self._emit_event("adaptation_manager.stopped", {
                'component': self.name,
                'stats': self.stats
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки адаптационного менеджера: {e}")
            return False
    
    def adapt_response(self, query: str, response: str, user_profile: Optional[Dict] = None) -> str:
        """Адаптирует ответ под профиль пользователя"""
        start_time = time.time()
        
        try:
            if self._original_manager and hasattr(self._original_manager, 'adapt_response'):
                adapted_response = self._original_manager.adapt_response(query, response, user_profile)
            else:
                # Базовая адаптация
                adapted_response = self._basic_adaptation(query, response, user_profile)
            
            # Обновляем статистику
            self.stats["adaptations_performed"] += 1
            
            # Публикуем событие адаптации
            self._emit_event("adaptation_manager.response_adapted", {
                'query_length': len(query),
                'response_length': len(response),
                'adapted_length': len(adapted_response),
                'processing_time': time.time() - start_time
            })
            
            return adapted_response
            
        except Exception as e:
            logger.error(f"Ошибка адаптации ответа: {e}")
            self.stats["errors"] += 1
            return response  # Возвращаем оригинальный ответ при ошибке
    
    def _basic_adaptation(self, query: str, response: str, user_profile: Optional[Dict] = None) -> str:
        """Базовая адаптация ответа"""
        # Простая логика адаптации
        if user_profile:
            # Если профиль указывает на формальный стиль
            if user_profile.get('style') == 'formal':
                response = response.replace("Привет", "Здравствуйте")
            elif user_profile.get('style') == 'casual':
                response = response.replace("Здравствуйте", "Привет")
        
        return response
    
    def create_user_profile(self, user_id: str, preferences: Dict) -> bool:
        """Создает профиль пользователя"""
        try:
            if self._original_manager and hasattr(self._original_manager, 'create_user_profile'):
                success = self._original_manager.create_user_profile(user_id, preferences)
            else:
                # Базовое создание профиля
                success = self._create_basic_profile(user_id, preferences)
            
            if success:
                self.stats["profiles_created"] += 1
                self._emit_event("adaptation_manager.profile_created", {
                    'user_id': user_id,
                    'preferences': preferences
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка создания профиля: {e}")
            self.stats["errors"] += 1
            return False
    
    def _create_basic_profile(self, user_id: str, preferences: Dict) -> bool:
        """Базовое создание профиля"""
        profile_path = os.path.join(self.cache_dir, f"profile_{user_id}.json")
        try:
            import json
            with open(profile_path, 'w') as f:
                json.dump({
                    'user_id': user_id,
                    'preferences': preferences,
                    'created_at': datetime.now().isoformat()
                }, f)
            return True
        except Exception:
            return False
    
    def process_feedback(self, user_id: str, feedback: Dict) -> bool:
        """Обрабатывает обратную связь"""
        try:
            if self._original_manager and hasattr(self._original_manager, 'process_feedback'):
                success = self._original_manager.process_feedback(user_id, feedback)
            else:
                # Базовая обработка
                success = self._process_basic_feedback(user_id, feedback)
            
            if success:
                self.stats["feedback_processed"] += 1
                self._emit_event("adaptation_manager.feedback_processed", {
                    'user_id': user_id,
                    'feedback_type': feedback.get('type'),
                    'rating': feedback.get('rating')
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка обработки обратной связи: {e}")
            self.stats["errors"] += 1
            return False
    
    def _process_basic_feedback(self, user_id: str, feedback: Dict) -> bool:
        """Базовая обработка обратной связи"""
        feedback_path = os.path.join(self.cache_dir, f"feedback_{user_id}.json")
        try:
            import json
            # Загружаем существующие отзывы
            existing_feedback = []
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    existing_feedback = json.load(f)
            
            # Добавляем новый отзыв
            existing_feedback.append({
                **feedback,
                'timestamp': datetime.now().isoformat()
            })
            
            # Сохраняем
            with open(feedback_path, 'w') as f:
                json.dump(existing_feedback, f)
            
            return True
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы"""
        stats = self.stats.copy()
        
        # Добавляем статистику из оригинального менеджера
        if self._original_manager and hasattr(self._original_manager, 'get_statistics'):
            original_stats = self._original_manager.get_statistics()
            stats.update(original_stats)
        
        return stats
'''
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "adaptation", 
        "adaptation_integrated.py"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"   ✅ Создан файл: {output_path}")
    return output_path

def create_integrated_analytics():
    """Создает интегрированную версию аналитического модуля"""
    print("🔧 Создание интегрированного аналитического модуля...")
    
    template = '''"""
Интегрированный аналитический менеджер CogniFlex
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger("cogniflex.analytics")

from cogniflex.core.base_component import BaseComponent, ComponentState
from cogniflex.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный AnalyticsManager
try:
    from cogniflex.analytics.analytics_manager import AnalyticsManager
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный AnalyticsManager недоступен")


class IntegratedAnalyticsManager(BaseComponent):
    """Интегрированный менеджер аналитики с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("analytics_manager", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'analytics_cache')
        
        # Инициализируем оригинальный менеджер если доступен
        self._original_manager = None
        if ORIGINAL_AVAILABLE:
            try:
                self._original_manager = AnalyticsManager(brain, cache_dir)
                logger.info("Оригинальный AnalyticsManager инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального менеджера: {e}")
        
        # Метрики
        self.metrics = defaultdict(deque)
        self.performance_data = deque(maxlen=1000)
        
        logger.info(f"IntegratedAnalyticsManager {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация аналитического менеджера...")
            
            # Инициализируем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'initialize'):
                self._original_manager.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Публикуем событие инициализации
            self._emit_event("analytics_manager.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации аналитического менеджера: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск аналитического менеджера...")
            
            # Запускаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'start'):
                self._original_manager.start()
            
            # Публикуем событие запуска
            self._emit_event("analytics_manager.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска аналитического менеджера: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка аналитического менеджера...")
            
            # Останавливаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'stop'):
                self._original_manager.stop()
            
            # Сохраняем метрики
            self._save_metrics()
            
            # Публикуем событие остановки
            self._emit_event("analytics_manager.stopped", {
                'component': self.name,
                'metrics_count': len(self.metrics)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки аналитического менеджера: {e}")
            return False
    
    def track_query(self, query: str, response_time: float, success: bool = True):
        """Отслеживает выполнение запроса"""
        try:
            timestamp = datetime.now()
            
            # Сохраняем метрику
            self.metrics['queries'].append({
                'timestamp': timestamp,
                'query_length': len(query),
                'response_time': response_time,
                'success': success
            })
            
            # Сохраняем данные производительности
            self.performance_data.append({
                'timestamp': timestamp,
                'response_time': response_time,
                'success': success
            })
            
            # Публикуем событие
            self._emit_event("analytics_manager.query_tracked", {
                'query_length': len(query),
                'response_time': response_time,
                'success': success
            })
            
            # Передаем в оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'track_query'):
                self._original_manager.track_query(query, response_time, success)
                
        except Exception as e:
            logger.error(f"Ошибка отслеживания запроса: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики производительности"""
        try:
            if not self.performance_data:
                return {}
            
            response_times = [d['response_time'] for d in self.performance_data]
            success_count = sum(1 for d in self.performance_data if d['success'])
            
            metrics = {
                'total_queries': len(self.performance_data),
                'success_rate': success_count / len(self.performance_data),
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            }
            
            # Добавляем метрики из оригинального менеджера
            if self._original_manager and hasattr(self._original_manager, 'get_performance_metrics'):
                original_metrics = self._original_manager.get_performance_metrics()
                metrics.update(original_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка получения метрик: {e}")
            return {}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает информацию о здоровье системы"""
        try:
            health = {
                'component_status': 'healthy',
                'metrics_count': len(self.metrics),
                'performance_data_count': len(self.performance_data),
                'cache_dir_exists': os.path.exists(self.cache_dir)
            }
            
            # Добавляем информацию из оригинального менеджера
            if self._original_manager and hasattr(self._original_manager, 'get_system_health'):
                original_health = self._original_manager.get_system_health()
                health.update(original_health)
            
            return health
            
        except Exception as e:
            logger.error(f"Ошибка получения здоровья системы: {e}")
            return {'component_status': 'error', 'error': str(e)}
    
    def _save_metrics(self):
        """Сохраняет метрики в файл"""
        try:
            import json
            
            metrics_file = os.path.join(self.cache_dir, 'metrics.json')
            
            # Конвертируем deque в списки для сериализации
            serializable_metrics = {
                key: list(deque_data) for key, deque_data in self.metrics.items()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, default=str)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")
    
    def generate_report(self, period_hours: int = 24) -> Dict[str, Any]:
        """Генерирует отчет за указанный период"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=period_hours)
            
            # Фильтруем данные за период
            period_data = [
                d for d in self.performance_data 
                if d['timestamp'] > cutoff_time
            ]
            
            if not period_data:
                return {'period_hours': period_hours, 'data_points': 0}
            
            response_times = [d['response_time'] for d in period_data]
            success_count = sum(1 for d in period_data if d['success'])
            
            report = {
                'period_hours': period_hours,
                'data_points': len(period_data),
                'success_rate': success_count / len(period_data),
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            }
            
            # Публикуем событие генерации отчета
            self._emit_event("analytics_manager.report_generated", {
                'period_hours': period_hours,
                'data_points': len(period_data)
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return {'error': str(e)}
'''
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "analytics", 
        "analytics_integrated.py"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"   ✅ Создан файл: {output_path}")
    return output_path

def update_component_initializer():
    """Обновляет ComponentInitializer с новыми интегрированными модулями"""
    print("🔧 Обновление ComponentInitializer...")
    
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
        
        # Добавляем импорты новых модулей
        imports_section = '''# Импорты интегрированных модулей
from ..adaptation.adaptation_integrated import IntegratedAdaptationManager
from ..analytics.analytics_integrated import IntegratedAnalyticsManager
'''
        
        # Находим место для вставки импортов
        import_pos = content.find("# Импорты компонентов")
        if import_pos == -1:
            import_pos = content.find("from typing import")
        
        if import_pos != -1:
            # Вставляем импорты после существующих
            insert_pos = content.find("\n\n", import_pos) + 2
            content = content[:insert_pos] + imports_section + content[insert_pos:]
        
        # Добавляем фабрики
        factories_section = '''
    def create_adaptation_manager(self) -> IntegratedAdaptationManager:
        """Создает интегрированный менеджер адаптации."""
        try:
            logger.debug("Создание интегрированного адаптационного менеджера...")
            event_bus = self.brain.get_event_bus() if hasattr(self.brain, 'get_event_bus') else self.event_bus
            component = IntegratedAdaptationManager(
                event_bus=event_bus,
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, "adaptation")
            )
            logger.debug("Интегрированный адаптационный менеджер создан")
            return component
        except Exception as e:
            logger.error(f"Ошибка создания интегрированного адаптационного менеджера: {e}")
            raise
    
    def create_analytics_manager(self) -> IntegratedAnalyticsManager:
        """Создает интегрированный менеджер аналитики."""
        try:
            logger.debug("Создание интегрированного аналитического менеджера...")
            event_bus = self.brain.get_event_bus() if hasattr(self.brain, 'get_event_bus') else self.event_bus
            component = IntegratedAnalyticsManager(
                event_bus=event_bus,
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, "analytics")
            )
            logger.debug("Интегрированный аналитический менеджер создан")
            return component
        except Exception as e:
            logger.error(f"Ошибка создания интегрированного аналитического менеджера: {e}")
            raise
'''
        
        # Находим место для вставки фабрик
        factory_pos = content.find("def create_")
        if factory_pos != -1:
            # Вставляем перед первой фабрикой
            content = content[:factory_pos] + factories_section + content[factory_pos:]
        
        # Сохраняем обновленный файл
        with open(initializer_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"   ✅ Обновлен файл: {initializer_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка обновления ComponentInitializer: {e}")
        return False

def main():
    """Основная функция интеграции"""
    print("🚀 Интеграция модулей CogniFlex")
    print("=" * 50)
    
    results = []
    
    # 1. Создаем интегрированный адаптационный модуль
    try:
        adaptation_path = create_integrated_adaptation()
        results.append(("adaptation", True, adaptation_path))
    except Exception as e:
        results.append(("adaptation", False, str(e)))
    
    # 2. Создаем интегрированный аналитический модуль
    try:
        analytics_path = create_integrated_analytics()
        results.append(("analytics", True, analytics_path))
    except Exception as e:
        results.append(("analytics", False, str(e)))
    
    # 3. Обновляем ComponentInitializer
    try:
        initializer_success = update_component_initializer()
        results.append(("component_initializer", initializer_success, ""))
    except Exception as e:
        results.append(("component_initializer", False, str(e)))
    
    # 4. Итоги
    print(f"\n📊 ИТОГИ ИНТЕГРАЦИИ:")
    
    success_count = 0
    for module, success, path in results:
        status = "✅ УСПЕХ" if success else "❌ НЕУДАЧА"
        print(f"   {module}: {status}")
        if success and path:
            print(f"      📁 {path}")
        elif not success:
            print(f"      ⚠️ {path}")
        
        if success:
            success_count += 1
    
    print(f"\n🎯 Результат: {success_count}/{len(results)} модулей интегрировано")
    
    if success_count == len(results):
        print("🎉 Интеграция успешно завершена!")
        return True
    else:
        print("⚠️ Некоторые модули требуют внимания.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
