# system_optimizer.py
"""Оптимизатор системы для ЕВА."""

import os
import sys
import time
import json
import logging
import threading
try:
    import psutil
except ImportError:
    psutil = None
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("eva_ai.system_optimizer")

class SystemOptimizer:
    """Система оптимизации ресурсов и производительности."""

    def __init__(self, core_brain):
        self.core_brain = core_brain
        self.optimization_history = []
        self.current_optimization_level = "normal"
        self.power_saving_mode = False
        self.optimization_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger("eva_ai.system_optimizer")
        self.optimization_interval = 30.0  # Интервал оптимизации в секундах

    def start_optimization_monitor(self):
        """Запускает фоновый мониторинг для автоматической оптимизации."""
        if self.optimization_thread and self.optimization_thread.is_alive():
            return  # Уже запущен

        self.stop_event.clear()

        def optimization_loop():
            while not self.stop_event.is_set():
                try:
                    # Проверяем необходимость оптимизации
                    self._check_optimization_needed()

                    # Задержка перед следующей проверкой
                    self.stop_event.wait(self.optimization_interval)
                except Exception as e:
                    self.logger.error(f"Ошибка в фоновом потоке оптимизации: {e}")
                    time.sleep(5)  # Пауза перед повторной попыткой

        self.optimization_thread = threading.Thread(
            target=optimization_loop,
            name="OptimizationMonitor",
            daemon=True
        )
        self.optimization_thread.start()
        self.logger.info("Запущен фоновый мониторинг оптимизации системы")

    def stop_optimization_monitor(self):
        """Останавливает фоновый мониторинг оптимизации."""
        self.stop_event.set()
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=2.0)
        self.logger.info("Фоновый мониторинг оптимизации системы остановлен")

    def enter_power_saving_mode(self):
        """Переходит в режим энергосбережения."""
        self.logger.info("Переход в режим энергосбережения")
        self.power_saving_mode = True
        self.current_optimization_level = "power_saving"

        # Выполняем оптимизацию для режима энергосбережения
        self._apply_power_saving_optimizations()

    def exit_power_saving_mode(self):
        """Выходит из режима энергосбережения."""
        if not self.power_saving_mode:
            return

        self.logger.info("Выход из режима энергосбережения")
        self.power_saving_mode = False
        self.current_optimization_level = "normal"

        # Восстанавливаем нормальные настройки
        self._restore_normal_operations()

    def _check_optimization_needed(self):
        """Проверяет, требуется ли оптимизация системы."""
        try:
            # Получаем данные о состоянии системы
            system_health = self._get_system_health()

            # Проверяем использование ресурсов
            if system_health["resource_usage"]["memory_percent"] > 85:
                self._optimize_for_memory()
            elif system_health["resource_usage"]["cpu_percent"] > 90:
                self._optimize_for_cpu()

            # Проверяем активность пользователя
            if self._is_user_inactive():
                self.enter_power_saving_mode()
            else:
                self.exit_power_saving_mode()
        except Exception as e:
            self.logger.error(f"Ошибка проверки необходимости оптимизации: {e}")

    def _get_system_health(self) -> Dict[str, Any]:
        """Получает информацию о состоянии системы, используя psutil."""
        try:
            vmem = psutil.virtual_memory()
            active_components = 0
            if hasattr(self.core_brain, 'components') and self.core_brain.components:
                try:
                    active_components = len(self.core_brain.components)
                except (TypeError, AttributeError):
                    pass
            
            health = {
                "status": "healthy",
                "resource_usage": {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": vmem.percent,
                    "memory_available_gb": round(vmem.available / (1024**3), 2),
                    "gpu_usage": 0
                },
                "active_components": active_components,
                "system_load": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }

            if hasattr(self.core_brain, 'system_monitor') and self.core_brain.system_monitor:
                monitor_health = self.core_brain.system_monitor.get_system_status()
                health.update(monitor_health)
                # Перезаписываем данные о ресурсах более точными данными от psutil
                health["resource_usage"]["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                health["resource_usage"]["memory_percent"] = vmem.percent

            return health
        except Exception as e:
            self.logger.error(f"Ошибка получения состояния системы: {e}")
            return {
                "status": "unknown",
                "resource_usage": {
                    "cpu_percent": 50,
                    "memory_percent": 50,
                    "gpu_usage": 0
                },
                "active_components": 0,
                "system_load": 0.5
            }

    def _is_user_inactive(self) -> bool:
        """Проверяет, неактивен ли пользователь."""
        try:
            # Здесь должна быть интеграция с системой отслеживания активности
            # Для упрощения возвращаем заглушку
            return False
        except Exception as e:
            self.logger.debug(f"Ошибка проверки активности пользователя: {e}")
            return False

    def _optimize_for_memory(self):
        """Оптимизирует систему для экономии памяти."""
        self.logger.info("Оптимизация использования памяти...")
        try:
            # Очищаем неиспользуемые кэши
            self._clear_inactive_caches()

            # Сжимаем данные в памяти
            self._compress_memory_data()

            # Выгружаем неактивные модули
            self._unload_inactive_modules()

            # Обновляем уровень оптимизации
            self.current_optimization_level = "memory_optimized"

            # Записываем результаты
            self._record_optimization("memory", "optimized")
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации памяти: {e}")

    def _optimize_for_cpu(self):
        """Оптимизирует систему для экономии CPU."""
        self.logger.info("Оптимизация использования CPU...")
        try:
            # Уменьшаем количество фоновых задач
            self._reduce_background_tasks()

            # Оптимизируем алгоритмы
            self._optimize_algorithms()

            # Обновляем уровень оптимизации
            self.current_optimization_level = "cpu_optimized"

            # Записываем результаты
            self._record_optimization("cpu", "optimized")
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации CPU: {e}")

    def _apply_power_saving_optimizations(self):
        """Применяет оптимизации для режима энергосбережения."""
        self.logger.info("Применение оптимизаций для режима энергосбережения")
        try:
            # Снижаем частоту обновления
            self._reduce_update_frequency()

            # Отключаем не критические функции
            self._disable_non_critical_features()

            # Уменьшаем качество обработки
            self._reduce_processing_quality()

            # Записываем результаты
            self._record_optimization("power_saving", "activated")
        except Exception as e:
            self.logger.error(f"Ошибка применения оптимизаций энергосбережения: {e}")

    def _restore_normal_operations(self):
        """Восстанавливает нормальные операции после режима энергосбережения."""
        self.logger.info("Восстановление нормальных операций")
        try:
            # Восстанавливаем частоту обновления
            self._restore_update_frequency()

            # Включаем все функции
            self._enable_all_features()

            # Восстанавливаем качество обработки
            self._restore_processing_quality()

            # Записываем результаты
            self._record_optimization("power_saving", "deactivated")
        except Exception as e:
            self.logger.error(f"Ошибка восстановления нормальных операций: {e}")

    def _clear_inactive_caches(self):
        """Очищает неиспользуемые кэши."""
        try:
            if hasattr(self.core_brain, 'token_cache') and self.core_brain.token_cache and hasattr(self.core_brain.token_cache, 'clear_inactive'):
                self.core_brain.token_cache.clear_inactive()

            if hasattr(self.core_brain, 'memory_manager') and self.core_brain.memory_manager and hasattr(self.core_brain.memory_manager, 'clear_inactive_caches'):
                self.core_brain.memory_manager.clear_inactive_caches()
            else:
                self.logger.debug("Memory manager or clear_inactive_caches method not available.")
        except Exception as e:
            self.logger.error(f"Ошибка очистки неактивных кэшей: {e}")

    def _compress_memory_data(self):
        """Сжимает данные в памяти."""
        try:
            if hasattr(self.core_brain, 'memory_manager') and self.core_brain.memory_manager and hasattr(self.core_brain.memory_manager, 'compress_data'):
                self.core_brain.memory_manager.compress_data()
            else:
                self.logger.debug("Memory manager or compress_data method not available.")
        except Exception as e:
            self.logger.error(f"Ошибка сжатия данных в памяти: {e}")

    def _unload_inactive_modules(self):
        """Выгружает неактивные модули."""
        try:
            # Проверяем наличие component_initializer и метода unload_inactive_modules
            if hasattr(self.core_brain, 'component_initializer') and self.core_brain.component_initializer is not None:
                if hasattr(self.core_brain.component_initializer, 'unload_inactive_modules'):
                    # Используем отложенную команду, если доступна система отложенных команд
                    if hasattr(self.core_brain, 'deferred_system') and self.core_brain.deferred_system is not None:
                        self.logger.debug("Использование отложенной команды для выгрузки неактивных модулей")
                        self.core_brain.deferred_system.defer_command(
                            self.core_brain.component_initializer.unload_inactive_modules,
                            priority='low',
                            description="Выгрузка неактивных модулей"
                        )
                    else:
                        # Прямой вызов, если система отложенных команд недоступна
                        self.core_brain.component_initializer.unload_inactive_modules()
                    self.logger.debug("Запрос на выгрузку неактивных модулей выполнен")
                else:
                    self.logger.warning("Метод unload_inactive_modules недоступен в component_initializer")
            else:
                self.logger.debug("component_initializer не доступен для выгрузки модулей")
        except Exception as e:
            self.logger.warning(f"Не удалось выгрузить неактивные модули: {e}", exc_info=True)

    def _reduce_background_tasks(self):
        """Снижает количество фоновых задач."""
        try:
            # FIX: Check if deferred_system attribute exists before accessing
            deferred_system = getattr(self.core_brain, 'deferred_system', None)
            if deferred_system:
                deferred_system.reduce_background_tasks()
        except Exception as e:
            self.logger.error(f"Ошибка снижения фоновых задач: {e}")

    def _optimize_algorithms(self):
        """Оптимизирует алгоритмы для снижения нагрузки на CPU."""
        try:
            # Здесь должна быть логика оптимизации алгоритмов
            pass
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации алгоритмов: {e}")

    def _reduce_update_frequency(self):
        """Снижает частоту обновления систем."""
        try:
            # FIX: Check attribute exists with hasattr BEFORE accessing
            fractal_attention = getattr(self.core_brain, 'fractal_attention', None)
            if fractal_attention and hasattr(fractal_attention, 'set_update_interval'):
                fractal_attention.set_update_interval(5.0)  # Увеличиваем интервал

            system_monitor = getattr(self.core_brain, 'system_monitor', None)
            if system_monitor and hasattr(system_monitor, 'set_update_interval'):
                system_monitor.set_update_interval(10.0)
        except Exception as e:
            self.logger.error(f"Ошибка снижения частоты обновления: {e}")

    def _disable_non_critical_features(self):
        """Отключает не критические функции."""
        try:
            # Placeholder for disabling features through core_brain
            pass
        except Exception as e:
            self.logger.error(f"Ошибка отключения некритических функций: {e}")

    def _reduce_processing_quality(self):
        """Уменьшает качество обработки для экономии ресурсов."""
        try:
            # FIX: Check attribute exists with hasattr BEFORE accessing
            generation_coordinator = getattr(self.core_brain, 'generation_coordinator', None)
            if generation_coordinator and hasattr(generation_coordinator, 'set_quality_level'):
                generation_coordinator.set_quality_level("low")
        except Exception as e:
            self.logger.error(f"Ошибка снижения качества обработки: {e}")

    def _restore_update_frequency(self):
        """Восстанавливает нормальную частоту обновления."""
        try:
            # FIX: Check attribute exists with hasattr BEFORE accessing
            fractal_attention = getattr(self.core_brain, 'fractal_attention', None)
            if fractal_attention and hasattr(fractal_attention, 'set_update_interval'):
                fractal_attention.set_update_interval(1.0)

            system_monitor = getattr(self.core_brain, 'system_monitor', None)
            if system_monitor and hasattr(system_monitor, 'set_update_interval'):
                system_monitor.set_update_interval(2.0)
        except Exception as e:
            self.logger.error(f"Ошибка восстановления частоты обновления: {e}")

    def _enable_all_features(self):
        """Включает все функции системы."""
        try:
            # Placeholder for enabling features through core_brain
            pass
        except Exception as e:
            self.logger.error(f"Ошибка включения всех функций: {e}")

    def _restore_processing_quality(self):
        """Восстанавливает нормальное качество обработки."""
        try:
            # FIX: Check attribute exists with hasattr BEFORE accessing
            generation_coordinator = getattr(self.core_brain, 'generation_coordinator', None)
            if generation_coordinator and hasattr(generation_coordinator, 'set_quality_level'):
                generation_coordinator.set_quality_level("high")
        except Exception as e:
            self.logger.error(f"Ошибка восстановления качества обработки: {e}")

    def _record_optimization(self, optimization_type: str, status: str):
        """Записывает результат оптимизации в историю."""
        self.optimization_history.append({
            "type": optimization_type,
            "status": status,
            "timestamp": time.time(),
            "level": self.current_optimization_level
        })
        self.logger.info(f"Оптимизация {optimization_type} {status}")

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Возвращает историю оптимизаций."""
        return self.optimization_history

    def get_current_optimization_level(self) -> str:
        """Возвращает текущий уровень оптимизации."""
        return self.current_optimization_level

    def is_in_power_saving_mode(self) -> bool:
        """Проверяет, находится ли система в режиме энергосбережения."""
        return self.power_saving_mode
