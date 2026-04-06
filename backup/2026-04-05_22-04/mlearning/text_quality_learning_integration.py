"""
Интеграция новой системы улучшения генерации с существующим модулем обучения GUI
"""
from __future__ import annotations

import logging
import os
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("eva.learning_integration")


class TextQualityLearningIntegration:
    """Интеграция системы улучшения генерации с GUI модулем обучения"""
    
    def __init__(self, learning_module, fractal_model_manager):
        self.learning_module = learning_module
        self.fractal_model_manager = fractal_model_manager
        
        # Состояние интеграции
        self.integration_active = False
        self.last_quality_check = 0
        self.auto_improvement_enabled = True
        self.quality_threshold = 0.7  # Порог для автоматического улучшения
        
        logger.info("TextQualityLearningIntegration инициализирована")
    
    def activate_integration(self):
        """Активирует интеграцию улучшения генерации с обучением"""
        try:
            # Проверяем наличие необходимых компонентов
            if not self._check_components():
                return False
            
            # Расширяем GUI модуль обучения функционалом улучшения генерации
            self._extend_learning_module()
            
            # Добавляем мониторинг качества генерации
            self._setup_quality_monitoring()
            
            # Настраиваем автоматическое улучшение
            self._setup_auto_improvement()
            
            self.integration_active = True
            logger.info("Интеграция улучшения генерации активирована")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка активации интеграции: {e}", exc_info=True)
            return False
    
    def _check_components(self) -> bool:
        """Проверяет наличие необходимых компонентов и создает их при необходимости"""
        components_ok = True
        
        # Проверяем FractalModelManager
        if not self.fractal_model_manager:
            logger.warning("FractalModelManager недоступен")
            components_ok = False
        
        # Создаем или проверяем наличие тренера
        try:
            if not hasattr(self.fractal_model_manager, 'trainer') or not self.fractal_model_manager.trainer:
                if hasattr(self.fractal_model_manager, 'model') and self.fractal_model_manager.model:
                    from .text_quality_trainer import TextQualityTrainer, TrainingConfig
                    self.fractal_model_manager.trainer = TextQualityTrainer(
                        model=self.fractal_model_manager.model,
                        tokenizer=self.fractal_model_manager.tokenizer,
                        config=TrainingConfig(num_epochs=50)
                    )
                    logger.info("TextQualityTrainer создан")
        except Exception as e:
            logger.debug(f"TextQualityTrainer недоступен: {e}")
        
        # Создаем или проверяем наличие улучшателя качества
        try:
            if not hasattr(self.fractal_model_manager, 'quality_improver') or not self.fractal_model_manager.quality_improver:
                from .text_quality_improver import TextQualityImprover
                self.fractal_model_manager.quality_improver = TextQualityImprover(None)
                logger.info("TextQualityImprover создан")
        except Exception as e:
            logger.debug(f"TextQualityImprover недоступен: {e}")
        
        return components_ok
    
    def _ensure_components(self) -> bool:
        """Гарантирует наличие необходимых компонентов перед использованием"""
        try:
            # Создаем тренер если нужно
            if not hasattr(self.fractal_model_manager, 'trainer') or not self.fractal_model_manager.trainer:
                if hasattr(self.fractal_model_manager, 'model') and self.fractal_model_manager.model:
                    from .text_quality_trainer import TextQualityTrainer, TrainingConfig
                    self.fractal_model_manager.trainer = TextQualityTrainer(
                        model=self.fractal_model_manager.model,
                        tokenizer=self.fractal_model_manager.tokenizer,
                        config=TrainingConfig(num_epochs=50)
                    )
                    logger.info("TextQualityTrainer создан по требованию")
            
            # Создаем улучшатель если нужно
            if not hasattr(self.fractal_model_manager, 'quality_improver') or not self.fractal_model_manager.quality_improver:
                from .text_quality_improver import TextQualityImprover
                self.fractal_model_manager.quality_improver = TextQualityImprover(None)
                logger.info("TextQualityImprover создан по требованию")
            
            return True
        except Exception as e:
            logger.warning(f"Не удалось создать компоненты: {e}")
            return False
    
    def check_components_availability(self) -> bool:
        """Публичный метод для проверки доступности компонентов"""
        return self._ensure_components()
    
    def _extend_learning_module(self):
        """Расширяет GUI модуль обучения функционалом улучшения генерации"""
        try:
            # Добавляем новые элементы в интерфейс обучения
            self._add_quality_controls()
            self._add_model_improvement_section()
            self._add_real_time_quality_monitoring()
            
            logger.info("GUI модуль обучения расширен функционалом улучшения генерации")
            
        except Exception as e:
            logger.error(f"Ошибка расширения модуля обучения: {e}", exc_info=True)
    
    def _add_quality_controls(self):
        """Добавляет элементы управления качеством генерации"""
        if not hasattr(self.learning_module, 'learning_frame'):
            return
        
        # Создаем секцию качества генерации
        quality_frame = ttk.LabelFrame(
            self.learning_module.learning_frame, 
            text="🎯 Качество генерации",
            padding="10"
        )
        quality_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Метрики качества
        metrics_frame = ttk.Frame(quality_frame)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Кнопка проверки качества
        check_quality_btn = ttk.Button(
            metrics_frame,
            text="🔍 Проверить качество",
            command=self._check_model_quality
        )
        check_quality_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка улучшения
        improve_btn = ttk.Button(
            metrics_frame,
            text="🚀 Улучшить модель",
            command=self._improve_model_quality
        )
        improve_btn.pack(side=tk.LEFT, padx=5)
        
        # Автоматическое улучшение
        auto_var = tk.BooleanVar(value=self.auto_improvement_enabled)
        auto_check = ttk.Checkbutton(
            metrics_frame,
            text="🔄 Автоулучшение",
            variable=auto_var,
            command=self._toggle_auto_improvement
        )
        auto_check.pack(side=tk.LEFT, padx=5)
        
        # Сохраняем ссылки на виджеты
        self.quality_check_btn = check_quality_btn
        self.improve_btn = improve_btn
        self.auto_var = auto_var
        
        # Область отображения метрик
        self.quality_metrics_frame = ttk.Frame(quality_frame)
        self.quality_metrics_frame.pack(fill=tk.X, pady=5)
        
        # Метрики
        self.quality_labels = {}
        metrics = ["overall", "coherence", "diversity", "grammar"]
        for metric in metrics:
            frame = ttk.Frame(self.quality_metrics_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"📊 {metric.title()}:").pack(side=tk.LEFT)
            label = ttk.Label(frame, text="---")
            label.pack(side=tk.RIGHT, padx=(10, 0))
            
            self.quality_labels[metric] = label
    
    def _add_model_improvement_section(self):
        """Добавляет секцию улучшения модели"""
        if not hasattr(self.learning_module, 'learning_frame'):
            return
        
        improvement_frame = ttk.LabelFrame(
            self.learning_module.learning_frame,
            text="🧠 Улучшение модели",
            padding="10"
        )
        improvement_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Параметры обучения
        params_frame = ttk.Frame(improvement_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Количество эпох
        ttk.Label(params_frame, text="Эпохи:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.epochs_var = tk.IntVar(value=2)
        epochs_spin = ttk.Spinbox(
            params_frame, 
            from_=1, to=1000, 
            textvariable=self.epochs_var,
            width=10
        )
        epochs_spin.grid(row=0, column=1, padx=5)
        
        # Скорость обучения
        ttk.Label(params_frame, text="Скорость:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.learning_rate_var = tk.StringVar(value="3e-5")
        rate_combo = ttk.Combobox(
            params_frame,
            textvariable=self.learning_rate_var,
            values=["1e-5", "3e-5", "5e-5", "1e-4"],
            width=10
        )
        rate_combo.grid(row=0, column=3, padx=5)
        
        # Кнопка запуска улучшения
        start_improvement_btn = ttk.Button(
            improvement_frame,
            text="🚀 Запустить улучшение",
            command=self._start_model_improvement
        )
        start_improvement_btn.pack(pady=10)
        
        # Прогресс бар
        improvement_progress = ttk.Progressbar(
            improvement_frame,
            mode='determinate',
            length=300
        )
        improvement_progress.pack(fill=tk.X, pady=5)
        
        # Сохраняем ссылки
        self.epochs_var = self.epochs_var
        self.learning_rate_var = self.learning_rate_var
        self.improvement_progress = improvement_progress
    
    def _add_real_time_quality_monitoring(self):
        """Добавляет мониторинг качества в реальном времени"""
        if not hasattr(self.learning_module, 'learning_frame'):
            return
        
        monitoring_frame = ttk.LabelFrame(
            self.learning_module.learning_frame,
            text="📈 Мониторинг качества",
            padding="10"
        )
        monitoring_frame.pack(fill=tk.X, pady=(10, 0))
        
        # График качества
        self.quality_canvas_frame = ttk.Frame(monitoring_frame)
        self.quality_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Статистика
        stats_frame = ttk.Frame(monitoring_frame)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.monitoring_stats = {}
        stats = [
            ("Запросов обработано", "total_requests"),
            ("Среднее качество", "avg_quality"),
            ("Улучшений модели", "model_improvements"),
            ("Последнее улучшение", "last_improvement")
        ]
        
        for i, (label, key) in enumerate(stats):
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            stat_label = ttk.Label(frame, text="0")
            stat_label.pack(side=tk.RIGHT, padx=(10, 0))
            
            self.monitoring_stats[key] = stat_label
    
    def _setup_quality_monitoring(self):
        """Настраивает фоновое мониторинг качества"""
        def monitor_quality():
            if not self.integration_active:
                return
            
            try:
                # Проверяем качество каждые 30 секунд
                current_time = time.time()
                if current_time - self.last_quality_check > 30:
                    self._check_model_quality()
                    self.last_quality_check = current_time
                    
                    # Автоматическое улучшение если качество низкое
                    if self.auto_improvement_enabled:
                        self._check_auto_improvement()
                        
            except Exception as e:
                logger.error(f"Ошибка мониторинга качества: {e}")
        
        # Запускаем в фоновом потоке
        import threading
        self.monitoring_thread = threading.Thread(target=monitor_quality, daemon=True)
        self.monitoring_thread.start()
    
    def _setup_auto_improvement(self):
        """Настраивает автоматическое улучшение"""
        def check_auto():
            if not self.auto_improvement_enabled:
                return
            
            try:
                self._check_auto_improvement()
            except Exception as e:
                logger.error(f"Ошибка автоулучшения: {e}")
        
        # Проверяем каждые 5 минут
        import threading
        self.auto_improvement_thread = threading.Thread(target=check_auto, daemon=True)
        self.auto_improvement_thread.start()
    
    def _check_model_quality(self):
        """Проверяет и отображает качество модели"""
        try:
            if not self.fractal_model_manager:
                return
            
            # Получаем метрики качества
            metrics = self.fractal_model_manager.get_quality_metrics()
            
            # Обновляем метрики в GUI
            if hasattr(self, 'quality_labels'):
                for metric, label in self.quality_labels.items():
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, float):
                            label.config(text=f"{value:.3f}")
                        elif isinstance(value, bool):
                            label.config(text="✅" if value else "❌")
                        else:
                            label.config(text=str(value))
            
            # Обновляем статистику мониторинга
            if hasattr(self, 'monitoring_stats'):
                # Здесь можно обновить статистику
                pass
            
            logger.info(f"Качество модели проверено: {metrics}")
            
        except Exception as e:
            logger.error(f"Ошибка проверки качества: {e}")
    
    def _improve_model_quality(self):
        """Запускает улучшение качества модели"""
        try:
            # Создаем компоненты если нужно
            self._ensure_components()
            
            if not self.fractal_model_manager:
                self._show_error("Ошибка", "FractalModelManager недоступен")
                return
            
            if not hasattr(self.fractal_model_manager, 'model') or not self.fractal_model_manager.model:
                self._show_error("Ошибка", "Модель недоступна")
                return
            
            # Показываем прогресс
            if hasattr(self, 'improvement_progress'):
                self.improvement_progress['value'] = 0
                self.improvement_progress['maximum'] = 100
            
            # Получаем параметры обучения
            epochs = self.epochs_var.get() if hasattr(self, 'epochs_var') else 2
            learning_rate = self.learning_rate_var.get() if hasattr(self, 'learning_rate_var') else "3e-5"
            
            # Запускаем улучшение в фоновом потоке
            def improve_in_background():
                try:
                    config = {
                        'learning_rate': float(learning_rate),
                        'num_epochs': epochs,
                        'batch_size': 2,
                        'max_length': 128
                    }
                    
                    # Создаем временный тренер с новыми параметрами
                    from .text_quality_trainer import TextQualityTrainer, TrainingConfig
                    
                    training_config = TrainingConfig(**config)
                    trainer = TextQualityTrainer(
                        model=self.fractal_model_manager.model,
                        tokenizer=self.fractal_model_manager.tokenizer,
                        config=training_config
                    )
                    
                    # Запускаем обучение
                    result = trainer.train()
                    
                    # Обновляем прогресс
                    if hasattr(self, 'improvement_progress'):
                        self.improvement_progress['value'] = 100
                    
                    # Обновляем модель
                    if result['status'] == 'success':
                        logger.info("Модель успешно улучшена")
                    
                except Exception as e:
                    logger.error(f"Ошибка улучшения модели: {e}")
                    if hasattr(self, 'improvement_progress'):
                        self.improvement_progress['value'] = 0
            
            import threading
            improvement_thread = threading.Thread(target=improve_in_background, daemon=True)
            improvement_thread.start()
            
            logger.info("Улучшение модели запущено в фоне")
            
        except Exception as e:
            logger.error(f"Ошибка запуска улучшения: {e}")
    
    def _start_model_improvement(self):
        """Запускает процесс улучшения модели"""
        self._improve_model_quality()
    
    def _check_auto_improvement(self):
        """Проверяет необходимость автоматического улучшения"""
        try:
            # Создаем компоненты если нужно
            if not self._ensure_components():
                return
            
            if not self.fractal_model_manager:
                return
            
            # Тестируем генерацию на простом запросе
            test_query = "Привет, как дела?"
            response = self.fractal_model_manager.generate_response(test_query, max_new_tokens=50)
            
            # Анализируем качество
            quality_metrics = self.fractal_model_manager.quality_improver.analyze_text_quality(response)
            overall_quality = quality_metrics.overall_score
            
            # Если качество ниже порога, запускаем улучшение
            if overall_quality < self.quality_threshold:
                logger.info(f"Качество {overall_quality:.2f} ниже порога {self.quality_threshold}, запускаем улучшение")
                self._improve_model_quality()
            else:
                logger.debug(f"Качество {overall_quality:.2f} в норме")
                
        except Exception as e:
            logger.error(f"Ошибка автоулучшения: {e}")
    
    def _toggle_auto_improvement(self):
        """Переключает автоматическое улучшение"""
        self.auto_improvement_enabled = self.auto_var.get()
        status = "включено" if self.auto_improvement_enabled else "выключено"
        logger.info(f"Автоулучшение {status}")
    
    def _show_error(self, title: str, message: str):
        """Показывает ошибку через GUI"""
        try:
            from tkinter import messagebox
            messagebox.showerror(title, message)
        except Exception as e:
            logger.error(f"Ошибка показа сообщения: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Возвращает статус интеграции"""
        return {
            "active": self.integration_active,
            "auto_improvement_enabled": self.auto_improvement_enabled,
            "last_quality_check": self.last_quality_check,
            "quality_threshold": self.quality_threshold,
            "components_available": self._check_components()
        }
    
    def deactivate_integration(self):
        """Деактивирует интеграцию"""
        self.integration_active = False
        logger.info("Интеграция улучшения генерации деактивирована")
    
    def activate(self):
        """Активирует интеграцию (метод для совместимости)"""
        try:
            self._extend_learning_module()
            self.integration_active = True
            logger.info("Интеграция улучшения генерации активирована")
            return True
        except Exception as e:
            logger.error(f"Ошибка активации интеграции: {e}")
            return False
    
    def deactivate(self):
        """Деактивирует интеграцию (метод для совместимости)"""
        self.deactivate_integration()
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус интеграции (метод для совместимости)"""
        return self.get_integration_status()
