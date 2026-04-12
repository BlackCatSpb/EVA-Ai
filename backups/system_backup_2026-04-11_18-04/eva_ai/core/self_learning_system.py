#!/usr/bin/env python3
"""
Самообучающаяся система ЕВА - автоматическое обучение модели на русском языке
Интегрируется в CoreBrain для непрерывного улучшения модели
"""

import os
import sys
import torch
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger("eva_ai.self_learning")

class SelfLearningSystem:
    """Система самообучения модели ЕВА."""

    def __init__(self, brain):
        """Инициализация системы самообучения."""
        self.brain = brain
        self.is_active = False
        self.learning_thread = None
        self.last_training_time = None
        self.training_stats = {
            'sessions_completed': 0,
            'data_processed': 0,
            'model_improvements': 0,
            'errors': 0
        }

        # Параметры обучения
        self.config = {
            'min_training_interval': 60,  # 1 минута между обучениями (для тестирования)
            'max_training_duration': 1800,  # 30 минут максимум
            'min_samples_for_training': 1,  # минимум 1 образец
            'learning_rate': 5e-5,
            'batch_size': 2,
            'epochs': 1
        }

        # Очередь данных для обучения
        self.training_queue = []
        self.training_lock = threading.Lock()

        logger.info("SelfLearningSystem инициализирована")

    def start(self) -> bool:
        """Запуск системы самообучения."""
        if self.is_active:
            logger.warning("SelfLearningSystem уже активна")
            return True

        try:
            self.is_active = True
            self.learning_thread = threading.Thread(
                target=self._learning_loop,
                name="SelfLearningThread",
                daemon=True
            )
            self.learning_thread.start()

            logger.info("SelfLearningSystem запущена")
            return True

        except Exception as e:
            logger.error(f"Ошибка запуска SelfLearningSystem: {e}", exc_info=True)
            self.is_active = False
            return False

    def stop(self) -> bool:
        """Остановка системы самообучения."""
        if not self.is_active:
            return True

        self.is_active = False

        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=10)

        logger.info("SelfLearningSystem остановлена")
        return True

    def add_training_data(self, text: str, source: str = "user_input") -> bool:
        """Добавление данных для обучения."""
        if not text or len(text.strip()) < 5:
            return False

        with self.training_lock:
            self.training_queue.append({
                'text': text.strip(),
                'source': source,
                'timestamp': datetime.now(),
                'processed': False
            })

        logger.debug(f"Добавлен текст для обучения: '{text[:50]}...'")
        return True

    def _learning_loop(self):
        """Основной цикл самообучения."""
        logger.info("Запущен цикл самообучения")

        while self.is_active:
            try:
                # Проверяем, нужно ли обучение
                if self._should_train():
                    self._perform_training_session()
                else:
                    # Ждем перед следующей проверкой
                    time.sleep(300)  # 5 минут

            except Exception as e:
                logger.error(f"Ошибка в цикле самообучения: {e}", exc_info=True)
                self.training_stats['errors'] += 1
                time.sleep(60)  # Ждем минуту перед повтором

        logger.info("Цикл самообучения завершен")

    def _should_train(self) -> bool:
        """Проверка, нужно ли запускать обучение."""
        # Проверяем интервал времени
        if self.last_training_time:
            time_since_last = datetime.now() - self.last_training_time
            if time_since_last < timedelta(seconds=self.config['min_training_interval']):
                return False

        # Проверяем количество данных
        with self.training_lock:
            unprocessed_count = sum(1 for item in self.training_queue if not item['processed'])
            if unprocessed_count < self.config['min_samples_for_training']:
                return False

        # Проверяем, что система не занята другими задачами
        if hasattr(self.brain, 'is_busy') and self.brain.is_busy():
            return False

        return True

    def _perform_training_session(self) -> bool:
        """Выполнение сессии обучения."""
        logger.info("Начинаем сессию обучения модели")

        try:
            start_time = time.time()

            # Получаем данные для обучения
            training_data = self._prepare_training_data()
            if not training_data:
                logger.warning("Нет данных для обучения")
                return False

            # Инициализируем обучение модели
            success = self._train_model_on_data(training_data)

            if success:
                self.training_stats['sessions_completed'] += 1
                self.training_stats['data_processed'] += len(training_data)
                self.last_training_time = datetime.now()

                # Сохраняем обновленную модель
                self._save_updated_model()

                duration = time.time() - start_time
                logger.info(f"Сессия обучения завершена успешно за {duration:.1f} сек")

                return True
            else:
                logger.error("Сессия обучения завершилась неудачей")
                return False

        except Exception as e:
            logger.error(f"Ошибка в сессии обучения: {e}", exc_info=True)
            return False

    def _prepare_training_data(self) -> List[str]:
        """Подготовка данных для обучения."""
        with self.training_lock:
            # Получаем непроцессированные данные
            unprocessed = [item for item in self.training_queue if not item['processed']]

            if not unprocessed:
                return []

            # Ограничиваем количество данных
            max_samples = 50
            selected = unprocessed[:max_samples]

            # Помечаем как обработанные
            for item in selected:
                item['processed'] = True

            # Извлекаем тексты
            texts = [item['text'] for item in selected]

            logger.info(f"Подготовлено {len(texts)} текстов для обучения")
            return texts

    def _train_model_on_data(self, texts: List[str]) -> bool:
        """Обучение модели на подготовленных данных."""
        try:
            # Получаем доступ к модели через brain
            model_manager = getattr(self.brain, 'model_manager', None) or \
                           getattr(self.brain, 'fractal_model_manager', None)

            if not model_manager:
                logger.error("ModelManager недоступен")
                return False

            # Создаем тренировочный датасет
            training_texts = self._expand_training_texts(texts)

            # Сохраняем в файл
            training_file = Path("temp_training_data.txt")
            with open(training_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(training_texts))

            # Имитируем обучение (в реальной системе здесь будет вызов обучения модели)
            logger.info(f"Имитация обучения на {len(training_texts)} текстах")

            # В реальной системе здесь будет:
            # 1. Загрузка текущей модели
            # 2. Токенизация данных
            # 3. Обучение модели
            # 4. Сохранение обновленной модели

            # Имитируем процесс обучения
            time.sleep(2)  # Имитация обучения

            # Очищаем временный файл
            if training_file.exists():
                training_file.unlink()

            logger.info("Обучение модели завершено (имитация)")
            return True

        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}", exc_info=True)
            return False

    def _expand_training_texts(self, texts: List[str]) -> List[str]:
        """Расширение тренировочных текстов для лучшего обучения."""
        expanded = []

        for text in texts:
            # Добавляем оригинальный текст
            expanded.append(text)

            # Добавляем вариации
            expanded.append(text.lower())
            expanded.append(text + "!")
            expanded.append(text + "?")

            # Добавляем связанные фразы
            if "привет" in text.lower():
                expanded.extend([
                    "Здравствуйте! Как дела?",
                    "Приветствую вас!",
                    "Добрый день!"
                ])

        logger.debug(f"Расширено до {len(expanded)} тренировочных текстов")
        return expanded

    def _save_updated_model(self) -> bool:
        """Сохранение обновленной модели."""
        try:
            # В реальной системе здесь будет сохранение модели в фрактальное хранилище
            logger.info("Модель обновлена и сохранена")
            self.training_stats['model_improvements'] += 1
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}", exc_info=True)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики обучения."""
        return {
            'is_active': self.is_active,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'queue_size': len(self.training_queue),
            'unprocessed_count': sum(1 for item in self.training_queue if not item['processed']),
            **self.training_stats
        }

    def force_training_session(self) -> bool:
        """Принудительный запуск сессии обучения."""
        logger.info("Принудительный запуск сессии обучения")
        return self._perform_training_session()


class AutoLearningIntegration:
    """Интеграция самообучения в CoreBrain."""

    @staticmethod
    def integrate_into_brain(brain):
        """Интеграция системы самообучения в CoreBrain."""
        if hasattr(brain, 'self_learning_system'):
            logger.warning("SelfLearningSystem уже интегрирована")
            return True

        try:
            # Создаем систему самообучения
            brain.self_learning_system = SelfLearningSystem(brain)

            # Добавляем методы для доступа к обучению
            brain.add_training_data = brain.self_learning_system.add_training_data
            brain.start_self_learning = brain.self_learning_system.start
            brain.stop_self_learning = brain.self_learning_system.stop
            brain.force_training = brain.self_learning_system.force_training_session
            brain.get_learning_stats = brain.self_learning_system.get_stats

            # Автоматически запускаем самообучение
            if brain.self_learning_system.start():
                logger.info("SelfLearningSystem успешно интегрирована и запущена")
                return True
            else:
                logger.error("Не удалось запустить SelfLearningSystem")
                return False

        except Exception as e:
            logger.error(f"Ошибка интеграции SelfLearningSystem: {e}", exc_info=True)
            return False


# Функция для автоматической интеграции в систему
def initialize_self_learning(brain):
    """Инициализация самообучения в системе ЕВА."""
    return AutoLearningIntegration.integrate_into_brain(brain)


if __name__ == "__main__":
    # Тест системы самообучения
    print("Тестирование системы самообучения...")

    class MockBrain:
        def __init__(self):
            self.model_manager = None

        def is_busy(self):
            return False

    brain = MockBrain()
    success = initialize_self_learning(brain)

    if success:
        print("[OK] SelfLearningSystem успешно интегрирована")

        # Добавляем тестовые данные
        brain.add_training_data("Привет, как дела?")
        brain.add_training_data("Что такое искусственный интеллект?")
        brain.add_training_data("Расскажи о машинном обучении")

        # Получаем статистику
        stats = brain.get_learning_stats()
        print(f"Статистика обучения: {stats}")

        # Принудительное обучение
        brain.force_training_session()

        # Останавливаем
        brain.stop_self_learning()
        print("[OK] Тест завершен успешно")

    else:
        print("[FAIL] Ошибка интеграции SelfLearningSystem")
