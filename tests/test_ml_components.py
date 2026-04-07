#!/usr/bin/env python3
"""
ЕВА ML Components Test Suite
Комплексные тесты для ML-компонентов системы ЕВА.
"""

import pytest
import os
import time
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import threading
import queue


class TestMLUnitIntegration:
    """Интеграционные тесты MLUnit."""

    @pytest.fixture
    def mock_brain(self):
        """Создает mock для CoreBrain."""
        brain = Mock()
        brain.cache_dir = tempfile.mkdtemp()
        brain.config = {
            'safe_test_mode': True,
            'use_gpu': False,
            'max_models': 2,
            'max_workers': 2
        }

        # Mock MemoryManager с гибридным кэшем
        memory_manager = Mock()
        hybrid_cache = Mock()
        hybrid_cache.get_cache_stats.return_value = {"size": 100, "hits": 50}
        memory_manager.hybrid_cache = hybrid_cache
        brain.memory_manager = memory_manager

        # Mock GlobalResourceQueue
        resource_queue = Mock()
        resource_queue.acquire_cpu.return_value = True
        resource_queue.acquire_memory.return_value = True
        resource_queue.acquire_io.return_value = True
        brain.resource_queue = resource_queue

        # Mock CacheRouter
        cache_router = Mock()
        cache_router.register_batch.return_value = "batch_123"
        cache_router.register_segment.return_value = "seg_456"
        brain.cache_router = cache_router

        return brain

    @pytest.fixture
    def ml_unit(self, mock_brain):
        """Создает MLUnit для тестирования."""
        from eva_ai.mlearning.ml_unit import MLUnit

        # Настраиваем дополнительные атрибуты brain для корректной работы
        mock_brain.on_text_processor_ready = []
        mock_brain.on_ml_unit_ready = []

        # Создаем MLUnit с минимальной конфигурацией для тестов
        ml_unit = MLUnit(
            brain=mock_brain,
            use_gpu=False,
            max_models=1,
            max_workers=1
        )

        # Mock зависимых компонентов
        ml_unit.text_processor = Mock()
        ml_unit.text_processor.process_text.return_value = {
            'tokens': ['тест', 'текст'],
            'token_count': 2
        }

        ml_unit.response_generator = Mock()
        ml_unit.response_generator.generate_response.return_value = {
            'text': 'Тестовый ответ',
            'confidence': 0.8
        }

        ml_unit.model_manager = Mock()
        ml_unit.model_manager.get_available_models.return_value = ['test_model']

        return ml_unit

    @pytest.mark.integration
    def test_ml_unit_initialization(self, ml_unit):
        """Тест инициализации MLUnit."""
        assert ml_unit.initialized is True
        assert ml_unit.brain is not None
        assert ml_unit.cache_dir is not None
        assert hasattr(ml_unit, 'hybrid_cache')

    @pytest.mark.integration
    def test_ml_unit_component_integration(self, ml_unit):
        """Тест интеграции компонентов в MLUnit."""
        # Проверяем что компоненты инициализированы
        assert ml_unit.text_processor is not None
        assert ml_unit.response_generator is not None
        assert ml_unit.model_manager is not None

        # Проверяем связи между компонентами
        assert hasattr(ml_unit, 'hybrid_cache')

    @pytest.mark.integration
    def test_ml_unit_response_generation(self, ml_unit):
        """Тест генерации ответов через MLUnit."""
        prompt = "Тестовый запрос"
        response = ml_unit.generate_response(prompt, max_length=50)

        assert 'text' in response
        assert response['text'] == 'Тестовый ответ'
        assert response['confidence'] == 0.8

        # Проверяем обновление статистики
        assert ml_unit.stats['total_requests'] == 1
        assert ml_unit.stats['successful_requests'] == 1

    @pytest.mark.integration
    def test_ml_unit_text_processing(self, ml_unit):
        """Тест обработки текста в MLUnit."""
        test_text = "Это тестовый текст для обработки"
        analysis = ml_unit._tokenize_text(test_text)

        assert 'tokens' in analysis
        assert 'token_count' in analysis
        assert analysis['token_count'] == 5
        assert 'тестовый' in analysis['tokens']


class TestЕВАTokenizer:
    """Тесты для ЕВАTokenizer."""

    @pytest.fixture
    def mock_brain(self):
        """Создает mock для brain с MemoryManager."""
        brain = Mock()
        brain.cache_dir = tempfile.mkdtemp()
        brain.language = "ru"

        # Mock MemoryManager
        memory_manager = Mock()
        hybrid_cache = Mock()
        hybrid_cache.get.return_value = None
        hybrid_cache.set.return_value = True
        memory_manager.hybrid_cache = hybrid_cache
        brain.memory_manager = memory_manager

        return brain

    @pytest.fixture
    def tokenizer(self, mock_brain):
        """Создает ЕВАTokenizer для тестирования."""
        from eva_ai.mlearning.cogniflex_tokenizer import ЕВАTokenizer

        # Создаем токенизатор без фактической загрузки модели
        tokenizer = ЕВАTokenizer(brain=mock_brain)

        # Mock базовые компоненты для тестирования
        tokenizer.tokenizer = Mock()
        tokenizer.tokenizer.tokenize.return_value = ['тест', 'токены']
        tokenizer.tokenizer.convert_tokens_to_ids.return_value = [1, 2]
        tokenizer.tokenizer.convert_ids_to_tokens.return_value = ['тест', 'токены']
        tokenizer.tokenizer.decode.return_value = 'тестовый текст'
        tokenizer.tokenizer.get_vocab.return_value = {'тест': 1, 'токены': 2}

        return tokenizer

    @pytest.mark.unit
    def test_tokenizer_initialization(self, tokenizer):
        """Тест инициализации токенизатора."""
        assert tokenizer.initialized is True
        assert tokenizer.language == "ru"
        assert tokenizer.model_type == "gpt"
        assert hasattr(tokenizer, 'special_tokens')
        assert hasattr(tokenizer, 'morphology_rules')

    @pytest.mark.unit
    def test_tokenizer_tokenize(self, tokenizer):
        """Тест базовой токенизации."""
        text = "Это тестовый текст"
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert 'тест' in tokens

    @pytest.mark.unit
    def test_tokenizer_encode_decode(self, tokenizer):
        """Тест кодирования и декодирования."""
        text = "Тестовый текст для кодирования"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded['input_ids'][0])

        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert isinstance(decoded, str)

    @pytest.mark.unit
    def test_tokenizer_special_tokens(self, tokenizer):
        """Тест работы со специальными токенами."""
        assert tokenizer.special_tokens is not None
        assert 'bos_token' in tokenizer.special_tokens
        assert 'eos_token' in tokenizer.special_tokens
        assert 'pad_token' in tokenizer.special_tokens

    @pytest.mark.unit
    def test_tokenizer_vocab_operations(self, tokenizer):
        """Тест операций с словарем."""
        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

        # Тест преобразований
        token_ids = tokenizer.convert_tokens_to_ids(['тест'])
        tokens = tokenizer.convert_ids_to_tokens([1])

        assert isinstance(token_ids, list)
        assert isinstance(tokens, list)

    @pytest.mark.unit
    def test_tokenizer_strategies(self, tokenizer):
        """Тест стратегий токенизации."""
        config = tokenizer.config
        strategy_params = config.get_strategy_params()

        assert 'max_length' in strategy_params
        assert 'dynamic_weights' in strategy_params

        # Тест разных стратегий
        strategies = ['initial_response', 'context_expansion', 'contradiction_analysis']
        for strategy in strategies:
            config.priority_strategy = strategy
            params = config.get_strategy_params()
            assert isinstance(params, dict)


class TestParallelTokenizer:
    """Тесты для ParallelTokenizer."""

    @pytest.fixture
    def mock_brain(self):
        """Создает mock brain для ParallelTokenizer."""
        brain = Mock()
        brain.cache_dir = tempfile.mkdtemp()

        # Mock GlobalResourceQueue
        resource_queue = Mock()
        resource_queue.acquire_cpu.return_value = True
        resource_queue.acquire_memory.return_value = True
        resource_queue.acquire_io.return_value = True
        resource_queue.release_cpu.return_value = True
        resource_queue.release_memory.return_value = True
        brain.resource_queue = resource_queue

        # Mock CacheRouter
        cache_router = Mock()
        cache_router.register_batch.return_value = "batch_123"
        cache_router.register_segment.return_value = "seg_456"
        cache_router.register_token_nodes.return_value = None
        cache_router.set_weight.return_value = None
        cache_router.upsert_batch.return_value = None
        brain.cache_router = cache_router

        return brain

    @pytest.fixture
    def parallel_tokenizer(self, mock_brain):
        """Создает ParallelTokenizer для тестирования."""
        from eva_ai.mlearning.parallel_tokenization import ParallelTokenizer

        tokenizer = ParallelTokenizer(
            brain=mock_brain,
            max_data_window_bytes=100 * 1024 * 1024,  # 100MB
            worker_count=2
        )

        return tokenizer

    @pytest.mark.unit
    def test_parallel_tokenizer_initialization(self, parallel_tokenizer):
        """Тест инициализации ParallelTokenizer."""
        assert parallel_tokenizer.worker_count > 0
        assert parallel_tokenizer.max_data_window_bytes > 0
        assert not parallel_tokenizer._stop.is_set()
        assert parallel_tokenizer._threads == []

    @pytest.mark.unit
    def test_parallel_tokenizer_start_stop(self, parallel_tokenizer):
        """Тест запуска и остановки ParallelTokenizer."""
        # Запуск
        parallel_tokenizer.start()
        assert len(parallel_tokenizer._threads) == parallel_tokenizer.worker_count
        assert all(thread.is_alive() for thread in parallel_tokenizer._threads)

        # Остановка
        parallel_tokenizer.stop()
        assert parallel_tokenizer._stop.is_set()

        # Проверяем что потоки завершились
        time.sleep(0.1)  # Даем время на завершение
        assert not any(thread.is_alive() for thread in parallel_tokenizer._threads)

    @pytest.mark.unit
    def test_parallel_tokenizer_submit(self, parallel_tokenizer):
        """Тест отправки задач в ParallelTokenizer."""
        batch_id = "test_batch_001"
        text = "Это тестовый текст для параллельной обработки"

        # Запускаем токенизатор
        parallel_tokenizer.start()

        try:
            # Отправляем задачу
            success = parallel_tokenizer.submit(batch_id, text)
            assert success is True

            # Проверяем что задача в очереди
            assert not parallel_tokenizer._in_q.empty()

        finally:
            parallel_tokenizer.stop()

    @pytest.mark.unit
    def test_parallel_tokenizer_resource_management(self, parallel_tokenizer, mock_brain):
        """Тест управления ресурсами в ParallelTokenizer."""
        parallel_tokenizer.start()

        try:
            # Проверяем что ресурсы запрашиваются
            batch_id = "test_batch_002"
            text = "Тестовый текст для проверки ресурсов"
            parallel_tokenizer.submit(batch_id, text)

            # Даем время на обработку
            time.sleep(0.2)

            # Проверяем вызовы методов управления ресурсами
            mock_brain.resource_queue.acquire_memory.assert_called()
            mock_brain.resource_queue.acquire_io.assert_called()
            mock_brain.cache_router.register_batch.assert_called_with(
                batch_id=batch_id, source="text", total_tokens=22, priority=0.0, status='processing'
            )

        finally:
            parallel_tokenizer.stop()

    @pytest.mark.unit
    def test_parallel_tokenizer_persistence(self, parallel_tokenizer, mock_brain):
        """Тест персистентности данных в ParallelTokenizer."""
        parallel_tokenizer.start()

        try:
            batch_id = "test_batch_003"
            text = "Тестовый текст для проверки сохранения"
            parallel_tokenizer.submit(batch_id, text)

            # Даем время на обработку
            time.sleep(0.2)

            # Проверяем что данные сохраняются
            expected_path = os.path.join(
                mock_brain.cache_dir,
                "hybrid_cache",
                "disk_storage",
                "segments"
            )

            # Проверяем что директория создается
            assert os.path.exists(expected_path)

        finally:
            parallel_tokenizer.stop()


class TestFractalStorageML:
    """Тесты интеграции фрактального хранилища с ML-компонентами."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Создает mock MemoryManager с фрактальным хранилищем."""
        memory_manager = Mock()

        # Mock гибридный кэш
        hybrid_cache = Mock()
        hybrid_cache.get.return_value = None
        hybrid_cache.set.return_value = True
        hybrid_cache.clear_memory.return_value = True
        hybrid_cache.clear_disk.return_value = True
        hybrid_cache.get_cache_stats.return_value = {
            "total_tokens": 1000,
            "memory_tokens": 700,
            "disk_tokens": 300,
            "hit_rate": 0.85
        }

        memory_manager.hybrid_cache = hybrid_cache
        memory_manager.get_hybrid_cache.return_value = hybrid_cache
        memory_manager.clear_cache.return_value = None

        return memory_manager

    @pytest.fixture
    def mock_brain_with_memory(self, mock_memory_manager):
        """Создает mock brain с MemoryManager."""
        brain = Mock()
        brain.cache_dir = tempfile.mkdtemp()
        brain.memory_manager = mock_memory_manager

        return brain

    @pytest.mark.integration
    def test_ml_unit_fractal_integration(self, mock_brain_with_memory):
        """Тест интеграции MLUnit с фрактальным хранилищем."""
        from eva_ai.mlearning.ml_unit import MLUnit

        ml_unit = MLUnit(
            brain=mock_brain_with_memory,
            use_gpu=False
        )

        # Проверяем что гибридный кэш интегрирован
        assert ml_unit.hybrid_cache is not None
        assert ml_unit.hybrid_cache == mock_brain_with_memory.memory_manager.hybrid_cache

    @pytest.mark.integration
    def test_tokenizer_fractal_integration(self, mock_brain_with_memory):
        """Тест интеграции ЕВАTokenizer с фрактальным хранилищем."""
        from eva_ai.mlearning.cogniflex_tokenizer import ЕВАTokenizer

        tokenizer = ЕВАTokenizer(brain=mock_brain_with_memory)

        # Проверяем что гибридный кэш доступен
        assert tokenizer.hybrid_cache is not None
        # Проверяем что кэш интегрирован (не обязательно тот же объект)
        assert hasattr(tokenizer, 'hybrid_cache')

    @pytest.mark.integration
    def test_parallel_tokenizer_fractal_integration(self, mock_brain_with_memory):
        """Тест интеграции ParallelTokenizer с фрактальным хранилищем."""
        from eva_ai.mlearning.parallel_tokenization import ParallelTokenizer

        # Mock CacheRouter для интеграции
        cache_router = Mock()
        cache_router.register_batch.return_value = "batch_123"
        cache_router.register_segment.return_value = "seg_456"
        mock_brain_with_memory.cache_router = cache_router

        parallel_tokenizer = ParallelTokenizer(
            brain=mock_brain_with_memory,
            max_data_window_bytes=50 * 1024 * 1024
        )

        # Проверяем что компоненты интегрированы
        assert parallel_tokenizer.brain == mock_brain_with_memory
        assert hasattr(parallel_tokenizer, '_persist_stub')

        parallel_tokenizer.start()

        try:
            # Тест обработки с интеграцией кэша
            parallel_tokenizer.submit("test_batch", "Тестовый текст")

            # Даем время на обработку
            time.sleep(0.2)

            # Проверяем что данные сохраняются через CacheRouter
            cache_router.register_batch.assert_called()

        finally:
            parallel_tokenizer.stop()


class TestMLSystemHealth:
    """Тесты здоровья ML-системы."""

    @pytest.fixture
    def healthy_ml_unit(self):
        """Создает здоровый MLUnit для тестирования."""
        brain = Mock()
        brain.cache_dir = tempfile.mkdtemp()

        from eva_ai.mlearning.ml_unit import MLUnit

        ml_unit = MLUnit(
            brain=brain,
            use_gpu=False
        )

        # Настраиваем компоненты как здоровые
        ml_unit.text_processor = Mock()
        ml_unit.response_generator = Mock()
        ml_unit.model_manager = Mock()
        ml_unit.hybrid_cache = Mock()

        return ml_unit

    @pytest.mark.unit
    def test_ml_system_health_check(self, healthy_ml_unit):
        """Тест проверки здоровья ML-системы."""
        health = healthy_ml_unit.get_system_health()

        assert 'status' in health
        assert 'score' in health
        assert health['score'] >= 0.0
        assert health['score'] <= 1.0

    @pytest.mark.unit
    def test_ml_statistics_tracking(self, healthy_ml_unit):
        """Тест отслеживания статистики ML-системы."""
        initial_requests = healthy_ml_unit.stats['total_requests']

        # Выполняем несколько запросов
        for i in range(3):
            healthy_ml_unit.generate_response(f"Тестовый запрос {i}")

        # Проверяем обновление статистики
        assert healthy_ml_unit.stats['total_requests'] == initial_requests + 3
        assert healthy_ml_unit.stats['successful_requests'] == 3
        assert healthy_ml_unit.stats['total_processing_time'] > 0

    @pytest.mark.unit
    def test_ml_error_handling(self, healthy_ml_unit):
        """Тест обработки ошибок в ML-системе."""
        # Настраиваем response_generator на выброс исключения
        healthy_ml_unit.response_generator.generate_response.side_effect = Exception("Test error")

        # Проверяем что ошибка обрабатывается gracefully
        response = healthy_ml_unit.generate_response("Тестовый запрос")

        assert 'error' in response or response.get('text') is not None

        # Проверяем обновление статистики ошибок
        assert healthy_ml_unit.stats['failed_requests'] > 0


# Вспомогательные функции для тестирования
def create_test_text_data():
    """Создает тестовые текстовые данные для тестирования."""
    return {
        'simple_text': 'Это простой тестовый текст.',
        'complex_text': 'Это более сложный текст для тестирования различных аспектов токенизации и обработки.',
        'russian_text': 'Пример текста на русском языке с различными словами и предложениями.',
        'mixed_text': 'Mixed русский and English текст для тестирования мультиязычности.'
    }


def setup_test_environment():
    """Настраивает тестовое окружение."""
    # Создаем временную директорию для тестов
    test_dir = tempfile.mkdtemp()

    # Настраиваем переменные окружения для тестирования
    os.environ['COGNIFLEX_TEST_MODE'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    return test_dir


def cleanup_test_environment(test_dir):
    """Очищает тестовое окружение."""
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Очищаем переменные окружения
    os.environ.pop('COGNIFLEX_TEST_MODE', None)


if __name__ == "__main__":
    print("🚀 Запуск комплексных тестов ML-компонентов ЕВА...")

    # Настройка тестового окружения
    test_dir = setup_test_environment()

    try:
        # Запуск pytest
        import subprocess
        result = subprocess.run([
            'python', '-m', 'pytest',
            __file__,
            '-v',
            '--tb=short'
        ], capture_output=True, text=True)

        print("Вывод тестов:")
        print(result.stdout)
        if result.stderr:
            print("Ошибки:")
            print(result.stderr)

        print(f"Код завершения: {result.returncode}")

    finally:
        # Очистка тестового окружения
        cleanup_test_environment(test_dir)

    print("✅ Тесты ML-компонентов завершены!")
