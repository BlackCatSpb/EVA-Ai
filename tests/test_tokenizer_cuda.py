import asyncio
import torch
import logging
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Добавляем корневую директорию в PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from cogniflex.mlearning.cogniflex_tokenizer import CogniFlexTokenizer, TokenizationConfig
from cogniflex.core.core_brain import CoreBrain

# ModelMetadata is conditionally defined in cogniflex_tokenizer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cogniflex.mlearning.model_manager import ModelMetadata
else:
    class ModelMetadata:  # type: ignore
        pass

class TestTokenizer:
    """Класс для тестирования токенизатора с поддержкой CUDA."""
    
    def __init__(self):
        """Инициализация тестового класса."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.tokenizer_path = Path(__file__).parent.parent / "cogniflex" / "mlearning" / "cogniflex_models" / "fractal_unified_text-generation"
        self.brain = None  # Mock brain for testing
        self.loop = None
        self.brain = self._create_mock_brain()
        self.model_name = "fractal_unified_text-generation"
        self.tokenizer_path = Path("cogniflex/mlearning/cogniflex_models/fractal_unified_text-generation")
        
    def _create_mock_brain(self) -> CoreBrain:
        """Создает мок-объект CoreBrain для тестирования."""
        class MockBrain:
            def __init__(self):
                self.hybrid_cache = None
                self.config = {}
                
        return MockBrain()
    
    async def initialize_tokenizer(self) -> bool:
        """Инициализирует токенизатор CogniFlex с поддержкой CUDA."""
        try:
            # Создаем метаданные модели (используем пустую инициализацию, так как класс не принимает аргументы)
            model_metadata = ModelMetadata()
            
            # Устанавливаем атрибуты напрямую, если они доступны
            if hasattr(model_metadata, 'model_name'):
                model_metadata.model_name = self.model_name
            if hasattr(model_metadata, 'model_type'):
                model_metadata.model_type = "gpt"
            if hasattr(model_metadata, 'language'):
                model_metadata.language = "ru"
            if hasattr(model_metadata, 'model_path'):
                model_metadata.model_path = str(self.tokenizer_path.absolute())
            
            # Инициализируем конфигурацию токенизации
            config = TokenizationConfig(
                priority_strategy="initial_response",
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Создаем экземпляр токенизатора с правильными параметрами
            try:
                self.tokenizer = await CogniFlexTokenizer.from_pretrained(
                    pretrained_model_name_or_path=str(self.tokenizer_path.absolute()),
                    brain=self.brain,
                    model_metadata=model_metadata,
                    device_map="auto" if self.device == 'cuda' else None,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                )
                
                # Проверяем инициализацию токенизатора
                if not hasattr(self.tokenizer, 'tokenizer') or self.tokenizer.tokenizer is None:
                    raise RuntimeError("Не удалось инициализировать внутренний токенизатор")
                    
                # Проверяем, что токенизатор готов к использованию
                if not hasattr(self.tokenizer, 'initialized') or not self.tokenizer.initialized:
                    raise RuntimeError("Токенизатор не инициализирован")
                
                # Тестируем токенизацию
                test_text = "Привет, мир!"
                tokens = self.tokenizer.tokenizer.tokenize(test_text)
                logger.info(f"Токенизатор успешно загружен. Пример токенизации: {tokens[:5]}...")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке токенизатора: {e}", exc_info=True)
                return False
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации токенизатора: {e}", exc_info=True)
            return False
    
    async def test_sync_tokenization(self) -> None:
        """Тестирует синхронную токенизацию."""
        if not self.tokenizer or not hasattr(self.tokenizer, 'tokenizer') or self.tokenizer.tokenizer is None:
            self.logger.error("Токенизатор не инициализирован для синхронной токенизации")
            return
            
        test_texts = [
            "Привет, как дела?",
            "Сегодня отличная погода!",
            "Давай проверим работу токенизатора на разных типах текста.",
            "Этот текст содержит числа: 123, символы: !@# и специальные символы: %^&*()_+",
            "Многострочный\n            текст для проверки\n            обработки переносов строк"
        ]
        
        self.logger.info("\n=== Тест синхронной токенизации ===")
        for text in test_texts:
            try:
                # Используем прямой вызов метода токенизатора
                tokens = self.tokenizer.tokenizer.tokenize(text)
                # Декодируем токены для корректного отображения
                decoded_tokens = [self.tokenizer.tokenizer.convert_tokens_to_string([t]) for t in tokens[:3]]
                self.logger.info(f"Текст: {text[:30]}...")
                self.logger.info(f"Токены: {decoded_tokens}... (всего {len(tokens)} токенов)")
                self.logger.debug(f"Сырые токены: {tokens[:3]}...")
            except Exception as e:
                self.logger.error(f"Ошибка при токенизации текста: {e}")
                raise
    
    async def test_async_tokenization(self) -> None:
        """Тестирует асинхронную токенизацию."""
        if not self.tokenizer or not hasattr(self.tokenizer, 'tokenizer'):
            self.logger.error("Токенизатор не инициализирован для асинхронной токенизации")
            return
            
        test_texts = [
            "Асинхронная обработка текста 1",
            "Еще один тестовый текст для проверки",
            "Токенизация с использованием GPU и CUDA",
            "Параллельная обработка нескольких запросов",
            "Асинхронное выполнение ускоряет работу"
        ]
        
        logger.info("\n=== Тест асинхронной токенизации ===")
        
        # Создаем задачи для асинхронной обработки
        tasks = []
        for i, text in enumerate(test_texts, 1):
            try:
                # Создаем задачу на токенизацию
                task = asyncio.create_task(self.tokenizer.tokenize_async(text))
                tasks.append((i, text, task))
            except Exception as e:
                logger.error(f"Ошибка при создании задачи {i}: {e}")
        
        # Обрабатываем результаты по мере их выполнения
        for i, text, task in tasks:
            try:
                # Ожидаем завершения задачи
                tokens = await task
                
                # Декодируем токены для корректного отображения
                if tokens:
                    decoded_tokens = [self.tokenizer.tokenizer.convert_tokens_to_string([t]) for t in tokens[:3]]
                    logger.info(f"Задача {i}: успешно обработано {len(tokens)} токенов")
                    logger.info(f"Текст: {text[:30]}...")
                    logger.info(f"Токены: {decoded_tokens}...")
                    logger.debug(f"Сырые токены: {tokens[:3]}...")
            except Exception as e:
                logger.error(f"Ошибка в задаче {i}: {e}")
    
    async def test_encoding_decoding(self) -> None:
        """Тестирует кодирование и декодирование."""
        if not self.tokenizer or not hasattr(self.tokenizer, 'tokenizer') or self.tokenizer.tokenizer is None:
            logger.error("Токенизатор не инициализирован для тестирования кодирования/декодирования")
            return
            
        test_texts = [
            "Тестируем кодирование и декодирование",
            "Проверка работы с разными символами: 123 !@# $%^&*()_+",
            "Многострочный\nтекст\nс\nпереносами"
        ]
        
        logger.info("\n=== Тест кодирования/декодирования ===")
        for text in test_texts:
            try:
                # Используем прямой вызов методов токенизатора с правильными параметрами
                encoded = self.tokenizer.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True
                )
                
                if not encoded or "input_ids" not in encoded:
                    logger.error(f"Ошибка кодирования для текста: {text}")
                    continue
                    
                # Декодируем
                decoded = self.tokenizer.tokenizer.decode(
                    encoded["input_ids"][0], 
                    skip_special_tokens=True
                )
                
                logger.info(f"Оригинал: {text}")
                logger.info(f"Декодировано: {decoded}")
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"Ошибка при кодировании/декодировании: {e}", exc_info=True)
                
    async def run_tests(self) -> None:
        """Запускает все тесты токенизатора."""
        try:
            # Запускаем синхронные тесты
            self.logger.info("\n=== Запуск синхронных тестов ===")
            await self.test_sync_tokenization()
            
            # Запускаем асинхронные тесты
            self.logger.info("\n=== Запуск асинхронных тестов ===")
            await self.test_async_tokenization()
            
            # Тестируем кодирование/декодирование
            self.logger.info("\n=== Тестирование кодирования/декодирования ===")
            await self.test_encoding_decoding()
            
            self.logger.info("\n=== Все тесты завершены ===")
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении тестов: {e}", exc_info=True)
            raise

async def main():
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tokenizer_test.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Выводим информацию о системе
    logger.info("=== Начало тестирования токенизатора ===")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"Устройство CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"Версия CUDA: {torch.version.cuda}")
    
    # Инициализируем тестер
    tester = TestTokenizer()
    
    # Запускаем тесты
    await tester.run_tests()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
