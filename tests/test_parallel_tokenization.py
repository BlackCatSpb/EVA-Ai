import asyncio
import time
from typing import List
import torch
from cogniflex.core.core_brain import CoreBrain
from cogniflex.mlearning.cogniflex_tokenizer import CogniFlexTokenizer

class TokenizationCoordinator:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self._executor = None
        
    async def initialize(self):
        """Инициализация координатора и токенизатора"""
        self.tokenizer = CogniFlexTokenizer()
        await self.tokenizer.from_pretrained(
            self.config['tokenizer_path'],
            **self.config.get('tokenizer_kwargs', {})
        )
    
    async def process_batch_async(self, texts: List[str]) -> List[List[str]]:
        """Асинхронная обработка пакета текстов"""
        if not self.tokenizer:
            await self.initialize()
            
        start_time = time.time()
        tasks = [self.tokenizer.tokenize_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        print(f"Обработано {len(texts)} текстов за {elapsed:.4f} сек. "
              f"({len(texts)/elapsed:.2f} текстов/сек)")
        return results
    
    def process_batch_sync(self, texts: List[str]) -> List[List[str]]:
        """Синхронная обработка пакета текстов"""
        if not self.tokenizer:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.initialize())
            
        start_time = time.time()
        results = [self.tokenizer.tokenize(text) for text in texts]
        elapsed = time.time() - start_time
        
        print(f"Синхронно обработано {len(texts)} текстов за {elapsed:.4f} сек. "
              f"({len(texts)/elapsed:.2f} текстов/сек)")
        return results

async def main():
    # Конфигурация
    config = {
        'tokenizer_path': 'cogniflex/mlearning/cogniflex_models/fractal_unified_text-generation',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'tokenizer_kwargs': {
            'model_max_length': 2048,
            'padding_side': 'left',
            'truncation_side': 'right'
        }
    }
    
    # Создаем координатор
    coordinator = TokenizationCoordinator(config)
    
    # Тестовые данные
    test_texts = [
        "Привет, как твои дела?",
        "Сегодня отличная погода для прогулки в парке.",
        "Искусственный интеллект меняет наше будущее.",
        "Давай проверим работу параллельной токенизации.",
        "Этот тест покажет производительность системы.",
        "Параллельная обработка ускоряет выполнение задач.",
        "CogniFlex - это мощная платформа для ИИ.",
        "Давай протестируем обработку длинных текстов.",
        "Токенизация - важный этап обработки естественного языка.",
        "Асинхронность позволяет эффективно использовать ресурсы."
    ] * 10  # Увеличиваем количество текстов для наглядности
    
    print(f"Начинаем тестирование с {len(test_texts)} текстами...")
    
    # Тестируем синхронную обработку
    print("\n=== Синхронная обработка ===")
    start_sync = time.time()
    sync_results = coordinator.process_batch_sync(test_texts)
    sync_time = time.time() - start_sync
    
    # Тестируем асинхронную обработку
    print("\n=== Асинхронная обработка ===")
    start_async = time.time()
    async_results = await coordinator.process_batch_async(test_texts)
    async_time = time.time() - start_async
    
    # Сравниваем результаты
    print("\n=== Результаты тестирования ===")
    print(f"Синхронная обработка: {sync_time:.4f} сек")
    print(f"Асинхронная обработка: {async_time:.4f} сек")
    print(f"Ускорение: {sync_time/async_time:.2f}x")
    
    # Проверяем корректность результатов
    assert len(sync_results) == len(async_results), "Количество результатов не совпадает"
    for i, (sync, async_res) in enumerate(zip(sync_results, async_results)):
        assert sync == async_res, f"Результаты не совпадают для текста {i}"
    
    print("\nПроверка корректности результатов завершена успешно!")

if __name__ == "__main__":
    asyncio.run(main())
