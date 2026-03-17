import os
import time
import queue
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import GPT2Tokenizer
from .hybrid_cache import HybridCache

class PromptWorker:
    def __init__(self, worker_id: int, cache: HybridCache, model_name: str = "sberbank-ai/rugpt3small_based_on_gpt2"):
        """
        Инициализация воркера для обработки промптов
        :param worker_id: Уникальный идентификатор воркера
        :param cache: Экземпляр гибридного кеша
        :param model_name: Название модели для токенизатора
        """
        self.worker_id = worker_id
        self.cache = cache
        self.tokenizer = None
        self._init_tokenizer(model_name)
    
    def _init_tokenizer(self, model_name: str):
        """Инициализация токенизатора"""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Worker {self.worker_id}: Tokenizer initialized")
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to initialize tokenizer: {e}")
            raise
    
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Обработка одного промпта
        :param prompt: Текст промпта
        :return: Словарь с результатами обработки
        """
        start_time = time.time()
        
        try:
            # Токенизируем промпт
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding='max_length',
                max_length=512,
                truncation=True
            )
            
            # Сохраняем токены в кеш
            cache_key = f"prompt_{self.worker_id}_{int(time.time() * 1000)}"
            self.cache.store(f"{cache_key}_input_ids", inputs['input_ids'])
            self.cache.store(f"{cache_key}_attention_mask", inputs['attention_mask'])
            
            # Возвращаем метаданные
            return {
                'status': 'success',
                'worker_id': self.worker_id,
                'prompt': prompt,
                'cache_key': cache_key,
                'num_tokens': inputs['input_ids'].shape[1],
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'worker_id': self.worker_id,
                'prompt': prompt,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

class PromptProcessor:
    def __init__(self, num_workers: int = 4, cache_dir: str = "./cache"):
        """
        Инициализация процессора промптов с пулом воркеров
        :param num_workers: Количество воркеров
        :param cache_dir: Директория для кеша
        """
        self.num_workers = num_workers
        self.cache = HybridCache(cache_dir=cache_dir)
        self.workers = [PromptWorker(i, self.cache) for i in range(num_workers)]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        
    def split_prompt(self, prompt: str, max_length: int = 200) -> List[str]:
        """
        Разбивка промпта на подпромпты
        :param prompt: Исходный промпт
        :param max_length: Максимальная длина подпромпта
        :return: Список подпромптов
        """
        # Простая реализация разбивки по предложениям
        import re
        sentences = re.split(r'(?<=[.!?])\s+', prompt)
        
        subprompts = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Добавляем точку, если предложение заканчивается на букву
            if sentence[-1].isalpha():
                sentence += '.'
                
            # Проверяем длину текущего чанка
            if current_length + len(sentence) + 1 > max_length and current_chunk:
                subprompts.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 за пробел
        
        # Добавляем последний чанк, если он не пустой
        if current_chunk:
            subprompts.append(' '.join(current_chunk))
            
        return subprompts
    
    def process_prompts(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Параллельная обработка списка промптов
        :param prompts: Список промптов для обработки
        :return: Список результатов обработки
        """
        results = []
        
        # Раздаем промпты воркерам
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_prompt = {
                executor.submit(self.workers[i % self.num_workers].process_prompt, prompt): prompt 
                for i, prompt in enumerate(prompts)
            }
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'status': 'error',
                        'prompt': prompt,
                        'error': str(e)
                    })
        
        return results
    
    def process_single_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Обработка одного промпта с автоматической разбивкой
        :param prompt: Исходный промпт
        :return: Результаты обработки
        """
        # Разбиваем промпт на подпромпты
        subprompts = self.split_prompt(prompt)
        print(f"Разбито на {len(subprompts)} подпромптов")
        
        # Обрабатываем подпромпты параллельно
        results = self.process_prompts(subprompts)
        
        # Собираем результаты
        success_results = [r for r in results if r.get('status') == 'success']
        error_results = [r for r in results if r.get('status') == 'error']
        
        return {
            'status': 'completed',
            'total_subprompts': len(subprompts),
            'successful': len(success_results),
            'failed': len(error_results),
            'cache_keys': [r['cache_key'] for r in success_results],
            'errors': [r.get('error') for r in error_results]
        }
    
    def clear_cache(self):
        """Очистка кеша"""
        self.cache.clear()
        print("Cache cleared")

# Пример использования
if __name__ == "__main__":
    # Создаем процессор с 4 воркерами
    processor = PromptProcessor(num_workers=4)
    
    # Пример промпта
    prompt = """
    Искусственный интеллект - это область компьютерных наук, 
    занимающаяся созданием интеллектуальных машин, способных 
    выполнять задачи, которые обычно требуют человеческого интеллекта. 
    Машинное обучение - это подраздел ИИ, который фокусируется на 
    разработке алгоритмов, которые могут учиться на данных и делать 
    прогнозы или принимать решения на основе этих данных.
    """
    
    # Обрабатываем промпт
    result = processor.process_single_prompt(prompt)
    print(f"Обработка завершена: {result}")
    
    # Очищаем кеш
    processor.clear_cache()
