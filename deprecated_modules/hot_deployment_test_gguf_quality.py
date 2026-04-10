"""
Тест качества GGUF модели Qwen2.5-0.5B.
Проверяем корректность ответов на русском языке.
"""
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Путь к модели
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(project_root, "models", "qwen2.5-0.5b-instruct-q4_0.gguf")

logger.info(f"Модель: {model_path}")
logger.info(f"Существует: {os.path.exists(model_path)}")

from llama_cpp import Llama

ll = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=0
)

logger.info("Модель загружена!")


def test_response(prompt: str, max_tokens: int = 100) -> str:
    """Тестирует генерацию"""
    start = time.time()
    
    response = ll.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1
    )
    
    elapsed = time.time() - start
    
    text = response['choices'][0]['message']['content']
    tokens = response.get('usage', {}).get('completion_tokens', max_tokens)
    speed = tokens / elapsed if elapsed > 0 else 0
    
    logger.info(f"Время: {elapsed:.2f}s, скорость: {speed:.1f} tok/s")
    
    return text


# Тесты
tests = [
    # Простые приветствия
    ("Привет! Как тебя зовут?", 30),
    ("Привет! Как дела?", 30),
    
    # Вопросы о себе
    ("Кто ты?", 50),
    ("Что ты умеешь?", 80),
    
    # Русский язык
    ("Расскажи анекдот", 100),
    ("Объясни что такое машинное обучение простыми словами", 100),
    
    # Математика
    ("Сколько будет 2+2?", 20),
    ("Что больше: 100 или 200?", 30),
    
    # Креатив
    ("Напиши короткое стихотворение о кошке", 80),
    ("Придумай название для кофейни", 30),
    
    # Технические вопросы
    ("Как работает интернет?", 100),
    ("Что такое Python?", 80),
]

logger.info("\n" + "="*60)
logger.info("ТЕСТ КАЧЕСТВА GGUF МОДЕЛИ")
logger.info("="*60)

for i, (prompt, max_tokens) in enumerate(tests, 1):
    logger.info(f"\n--- Тест {i}/{len(tests)} ---")
    logger.info(f"Вопрос: {prompt}")
    
    try:
        response = test_response(prompt, max_tokens)
        logger.info(f"Ответ: {response}")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
    
    time.sleep(0.5)  # Пауза между тестами

logger.info("\n" + "="*60)
logger.info("ТЕСТЫ ЗАВЕРШЕНЫ")
logger.info("="*60)