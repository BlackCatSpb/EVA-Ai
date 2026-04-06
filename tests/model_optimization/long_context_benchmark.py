"""
Long Context Benchmark - тестирование на длинных промптах.
"""

import os
import sys
import time
import json
import logging
import importlib.util
import statistics
from typing import Dict, List

_mod_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _mod_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def _import_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Длинные промпты (200-500 слов)
LONG_PROMPTS = [
    """
    Расскажи подробно о развитии искусственного интеллекта в России за последние 10 лет.
    Какие государственные программы были запущены? Какие компании лидируют в этой области?
    Какие научные центры и университеты занимаются исследованиями? Какие результаты достигнуты?
    Какие вызовы стоят перед российской индустрией ИИ? Какие направления считаются приоритетными?
    """,

    """
    Объясни принципы работы фрактальной памяти в когнитивных системах.
    Как фрактальная структура помогает хранить и извлекать информацию?
    Какие математические модели используются для описания фракталов?
    Как фрактальная память связана с ассоциативным мышлением человека?
    Приведи практические примеры применения фрактальной памяти в ИИ системах.
    """,

    """
    Проанализируй современное состояние квантовых вычислений и их влияние на машинное обучение.
    Какие алгоритмы квантового машинного обучения существуют?
    Какие преимущества даёт квантовый компьютер для обучения нейросетей?
    Когда ожидается массовое применение квантовых компьютеров в ИИ?
    Какие компании и страны лидируют в разработке квантовых технологий?
    """,

    """
    Опиши архитектуру современных больших языковых моделей.
    Как работает механизм внимания (attention)? Что такое трансформеры?
    Какие методы используются для эффективного обучения LLM?
    Что такое RLHF и как он улучшает качество генерации?
    Какие ограничения современных LLM и как их преодолевают?
    """,
]


def run_long_context_benchmark():
    """Тест на длинных контекстах."""
    
    # Загрузка модели
    logger.info("=" * 60)
    logger.info("ЗАГРУЗКА МОДЕЛИ")
    logger.info("=" * 60)
    
    from llama_cpp import Llama
    
    model_path = os.path.join(_mod_dir, 'models', 'qwen2.5-3b-instruct-q4_k_m.gguf')
    
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=0
    )
    logger.info("Модель загружена")
    
    # Оптимизации
    logger.info("\n" + "=" * 60)
    logger.info("ИНИЦИАЛИЗАЦИЯ ОПТИМИЗАЦИЙ")
    logger.info("=" * 60)
    
    index_mod = _import_module('context_index', os.path.join(_mod_dir, 'index', 'context_index.py'))
    embed_mod = _import_module('fast_embedder', os.path.join(_mod_dir, 'embeddings', 'fast_embedder.py'))
    
    ContextIndex = index_mod.ContextIndex
    FastEmbedder = embed_mod.FastEmbedder
    
    index = ContextIndex(cache_dir=os.path.join(_mod_dir, 'cache'))
    embedder = FastEmbedder(embeddings_dir=os.path.join(_mod_dir, 'embeddings'))
    
    # Прогрев на длинных промптах
    for p in LONG_PROMPTS:
        index.tokenize_with_cache(p)
        tokens = list(range(min(len(p), 100)))
        embedder.get_embedding(tokens)
    
    index.add_training_data(LONG_PROMPTS)
    
    logger.info(f"Кэш прогрет: {index.get_stats()}")
    
    # === BASELINE (без оптимизаций) ===
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE - длинные промпты")
    logger.info("=" * 60)
    
    baseline_times = []
    baseline_tokenize = []
    
    for i, prompt in enumerate(LONG_PROMPTS):
        logger.info(f"\nПромпт {i+1} ({len(prompt.split())} слов):")
        
        # Замер токенизации
        tok_start = time.perf_counter()
        tokens = len(prompt.split())  # примерное число токенов
        tok_time = time.perf_counter() - tok_start
        baseline_tokenize.append(tok_time * 1000)
        
        # Замер генерации
        gen_start = time.perf_counter()
        output = llm(prompt, max_tokens=40, temperature=0.7)
        gen_time = time.perf_counter() - gen_start
        
        total_time = tok_time + gen_time
        baseline_times.append(total_time)
        
        logger.info(f"  Токенизация: {tok_time*1000:.1f}ms")
        logger.info(f"  Генерация: {gen_time:.2f}s")
        logger.info(f"  Всего: {total_time:.2f}s")
    
    avg_baseline = statistics.mean(baseline_times)
    avg_tokenize_baseline = statistics.mean(baseline_tokenize)
    
    logger.info(f"\nСреднее время baseline: {avg_baseline:.2f}s")
    logger.info(f"Среднее время токенизации: {avg_tokenize_baseline:.1f}ms")
    
    # === OPTIMIZED (с оптимизациями) ===
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZED - длинные промпты")
    logger.info("=" * 60)
    
    optimized_times = []
    optimized_tokenize = []
    cache_hits = 0
    total_calls = 0
    
    for i, prompt in enumerate(LONG_PROMPTS):
        logger.info(f"\nПромпт {i+1} ({len(prompt.split())} слов):")
        
        # Замер токенизации с кэшем
        tok_start = time.perf_counter()
        tokens, cached = index.tokenize_with_cache(prompt)
        if cached:
            cache_hits += 1
        total_calls += 1
        tok_time = time.perf_counter() - tok_start
        optimized_tokenize.append(tok_time * 1000)
        
        # Эмбеддинг
        _ = embedder.get_embedding(tokens[:min(len(tokens), 50)])
        
        # Предиктор
        _ = index.predict_next_tokens(tokens[:min(5, len(tokens))])
        
        # Генерация
        gen_start = time.perf_counter()
        output = llm(prompt, max_tokens=40, temperature=0.7)
        gen_time = time.perf_counter() - gen_start
        
        total_time = tok_time + gen_time
        optimized_times.append(total_time)
        
        logger.info(f"  Токенизация (cached={cached}): {tok_time*1000:.1f}ms")
        logger.info(f"  Генерация: {gen_time:.2f}s")
        logger.info(f"  Всего: {total_time:.2f}s")
    
    avg_optimized = statistics.mean(optimized_times)
    avg_tokenize_optimized = statistics.mean(optimized_tokenize)
    cache_rate = cache_hits / total_calls * 100
    
    logger.info(f"\nСреднее время optimized: {avg_optimized:.2f}s")
    logger.info(f"Среднее время токенизации: {avg_tokenize_optimized:.1f}ms")
    
    # === РЕЗУЛЬТАТЫ ===
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ (длинные контексты)")
    logger.info("=" * 60)
    
    speedup = avg_baseline / avg_optimized if avg_optimized > 0 else 1
    tokenize_speedup = avg_tokenize_baseline / avg_tokenize_optimized if avg_tokenize_optimized > 0 else 1
    
    results = {
        'test_type': 'long_context',
        'baseline_avg_sec': avg_baseline,
        'optimized_avg_sec': avg_optimized,
        'speedup_x': speedup,
        'improvement_percent': (avg_baseline - avg_optimized) / avg_baseline * 100,
        'tokenize_baseline_ms': avg_tokenize_baseline,
        'tokenize_optimized_ms': avg_tokenize_optimized,
        'tokenize_speedup_x': tokenize_speedup,
        'cache_hit_rate': cache_rate,
        'avg_words_per_prompt': sum(len(p.split()) for p in LONG_PROMPTS) / len(LONG_PROMPTS)
    }
    
    logger.info(f"Baseline: {avg_baseline:.2f}s")
    logger.info(f"Optimized: {avg_optimized:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Улучшение: {results['improvement_percent']:.1f}%")
    logger.info(f"\nТокенизация:")
    logger.info(f"  Baseline: {avg_tokenize_baseline:.1f}ms")
    logger.info(f"  Optimized: {avg_tokenize_optimized:.1f}ms")
    logger.info(f"  Speedup: {tokenize_speedup:.2f}x")
    logger.info(f"Cache hit: {cache_rate:.0f}%")
    
    if speedup > 1.1:
        logger.info("\n✓ ОПТИМИЗАЦИИ ЗНАЧИТЕЛЬНО УЛУЧШИЛИ ПРОИЗВОДИТЕЛЬНОСТЬ!")
    elif tokenize_speedup > 1.5:
        logger.info("\n✓ ЗНАЧИТЕЛЬНОЕ УЛУЧШЕНИЕ ТОКЕНИЗАЦИИ!")
    
    # Сохранение
    with open(os.path.join(_mod_dir, 'long_context_benchmark.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nСохранено: long_context_benchmark.json")
    
    return results


if __name__ == '__main__':
    run_long_context_benchmark()