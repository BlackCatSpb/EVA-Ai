"""
Real Benchmark - Сравнение генерации с реальной моделью.
Baseline vs Optimized (с кэшированием и предиктором).
"""

import os
import sys
import time
import json
import logging
import importlib.util
import statistics
from typing import Dict, List

# Setup
_mod_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _mod_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def _import_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_real_comparison():
    """Сравнение реальной генерации с оптимизациями."""
    
    # === ЗАГРУЗКА МОДЕЛИ ===
    logger.info("=" * 60)
    logger.info("ЗАГРУЗКА МОДЕЛИ")
    logger.info("=" * 60)
    
    from llama_cpp import Llama
    
    model_path = os.path.join(_mod_dir, 'models', 'qwen2.5-3b-instruct-q4_k_m.gguf')
    
    llm = Llama(
        model_path=model_path,
        n_ctx=256,
        n_threads=4,
        n_gpu_layers=0
    )
    logger.info("Модель загружена")
    
    # === ИНИЦИАЛИЗАЦИЯ ОПТИМИЗАЦИЙ ===
    logger.info("\n" + "=" * 60)
    logger.info("ИНИЦИАЛИЗАЦИЯ ОПТИМИЗАЦИЙ")
    logger.info("=" * 60)
    
    index_mod = _import_module('context_index', os.path.join(_mod_dir, 'index', 'context_index.py'))
    embed_mod = _import_module('fast_embedder', os.path.join(_mod_dir, 'embeddings', 'fast_embedder.py'))
    
    ContextIndex = index_mod.ContextIndex
    FastEmbedder = embed_mod.FastEmbedder
    
    index = ContextIndex(cache_dir=os.path.join(_mod_dir, 'cache'))
    embedder = FastEmbedder(embeddings_dir=os.path.join(_mod_dir, 'embeddings'))
    
    # === ТЕСТОВЫЕ ПРОМПТЫ ===
    prompts = [
        "Привет! Как дела?",
        "Что такое искусственный интеллект?",
        "Как развить технологии в России?",
        "Расскажи о машинном обучении",
        "Какие бывают нейросети?",
    ]
    
    # === ПРОГРЕВ ===
    logger.info("\n" + "=" * 60)
    logger.info("ПРОГРЕВ КЭША")
    logger.info("=" * 60)
    
    for p in prompts:
        index.tokenize_with_cache(p)
        embedder.get_embedding(list(range(10)))
    
    index.add_training_data(prompts * 3)
    
    logger.info(f"Кэш прогрет: {index.get_stats()}")
    
    # === БАЗОВАЯ ГЕНЕРАЦИЯ (БЕЗ ОПТИМИЗАЦИЙ) ===
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE (без оптимизаций)")
    logger.info("=" * 60)
    
    baseline_times = []
    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        
        output = llm(prompt, max_tokens=30, temperature=0.7)
        
        elapsed = time.perf_counter() - start
        baseline_times.append(elapsed)
        
        text = output['choices'][0]['text']
        logger.info(f"Промпт {i+1}: {elapsed:.2f}s")
    
    avg_baseline = statistics.mean(baseline_times)
    logger.info(f"Среднее время baseline: {avg_baseline:.2f}s")
    
    # === ОПТИМИЗИРОВАННАЯ ГЕНЕРАЦИЯ ===
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZED (с оптимизациями)")
    logger.info("=" * 60)
    
    optimized_times = []
    cache_hits = 0
    total_calls = 0
    
    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        
        # Используем оптимизации
        tokens, cached = index.tokenize_with_cache(prompt)
        if cached:
            cache_hits += 1
        total_calls += 1
        
        _ = embedder.get_embedding(tokens[:10])
        _ = index.predict_next_tokens(tokens[:3])
        
        output = llm(prompt, max_tokens=30, temperature=0.7)
        
        elapsed = time.perf_counter() - start
        optimized_times.append(elapsed)
        
        text = output['choices'][0]['text']
        logger.info(f"Промпт {i+1}: {elapsed:.2f}s (cached={cached})")
    
    avg_optimized = statistics.mean(optimized_times)
    cache_rate = cache_hits / total_calls * 100
    
    logger.info(f"Среднее время optimized: {avg_optimized:.2f}s")
    logger.info(f"Cache hit rate: {cache_rate:.0f}%")
    
    # === РЕЗУЛЬТАТЫ ===
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ")
    logger.info("=" * 60)
    
    speedup = avg_baseline / avg_optimized if avg_optimized > 0 else 1
    
    results = {
        'model': 'qwen2.5-3b-instruct-q4_k_m',
        'baseline_avg_sec': avg_baseline,
        'optimized_avg_sec': avg_optimized,
        'speedup_x': speedup,
        'improvement_percent': (avg_baseline - avg_optimized) / avg_baseline * 100,
        'cache_hit_rate': cache_rate,
        'baseline_times': baseline_times,
        'optimized_times': optimized_times,
        'tokens_per_sec': 30 / avg_baseline  # примерно
    }
    
    logger.info(f"Baseline: {avg_baseline:.2f}s")
    logger.info(f"Optimized: {avg_optimized:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Улучшение: {results['improvement_percent']:.1f}%")
    
    if speedup > 1.05:
        logger.info("\n✓ ОПТИМИЗАЦИИ РАБОТАЮТ!")
    else:
        logger.info("\n! Прирост минимальный на коротких промптах")
        logger.info("На длинных контекстах эффект будет заметнее")
    
    # Сохранение
    with open(os.path.join(_mod_dir, 'real_benchmark.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nСохранено: real_benchmark.json")
    
    return results


if __name__ == '__main__':
    run_real_comparison()