"""
UnifiedCacheBridge Benchmark - тестирование объединённого кэша модели и графа знаний.

Сравнивает генерацию:
1. Baseline: без кэша графа
2. With Graph Cache: с предзагрузкой узлов графа в токен-кэш
3. With Enriched Prompt: с обогащённым промптом из графа
4. Full Unified: полный цикл (кэш генерации + граф + обогащение)
"""

import os
import sys
import time
import json
import logging
import statistics
from typing import Dict, List, Optional

_mod_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_mod_dir)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MockKnowledgeGraph:
    """Моковый граф знаний для тестирования."""
    
    def __init__(self):
        self.nodes = {
            "ai_russia": {
                "name": "ИИ в России",
                "description": "Россия развивает ИИ: нацстратегия 2019, нацпрограмма 2024, Сколково, Иннополис, МФТИ",
                "node_type": "concept",
                "domain": "technology",
                "strength": 0.9,
            },
            "ml_algorithms": {
                "name": "Алгоритмы машинного обучения",
                "description": "Нейросети, трансформеры, LSTM, CNN, RNN, случайный лес, градиентный бустинг",
                "node_type": "concept",
                "domain": "machine_learning",
                "strength": 0.85,
            },
            "quantum_computing": {
                "name": "Квантовые вычисления",
                "description": "Квантовые компьютеры используют кубиты, квантовое превосходство, алгоритм Шора, Гровера",
                "node_type": "concept",
                "domain": "quantum",
                "strength": 0.7,
            },
            "nlp_russian": {
                "name": "NLP для русского языка",
                "description": "Обработка естественного языка для русского: токенизация, лемматизация, синтаксический анализ, ruGPT, ruBERT",
                "node_type": "concept",
                "domain": "nlp",
                "strength": 0.8,
            },
            "cogniflex": {
                "name": "CogniFlex EVA",
                "description": "Когнитивная система с рекурсивным рассуждением, самодиалогом и графом знаний",
                "node_type": "system",
                "domain": "ai_systems",
                "strength": 0.95,
            },
            "fractal_memory": {
                "name": "Фрактальная память",
                "description": "Самоподобные структуры для эффективного хранения и извлечения информации в нейросетях",
                "node_type": "concept",
                "domain": "memory",
                "strength": 0.6,
            },
        }
        
        self.edges = {
            "ai_russia_ml": {
                "source": "ai_russia",
                "target": "ml_algorithms",
                "relation": "uses",
            },
            "ml_nlp": {
                "source": "ml_algorithms",
                "target": "nlp_russian",
                "relation": "applied_to",
            },
            "cogniflex_fractal": {
                "source": "cogniflex",
                "target": "fractal_memory",
                "relation": "implements",
            },
        }
        
        self.stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_queries": 0,
            "successful_queries": 0,
        }


class MockTokenCache:
    """Моковый токен-кэш для тестирования."""
    
    def __init__(self, max_items: int = 1000):
        self._cache = {}
        self._max_items = max_items
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total': 0,
        }
    
    def get(self, key: str) -> Optional[Dict]:
        self.stats['total'] += 1
        if key in self._cache:
            self.stats['hits'] += 1
            return self._cache[key]
        self.stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Dict) -> None:
        if len(self._cache) >= self._max_items:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value
    
    def get_cache_stats(self) -> Dict:
        total = self.stats['total']
        return {
            **self.stats,
            'hit_rate': self.stats['hits'] / total if total > 0 else 0,
            'size': len(self._cache),
        }


class MockLLM:
    """Моковая LLM для тестирования."""
    
    def __init__(self, base_latency: float = 0.5):
        self.base_latency = base_latency
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        
        # Имитация задержки генерации
        # Более длинные промпты = чуть больше задержка
        length_factor = 1 + (len(prompt) / 10000)
        time.sleep(self.base_latency * length_factor)
        
        return f"Ответ на: {prompt[:50]}..."


def run_benchmark():
    """Запуск бенчмарка UnifiedCacheBridge."""
    
    from eva.core.unified_cache_bridge import UnifiedCacheBridge
    
    logger.info("=" * 60)
    logger.info("UNIFIED CACHE BRIDGE BENCHMARK")
    logger.info("=" * 60)
    
    # Тестовые запросы
    queries = [
        "Как развивают ИИ в России?",
        "Что такое машинное обучение?",
        "Расскажи о квантовых вычислениях",
        "Как работает NLP для русского языка?",
        "Что такое CogniFlex?",
        "Привет, как дела?",
        "Какие бывают нейросети?",
        "Объясни фрактальную память",
    ]
    
    # === ТЕСТ 1: BASELINE (без кэша графа) ===
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТ 1: BASELINE (без кэша графа)")
    logger.info("=" * 60)
    
    llm_baseline = MockLLM(base_latency=0.5)
    baseline_times = []
    
    for query in queries:
        start = time.perf_counter()
        _ = llm_baseline.generate(query)
        elapsed = time.perf_counter() - start
        baseline_times.append(elapsed)
        logger.info(f"  Query: {query[:40]}... -> {elapsed:.3f}s")
    
    avg_baseline = statistics.mean(baseline_times)
    logger.info(f"  Среднее: {avg_baseline:.3f}s")
    
    # === ТЕСТ 2: С ПРЕДЗАГРУЗКОЙ ГРАФА ===
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТ 2: С ПРЕДЗАГРУЗКОЙ ГРАФА В ТОКЕН-КЭШ")
    logger.info("=" * 60)
    
    graph = MockKnowledgeGraph()
    token_cache = MockTokenCache()
    bridge = UnifiedCacheBridge(
        token_cache=token_cache,
        knowledge_graph=graph,
        cache_dir=os.path.join(_mod_dir, 'cache')
    )
    
    llm_graph = MockLLM(base_latency=0.5)
    graph_times = []
    
    for query in queries:
        start = time.perf_counter()
        
        # Предзагружаем граф
        prep = bridge.prepare_for_generation(query)
        
        if prep['cached_response']:
            elapsed = time.perf_counter() - start
            graph_times.append(elapsed)
            logger.info(f"  Query: {query[:40]}... -> {elapsed:.3f}s (CACHED)")
            continue
        
        _ = llm_graph.generate(prep['prompt'])
        elapsed = time.perf_counter() - start
        graph_times.append(elapsed)
        logger.info(f"  Query: {query[:40]}... -> {elapsed:.3f}s (nodes={len(prep['graph_nodes'])}, preloaded={prep['preloaded_count']})")
    
    avg_graph = statistics.mean(graph_times)
    logger.info(f"  Среднее: {avg_graph:.3f}s")
    logger.info(f"  Статистика моста: {bridge.get_stats()}")
    
    # === ТЕСТ 3: С ОБОГАЩЁННЫМ ПРОМПТОМ ===
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТ 3: С ОБОГАЩЁННЫМ ПРОМПТОМ ИЗ ГРАФА")
    logger.info("=" * 60)
    
    bridge2 = UnifiedCacheBridge(
        token_cache=MockTokenCache(),
        knowledge_graph=MockKnowledgeGraph(),
        cache_dir=os.path.join(_mod_dir, 'cache2')
    )
    
    llm_enriched = MockLLM(base_latency=0.5)
    enriched_times = []
    
    for query in queries:
        start = time.perf_counter()
        
        enriched_prompt = bridge2.build_enriched_prompt(query)
        _ = llm_enriched.generate(enriched_prompt)
        
        elapsed = time.perf_counter() - start
        enriched_times.append(elapsed)
        logger.info(f"  Query: {query[:40]}... -> {elapsed:.3f}s (prompt_len={len(enriched_prompt)})")
    
    avg_enriched = statistics.mean(enriched_times)
    logger.info(f"  Среднее: {avg_enriched:.3f}s")
    
    # === ТЕСТ 4: ПОЛНЫЙ ЦИКЛ (кэш генерации + граф) ===
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТ 4: ПОЛНЫЙ ЦИКЛ (кэш генерации + граф)")
    logger.info("=" * 60)
    
    bridge3 = UnifiedCacheBridge(
        token_cache=MockTokenCache(),
        knowledge_graph=MockKnowledgeGraph(),
        cache_dir=os.path.join(_mod_dir, 'cache3')
    )
    
    llm_full = MockLLM(base_latency=0.5)
    full_times = []
    
    # Первый проход - генерация
    logger.info("  Первый проход (генерация):")
    for query in queries:
        start = time.perf_counter()
        prep = bridge3.prepare_for_generation(query)
        
        if prep['cached_response']:
            elapsed = time.perf_counter() - start
            full_times.append(elapsed)
            continue
        
        response = llm_full.generate(prep['prompt'])
        bridge3.cache_generation_result(query, response)
        
        elapsed = time.perf_counter() - start
        full_times.append(elapsed)
        logger.info(f"  Query: {query[:40]}... -> {elapsed:.3f}s (generated)")
    
    avg_full_first = statistics.mean(full_times)
    logger.info(f"  Среднее (первый проход): {avg_full_first:.3f}s")
    
    # Второй проход - кэш
    logger.info("\n  Второй проход (кэш):")
    full_times_2 = []
    
    for query in queries:
        start = time.perf_counter()
        prep = bridge3.prepare_for_generation(query)
        elapsed = time.perf_counter() - start
        full_times_2.append(elapsed)
        
        if prep['cached_response']:
            logger.info(f"  Query: {query[:40]}... -> {elapsed:.3f}s (CACHED)")
        else:
            logger.info(f"  Query: {query[:40]}... -> {elapsed:.3f}s (miss)")
    
    avg_full_cached = statistics.mean(full_times_2)
    logger.info(f"  Среднее (кэш): {avg_full_cached:.3f}s")
    
    # === РЕЗУЛЬТАТЫ ===
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ")
    logger.info("=" * 60)
    
    results = {
        'baseline_avg': avg_baseline,
        'graph_cache_avg': avg_graph,
        'enriched_avg': avg_enriched,
        'full_first_avg': avg_full_first,
        'full_cached_avg': avg_full_cached,
        'graph_vs_baseline_improvement': (avg_baseline - avg_graph) / avg_baseline * 100,
        'cached_vs_baseline_speedup': avg_baseline / avg_full_cached if avg_full_cached > 0 else float('inf'),
        'bridge_stats': bridge3.get_stats(),
        'token_cache_stats': bridge3.token_cache.get_cache_stats() if bridge3.token_cache else {},
    }
    
    logger.info(f"Baseline:              {avg_baseline:.3f}s")
    logger.info(f"Graph Cache:           {avg_graph:.3f}s ({results['graph_vs_baseline_improvement']:+.1f}%)")
    logger.info(f"Enriched Prompt:       {avg_enriched:.3f}s")
    logger.info(f"Full (first pass):     {avg_full_first:.3f}s")
    logger.info(f"Full (cached):         {avg_full_cached:.3f}s (x{results['cached_vs_baseline_speedup']:.1f} быстрее baseline)")
    
    logger.info(f"\nСтатистика моста:")
    for k, v in bridge3.get_stats().items():
        if not isinstance(v, dict):
            logger.info(f"  {k}: {v}")
    
    # Сохранение
    out_path = os.path.join(_mod_dir, 'unified_benchmark.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\nСохранено: {out_path}")
    
    return results


if __name__ == '__main__':
    run_benchmark()
