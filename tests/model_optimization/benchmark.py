"""
Advanced Test Runner - сравнение качества и скорости генерации.
Запускает базовую и оптимизированную версии, сравнивает метрики.
"""

import os
import sys
import time
import json
import logging
import threading
import statistics
import importlib.util
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Добавляем путь для импорта локальных модулей
_mod_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _mod_dir)

logger = logging.getLogger(__name__)


def _import_module_from_file(module_name: str, file_path: str):
    """Динамический импорт модуля из файла."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class GenerationResult:
    """Результат генерации."""
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    quality_score: float = 0.0


@dataclass
class TestReport:
    """Отчет тестирования."""
    test_name: str
    baseline_stats: Dict[str, float]
    optimized_stats: Dict[str, float]
    comparison: Dict[str, float]
    recommendations: List[str]


class QualityEvaluator:
    """Оценщик качества генерации."""
    
    def __init__(self):
        self.russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        self.punctuation = set('.,!?;:"()[]{}')
    
    def evaluate(self, generated_text: str, reference_text: str = None) -> Dict[str, float]:
        """Оценка качества текста."""
        scores = {}
        
        # 1. Отсутствие иностранных символов (для русского языка)
        foreign_chars = sum(1 for c in generated_text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in generated_text if c.isalpha())
        scores['language_purity'] = 1.0 - (foreign_chars / max(total_chars, 1)) if total_chars > 0 else 1.0
        
        # 2. Пунктуация
        punct_count = sum(1 for c in generated_text if c in self.punctuation)
        scores['punctuation_ratio'] = punct_count / max(len(generated_text), 1)
        
        # 3. Длина (не слишком короткий/длинный)
        words = generated_text.split()
        if 5 <= len(words) <= 100:
            scores['length_score'] = 1.0
        elif len(words) < 5:
            scores['length_score'] = len(words) / 5
        else:
            scores['length_score'] = max(0, 1.0 - (len(words) - 100) / 100)
        
        # 4. Уникальность слов
        unique_ratio = len(set(words)) / max(len(words), 1)
        scores['uniqueness'] = unique_ratio
        
        # 5. Общий скор
        scores['overall'] = statistics.mean([
            scores['language_purity'],
            scores['length_score'],
            scores['uniqueness']
        ])
        
        return scores
    
    def compare_responses(self, baseline: str, optimized: str) -> Dict[str, Any]:
        """Сравнение двух ответов."""
        base_scores = self.evaluate(baseline)
        opt_scores = self.evaluate(optimized)
        
        return {
            'baseline_score': base_scores['overall'],
            'optimized_score': opt_scores['overall'],
            'improvement': opt_scores['overall'] - base_scores['overall'],
            'details': {
                'baseline': base_scores,
                'optimized': opt_scores
            }
        }


class PerformanceProfiler:
    """Профилировщик производительности."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    def start_operation(self, name: str) -> float:
        """Начать замер операции."""
        return time.perf_counter()
    
    def end_operation(self, name: str, start_time: float):
        """Завершить замер операции."""
        elapsed = time.perf_counter() - start_time
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Получить статистику по всем операциям."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'mean_ms': statistics.mean(times) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'stdev_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                    'count': len(times)
                }
        return stats


class ModelBenchmark:
    """Бенчмарк модели с базовой и оптимизированной версиями."""
    
    def __init__(
        self,
        model_path: str = None,
        use_real_model: bool = False
    ):
        self.model_path = model_path
        self.use_real_model = use_real_model
        self.llama_model = None
        
        self.quality_evaluator = QualityEvaluator()
        self.profiler = PerformanceProfiler()
        
        self._init_model()
    
    def _init_model(self):
        """Инициализация модели (реальной или симуляции)."""
        if self.use_real_model and self.model_path and os.path.exists(self.model_path):
            try:
                from llama_cpp import Llama
                self.llama_model = Llama(
                    model_path=self.model_path,
                    n_ctx=512,
                    n_threads=4
                )
                logger.info(f"Загружена реальная модель: {self.model_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель: {e}")
                self.use_real_model = False
        else:
            logger.info("Используется симуляция модели")
    
    def generate_baseline(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7
    ) -> GenerationResult:
        """Базовая генерация (без оптимизаций)."""
        start = self.profiler.start_operation('baseline_total')
        
        # Токенизация
        tok_start = self.profiler.start_operation('baseline_tokenize')
        tokens = self._tokenize(prompt)
        self.profiler.end_operation('baseline_tokenize', tok_start)
        
        # Генерация
        gen_start = self.profiler.start_operation('baseline_generate')
        if self.llama_model:
            output = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=['</s>']
            )
            text = output['choices'][0]['text']
        else:
            text = self._simulate_generation(prompt, max_tokens)
        self.profiler.end_operation('baseline_generate', gen_start)
        
        self.profiler.end_operation('baseline_total', start)
        
        tokens_generated = len(text.split())
        time_taken = time.perf_counter() - start
        
        return GenerationResult(
            text=text,
            tokens_generated=tokens_generated,
            time_seconds=time_taken,
            tokens_per_second=tokens_generated / max(time_taken, 0.001),
            quality_score=self.quality_evaluator.evaluate(text)['overall']
        )
    
    def generate_optimized(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        use_cache: bool = True,
        use_predictor: bool = True
    ) -> GenerationResult:
        """Оптимизированная генерация."""
        # Импорт локальных модулей
        index_mod = _import_module_from_file('context_index', 
            os.path.join(_mod_dir, 'index', 'context_index.py'))
        embed_mod = _import_module_from_file('fast_embedder',
            os.path.join(_mod_dir, 'embeddings', 'fast_embedder.py'))
        
        ContextIndex = index_mod.ContextIndex
        FastEmbedder = embed_mod.FastEmbedder
        
        start = self.profiler.start_operation('optimized_total')
        
        # Инициализация оптимизаций
        index = ContextIndex()
        embedder = FastEmbedder()
        
        # Токенизация с кэшем
        tok_start = self.profiler.start_operation('optimized_tokenize')
        tokens, cached = index.tokenize_with_cache(prompt)
        self.profiler.end_operation('optimized_tokenize', tok_start)
        
        # Предсказание следующих токенов (опционально)
        if use_predictor:
            pred_start = self.profiler.start_operation('optimized_predict')
            predictions = index.predict_next_tokens(tokens[:5])
            self.profiler.end_operation('optimized_predict', pred_start)
        
        # Генерация с использованием эмбеддинга
        emb_start = self.profiler.start_operation('optimized_embed')
        embedding = embedder.get_embedding(tokens)
        self.profiler.end_operation('optimized_embed', emb_start)
        
        # Генерация
        gen_start = self.profiler.start_operation('optimized_generate')
        if self.llama_model:
            output = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=['</s>']
            )
            text = output['choices'][0]['text']
        else:
            text = self._simulate_generation(prompt, max_tokens)
        self.profiler.end_operation('optimized_generate', gen_start)
        
        self.profiler.end_operation('optimized_total', start)
        
        tokens_generated = len(text.split())
        time_taken = time.perf_counter() - start
        
        return GenerationResult(
            text=text,
            tokens_generated=tokens_generated,
            time_seconds=time_taken,
            tokens_per_second=tokens_generated / max(time_taken, 0.001),
            quality_score=self.quality_evaluator.evaluate(text)['overall']
        )
    
    def _tokenize(self, text: str) -> List[int]:
        """Токенизация (базовая)."""
        return [hash(c) % 50000 for c in text]
    
    def _simulate_generation(self, prompt: str, max_tokens: int) -> str:
        """Симуляция генерации."""
        time.sleep(0.01)  # Имитация работы модели
        
        templates = [
            "Это интересный вопрос. {topic} развивается очень быстро.",
            "Согласно последним исследованиям, {topic} имеет большой потенциал.",
            "Основные направления {topic} включают машинное обучение и нейросети.",
            "Для развития {topic} в России нужно больше инвестиций в исследования.",
            "Современные технологии {topic} позволяют решать сложные задачи."
        ]
        
        words = prompt.split()[:3]
        topic = ' '.join(words) if words else 'ИИ'
        
        import random
        template = random.choice(templates)
        return template.format(topic=topic)
    
    def run_comparison(
        self,
        test_prompts: List[str],
        num_runs: int = 5
    ) -> TestReport:
        """Запуск полного сравнения базовой и оптимизированной версий."""
        logger.info(f"Запуск сравнения: {len(test_prompts)} промптов, {num_runs} запусков")
        
        baseline_results: List[GenerationResult] = []
        optimized_results: List[GenerationResult] = []
        
        for run in range(num_runs):
            for prompt in test_prompts:
                # Базовая генерация
                baseline_result = self.generate_baseline(prompt)
                baseline_results.append(baseline_result)
                
                # Оптимизированная генерация
                optimized_result = self.generate_optimized(prompt)
                optimized_results.append(optimized_result)
        
        # Статистика
        baseline_stats = self._calc_stats(baseline_results)
        optimized_stats = self._calc_stats(optimized_results)
        
        # Сравнение
        speedup = baseline_stats['avg_time'] / optimized_stats['avg_time'] if optimized_stats['avg_time'] > 0 else 1
        quality_delta = optimized_stats['avg_quality'] - baseline_stats['avg_quality']
        
        # Рекомендации
        recommendations = []
        if speedup > 1.2:
            recommendations.append(f"Оптимизация ускорила генерацию в {speedup:.2f}x раз")
        if quality_delta > 0.05:
            recommendations.append(f"Качество улучшилось на {quality_delta*100:.1f}%")
        if speedup < 1.0:
            recommendations.append("ВНИМАНИЕ: оптимизация замедлила генерацию")
        
        comparison = {
            'speedup_x': speedup,
            'quality_delta': quality_delta,
            'time_baseline_ms': baseline_stats['avg_time'] * 1000,
            'time_optimized_ms': optimized_stats['avg_time'] * 1000,
            'tokens_per_sec_baseline': baseline_stats['avg_tokens_per_sec'],
            'tokens_per_sec_optimized': optimized_stats['avg_tokens_per_sec']
        }
        
        # Профилировка
        perf_stats = self.profiler.get_stats()
        logger.info(f"Производительность: {perf_stats}")
        
        return TestReport(
            test_name="Model Optimization Benchmark",
            baseline_stats=baseline_stats,
            optimized_stats=optimized_stats,
            comparison=comparison,
            recommendations=recommendations
        )
    
    def _calc_stats(self, results: List[GenerationResult]) -> Dict[str, float]:
        """Расчет статистики результатов."""
        return {
            'avg_time': statistics.mean([r.time_seconds for r in results]),
            'min_time': min([r.time_seconds for r in results]),
            'max_time': max([r.time_seconds for r in results]),
            'avg_tokens_per_sec': statistics.mean([r.tokens_per_second for r in results]),
            'avg_quality': statistics.mean([r.quality_score for r in results]),
            'total_generations': len(results)
        }


def run_full_benchmark():
    """Запуск полного бенчмарка."""
    logger.info("=" * 60)
    logger.info("FULL MODEL OPTIMIZATION BENCHMARK")
    logger.info("=" * 60)
    
    test_prompts = [
        "Как развить искусственный интеллект в России?",
        "Какие технологии машинного обучения самые эффективные?",
        "Что такое фрактальная память и как она работает?",
        "Какие преимущества у квантованных моделей?",
        "Как улучшить качество генерации текста?",
    ]
    
    benchmark = ModelBenchmark(use_real_model=False)
    
    report = benchmark.run_comparison(test_prompts, num_runs=3)
    
    # Вывод результатов
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nBaseline:")
    for k, v in report.baseline_stats.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\nOptimized:")
    for k, v in report.optimized_stats.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\nComparison:")
    for k, v in report.comparison.items():
        logger.info(f"  {k}: {v:.4f}")
    
    if report.recommendations:
        logger.info(f"\nRecommendations:")
        for rec in report.recommendations:
            logger.info(f"  - {rec}")
    
    # Сохранение отчета
    report_file = 'tests/model_optimization/benchmark_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_name': report.test_name,
            'baseline_stats': report.baseline_stats,
            'optimized_stats': report.optimized_stats,
            'comparison': report.comparison,
            'recommendations': report.recommendations
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nОтчет сохранен: {report_file}")
    logger.info("=" * 60)
    
    return report


if __name__ == '__main__':
    run_full_benchmark()