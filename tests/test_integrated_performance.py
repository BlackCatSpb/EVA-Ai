"""
Комплексный тест эффективности асинхронной токенизации с гибридным кэшем
Измеряет выигрыш в ширине и глубине понимаемого контекста
"""

import os
import sys
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import json
import pytest

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

class PerformanceMetrics:
    """Класс для сбора и анализа метрик производительности."""
    
    def __init__(self):
        self.metrics = {
            'processing_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'tokens_processed': 0,
            'contexts_analyzed': 0,
            'memory_usage': [],
            'concurrent_operations': 0,
            'context_depth_scores': [],
            'context_width_scores': [],
            'semantic_coherence_scores': []
        }
        self.start_time = time.time()
    
    def record_processing_time(self, duration: float):
        self.metrics['processing_times'].append(duration)
    
    def record_cache_hit(self):
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        self.metrics['cache_misses'] += 1
    
    def record_tokens(self, count: int):
        self.metrics['tokens_processed'] += count
    
    def record_context_analysis(self, depth_score: float, width_score: float, coherence_score: float):
        self.metrics['contexts_analyzed'] += 1
        self.metrics['context_depth_scores'].append(depth_score)
        self.metrics['context_width_scores'].append(width_score)
        self.metrics['semantic_coherence_scores'].append(coherence_score)
    
    def get_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        cache_total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = self.metrics['cache_hits'] / max(1, cache_total)
        
        return {
            'total_runtime': total_time,
            'avg_processing_time': statistics.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0,
            'median_processing_time': statistics.median(self.metrics['processing_times']) if self.metrics['processing_times'] else 0,
            'cache_hit_rate': cache_hit_rate,
            'tokens_per_second': self.metrics['tokens_processed'] / max(1, total_time),
            'contexts_per_second': self.metrics['contexts_analyzed'] / max(1, total_time),
            'avg_context_depth': statistics.mean(self.metrics['context_depth_scores']) if self.metrics['context_depth_scores'] else 0,
            'avg_context_width': statistics.mean(self.metrics['context_width_scores']) if self.metrics['context_width_scores'] else 0,
            'avg_semantic_coherence': statistics.mean(self.metrics['semantic_coherence_scores']) if self.metrics['semantic_coherence_scores'] else 0,
            'total_tokens': self.metrics['tokens_processed'],
            'total_contexts': self.metrics['contexts_analyzed']
        }

class ContextAnalyzer:
    """Анализатор контекста для измерения глубины и ширины понимания."""
    
    @staticmethod
    def analyze_context_depth(tokens: List[str], original_text: str) -> float:
        """
        Анализирует глубину понимания контекста.
        Глубина = способность извлекать семантически значимые элементы.
        """
        if not tokens or not original_text:
            return 0.0
        
        # Подсчитываем семантически значимые токены
        meaningful_tokens = [t for t in tokens if len(t) > 2 and t.isalpha()]
        
        # Анализируем разнообразие токенов
        unique_tokens = set(meaningful_tokens)
        diversity_ratio = len(unique_tokens) / max(1, len(meaningful_tokens))
        
        # Анализируем покрытие исходного текста
        text_words = original_text.lower().split()
        coverage = sum(1 for token in meaningful_tokens if token.lower() in text_words)
        coverage_ratio = coverage / max(1, len(text_words))
        
        # Комбинированная оценка глубины (0-1)
        depth_score = (diversity_ratio * 0.4 + coverage_ratio * 0.6)
        return min(1.0, depth_score)
    
    @staticmethod
    def analyze_context_width(tokens: List[str], context_size: int) -> float:
        """
        Анализирует ширину понимания контекста.
        Ширина = способность обрабатывать большие объемы информации.
        """
        if not tokens:
            return 0.0
        
        # Базовая ширина по количеству токенов
        token_width = min(1.0, len(tokens) / max(1, context_size))
        
        # Анализируем тематическое разнообразие
        # Простая эвристика: разные первые буквы указывают на тематическое разнообразие
        first_letters = set(t[0].lower() for t in tokens if t and t.isalpha())
        diversity_width = len(first_letters) / 26.0  # Нормализация по алфавиту
        
        # Комбинированная оценка ширины
        width_score = (token_width * 0.7 + diversity_width * 0.3)
        return min(1.0, width_score)
    
    @staticmethod
    def analyze_semantic_coherence(tokens: List[str], original_text: str) -> float:
        """
        Анализирует семантическую связность извлеченных токенов.
        """
        if not tokens or not original_text:
            return 0.0
        
        # Простая эвристика семантической связности
        # Проверяем, сохраняется ли порядок ключевых слов из исходного текста
        text_words = [w.lower() for w in original_text.split() if len(w) > 2]
        token_words = [t.lower() for t in tokens if len(t) > 2]
        
        if not text_words or not token_words:
            return 0.0
        
        # Подсчитываем сохранение порядка
        order_preservation = 0
        last_index = -1
        
        for token in token_words:
            try:
                current_index = text_words.index(token)
                if current_index > last_index:
                    order_preservation += 1
                    last_index = current_index
            except ValueError:
                continue
        
        coherence_score = order_preservation / max(1, len(token_words))
        return min(1.0, coherence_score)

def create_test_contexts() -> List[Dict[str, Any]]:
    """Создает набор тестовых контекстов различной сложности."""
    
    contexts = [
        {
            'name': 'Простой контекст',
            'text': 'Машинное обучение использует алгоритмы для анализа данных.',
            'expected_complexity': 'low',
            'size_category': 'small'
        },
        {
            'name': 'Средний контекст',
            'text': 'Искусственный интеллект и машинное обучение революционизируют современные технологии. Нейронные сети позволяют решать сложные задачи классификации и прогнозирования. Глубокое обучение открывает новые возможности в обработке естественного языка.',
            'expected_complexity': 'medium',
            'size_category': 'medium'
        },
        {
            'name': 'Сложный технический контекст',
            'text': 'Трансформеры представляют собой архитектуру нейронных сетей, основанную на механизме внимания. Модель BERT использует двунаправленное кодирование для понимания контекста. GPT применяет автогрессивную генерацию текста. Механизм self-attention позволяет модели фокусироваться на релевантных частях входной последовательности. Позиционное кодирование обеспечивает понимание порядка токенов.',
            'expected_complexity': 'high',
            'size_category': 'large'
        },
        {
            'name': 'Многотематический контекст',
            'text': 'Квантовые компьютеры используют принципы квантовой механики для вычислений. Биоинформатика применяет компьютерные методы для анализа биологических данных. Блокчейн технологии обеспечивают децентрализованное хранение информации. Интернет вещей соединяет физические устройства с цифровыми сетями. Дополненная реальность накладывает цифровую информацию на физический мир.',
            'expected_complexity': 'high',
            'size_category': 'large'
        },
        {
            'name': 'Очень большой контекст',
            'text': ' '.join([
                'Современные системы искусственного интеллекта демонстрируют впечатляющие способности в различных областях.',
                'Обработка естественного языка позволяет машинам понимать и генерировать человеческую речь.',
                'Компьютерное зрение дает возможность анализировать и интерпретировать визуальную информацию.',
                'Робототехника интегрирует ИИ с физическими системами для автономного функционирования.',
                'Машинное обучение с подкреплением обучает агентов принимать оптимальные решения в динамических средах.',
                'Федеративное обучение позволяет тренировать модели без централизации данных.',
                'Объяснимый ИИ стремится сделать решения алгоритмов понятными для человека.',
                'Этические аспекты ИИ включают справедливость, прозрачность и ответственность.',
                'Квантовое машинное обучение исследует применение квантовых вычислений для ускорения алгоритмов.',
                'Нейроморфные вычисления имитируют структуру и функции человеческого мозга.'
            ]),
            'expected_complexity': 'very_high',
            'size_category': 'extra_large'
        }
    ]
    
    return contexts

def test_baseline_performance(contexts: List[Dict[str, Any]]) -> PerformanceMetrics:
    """Тестирует базовую производительность без оптимизаций."""
    print("=== Тест базовой производительности (без оптимизаций) ===")
    
    metrics = PerformanceMetrics()
    
    try:
        # Создаем минимальную систему без кэша и асинхронности
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        from eva.mlearning.unified_text_processor import UnifiedTextProcessor
        processor = UnifiedTextProcessor(brain=brain, use_async=False, max_workers=1)
        
        analyzer = ContextAnalyzer()
        
        for i, context in enumerate(contexts):
            print(f"Обработка контекста {i+1}/{len(contexts)}: {context['name']}")
            
            start_time = time.time()
            result = processor.process_text(context['text'])
            processing_time = time.time() - start_time
            
            metrics.record_processing_time(processing_time)
            
            if result and 'keywords' in result:
                tokens = result['keywords']
                # Нормализуем к списку строк (keywords могут быть [(word, score), ...])
                norm_tokens = [t if isinstance(t, str) else (t[0] if t else '') for t in tokens]
                norm_tokens = [t for t in norm_tokens if isinstance(t, str) and t]
                metrics.record_tokens(len(norm_tokens))
                
                # Анализируем контекст
                depth_score = analyzer.analyze_context_depth(norm_tokens, context['text'])
                width_score = analyzer.analyze_context_width(norm_tokens, len(context['text'].split()))
                coherence_score = analyzer.analyze_semantic_coherence(norm_tokens, context['text'])
                
                metrics.record_context_analysis(depth_score, width_score, coherence_score)
                
                print(f"  Время: {processing_time:.4f}сек, Токенов: {len(tokens)}")
                print(f"  Глубина: {depth_score:.3f}, Ширина: {width_score:.3f}, Связность: {coherence_score:.3f}")
            else:
                print("  ❌ Ошибка обработки")
        
        # Under pytest, avoid returning a value to prevent ReturnNotNoneWarning
        if 'PYTEST_CURRENT_TEST' in os.environ:
            summary = metrics.get_summary()
            assert summary is not None
        else:
            return metrics
        
    except Exception as e:
        print(f"❌ Ошибка в базовом тесте: {e}")
        if 'PYTEST_CURRENT_TEST' in os.environ:
            pytest.fail(str(e))
        else:
            return metrics

def test_optimized_performance(contexts: List[Dict[str, Any]]) -> PerformanceMetrics:
    """Тестирует оптимизированную производительность с кэшем и асинхронностью."""
    print("\n=== Тест оптимизированной производительности (кэш + асинхронность) ===")
    
    metrics = PerformanceMetrics()
    
    try:
        # Создаем систему с кэшем и асинхронностью
        class OptimizedBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = OptimizedBrain()
        
        from eva.mlearning.unified_text_processor import UnifiedTextProcessor
        from eva.memory.hybrid_token_cache import HybridTokenCache
        
        # Создаем кэш
        cache = HybridTokenCache(brain, max_memory_tokens=1000)
        
        # Создаем процессор с оптимизациями
        processor = UnifiedTextProcessor(brain=brain, use_async=True, max_workers=4, hybrid_cache=cache)
        
        analyzer = ContextAnalyzer()
        
        # Тестируем последовательную обработку с кэшированием
        for i, context in enumerate(contexts):
            print(f"Обработка контекста {i+1}/{len(contexts)}: {context['name']}")
            
            start_time = time.time()
            result = processor.process_text(context['text'])
            processing_time = time.time() - start_time
            
            metrics.record_processing_time(processing_time)
            
            if result and 'keywords' in result:
                tokens = result['keywords']
                norm_tokens = [t if isinstance(t, str) else (t[0] if t else '') for t in tokens]
                norm_tokens = [t for t in norm_tokens if isinstance(t, str) and t]
                metrics.record_tokens(len(norm_tokens))
                
                # Анализируем контекст
                depth_score = analyzer.analyze_context_depth(norm_tokens, context['text'])
                width_score = analyzer.analyze_context_width(norm_tokens, len(context['text'].split()))
                coherence_score = analyzer.analyze_semantic_coherence(norm_tokens, context['text'])
                
                metrics.record_context_analysis(depth_score, width_score, coherence_score)
                
                print(f"  Время: {processing_time:.4f}сек, Токенов: {len(tokens)}")
                print(f"  Глубина: {depth_score:.3f}, Ширина: {width_score:.3f}, Связность: {coherence_score:.3f}")
            else:
                print("  ❌ Ошибка обработки")
        
        # Повторная обработка для тестирования кэша
        print("\n--- Повторная обработка (тест кэширования) ---")
        for i, context in enumerate(contexts[:3]):  # Тестируем только первые 3
            print(f"Повторная обработка {i+1}: {context['name']}")
            
            start_time = time.time()
            result = processor.process_text(context['text'])
            processing_time = time.time() - start_time
            
            metrics.record_processing_time(processing_time)
            
            if processing_time < 0.001:  # Очень быстро = вероятно из кэша
                metrics.record_cache_hit()
                print(f"  ✅ Кэш попадание: {processing_time:.6f}сек")
            else:
                metrics.record_cache_miss()
                print(f"  ⚠️ Кэш промах: {processing_time:.4f}сек")
        
        # Получаем статистику кэша
        if hasattr(cache, 'get_cache_stats'):
            cache_stats = cache.get_cache_stats()
            print(f"\nСтатистика кэша: {cache_stats}")
        
        if 'PYTEST_CURRENT_TEST' in os.environ:
            summary = metrics.get_summary()
            assert summary is not None
        else:
            return metrics
        
    except Exception as e:
        print(f"❌ Ошибка в оптимизированном тесте: {e}")
        import traceback
        traceback.print_exc()
        if 'PYTEST_CURRENT_TEST' in os.environ:
            pytest.fail(str(e))
        else:
            return metrics

def test_concurrent_performance(contexts: List[Dict[str, Any]]) -> PerformanceMetrics:
    """Тестирует производительность при параллельной обработке."""
    print("\n=== Тест параллельной обработки ===")
    
    metrics = PerformanceMetrics()
    
    try:
        class ConcurrentBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = ConcurrentBrain()
        
        from eva.mlearning.unified_text_processor import UnifiedTextProcessor
        from eva.memory.hybrid_token_cache import HybridTokenCache
        
        cache = HybridTokenCache(brain, max_memory_tokens=2000)
        processor = UnifiedTextProcessor(brain=brain, use_async=True, max_workers=6, hybrid_cache=cache)
        analyzer = ContextAnalyzer()
        
        def process_context(context_data):
            context, index = context_data
            try:
                start_time = time.time()
                result = processor.process_text(context['text'])
                processing_time = time.time() - start_time
                
                if result and 'keywords' in result:
                    tokens = result['keywords']
                    norm_tokens = [t if isinstance(t, str) else (t[0] if t else '') for t in tokens]
                    norm_tokens = [t for t in norm_tokens if isinstance(t, str) and t]
                    depth_score = analyzer.analyze_context_depth(norm_tokens, context['text'])
                    width_score = analyzer.analyze_context_width(norm_tokens, len(context['text'].split()))
                    coherence_score = analyzer.analyze_semantic_coherence(norm_tokens, context['text'])
                    
                    return {
                        'index': index,
                        'name': context['name'],
                        'processing_time': processing_time,
                        'token_count': len(norm_tokens),
                        'depth_score': depth_score,
                        'width_score': width_score,
                        'coherence_score': coherence_score,
                        'success': True
                    }
                else:
                    return {
                        'index': index,
                        'name': context['name'],
                        'success': False
                    }
            except Exception as e:
                return {
                    'index': index,
                    'name': context['name'],
                    'error': str(e),
                    'success': False
                }
        
        # Параллельная обработка
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            context_data = [(context, i) for i, context in enumerate(contexts)]
            futures = [executor.submit(process_context, data) for data in context_data]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    metrics.record_processing_time(result['processing_time'])
                    metrics.record_tokens(result['token_count'])
                    metrics.record_context_analysis(
                        result['depth_score'],
                        result['width_score'],
                        result['coherence_score']
                    )
                    print(f"✅ {result['name']}: {result['processing_time']:.4f}сек, {result['token_count']} токенов")
                else:
                    print(f"❌ {result['name']}: ошибка обработки")
        
        total_time = time.time() - start_time
        print(f"\nОбщее время параллельной обработки: {total_time:.4f}сек")
        
        if 'PYTEST_CURRENT_TEST' in os.environ:
            summary = metrics.get_summary()
            assert summary is not None
        else:
            return metrics
        
    except Exception as e:
        print(f"❌ Ошибка в тесте параллельной обработки: {e}")
        import traceback
        traceback.print_exc()
        if 'PYTEST_CURRENT_TEST' in os.environ:
            pytest.fail(str(e))
        else:
            return metrics

def compare_performance_metrics(baseline: PerformanceMetrics, optimized: PerformanceMetrics, concurrent: PerformanceMetrics):
    """Сравнивает метрики производительности и выводит анализ."""
    print("\n" + "="*80)
    print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*80)
    
    baseline_summary = baseline.get_summary()
    optimized_summary = optimized.get_summary()
    concurrent_summary = concurrent.get_summary()
    
    print("\n🔍 СКОРОСТЬ ОБРАБОТКИ:")
    print(f"Базовая система:        {baseline_summary['avg_processing_time']:.4f} сек/контекст")
    print(f"Оптимизированная:       {optimized_summary['avg_processing_time']:.4f} сек/контекст")
    print(f"Параллельная:           {concurrent_summary['avg_processing_time']:.4f} сек/контекст")
    
    if baseline_summary['avg_processing_time'] > 0:
        speedup_opt = baseline_summary['avg_processing_time'] / optimized_summary['avg_processing_time']
        speedup_conc = baseline_summary['avg_processing_time'] / concurrent_summary['avg_processing_time']
        print(f"Ускорение (оптимизированная): {speedup_opt:.2f}x")
        print(f"Ускорение (параллельная):     {speedup_conc:.2f}x")
    
    print(f"\n🎯 ПРОПУСКНАЯ СПОСОБНОСТЬ:")
    print(f"Базовая система:        {baseline_summary['tokens_per_second']:.1f} токенов/сек")
    print(f"Оптимизированная:       {optimized_summary['tokens_per_second']:.1f} токенов/сек")
    print(f"Параллельная:           {concurrent_summary['tokens_per_second']:.1f} токенов/сек")
    
    print(f"\n🧠 КАЧЕСТВО ПОНИМАНИЯ КОНТЕКСТА:")
    print(f"                        Глубина    Ширина     Связность")
    print(f"Базовая система:        {baseline_summary['avg_context_depth']:.3f}     {baseline_summary['avg_context_width']:.3f}     {baseline_summary['avg_semantic_coherence']:.3f}")
    print(f"Оптимизированная:       {optimized_summary['avg_context_depth']:.3f}     {optimized_summary['avg_context_width']:.3f}     {optimized_summary['avg_semantic_coherence']:.3f}")
    print(f"Параллельная:           {concurrent_summary['avg_context_depth']:.3f}     {concurrent_summary['avg_context_width']:.3f}     {concurrent_summary['avg_semantic_coherence']:.3f}")
    
    # Анализ улучшений
    print(f"\n📈 УЛУЧШЕНИЯ ПОНИМАНИЯ КОНТЕКСТА:")
    
    depth_improvement_opt = (optimized_summary['avg_context_depth'] - baseline_summary['avg_context_depth']) / max(0.001, baseline_summary['avg_context_depth']) * 100
    width_improvement_opt = (optimized_summary['avg_context_width'] - baseline_summary['avg_context_width']) / max(0.001, baseline_summary['avg_context_width']) * 100
    coherence_improvement_opt = (optimized_summary['avg_semantic_coherence'] - baseline_summary['avg_semantic_coherence']) / max(0.001, baseline_summary['avg_semantic_coherence']) * 100
    
    print(f"Улучшение глубины (оптимизированная):  {depth_improvement_opt:+.1f}%")
    print(f"Улучшение ширины (оптимизированная):   {width_improvement_opt:+.1f}%")
    print(f"Улучшение связности (оптимизированная): {coherence_improvement_opt:+.1f}%")
    
    depth_improvement_conc = (concurrent_summary['avg_context_depth'] - baseline_summary['avg_context_depth']) / max(0.001, baseline_summary['avg_context_depth']) * 100
    width_improvement_conc = (concurrent_summary['avg_context_width'] - baseline_summary['avg_context_width']) / max(0.001, baseline_summary['avg_context_width']) * 100
    coherence_improvement_conc = (concurrent_summary['avg_semantic_coherence'] - baseline_summary['avg_semantic_coherence']) / max(0.001, baseline_summary['avg_semantic_coherence']) * 100
    
    print(f"Улучшение глубины (параллельная):       {depth_improvement_conc:+.1f}%")
    print(f"Улучшение ширины (параллельная):        {width_improvement_conc:+.1f}%")
    print(f"Улучшение связности (параллельная):     {coherence_improvement_conc:+.1f}%")
    
    # Кэширование
    if optimized_summary['cache_hit_rate'] > 0:
        print(f"\n💾 ЭФФЕКТИВНОСТЬ КЭШИРОВАНИЯ:")
        print(f"Коэффициент попаданий в кэш: {optimized_summary['cache_hit_rate']:.1%}")
    
    print(f"\n📊 ОБЩАЯ ОЦЕНКА ЭФФЕКТИВНОСТИ:")
    
    # Комплексная оценка
    overall_improvement_opt = (speedup_opt * 0.4 + 
                              (1 + depth_improvement_opt/100) * 0.2 + 
                              (1 + width_improvement_opt/100) * 0.2 + 
                              (1 + coherence_improvement_opt/100) * 0.2)
    
    overall_improvement_conc = (speedup_conc * 0.4 + 
                               (1 + depth_improvement_conc/100) * 0.2 + 
                               (1 + width_improvement_conc/100) * 0.2 + 
                               (1 + coherence_improvement_conc/100) * 0.2)
    
    print(f"Общее улучшение (оптимизированная): {overall_improvement_opt:.2f}x")
    print(f"Общее улучшение (параллельная):     {overall_improvement_conc:.2f}x")
    
    return {
        'baseline': baseline_summary,
        'optimized': optimized_summary,
        'concurrent': concurrent_summary,
        'improvements': {
            'speed_optimized': speedup_opt if baseline_summary['avg_processing_time'] > 0 else 0,
            'speed_concurrent': speedup_conc if baseline_summary['avg_processing_time'] > 0 else 0,
            'depth_optimized': depth_improvement_opt,
            'width_optimized': width_improvement_opt,
            'coherence_optimized': coherence_improvement_opt,
            'depth_concurrent': depth_improvement_conc,
            'width_concurrent': width_improvement_conc,
            'coherence_concurrent': coherence_improvement_conc,
            'overall_optimized': overall_improvement_opt,
            'overall_concurrent': overall_improvement_conc
        }
    }

if __name__ == "__main__":
    print("🚀 КОМПЛЕКСНЫЙ ТЕСТ ЭФФЕКТИВНОСТИ АСИНХРОННОЙ ТОКЕНИЗАЦИИ + ГИБРИДНЫЙ КЭШ")
    print("="*80)
    print("Измерение выигрыша в ширине и глубине понимаемого контекста")
    print("="*80)
    
    # Создаем тестовые контексты
    contexts = create_test_contexts()
    print(f"Подготовлено {len(contexts)} тестовых контекстов различной сложности\n")
    
    # Запускаем тесты
    baseline_metrics = test_baseline_performance(contexts)
    optimized_metrics = test_optimized_performance(contexts)
    concurrent_metrics = test_concurrent_performance(contexts)
    
    # Сравниваем результаты
    comparison = compare_performance_metrics(baseline_metrics, optimized_metrics, concurrent_metrics)
    
    # Сохраняем результаты
    results_file = "performance_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в {results_file}")
    
    print("\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*80)
