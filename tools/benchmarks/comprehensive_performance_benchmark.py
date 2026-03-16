"""
Комплексный бенчмарк производительности CogniFlex
Измеряет эффективность асинхронной токенизации + гибридный кэш
Анализирует выигрыш в ширине и глубине понимания контекста
"""

import os
import sys
import time
import threading
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import hashlib

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

@dataclass
class BenchmarkResult:
    """Результат бенчмарка."""
    scenario: str
    avg_processing_time: float
    median_processing_time: float
    total_tokens: int
    tokens_per_second: float
    cache_hit_rate: float
    context_depth_score: float
    context_width_score: float
    semantic_coherence_score: float
    memory_efficiency: float
    concurrent_capacity: int

class ContextAnalyzer:
    """Продвинутый анализатор контекста."""
    
    @staticmethod
    def calculate_depth_score(tokens: List[str], original_text: str) -> float:
        """
        Глубина понимания = способность извлекать семантически значимые элементы
        Учитывает: разнообразие токенов, семантическую значимость, покрытие ключевых концепций
        """
        if not tokens or not original_text:
            return 0.0
        
        # Извлекаем только значимые токены (длина > 2, буквенные)
        meaningful_tokens = [t for t in tokens if len(t) > 2 and t.isalpha()]
        if not meaningful_tokens:
            return 0.0
        
        # 1. Разнообразие токенов (уникальность)
        unique_ratio = len(set(meaningful_tokens)) / len(meaningful_tokens)
        
        # 2. Семантическое покрытие (покрытие слов исходного текста)
        text_words = [w.lower() for w in original_text.split() if len(w) > 2]
        if not text_words:
            return unique_ratio * 0.5
        
        coverage_count = sum(1 for token in meaningful_tokens 
                           if any(token.lower() in word.lower() or word.lower() in token.lower() 
                                 for word in text_words))
        coverage_ratio = coverage_count / len(text_words)
        
        # 3. Концептуальная плотность (отношение значимых токенов к общему количеству слов)
        density_ratio = len(meaningful_tokens) / len(original_text.split())
        density_score = min(1.0, density_ratio)
        
        # Комбинированная оценка глубины
        depth_score = (unique_ratio * 0.3 + coverage_ratio * 0.5 + density_score * 0.2)
        return min(1.0, depth_score)
    
    @staticmethod
    def calculate_width_score(tokens: List[str], text_length: int) -> float:
        """
        Ширина понимания = способность обрабатывать большие объемы информации
        Учитывает: количество обработанных токенов, тематическое разнообразие, масштабируемость
        """
        if not tokens:
            return 0.0
        
        # 1. Масштаб обработки (нормализованное количество токенов)
        scale_score = min(1.0, len(tokens) / max(1, text_length / 10))
        
        # 2. Тематическое разнообразие (разные начальные буквы как индикатор тематик)
        if tokens:
            first_letters = set(t[0].lower() for t in tokens if t and t.isalpha())
            diversity_score = len(first_letters) / 26.0  # Нормализация по алфавиту
        else:
            diversity_score = 0.0
        
        # 3. Лексическое богатство (средняя длина токенов)
        if tokens:
            avg_token_length = sum(len(t) for t in tokens) / len(tokens)
            richness_score = min(1.0, avg_token_length / 8.0)  # Нормализация к 8 символам
        else:
            richness_score = 0.0
        
        # Комбинированная оценка ширины
        width_score = (scale_score * 0.5 + diversity_score * 0.3 + richness_score * 0.2)
        return min(1.0, width_score)
    
    @staticmethod
    def calculate_coherence_score(tokens: List[str], original_text: str) -> float:
        """
        Семантическая связность = сохранение смысловых связей между токенами
        Учитывает: порядок токенов, контекстуальные связи, смысловую целостность
        """
        if not tokens or not original_text:
            return 0.0
        
        text_words = original_text.lower().split()
        token_words = [t.lower() for t in tokens]
        
        if not text_words or not token_words:
            return 0.0
        
        # 1. Сохранение порядка (последовательность токенов соответствует тексту)
        order_preservation = 0
        last_position = -1
        
        for token in token_words:
            # Ищем позицию токена в исходном тексте
            for i, word in enumerate(text_words):
                if token in word or word in token:
                    if i > last_position:
                        order_preservation += 1
                        last_position = i
                    break
        
        order_score = order_preservation / max(1, len(token_words))
        
        # 2. Контекстуальная близость (токены из близких частей текста)
        proximity_score = 0.0
        if len(token_words) > 1:
            proximity_distances = []
            for i in range(len(token_words) - 1):
                pos1 = next((j for j, word in enumerate(text_words) 
                           if token_words[i] in word or word in token_words[i]), -1)
                pos2 = next((j for j, word in enumerate(text_words) 
                           if token_words[i+1] in word or word in token_words[i+1]), -1)
                
                if pos1 != -1 and pos2 != -1:
                    distance = abs(pos2 - pos1)
                    proximity_distances.append(min(1.0, 1.0 / (distance + 1)))
            
            if proximity_distances:
                proximity_score = statistics.mean(proximity_distances)
        
        # 3. Смысловая полнота (покрытие ключевых частей речи)
        # Простая эвристика: наличие существительных, глаголов, прилагательных
        pos_indicators = {
            'nouns': sum(1 for t in token_words if len(t) > 4),  # Длинные слова часто существительные
            'verbs': sum(1 for t in token_words if t.endswith(('ать', 'ить', 'еть', 'ует', 'ает'))),
            'adjectives': sum(1 for t in token_words if t.endswith(('ный', 'ский', 'ной', 'ая', 'ое')))
        }
        
        completeness_score = min(1.0, sum(1 for count in pos_indicators.values() if count > 0) / 3.0)
        
        # Комбинированная оценка связности
        coherence_score = (order_score * 0.4 + proximity_score * 0.3 + completeness_score * 0.3)
        return min(1.0, coherence_score)

class PerformanceBenchmark:
    """Класс для проведения комплексного бенчмарка."""
    
    def __init__(self):
        self.analyzer = ContextAnalyzer()
        self.results = {}
    
    def create_test_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Создает тестовые наборы данных различной сложности."""
        
        datasets = {
            'simple': [
                {
                    'text': 'Машинное обучение использует алгоритмы для анализа данных.',
                    'category': 'technical_basic',
                    'expected_tokens': 6
                },
                {
                    'text': 'Искусственный интеллект помогает решать сложные задачи.',
                    'category': 'technical_basic', 
                    'expected_tokens': 7
                },
                {
                    'text': 'Python является популярным языком программирования.',
                    'category': 'technical_basic',
                    'expected_tokens': 5
                }
            ],
            
            'medium': [
                {
                    'text': 'Нейронные сети глубокого обучения революционизируют обработку естественного языка. Трансформеры показывают выдающиеся результаты в задачах понимания текста.',
                    'category': 'technical_advanced',
                    'expected_tokens': 15
                },
                {
                    'text': 'Квантовые вычисления открывают новые возможности для криптографии и оптимизации. Квантовые алгоритмы могут решать определенные задачи экспоненциально быстрее.',
                    'category': 'scientific',
                    'expected_tokens': 16
                },
                {
                    'text': 'Блокчейн технологии обеспечивают децентрализованное хранение данных. Смарт-контракты автоматизируют выполнение соглашений без посредников.',
                    'category': 'fintech',
                    'expected_tokens': 14
                }
            ],
            
            'complex': [
                {
                    'text': 'Архитектура трансформеров основана на механизме самовнимания, который позволяет модели динамически фокусироваться на релевантных частях входной последовательности. Позиционное кодирование обеспечивает понимание порядка токенов, а многоголовое внимание захватывает различные типы зависимостей между элементами последовательности.',
                    'category': 'deep_technical',
                    'expected_tokens': 25
                },
                {
                    'text': 'Федеративное обучение представляет парадигму машинного обучения, где модель тренируется на децентрализованных данных без их централизации. Дифференциальная приватность защищает индивидуальную информацию участников, а агрегация градиентов обеспечивает конвергенцию глобальной модели.',
                    'category': 'privacy_ml',
                    'expected_tokens': 22
                },
                {
                    'text': 'Квантовое машинное обучение исследует применение квантовых вычислений для ускорения алгоритмов обучения. Квантовые нейронные сети используют суперпозицию и запутанность для представления и обработки информации в экспоненциально большем пространстве состояний.',
                    'category': 'quantum_ml',
                    'expected_tokens': 24
                }
            ],
            
            'large_context': [
                {
                    'text': ' '.join([
                        'Современные системы искусственного интеллекта демонстрируют впечатляющие способности в различных областях применения.',
                        'Обработка естественного языка позволяет машинам понимать, анализировать и генерировать человеческую речь с высокой точностью.',
                        'Компьютерное зрение обеспечивает интерпретацию визуальной информации, включая распознавание объектов, сегментацию изображений и анализ сцен.',
                        'Робототехника интегрирует ИИ с физическими системами для создания автономных агентов, способных взаимодействовать с окружающей средой.',
                        'Машинное обучение с подкреплением обучает агентов принимать оптимальные решения в динамических и неопределенных средах.',
                        'Объяснимый искусственный интеллект стремится сделать решения алгоритмов понятными и интерпретируемыми для человека.',
                        'Этические аспекты ИИ включают вопросы справедливости, прозрачности, ответственности и влияния на общество.',
                        'Нейроморфные вычисления имитируют структуру и функционирование человеческого мозга для создания энергоэффективных систем.',
                        'Автономные системы используют ИИ для независимого функционирования в сложных и изменяющихся условиях.',
                        'Мультимодальное обучение объединяет информацию из различных источников данных для более полного понимания контекста.'
                    ]),
                    'category': 'comprehensive_ai',
                    'expected_tokens': 50
                }
            ]
        }
        
        return datasets
    
    def benchmark_scenario(self, scenario_name: str, texts: List[Dict], use_cache: bool = False, 
                          use_async: bool = False, max_workers: int = 1) -> BenchmarkResult:
        """Проводит бенчмарк для конкретного сценария."""
        
        print(f"\n🔬 Бенчмарк сценария: {scenario_name}")
        print(f"   Кэш: {'✅' if use_cache else '❌'}, Асинхронность: {'✅' if use_async else '❌'}, Воркеры: {max_workers}")
        
        try:
            # Создаем систему
            class BenchmarkBrain:
                def __init__(self):
                    self.cache_dir = os.path.join(os.path.dirname(__file__), "benchmark_cache")
                    os.makedirs(self.cache_dir, exist_ok=True)
                    self.components = {}
            
            brain = BenchmarkBrain()
            
            from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
            from cogniflex.memory.hybrid_token_cache import HybridTokenCache
            
            # Настраиваем компоненты
            cache = HybridTokenCache(brain, max_memory_tokens=1000) if use_cache else None
            processor = UnifiedTextProcessor(
                brain=brain, 
                use_async=use_async, 
                max_workers=max_workers,
                hybrid_cache=cache
            )
            
            # Метрики
            processing_times = []
            total_tokens = 0
            cache_hits = 0
            cache_misses = 0
            context_scores = {'depth': [], 'width': [], 'coherence': []}
            
            # Обрабатываем тексты
            start_time = time.time()
            
            for i, text_data in enumerate(texts):
                text = text_data['text']
                print(f"   Обработка {i+1}/{len(texts)}: {len(text)} символов")
                
                # Измеряем время обработки
                process_start = time.time()
                result = processor.process_text(text)
                process_time = time.time() - process_start
                
                processing_times.append(process_time)
                
                if result and 'keywords' in result:
                    # Обрабатываем токены
                    raw_keywords = result['keywords']
                    if isinstance(raw_keywords, list) and raw_keywords:
                        if isinstance(raw_keywords[0], tuple):
                            tokens = [kw[0] for kw in raw_keywords]
                        else:
                            tokens = [str(kw) for kw in raw_keywords]
                    else:
                        tokens = []
                    
                    total_tokens += len(tokens)
                    
                    # Анализируем качество контекста
                    depth_score = self.analyzer.calculate_depth_score(tokens, text)
                    width_score = self.analyzer.calculate_width_score(tokens, len(text))
                    coherence_score = self.analyzer.calculate_coherence_score(tokens, text)
                    
                    context_scores['depth'].append(depth_score)
                    context_scores['width'].append(width_score)
                    context_scores['coherence'].append(coherence_score)
                    
                    print(f"      Токенов: {len(tokens)}, Время: {process_time:.4f}с")
                    print(f"      Глубина: {depth_score:.3f}, Ширина: {width_score:.3f}, Связность: {coherence_score:.3f}")
            
            total_time = time.time() - start_time
            
            # Повторная обработка для тестирования кэша
            if use_cache and len(texts) > 0:
                print("   Тестирование кэша...")
                cache_test_times = []
                
                for text_data in texts[:3]:  # Тестируем первые 3
                    cache_start = time.time()
                    result = processor.process_text(text_data['text'])
                    cache_time = time.time() - cache_start
                    cache_test_times.append(cache_time)
                    
                    if cache_time < 0.01:  # Быстро = из кэша
                        cache_hits += 1
                    else:
                        cache_misses += 1
            
            # Вычисляем метрики
            avg_processing_time = statistics.mean(processing_times) if processing_times else 0
            median_processing_time = statistics.median(processing_times) if processing_times else 0
            tokens_per_second = total_tokens / max(0.001, total_time)
            cache_hit_rate = cache_hits / max(1, cache_hits + cache_misses) if use_cache else 0
            
            avg_depth = statistics.mean(context_scores['depth']) if context_scores['depth'] else 0
            avg_width = statistics.mean(context_scores['width']) if context_scores['width'] else 0
            avg_coherence = statistics.mean(context_scores['coherence']) if context_scores['coherence'] else 0
            
            # Оценка эффективности памяти (упрощенная)
            memory_efficiency = min(1.0, total_tokens / max(1, len(' '.join(t['text'] for t in texts))))
            
            return BenchmarkResult(
                scenario=scenario_name,
                avg_processing_time=avg_processing_time,
                median_processing_time=median_processing_time,
                total_tokens=total_tokens,
                tokens_per_second=tokens_per_second,
                cache_hit_rate=cache_hit_rate,
                context_depth_score=avg_depth,
                context_width_score=avg_width,
                semantic_coherence_score=avg_coherence,
                memory_efficiency=memory_efficiency,
                concurrent_capacity=max_workers
            )
            
        except Exception as e:
            print(f"❌ Ошибка в бенчмарке {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
            
            return BenchmarkResult(
                scenario=scenario_name,
                avg_processing_time=0, median_processing_time=0, total_tokens=0,
                tokens_per_second=0, cache_hit_rate=0, context_depth_score=0,
                context_width_score=0, semantic_coherence_score=0,
                memory_efficiency=0, concurrent_capacity=0
            )
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Запускает комплексный бенчмарк всех сценариев."""
        
        print("🚀 ЗАПУСК КОМПЛЕКСНОГО БЕНЧМАРКА ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*80)
        
        datasets = self.create_test_datasets()
        
        # Определяем сценарии тестирования
        scenarios = [
            ('baseline_simple', datasets['simple'], False, False, 1),
            ('optimized_simple', datasets['simple'], True, True, 2),
            
            ('baseline_medium', datasets['medium'], False, False, 1),
            ('optimized_medium', datasets['medium'], True, True, 3),
            
            ('baseline_complex', datasets['complex'], False, False, 1),
            ('optimized_complex', datasets['complex'], True, True, 4),
            
            ('baseline_large', datasets['large_context'], False, False, 1),
            ('optimized_large', datasets['large_context'], True, True, 6),
        ]
        
        results = {}
        
        for scenario_name, texts, use_cache, use_async, workers in scenarios:
            result = self.benchmark_scenario(scenario_name, texts, use_cache, use_async, workers)
            results[scenario_name] = result
        
        return results
    
    def analyze_improvements(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Анализирует улучшения производительности."""
        
        print("\n" + "="*80)
        print("📊 АНАЛИЗ УЛУЧШЕНИЙ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*80)
        
        analysis = {}
        
        # Сравниваем пары baseline vs optimized
        comparisons = [
            ('simple', 'baseline_simple', 'optimized_simple'),
            ('medium', 'baseline_medium', 'optimized_medium'),
            ('complex', 'baseline_complex', 'optimized_complex'),
            ('large', 'baseline_large', 'optimized_large')
        ]
        
        for category, baseline_key, optimized_key in comparisons:
            if baseline_key in results and optimized_key in results:
                baseline = results[baseline_key]
                optimized = results[optimized_key]
                
                # Вычисляем улучшения
                speed_improvement = (baseline.avg_processing_time / max(0.0001, optimized.avg_processing_time)) if baseline.avg_processing_time > 0 else 1.0
                throughput_improvement = optimized.tokens_per_second / max(0.1, baseline.tokens_per_second)
                
                depth_improvement = ((optimized.context_depth_score - baseline.context_depth_score) / max(0.001, baseline.context_depth_score)) * 100
                width_improvement = ((optimized.context_width_score - baseline.context_width_score) / max(0.001, baseline.context_width_score)) * 100
                coherence_improvement = ((optimized.semantic_coherence_score - baseline.semantic_coherence_score) / max(0.001, baseline.semantic_coherence_score)) * 100
                
                analysis[category] = {
                    'speed_improvement': speed_improvement,
                    'throughput_improvement': throughput_improvement,
                    'depth_improvement': depth_improvement,
                    'width_improvement': width_improvement,
                    'coherence_improvement': coherence_improvement,
                    'cache_hit_rate': optimized.cache_hit_rate,
                    'baseline_performance': {
                        'time': baseline.avg_processing_time,
                        'throughput': baseline.tokens_per_second,
                        'depth': baseline.context_depth_score,
                        'width': baseline.context_width_score,
                        'coherence': baseline.semantic_coherence_score
                    },
                    'optimized_performance': {
                        'time': optimized.avg_processing_time,
                        'throughput': optimized.tokens_per_second,
                        'depth': optimized.context_depth_score,
                        'width': optimized.context_width_score,
                        'coherence': optimized.semantic_coherence_score
                    }
                }
                
                print(f"\n🔍 Категория: {category.upper()}")
                print(f"   Ускорение обработки:     {speed_improvement:.2f}x")
                print(f"   Рост пропускной способности: {throughput_improvement:.2f}x")
                print(f"   Улучшение глубины:       {depth_improvement:+.1f}%")
                print(f"   Улучшение ширины:        {width_improvement:+.1f}%")
                print(f"   Улучшение связности:     {coherence_improvement:+.1f}%")
                print(f"   Эффективность кэша:      {optimized.cache_hit_rate:.1%}")
        
        return analysis

def main():
    """Основная функция запуска бенчмарка."""
    
    benchmark = PerformanceBenchmark()
    
    # Запускаем бенчмарк
    results = benchmark.run_comprehensive_benchmark()
    
    # Анализируем результаты
    analysis = benchmark.analyze_improvements(results)
    
    # Сохраняем результаты
    output_data = {
        'timestamp': time.time(),
        'results': {k: {
            'scenario': v.scenario,
            'avg_processing_time': v.avg_processing_time,
            'median_processing_time': v.median_processing_time,
            'total_tokens': v.total_tokens,
            'tokens_per_second': v.tokens_per_second,
            'cache_hit_rate': v.cache_hit_rate,
            'context_depth_score': v.context_depth_score,
            'context_width_score': v.context_width_score,
            'semantic_coherence_score': v.semantic_coherence_score,
            'memory_efficiency': v.memory_efficiency,
            'concurrent_capacity': v.concurrent_capacity
        } for k, v in results.items()},
        'analysis': analysis
    }
    
    # Сохраняем в файл
    with open('comprehensive_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в comprehensive_benchmark_results.json")
    
    # Итоговый отчет
    print("\n" + "="*80)
    print("🎯 ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    
    if analysis:
        avg_speed_improvement = statistics.mean([a['speed_improvement'] for a in analysis.values()])
        avg_depth_improvement = statistics.mean([a['depth_improvement'] for a in analysis.values()])
        avg_width_improvement = statistics.mean([a['width_improvement'] for a in analysis.values()])
        avg_coherence_improvement = statistics.mean([a['coherence_improvement'] for a in analysis.values()])
        
        print(f"Среднее ускорение обработки:        {avg_speed_improvement:.2f}x")
        print(f"Среднее улучшение глубины понимания: {avg_depth_improvement:+.1f}%")
        print(f"Среднее улучшение ширины понимания:  {avg_width_improvement:+.1f}%")
        print(f"Среднее улучшение связности:         {avg_coherence_improvement:+.1f}%")
        
        # Общая оценка эффективности
        overall_improvement = (avg_speed_improvement * 0.4 + 
                             (1 + avg_depth_improvement/100) * 0.2 + 
                             (1 + avg_width_improvement/100) * 0.2 + 
                             (1 + avg_coherence_improvement/100) * 0.2)
        
        print(f"\n🏆 ОБЩАЯ ЭФФЕКТИВНОСТЬ ОПТИМИЗАЦИЙ: {overall_improvement:.2f}x")
        
        if overall_improvement > 1.5:
            print("🎉 ОТЛИЧНЫЙ РЕЗУЛЬТАТ! Оптимизации значительно улучшили производительность.")
        elif overall_improvement > 1.2:
            print("✅ ХОРОШИЙ РЕЗУЛЬТАТ! Оптимизации показали заметные улучшения.")
        else:
            print("⚠️ УМЕРЕННЫЙ РЕЗУЛЬТАТ. Есть потенциал для дальнейших оптимизаций.")
    
    print("\n🔬 Бенчмарк завершен!")

if __name__ == "__main__":
    main()
