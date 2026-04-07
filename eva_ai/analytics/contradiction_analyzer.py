"""
Модуль аналитики противоречий для системы ЕВА
Обеспечивает обнаружение и анализ противоречий между фактами модели и результатами веб-поиска

Основные возможности:
- Сравнение фактов модели с результатами поиска
- Расчёт уровня расхождения между утверждениями
- Фильтрация значимых противоречий
- Анализ паттернов для самообучения
- Централизованные метрики
"""

import os
import logging
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger("eva_ai.analytics.contradiction_analyzer")


class ContradictionAnalyzer:
    """
    Анализатор противоречий для обнаружения расхождений между
    фактами модели и результатами веб-поиска
    """
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует анализатор противоречий.
        
        Args:
            brain: Ссылка на CoreBrain
            cache_dir: Директория для кэширования
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'analytics_cache', 'contradictions')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.contradictions_history: List[Dict[str, Any]] = []
        self.patterns: Dict[str, int] = defaultdict(int)
        self.metrics = {
            'total_checked': 0,
            'significant_found': 0,
            'resolved': 0,
            'divergence_scores': [],
            'by_category': defaultdict(int)
        }
        
        self.min_divergence_threshold = 0.3
        self.significant_divergence_threshold = 0.6
        
        self.keywords_negation = ['не', 'нет', 'никогда', 'без', 'отсутствует', 'неверно', 'ошибочно']
        self.keywords_comparison = ['больше', 'меньше', 'лучше', 'хуже', 'раньше', 'позже', 'выше', 'ниже']
        
        logger.info("ContradictionAnalyzer инициализирован")
    
    def detect_contradictions(
        self, 
        model_facts: str, 
        web_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Сравнивает факты модели с результатами веб-поиска.
        
        Args:
            model_facts: Факты, полученные от GGUF модели (Модуль А)
            web_results: Результаты веб-поиска (список dict с title, url, snippet)
            
        Returns:
            Dict с результатами анализа:
            - significant_count: количество значимых противоречий
            - minor_count: количество второстепенных расхождений
            - details: список найденных противоречий
            - divergence_level: общий уровень расхождения (0-1)
            - has_contradiction: флаг наличия противоречий
        """
        self.metrics['total_checked'] += 1
        
        result = {
            'significant_count': 0,
            'minor_count': 0,
            'details': [],
            'divergence_level': 0.0,
            'has_contradiction': False,
            'checked_at': datetime.now().isoformat(),
            'model_facts_preview': model_facts[:200] if model_facts else ""
        }
        
        if not model_facts or not web_results:
            result['details'].append({
                'type': 'insufficient_data',
                'message': 'Недостаточно данных для сравнения'
            })
            return result
        
        fact_sentences = self._split_into_sentences(model_facts)
        
        for web_result in web_results:
            snippet = web_result.get('snippet', '')
            title = web_result.get('title', '')
            
            if not snippet:
                continue
            
            web_text = f"{title}. {snippet}"
            
            for fact_sentence in fact_sentences:
                if len(fact_sentence.strip()) < 10:
                    continue
                
                divergence = self.calculate_divergence(fact_sentence, web_text)
                
                if divergence >= self.significant_divergence_threshold:
                    contradiction = {
                        'type': 'significant',
                        'fact': fact_sentence,
                        'web_source': title[:100],
                        'web_url': web_result.get('url', ''),
                        'divergence': divergence,
                        'category': self._categorize_contradiction(fact_sentence, web_text),
                        'timestamp': time.time()
                    }
                    result['details'].append(contradiction)
                    result['significant_count'] += 1
                    self.patterns[contradiction['category']] += 1
                    
                elif divergence >= self.min_divergence_threshold:
                    minor_contradiction = {
                        'type': 'minor',
                        'fact': fact_sentence,
                        'web_source': title[:100],
                        'divergence': divergence,
                        'category': 'minor_discrepancy',
                        'timestamp': time.time()
                    }
                    result['details'].append(minor_contradiction)
                    result['minor_count'] += 1
        
        if result['significant_count'] > 0:
            result['has_contradiction'] = True
            self.metrics['significant_found'] += 1
        
        total_contradictions = result['significant_count'] + result['minor_count']
        if total_contradictions > 0:
            divergences = [c['divergence'] for c in result['details']]
            result['divergence_level'] = sum(divergences) / len(divergences)
            self.metrics['divergence_scores'].append(result['divergence_level'])
        
        self.contradictions_history.append({
            'timestamp': time.time(),
            'model_facts': model_facts[:500],
            'result': result
        })
        
        self._save_to_history(result)
        
        logger.debug(f"Противоречия: значимых={result['significant_count']}, "
                    f"второстепенных={result['minor_count']}, "
                    f"расхождение={result['divergence_level']:.2f}")
        
        return result
    
    def calculate_divergence(self, fact_statement: str, web_statement: str) -> float:
        """
        Вычисляет уровень расхождения между двумя утверждениями.
        
        Args:
            fact_statement: Утверждение из модели
            web_statement: Утверждение из веб-поиска
            
        Returns:
            float: Уровень расхождения (0-1), где 1 - полное противоречие
        """
        if not fact_statement or not web_statement:
            return 0.0
        
        fact_lower = fact_statement.lower()
        web_lower = web_statement.lower()
        
        divergence = 0.0
        
        negation_matches = self._check_negation_contradiction(fact_lower, web_lower)
        divergence = max(divergence, negation_matches)
        
        number_matches = self._check_number_contradiction(fact_lower, web_lower)
        divergence = max(divergence, number_matches)
        
        temporal_matches = self._check_temporal_contradiction(fact_lower, web_lower)
        divergence = max(divergence, temporal_matches)
        
        keyword_divergence = self._check_keyword_contradiction(fact_lower, web_lower)
        divergence = max(divergence, keyword_divergence)
        
        return min(divergence, 1.0)
    
    def _check_negation_contradiction(self, fact: str, web: str) -> float:
        """Проверяет противоречие через отрицание."""
        for neg_word in self.keywords_negation:
            if neg_word in fact:
                for neg_word_web in self.keywords_negation:
                    if neg_word_web in web:
                        return 0.0
                
                fact_without_neg = fact.replace(neg_word, '').strip()
                if fact_without_neg in web:
                    return 0.8
        
        return 0.0
    
    def _check_number_contradiction(self, fact: str, web: str) -> float:
        """Проверяет численные противоречия."""
        fact_numbers = re.findall(r'\d+(?:[.,]\d+)?', fact)
        web_numbers = re.findall(r'\d+(?:[.,]\d+)?', web)
        
        if not fact_numbers or not web_numbers:
            return 0.0
        
        for fn in fact_numbers:
            for wn in web_numbers:
                try:
                    f_num = float(fn.replace(',', '.'))
                    w_num = float(wn.replace(',', '.'))
                    
                    if f_num != w_num and abs(f_num - w_num) > 0.01:
                        if (f_num > 0 and w_num > 0 and 
                            (f_num > w_num * 2 or w_num > f_num * 2)):
                            return 0.9
                except ValueError:
                    continue
        
        return 0.0
    
    def _check_temporal_contradiction(self, fact: str, web: str) -> float:
        """Проверяет временные противоречия."""
        time_words_fact = [w for w in fact.split() if w in self.keywords_comparison]
        
        if not time_words_fact:
            return 0.0
        
        for word in time_words_fact:
            if word in fact and word in web:
                fact_pos = fact.find(word)
                web_pos = web.find(word)
                
                if abs(fact_pos - web_pos) < 20:
                    return 0.0
                
                return 0.7
        
        return 0.0
    
    def _check_keyword_contradiction(self, fact: str, web: str) -> float:
        """Проверяет ключевые слова-противоречия."""
        contradiction_keywords = [
            ('да', 'нет'),
            ('есть', 'нет'),
            ('может', 'невозможно'),
            ('известно', 'неизвестно'),
            ('верно', 'неверно'),
            ('правда', 'ложь'),
            ('истина', 'ложь'),
        ]
        
        for kw1, kw2 in contradiction_keywords:
            if kw1 in fact and kw2 in web:
                return 0.7
            if kw2 in fact and kw1 in web:
                return 0.7
        
        return 0.0
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _categorize_contradiction(self, fact: str, web: str) -> str:
        """Определяет категорию противоречия."""
        if self._check_negation_contradiction(fact.lower(), web.lower()) > 0:
            return 'negation'
        if self._check_number_contradiction(fact.lower(), web.lower()) > 0:
            return 'numerical'
        if self._check_temporal_contradiction(fact.lower(), web.lower()) > 0:
            return 'temporal'
        if self._check_keyword_contradiction(fact.lower(), web.lower()) > 0:
            return 'keyword'
        return 'semantic'
    
    def get_significant_contradictions(self, contradictions: List[Dict]) -> List[Dict]:
        """
        Фильтрует только значимые противоречия.
        
        Args:
            contradictions: Список противоречий от detect_contradictions
            
        Returns:
            List: Отфильтрованный список значимых противоречий
        """
        significant = [c for c in contradictions if c.get('type') == 'significant']
        return sorted(significant, key=lambda x: x.get('divergence', 0), reverse=True)
    
    def analyze_contradiction_patterns(self) -> Dict[str, Any]:
        """
        Анализирует паттерны противоречий для самообучения.
        
        Returns:
            Dict с анализом паттернов:
            - most_common_category: наиболее частая категория
            - trend: тренд (increasing, decreasing, stable)
            - average_divergence: среднее расхождение
            - total_contradictions: общее количество
            - resolution_rate: процент разрешённых
        """
        if not self.patterns:
            return {
                'most_common_category': 'none',
                'trend': 'stable',
                'average_divergence': 0.0,
                'total_contradictions': 0,
                'resolution_rate': 0.0
            }
        
        most_common = max(self.patterns.items(), key=lambda x: x[1])
        
        recent_divergences = self.metrics['divergence_scores'][-10:] if self.metrics['divergence_scores'] else []
        avg_divergence = sum(recent_divergences) / len(recent_divergences) if recent_divergences else 0.0
        
        trend = 'stable'
        if len(self.metrics['divergence_scores']) >= 10:
            old_avg = sum(self.metrics['divergence_scores'][:5]) / 5
            new_avg = sum(self.metrics['divergence_scores'][-5:]) / 5
            if new_avg > old_avg + 0.1:
                trend = 'increasing'
            elif new_avg < old_avg - 0.1:
                trend = 'decreasing'
        
        resolution_rate = self.metrics['resolved'] / max(self.metrics['significant_found'], 1)
        
        return {
            'most_common_category': most_common[0],
            'category_count': most_common[1],
            'trend': trend,
            'average_divergence': avg_divergence,
            'total_contradictions': self.metrics['significant_found'],
            'resolution_rate': resolution_rate,
            'patterns': dict(self.patterns)
        }
    
    def get_contradiction_metrics(self) -> Dict[str, Any]:
        """
        Возвращает метрики противоречий для конвейера.
        
        Returns:
            Dict с метриками:
            - count: общее количество проверенных
            - significant: количество значимых
            - resolved: количество разрешённых
            - avg_divergence: среднее расхождение
            - category_breakdown: по категориям
        """
        avg_div = 0.0
        if self.metrics['divergence_scores']:
            avg_div = sum(self.metrics['divergence_scores']) / len(self.metrics['divergence_scores'])
        
        return {
            'count': self.metrics['total_checked'],
            'significant': self.metrics['significant_found'],
            'resolved': self.metrics['resolved'],
            'avg_divergence': avg_div,
            'category_breakdown': dict(self.metrics['by_category']),
            'last_checked': self.contradictions_history[-1]['timestamp'] if self.contradictions_history else None
        }
    
    def resolve_contradiction(self, contradiction_id: str, resolution: str = "resolved") -> bool:
        """
        Отмечает противоречие как разрешённое.
        
        Args:
            contradiction_id: ID противоречия
            resolution: Тип разрешения
            
        Returns:
            True если успешно
        """
        self.metrics['resolved'] += 1
        logger.info(f"Противоречие {contradiction_id} разрешено: {resolution}")
        return True
    
    def _save_to_history(self, result: Dict[str, Any]):
        """Сохраняет результат в историю."""
        try:
            history_file = os.path.join(self.cache_dir, 'contradiction_history.json')
            
            existing = []
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            
            existing.append({
                'timestamp': datetime.now().isoformat(),
                'result': result
            })
            
            if len(existing) > 100:
                existing = existing[-100:]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.debug(f"Не удалось сохранить историю: {e}")
    
    def reset_metrics(self):
        """Сбрасывает метрики."""
        self.contradictions_history.clear()
        self.patterns.clear()
        self.metrics = {
            'total_checked': 0,
            'significant_found': 0,
            'resolved': 0,
            'divergence_scores': [],
            'by_category': defaultdict(int)
        }
        logger.info("Метрики противоречий сброшены")


class RelevanceCalculator:
    """
    Калькулятор релевантности для оценки качества ответа.
    Использует локальные эмбеддинги для расчёта косинусного сходства.
    """
    
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        """
        Инициализирует калькулятор релевантности.
        
        Args:
            model_name: Название модели эмбеддингов
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализирует модель эмбеддингов."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"RelevanceCalculator: модель {self.model_name} загружена")
        except ImportError:
            logger.warning("sentence-transformers не установлен, используем упрощённый метод")
            self.model = None
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель эмбеддингов: {e}")
            self.model = None
    
    def calculate_similarity(self, query: str, answer: str) -> float:
        """
        Вычисляет косинусное сходство между запросом и ответом.
        
        Args:
            query: Запрос пользователя
            answer: Ответ модели
            
        Returns:
            float: Косинусное сходство (0-1)
        """
        if not query or not answer:
            return 0.0
        
        if self.model is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                query_emb = self.model.encode([query])
                answer_emb = self.model.encode([answer])
                
                similarity = cosine_similarity(query_emb, answer_emb)[0][0]
                return float(similarity)
            except Exception as e:
                logger.debug(f"Ошибка расчёта эмбеддингов: {e}")
        
        return self._fallback_similarity(query, answer)
    
    def _fallback_similarity(self, query: str, answer: str) -> float:
        """Упрощённый метод при отсутствии модели."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words or not answer_words:
            return 0.0
        
        intersection = query_words & answer_words
        union = query_words | answer_words
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return min(jaccard * 2, 1.0)
    
    def check_threshold(self, query: str, answer: str, threshold: float = 0.85) -> Dict[str, Any]:
        """
        Проверяет порог релевантности.
        
        Args:
            query: Запрос
            answer: Ответ
            threshold: Порог (по умолчанию 0.85)
            
        Returns:
            Dict с результатами проверки:
            - similarity: косинусное сходство
            - passes: проходит ли порог
            - threshold: использованный порог
        """
        similarity = self.calculate_similarity(query, answer)
        
        return {
            'similarity': similarity,
            'passes': similarity >= threshold,
            'threshold': threshold,
            'status': 'ok' if similarity >= threshold else 'needs_refinement'
        }
