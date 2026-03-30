"""
Утилиты для детекции и исправления бессвязного текста
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class TextQualityChecker:
    """Проверка качества сгенерированного текста"""
    
    def __init__(self):
        # Пороговые значения для определения проблем
        self.thresholds = {
            'max_word_length': 20,  # Максимальная длина слова
            'min_avg_word_length': 3,  # Минимальная средняя длина слова
            'max_non_alpha_ratio': 0.3,  # Максимальная доля не-буквенных символов
            'min_unique_words_ratio': 0.4,  # Минимальная доля уникальных слов
            'max_repetition_ratio': 0.5,  # Максимальная доля повторов
        }
        
        # Частые русские слова для проверки
        self.common_russian_words = {
            'и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а', 'по', 'это',
            'она', 'к', 'но', 'мы', 'как', 'из', 'у', 'то', 'за', 'свой', 'ее',
            'мочь', 'вы', 'весь', 'так', 'его', 'сказать', 'для', 'уже', 'кто',
            'да', 'этот', 'время', 'если', 'еще', 'каждый', 'другой', 'себя',
            'тот', 'только', 'такой', 'который', 'человек', 'их', 'много', 'все',
            'знать', 'можно', 'чем', 'мой', 'таки', 'русский', 'год', 'работа',
            'жизнь', 'день', 'рука', 'раз', 'работать', 'слово', 'место', 'лицо',
            'друг', 'глаз', 'вопрос', 'дом', 'сторона', 'страна', 'мир', 'случай',
            'голова', 'ребенок', 'сила', 'конец', 'видеть', 'система', 'часть',
            'город', 'отношение', 'женщина', 'деньги', 'земля', 'машина', 'вода',
            'отец', 'проблема', 'час', 'право', 'нога', 'решение', 'дверь',
            'образ', 'история', 'власть', 'закон', 'война', 'бог', 'голос'
        }
        
    def check_quality(self, text: str) -> Dict[str, Any]:
        """
        Проверяет качество текста и возвращает оценку
        
        Returns:
            Dict с метриками качества и флагом is_gibberish
        """
        if not text or not isinstance(text, str):
            return {'is_gibberish': True, 'score': 0.0, 'reasons': ['Empty or invalid text']}
        
        text = text.strip()
        if len(text) < 10:
            return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Text too short']}
        
        metrics = {}
        reasons = []
        
        # 1. Извлекаем слова
        words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]+\b', text.lower())
        if not words:
            return {'is_gibberish': True, 'score': 0.0, 'reasons': ['No valid words found']}
        
        # 2. Проверяем длины слов
        word_lengths = [len(w) for w in words]
        max_len = max(word_lengths)
        avg_len = sum(word_lengths) / len(word_lengths)
        
        metrics['max_word_length'] = max_len
        metrics['avg_word_length'] = avg_len
        
        if max_len > self.thresholds['max_word_length']:
            reasons.append(f'Words too long (max: {max_len})')
        
        if avg_len < self.thresholds['min_avg_word_length']:
            reasons.append(f'Average word length too low ({avg_len:.1f})')
        
        # 3. Проверяем символы
        total_chars = len(text)
        alpha_chars = len(re.findall(r'[а-яА-ЯёЁa-zA-Z]', text))
        non_alpha_ratio = 1 - (alpha_chars / total_chars) if total_chars > 0 else 1
        
        metrics['non_alpha_ratio'] = non_alpha_ratio
        
        if non_alpha_ratio > self.thresholds['max_non_alpha_ratio']:
            reasons.append(f'Too many non-alpha characters ({non_alpha_ratio:.1%})')
        
        # 4. Проверяем уникальность слов
        unique_words = len(set(words))
        unique_ratio = unique_words / len(words) if words else 0
        
        metrics['unique_words_ratio'] = unique_ratio
        
        if unique_ratio < self.thresholds['min_unique_words_ratio']:
            reasons.append(f'Too many repeated words ({unique_ratio:.1%} unique)')
        
        # 5. Проверяем на наличие осмысленных слов
        meaningful_words = [w for w in words if w in self.common_russian_words]
        meaningful_ratio = len(meaningful_words) / len(words) if words else 0
        
        metrics['meaningful_ratio'] = meaningful_ratio
        
        if meaningful_ratio < 0.1:  # Меньше 10% осмысленных слов
            reasons.append(f'Too few meaningful words ({meaningful_ratio:.1%})')
        
        # 6. Проверяем паттерны повторов
        word_counts = Counter(words)
        most_common = word_counts.most_common(3)
        repetition_score = sum(c for _, c in most_common) / len(words)
        
        metrics['repetition_score'] = repetition_score
        
        if repetition_score > self.thresholds['max_repetition_ratio']:
            reasons.append(f'Too much repetition ({repetition_score:.1%})')
        
        # 7. Проверяем на случайные комбинации букв
        gibberish_words = [w for w in words if len(w) > 12 and self._is_gibberish_word(w)]
        gibberish_ratio = len(gibberish_words) / len(words) if words else 0
        
        metrics['gibberish_ratio'] = gibberish_ratio
        
        if gibberish_ratio > 0.3:
            reasons.append(f'Too many gibberish words ({gibberish_ratio:.1%})')
        
        # Вычисляем общий score
        score = self._calculate_score(metrics)
        is_gibberish = len(reasons) > 0 or score < 0.5
        
        return {
            'is_gibberish': is_gibberish,
            'score': score,
            'reasons': reasons,
            'metrics': metrics
        }
    
    def _is_gibberish_word(self, word: str) -> bool:
        """Проверяет, является ли слово бессвязным набором букв"""
        # Проверяем на чередование согласных и гласных
        vowels = set('аеёиоуыэюяaeiouy')
        
        if len(word) < 6:
            return False
        
        # Считаем чередование
        vowel_count = sum(1 for c in word.lower() if c in vowels)
        consonant_count = len(word) - vowel_count
        
        # Если все согласные или все гласные - подозрительно
        if vowel_count == 0 or consonant_count == 0:
            return True
        
        # Проверяем соотношение
        ratio = vowel_count / len(word)
        if ratio < 0.1 or ratio > 0.9:  # Слишком мало или много гласных
            return True
        
        # Проверяем на три одинаковые буквы подряд
        if re.search(r'(.)\1\1', word.lower()):
            return True
        
        return False
    
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """Вычисляет общий score на основе метрик"""
        score = 1.0
        
        # Штрафуем за проблемы
        if metrics.get('max_word_length', 0) > 15:
            score -= 0.2
        
        if metrics.get('non_alpha_ratio', 0) > 0.2:
            score -= 0.2
        
        if metrics.get('unique_words_ratio', 1) < 0.5:
            score -= 0.15
        
        if metrics.get('meaningful_ratio', 1) < 0.2:
            score -= 0.3
        
        if metrics.get('gibberish_ratio', 0) > 0.2:
            score -= 0.25
        
        return max(0.0, min(1.0, score))


class TextPostProcessor:
    """Пост-обработка сгенерированного текста"""
    
    def __init__(self):
        self.quality_checker = TextQualityChecker()
        
    def process(self, text: str, query: str = "") -> Dict[str, Any]:
        """
        Пост-обрабатывает текст и проверяет качество
        
        Returns:
            Dict с обработанным текстом и информацией о качестве
        """
        if not text:
            return {'text': '', 'quality': {'is_gibberish': True, 'score': 0.0}}
        
        # Шаг 1: Базовая очистка
        cleaned = self._basic_clean(text)
        
        # Шаг 2: Убираем запрос из ответа
        if query:
            cleaned = self._remove_query(cleaned, query)
        
        # Шаг 3: Проверяем качество
        quality = self.quality_checker.check_quality(cleaned)
        
        # Шаг 4: Дополнительная очистка если нужно
        if quality['is_gibberish']:
            cleaned = self._aggressive_clean(cleaned)
            quality = self.quality_checker.check_quality(cleaned)
        
        return {
            'text': cleaned,
            'original': text,
            'quality': quality
        }
    
    def _basic_clean(self, text: str) -> str:
        """Базовая очистка текста"""
        # Убираем специальные токены
        text = text.replace('<|endoftext|>', '')
        text = text.replace('<pad>', '')
        text = text.replace('<s>', '')
        text = text.replace('</s>', '')
        text = text.replace('[MASK]', '')
        text = text.replace('[CLS]', '')
        text = text.replace('[SEP]', '')
        
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _remove_query(self, text: str, query: str) -> str:
        """Убирает запрос из начала ответа"""
        query_lower = query.lower().strip()
        text_lower = text.lower().strip()
        
        if text_lower.startswith(query_lower):
            return text[len(query):].strip()
        
        return text
    
    def _aggressive_clean(self, text: str) -> str:
        """Агрессивная очистка для бессвязного текста"""
        # Убираем длинные бессвязные слова
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Пропускаем слова длиннее 20 символов
            if len(word) > 20:
                continue
            
            # Пропускаем слова с тремя одинаковыми буквами подряд
            if re.search(r'(.)\1\1', word.lower()):
                continue
            
            cleaned_words.append(word)
        
        return ' '.join(cleaned_words)


def check_and_fix_response(text: str, query: str = "") -> Tuple[str, bool, Dict]:
    """
    Быстрая проверка и исправление ответа
    
    Returns:
        Tuple: (исправленный_текст, is_gibberish, метрики)
    """
    processor = TextPostProcessor()
    result = processor.process(text, query)
    
    return (
        result['text'],
        result['quality']['is_gibberish'],
        result['quality']
    )
