"""
Модуль улучшения генерации текста для CogniFlex
Обеспечивает качественную генерацию через обучение и постобработку
"""
from __future__ import annotations

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger("cogniflex.text_quality_improver")


@dataclass
class GenerationMetrics:
    """Метрики качества генерации"""
    coherence_score: float
    diversity_score: float
    length_score: float
    grammar_score: float
    readability_score: float
    relevance_score: float
    overall_score: float


class TextQualityImprover:
    """Класс для улучшения качества генерации текста"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'core', 'cogniflex_cache', 'ml_unit', 'fractal_storage', 'models'
        )
        
        # Загружаем модель для анализа качества
        self.quality_model = None
        self.quality_tokenizer = None
        
        # Паттерны для проверки качества
        self.good_patterns = [
            r'\b[а-я]+\b',  # Русские слова
            r'[.!?]',       # Знаки препинания
            r'\b\d+\b',     # Цифры
        ]
        
        self.bad_patterns = [
            r'(.)\1{4,}',   # Повторяющиеся символы
            r'[^\w\s\.\,\!\?\;\:\-\—\nа-яА-ЯёЁ]',  # Не-русские символы
            r'\b\w{1,2}\b\s+\b\w{1,2}\b',  # Много коротких слов подряд
        ]
        
        # Шаблоны хороших ответов
        self.response_templates = [
            "Я могу помочь вам с этим вопросом.",
            "Давайте разберем эту тему подробнее.",
            "Это интересный вопрос, который требует внимания.",
            "Я постараюсь дать вам полезный ответ.",
            "Пожалуйста, уточните ваш запрос для более точного ответа."
        ]
        
        self._initialize_quality_checker()
    
    def _initialize_quality_checker(self):
        """Инициализирует модель для проверки качества"""
        try:
            # Ищем модель в фрактальном хранилище
            model_files = os.listdir(self.model_path)
            if 'russian_gpt2' in model_files:
                model_dir = os.path.join(self.model_path, 'russian_gpt2')
                if os.path.exists(model_dir):
                    self.quality_tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
                    self.quality_model = GPT2LMHeadModel.from_pretrained(model_dir)
                    logger.info("Модель качества загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель качества: {e}")
    
    def improve_generation_parameters(self, query: str, current_response: str) -> Dict[str, Any]:
        """Улучшает параметры генерации на основе анализа запроса и ответа"""
        metrics = self.analyze_text_quality(current_response)
        
        # Базовые параметры
        params = {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.85,
            'repetition_penalty': 1.1,
            'no_repeat_ngram_size': 2,
        }
        
        # Адаптируем параметры на основе метрик
        if metrics.coherence_score < 0.5:
            params['temperature'] = 0.6  # Более предсказуемая генерация
            params['top_k'] = 30
            
        if metrics.diversity_score < 0.3:
            params['temperature'] = 0.8  # Более разнообразная генерация
            params['repetition_penalty'] = 1.3
            
        if metrics.length_score < 0.4:
            params['max_tokens'] = 150
            
        # Специфичные параметры для разных типов запросов
        if '?' in query:
            params['temperature'] = 0.65  # Более точные ответы на вопросы
            
        if 'расскажи' in query.lower() or 'опиши' in query.lower():
            params['max_tokens'] = 200
            
        return params
    
    def analyze_text_quality(self, text: str) -> GenerationMetrics:
        """Анализирует качество текста"""
        if not text:
            return GenerationMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Коэрентность (связность)
        coherence_score = self._calculate_coherence(text)
        
        # Разнообразие
        diversity_score = self._calculate_diversity(text)
        
        # Длина
        length_score = self._calculate_length_score(text)
        
        # Грамматика
        grammar_score = self._calculate_grammar_score(text)
        
        # Читаемость
        readability_score = self._calculate_readability_score(text)
        
        # Релевантность
        relevance_score = self._calculate_relevance_score(text)
        
        # Общая оценка (с учетом всех метрик)
        overall_score = (coherence_score + diversity_score + length_score + 
                        grammar_score + readability_score + relevance_score) / 6
        
        return GenerationMetrics(
            coherence_score=coherence_score,
            diversity_score=diversity_score,
            length_score=length_score,
            grammar_score=grammar_score,
            readability_score=readability_score,
            relevance_score=relevance_score,
            overall_score=overall_score
        )
    
    def _calculate_coherence(self, text: str) -> float:
        """Рассчитывает связность текста"""
        # Проверяем наличие знаков препинания
        punctuation_count = len(re.findall(r'[.!?]', text))
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Средняя длина предложений
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Оценка связности
        score = 0.0
        
        # Наличие знаков препинания
        if punctuation_count > 0:
            score += 0.3
            
        # Средняя длина предложений
        if 5 <= avg_sentence_length <= 20:
            score += 0.4
        elif 3 <= avg_sentence_length <= 30:
            score += 0.2
            
        # Наличие русских слов
        russian_words = len(re.findall(r'\b[а-я]+\b', text.lower()))
        total_words = len(text.split())
        if total_words > 0:
            russian_ratio = russian_words / total_words
            score += russian_ratio * 0.3
            
        return min(score, 1.0)
    
    def _calculate_diversity(self, text: str) -> float:
        """Рассчитывает разнообразие лексики"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        
        # Штраф за слишком короткие слова
        short_words = [w for w in words if len(w) < 3]
        if len(short_words) > len(words) * 0.5:
            diversity *= 0.7
            
        return min(diversity, 1.0)
    
    def _calculate_length_score(self, text: str) -> float:
        """Рассчитывает оценку длины"""
        length = len(text.strip())
        
        if length < 10:
            return 0.0
        elif length < 30:
            return 0.3
        elif length < 100:
            return 0.7
        elif length < 300:
            return 1.0
        else:
            return 0.8  # Слишком длинные ответы не всегда хороши
    
    def _calculate_grammar_score(self, text: str) -> float:
        """Рассчитывает грамматическую оценку"""
        score = 1.0
        
        # Штраф за повторяющиеся символы
        if re.search(r'(.)\1{4,}', text):
            score -= 0.3
            
        # Штраф за странные символы
        strange_chars = len(re.findall(r'[^\w\s\.\,\!\?\;\:\-\—\nа-яА-ЯёЁ]', text))
        if strange_chars > 0:
            score -= min(strange_chars * 0.1, 0.4)
            
        # Штраф за слишком много коротких слов подряд
        short_word_sequences = re.findall(r'\b\w{1,2}\b\s+\b\w{1,2}\b', text)
        if len(short_word_sequences) > 2:
            score -= 0.2
            
        return max(score, 0.0)
    
    def post_process_response(self, response: str, query: str) -> str:
        """Постобработка ответа"""
        if not response:
            return self._get_fallback_response(query)
        
        # Очищаем текст
        cleaned = self._clean_text(response)
        
        # Проверяем качество
        metrics = self.analyze_text_quality(cleaned)
        
        # Если качество низкое, пробуем улучшить
        if metrics.overall_score < 0.3:
            return self._improve_low_quality_response(cleaned, query)
        
        # Финальная обработка
        return self._finalize_response(cleaned)
    
    def _clean_text(self, text: str) -> str:
        """Очищает текст от артефактов"""
        # Убираем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем множественные знаки препинания
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '—', text)
        
        # Убираем странные символы
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\—\nа-яА-ЯёЁ]', '', text)
        
        # Убираем очень короткие бессмысленные слова
        words = text.split()
        filtered_words = [word for word in words if len(word) > 1 or word in ['и', 'в', 'на', 'с', 'по', 'к', 'у', 'я', 'ты', 'он', 'она']]
        
        return ' '.join(filtered_words).strip()
    
    def _improve_low_quality_response(self, response: str, query: str) -> str:
        """Улучшает ответ низкого качества"""
        # Если ответ совсем плохой, используем шаблон
        if len(response.strip()) < 10:
            return self._get_fallback_response(query)
        
        # Пробуем исправить основные проблемы
        improved = response
        
        # Добавляем связующие слова если нужно
        if not any(word in improved for word in ['потому', 'поскольку', 'так как', 'из-за']):
            if '?' in query:
                improved = "На ваш вопрос: " + improved
        
        # Добавляем завершение если нужно
        if not improved.endswith(('.', '!', '?', '...', '—')):
            improved += '.'
            
        return improved
    
    def _finalize_response(self, text: str) -> str:
        """Финальная обработка ответа"""
        text = text.strip()
        
        # Капитализация первого символа
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Добавляем точку в конце если нужно
        if text and not text.endswith(('.', '!', '?', '...', '—')):
            text += '.'
            
        return text
    
    def _get_fallback_response(self, query: str) -> str:
        """Возвращает запасной ответ"""
        if '?' in query:
            return "Я постараюсь ответить на ваш вопрос, но мне нужно больше информации для точного ответа."
        elif 'что' in query.lower():
            return "Это интересная тема. Давайте рассмотрим ее подробнее."
        elif 'как' in query.lower():
            return "Я могу помочь вам разобраться в этом вопросе."
        else:
            return "Спасибо за ваш запрос. Я постараюсь дать полезный ответ."
    
    def _calculate_readability_score(self, text: str) -> float:
        """Рассчитывает читаемость текста"""
        if not text:
            return 0.0
        
        # Длина слов
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Длина предложений
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Оценка читаемости (0-1)
        score = 0.0
        
        # Оптимальная длина слов (4-8 символов)
        if 4 <= avg_word_length <= 8:
            score += 0.4
        elif 3 <= avg_word_length <= 10:
            score += 0.2
        
        # Оптимальная длина предложений (10-25 слов)
        if 10 <= avg_sentence_length <= 25:
            score += 0.4
        elif 5 <= avg_sentence_length <= 35:
            score += 0.2
        
        # Наличие заглавных букв
        if any(word[0].isupper() for word in words if word):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Рассчитывает релевантность текста (базовая оценка)"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Проверка на осмысленность
        words = text.split()
        if len(words) < 3:
            return 0.0
        
        # Наличие русских слов
        russian_words = len(re.findall(r'\b[а-я]+\b', text.lower()))
        if russian_words > 0:
            score += 0.3
        
        # Отсутствие повторяющихся символов
        if not re.search(r'(.)\1{4,}', text):
            score += 0.3
        
        # Наличие знаков препинания
        if re.search(r'[.!?]', text):
            score += 0.2
        
        # Разнообразие слов
        unique_words = len(set(word.lower() for word in words))
        word_diversity = unique_words / len(words)
        if word_diversity > 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def improve_text(self, text: str) -> str:
        """Улучшает качество текста"""
        if not text or len(text.strip()) < 3:
            return text
        
        improved = text.strip()
        
        # Исправляем заглавные буквы в начале предложений
        improved = re.sub(r'([.!?]\s*)([а-я])', lambda m: m.group(1) + m.group(2).upper(), improved)
        improved = improved[0].upper() + improved[1:] if improved else improved
        
        # Добавляем недостающие знаки препинания
        if improved and not improved[-1] in '.!?':
            improved += '.'
        
        # Удаляем лишние пробелы
        improved = re.sub(r'\s+', ' ', improved)
        
        # Исправляем распространенные ошибки
        corrections = {
            'чо': 'что',
            'чтоб': 'чтобы',
            'есчо': 'ещё',
            'пожалуста': 'пожалуйста',
            'спасиба': 'спасибо',
            'зделай': 'сделай',
            'сделано': 'сделано'
        }
        
        for wrong, correct in corrections.items():
            improved = re.sub(rf'\b{wrong}\b', correct, improved, flags=re.IGNORECASE)
        
        return improved
    
    def enhance_response(self, response: str, context: str = None) -> str:
        """Улучшает ответ в зависимости от контекста"""
        if not response:
            return self.response_templates[0] if self.response_templates else "Я понимаю ваш вопрос."
        
        # Улучшаем базовое качество
        enhanced = self.improve_text(response)
        
        # Если ответ слишком короткий, расширяем его
        if len(enhanced.split()) < 5:
            if context and len(context.split()) > 3:
                enhanced = f"Учитывая контекст '{context[:50]}...', {enhanced.lower()}"
            else:
                enhanced = f"Хорошо, {enhanced.lower()}. Давайте рассмотрим это подробнее."
        
        # Добавляем вежливость, если нужно
        polite_phrases = ['пожалуйста', 'спасибо', 'будьте добры']
        if not any(phrase in enhanced.lower() for phrase in polite_phrases):
            if '?' in enhanced:
                enhanced = enhanced.replace('?', ', пожалуйста?')
        
        return enhanced
    
    def correct_grammar(self, text: str) -> str:
        """Исправляет грамматические ошибки"""
        if not text:
            return text
        
        corrected = text
        
        # Исправляем заглавные буквы
        sentences = re.split('([.!?]+)', corrected)
        for i in range(0, len(sentences), 2):
            if sentences[i].strip():
                sentences[i] = sentences[i][0].upper() + sentences[i][1:] if len(sentences[i]) > 1 else sentences[i]
        
        corrected = ''.join(sentences)
        
        # Исправляем пробелы перед знаками препинания
        corrected = re.sub(r'\s+([,.!?])', r'\1', corrected)
        
        # Добавляем пробелы после знаков препинания
        corrected = re.sub(r'([.!?])([а-яА-Я])', r'\1 \2', corrected)
        
        # Исправляем множественные пробелы
        corrected = re.sub(r'\s+', ' ', corrected)
        
        # Удаляем пробелы в начале и конце
        corrected = corrected.strip()
        
        # Добавляем точку в конец, если нужно
        if corrected and not corrected[-1] in '.!?':
            corrected += '.'
        
        return corrected
