"""
SRE Quality Module — response quality checking, sanitization, looping detection.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def check_quality(self, response: str, query: str) -> Dict[str, Any]:
    """Проверяет качество ответа модели"""
    if not response or len(response.strip()) < 5:
        return {'score': 0.1, 'is_gibberish': True, 'reasons': ['Пустой ответ']}

    is_gibberish = False
    reasons = []
    words = response.split()
    if len(words) > 5:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            is_gibberish = True
            reasons.append('Много повторений')

    vowels = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')
    if not any(v in response for v in vowels):
        is_gibberish = True
        reasons.append('Нет гласных')

    score = 0.8
    if is_gibberish:
        score = 0.2
    elif len(response) < 50:
        score = 0.5

    return {'score': score, 'is_gibberish': is_gibberish, 'reasons': reasons if reasons else ['OK']}


def _sanitize_response(self, response: str) -> str:
    """Очистка ответа от артефактов генерации"""
    if not response:
        return ""

    response = _clean_filler_start(self, response)
    response = _remove_looping_blocks(self, response)

    return response.strip()


def _clean_filler_start(self, response: str) -> str:
    """Убирает начальные filler-фразы"""
    filler_prefixes = [
        'хорошо,', 'давайте', 'начнём', 'итак,', 'что ж,',
        'ok,', 'okay,', 'well,', 'sure,', 'of course,',
        'конечно,', 'разумеется,', 'без проблем,',
    ]

    lines = response.split('\n')
    if lines:
        first_line_lower = lines[0].lower().strip()
        for prefix in filler_prefixes:
            if first_line_lower.startswith(prefix):
                lines[0] = lines[0][len(prefix):].strip()
                break

    return '\n'.join(line for line in lines if line.strip())


def _remove_looping_blocks(self, response: str) -> str:
    """Удаляет зацикленные блоки текста"""
    if not response:
        return ""

    sentences = response.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 4:
        return response

    seen = {}
    loop_start = None
    for i, sent in enumerate(sentences):
        sent_normalized = ' '.join(sent.lower().split())
        if sent_normalized in seen:
            loop_start = seen[sent_normalized]
            if i - loop_start >= 2:
                sentences = sentences[:i]
                break
        else:
            seen[sent_normalized] = i

    return '. '.join(sentences) + '.' if sentences else response


def _check_relevance(self, response: str, query: str) -> Dict[str, Any]:
    """Проверяет релевантность ответа к запросу"""
    if not response or not query:
        return {'score': 0.5, 'match_type': 'empty'}

    query_lower = query.lower()
    response_lower = response.lower()
    score = 0.7

    query_keywords = set(query_lower.split()[:5])
    response_keywords = set(response_lower.split()[:10])
    overlap = len(query_keywords.intersection(response_keywords))

    if overlap == 0:
        score -= 0.3

    score = max(0.0, min(1.0, score))
    return {'score': score, 'match_type': 'semantic' if score > 0.6 else 'weak'}
