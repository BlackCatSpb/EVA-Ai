"""
Quality checking and response sanitization for the Recursive Model Pipeline.
"""

import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_generation_executor = ThreadPoolExecutor(max_workers=4)
import atexit
atexit.register(lambda: _generation_executor.shutdown(wait=False))


def check_quality(self, text: str) -> Dict[str, Any]:
    """Проверка качества текста с детекцией зацикливания"""
    if not text or len(text.strip()) < 5:
        return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Пустой или слишком короткий текст']}
    
    words = text.split()
    unique_words = set(words) if words else set()
    
    if len(words) > 10 and len(unique_words) / len(words) < 0.3:
        return {'is_gibberish': True, 'score': 0.2, 'reasons': ['Слишком много повторений слов']}
    
    vowels = set('аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouAEIOU')
    if not any(v in text for v in vowels):
        return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Нет гласных букв']}
    
    lines = text.split('\n')
    repeating_lines = {}
    for line in lines:
        line = line.strip()
        if len(line) > 20:
            repeating_lines[line] = repeating_lines.get(line, 0) + 1
    max_repeats = max(repeating_lines.values()) if repeating_lines else 1
    if max_repeats > 2:
        return {'is_gibberish': True, 'score': 0.3, 'reasons': ['Зацикливание: одинаковые предложения повторяются']}
    
    filler_starts = ['Вот более', 'Вот что', '---', '***']
    for filler in filler_starts:
        if text.startswith(filler):
            return {'is_gibberish': True, 'score': 0.4, 'reasons': ['Начинается с фразы-паразита']}
    
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    if chinese_chars > 5:
        return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Содержит китайские символы']}
    
    english_words = sum(1 for w in words if w.isascii() and len(w) > 3)
    if len(words) > 10 and english_words / len(words) > 0.3:
        return {'is_gibberish': True, 'score': 0.3, 'reasons': ['Слишком много английских слов']}
    
    return {'is_gibberish': False, 'score': 0.8, 'reasons': ['OK']}


def _sanitize_response(self, text: str) -> str:
    """Постобработка: удаление артефактов, латиницы в кириллических словах"""
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"
    
    text = re.sub(r'```[\s\S]*?```', save_code_block, text)
    
    retry_artifacts = [
        'Прости за ошибку',
        'Извините за предыдущий',
        'ВНИМАНИЕ:',
        'предыдущий ответ был некорректным',
    ]
    for artifact in retry_artifacts:
        if artifact in text:
            lines = text.split('\n')
            lines = [l for l in lines if artifact not in l]
            text = '\n'.join(lines)
    
    replacements = {
        'ИнтелLECT': 'Интеллект',
        'LECT': 'лект',
        'TEL': 'тел',
        'IA': 'ИИ',
        'AI': 'ИИ',
        'system': 'система',
        'software': 'программа',
        'rek': 'рек',
        'omend': 'оменд',
    }
    for eng, rus in replacements.items():
        text = text.replace(eng, rus)
    
    mixed_pattern = re.compile(r'[а-яА-ЯёЁ]+[a-zA-Z]+[а-яА-ЯёЁ]+|[a-zA-Z]+[а-яА-ЯёЁ]+[a-zA-Z]+')
    def fix_mixed(match):
        word = match.group(0)
        latin_to_cyrillic = {
            'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
            'x': 'х', 'y': 'у', 'A': 'А', 'E': 'Е', 'O': 'О',
            'P': 'Р', 'C': 'С', 'X': 'Х', 'Y': 'У',
            'i': 'і', 'I': 'І', 'j': 'ј', 'J': 'Ј',
            'k': 'к', 'K': 'К', 'm': 'м', 'M': 'М',
            'T': 'Т', 't': 'т', 'B': 'В', 'b': 'ь',
            'H': 'Н', 'h': 'н',
        }
        result = ''
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in word)
        for ch in word:
            if has_cyrillic and ch in latin_to_cyrillic:
                result += latin_to_cyrillic[ch]
            else:
                result += ch
        return result
    
    text = mixed_pattern.sub(fix_mixed, text)
    
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODEBLOCK{i}\x00", block)
    
    return text.strip()


def _clean_filler_start(self, text: str) -> str:
    """Удаляет вводные фразы-паразиты из начала ответа"""
    fillers = [
        'Конечно! ', 'Конечно, ', 'Конечно!', 'Конечно',
        'Вот более ', 'Вот что ', 'Вот ',
        'Это всё ', 'Это все ',
        'Ваш вопрос ', 'Ваш запрос ',
        '---', '***',
    ]
    cleaned = text
    for filler in fillers:
        if cleaned.startswith(filler):
            cleaned = cleaned[len(filler):].strip()
            break
    return cleaned


def _remove_looping_blocks(self, text: str, max_repeats: int = 2) -> str:
    """Удаляет зацикливающиеся блоки текста"""
    lines = text.split('\n')
    seen_blocks = {}
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 30:
            if stripped in seen_blocks:
                seen_blocks[stripped] += 1
                if seen_blocks[stripped] > max_repeats:
                    continue
            else:
                seen_blocks[stripped] = 1
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def check_russian_quality(text: str) -> Dict[str, Any]:
    """
    Строгая проверка грамматики, синтаксиса и пунктуации для Model B.
    Возвращает причины ошибок для перегенерации.
    """
    if not text or len(text.strip()) < 10:
        return {'is_valid': False, 'score': 0.0, 'reasons': ['Слишком короткий текст']}
    
    reasons = []
    score = 1.0
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        reasons.append('Нет законченных предложений')
        score -= 0.5
    
    has_proper_punctuation = False
    for sent in sentences:
        if sent.endswith(('.', '!', '?', '...')):
            has_proper_punctuation = True
            break
    if not has_proper_punctuation and len(sentences) > 1:
        reasons.append('Предложения не заканчиваются знаками препинания')
        score -= 0.3
    
    words = text.split()
    if words:
        first_word = words[0]
        if first_word and first_word[0].isupper() and len(first_word) > 1:
            pass
        elif first_word and first_word.isalpha():
            reasons.append('Предложение начинается со строчной буквы')
            score -= 0.2
        
        last_word = words[-1] if words else ''
        if last_word and not any(last_word.endswith(p) for p in '.!?...'):
            if len(sentences) > 1:
                reasons.append('Последнее предложение без завершающего знака')
                score -= 0.1
    
    cyrillic_pattern = re.compile(r'[а-яА-ЯёЁ]')
    latin_pattern = re.compile(r'[a-zA-Z]')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    has_cyrillic = cyrillic_pattern.search(text)
    has_latin = latin_pattern.search(text)
    has_chinese = chinese_pattern.search(text)
    
    if has_latin and not has_cyrillic:
        reasons.append('Текст на английском языке')
        score -= 0.8
    elif has_latin:
        reasons.append('Содержит английские слова')
        score -= 0.2
    if has_chinese:
        reasons.append('Содержит китайские символы')
        score -= 0.8
    
    russian_prepositions = ['в', 'на', 'с', 'со', 'по', 'к', 'за', 'из', 'от', 'до', 'о', 'об', 'у', 'при', 'для', 'без', 'под', 'над']
    russian_conjunctions = ['и', 'а', 'но', 'или', 'что', 'как', 'если', 'когда', 'потому', 'поэтому']
    
    has_russian_structure = False
    for word in words[:20]:
        word_lower = word.lower().strip('.,!?;:')
        if word_lower in russian_prepositions or word_lower in russian_conjunctions:
            has_russian_structure = True
            break
    
    if not has_russian_structure and len(words) > 10:
        reasons.append('Нет структуры русского текста (предлогов/союзов)')
        score -= 0.2
    
    repeated_words = {}
    for word in words:
        word_clean = word.lower().strip('.,!?;:()[]')
        if len(word_clean) > 3:
            repeated_words[word_clean] = repeated_words.get(word_clean, 0) + 1
    
    max_repeat = max(repeated_words.values()) if repeated_words else 1
    if max_repeat > 5 and len(words) > 20:
        reasons.append('Слишком много повторений слов')
        score -= 0.3
    
    if re.search(r'[а-яё]{3,}[A-Z]{2,}|[A-Z]{2,}[а-яё]{3,}', text):
        reasons.append('Смешанный регистр (кириллица + латиница)')
        score -= 0.3
    
    empty_lines = [line.strip() for line in text.split('\n') if line.strip() == '']
    if len(empty_lines) > len(text.split('\n')) * 0.5:
        reasons.append('Слишком много пустых строк')
        score -= 0.1
    
    if score < 0:
        score = 0.0
    
    return {
        'is_valid': score >= 0.5,
        'score': score,
        'reasons': reasons if reasons else ['OK'],
        'sentences_count': len(sentences),
        'has_proper_punctuation': has_proper_punctuation
    }


def _fix_russian_punctuation(text: str) -> str:
    """Автоматическое исправление пунктуации"""
    lines = text.split('\n')
    result_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line and not line[0].isupper() and not line.startswith('•') and not line.startswith('-'):
            line = line[0].upper() + line[1:] if len(line) > 1 else line.upper()
        
        if line and not any(line.endswith(p) for p in '.!?...'):
            if len(line) > 20:
                line = line.rstrip() + '.'
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def _generate_with_timeout(self, model, messages: list, params: dict, timeout: int = None, logit_bias: dict = None) -> Optional[Dict]:
    """Генерация с таймаутом через ThreadPoolExecutor."""
    def _generate():
        kwargs = {
            'messages': messages,
            'max_tokens': params['max_tokens'],
            'temperature': params['temperature'],
            'top_p': params['top_p'],
            'top_k': params['top_k'],
            'repeat_penalty': params['repeat_penalty'],
            'stop': self.STOP_TOKENS
        }
        if logit_bias:
            kwargs['logit_bias'] = logit_bias
        return model.create_chat_completion(**kwargs)
    
    future = _generation_executor.submit(_generate)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning(f"Генерация прервана по таймауту ({timeout}с)")
        return None
    except Exception as e:
        logger.warning(f"Ошибка генерации: {e}")
        return None
