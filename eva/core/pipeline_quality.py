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


def _generate_with_timeout(self, model, messages: list, params: dict, timeout: int = 45) -> Optional[Dict]:
    """Генерация с таймаутом через ThreadPoolExecutor."""
    def _generate():
        return model.create_chat_completion(
            messages=messages,
            max_tokens=params['max_tokens'],
            temperature=params['temperature'],
            top_p=params['top_p'],
            top_k=params['top_k'],
            repeat_penalty=params['repeat_penalty'],
            stop=self.STOP_TOKENS
        )
    
    future = _generation_executor.submit(_generate)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning(f"Генерация прервана по таймауту ({timeout}с)")
        return None
    except Exception as e:
        logger.warning(f"Ошибка генерации: {e}")
        return None
