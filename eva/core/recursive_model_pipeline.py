"""
Recursive Model Pipeline - Three GGUF models working sequentially
для рекурсивной генерации ответов с переводом в натуральный язык и проверкой качества

Использует chat_format="qwen" для автоматической обработки темплейта Qwen

Pipeline:
  Model A (Qwen 2.5 3B) - логика, краткий ответ
  Model B (Qwen 2.5 3B) - развитие мысли, развёрнутый ответ
  Model C (Qwen 2.5 Coder 1.5B) - генерация кода (если запрос содержит код)
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class RecursiveModelPipeline:
    """
    Пайплайн для последовательной работы GGUF моделей:
    1. Model A (Qwen 2.5 3B) - даёт краткий логичный ответ
    2. Model B (Qwen 2.5 3B) - развивает мысль, добавляет детали
    3. Model C (Qwen 2.5 Coder 1.5B) - генерирует код, если нужен
    
    Использует create_chat_completion с автоматическим форматированием Qwen
    """
    
    # Настройки генерации для Model A (основная модель)
    MODEL_A_MAX_TOKENS = 1024
    MODEL_A_TEMPERATURE = 0.3
    MODEL_A_TOP_P = 0.9
    MODEL_A_TOP_K = 40
    MODEL_A_REPEAT_PENALTY = 1.5
    
    # Настройки генерации для Model B (развитие)
    MODEL_B_MAX_TOKENS = 512
    MODEL_B_TEMPERATURE = 0.3
    MODEL_B_TOP_P = 0.9
    MODEL_B_TOP_K = 40
    MODEL_B_REPEAT_PENALTY = 2.0
    
    # Настройки генерации для Model C (кодер)
    MODEL_C_MAX_TOKENS = 512
    MODEL_C_TEMPERATURE = 0.1
    MODEL_C_TOP_P = 0.9
    MODEL_C_TOP_K = 50
    MODEL_C_REPEAT_PENALTY = 1.3
    
    def __init__(
        self,
        model_a_path: str,
        model_b_path: str,
        model_c_path: str = None,
        n_ctx: int = 8192,
        n_threads: int = 8,
        fractal_memory = None
    ):
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.model_c_path = model_c_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.model_a = None
        self.model_b = None
        self.model_c = None
        self.fractal_memory = fractal_memory
        self.quality_checker = None
        
        logger.info(f"RecursiveModelPipeline инициализирован (3-модельный)")
    
    def load_models(self):
        """Загрузка GGUF моделей - Model A и B как отдельные экземпляры"""
        a_ctx = min(self.n_ctx, 2048)
        
        logger.info(f"Загрузка Model A: {self.model_a_path}")
        self.model_a = Llama(
            model_path=self.model_a_path,
            chat_format="qwen",
            n_ctx=a_ctx,
            n_threads=self.n_threads,
            verbose=False
        )
        logger.info(f"Model A загружена с контекстом {a_ctx}")
        
        if self.fractal_memory:
            self.fractal_memory.register_model_instance("model_a", self.model_a)
        
        # Model B - отдельный экземпляр (памяти хватает благодаря mmap)
        logger.info(f"Загрузка Model B: {self.model_b_path}")
        self.model_b = Llama(
            model_path=self.model_b_path,
            chat_format="qwen",
            n_ctx=a_ctx,
            n_threads=self.n_threads,
            verbose=False
        )
        logger.info(f"Model B загружена с контекстом {a_ctx}")
        
        if self.fractal_memory:
            self.fractal_memory.register_model_instance("model_b", self.model_b)
        
        # Model C загружается лениво (только при запросе кода)
        self.model_c = None
        if self.model_c_path and os.path.exists(self.model_c_path):
            logger.info(f"Model C будет загружена лениво при запросе кода")
        else:
            logger.info("Model C не указана")
    
    def _is_code_request(self, query: str) -> bool:
        """Определяет, нужен ли код в ответе"""
        code_keywords = [
            'напиши код', 'напиши функцию', 'напиши скрипт', 'код для',
            'функцию для', 'скрипт для', 'программу', 'код на python',
            'код на js', 'код на javascript', 'напиши программу',
            'реализуй', 'реализовать', 'функция которая', 'класс для',
            'def ', 'import ', 'function ', 'class ', 'const ', 'let ',
            '```', 'print(', 'return ', 'async ', 'await '
        ]
        query_lower = query.lower()
        for kw in code_keywords:
            if kw in query_lower:
                return True
        return False
    
    def check_quality(self, text: str) -> Dict[str, Any]:
        """Проверка качества текста с детекцией зацикливания"""
        if not text or len(text.strip()) < 5:
            return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Пустой или слишком короткий текст']}
        
        words = text.split()
        unique_words = set(words) if words else set()
        
        # Проверка на повторения слов
        if len(words) > 10 and len(unique_words) / len(words) < 0.3:
            return {'is_gibberish': True, 'score': 0.2, 'reasons': ['Слишком много повторений слов']}
        
        # Проверка на гласные
        vowels = set('аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouAEIOU')
        if not any(v in text for v in vowels):
            return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Нет гласных букв']}
        
        # Проверка на зацикливание предложений
        lines = text.split('\n')
        repeating_lines = {}
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                repeating_lines[line] = repeating_lines.get(line, 0) + 1
        max_repeats = max(repeating_lines.values()) if repeating_lines else 1
        if max_repeats > 2:
            return {'is_gibberish': True, 'score': 0.3, 'reasons': ['Зацикливание: одинаковые предложения повторяются']}
        
        # Проверка на начальные фразы-паразиты
        filler_starts = ['Конечно!', 'Конечно', 'Вот более', 'Вот что', 'Это всё', '---', '***']
        for filler in filler_starts:
            if text.startswith(filler):
                return {'is_gibberish': True, 'score': 0.4, 'reasons': ['Начинается с фразы-паразита']}
        
        # Проверка на смешение языков (китайские/английские вставки)
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        if chinese_chars > 5:
            return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Содержит китайские символы']}
        
        # Проверка на английские блоки (более 30% английских слов)
        words = text.split()
        english_words = sum(1 for w in words if w.isascii() and len(w) > 3)
        if len(words) > 10 and english_words / len(words) > 0.3:
            return {'is_gibberish': True, 'score': 0.3, 'reasons': ['Слишком много английских слов']}
        
        return {'is_gibberish': False, 'score': 0.8, 'reasons': ['OK']}
    
    def _sanitize_response(self, text: str) -> str:
        """Постобработка: удаление артефактов, латиницы в кириллических словах"""
        import re
        
        # Удаляем артефакты retry
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
        
        # Заменяем смешанные слова с латиницей
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
            'рек': 'рек',
            'омend': 'оменд',
        }
        for eng, rus in replacements.items():
            text = text.replace(eng, rus)
        
        # Находим слова со смешением кириллицы и латиницы и помечаем
        # Паттерн: русское слово с латинскими буквами внутри
        mixed_pattern = re.compile(r'[а-яА-ЯёЁ]+[a-zA-Z]+[а-яА-ЯёЁ]+|[a-zA-Z]+[а-яА-ЯёЁ]+[a-zA-Z]+')
        def fix_mixed(match):
            word = match.group(0)
            # Заменяем латинские буквы на похожие кириллические
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
        
        return text.strip()

    def generate_with_model_a(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """Генерация ответа на Model A (логика)"""
        if not self.model_a:
            raise RuntimeError("Model A не загружена")
        
        logger.info(f"Model A query: {query[:100]}...")
        
        # Для коротких запросов добавляем контекст чтобы модель не генерировала китайский
        user_content = query
        if len(query.strip()) < 20:
            user_content = 'Пользователь написал: "' + query + '". Ответь вежливо на русском языке.'
        
        for attempt in range(max_retries + 1):
            messages = [
                {"role": "system", "content": "Ты — ЕВА, русскоязычный ИИ. Отвечай строго на русском языке. Не используй английские, китайские или другие иностранные слова и аббревиатуры."},
                {"role": "user", "content": user_content}
            ]
            
            output = self.model_a.create_chat_completion(
                messages=messages,
                max_tokens=self.MODEL_A_MAX_TOKENS,
                temperature=self.MODEL_A_TEMPERATURE + (attempt * 0.1),
                top_p=self.MODEL_A_TOP_P,
                top_k=self.MODEL_A_TOP_K,
                repeat_penalty=self.MODEL_A_REPEAT_PENALTY + (attempt * 0.15),
                stop=["</s>"]
            )
            
            raw_response = output['choices'][0]['message']['content'].strip()
            raw_response = self._sanitize_response(raw_response)
            quality = self.check_quality(raw_response)
            
            # Никогда не возвращаем ответ с китайскими символами
            has_chinese = sum(1 for c in raw_response if '一' <= c <= '鿿') > 5
            if has_chinese:
                logger.warning(f"Model A: попытка {attempt+1} содержит китайские символы, retry")
                continue
            
            if not quality['is_gibberish']:
                logger.info(f"Model A response (attempt {attempt+1}): '{raw_response[:200]}...'")
                logger.info(f"Model A tokens: {output['usage']['completion_tokens']}")
                
                return {
                    'raw_response': raw_response,
                    'natural_response': raw_response,
                    'quality': quality,
                    'model': 'Model A (Logic)',
                    'tokens': output['usage']['completion_tokens']
                }
            
            logger.warning(f"Model A: попытка {attempt+1} не прошла проверку качества: {quality['reasons']}")
        
        # Fallback
        return {
            'raw_response': 'Не удалось сформировать ответ.',
            'natural_response': 'Не удалось сформировать ответ.',
            'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Все попытки не прошли проверку']},
            'model': 'Model A (Logic)',
            'tokens': 0
        }
    
    def generate_with_model_b(self, query: str, previous_response: str, max_retries: int = 2) -> Dict[str, Any]:
        """Генерация ответа на Model B - независимый ответ на тот же вопрос"""
        if not self.model_b:
            raise RuntimeError("Model B не загружена")
        
        # Model B получает ТОЛЬКО оригинальный запрос, без контекста Model A
        # Это предотвращает каскадное загрязнение
        for attempt in range(max_retries + 1):
            messages = [
                {"role": "system", "content": "Ты — ЕВА, русскоязычный ИИ. Отвечай строго на русском языке. Не используй английские, китайские или другие иностранные слова. Давай развёрнутый ответ с примерами."},
                {"role": "user", "content": query}
            ]
            
            output = self.model_b.create_chat_completion(
                messages=messages,
                max_tokens=self.MODEL_B_MAX_TOKENS,
                temperature=self.MODEL_B_TEMPERATURE + (attempt * 0.1),
                top_p=self.MODEL_B_TOP_P,
                top_k=self.MODEL_B_TOP_K,
                repeat_penalty=self.MODEL_B_REPEAT_PENALTY + (attempt * 0.15),
                stop=["</s>"]
            )
            
            raw_response = output['choices'][0]['message']['content'].strip()
            raw_response = self._sanitize_response(raw_response)
            raw_response = self._clean_filler_start(raw_response)
            raw_response = self._remove_looping_blocks(raw_response)
            quality = self.check_quality(raw_response)
            
            # Никогда не возвращаем ответ с китайскими символами
            has_chinese = sum(1 for c in raw_response if '一' <= c <= '鿿') > 5
            if has_chinese:
                logger.warning(f"Model B: попытка {attempt+1} содержит китайские символы, retry")
                continue
            
            if not quality['is_gibberish']:
                logger.info(f"Model B response (attempt {attempt+1}): '{raw_response[:200]}...'")
                logger.info(f"Model B tokens: {output['usage']['completion_tokens']}")
                
                return {
                    'raw_response': raw_response,
                    'natural_response': raw_response,
                    'quality': quality,
                    'model': 'Model B (Concept)',
                    'tokens': output['usage']['completion_tokens']
                }
            
            logger.warning(f"Model B: попытка {attempt+1} не прошла проверку качества: {quality['reasons']}")
        
        # Fallback на Model A
        return {
            'raw_response': previous_response if previous_response else 'Не удалось сформировать ответ.',
            'natural_response': previous_response if previous_response else 'Не удалось сформировать ответ.',
            'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Все попытки не прошли проверку']},
            'model': 'Model B (Concept)',
            'tokens': 0
        }
    
    def _load_model_c(self):
        """Ленивая загрузка Model C (только при необходимости)"""
        if self.model_c is not None:
            return True
        if not self.model_c_path or not os.path.exists(self.model_c_path):
            logger.warning("Model C не найдена")
            return False
        try:
            logger.info(f"Ленивая загрузка Model C: {self.model_c_path}")
            model_c_ctx = min(self.n_ctx, 2048)
            self.model_c = Llama(
                model_path=self.model_c_path,
                chat_format="qwen",
                n_ctx=model_c_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            logger.info(f"Model C загружена с контекстом {model_c_ctx}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки Model C: {e}")
            return False

    def generate_with_model_c(self, query: str, context: str, max_retries: int = 2) -> Dict[str, Any]:
        """Генерация кода на Model C (Coder) с ленивой загрузкой"""
        if not self._load_model_c():
            raise RuntimeError("Model C (Coder) не может быть загружена")
        
        messages = [
            {"role": "system", "content": "Ты - ЕВА, помощница-программист. Пиши чистый, рабочий код. Комментарии на русском языке."},
            {"role": "user", "content": f"[НА РУССКОМ] Контекст: {context}\n\nЗапрос: {query}\n\nНапиши код."}
        ]
        
        logger.info(f"Model C (Coder) query: {query[:100]}...")
        
        for attempt in range(max_retries + 1):
            output = self.model_c.create_chat_completion(
                messages=messages,
                max_tokens=self.MODEL_C_MAX_TOKENS,
                temperature=self.MODEL_C_TEMPERATURE + (attempt * 0.05),
                top_p=self.MODEL_C_TOP_P,
                top_k=self.MODEL_C_TOP_K,
                repeat_penalty=self.MODEL_C_REPEAT_PENALTY + (attempt * 0.1),
                stop=["</s>"]
            )
            
            raw_response = output['choices'][0]['message']['content'].strip()
            quality = self.check_quality(raw_response)
            
            if not quality['is_gibberish'] or attempt == max_retries:
                logger.info(f"Model C response (attempt {attempt+1}): '{raw_response[:200]}...'")
                logger.info(f"Model C tokens: {output['usage']['completion_tokens']}")
                
                return {
                    'raw_response': raw_response,
                    'natural_response': raw_response,
                    'quality': quality,
                    'model': 'Model C (Coder)',
                    'tokens': output['usage']['completion_tokens']
                }
            
            logger.warning(f"Model C: попытка {attempt+1} не прошла проверку качества: {quality['reasons']}")
    
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
    
    def process_query(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Основной метод обработки запроса через 3-модельный пайплайн"""
        if not self.model_a or not self.model_b:
            raise RuntimeError("Модели не загружены. Вызовите load_models()")
        
        # Применяем динамические параметры (локальные переменные)
        a_max_tokens = self.MODEL_A_MAX_TOKENS
        a_temperature = self.MODEL_A_TEMPERATURE
        b_max_tokens = self.MODEL_B_MAX_TOKENS
        b_temperature = self.MODEL_B_TEMPERATURE
        
        if gen_params:
            params_a = gen_params.get('model_a', {})
            params_b = gen_params.get('model_b', {})
            a_max_tokens = params_a.get('max_tokens', self.MODEL_A_MAX_TOKENS)
            a_temperature = params_a.get('temperature', self.MODEL_A_TEMPERATURE)
            b_max_tokens = params_b.get('max_tokens', self.MODEL_B_MAX_TOKENS)
            b_temperature = params_b.get('temperature', self.MODEL_B_TEMPERATURE)
            logger.info(f"Применены динамические параметры: A(temp={self.MODEL_A_TEMPERATURE}), B(temp={self.MODEL_B_TEMPERATURE})")
        
        results = {
            'query': query,
            'model_a_result': None,
            'model_b_result': None,
            'model_c_result': None,
            'reasoning_steps': [],
            'has_code': False,
            'fractal_context': None
        }
        
        # Контекст из графа обучения (чистый — только концепты и качественные опыты)
        enriched_query = query
        if self.fractal_memory and hasattr(self.fractal_memory, 'get_context_for_query'):
            graph_context = self.fractal_memory.get_context_for_query(query)
            if graph_context:
                enriched_query = f"{query}\n\nКонтекст из опыта:\n{graph_context}"
                results['fractal_context'] = graph_context
                logger.info(f"Контекст из графа обучения: {len(graph_context)} символов")
        
        # Шаг 1: Model A - логический ответ
        logger.info("=== Шаг 1: Генерация ответа на Model A (логическое ядро) ===")
        model_a_result = self.generate_with_model_a(enriched_query)
        logger.info(f"Model A ответ: {model_a_result['natural_response'][:150]}...")
        results['model_a_result'] = model_a_result
        
        results['reasoning_steps'].append({
            'step': 1,
            'phase': 'model_a_generation',
            'thought': model_a_result['natural_response'][:200],
            'confidence': model_a_result['quality'].get('score', 0.8),
            'model': 'Model A (Logic)',
            'action': 'Извлечение фактов',
            'input': query,
            'output': model_a_result['natural_response']
        })
        
        # Шаг 2: Model B - развитие мысли (чистый контекст, без загрязнения)
        logger.info("=== Шаг 2: Генерация расширенного ответа на Model B ===")
        model_b_result = self.generate_with_model_b(query, model_a_result['natural_response'])
        logger.info(f"Model B ответ: {model_b_result['natural_response'][:150]}...")
        results['model_b_result'] = model_b_result
        
        results['reasoning_steps'].append({
            'step': 2,
            'phase': 'model_b_generation',
            'thought': model_b_result['natural_response'][:200],
            'confidence': model_b_result['quality'].get('score', 0.8),
            'model': 'Model B (Concept)',
            'action': 'Расширение концепций',
            'input': f"Факты: {model_a_result['natural_response'][:100]}",
            'output': model_b_result['natural_response']
        })
        
        # Финальный ответ - Model B (расширенный)
        results['final_response'] = model_b_result['natural_response']
        
        # Шаг 3: Model C - код (если нужен)
        if self.model_c and self._is_code_request(query):
            logger.info("=== Шаг 3: Генерация кода на Model C (Coder) ===")
            results['has_code'] = True
            model_c_result = self.generate_with_model_c(query, model_b_result['natural_response'])
            logger.info(f"Model C ответ: {model_c_result['natural_response'][:150]}...")
            results['model_c_result'] = model_c_result
            
            results['reasoning_steps'].append({
                'step': 3,
                'phase': 'model_c_generation',
                'thought': model_c_result['natural_response'][:200],
                'confidence': model_c_result['quality'].get('score', 0.8),
                'model': 'Model C (Coder)',
                'action': 'Генерация кода',
                'input': f"Контекст: {model_b_result['natural_response'][:100]}",
                'output': model_c_result['natural_response']
            })
            
            # Финальный ответ = текст Model B + код Model C
            results['final_response'] = model_b_result['natural_response'] + "\n\n" + model_c_result['natural_response']
        else:
            # Финальный ответ - используем Model B (расширенный)
            results['final_response'] = model_b_result['natural_response']
        
        results['final_quality'] = model_b_result['quality']
        
        # Сохраняем опыт в граф обучения (цикл обучения через граф)
        if self.fractal_memory and hasattr(self.fractal_memory, 'save_experience'):
            # Сохраняем оба ответа как опыт
            self.fractal_memory.save_experience(
                query=query,
                response=model_a_result['natural_response'],
                model_used='model_a',
                quality_score=model_a_result['quality'].get('score', 0.5)
            )
            self.fractal_memory.save_experience(
                query=query,
                response=model_b_result['natural_response'],
                model_used='model_b',
                quality_score=model_b_result['quality'].get('score', 0.5)
            )
        
        logger.info("Three-GGUF пайплайн завершён")
        
        return results


def create_recursive_pipeline(
    model_a_path: str = None,
    model_b_path: str = None,
    model_c_path: str = None,
    n_ctx: int = 8192,
    n_threads: int = 8,
    fractal_memory = None
) -> 'RecursiveModelPipeline':
    """Фабричная функция для создания пайплайна"""
    if model_a_path is None:
        model_a_path = r"C:/Users/black/OneDrive/Desktop/CogniFlex/eva/memory/fractal_torch_storage/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    if model_b_path is None:
        model_b_path = r"C:/Users/black/OneDrive/Desktop/CogniFlex/eva/memory/fractal_torch_storage/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    if model_c_path is None:
        model_c_path = r"C:/Users/black/OneDrive/Desktop/CogniFlex/eva/memory/fractal_torch_storage/gguf_models/qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    
    pipeline = RecursiveModelPipeline(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        model_c_path=model_c_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        fractal_memory=fractal_memory
    )
    pipeline.load_models()
    return pipeline


if __name__ == "__main__":
    import sys
    import io
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Создание Recursive Model Pipeline (Three-GGUF)")
    print("=" * 60)
    
    pipeline = create_recursive_pipeline()
    
    test_query = "Что такое искусственный интеллект?"
    print(f"\nЗапрос: {test_query}")
    print("-" * 40)
    
    results = pipeline.process_query(test_query)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    print("\n[Model A - Логическое ядро]")
    print(f"Ответ: {results['model_a_result']['natural_response']}")
    
    print("\n[Model B - Расширение концепций]")
    print(f"Ответ: {results['model_b_result']['natural_response']}")
    
    if results.get('model_c_result'):
        print("\n[Model C - Кодер]")
        print(f"Ответ: {results['model_c_result']['natural_response']}")
