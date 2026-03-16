#!/usr/bin/env python3
"""
Отладка генерации английского текста - показываем сырой вывод модели
"""

import sys
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_english_gen")

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_english_generation():
    """Отладка генерации английского текста."""
    try:
        logger.info("=== ОТЛАДКА ГЕНЕРАЦИИ АНГЛИЙСКОГО ТЕКСТА ===")

        # Импортируем исправленную версию
        from cogniflex.mlearning.fractal_model_manager_fixed import FractalModelManager

        # Создаем экземпляр и инициализируем
        logger.info("Создание и инициализация модели...")
        manager = FractalModelManager()

        init_result = manager.initialize()
        if not init_result:
            logger.error("Не удалось инициализировать модель")
            return False

        logger.info("Модель инициализирована успешно")

        # Тестируем генерацию английского текста с подробным выводом
        query = "Hello, how are you?"
        logger.info(f"Тестируем запрос: '{query}'")

        # Проверяем качество генерации напрямую
        if manager.model is not None and manager.tokenizer is not None:
            logger.info("Генерация ответа с подробным выводом...")

            try:
                # Токенизируем входной запрос
                from transformers import AutoTokenizer
                inputs = manager.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
                input_ids = inputs['input_ids'].to(manager.device)
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(manager.device)

                logger.info(f"Input tokens: {input_ids.tolist()}")
                logger.info(f"Decoded input: '{manager.tokenizer.decode(input_ids[0])}'")

                # Генерируем ответ БЕЗ наших проверок качества
                import torch
                with torch.no_grad():
                    raw_output = manager.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + 50,  # Короче для отладки
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.8,
                        top_k=40,
                        top_p=0.9,
                        pad_token_id=manager.tokenizer.eos_token_id,
                        eos_token_id=manager.tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                    )

                # Декодируем сырой вывод
                raw_decoded = manager.tokenizer.decode(raw_output[0], skip_special_tokens=True)
                logger.info(f"Сырой декодированный вывод: '{raw_decoded}'")

                # Показываем токены
                output_tokens = raw_output[0].tolist()
                logger.info(f"Output tokens: {output_tokens}")

                # Применяем нашу постобработку
                cleaned_response = manager._clean_generated_response(query, raw_decoded)
                logger.info(f"После очистки: '{cleaned_response}'")

                # Проверяем качество
                quality_good = manager._is_response_quality_good(cleaned_response)
                logger.info(f"Проходит проверку качества: {quality_good}")

                # Показываем причины, почему не проходит
                if not quality_good:
                    logger.info("Анализ причин низкого качества:")
                    logger.info(f"  - Длина: {len(cleaned_response)} (мин. 5)")
                    garbage_chars = ['�', '�', '�', '�', '�']
                    has_garbage = any(char in cleaned_response for char in garbage_chars)
                    logger.info(f"  - Содержит 'мусорные' символы: {has_garbage}")
                    if len(cleaned_response) > 10:
                        sample = cleaned_response[:20]
                        repetitions = cleaned_response.count(sample)
                        logger.info(f"  - Повторения sample[:20]: {repetitions} (макс. 2)")
                    words = cleaned_response.split()
                    logger.info(f"  - Количество слов: {len(words)} (мин. 3)")

                # Показываем fallback ответ
                fallback = manager._generate_info_response(query)
                logger.info(f"Fallback ответ: '{fallback[:100]}...'")

            except Exception as e:
                logger.error(f"Ошибка при генерации: {e}", exc_info=True)

        else:
            logger.warning("Модель или токенизатор недоступны - только fallback")

        # Тестируем несколько разных запросов
        test_queries = [
            "Hi there!",
            "What is AI?",
            "Tell me a joke.",
            "How does machine learning work?"
        ]

        logger.info("\n=== ТЕСТИРОВАНИЕ РАЗНЫХ ЗАПРОСОВ ===")
        for test_query in test_queries:
            logger.info(f"\nТест запроса: '{test_query}'")
            try:
                response = manager.generate_response(test_query)
                logger.info(f"Ответ: '{response[:100]}...'")
            except Exception as e:
                logger.error(f"Ошибка: {e}")

        logger.info("\n=== ОТЛАДКА ЗАВЕРШЕНА ===")
        return True

    except Exception as e:
        logger.error(f"Критическая ошибка при отладке: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = debug_english_generation()
    sys.exit(0 if success else 1)
