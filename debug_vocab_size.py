#!/usr/bin/env python3
"""
Отладочный тест для исследования проблемы несоответствия vocab_size.
Полное логгирование, отключен fallback.
"""

import sys
import os
import logging
import torch

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_vocab_size.log', mode='w')
    ]
)

logger = logging.getLogger("debug_vocab_size")

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_vocab_size():
    """Отладка проблемы несоответствия vocab_size."""
    try:
        logger.info("=== НАЧАЛО ОТЛАДКИ VOCAB_SIZE ===")

        # Импортируем исправленную версию
        from cogniflex.mlearning.fractal_model_manager_fixed import FractalModelManager

        # Создаем экземпляр
        logger.info("1. Создание экземпляра FractalModelManager...")
        manager = FractalModelManager()

        # Проверяем пути к файлам
        logger.info(f"Model path: {manager.model_path}")
        logger.info(f"Config path: {manager.config_path}")

        # Проверяем существование файлов
        logger.info(f"Model file exists: {os.path.exists(manager.model_path)}")
        logger.info(f"Config file exists: {os.path.exists(manager.config_path)}")

        # Ищем токенизатор
        logger.info("2. Поиск токенизатора в фрактальном хранилище...")
        tokenizer_paths = manager._find_tokenizer_in_fractal_storage()
        logger.info(f"Найден токенизатор: {tokenizer_paths}")

        # Проверяем содержимое config файла
        logger.info("3. Анализ конфигурации модели...")
        if os.path.exists(manager.config_path):
            with open(manager.config_path, 'r', encoding='utf-8') as f:
                config_data = f.read()
                logger.info(f"Config file content: {config_data[:500]}...")

        # Загружаем state_dict для анализа
        logger.info("4. Загрузка и анализ state_dict...")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(manager.model_path, device='cpu')
            logger.info(f"State dict loaded with {len(state_dict)} keys")

            # Анализируем ключи для определения параметров
            vocab_size_found = None
            n_embd_found = None
            n_layer_found = None

            for key, tensor in state_dict.items():
                logger.debug(f"Key: {key}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
                if 'wte.weight' in key:  # word token embeddings
                    vocab_size_found = tensor.shape[0]
                    n_embd_found = tensor.shape[1]
                    logger.info(f"Из wte.weight: vocab_size={vocab_size_found}, n_embd={n_embd_found}")
                elif key == 'transformer.h.0.attn.c_attn.weight':
                    logger.info(f"Из c_attn.weight: shape={tensor.shape}")
                    n_embd_found = tensor.shape[1] if n_embd_found is None else n_embd_found

            # Подсчитываем слои
            layer_keys = [k for k in state_dict.keys() if k.startswith('transformer.h.') and '.attn.c_attn.weight' in k]
            n_layer_found = len(layer_keys)
            logger.info(f"Найдено слоев: {n_layer_found}")

            logger.info(f"ОПРЕДЕЛЕННЫЕ ПАРАМЕТРЫ МОДЕЛИ: vocab_size={vocab_size_found}, n_embd={n_embd_found}, n_layer={n_layer_found}")

        except Exception as e:
            logger.error(f"Ошибка при анализе state_dict: {e}", exc_info=True)
            return False

        # Проверяем токенизатор отдельно
        logger.info("5. Проверка токенизатора...")
        if tokenizer_paths:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_paths, local_files_only=True)
                logger.info(f"Токенизатор загружен. vocab_size: {tokenizer.vocab_size}")
                logger.info(f"eos_token_id: {tokenizer.eos_token_id}")
                logger.info(f"bos_token_id: {tokenizer.bos_token_id}")
                logger.info(f"pad_token: {tokenizer.pad_token}")

                # Проверяем первые несколько токенов
                test_text = "Hello world"
                tokens = tokenizer.encode(test_text, add_special_tokens=True)
                logger.info(f"Токены для '{test_text}': {tokens}")

                decoded = tokenizer.decode(tokens)
                logger.info(f"Декодированные токены: '{decoded}'")

            except Exception as e:
                logger.error(f"Ошибка при проверке токенизатора: {e}", exc_info=True)

        # Проверяем альтернативные токенизаторы
        logger.info("6. Проверка альтернативных токенизаторов...")

        alt_tokenizers = [
            'gpt2',
            'sberbank-ai/rugpt3small_based_on_gpt2',
            'microsoft/DialoGPT-medium'
        ]

        for tok_name in alt_tokenizers:
            try:
                logger.info(f"Проверка токенизатора: {tok_name}")
                alt_tok = AutoTokenizer.from_pretrained(tok_name)
                logger.info(f"  vocab_size: {alt_tok.vocab_size}")
                if alt_tok.vocab_size == vocab_size_found:
                    logger.info(f"  ✅ СОВПАДЕНИЕ! Токенизатор {tok_name} подходит для модели")
                else:
                    logger.info(f"  ❌ Несоответствие: {alt_tok.vocab_size} != {vocab_size_found}")
            except Exception as e:
                logger.warning(f"  Не удалось загрузить {tok_name}: {e}")

        logger.info("=== ОТЛАДКА ЗАВЕРШЕНА ===")
        return True

    except Exception as e:
        logger.error(f"Критическая ошибка при отладке: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = debug_vocab_size()
    sys.exit(0 if success else 1)
