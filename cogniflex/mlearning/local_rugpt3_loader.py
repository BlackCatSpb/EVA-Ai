"""
Локальный загрузчик ruGPT-3 Large с улучшенной поддержкой фрактального хранилища
Только локальные модели, без загрузки с HuggingFace
"""

import os
import sys
import json
import logging
from typing import Optional, Tuple, Any, Dict, List

# Добавляем корневую директорию проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Настройка логгера
logger = logging.getLogger("cogniflex.mlearning.local_rugpt3_loader")

# Флаг отключения HuggingFace (только локальные модели)
DISABLE_HUGGINGFACE_FALLBACK = False  # Включаем fallback для надежности

# Пытаемся импортировать torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch недоступен - функции генерации будут ограничены")

# Импорты из transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    GPT2LMHeadModel = None
    GPT2Tokenizer = None
    GPT2Config = None
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers недоступен - загрузка моделей будет невозможна")

class Localrugpt3largeLoader:
    """Локальный загрузчик ruGPT-3 Large из фрактального хранилища"""
    
    def __init__(self, storage_path: str = None):
        # Автоматически определяем путь к модели
        if storage_path is None:
            # Пробуем найти модель в стандартных местах
            possible_paths = [
                # Новые пути к ruGPT-3 Large во фрактальном хранилище
                "cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_large_fractal",
                "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal",
                # Путь к rugpt3_large (полная модель ~3GB)
                os.path.join(os.path.dirname(__file__), "cogniflex_models", "rugpt3_large"),
                # Альтернативный путь от корня проекта
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cogniflex", "mlearning", "cogniflex_models", "rugpt3_large"),
                # Старые пути (для обратной совместимости)
                "./cogniflex_cache/ml_unit/fractal_storage/rugpt3large",
                "./cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/text-generation",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    # Проверяем наличие модели
                    model_file = os.path.join(path, "pytorch_model.bin")
                    if os.path.exists(model_file):
                        storage_path = path
                        logger.info(f"✅ Найдена локальная модель: {path}")
                        break
            
            if storage_path is None:
                storage_path = possible_paths[0]  # По умолчанию первый путь
                logger.warning(f"⚠️ Локальная модель не найдена, используем путь по умолчанию: {storage_path}")
        
        self.storage_path = storage_path
        logger.info(f"Localrugpt3largeLoader инициализирован с путем: {storage_path}")
        logger.info(f"Текущая рабочая директория: {os.getcwd()}")
        logger.info(f"Абсолютный путь: {os.path.abspath(storage_path)}")
        
        # Проверяем что путь не является HuggingFace репозиторием
        if storage_path.startswith('sberbank-ai/') or storage_path.startswith('huggingface/'):
            logger.error(f"Обнаружен HuggingFace путь: {storage_path}")
            logger.error("❌ Localrugpt3largeLoader работает только с локальными моделями!")
            raise ValueError(f"Localrugpt3largeLoader не поддерживает HuggingFace пути: {storage_path}")
        
        self.model_config = None
        self.tokenizer_config = None
        self.metadata = None
        
    def load_configs(self) -> bool:
        """Загружает конфигурации"""
        try:
            # Загружаем конфигурацию модели
            config_path = os.path.join(self.storage_path, "model", "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.model_config = json.load(f)
            
            # Загружаем конфигурацию токенизатора
            tokenizer_config_path = os.path.join(self.storage_path, "tokenizer", "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                    self.tokenizer_config = json.load(f)
            
            # Загружаем метаданные
            metadata_path = os.path.join(self.storage_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            
            logger.info("✅ Конфигурации загружены")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигураций: {e}")
            return False
    
    def create_tokenizer(self) -> "SimpleTokenizer":
        """Создает простой токенизатор"""
        try:
            from transformers import GPT2Tokenizer
            
            # Проверяем наличие файлов токенизатора
            # Для ruGPT-3 Large во фрактальном хранилище
            tokenizer_path = self.storage_path
            
            # Возможные пути к vocab.json
            possible_vocab_paths = [
                os.path.join(tokenizer_path, "vocab.json"),  # В корневой директории (приоритет)
                os.path.join(tokenizer_path, "tokenizer", "vocab.json"),  # В поддиректории tokenizer
            ]
            
            # Дополнительные проверки для фрактального хранилища
            if "rugpt3_large_fractal" in tokenizer_path or "rugpt3large_fractal" in tokenizer_path:
                # Если это путь к токенизатору ruGPT-3 Large, используем абсолютные пути
                base_fractal_path = "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal/model"
                current_dir = os.getcwd()
                absolute_base_path = os.path.join(current_dir, base_fractal_path)
                
                possible_vocab_paths.extend([
                    os.path.join(absolute_base_path, "vocab.json"),
                    os.path.join(absolute_base_path, "tokenizer", "vocab.json"),
                    os.path.join(current_dir, base_fractal_path, "vocab.json"),
                    os.path.join(current_dir, base_fractal_path, "tokenizer", "vocab.json"),
                ])
            
            vocab_path = None
            for i, path in enumerate(possible_vocab_paths):
                logger.info(f"Проверяем путь {i+1}: {path}")
                logger.info(f"  Существует: {os.path.exists(path)}")
                if os.path.exists(path):
                    vocab_path = path
                    logger.info(f"  ✅ Найден vocab по пути: {path}")
                    break
            
            tokenizer_subdir_path = os.path.join(tokenizer_path, "vocab.json")
            
            if vocab_path:
                logger.info(f"Проверяем путь к токенизатору: {vocab_path}")
                logger.info(f"Абсолютный путь: {os.path.abspath(vocab_path)}")
                logger.info(f"Файл существует в корне: {os.path.exists(vocab_path)}")
            else:
                logger.warning("vocab_path не найден")
            
            logger.info(f"Файл существует в поддиректории tokenizer: {os.path.exists(tokenizer_subdir_path)}")
            
            # Пробуем загрузить из корневой директории (для rugpt3_large)
            if vocab_path and os.path.exists(vocab_path):
                # Определяем правильный путь для загрузки
                if "rugpt3_large_fractal" in tokenizer_path or "rugpt3large_fractal" in tokenizer_path:
                    # Для фрактального хранилища используем путь к директории модели
                    load_path = tokenizer_path  # Используем tokenizer_path, а не vocab_path
                else:
                    # Для других случаев используем tokenizer_path
                    load_path = tokenizer_path
                
                logger.info(f"✅ Найден локальный токенизатор, загружаем из: {load_path}")
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained(
                        load_path,
                        local_files_only=True
                    )
                    logger.info("✅ Токенизатор успешно загружен")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить токенизатор: {e}")
                    if DISABLE_HUGGINGFACE_FALLBACK:
                        logger.error("❌ HuggingFace fallback отключен. Локальный токенизатор обязателен.")
                        return None
                    else:
                        # Fallback на HuggingFace
                        logger.warning("Используем fallback токенизатор ruGPT-3 Medium из HuggingFace")
                        tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
            # Пробуем загрузить из поддиректории tokenizer (для других структур)
            elif tokenizer_subdir_path and os.path.exists(tokenizer_subdir_path):
                logger.info("✅ Найден локальный токенизатор в поддиректории tokenizer, загружаем...")
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained(
                        os.path.join(tokenizer_path, "tokenizer"),
                        local_files_only=True
                    )
                    logger.info("✅ Токенизатор успешно загружен из поддиректории tokenizer")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить локальный токенизатор: {e}")
                    if DISABLE_HUGGINGFACE_FALLBACK:
                        logger.error("❌ HuggingFace fallback отключен. Локальный токенизатор обязателен.")
                        return None
                    else:
                        logger.warning("Используем fallback токенизатор ruGPT-3 Medium из HuggingFace")
                        tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
            else:
                # Fallback на ruGPT-3 Medium токенизатор из HuggingFace
                if DISABLE_HUGGINGFACE_FALLBACK:
                    logger.error("❌ HuggingFace fallback отключен. Локальный токенизатор обязателен.")
                    return None
                else:
                    logger.warning(f"Локальный токенизатор не найден по путям:")
                    for path in possible_vocab_paths:
                        logger.warning(f"  - {path}")
                    logger.warning("Используем fallback токенизатор ruGPT-3 Medium из HuggingFace")
                    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
            
            # Устанавливаем специальные токены
            if tokenizer.pad_token is None:
                # Для ruGPT-3/GPT-2 лучше оставить pad_token как None
                # и не устанавливать его равным eos_token
                # Это предотвратит предупреждения об attention_mask
                pass
            elif tokenizer.pad_token == tokenizer.eos_token:
                # Если pad_token равен eos_token, отключаем его
                tokenizer.pad_token = None
                logger.info("pad_token отключен для предотвращения предупреждений")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания токенизатора: {e}")
            return None
    
    def create_model(self, device: str = "auto") -> Optional["GPT2LMHeadModel"]:
        """Создает и загружает модель с весами"""
        try:
            from transformers import GPT2LMHeadModel, GPT2Config
            
            # Определяем устройство
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Проверяем наличие локальных весов модели
            # Для ruGPT-3 Large во фрактальном хранилище
            model_root_path = self.storage_path
            
            # Возможные пути к pytorch_model.bin
            possible_model_paths = [
                os.path.join(model_root_path, "pytorch_model.bin"),  # В корневой директории
                os.path.join(model_root_path, "model", "pytorch_model.bin"),  # В поддиректории model
            ]
            
            # Дополнительные проверки для фрактального хранилища
            if "rugpt3_large_fractal" in model_root_path:
                # Если это путь к токенизатору ruGPT-3 Large, ищем веса в директории модели
                possible_model_paths.extend([
                    "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal/pytorch_model.bin",
                    "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal/model/pytorch_model.bin",
                ])
            
            pytorch_bin_root = None
            for path in possible_model_paths:
                if os.path.exists(path):
                    pytorch_bin_root = path
                    break
            
            # Аналогично для config.json
            possible_config_paths = [
                os.path.join(model_root_path, "config.json"),
                os.path.join(model_root_path, "model", "config.json"),
            ]
            
            if "rugpt3_large_fractal" in model_root_path:
                possible_config_paths.extend([
                    "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal/config.json",
                    "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal/model/config.json",
                ])
            
            model_config_root = None
            for path in possible_config_paths:
                if os.path.exists(path):
                    model_config_root = path
                    break
            
            # Альтернативный путь (поддиректория model)
            model_subdir_path = os.path.join(self.storage_path, "model")
            pytorch_bin_subdir = os.path.join(model_subdir_path, "pytorch_model.bin")
            model_config_subdir = os.path.join(model_subdir_path, "config.json")
            
            logger.info(f"Проверяем локальные веса модели:")
            logger.info(f"  Корневая директория: {pytorch_bin_root} (exists: {os.path.exists(pytorch_bin_root)})")
            logger.info(f"  Поддиректория model: {pytorch_bin_subdir} (exists: {os.path.exists(pytorch_bin_subdir)})")
            
            # Пробуем загрузить из корневой директории (для rugpt3_large)
            if pytorch_bin_root and model_config_root and os.path.exists(pytorch_bin_root) and os.path.exists(model_config_root):
                # Определяем правильный путь для загрузки
                if "rugpt3_large_fractal" in pytorch_bin_root:
                    # Для фрактального хранилища используем директорию модели
                    load_path = os.path.dirname(pytorch_bin_root)
                else:
                    # Для других случаев используем model_root_path
                    load_path = model_root_path
                
                logger.info(f"✅ Найдены локальные веса, загружаем модель из: {load_path}")
                model = GPT2LMHeadModel.from_pretrained(load_path, local_files_only=True)
                logger.info("✅ Модель успешно загружена из локальных весов")
            # Пробуем загрузить из поддиректории model
            elif os.path.exists(pytorch_bin_subdir) and os.path.exists(model_config_subdir):
                logger.info("✅ Найдены локальные веса в поддиректории model, загружаем модель...")
                model = GPT2LMHeadModel.from_pretrained(model_subdir_path, local_files_only=True)
                logger.info("✅ Модель успешно загружена из поддиректории model")
            else:
                # Загружаем из HuggingFace
                if DISABLE_HUGGINGFACE_FALLBACK:
                    logger.error("❌ HuggingFace fallback отключен. Локальная модель обязательна.")
                    return None
                else:
                    logger.warning("Локальные веса не найдены, загружаем из HuggingFace...")
                    logger.info("⏳ Загрузка ruGPT-3 Medium из sberbank-ai/rugpt3large_based_on_gpt2...")
                    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
                    logger.info("✅ Модель успешно загружена из HuggingFace")
            
            # Переносим на устройство
            model = model.to(device)
            model.eval()
            
            logger.info(f"✅ Модель готова на устройстве: {device}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания модели: {e}")
            return None
    
    def load_model_and_tokenizer(self, device: str = "auto") -> Tuple[Optional[Any], Optional[Any]]:
        """Загружает модель и токенизатор"""
        if not self.load_configs():
            return None, None
        
        tokenizer = self.create_tokenizer()
        model = self.create_model(device)
        
        return model, tokenizer

def load_rugpt3_medium_local(storage_path: str = None, 
                            device: str = "auto") -> Tuple[Optional[Any], Optional[Any]]:
    """Удобная функция для загрузки ruGPT-3 Medium/Large"""
    # Если путь не указан, используем путь к ruGPT-3 Large во фрактальном хранилище
    if storage_path is None:
        storage_path = "cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_large_fractal"
    
    loader = Localrugpt3largeLoader(storage_path)
    return loader.load_model_and_tokenizer(device)

if __name__ == "__main__":
    # Тест загрузки
    model, tokenizer = load_rugpt3_medium_local()
    if model and tokenizer:
        print("✅ ruGPT-3 Medium успешно загружен из фрактального хранилища")
        print(f"Модель: {type(model).__name__}")
        print(f"Токенизатор: {type(tokenizer).__name__}")
        print(f"Устройство: {model.device}")
    else:
        print("❌ Ошибка загрузки ruGPT-3 Medium")
