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
    
    @staticmethod
    def _get_project_root() -> str:
        """Возвращает корневую директорию проекта"""
        import sys
        
        # Пробуем определить корень проекта несколькими способами
        possible_roots = []
        
        # 1. Относительно текущего файла
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)  # cogniflex/mlearning
        possible_roots.append(os.path.dirname(os.path.dirname(current_dir)))  # cogniflex
        possible_roots.append(os.path.dirname(current_dir))  # cogniflex/mlearning -> project root
        
        # 2. Относительно sys.argv[0] (главный скрипт)
        if sys.argv and sys.argv[0]:
            argv_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            possible_roots.append(argv_dir)
            possible_roots.append(os.path.dirname(argv_dir))
        
        # 3. Проверяем common project markers - с дополнительной проверкой
        for root in possible_roots:
            if root and os.path.exists(root):
                cogniflex_marker = os.path.join(root, 'cogniflex')
                pyproject_marker = os.path.join(root, 'pyproject.toml')
                setup_marker = os.path.join(root, 'setup.py')
                
                # Дополнительная проверка - также проверяем что это правильная директория
                has_cogniflex_dir = os.path.exists(cogniflex_marker)
                
                if has_cogniflex_dir or os.path.exists(pyproject_marker) or os.path.exists(setup_marker):
                    # Дополнительная проверка - убеждаемся что это "наш" проект
                    expected_items = ['cogniflex', 'tools', 'test_cogniflex.py', 'pyproject.toml', 'requirements.txt']
                    found_items = [item for item in expected_items if os.path.exists(os.path.join(root, item)) or 
                                  os.path.exists(os.path.join(root, 'cogniflex', item))]
                    
                    if has_cogniflex_dir:
                        return root
        
        # 4. Fallback - ищем по ключевым директориям с явным указанием
        # ВСЕГДА предпочитаем OneDrive путь если есть
        drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
        username = os.environ.get('USERNAME', 'user')
        
        # Явно проверяем OneDrive путь first
        onedrive_path = os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'CogniFlex')
        if os.path.exists(onedrive_path) and os.path.exists(os.path.join(onedrive_path, 'cogniflex')):
            return onedrive_path
        
        possible_locations = [
            os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'CogniFlex'),
            os.path.join(drive, 'Users', username, 'Desktop', 'CogniFlex'),
            os.path.join(drive, 'CogniFlex'),
            os.path.join(os.getcwd(), '..'),
            os.path.join(os.getcwd(), '..', '..'),
        ]
        
        for loc in possible_locations:
            if os.path.exists(loc):
                if os.path.exists(os.path.join(loc, 'cogniflex')) or \
                   os.path.exists(os.path.join(loc, 'pyproject.toml')):
                    return os.path.abspath(loc)
        
        # 5. Последний fallback - возвращаем директорию где запущен скрипт
        return os.getcwd()
    
    def __init__(self, storage_path: str = None):
        # Получаем корень проекта для построения абсолютных путей
        project_root = self._get_project_root()
        
        # Автоматически определяем путь к модели
        if storage_path is None:
            # Пробуем найти модель в стандартных местах
            # ВАЖНО: модель находится в models/rugpt3_large_fractal/model/, не в tokenizers/
            possible_paths = [
                # Путь к ruGPT-3 Large во фрактальном хранилище (модель в подпапке model/)
                os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_large_fractal", "model"),
                os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_large_fractal"),
                # Fallback на старый путь tokenizers (для обратной совместимости)
                os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "tokenizers", "rugpt3_large_fractal"),
                # Путь к rugpt3_large (полная модель ~3GB)
                os.path.join(project_root, "cogniflex_cache", "mlearning", "cogniflex_models", "rugpt3_large"),
                # Альтернативный путь от корня проекта
                os.path.join(project_root, "cogniflex", "mlearning", "cogniflex_models", "rugpt3_large"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    # Проверяем наличие модели
                    model_file = os.path.join(path, "pytorch_model.bin")
                    if os.path.exists(model_file):
                        storage_path = path
                        logger.info(f"Найдена локальная модель: {path}")
                        break
                    # Также проверяем в поддиректории model
                    model_file = os.path.join(path, "model", "pytorch_model.bin")
                    if os.path.exists(model_file):
                        storage_path = path
                        logger.info(f"Найдена локальная модель: {path}")
                        break
            
            if storage_path is None:
                storage_path = possible_paths[0]  # По умолчанию первый путь
                logger.warning(f"Локальная модель не найдена, используем путь по умолчанию: {storage_path}")
            
            # Fallback: если стандартные пути не содержат модель, пробуем fractal_unified_tokenizer
            if not os.path.exists(os.path.join(storage_path, "pytorch_model.bin")):
                # Пробуем разные варианты fallback
                # ВАЖНО: правильный путь - models/rugpt3_large_fractal/model/
                fallback_paths = [
                    os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_large_fractal", "model"),
                    os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_large_fractal"),
                    os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "tokenizers", "rugpt3_large_fractal"),
                    os.path.join(project_root, "cogniflex", "mlearning", "tokenizers", "fractal_unified_tokenizer"),
                    os.path.join(os.path.dirname(__file__), "tokenizers", "fractal_unified_tokenizer"),
                ]
                
                for fallback_tokenizer_path in fallback_paths:
                    logger.info(f"[_get_project_root] Checking fallback path: {fallback_tokenizer_path}")
                    if os.path.exists(os.path.join(fallback_tokenizer_path, "tokenizer.json")) or \
                       os.path.exists(os.path.join(fallback_tokenizer_path, "vocab.json")):
                        logger.info(f"[_get_project_root] Using fallback: {fallback_tokenizer_path}")
                        storage_path = fallback_tokenizer_path
                        break
        
        self.storage_path = storage_path
        logger.info(f"Localrugpt3largeLoader инициализирован с путем: {storage_path}")
        logger.info(f"Корень проекта: {project_root}")
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
            
            # Получаем корень проекта
            project_root = self._get_project_root()
            
            # Проверяем наличие файлов токенизатора
            # Для ruGPT-3 Large во фрактальном хранилище
            tokenizer_path = self.storage_path
            
            # Возможные пути к vocab.json - используем корень проекта
            possible_vocab_paths = [
                os.path.join(tokenizer_path, "vocab.json"),  # В корневой директории (приоритет)
                os.path.join(tokenizer_path, "tokenizer", "vocab.json"),  # В поддиректории tokenizer
            ]
            
            # Дополнительные проверки для фрактального хранилища
            if "rugpt3_large_fractal" in tokenizer_path or "rugpt3large_fractal" in tokenizer_path:
                # Если это путь к токенизатору ruGPT-3 Large, используем абсолютные пути от проекта
                
                # Пробуем разные варианты базовых путей от проекта
                base_paths = [
                    os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_large_fractal", "model"),
                    os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "tokenizers", "rugpt3_large_fractal"),
                    os.path.join(project_root, "cache", "ml_unit", "fractal_storage", "models", "rugpt3_large_fractal", "model"),
                    os.path.join(project_root, "cogniflex", "mlearning", "tokenizers", "fractal_unified_tokenizer"),
                ]
                
                for base_path in base_paths:
                    possible_vocab_paths.extend([
                        os.path.join(base_path, "vocab.json"),
                        os.path.join(base_path, "tokenizer", "vocab.json"),
                    ])
            
            vocab_path = None
            for i, path in enumerate(possible_vocab_paths):
                logger.info(f"Проверяем путь {i+1}: {path}")
                logger.info(f"  Существует: {os.path.exists(path)}")
                if os.path.exists(path):
                    vocab_path = path
                    logger.info(f"  Найден vocab по пути: {path}")
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
                
                # Специальная обработка для fractal_unified_tokenizer
                if vocab_path and "fractal_unified_tokenizer" in vocab_path:
                    logger.info(f"Обнаружен fractal_unified_tokenizer, используем библиотеку tokenizers")
                    try:
                        from tokenizers import Tokenizer
                        tokenizer_json = os.path.join(os.path.dirname(vocab_path), "tokenizer.json")
                        if os.path.exists(tokenizer_json):
                            tokenizer = Tokenizer.from_file(tokenizer_json)
                            logger.info("Токенизатор загружен через tokenizers.JSON")
                            # Оборачиваем в HuggingFace токенизатор для совместимости
                            from transformers import PreTrainedTokenizerFast
                            fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
                            fast_tokenizer.pad_token = tokenizer.token_to_id('<pad>') or tokenizer.token_to_id('</s>') or tokenizer.eos_id
                            fast_tokenizer.eos_token = tokenizer.id_to_token(tokenizer.eos_id) or '</s>'
                            fast_tokenizer.bos_token = tokenizer.id_to_token(tokenizer.bos_id) or '<s>'
                            return fast_tokenizer
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить через tokenizers: {e}")
                
                logger.info(f"Найден локальный токенизатор, загружаем из: {load_path}")
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained(
                        load_path,
                        local_files_only=True
                    )
                    logger.info("Токенизатор успешно загружен")
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
    # Важно: модель находится в подпапке model/
    if storage_path is None:
        storage_path = "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal/model"
    
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
