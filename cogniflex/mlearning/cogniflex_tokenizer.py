import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import logging
import threading
from dataclasses import dataclass

# Импорты для работы с токенизаторами
try:
    from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
    from tokenizers import Tokenizer as HFTokenizer
except ImportError:
    GPT2TokenizerFast = None
    PreTrainedTokenizerFast = None
    HFTokenizer = None

import torch

# Проверяем доступность CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizer = None
    PreTrainedTokenizerFast = None
    _TRANSFORMERS_AVAILABLE = False

try:
    from cogniflex.core import CoreBrain
except ImportError:
    class CoreBrain: pass

if TYPE_CHECKING:
    from cogniflex.mlearning.model_manager import ModelMetadata
else:
    class ModelMetadata: pass

logger = logging.getLogger(__name__)


class DummyTokenizer:
    """Резервный токенизатор, обеспечивающий минимальную функциональность для предотвращения сбоев."""

    def __init__(self, *args, **kwargs):
        logger.warning("Инициализация DummyTokenizer. Функциональность токенизатора ограничена.")
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.bos_token = "[BOS]"
        self.unk_token = "[UNK]"
        self.model_max_length = 512
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

    def tokenize(self, text: str, *args, **kwargs) -> List[str]:
        if not isinstance(text, str):
            return []
        return re.findall(r'\w+|[\s\.,!\?]', text)

    def encode(self, text: str, *args, **kwargs) -> List[int]:
        return [self.unk_token_id] * len(self.tokenize(text))

    def __call__(self, text: Union[str, List[str]], *args, **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(text, list):
            text = text[0] if text else ""
        
        tokens = self.encode(text)
        max_length = kwargs.get("max_length", self.model_max_length)
        padding = kwargs.get("padding", False)
        truncation = kwargs.get("truncation", False)

        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]

        attention_mask = [1] * len(tokens)

        if padding and len(tokens) < max_length:
            pad_len = max_length - len(tokens)
            tokens.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }


@dataclass
class TokenizationConfig:
    priority_strategy: str = "initial_response"
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "pt"
    language: str = "ru"


class CogniFlexTokenizer:
    """
    Deprecated: This tokenizer is deprecated and will be removed in a future version.
    Please migrate to ExtendedFractalTokenizer for fractal-aware tokenization.
    
    Migration guide:
    1. Replace `CogniFlexTokenizer.get_or_create()` with `ExtendedFractalTokenizer.from_pretrained()`
    2. Update your code to use the new fractal-aware tokenization methods:
       - Use `encode_fractal_path()` for fractal path encoding
       - Use `encode_metadata()` for metadata encoding
       - Use `prepare_fractal_input()` to combine text with fractal metadata
    """
    _instance_cache: Dict[str, "CogniFlexTokenizer"] = {}
    _global_lock = threading.Lock()

    def __init__(self, brain: Optional[CoreBrain] = None, model_metadata: Optional[ModelMetadata] = None):
        self.brain = brain
        self.model_metadata = model_metadata
        self.tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DummyTokenizer]] = None
        self._is_initialized = False
        self.model_path: Optional[str] = None
        self.load_time = 0.0
        self.config = TokenizationConfig()
        self._cpu_executor = ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 1) + 2))
        
        # Фрактальные компоненты
        self.fractal_levels = 4
        self.block_size = 64
        self.hybrid_cache_size = 10000
        self.fractal_metadata = {}
        self.hybrid_cache = None
        self.cache_dir = Path("./fractal_token_cache")
        self.metadata_dir = self.cache_dir / "metadata"

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized and self.tokenizer is not None

    def _create_fallback_tokenizer(self):
        """Создает и устанавливает DummyTokenizer в качестве резервного."""
        logger.warning("Создание резервного DummyTokenizer из-за сбоя загрузки основной модели.")
        self.tokenizer = DummyTokenizer()
        self._is_initialized = True

    def _ensure_initialized(self):
        """Гарантирует, что токенизатор инициализирован, создавая резервный при необходимости."""
        if not self.is_initialized:
            logger.warning("Токенизатор не был инициализирован. Попытка создать резервный.")
            self._create_fallback_tokenizer()

    @classmethod
    async def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "CogniFlexTokenizer":
        """Загружает токенизатор из указанного пути.
        
        Args:
            pretrained_model_name_or_path: Путь к директории с токенизатором или имя модели
            **kwargs: Дополнительные аргументы для загрузки токенизатора
            
        Returns:
            Экземпляр CogniFlexTokenizer
        """
        brain = kwargs.get("brain")
        model_metadata = kwargs.get("model_metadata")
        
        # Определяем путь к токенизатору
        model_path = pretrained_model_name_or_path
        
        # Если путь не абсолютный, пробуем найти в стандартных директориях
        if not os.path.isabs(model_path):
            # Пробуем найти в директории с моделями
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            possible_paths = [
                os.path.join(project_root, 'cogniflex', 'mlearning', 'cogniflex_models', 'fractal_unified_text-generation'),
                os.path.join(project_root, 'cogniflex_models', 'fractal_unified_text-generation'),
                os.path.join(project_root, 'models', 'fractal_unified_text-generation'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Найден токенизатор по пути: {model_path}")
                    break
        
        # Используем путь как ключ кеша
        cache_key = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
        
        with cls._global_lock:
            if cache_key in cls._instance_cache:
                logger.debug(f"Используем закешированный токенизатор для {cache_key}")
                return cls._instance_cache[cache_key]

            logger.info(f"Инициализация нового токенизатора из: {model_path}")
            instance = cls(brain, model_metadata)
            instance.model_path = model_path
            
            try:
                await instance._load_tokenizer_async(**kwargs)
                if not instance.is_initialized():
                    raise RuntimeError("Не удалось инициализировать токенизатор")
                
                cls._instance_cache[cache_key] = instance
                logger.info(f"Токенизатор успешно загружен и закеширован с ключом: {cache_key}")
                return instance
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке токенизатора: {e}", exc_info=True)
                # В случае ошибки создаем и возвращаем токенизатор-заглушку
                instance._create_fallback_tokenizer()
                return instance

    async def _load_tokenizer_async(self, **kwargs):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self._cpu_executor,
                self._load_inner_tokenizer, 
                self.model_path, 
                kwargs
            )
        except Exception as e:
            logger.critical(f"Критический сбой при асинхронной загрузке токенизатора: {e}", exc_info=True)
            self._create_fallback_tokenizer()

    def _load_inner_tokenizer(self, path: str, kwargs: Dict[str, Any] = None) -> None:
        """Загружает внутренний токенизатор из указанного пути.
        
        Args:
            path: Путь к директории с моделью токенизатора
            kwargs: Дополнительные аргументы для загрузки токенизатора
        """
        if kwargs is None:
            kwargs = {}
            
        start_time = time.time()
        
        try:
            # Проверяем существование пути и необходимых файлов
            if not os.path.exists(path):
                # Пробуем найти путь относительно корня проекта
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                alt_paths = [
                    os.path.join(project_root, 'cogniflex', 'mlearning', 'tokenizers', 'fractal_unified_tokenizer'),
                    os.path.join(project_root, 'cogniflex', 'mlearning', 'cogniflex_models', 'fractal_unified_text-generation')
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        path = alt_path
                        logger.info(f"Используем альтернативный путь к токенизатору: {path}")
                        break
                else:
                    raise FileNotFoundError(f"Tokenizer path does not exist: {path} and {', '.join(alt_paths)}")
            
            # Проверяем наличие необходимых файлов
            required_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
            missing_files = [f for f in required_files if not os.path.isfile(os.path.join(path, f))]
            
            if missing_files:
                raise FileNotFoundError(f"Missing required tokenizer files: {', '.join(missing_files)} in {path}")
            
            logger.info(f"Загрузка токенизатора из: {path}")
            
            # Загружаем локальный токенизатор без попыток загрузки из интернета
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
                **kwargs
            )
            
            if self.tokenizer is None:
                raise ValueError("Failed to initialize tokenizer - tokenizer is None")
                
            self._setup_special_tokens()
            self._is_initialized = True
            self.load_time = time.time() - start_time
            logger.info(f"Токенизатор успешно загружен из локальных файлов за {self.load_time:.2f} сек.")
            
            # Инициализируем фрактальные компоненты
            self._initialize_fractal_components()
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке токенизатора: {e}", exc_info=True)
            self._create_fallback_tokenizer()

    def _initialize_fractal_components(self):
        """Инициализирует фрактальные компоненты токенизатора"""
        try:
            # Создаем директории
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)
            
            # Инициализируем фрактальные метаданные
            self._initialize_fractal_metadata()
            
            # Инициализируем гибридный кэш
            self._initialize_hybrid_cache()
            
            # Добавляем фрактальные специальные токены
            self._add_fractal_special_tokens()
            
            logger.info("Фрактальные компоненты токенизатора инициализированы")
            
        except Exception as e:
            logger.warning(f"Не удалось инициализировать фрактальные компоненты: {e}")
    
    def _add_fractal_special_tokens(self):
        """Добавляет специальные токены для фрактальной структуры"""
        if not self.is_initialized or isinstance(self.tokenizer, DummyTokenizer):
            return
            
        try:
            # Фрактальные специальные токены
            fractal_special_tokens = [
                "<fractal_start>", "<fractal_end>", "<fractal_node>", "<fractal_edge>",
                "<fractal_level>", "<fractal_block>", "<fractal_compress>",
                "<fractal_expand>", "<fractal_memory>", "<fractal_cache>",
                "<fractal_reconstruct>", "<fractal_optimize>", "<fractal_stream>"
            ]
            
            # Добавляем токены как специальные
            special_tokens_dict = {
                "additional_special_tokens": fractal_special_tokens
            }
            added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Добавлено {added_tokens} фрактальных специальных токенов")
            
        except Exception as e:
            logger.warning(f"Не удалось добавить фрактальные специальные токены: {e}")
    
    def _initialize_fractal_metadata(self):
        """Инициализирует фрактальные метаданные токенизатора"""
        self.fractal_metadata = {
            "tokenizer_type": "CogniFlexTokenizer",
            "base_model": getattr(self.tokenizer, 'name_or_path', 'unknown') if self.tokenizer else 'unknown',
            "vocab_size": len(self.tokenizer) if self.tokenizer and hasattr(self.tokenizer, '__len__') else 0,
            "fractal_levels": self.fractal_levels,
            "block_size": self.block_size,
            "special_tokens": {
                "fractal_start": "<fractal_start>",
                "fractal_end": "<fractal_end>",
                "fractal_node": "<fractal_node>",
                "fractal_edge": "<fractal_edge>",
                "fractal_level": "<fractal_level>",
                "fractal_block": "<fractal_block>",
                "fractal_compress": "<fractal_compress>",
                "fractal_expand": "<fractal_expand>",
                "fractal_memory": "<fractal_memory>",
                "fractal_cache": "<fractal_cache>"
            },
            "created_at": datetime.now().isoformat(),
            "version": "2.0.0"
        }
        
        # Сохраняем метаданные
        try:
            metadata_file = self.metadata_dir / "fractal_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.fractal_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Не удалось сохранить фрактальные метаданные: {e}")
    
    def _initialize_hybrid_cache(self):
        """Инициализирует гибридный кэш токенов"""
        try:
            # Импортируем HybridTokenCache
            from cogniflex.memory.hybrid_token_cache import HybridTokenCache
            
            # Создаем гибридный кэш
            self.hybrid_cache = HybridTokenCache(
                brain=self.brain,
                max_memory_tokens=self.hybrid_cache_size,
                disk_cache_dir=str(self.cache_dir / "hybrid_cache"),
                target_memory_gb=2.0,  # 2GB для токенов
                dynamic_memory_limit=True,
                max_ram_usage_percent=75.0,
                vram_threshold=0.2,
            )
            
            logger.info("Гибридный кэш токенов инициализирован")
            
        except Exception as e:
            logger.warning(f"Не удалось инициализировать гибридный кэш: {e}")
            self.hybrid_cache = None

    def _setup_special_tokens(self):
        """Настраивает специальные токены, особенно pad_token."""
        if isinstance(self.tokenizer, DummyTokenizer) or not self.tokenizer:
            return

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Установлен pad_token, равный eos_token: '{self.tokenizer.eos_token}'")
            else:
                new_pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': new_pad_token})
                logger.warning(f"Установлен новый pad_token: '{new_pad_token}'")

    def tokenize(self, text: str, **kwargs) -> List[str]:
        self._ensure_initialized()
        return self.tokenizer.tokenize(text, **kwargs)

    def encode(self, text: str, **kwargs) -> List[int]:
        self._ensure_initialized()
        return self.tokenizer.encode(text, **kwargs)

    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        self._ensure_initialized()
        # Убедимся, что padding и truncation передаются корректно
        final_kwargs = {
            'return_tensors': self.config.return_tensors,
            'padding': self.config.padding,
            'truncation': self.config.truncation,
            'max_length': self.config.max_length,
            **kwargs
        }
        return self.tokenizer(text, **final_kwargs)


    # ---------- Публичные методы токенизации ----------
    async def tokenize_async(self, text: str, config: Optional[TokenizationConfig] = None) -> List[str]:
        """
        Асинхронно токенизирует входной текст с учетом конфигурации.
        
        Args:
            text: Входной текст для токенизации
            config: Конфигурация токенизации (опционально)
            
        Returns:
            Список токенов
        """
        if not self.is_initialized or self.tokenizer is None:
            logger.error("Токенизатор не инициализирован")
            return text.split()
            
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если событийный цикл уже запущен, запускаем асинхронно
            return loop.run_until_complete(self.tokenize_async(text, config))
        else:
            # Иначе запускаем новый цикл
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.tokenize_async(text, config))
                )
                return future.result()

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        **kwargs,
    ) -> List[str]:
        """Токенизирует текст, используя внутренний токенизатор."""
        if not self.is_initialized or self.tokenizer is None:
            logger.error("Токенизатор не инициализирован")
            return text.split()
            
        try:
            return self.tokenizer.tokenize(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Ошибка во время токенизации: {e}", exc_info=True)
            return text.split()

    def encode(self, text: str, config: Optional[TokenizationConfig] = None) -> Dict[str, torch.Tensor]:
        if not self.is_initialized or self.tokenizer is None:
            logger.error("Токенизатор не инициализирован")
            return {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]])}
        cfg = config or self.config
        strat = cfg.get_strategy_params()
        try:
            processed = self._preprocess_text(text)
            enc = self.tokenizer(
                processed,
                max_length=int(strat.get("max_length", cfg.max_length)),
                padding=bool(cfg.padding),
                truncation=bool(cfg.truncation),
                return_tensors=str(cfg.return_tensors),
            )
            # Возвращаем обычный dict с тензорами, а не BatchEncoding
            try:
                input_ids = enc["input_ids"]
                attention_mask = enc.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}
            except Exception:
                # На всякий случай приведём к dict через items
                d = {k: v for k, v in getattr(enc, "items", lambda: [])()}
                if "input_ids" not in d:
                    d["input_ids"] = torch.tensor([[0]])
                if "attention_mask" not in d:
                    d["attention_mask"] = torch.ones_like(d["input_ids"])
                return d
        except Exception as e:
            logger.error(f"Ошибка кодирования: {e}", exc_info=True)
            # Возвращаем согласованные тензоры минимального размера
            return {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]])}

    def decode(self, token_ids: Union[List[int], torch.Tensor], config: Optional[TokenizationConfig] = None) -> str:
        if not self.is_initialized or self.tokenizer is None:
            logger.error("Токенизатор не инициализирован")
            return ""
        try:
            if isinstance(token_ids, torch.Tensor):
                ids = token_ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            else:
                ids = token_ids
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            return self._postprocess_decoded_text(text, config)
        except Exception as e:
            logger.error(f"Ошибка преобразования токенов в ID: {e}", exc_info=True)
            return ""

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if not self.is_initialized or self.tokenizer is None:
            logger.error("Токенизатор не инициализирован")
            return "" if isinstance(ids, int) else [""]
        try:
            return self.tokenizer.convert_ids_to_tokens(ids)
        except Exception as e:
            logger.error(f"Ошибка преобразования ID в токены: {e}", exc_info=True)
            return "" if isinstance(ids, int) else [""]

    def add_special_tokens(self, special_tokens_dict: Dict[str, Any]) -> None:
        if not self.is_initialized or self.tokenizer is None:
            logger.error("Токенизатор не инициализирован")
            return
        try:
            self.tokenizer.add_special_tokens(special_tokens_dict)
        except Exception as e:
            logger.error(f"Ошибка добавления специальных токенов: {e}", exc_info=True)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]) -> None:
        if not self.is_initialized or self.tokenizer is None:
            logger.error("Токенизатор не инициализирован")
            return
        try:
            self.tokenizer.save_pretrained(save_directory)
        except Exception as e:
            logger.error(f"Ошибка сохранения токенизатора: {e}", exc_info=True)

    # ---------- Внутренние вспомогательные ----------
    def _preprocess_text(self, text: str) -> str:
        processed = text
        if self.token_streamer:
            try:
                processed = self.token_streamer.preprocess_text(text)
            except Exception as e:
                logger.debug(f"Ошибка препроцессинга текста: {e}")
        return processed

    def _postprocess_tokens(self, tokens: List[str], config: TokenizationConfig) -> List[str]:
        # Минимальная безопасная постобработка
        out: List[str] = []
        stops = set(self.stop_words) if config.dynamic_weights else set()
        for t in tokens:
            if t.startswith("▁") or t.startswith("Ġ"):
                t = t[1:]
            t = t.strip().lower()
            if not t or t in stops:
                continue
            out.append(t)
        return out

    def _postprocess_decoded_text(self, text: str, config: TokenizationConfig) -> str:
        # Удаляем известные спец-токены из текста и приводим пробелы
        for v in self.special_tokens.values():
            if isinstance(v, list):
                for sv in v:
                    text = text.replace(sv, "")
            else:
                text = text.replace(str(v), "")
        return re.sub(r"\s+", " ", text).strip()
