"""
Фрактальный токенизатор для ruGPT3 с интеграцией гибридного кеша
"""
import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Импорты для работы с токенизаторами
try:
    from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
    from tokenizers import Tokenizer as HFTokenizer
    import torch
except ImportError as e:
    logging.error(f"Необходимые библиотеки не установлены: {e}")
    raise

logger = logging.getLogger("fractal_tokenizer")

class FractalTokenizer:
    """
    Фрактальный токенизатор для ruGPT3 с поддержкой:
    - Фрактальной метаданных
    - Гибридного кеша
    - Оптимизированного хранения
    - Адаптивной токенизации
    """
    
    def __init__(
        self,
        base_tokenizer_path: str = "gpt2",
        cache_dir: str = "fractal_token_cache",
        fractal_levels: int = 4,
        block_size: int = 64,
        hybrid_cache_size: int = 10000
    ):
        self.base_tokenizer_path = base_tokenizer_path
        self.cache_dir = Path(cache_dir)
        self.fractal_levels = fractal_levels
        self.block_size = block_size
        self.hybrid_cache_size = hybrid_cache_size
        
        # Создаем директории
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Инициализируем базовый токенизатор
        self._initialize_base_tokenizer()
        
        # Инициализируем фрактальные метаданные
        self._initialize_fractal_metadata()
        
        # Инициализируем гибридный кеш
        self._initialize_hybrid_cache()
        
        logger.info(f"FractalTokenizer инициализирован: vocab_size={len(self.vocab)}")
    
    def _initialize_base_tokenizer(self):
        """Инициализирует базовый токенизатор GPT-2"""
        try:
            self.base_tokenizer = GPT2TokenizerFast.from_pretrained(self.base_tokenizer_path)
            
            # Добавляем специальные токены для фрактальной структуры
            fractal_special_tokens = [
                "<fractal_start>", "<fractal_end>", "<fractal_node>", "<fractal_edge>",
                "<fractal_level>", "<fractal_block>", "<fractal_compress>",
                "<fractal_expand>", "<fractal_memory>", "<fractal_cache>",
                "<fractal_reconstruct>", "<fractal_optimize>", "<fractal_stream>"
            ]
            
            # Добавляем русские токены
            russian_tokens = [
                "Привет", "Здравствуйте", "Спасибо", "Пожалуйста", "Да", "Нет",
                "Что", "Как", "Почему", "Где", "Когда", "Кто", "Чей",
                "Россия", "Москва", "Санкт-Петербург", "Русский", "Язык",
                "Искусственный", "Интеллект", "Машина", "Обучение", 
                "Нейронная", "Сеть", "Трансформер", "Модель", "Данные",
                "Алгоритм", "Программа", "Код", "Система", "Компьютер",
                "Процессор", "Память", "Хранилище", "Фрактал", "Кеш"
            ]
            
            # Добавляем все токены
            all_new_tokens = fractal_special_tokens + russian_tokens
            added_tokens = self.base_tokenizer.add_tokens(all_new_tokens)
            
            logger.info(f"Добавлено токенов: {added_tokens}")
            
            # Устанавливаем специальные токены
            self.base_tokenizer.pad_token = "<pad>"
            self.base_tokenizer.eos_token = ""
            self.base_tokenizer.bos_token = "<|startoftext|>"
            self.base_tokenizer.unk_token = "<unk>"
            
            # Сохраняем словарь
            self.vocab = self.base_tokenizer.get_vocab()
            self.vocab_size = len(self.vocab)
            
        except Exception as e:
            logger.error(f"Ошибка инициализации токенизатора: {e}")
            raise
    
    def _initialize_fractal_metadata(self):
        """Инициализирует фрактальные метаданные токенизатора"""
        self.fractal_metadata = {
            "tokenizer_type": "FractalTokenizer",
            "base_model": self.base_tokenizer_path,
            "vocab_size": self.vocab_size,
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
            "russian_tokens_count": 30,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        # Сохраняем метаданные
        metadata_file = self.metadata_dir / "fractal_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.fractal_metadata, f, indent=2, ensure_ascii=False)
    
    def _initialize_hybrid_cache(self):
        """Инициализирует гибридный кеш токенов"""
        try:
            # Импортируем HybridTokenCache
            from cogniflex.memory.hybrid_token_cache import HybridTokenCache
            
            # Создаем гибридный кеш
            self.hybrid_cache = HybridTokenCache(
                brain=None,  # Будет установлен позже
                max_memory_tokens=self.hybrid_cache_size,
                disk_cache_dir=str(self.cache_dir / "hybrid_cache"),
                target_memory_gb=2.0,  # 2GB для токенов
                dynamic_memory_limit=True,
                max_ram_usage_percent=75.0,
                vram_threshold=0.2,
                ram_threshold=0.15,
                eviction_policy="hybrid",
                vram_ratio=0.7,
                ram_cache_ratio=0.3
            )
            
            logger.info(f"Гибридный кеш инициализирован: {self.hybrid_cache_size} токенов")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации гибридного кеша: {e}")
            # Fallback на простой кеш
            self.hybrid_cache = {}
            logger.warning("Используется fallback кеш (dict)")
    
    def encode(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        use_fractal_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Кодирует текст с использованием фрактального кеша
        
        Args:
            text: Текст для кодирования
            return_tensors: Формат возвращаемых тензоров
            add_special_tokens: Добавлять ли специальные токены
            truncation: Обрезать ли длинные последовательности
            max_length: Максимальная длина
            use_fractal_cache: Использовать ли фрактальный кеш
            
        Returns:
            Словарь с закодированными токенами и метаданными
        """
        try:
            # Проверяем кеш
            cache_key = hashlib.md5(text.encode()).hexdigest()
            
            if use_fractal_cache and hasattr(self.hybrid_cache, 'get'):
                cached_result = self.hybrid_cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Токен из кеша: {cache_key[:8]}...")
                    return cached_result
            
            # Базовая токенизация
            encoded = self.base_tokenizer(
                text,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length
            )
            
            # Добавляем фрактальные метаданные
            result = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded.get("attention_mask"),
                "token_type_ids": encoded.get("token_type_ids"),
                "fractal_metadata": {
                    "cache_key": cache_key,
                    "text_length": len(text),
                    "token_count": encoded["input_ids"].shape[-1] if hasattr(encoded["input_ids"], 'shape') else len(encoded["input_ids"]),
                    "encoding_time": datetime.now().isoformat(),
                    "fractal_levels": self.fractal_levels,
                    "block_size": self.block_size
                }
            }
            
            # Сохраняем в кеш
            if use_fractal_cache and hasattr(self.hybrid_cache, 'put'):
                self.hybrid_cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка кодирования: {e}")
            # Fallback на базовую токенизацию
            return self.base_tokenizer(
                text,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length
            )
    
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        use_fractal_cache: bool = True
    ) -> str:
        """
        Декодирует токены с использованием фрактального кеша
        
        Args:
            token_ids: ID токенов для декодирования
            skip_special_tokens: Пропускать ли специальные токены
            clean_up_tokenization_spaces: Очищать ли пробелы
            use_fractal_cache: Использовать ли фрактальный кеш
            
        Returns:
            Декодированный текст
        """
        try:
            # Проверяем кеш
            if isinstance(token_ids, torch.Tensor):
                cache_key = hashlib.md5(token_ids.cpu().numpy().tobytes()).hexdigest()
            else:
                cache_key = hashlib.md5(str(token_ids).encode()).hexdigest()
            
            if use_fractal_cache and hasattr(self.hybrid_cache, 'get'):
                cached_result = self.hybrid_cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Декодирование из кеша: {cache_key[:8]}...")
                    return cached_result
            
            # Базовое декодирование
            decoded = self.base_tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            
            # Сохраняем в кеш
            if use_fractal_cache and hasattr(self.hybrid_cache, 'put'):
                self.hybrid_cache.put(cache_key, decoded)
            
            return decoded
            
        except Exception as e:
            logger.error(f"Ошибка декодирования: {e}")
            # Fallback на базовое декодирование
            return self.base_tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
    
    def get_fractal_stats(self) -> Dict[str, Any]:
        """Возвращает статистику фрактального токенизатора"""
        stats = {
            "vocab_size": self.vocab_size,
            "fractal_levels": self.fractal_levels,
            "block_size": self.block_size,
            "cache_size": self.hybrid_cache_size,
            "metadata": self.fractal_metadata
        }
        
        # Добавляем статистику кеша если доступна
        if hasattr(self.hybrid_cache, 'cache_stats'):
            stats["cache_stats"] = self.hybrid_cache.cache_stats
        
        return stats
    
    def save_pretrained(self, save_directory: str):
        """Сохраняет токенизатор с фрактальными метаданными"""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем базовый токенизатор
        self.base_tokenizer.save_pretrained(str(save_dir))
        
        # Сохраняем фрактальные метаданные
        fractal_file = save_dir / "fractal_tokenizer.json"
        with open(fractal_file, 'w', encoding='utf-8') as f:
            json.dump(self.fractal_metadata, f, indent=2, ensure_ascii=False)
        
        # Сохраняем статистику
        stats_file = save_dir / "fractal_stats.json"
        stats = self.get_fractal_stats()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Фрактальный токенизатор сохранен: {save_dir}")
    
    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs):
        """Загружает предобученный фрактальный токенизатор"""
        try:
            # Проверяем наличие фрактальных метаданных
            fractal_file = Path(tokenizer_path) / "fractal_tokenizer.json"
            
            if fractal_file.exists():
                with open(fractal_file, 'r', encoding='utf-8') as f:
                    fractal_metadata = json.load(f)
                
                logger.info(f"Загрузка фрактального токенизатора: {fractal_metadata.get('version', 'unknown')}")
                
                # Создаем экземпляр с параметрами из метаданных
                instance = cls(
                    base_tokenizer_path=fractal_metadata.get('base_model', 'gpt2'),
                    cache_dir=fractal_metadata.get('cache_dir', 'fractal_token_cache'),
                    fractal_levels=fractal_metadata.get('fractal_levels', 4),
                    block_size=fractal_metadata.get('block_size', 64),
                    hybrid_cache_size=kwargs.get('hybrid_cache_size', 10000)
                )
                
                instance.fractal_metadata = fractal_metadata
                return instance
            else:
                # Fallback на обычный токенизатор
                logger.warning("Фрактальные метаданные не найдены, используется базовый токенизатор")
                return cls(**kwargs)
                
        except Exception as e:
            logger.error(f"Ошибка загрузки фрактального токенизатора: {e}")
            raise

# Фабричные функции для создания токенизаторов
def create_fractal_tokenizer_for_rugpt3(
    model_path: str = "sberbank-ai/rugpt3large_based_on_gpt2",
    cache_dir: str = "fractal_rugpt3_cache",
    fractal_levels: int = 3,
    block_size: int = 32
) -> FractalTokenizer:
    """
    Создает фрактальный токенизатор для ruGPT3
    
    Args:
        model_path: Путь к модели ruGPT3
        cache_dir: Директория кеша
        fractal_levels: Уровни фрактала
        block_size: Размер блока
        
    Returns:
        FractalTokenizer адаптированный для ruGPT3
    """
    try:
        # Создаем базовый токенизатор ruGPT3
        tokenizer = FractalTokenizer(
            base_tokenizer_path=model_path,
            cache_dir=cache_dir,
            fractal_levels=fractal_levels,
            block_size=block_size,
            hybrid_cache_size=15000  # Увеличенный кеш для большой модели
        )
        
        logger.info(f"Фрактальный токенизатор для ruGPT3 создан")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Ошибка создания токенизатора для ruGPT3: {e}")
        raise

def test_fractal_tokenizer():
    """Тестирует фрактальный токенизатор"""
    try:
        print("🧪 ТЕСТИРОВАНИЕ ФРАКТАЛЬНОГО ТОКЕНИЗАТОРА")
        print("="*50)
        
        # Создаем токенизатор
        tokenizer = create_fractal_tokenizer_for_rugpt3()
        
        # Тестовые тексты
        test_texts = [
            "Привет, как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о России",
            "Объясни фрактальную структуру данных",
            "Как работает нейронная сеть?"
        ]
        
        print("📝 ТЕСТЫ ТОКЕНИЗАЦИИ:")
        for i, text in enumerate(test_texts, 1):
            try:
                result = tokenizer.encode(text, use_fractal_cache=True)
                
                print(f"\n{i}. Текст: '{text}'")
                print(f"   📊 Длина: {result['fractal_metadata']['text_length']}")
                print(f"   🔢 Токенов: {result['fractal_metadata']['token_count']}")
                print(f"   🔑 Ключ кеша: {result['fractal_metadata']['cache_key'][:8]}...")
                
                # Декодирование
                decoded = tokenizer.decode(result["input_ids"], use_fractal_cache=True)
                print(f"   💬 Декодировано: '{decoded}'")
                print(f"   ✅ Совпадение: {text == decoded}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        # Статистика
        stats = tokenizer.get_fractal_stats()
        print(f"\n📊 СТАТИСТИКА ТОКЕНИЗАТОРА:")
        print(f"   📚 Размер словаря: {stats['vocab_size']:,}")
        print(f"   🏗️ Уровней фрактала: {stats['fractal_levels']}")
        print(f"   📦 Размер блока: {stats['block_size']}")
        print(f"   💾 Размер кеша: {stats['cache_size']:,}")
        
        if 'cache_stats' in stats:
            cache_stats = stats['cache_stats']
            print(f"   🎯 Хитов кеша: {cache_stats.get('total_requests', 0) - cache_stats.get('misses', 0)}")
            print(f"   ❌ Промахов: {cache_stats.get('misses', 0)}")
        
        # Сохраняем токенизатор
        tokenizer.save_pretrained("test_fractal_tokenizer")
        print(f"\n✅ Токенизатор сохранен: test_fractal_tokenizer")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    test_fractal_tokenizer()
