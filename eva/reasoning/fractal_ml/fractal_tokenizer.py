"""
Фрактальный токенизатор - собственная токенизация для хранилища
Отдельная от Qwen для независимой работы
"""

import re
import logging
from typing import List, Dict, Set, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class FractalTokenizer:
    """
    Собственный токенизатор для фрактального хранилища
    Использует character-level + word-level токенизацию
    """
    
    # Специальные токены
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._is_trained = False
        
        # Инициализация специальных токенов
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Инициализация специальных токенов"""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.SEP_TOKEN]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        if not self._is_trained:
            # Простая токенизация если словарь не обучен
            return self._simple_tokenize(text)
        
        # Токенизация по словарю
        tokens = []
        for word in self._simple_tokenize(text):
            if word in self.vocab:
                tokens.append(word)
            else:
                # Разбиваем на символы для неизвестных слов
                tokens.extend(list(word))
                tokens.append(self.UNK_TOKEN)
        
        return tokens
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Простая токенизация (word-level)"""
        # Очистка текста
        text = text.lower().strip()
        
        # Разбиение по пробелам и пунктуации
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        tokens = text.split()
        
        return [t for t in tokens if t.strip()]
    
    def tokenize_to_ids(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Токенизация с преобразованием в ID"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        
        # Преобразование в ID
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab[self.UNK_TOKEN]))
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Де-кодирование ID обратно в текст"""
        tokens = [self.id_to_token.get(i, self.UNK_TOKEN) for i in ids]
        
        # Удаление специальных токенов
        special = {self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.SEP_TOKEN}
        tokens = [t for t in tokens if t not in special]
        
        return ' '.join(tokens)
    
    def train(self, texts: List[str]):
        """Обучение словаря на текстах"""
        logger.info(f"Обучение токенизатора на {len(texts)} текстах...")
        
        # Подсчёт частот
        word_counts = Counter()
        for text in texts:
            tokens = self._simple_tokenize(text)
            word_counts.update(tokens)
        
        # Фильтрация по частоте и лимиту
        filtered = {w: c for w, c in word_counts.items() 
                   if c >= self.min_freq and w not in self.vocab}
        
        # Сортировка по частоте
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.vocab_size - len(self.vocab)]
        
        # Добавление в словарь
        next_id = len(self.vocab)
        for word, _ in sorted_words:
            self.vocab[word] = next_id
            self.id_to_token[next_id] = word
            next_id += 1
        
        self._is_trained = True
        logger.info(f"Словарь обучен: {len(self.vocab)} токенов")
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab.get(self.PAD_TOKEN, 0)
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab.get(self.UNK_TOKEN, 0)
    
    @property
    def bos_token_id(self) -> int:
        return self.vocab.get(self.BOS_TOKEN, 0)
    
    @property
    def eos_token_id(self) -> int:
        return self.vocab.get(self.EOS_TOKEN, 0)


class FractalTokenizerWrapper:
    """
    Обёртка для использования с sentence-transformers
    Адаптирует фрактальный токенизатор под интерфейс HuggingFace
    """
    
    def __init__(self, tokenizer: FractalTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, text: str, **kwargs):
        """Вызов как функция"""
        return self.tokenize(text, **kwargs)
    
    def tokenize(self, text: str, add_special_tokens: bool = True):
        """Токенизация"""
        return self.tokenizer.tokenize_to_ids(text, add_special_tokens)
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True):
        """Де-кодирование"""
        return self.tokenizer.decode(ids)
    
    def save_pretrained(self, path: str):
        """Сохранение"""
        import json
        data = {
            "vocab": self.tokenizer.vocab,
            "id_to_token": self.tokenizer.id_to_token,
            "vocab_size": self.tokenizer.vocab_size,
            "min_freq": self.tokenizer.min_freq
        }
        with open(f"{path}/fractal_tokenizer.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def from_pretrained(self, path: str):
        """Загрузка"""
        import json
        with open(f"{path}/fractal_tokenizer.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.tokenizer.vocab = data["vocab"]
        self.tokenizer.id_to_token = data["id_to_token"]
        self.tokenizer.vocab_size = data["vocab_size"]
        self.tokenizer.min_freq = data["min_freq"]
        self.tokenizer._is_trained = True
        return self
