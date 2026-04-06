"""
Модуль для управления метаданными токенов
"""
import logging
import time
import threading
import os
import json
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class MetadataManager:
    """Класс для управления метаданными токенов"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.token_metadata = {}
        self.metadata_lock = threading.Lock()
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")

        # Загружаем существующие метаданные при инициализации
        self._load_metadata()

    def update_token_metadata(self, token_hash: str, token_data: Dict[str, Any]) -> None:
        """Обновляет метаданные для токена"""
        with self.metadata_lock:
            self.token_metadata[token_hash] = {
                'timestamp': time.time(),
                'access_count': self.token_metadata.get(token_hash, {}).get('access_count', 0) + 1,
                'size': len(str(token_data))
            }

    def get_expired_tokens(self, max_age_seconds: int = 86400) -> List[str]:
        """Возвращает список устаревших токенов"""
        current_time = time.time()
        expired_tokens = []

        with self.metadata_lock:
            for token_hash, metadata in self.token_metadata.items():
                if current_time - metadata.get('timestamp', 0) > max_age_seconds:
                    expired_tokens.append(token_hash)

        return expired_tokens

    def remove_metadata(self, token_hashes: List[str]) -> None:
        """Удаляет метаданные для указанных токенов"""
        with self.metadata_lock:
            for token_hash in token_hashes:
                if token_hash in self.token_metadata:
                    del self.token_metadata[token_hash]

    def clear_all_metadata(self) -> None:
        """Очищает все метаданные"""
        with self.metadata_lock:
            self.token_metadata.clear()

    def save_metadata(self) -> None:
        """Сохраняет метаданные на диск"""
        try:
            with self.metadata_lock:
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.token_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных: {e}")

    def _load_metadata(self) -> None:
        """Загружает метаданные с диска"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.token_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
            self.token_metadata = {}

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику метаданных"""
        with self.metadata_lock:
            return {
                'total_tokens': len(self.token_metadata),
                'metadata_file': self.metadata_file
            }
