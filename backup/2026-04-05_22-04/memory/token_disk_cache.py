"""
Простой дисковый кэш для токенов с JSON файлами
"""
import logging
import os
import json
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class TokenDiskCache:
    """Класс для простого управления дисковым кэшем токенов с JSON файлами"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_token(self, token_hash: str) -> Dict[str, Any]:
        """Загружает токен с диска"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{token_hash}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки токена с диска {token_hash}: {e}")
        return {}

    def save_token(self, token_hash: str, token_data: Dict[str, Any]) -> None:
        """Сохраняет токен на диск"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{token_hash}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(token_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения токена на диск {token_hash}: {e}")

    def delete_token(self, token_hash: str) -> None:
        """Удаляет токен с диска"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{token_hash}.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except Exception as e:
            logger.error(f"Ошибка удаления файла кэша {cache_file}: {e}")

    def clear_all(self) -> None:
        """Очищает все файлы кэша на диске"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Ошибка удаления файла {file_path}: {e}")
        except Exception as e:
            logger.error(f"Ошибка очистки дискового кэша: {e}")

    def delete_multiple(self, token_hashes: List[str]) -> None:
        """Удаляет несколько токенов с диска"""
        for token_hash in token_hashes:
            self.delete_token(token_hash)

    def has_token(self, token_hash: str) -> bool:
        """Проверяет наличие токена на диске"""
        cache_file = os.path.join(self.cache_dir, f"{token_hash}.json")
        return os.path.exists(cache_file)
