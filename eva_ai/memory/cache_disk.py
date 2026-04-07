"""Disk cache management, persistence, SSD storage."""
import os
import json
import time
import threading
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TokenDiskCache:
    """Дисковый кэш токенов с поддержкой больших объемов (до 50 ГБ)."""

    def __init__(self, cache_dir: str, max_size_gb: float = 50.0):
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self.index_file = os.path.join(cache_dir, "disk_cache_index.json")
        self.data_dir = os.path.join(cache_dir, "data")

        os.makedirs(self.data_dir, exist_ok=True)

        self.file_index = {}
        self.current_size_bytes = 0
        self._lock = threading.RLock()

        self._load_index()

        logger.info(f"TokenDiskCache инициализирован: {cache_dir}, лимит={max_size_gb}GB")

    def _load_index(self):
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.file_index = data.get('files', {})
                    self.current_size_bytes = data.get('total_size', 0)
                logger.debug(f"Загружен индекс дискового кэша: {len(self.file_index)} файлов")
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса дискового кэша: {e}")
            self.file_index = {}
            self.current_size_bytes = 0

    def _save_index(self):
        try:
            data = {
                'files': self.file_index,
                'total_size': self.current_size_bytes,
                'last_updated': time.time()
            }
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса дискового кэша: {e}")

    def _get_file_path(self, token_id: str) -> str:
        hash_prefix = token_id[:2]
        subdir = os.path.join(self.data_dir, hash_prefix)
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{token_id}.bin")

    def get(self, token_id: str) -> Optional[Dict]:
        with self._lock:
            if token_id not in self.file_index:
                return None

            file_path = self._get_file_path(token_id)
            if not os.path.exists(file_path):
                del self.file_index[token_id]
                self._save_index()
                return None

            try:
                with open(file_path, 'rb') as f:
                    data = f.read()

                if not data or len(data) < 2:
                    logger.error(f"Пустые или поврежденные данные для токена {token_id}")
                    self._remove_file(token_id)
                    return None

                import pickle
                import re

                SAFE_TOKEN_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
                if not SAFE_TOKEN_ID_PATTERN.match(token_id):
                    logger.error(f"Invalid token_id format: {token_id}")
                    self._remove_file(token_id)
                    return None

                if len(data) > 100 * 1024 * 1024:
                    logger.error(f"Token data too large: {len(data)} bytes")
                    self._remove_file(token_id)
                    return None

                try:
                    token_data = pickle.loads(data, fix_imports=True, encoding='bytes', errors='strict')
                    if not isinstance(token_data, dict):
                        logger.error(f"Invalid token data type: {type(token_data)}")
                        self._remove_file(token_id)
                        return None
                except (pickle.UnpicklingError, AttributeError, ValueError) as e:
                    logger.error(f"Ошибка десериализации токена {token_id}: {e}")
                    self._remove_file(token_id)
                    return None

                self.file_index[token_id]['last_access'] = time.time()
                self.file_index[token_id]['access_count'] = self.file_index[token_id].get('access_count', 0) + 1
                self._save_index()

                return token_data

            except Exception as e:
                logger.error(f"Ошибка загрузки токена {token_id} с диска: {e}")
                self._remove_file(token_id)
                return None

    def put(self, token_id: str, token_data: Dict) -> bool:
        with self._lock:
            try:
                import pickle
                data = pickle.dumps(token_data)
                data_size = len(data)

                if data_size > 100 * 1024 * 1024:
                    logger.warning(f"Токен {token_id} слишком большой: {data_size / 1024 / 1024:.2f} MB")
                    return False

                while (self.current_size_bytes + data_size > self.max_size_bytes and
                       len(self.file_index) > 0):
                    self._evict_lru()

                file_path = self._get_file_path(token_id)
                with open(file_path, 'wb') as f:
                    f.write(data)

                old_size = 0
                if token_id in self.file_index:
                    old_size = self.file_index[token_id].get('size', 0)

                self.file_index[token_id] = {
                    'size': data_size,
                    'created': time.time(),
                    'last_access': time.time(),
                    'access_count': 1
                }

                self.current_size_bytes += data_size - old_size
                self._save_index()

                return True

            except Exception as e:
                logger.error(f"Ошибка сохранения токена {token_id} на диск: {e}")
                return False

    def _evict_lru(self):
        if not self.file_index:
            return

        lru_token_id = min(self.file_index.keys(),
                          key=lambda k: self.file_index[k].get('last_access', 0))

        self._remove_file(lru_token_id)

    def _remove_file(self, token_id: str):
        if token_id not in self.file_index:
            return

        file_path = self._get_file_path(token_id)
        file_size = self.file_index[token_id].get('size', 0)

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Ошибка удаления файла {file_path}: {e}")

        del self.file_index[token_id]
        self.current_size_bytes -= file_size
        self._save_index()

    def remove(self, token_id: str) -> bool:
        with self._lock:
            if token_id not in self.file_index:
                return False

            self._remove_file(token_id)
            return True

    def clear(self):
        with self._lock:
            try:
                for subdir in os.listdir(self.data_dir):
                    subdir_path = os.path.join(self.data_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for file_name in os.listdir(subdir_path):
                            file_path = os.path.join(subdir_path, file_name)
                            os.remove(file_path)

                self.file_index.clear()
                self.current_size_bytes = 0
                self._save_index()

                logger.info("Дисковый кэш очищен")

            except Exception as e:
                logger.error(f"Ошибка очистки дискового кэша: {e}")

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                'total_files': len(self.file_index),
                'total_size_bytes': self.current_size_bytes,
                'total_size_mb': self.current_size_bytes / (1024 * 1024),
                'total_size_gb': self.current_size_bytes / (1024 * 1024 * 1024),
                'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                'usage_percent': (self.current_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes else 0.0
            }

    def __contains__(self, token_id: str) -> bool:
        with self._lock:
            return token_id in self.file_index

    def __len__(self) -> int:
        with self._lock:
            return len(self.file_index)
