import os
import sqlite3
import json
from typing import Dict, List, Optional
import torch
import numpy as np
import mmap
import struct
from pathlib import Path

class HybridCache:
    def __init__(self, cache_dir: str = "./cache", max_size_gb: int = 50):
        """
        Инициализация гибридного кеша
        :param cache_dir: Директория для хранения кеша
        :param max_size_gb: Максимальный размер кеша в ГБ
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_gb * 1024 ** 3
        self.current_size = 0
        self.cache_index = {}
        
        # Создаем директорию, если она не существует
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем базу данных для индекса
        self._init_db()
        
    def _init_db(self):
        """Инициализация базы данных SQLite для хранения индекса кеша"""
        self.db_path = self.cache_dir / "cache_index.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        
        # Создаем таблицу, если она не существует
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_index (
            key TEXT PRIMARY KEY,
            start_pos INTEGER,
            length INTEGER,
            dtype TEXT,
            shape TEXT,
            timestamp REAL
        )
        """)
        self.conn.commit()
        
        # Загружаем существующий индекс в память
        self._load_index()
    
    def _load_index(self):
        """Загрузка индекса из базы данных в память"""
        self.cursor.execute("SELECT key, start_pos, length, dtype, shape FROM cache_index")
        for key, start_pos, length, dtype, shape in self.cursor.fetchall():
            self.cache_index[key] = {
                'start_pos': start_pos,
                'length': length,
                'dtype': dtype,
                'shape': tuple(map(int, shape.split(','))) if shape else None
            }
            self.current_size += length
    
    def _get_data_file(self):
        """Получение файла данных для записи"""
        data_file = self.cache_dir / "cache_data.bin"
        if not data_file.exists():
            data_file.touch()
        return data_file
    
    def _get_dtype_size(self, dtype: str) -> int:
        """Получение размера типа данных в байтах"""
        if dtype == 'float32':
            return 4
        elif dtype == 'int64':
            return 8
        elif dtype == 'int32':
            return 4
        elif dtype == 'int16':
            return 2
        elif dtype == 'int8':
            return 1
        elif dtype == 'bool':
            return 1
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    
    def store(self, key: str, tensor: torch.Tensor):
        """
        Сохранение тензора в кеш
        :param key: Ключ для доступа к данным
        :param tensor: Тензор для сохранения
        """
        # Проверяем, есть ли уже такой ключ в кеше
        if key in self.cache_index:
            self._remove(key)
        
        # Преобразуем тензор в numpy массив
        if tensor.is_cuda:
            tensor = tensor.cpu()
        np_array = tensor.numpy()
        
        # Получаем метаданные
        dtype = str(np_array.dtype)
        shape = np_array.shape
        
        # Рассчитываем размер данных
        element_size = self._get_dtype_size(dtype)
        num_elements = np.prod(shape)
        data_size = element_size * num_elements
        
        # Проверяем, не превысим ли мы лимит кеша
        self._ensure_space(data_size)
        
        # Открываем файл данных для записи
        with open(self._get_data_file(), 'r+b') as f:
            # Перемещаем указатель в конец файла
            f.seek(0, 2)
            start_pos = f.tell()
            
            # Записываем данные
            np_array.tofile(f)
            
            # Обновляем индекс
            self.cache_index[key] = {
                'start_pos': start_pos,
                'length': data_size,
                'dtype': dtype,
                'shape': shape
            }
            
            # Обновляем размер кеша
            self.current_size += data_size
            
            # Обновляем базу данных
            self.cursor.execute(
                "INSERT OR REPLACE INTO cache_index VALUES (?, ?, ?, ?, ?, ?)",
                (key, start_pos, data_size, dtype, ','.join(map(str, shape)) if shape else '', 0.0)
            )
            self.conn.commit()
    
    def retrieve(self, key: str) -> Optional[torch.Tensor]:
        """
        Получение тензора из кеша
        :param key: Ключ для доступа к данным
        :return: Тензор или None, если данные не найдены
        """
        if key not in self.cache_index:
            return None
            
        meta = self.cache_index[key]
        
        try:
            with open(self._get_data_file(), 'rb') as f:
                # Используем mmap для эффективного чтения
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Читаем данные
                    mm.seek(meta['start_pos'])
                    buffer = mm.read(meta['length'])
                    
                    # Преобразуем данные в numpy массив
                    np_array = np.frombuffer(buffer, dtype=meta['dtype'])
                    if meta['shape']:
                        np_array = np_array.reshape(meta['shape'])
                    
                    # Преобразуем в тензор
                    return torch.from_numpy(np_array)
        except Exception as e:
            print(f"Error retrieving {key} from cache: {e}")
            self._remove(key)
            return None
    
    def _ensure_space(self, required_size: int):
        """
        Освобождение места в кеше, если необходимо
        :param required_size: Требуемый размер в байтах
        """
        if self.current_size + required_size <= self.max_size_bytes:
            return
            
        # Получаем список ключей, отсортированный по времени последнего доступа
        self.cursor.execute("SELECT key, timestamp FROM cache_index ORDER BY timestamp ASC")
        keys_to_remove = []
        
        # Удаляем старые записи, пока не освободим достаточно места
        for key, _ in self.cursor.fetchall():
            if key in self.cache_index:
                keys_to_remove.append(key)
                self.current_size -= self.cache_index[key]['length']
                
                if self.current_size + required_size <= self.max_size_bytes:
                    break
        
        # Удаляем выбранные ключи
        for key in keys_to_remove:
            self._remove(key)
    
    def _remove(self, key: str):
        """Удаление записи из кеша"""
        if key in self.cache_index:
            # Удаляем из базы данных
            self.cursor.execute("DELETE FROM cache_index WHERE key = ?", (key,))
            self.conn.commit()
            
            # Удаляем из памяти
            del self.cache_index[key]
    
    def clear(self):
        """Очистка всего кеша"""
        # Закрываем соединение с базой данных
        self.conn.close()
        
        # Удаляем файлы кеша
        if self.db_path.exists():
            self.db_path.unlink()
            
        data_file = self._get_data_file()
        if data_file.exists():
            data_file.unlink()
        
        # Пересоздаем базу данных
        self._init_db()
        self.cache_index = {}
        self.current_size = 0
    
    def __del__(self):
        """Деструктор - закрываем соединение с базой данных"""
        if hasattr(self, 'conn'):
            self.conn.close()

# Пример использования
if __name__ == "__main__":
    # Создаем кеш с лимитом 1 ГБ
    cache = HybridCache(max_size_gb=1)
    
    # Сохраняем тензор
    tensor = torch.randn(3, 3)
    cache.store("test_tensor", tensor)
    
    # Получаем тензор обратно
    loaded_tensor = cache.retrieve("test_tensor")
    print("Original tensor:", tensor)
    print("Loaded tensor:", loaded_tensor)
    
    # Очищаем кеш
    cache.clear()
