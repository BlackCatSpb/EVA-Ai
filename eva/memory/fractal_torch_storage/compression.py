"""
WeightCompressor - сжатие весов моделей.
Поддерживает квантизацию и sparse-компрессию.
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger("eva.memory.fractal_torch_storage.compression")


class WeightCompressor:
    """
    Сжимает веса моделей для экономии памяти.
    
    Методы:
    - INT8 квантизация: 8-bit вместо 32-bit (4x сжатие)
    - Sparse-компрессия: Хранение только ненулевых элементов
    - Mixed precision: Разные точности для разных слоёв
    """
    
    def __init__(self, method: str = "int8", sparsity_threshold: float = 0.01):
        self.method = method
        self.sparsity_threshold = sparsity_threshold
        
        # Статистика
        self.stats = {
            "compressions": 0,
            "decompressions": 0,
            "bytes_saved": 0,
            "original_bytes": 0
        }
        
        logger.info(f"WeightCompressor инициализирован: method={method}")
    
    def should_compress(self, shape: tuple) -> bool:
        """
        Определяет нужно ли сжимать тензор.
        
        Args:
            shape: Форма тензора
            
        Returns:
            bool: True если нужно сжимать
        """
        # Сжимаем только большие тензоры
        size = 1
        for dim in shape:
            size *= dim
        
        return size > 1000  # Минимум 1000 элементов
    
    def compress(self, data: bytes, shape: tuple) -> bytes:
        """
        Сжимает данные тензора.
        
        Args:
            data: Исходные данные (float32)
            shape: Форма тензора
            
        Returns:
            bytes: Сжатые данные
        """
        import struct
        
        try:
            # Конвертируем в float32
            count = len(data) // 4
            floats = list(struct.unpack(f'{count}f', data))
            
            if self.method == "int8":
                return self._compress_int8(floats, shape)
            elif self.method == "sparse":
                return self._compress_sparse(floats, shape)
            else:
                return data
                
        except Exception as e:
            logger.warning(f"Ошибка компрессии: {e}")
            return data
    
    def decompress(self, data: bytes, shape: tuple) -> bytes:
        """
        Декомпрессия данных.
        
        Args:
            data: Сжатые данные
            shape: Форма тензора
            
        Returns:
            bytes: Восстановленные данные
        """
        import struct
        
        try:
            if self.method == "int8":
                return self._decompress_int8(data, shape)
            elif self.method == "sparse":
                return self._decompress_sparse(data, shape)
            else:
                return data
                
        except Exception as e:
            logger.warning(f"Ошибка декомпрессии: {e}")
            return data
    
    def _compress_int8(self, floats: list, shape: tuple) -> bytes:
        """
        INT8 квантизация.
        
        Конвертирует float32 в int8 с масштабированием.
        """
        import struct
        
        # Находим min/max
        if not floats:
            return b''
        
        min_val = min(floats)
        max_val = max(floats)
        
        # Вычисляем масштаб и смещение
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        zero_point = min_val
        
        # Квантизуем
        quantized = []
        for val in floats:
            q = int((val - zero_point) / scale)
            q = max(0, min(255, q))
            quantized.append(q)
        
        # Упаковываем: [min, max, bytes]
        result = struct.pack('ff', min_val, max_val)
        result += bytes(quantized)
        
        self.stats["compressions"] += 1
        self.stats["original_bytes"] += len(floats) * 4
        self.stats["bytes_saved"] += len(floats) * 4 - len(result)
        
        return result
    
    def _decompress_int8(self, data: bytes, shape: tuple) -> bytes:
        """
        INT8 деквантизация.
        """
        import struct
        
        if len(data) < 8:
            return data
        
        # Извлекаем min/max
        min_val, max_val = struct.unpack('ff', data[:8])
        quantized = data[8:]
        
        # Восстанавливаем масштаб
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        zero_point = min_val
        
        # Деквантизуем
        floats = []
        for q in quantized:
            val = q * scale + zero_point
            floats.append(val)
        
        # Конвертируем в bytes
        result = struct.pack(f'{len(floats)}f', *floats)
        
        self.stats["decompressions"] += 1
        
        return result
    
    def _compress_sparse(self, floats: list, shape: tuple) -> bytes:
        """
        Sparse-компрессия.
        Хранит только ненулевые элементы и их индексы.
        """
        import struct
        
        # Находим ненулевые элементы
        non_zero = [
            (i, val) for i, val in enumerate(floats)
            if abs(val) > self.sparsity_threshold
        ]
        
        # Формат: [count, indices..., values...]
        result = struct.pack('I', len(non_zero))
        
        for idx, val in non_zero:
            result += struct.pack('If', idx, val)
        
        self.stats["compressions"] += 1
        self.stats["original_bytes"] += len(floats) * 4
        self.stats["bytes_saved"] += len(floats) * 4 - len(result)
        
        return result
    
    def _decompress_sparse(self, data: bytes, shape: tuple) -> bytes:
        """
        Декомпрессия sparse.
        """
        import struct
        
        if len(data) < 4:
            return data
        
        # Извлекаем количество
        count = struct.unpack('I', data[:4])[0]
        
        # Восстанавливаем массив
        total = 1
        for dim in shape:
            total *= dim
        
        floats = [0.0] * total
        
        offset = 4
        for _ in range(count):
            idx, val = struct.unpack('If', data[offset:offset+8])
            if idx < total:
                floats[idx] = val
            offset += 8
        
        result = struct.pack(f'{len(floats)}f', *floats)
        
        self.stats["decompressions"] += 1
        
        return result
    
    def get_stats(self) -> dict:
        """Возвращает статистику."""
        ratio = 0.0
        if self.stats["original_bytes"] > 0:
            ratio = self.stats["bytes_saved"] / self.stats["original_bytes"]
        
        return {
            **self.stats,
            "compression_ratio": ratio
        }
