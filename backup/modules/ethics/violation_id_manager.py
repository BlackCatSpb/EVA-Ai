"""
Модуль для управления идентификаторами нарушений в этической рамке ЕВА
"""
import time
import hashlib
import re
import logging
import os
from typing import List
from typing import Dict, Optional, Tuple, NamedTuple

logger = logging.getLogger("eva.ethics.id_manager")

class ViolationIDComponents(NamedTuple):
    """Компоненты идентификатора нарушения."""
    timestamp: float
    principle: str
    hash: str
    version: str = "1"

class ViolationIDManager:
    """
    Менеджер идентификаторов нарушений.
    
    Обеспечивает генерацию, валидацию и парсинг уникальных идентификаторов нарушений.
    Поддерживает несколько форматов для обратной совместимости.
    """
    
    # Регулярные выражения для различных форматов ID
    ID_PATTERNS = {
        # Новый формат: violation:v1:1725197415:privacy:a1b2c3
        "v1": re.compile(r'^violation:v([1-9]):(\d+):([a-z]+):([a-f0-9]{6,8})$'),
        
        # Старый формат: principle_1725197415
        "legacy": re.compile(r'^([a-z]+)_(\d+)$')
    }
    
    # Сокращения для принципов (чтобы уменьшить длину ID)
    PRINCIPLE_SHORTCODES = {
        "Non-Maleficence": "nm",
        "Beneficence": "bf",
        "Autonomy": "at",
        "Justice": "js",
        "Transparency": "tr",
        "Privacy": "pr",
        "Accountability": "ac",
        "System": "sys"
    }
    
    # Обратное отображение для расшифровки
    SHORTCODES_TO_PRINCIPLES = {v: k for k, v in PRINCIPLE_SHORTCODES.items()}
    
    def __init__(self):
        """Инициализирует менеджер идентификаторов."""
        logger.debug("Инициализирован менеджер идентификаторов нарушений")
    
    def generate_id(self, principle: str, timestamp: Optional[float] = None) -> str:
        """
        Генерирует уникальный идентификатор нарушения.
        
        Args:
            principle: Название принципа
            timestamp: Временная метка (опционально, по умолчанию текущее время)
            
        Returns:
            str: Уникальный идентификатор нарушения
        """
        # Используем текущее время, если временная метка не предоставлена
        timestamp = timestamp or time.time()
        timestamp_int = int(timestamp * 1000)  # Миллисекунды для большей уникальности
        
        # Получаем короткий код принципа
        principle_code = self._get_principle_shortcode(principle)
        if not principle_code:
            logger.warning(f"Неизвестный принцип '{principle}', используем код 'unk'")
            principle_code = "unk"
        
        # Генерируем уникальный хеш
        hash_input = f"{timestamp_int}_{principle_code}_{os.urandom(8).hex()}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        # Формируем ID в новом формате
        violation_id = f"violation:v1:{timestamp_int}:{principle_code}:{hash_value}"
        
        logger.debug(f"Сгенерирован новый ID нарушения: {violation_id}")
        return violation_id
    
    def _get_principle_shortcode(self, principle: str) -> str:
        """
        Возвращает короткий код для принципа.
        
        Args:
            principle: Название принципа
            
        Returns:
            str: Короткий код принципа
        """
        # Приводим к нижнему регистру и удаляем дефисы
        normalized = principle.lower().replace("-", "")
        return self.PRINCIPLE_SHORTCODES.get(principle, normalized[:3])
    
    def parse_id(self, violation_id: str) -> Optional[ViolationIDComponents]:
        """
        Парсит идентификатор нарушения и возвращает его компоненты.
        
        Args:
            violation_id: Идентификатор нарушения
            
        Returns:
            Optional[ViolationIDComponents]: Компоненты ID или None при ошибке
        """
        # Пытаемся распознать формат
        for format_name, pattern in self.ID_PATTERNS.items():
            match = pattern.match(violation_id)
            if match:
                try:
                    if format_name == "v1":
                        # violation:v1:1725197415:privacy:a1b2c3
                        version = match.group(1)
                        timestamp = int(match.group(2)) / 1000  # Конвертируем обратно в секунды
                        principle_code = match.group(3)
                        hash_value = match.group(4)
                        
                        # Расшифровываем принцип
                        principle = self.SHORTCODES_TO_PRINCIPLES.get(principle_code, principle_code)
                        
                        return ViolationIDComponents(
                            timestamp=timestamp,
                            principle=principle,
                            hash=hash_value,
                            version=version
                        )
                    
                    elif format_name == "legacy":
                        # principle_1725197415
                        principle = match.group(1)
                        timestamp = int(match.group(2))
                        
                        # Генерируем хеш для совместимости
                        hash_input = f"{timestamp}_{principle}"
                        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]
                        
                        return ViolationIDComponents(
                            timestamp=timestamp,
                            principle=principle,
                            hash=hash_value,
                            version="legacy"
                        )
                
                except Exception as e:
                    logger.error(f"Ошибка парсинга ID '{violation_id}' в формате {format_name}: {e}")
                    return None
        
        logger.warning(f"Неизвестный формат ID нарушения: {violation_id}")
        return None
    
    def is_valid_id(self, violation_id: str) -> bool:
        """
        Проверяет, является ли строка валидным идентификатором нарушения.
        
        Args:
            violation_id: Строка для проверки
            
        Returns:
            bool: True, если строка является валидным ID
        """
        for pattern in self.ID_PATTERNS.values():
            if pattern.match(violation_id):
                return True
        return False
    
    def convert_legacy_id(self, legacy_id: str) -> str:
        """
        Конвертирует старый формат ID в новый формат.
        
        Args:
            legacy_id: Старый формат ID
            
        Returns:
            str: Новый формат ID
        """
        match = self.ID_PATTERNS["legacy"].match(legacy_id)
        if not match:
            logger.error(f"Невозможно конвертировать ID: '{legacy_id}' не соответствует старому формату")
            return legacy_id
        
        principle = match.group(1)
        timestamp = int(match.group(2))
        
        # Генерируем новый ID
        return self.generate_id(principle, timestamp)
    
    def get_timestamp(self, violation_id: str) -> Optional[float]:
        """
        Извлекает временную метку из идентификатора.
        
        Args:
            violation_id: Идентификатор нарушения
            
        Returns:
            Optional[float]: Временная метка или None при ошибке
        """
        components = self.parse_id(violation_id)
        return components.timestamp if components else None
    
    def get_principle(self, violation_id: str) -> Optional[str]:
        """
        Извлекает принцип из идентификатора.
        
        Args:
            violation_id: Идентификатор нарушения
            
        Returns:
            Optional[str]: Название принципа или None при ошибке
        """
        components = self.parse_id(violation_id)
        return components.principle if components else None
    
    def get_hash(self, violation_id: str) -> Optional[str]:
        """
        Извлекает хеш из идентификатора.
        
        Args:
            violation_id: Идентификатор нарушения
            
        Returns:
            Optional[str]: Хеш или None при ошибке
        """
        components = self.parse_id(violation_id)
        return components.hash if components else None
    
    def get_version(self, violation_id: str) -> Optional[str]:
        """
        Извлекает версию формата из идентификатора.
        
        Args:
            violation_id: Идентификатор нарушения
            
        Returns:
            Optional[str]: Версия формата или None при ошибке
        """
        components = self.parse_id(violation_id)
        return components.version if components else None
    
    def is_new_format(self, violation_id: str) -> bool:
        """
        Проверяет, использует ли ID новый формат.
        
        Args:
            violation_id: Идентификатор нарушения
            
        Returns:
            bool: True, если ID в новом формате
        """
        components = self.parse_id(violation_id)
        return components is not None and components.version != "legacy"
    
    def create_compatibility_layer(self) -> Dict[str, str]:
        """
        Создает слой совместимости для перехода со старого формата на новый.
        
        Returns:
            Dict[str, str]: Словарь соответствия старых и новых ID
        """
        # В реальной системе здесь был бы код для массового обновления ID
        # Для примера возвращаем пустой словарь
        logger.info("Создан слой совместимости для идентификаторов нарушений")
        return {}
    
    def validate_id_consistency(self, violation_id: str, principle: str, 
                              timestamp: float) -> bool:
        """
        Проверяет, соответствует ли ID предоставленным данным.
        
        Args:
            violation_id: Идентификатор нарушения
            principle: Ожидаемый принцип
            timestamp: Ожидаемая временная метка
            
        Returns:
            bool: True, если ID соответствует данным
        """
        components = self.parse_id(violation_id)
        if not components:
            return False
        
        # Нормализуем принцип для сравнения
        principle_code = self._get_principle_shortcode(principle)
        expected_principle = self.SHORTCODES_TO_PRINCIPLES.get(principle_code, principle_code)
        
        # Проверяем принцип с учетом возможных различий в формате
        principle_match = (
            components.principle == principle or
            components.principle == principle_code or
            self._get_principle_shortcode(components.principle) == principle_code
        )
        
        # Проверяем временную метку с небольшой погрешностью
        time_match = abs(components.timestamp - timestamp) < 0.1
        
        is_consistent = principle_match and time_match
        
        if not is_consistent:
            logger.warning(
                f"Несоответствие ID: {violation_id}\n"
                f"Ожидалось: принцип={principle}, время={timestamp}\n"
                f"Получено: принцип={components.principle}, время={components.timestamp}"
            )
        
        return is_consistent
    
    def generate_batch_ids(self, principle: str, count: int, 
                          base_timestamp: Optional[float] = None) -> List[str]:
        """
        Генерирует пакет идентификаторов для одного принципа.
        
        Args:
            principle: Название принципа
            count: Количество ID для генерации
            base_timestamp: Базовая временная метка (опционально)
            
        Returns:
            List[str]: Список идентификаторов
        """
        base_timestamp = base_timestamp or time.time()
        ids = []
        
        for i in range(count):
            # Слегка сдвигаем временную метку для каждого ID
            timestamp = base_timestamp + (i * 0.001)
            ids.append(self.generate_id(principle, timestamp))
        
        logger.debug(f"Сгенерировано {count} ID для принципа '{principle}'")
        return ids
    
    def get_id_age(self, violation_id: str) -> Optional[float]:
        """
        Возвращает возраст ID в секундах.
        
        Args:
            violation_id: Идентификатор нарушения
            
        Returns:
            Optional[float]: Возраст ID в секундах или None при ошибке
        """
        timestamp = self.get_timestamp(violation_id)
        if timestamp is None:
            return None
        return time.time() - timestamp
    
    def is_recent_id(self, violation_id: str, max_age: float = 86400) -> bool:
        """
        Проверяет, является ли ID недавно созданным.
        
        Args:
            violation_id: Идентификатор нарушения
            max_age: Максимальный возраст в секундах (по умолчанию 24 часа)
            
        Returns:
            bool: True, если ID создан недавно
        """
        age = self.get_id_age(violation_id)
        return age is not None and age <= max_age
    
    def normalize_id(self, violation_id: str) -> str:
        """
        Нормализует ID, преобразуя его в новый формат при необходимости.
        
        Args:
            violation_id: Исходный идентификатор
            
        Returns:
            str: Нормализованный идентификатор в новом формате
        """
        if self.is_new_format(violation_id):
            return violation_id
        
        # Проверяем, является ли это старым форматом
        if self.ID_PATTERNS["legacy"].match(violation_id):
            return self.convert_legacy_id(violation_id)
        
        # Если ID не соответствует ни одному известному формату,
        # пытаемся извлечь информацию и создать новый ID
        logger.warning(f"Получен ID неизвестного формата: {violation_id}")
        
        # Пытаемся извлечь принцип из строки
        principle_match = re.search(r'([a-z]+)', violation_id.lower())
        principle = principle_match.group(1) if principle_match else "unknown"
        
        # Используем текущее время
        return self.generate_id(principle)

# Глобальный экземпляр для удобства использования
id_manager = ViolationIDManager()

# Функции для удобного доступа к функциональности менеджера
def generate_violation_id(principle: str, timestamp: Optional[float] = None) -> str:
    """Генерирует новый уникальный ID нарушения."""
    return id_manager.generate_id(principle, timestamp)

def parse_violation_id(violation_id: str) -> Optional[ViolationIDComponents]:
    """Парсит ID нарушения и возвращает его компоненты."""
    return id_manager.parse_id(violation_id)

def is_valid_violation_id(violation_id: str) -> bool:
    """Проверяет валидность ID нарушения."""
    return id_manager.is_valid_id(violation_id)

def convert_legacy_violation_id(legacy_id: str) -> str:
    """Конвертирует старый ID в новый формат."""
    return id_manager.convert_legacy_id(legacy_id)

def get_violation_timestamp(violation_id: str) -> Optional[float]:
    """Извлекает временную метку из ID нарушения."""
    return id_manager.get_timestamp(violation_id)

def get_violation_principle(violation_id: str) -> Optional[str]:
    """Извлекает принцип из ID нарушения."""
    return id_manager.get_principle(violation_id)