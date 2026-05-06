"""
Менеджер конфигурации для ЕВА
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Управляет конфигурацией системы ЕВА."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Инициализирует менеджер конфигурации.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path or "eva_config.json"
        self.defaults = self._load_default_config()
        self.config = self._load_default_config()
        self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию по умолчанию."""
        return {
            "system": {
                "max_memory_tokens": 50000,
                "cache_size_gb": 50.0,
                "log_level": "INFO",
                "enable_async": True,
                "max_workers": 8
            },
            "ml": {
                "model_cache_size": 1000,
                "tokenizer_batch_size": 32,
                "embedding_dim": 768,
                "use_gpu": True
            },
            "knowledge": {
                "max_graph_nodes": 100000,
                "relationship_threshold": 0.7,
                "auto_expand": True
            },
            "memory": {
                "retention_days": 30,
                "compression_level": 6,
                "backup_interval": 3600
            },
            "ethics": {
                "strict_mode": True,
                "bias_detection": True,
                "content_filtering": True
            },
            "gui": {
                "theme": "dark",
                "auto_refresh": True,
                "max_history": 1000
            }
        }
    
    def _load_config_file(self):
        """Загружает конфигурацию из файла."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._merge_config(file_config)
                logger.info(f"Конфигурация загружена из {self.config_path}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Объединяет новую конфигурацию с существующей."""
        for section, values in new_config.items():
            if section in self.config and isinstance(values, dict):
                self.config[section].update(values)
            else:
                self.config[section] = values
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Получает значение конфигурации.
        
        Args:
            section: Секция конфигурации
            key: Ключ параметра
            default: Значение по умолчанию
            
        Returns:
            Значение параметра или default
        """
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Устанавливает значение конфигурации.
        
        Args:
            section: Секция конфигурации
            key: Ключ параметра
            value: Новое значение
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Получает всю секцию конфигурации.
        
        Args:
            section: Название секции
            
        Returns:
            Словарь с параметрами секции
        """
        return self.config.get(section, {})
    
    def save_config(self):
        """Сохраняет текущую конфигурацию в файл."""
        try:
            config_dir = os.path.dirname(self.config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Конфигурация сохранена в {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def reload_config(self):
        """Перезагружает конфигурацию из файла."""
        self.config = self._load_default_config()
        self._load_config_file()
        logger.info("Конфигурация перезагружена")
    
    def validate_config(self) -> bool:
        """Проверяет корректность конфигурации.
        
        Returns:
            True если конфигурация корректна
        """
        required_sections = ["system", "ml", "knowledge", "memory"]
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Отсутствует обязательная секция: {section}")
                return False
        
        # Проверка типов и диапазонов значений
        system_config = self.config.get("system", {})
        
        if not isinstance(system_config.get("max_memory_tokens", 0), int):
            logger.error("max_memory_tokens должно быть целым числом")
            return False
        
        if system_config.get("max_memory_tokens", 0) <= 0:
            logger.error("max_memory_tokens должно быть положительным")
            return False
        
        cache_size = system_config.get("cache_size_gb", 0)
        if not isinstance(cache_size, (int, float)) or cache_size <= 0:
            logger.error("cache_size_gb должно быть положительным числом")
            return False
        
        return True
    
    def get_all_config(self) -> Dict[str, Any]:
        """Возвращает всю конфигурацию.
        
        Returns:
            Полная конфигурация системы
        """
        return self.config.copy()
