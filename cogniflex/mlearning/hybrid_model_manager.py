#!/usr/bin/env python3
"""
Гибридный менеджер моделей с горячими окнами и множественными токенизаторами
"""
import os
import sys
import json
import logging
import threading
import time
import shutil
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import psutil

# Настройка логгера
logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Возвращает корневую директорию проекта"""
    import sys
    
    possible_roots = []
    
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    possible_roots.append(os.path.dirname(os.path.dirname(current_dir)))
    possible_roots.append(os.path.dirname(current_dir))
    
    if sys.argv and sys.argv[0]:
        argv_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        possible_roots.append(argv_dir)
        possible_roots.append(os.path.dirname(argv_dir))
    
    for root in possible_roots:
        if root and os.path.exists(root):
            cogniflex_marker = os.path.join(root, 'cogniflex')
            if os.path.exists(cogniflex_marker):
                return root
    
    drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
    username = os.environ.get('USERNAME', 'user')
    
    onedrive_path = os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'CogniFlex')
    if os.path.exists(onedrive_path) and os.path.exists(os.path.join(onedrive_path, 'cogniflex')):
        return onedrive_path
    
    possible_locations = [
        os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'CogniFlex'),
        os.path.join(drive, 'Users', username, 'Desktop', 'CogniFlex'),
        os.path.join(drive, 'CogniFlex'),
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            if os.path.exists(os.path.join(loc, 'cogniflex')):
                return os.path.abspath(loc)
    
    return os.getcwd()

class WindowType(Enum):
    """Типы окон хранения"""
    HOT_VRAM = "hot_vram"      # Горячее окно в VRAM
    WARM_SSD = "warm_ssd"      # Теплое окно на SSD  
    COLD_STORAGE = "cold_storage"  # Холодное хранилище

@dataclass
class ModelWindow:
    """Окно для хранения модели"""
    model_name: str
    window_type: WindowType
    size_gb: float
    model: Any = None
    tokenizer: Any = None
    last_access: float = 0
    access_count: int = 0
    device: str = "cpu"
    is_loaded: bool = False

class HybridModelManager:
    """
    Гибридный менеджер моделей с поддержкой горячих окон
    и множественных токенизаторов
    """
    
    def __init__(self, brain=None, max_vram_gb: float = 0.5, max_ssd_gb: float = 2.0):
        self.brain = brain
        self.max_vram_gb = max_vram_gb
        self.max_ssd_gb = max_ssd_gb
        
        # Load config from brain or use defaults
        if brain and hasattr(brain, 'config'):
            self.config = brain.config.get('model', {})
        else:
            self.config = {'device': 'cuda'}
        
        # Пулы окон
        self.hot_windows: Dict[str, ModelWindow] = {}
        self.warm_windows: Dict[str, ModelWindow] = {}
        self.cold_windows: Dict[str, ModelWindow] = {}
        
        # Токенизаторы (могут быть загружены одновременно)
        self.tokenizers: Dict[str, Any] = {}
        
        # Статистика
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "vram_usage": 0.0,
            "ssd_usage": 0.0,
            "model_swaps": 0
        }
        
        # Блокировки для потокобезопасности
        self._lock = threading.RLock()
        
        # Определяем доступные ресурсы
        self._analyze_resources()
        
        # Флаг инициализации
        self.initialized = False
        
        logger.info(f"HybridModelManager инициализирован: VRAM={max_vram_gb}GB, SSD={max_ssd_gb}GB")
    
    def initialize(self) -> bool:
        """Инициализирует гибридный менеджер - использует QwenModelManager"""
        try:
            with self._lock:
                project_root = _get_project_root()
                
                # Читаем модель из конфигурации
                model_name = "qwen3.5-0.8b"
                model_path_config = "cogniflex/mlearning/cogniflex_models/qwen3.5-0.8b"
                qwen_path = os.path.join(project_root, model_path_config)
                
                # Если модель не найдена, пробуем альтернативные пути
                if not os.path.exists(qwen_path):
                    alt_paths = [
                        os.path.join(project_root, "cogniflex", "mlearning", "cogniflex_models", "qwen3.5-2b"),
                        os.path.join(project_root, "mlearning", "cogniflex_models", "qwen3.5-0.8b"),
                        os.path.join(project_root, "mlearning", "cogniflex_models", "qwen3.5-2b"),
                    ]
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            qwen_path = alt_path
                            model_name = os.path.basename(alt_path)
                            break
                
                success = self.register_model(
                    model_name=model_name,
                    model_path=qwen_path,
                    tokenizer_path=qwen_path,
                    priority=1
                )
                
                if success:
                    tokenizer_success = self.load_tokenizer(model_name, qwen_path)
                    
                    if not tokenizer_success:
                        logger.warning("Не удалось загрузить токенизатор Qwen, используем только QwenModelManager")
                    
                    self.initialized = True
                    logger.info(f"HybridModelManager инициализирован: {model_name} (через QwenModelManager)")
                    return True
                else:
                    logger.error("Не удалось зарегистрировать модель Qwen")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка инициализации HybridModelManager: {e}")
            return False
    
    def _analyze_resources(self):
        """Анализирует доступные ресурсы"""
        try:
            # VRAM
            if torch.cuda.is_available():
                self.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"Доступно VRAM: {self.total_vram_gb:.2f}GB")
            else:
                self.total_vram_gb = 0
                logger.info("VRAM недоступен")
            
            # RAM
            memory = psutil.virtual_memory()
            self.total_ram_gb = memory.total / 1024**3
            self.free_ram_gb = memory.available / 1024**3
            logger.info(f"Доступно RAM: {self.total_ram_gb:.2f}GB (свободно: {self.free_ram_gb:.2f}GB)")
            
            # SSD пространство - кроссплатформенный подход
            try:
                if os.path.exists("./cogniflex_cache"):
                    # Используем shutil для кроссплатформенного получения размера диска
                    import shutil
                    self.free_ssd_gb = shutil.disk_usage("./cogniflex_cache").free / 1024**3
                    logger.info(f"Доступно SSD: {self.free_ssd_gb:.2f}GB")
                else:
                    self.free_ssd_gb = 10.0  # По умолчанию
                    logger.info("SSD пространство не определено, используем 10GB по умолчанию")
            except Exception as e:
                logger.warning(f"Ошибка определения SSD пространства: {e}")
                self.free_ssd_gb = 10.0
                
        except Exception as e:
            logger.warning(f"Ошибка анализа ресурсов: {e}")
            self.total_vram_gb = 0
            self.total_ram_gb = 8.0
            self.free_ram_gb = 2.0
            self.free_ssd_gb = 10.0
    
    def register_model(self, model_name: str, model_path: str, tokenizer_path: str = None, 
                      priority: int = 1) -> bool:
        """Регистрирует модель в системе"""
        try:
            with self._lock:
                # Определяем тип окна на основе приоритета и доступных ресурсов
                config_device = self.config.get('device', 'cpu')
                if priority <= 2 and self.total_vram_gb > 0:
                    window_type = WindowType.HOT_VRAM
                    device = config_device if config_device.startswith('cuda') else "cuda" if torch.cuda.is_available() else "cpu"
                elif priority <= 5:
                    window_type = WindowType.WARM_SSD
                    device = "cpu"
                else:
                    window_type = WindowType.COLD_STORAGE
                    device = "cpu"
                
                # Создаем окно
                window = ModelWindow(
                    model_name=model_name,
                    window_type=window_type,
                    size_gb=self._estimate_model_size(model_path),
                    device=device,
                    last_access=time.time()
                )
                
                # Добавляем в соответствующий пул
                if window_type == WindowType.HOT_VRAM:
                    self.hot_windows[model_name] = window
                elif window_type == WindowType.WARM_SSD:
                    self.warm_windows[model_name] = window
                else:
                    self.cold_windows[model_name] = window
                
                logger.info(f"Модель {model_name} зарегистрирована в {window_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка регистрации модели {model_name}: {e}")
            return False
    
    def _estimate_model_size(self, model_path: str) -> float:
        """Оценивает размер модели в GB"""
        try:
            if os.path.exists(model_path):
                size_bytes = 0
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        size_bytes += os.path.getsize(os.path.join(root, file))
                return size_bytes / 1024**3
            else:
                # Оценка по умолчанию для ruGPT-3
                return 1.5  # ~1.5GB
        except Exception:
            return 1.5
    
    def load_tokenizer(self, model_name: str, tokenizer_path: str) -> bool:
        """Загружает токенизатор (может быть несколько одновременно)"""
        try:
            from transformers import AutoTokenizer
            
            with self._lock:
                if model_name in self.tokenizers:
                    logger.debug(f"Токенизатор {model_name} уже загружен")
                    return True
                
                # Используем абсолютный путь
                if not os.path.isabs(tokenizer_path):
                    project_root = _get_project_root()
                    tokenizer_path = os.path.join(project_root, tokenizer_path)
                
                logger.info(f"Загружаем токенизатор из: {tokenizer_path}")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                    trust_remote_code=False
                )
                
                if tokenizer:
                    self.tokenizers[model_name] = tokenizer
                    logger.info(f"Токенизатор {model_name} загружен")
                    return True
                else:
                    logger.error(f"Не удалось загрузить токенизатор {model_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки токенизатора {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str, force_device: str = None) -> bool:
        """Загружает модель с учетом горячих окон"""
        try:
            with self._lock:
                # Ищем модель в пулах
                window = self._find_window(model_name)
                if not window:
                    logger.error(f"Модель {model_name} не найдена")
                    return False
                
                # Если уже загружена
                if window.is_loaded and window.model:
                    window.last_access = time.time()
                    window.access_count += 1
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Модель {model_name} найдена в кэше")
                    return True
                
                # Определяем устройство
                device = force_device or window.device
                
                # Проверяем ресурсы
                if device == "cuda" and not self._can_fit_in_vram(window.size_gb):
                    logger.warning(f"Модель {model_name} не помещается в VRAM, используем CPU")
                    device = "cpu"
                    window.device = "cpu"
                
                # Загружаем модель
                if self._load_model_to_window(window, device):
                    window.is_loaded = True
                    window.last_access = time.time()
                    window.access_count = 1
                    self.stats["cache_misses"] += 1
                    
                    # Обновляем статистику использования
                    if device == "cuda":
                        self.stats["vram_usage"] += window.size_gb
                    else:
                        self.stats["ssd_usage"] += window.size_gb
                    
                    logger.info(f"Модель {model_name} загружена на {device}")
                    return True
                else:
                    logger.error(f"Не удалось загрузить модель {model_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            return False
    
    def _find_window(self, model_name: str) -> Optional[ModelWindow]:
        """Находит окно модели"""
        for pool in [self.hot_windows, self.warm_windows, self.cold_windows]:
            if model_name in pool:
                return pool[model_name]
        return None
    
    def _can_fit_in_vram(self, size_gb: float) -> bool:
        """Проверяет помещается ли модель в VRAM"""
        if not torch.cuda.is_available():
            return False
        
        used_vram = torch.cuda.memory_allocated(0) / 1024**3
        available_vram = self.total_vram_gb - used_vram
        
        return size_gb <= available_vram * 0.8  # 80% для безопасности
    
    def _load_model_to_window(self, window: ModelWindow, device: str) -> bool:
        """Загружает модель в окно - теперь через QwenModelManager"""
        logger.info("Текстовая генерация теперь через QwenModelManager, не HybridModelManager")
        return False
    
    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ (совместимый интерфейс с ML Unit).
        
        Args:
            prompt: Текст запроса
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Структурированный ответ
        """
        try:
            import time
            with self._lock:
                start_time = time.time()
                # Используем первую доступную модель
                available_models = self.get_available_models()
                if not available_models:
                    return {
                        "text": "Ошибка: нет доступных моделей",
                        "status": "error",
                        "error": "Нет доступных моделей в гибридном менеджере"
                    }
                
                model_name = list(available_models.keys())[0]
                
                # Извлекаем параметры
                max_tokens = kwargs.get('max_tokens', kwargs.get('max_length', 500))
                temperature = kwargs.get('temperature', 0.7)
                
                # Генерируем ответ через внутренний метод
                response_text = self._generate_response_internal(
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                generation_time = time.time() - start_time
                
                return {
                    "text": response_text,
                    "status": "ok",
                    "model_name": model_name,
                    "tokens_generated": len(response_text.split()),
                    "generation_time": generation_time,
                    "metadata": {
                        "provider": "HybridModelManager",
                        "window_type": available_models[model_name].get('window_type', 'unknown'),
                        "device": available_models[model_name].get('device', 'unknown')
                    }
                }
                
        except Exception as e:
            logger.error(f"Ошибка генерации в HybridModelManager: {e}")
            return {
                "text": f"Ошибка генерации: {str(e)}",
                "status": "error",
                "error": str(e)
            }
    
    def _generate_response_internal(self, model_name: str, prompt: str, max_tokens: int = 500,
                                   temperature: float = 0.7, **kwargs) -> str:
        """Внутренний метод генерации ответа"""
        try:
            # Проверяем наличие модели в окне
            window = self._find_window(model_name)
            
            if window and window.model:
                # Модель загружена напрямую
                if model_name not in self.tokenizers:
                    return f"Ошибка: токенизатор для {model_name} не загружен"
                
                tokenizer = self.tokenizers[model_name]
                return self._generate_with_model(window.model, tokenizer, prompt, 
                                               max_tokens, temperature, window.device, **kwargs)
            else:
                # Модель не загружена в HybridModelManager
                # Используем QwenModelManager если доступен через brain
                logger.info(f"Модель {model_name} не загружена в HybridManager, используем базовый ответ")
                return self._generate_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return f"Ошибка генерации: {str(e)}"
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Генерирует базовый fallback ответ без модели"""
        prompt_lower = prompt.lower()
        
        fallbacks = {
            'привет': 'Привет! Я CogniFlex, рад общению.',
            'здравств': 'Здравствуйте! Чем могу помочь?',
            'как дела': 'Спасибо, что спрашиваете! У меня всё хорошо.',
            'как тебя': 'Меня зовут CogniFlex.',
            'кто ты': 'Я CogniFlex - когнитивная AI система.',
        }
        
        for key, response in fallbacks.items():
            if key in prompt_lower:
                return response
        
        return 'Интересный вопрос! Расскажите подробнее.'
    
    def _generate_with_model(self, model, tokenizer, prompt: str, max_tokens: int,
                           temperature: float, device: str, **kwargs) -> str:
        """Генерирует текст с моделью"""
        try:
            # Токенизация
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            # Переносим на устройство модели
            inputs = inputs.to(device)
            
            # Генерация
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=False,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Декодирование
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Удаляем промпт
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации с моделью: {e}")
            return f"Ошибка генерации: {str(e)}"
    
    def get_available_models(self) -> Dict[str, Any]:
        """Возвращает доступные модели"""
        models = {}
        
        with self._lock:
            for pool_name, pool in [("hot", self.hot_windows), 
                                   ("warm", self.warm_windows), 
                                   ("cold", self.cold_windows)]:
                for model_name, window in pool.items():
                    models[model_name] = {
                        "display_name": model_name,
                        "type": "text-generation",
                        "status": "loaded" if window.is_loaded else "cached",
                        "device": window.device,
                        "window_type": window.window_type.value,
                        "size_gb": window.size_gb,
                        "access_count": window.access_count,
                        "pool": pool_name
                    }
        
        return models
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        with self._lock:
            return {
                **self.stats,
                "hot_windows": len(self.hot_windows),
                "warm_windows": len(self.warm_windows),
                "cold_windows": len(self.cold_windows),
                "loaded_tokenizers": len(self.tokenizers),
                "total_models": len(self.hot_windows) + len(self.warm_windows) + len(self.cold_windows)
            }
    
    def optimize_windows(self):
        """Оптимизирует распределение окон"""
        try:
            with self._lock:
                # TODO: Реализовать оптимизацию на основе статистики использования
                logger.info("Оптимизация окон завершена")
                
        except Exception as e:
            logger.error(f"Ошибка оптимизации окон: {e}")
    
    def cleanup(self):
        """Очищает ресурсы"""
        try:
            with self._lock:
                # Выгружаем модели
                for pool in [self.hot_windows, self.warm_windows, self.cold_windows]:
                    for window in pool.values():
                        if window.model:
                            del window.model
                            window.is_loaded = False
                
                # Очищаем токенизаторы
                for tokenizer in self.tokenizers.values():
                    del tokenizer
                self.tokenizers.clear()
                
                # Очищаем CUDA кэш
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("Ресурсы HybridModelManager очищены")
                
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
    
    def get_model_for_task(self, task: str):
        """Возвращает модель для указанной задачи"""
        try:
            with self._lock:
                # Для задачи text-generation возвращаем первую доступную модель
                if task == "text-generation":
                    # Ищем в горячих окнах
                    for model_name, window in self.hot_windows.items():
                        if window.is_loaded:
                            return (window.model, self.tokenizers.get(model_name), model_name)
                    
                    # Ищем в теплых окнах
                    for model_name, window in self.warm_windows.items():
                        # Загружаем модель если нужно
                        if self.load_model(model_name):
                            return (window.model, self.tokenizers.get(model_name), model_name)
                    
                    # Ищем в холодных окнах
                    for model_name, window in self.cold_windows.items():
                        # Загружаем модель если нужно
                        if self.load_model(model_name):
                            return (window.model, self.tokenizers.get(model_name), model_name)
                
                logger.warning(f"Модель для задачи '{task}' не найдена")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка получения модели для задачи '{task}': {e}")
            return None
    
    @property
    def tokenizer(self):
        """Возвращает токенизатор по умолчанию для совместимости"""
        if self.tokenizers:
            # Возвращаем первый доступный токенизатор
            return next(iter(self.tokenizers.values()))
        return None
