
"""
Model Manager для работы с моделями в CogniFlex.

Этот модуль предоставляет интерфейс для работы с моделями через CoreBrain.
"""
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, TypeVar

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoConfig = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

# Импортируем базовый класс компонента
try:
    from cogniflex.core.base_component import BaseComponent
except ImportError:
    # Для обратной совместимости
    BaseComponent = object
    logger = logging.getLogger("cogniflex.model_manager")
else:
    logger = logging.getLogger("cogniflex.model_manager")

# Тип для аннотаций
T = TypeVar('T')

class ModelManager(BaseComponent):
    """
    Менеджер моделей для работы с CoreBrain.
    
    Основные возможности:
    - Загрузка моделей через CoreBrain
    - Регистрация инициализаторов моделей
    - Управление зависимостями между моделями
    """
    
    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Инициализация менеджера моделей.
        
        Args:
            brain: Ссылка на ядро системы
            config: Конфигурация менеджера моделей
            **kwargs: Дополнительные параметры для совместимости
        """
        super().__init__(brain, config or {})
        
        # Устанавливаем логгер
        self.logger = logger.getChild('ModelManager')
        
        # Конфигурация моделей по умолчанию
        self.model_configs = {
            'text-generation': {
                'dependencies': [],
                'model_class': 'AutoModelForCausalLM',
                'tokenizer_class': 'AutoTokenizer',
            },
            'fractal-text-generation': {
                'dependencies': [],
                'model_class': 'FractalTransformer',
                'tokenizer_class': 'ExtendedFractalTokenizer',
                'config_class': 'FractalConfig',
                'use_fractal_storage': True,
                'memory_slots': 32,
                'memory_size': 512,
            },
            'text-embedding': {
                'dependencies': ['text-generation'],
                'model_class': 'AutoModel',
                'tokenizer_class': 'AutoTokenizer',
            },
        }
        
        # Обновляем конфигурацию из переданных параметров
        if 'model_configs' in self.config:
            self.model_configs.update(self.config['model_configs'])
        
        # Инициализируем кэш моделей
        self._model_cache: Dict[str, Any] = {}
        
        # Директория, где лежат модели в фрактальном формате (поддиректории по task)
        self.fractal_models_dir: str = os.path.abspath(
            str(kwargs.get('model_dir') or self.config.get('model_dir') or self.config.get('models_dir') or './cache/fractal_storage/models')
        )

        # Инициализируем фрактальное хранилище
        self.fractal_store = None
        self._init_fractal_store()
        
        # Определяем зависимости
        self._required_components = []
        self._optional_components = []
        
        self.logger.debug("ModelManager инициализирован")
        
    def _init_fractal_store(self, deferred_system=None):
        """Инициализирует фрактальное хранилище для моделей с отложенной инициализацией.
        
        Args:
            deferred_system: Система отложенных команд для асинхронной инициализации
        """
        def _init_store():
            try:
                from .storage.fractal_store import FractalWeightStore

                store_config = self.config.get('fractal_store', {})
                block_size = int(store_config.get('block_size', 64))
                fractal_levels = int(store_config.get('fractal_levels', 4))
                containers_per_group = int(store_config.get('containers_per_group', 4))

                device = 'cpu'
                try:
                    if torch is not None and torch.cuda.is_available():
                        device = 'cuda'
                except Exception:
                    device = 'cpu'

                self.fractal_store = FractalWeightStore(
                    block_size=block_size,
                    fractal_levels=fractal_levels,
                    containers_per_group=containers_per_group,
                    device=device,
                )

                self.logger.info(f"Фрактальное хранилище моделей (FractalWeightStore) инициализировано на устройстве: {device}")

                if hasattr(self, 'brain') and hasattr(self.brain, 'register_component'):
                    try:
                        self.brain.register_component('fractal_weight_store', self.fractal_store)
                    except Exception:
                        pass

                return True
                    
            except ImportError as e:
                error_msg = f"Не удалось импортировать FractalWeightStore: {e}"
                self.logger.warning(error_msg)
                return False
            except Exception as e:
                error_msg = f"Ошибка при инициализации фрактального хранилища: {e}"
                self.logger.error(error_msg, exc_info=True)
                return False
        
        # Если передан deferred_system, используем отложенную инициализацию
        if deferred_system is not None and hasattr(deferred_system, 'defer_command'):
            # Проверяем, доступен ли torch для импорта
            def check_torch_available():
                try:
                    import torch
                    return True
                except ImportError:
                    return False
            
            # Откладываем инициализацию с проверкой доступности torch
            deferred_system.defer_command(
                _init_store,
                priority='critical',
                condition=check_torch_available,
                retries=3,
                delay=5,
                name='init_fractal_store'
            )
            self.logger.info("Отложенная инициализация фрактального хранилища запланирована")
        else:
            # Прямая синхронная инициализация (для обратной совместимости)
            return _init_store()
    
    def _setup_component(self) -> None:
        """Настраивает компонент после проверки зависимостей."""
        # Регистрируем модели в CoreBrain
        self.register_models_with_core_brain()
        self.logger.info("ModelManager готов к работе")
    
    def get_model_for_task(self, task_type: str, model_name: Optional[str] = None, **kwargs) -> Tuple[Any, Any, str]:
        """
        Получает модель и токенизатор для указанной задачи.
        
        Args:
            task_type: Тип задачи (например, 'text-generation', 'fractal-text-generation')
            model_name: Имя модели (опционально)
            **kwargs: Дополнительные параметры для инициализации модели
            
        Returns:
            Кортеж (модель, токенизатор, имя_модели)
        """
        model_name = model_name or str(task_type)

        if model_name in self._model_cache and not kwargs.get('force_reload', False):
            return self._model_cache[model_name]

        if AutoConfig is None or AutoModelForCausalLM is None:
            self.logger.error("transformers недоступен: невозможно собрать HF-модель из фрактального хранилища")
            return None, None, None

        # Ожидаем структуру:
        #   <fractal_models_dir>/<task_type>/index.json + (shards_manifest.jsonl|containers.jsonl|data/) + config.json + tokenizer/
        task_dir = os.path.join(self.fractal_models_dir, str(task_type))
        if not os.path.isdir(task_dir):
            self.logger.error(
                f"Фрактальная модель для task='{task_type}' не найдена: {task_dir}. "
                f"Сначала экспортируйте модель (ruGPT-small) в это хранилище."
            )
            return None, None, None

        try:
            from .storage.fractal_store import FractalWeightStore

            store_config = self.config.get('fractal_store', {})
            block_size = int(store_config.get('block_size', 64))
            fractal_levels = int(store_config.get('fractal_levels', 4))
            containers_per_group = int(store_config.get('containers_per_group', 4))

            device = 'cpu'
            try:
                if torch is not None and torch.cuda.is_available() and bool(kwargs.get('use_gpu', True)):
                    device = 'cuda'
            except Exception:
                device = 'cpu'

            fs = FractalWeightStore(
                block_size=block_size,
                fractal_levels=fractal_levels,
                containers_per_group=containers_per_group,
                device=device,
            )

            # Шардированный формат читаем лениво (экономия RAM)
            fs.load_from_disk(task_dir, lazy=True)

            state_dict = fs.reconstruct_state_dict(output_dtype='float32', device='cpu')

            cfg = AutoConfig.from_pretrained(task_dir, local_files_only=True)
            model = AutoModelForCausalLM.from_config(cfg)
            model.load_state_dict(state_dict, strict=False)

            if device == 'cuda' and torch is not None and torch.cuda.is_available():
                model.to('cuda')
            else:
                model.to('cpu')
            model.eval()

            tokenizer = None
            tok_dir = os.path.join(task_dir, 'tokenizer')
            try:
                if AutoTokenizer is not None and os.path.isdir(tok_dir):
                    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, use_fast=True)
            except Exception:
                tokenizer = None
            if tokenizer is None:
                tokenizer = getattr(self.brain, 'tokenizer', None)

            result = (model, tokenizer, model_name)
            self._model_cache[model_name] = result

            if self.brain is not None:
                try:
                    self.brain.register_component(f"model_{task_type}", model)
                    if tokenizer is not None:
                        self.brain.register_component(f"tokenizer_{task_type}", tokenizer)
                except Exception:
                    pass

            return result
        except Exception as e:
            self.logger.error(
                f"Критическая ошибка при загрузке модели из фрактального хранилища (task='{task_type}'): {e}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
            return None, None, None

    def export_task_model_to_fractal(self, task_type: str, hf_model_dir_or_id: str, model_id: str, **kwargs) -> bool:
        """Явный экспорт HF-модели (ruGPT-small) в фрактальное хранилище под task.

        Пишет в: <fractal_models_dir>/<task_type>/
        """
        try:
            from .storage.fractal_store import export_hf_model_to_fractal
            task_dir = os.path.join(self.fractal_models_dir, str(task_type))
            os.makedirs(task_dir, exist_ok=True)
            return bool(export_hf_model_to_fractal(
                hf_model_dir_or_id=hf_model_dir_or_id,
                output_path=task_dir,
                model_id=model_id,
                device=str(kwargs.get('device', 'cpu')),
                fractal_levels=int(kwargs.get('fractal_levels', 4)),
                block_size=int(kwargs.get('block_size', 64)),
                local_files_only=bool(kwargs.get('local_files_only', True)),
            ))
        except Exception as e:
            self.logger.error(f"Ошибка экспорта модели в фрактальное хранилище: {e}", exc_info=True)
            return False
            
    def scan_models_directory(self) -> Optional[int]:
        """Сканирует директорию моделей и возвращает количество найденных моделей.
        
        Returns:
            Optional[int]: Количество найденных моделей или None в случае ошибки
        """
        try:
            models_found = 0
            
            # Сканируем директорию кэша моделей
            cache_dir = getattr(self, 'cache_dir', None)
            if cache_dir and os.path.exists(cache_dir):
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        # Проверяем наличие файлов модели
                        model_files = [f for f in os.listdir(item_path) 
                                     if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
                        if model_files:
                            models_found += 1
                            self.logger.debug(f"Найдена модель в директории: {item}")
            
            # Сканируем фрактальную директорию моделей
            fractal_models_dir = getattr(self, 'fractal_models_dir', None)
            if fractal_models_dir and os.path.exists(fractal_models_dir):
                for item in os.listdir(fractal_models_dir):
                    item_path = os.path.join(fractal_models_dir, item)
                    if os.path.isdir(item_path):
                        # Проверяем наличие индексного файла
                        if os.path.exists(os.path.join(item_path, 'index.json')):
                            models_found += 1
                            self.logger.debug(f"Найдена фрактальная модель: {item}")
            
            self.logger.info(f"Сканирование завершено. Найдено моделей: {models_found}")
            return models_found
            
        except Exception as e:
            self.logger.error(f"Ошибка при сканировании директории моделей: {e}", exc_info=True)
            return None
            
    def _save_to_fractal_store(self, model_name: str, model: Any, tokenizer: Any) -> None:
        """Сохраняет модель и токенизатор в фрактальное хранилище.
        
        Args:
            model_name: Имя модели
            model: Загруженная модель
            tokenizer: Токенизатор для модели
        """
        if self.fractal_store is None:
            return

        if not hasattr(self.fractal_store, 'store'):
            return
            
        try:
            # Подготавливаем данные для сохранения
            model_data = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name,
                'timestamp': time.time()
            }
            
            # Сохраняем в хранилище
            self.fractal_store.store(f"model_{model_name}", model_data)
            self.logger.debug(f"Модель '{model_name}' сохранена в фрактальное хранилище")
            
        except Exception as e:
            self.logger.warning(
                f"Не удалось сохранить модель '{model_name}' в фрактальное хранилище: {e}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
    
    def register_models_with_core_brain(self) -> None:
        """Регистрирует модели в CoreBrain для ленивой загрузки."""
        if not hasattr(self.brain, 'register_model_initializer'):
            self.logger.warning("CoreBrain не поддерживает регистрацию моделей")
            return

        # В текущей архитектуре каноничный путь получения модели — через get_model_for_task,
        # поэтому здесь регистрируем только ленивые обёртки, которые вызывают get_model_for_task.
            
        # Регистрируем модели из конфигурации
        for model_name, config in self.model_configs.items():
            try:
                # Получаем зависимости модели из конфигурации
                dependencies = config.get('dependencies', [])
                model_class = config.get('model_class', 'AutoModel')
                tokenizer_class = config.get('tokenizer_class', 'AutoTokenizer')
                
                # Создаем замыкание для ленивой загрузки модели
                def make_model_initializer(name=model_name):
                    def initializer():
                        return self.get_model_for_task(name)

                    return initializer
                        
                # Регистрируем модель в CoreBrain
                self.brain.register_model_initializer(
                    model_name=model_name,
                    initializer=make_model_initializer(model_name),
                    dependencies=dependencies
                )
                
                self.logger.info(f"Модель '{model_name}' зарегистрирована в CoreBrain")
                
            except Exception as e:
                self.logger.error(f"Ошибка при регистрации модели '{model_name}': {e}", exc_info=True)
                
    def cleanup(self) -> None:
        """Очищает ресурсы, используемые менеджером моделей."""
        try:
            # Очищаем кэш моделей
            for model_name, (model, tokenizer, _) in self._model_cache.items():
                try:
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    del model
                    
                    if hasattr(tokenizer, 'save_pretrained'):
                        tokenizer.save_pretrained('.')
                    del tokenizer
                    
                    self.logger.debug(f"Модель {model_name} выгружена из памяти")
                except Exception as e:
                    self.logger.error(f"Ошибка при выгрузке модели {model_name}: {e}", exc_info=True)
            
            # Очищаем кэш
            self._model_cache.clear()
            
            # Вызываем cleanup родительского класса
            super().cleanup()
            
            self.logger.info("Ресурсы ModelManager успешно освобождены")
            
        except Exception as e:
            self.logger.error(f"Ошибка при очистке ресурсов: {e}", exc_info=True)
            raise

"""
Model Manager для работы с моделями в CogniFlex.

Этот модуль предоставляет интерфейс для работы с моделями через CoreBrain.
"""
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, TypeVar

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoConfig = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

# Импортируем базовый класс компонента
try:
    from cogniflex.core.base_component import BaseComponent
except ImportError:
    # Для обратной совместимости
    BaseComponent = object
    logger = logging.getLogger("cogniflex.model_manager")
else:
    logger = logging.getLogger("cogniflex.model_manager")

# Тип для аннотаций
T = TypeVar('T')

class ModelManager(BaseComponent):
    """
    Менеджер моделей для работы с CoreBrain.
    
    Основные возможности:
    - Загрузка моделей через CoreBrain
    - Регистрация инициализаторов моделей
    - Управление зависимостями между моделями
    """
    
    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Инициализация менеджера моделей.
        
        Args:
            brain: Ссылка на ядро системы
            config: Конфигурация менеджера моделей
            **kwargs: Дополнительные параметры для совместимости
        """
        super().__init__(brain, config or {})
        
        # Устанавливаем логгер
        self.logger = logger.getChild('ModelManager')
        
        # Конфигурация моделей по умолчанию
        self.model_configs = {
            'text-generation': {
                'dependencies': [],
                'model_class': 'AutoModelForCausalLM',
                'tokenizer_class': 'AutoTokenizer',
            },
            'fractal-text-generation': {
                'dependencies': [],
                'model_class': 'FractalTransformer',
                'tokenizer_class': 'ExtendedFractalTokenizer',
                'config_class': 'FractalConfig',
                'use_fractal_storage': True,
                'memory_slots': 32,
                'memory_size': 512,
            },
            'text-embedding': {
                'dependencies': ['text-generation'],
                'model_class': 'AutoModel',
                'tokenizer_class': 'AutoTokenizer',
            },
        }
        
        # Обновляем конфигурацию из переданных параметров
        if 'model_configs' in self.config:
            self.model_configs.update(self.config['model_configs'])
        
        # Инициализируем кэш моделей
        self._model_cache: Dict[str, Any] = {}
        
        # Директория, где лежат модели в фрактальном формате (поддиректории по task)
        default_path = os.path.join(os.path.dirname(__file__), '..', 'core', 'cogniflex_cache', 'ml_unit', 'fractal_storage', 'models')
        self.fractal_models_dir: str = os.path.abspath(
            str(kwargs.get('model_dir') or self.config.get('model_dir') or self.config.get('models_dir') or default_path)
        )

        # Инициализируем фрактальное хранилище
        self.fractal_store = None
        self._init_fractal_store()
        
        # Определяем зависимости
        self._required_components = []
        self._optional_components = []
        
        self.logger.debug("ModelManager инициализирован")
        
    def _init_fractal_store(self, deferred_system=None):
        """Инициализирует фрактальное хранилище для моделей с отложенной инициализацией.
        
        Args:
            deferred_system: Система отложенных команд для асинхронной инициализации
        """
        def _init_store():
            try:
                from .storage.fractal_store import FractalWeightStore

                store_config = self.config.get('fractal_store', {})
                block_size = int(store_config.get('block_size', 64))
                fractal_levels = int(store_config.get('fractal_levels', 4))
                containers_per_group = int(store_config.get('containers_per_group', 4))

                device = 'cpu'
                try:
                    if torch is not None and torch.cuda.is_available():
                        device = 'cuda'
                except Exception:
                    device = 'cpu'

                self.fractal_store = FractalWeightStore(
                    block_size=block_size,
                    fractal_levels=fractal_levels,
                    containers_per_group=containers_per_group,
                    device=device,
                )

                self.logger.info(f"Фрактальное хранилище моделей (FractalWeightStore) инициализировано на устройстве: {device}")

                if hasattr(self, 'brain') and hasattr(self.brain, 'register_component'):
                    try:
                        self.brain.register_component('fractal_weight_store', self.fractal_store)
                    except Exception:
                        pass

                return True
                    
            except ImportError as e:
                error_msg = f"Не удалось импортировать FractalWeightStore: {e}"
                self.logger.warning(error_msg)
                return False
            except Exception as e:
                error_msg = f"Ошибка при инициализации фрактального хранилища: {e}"
                self.logger.error(error_msg, exc_info=True)
                return False
        
        # Если передан deferred_system, используем отложенную инициализацию
        if deferred_system is not None and hasattr(deferred_system, 'defer_command'):
            # Проверяем, доступен ли torch для импорта
            def check_torch_available():
                try:
                    import torch
                    return True
                except ImportError:
                    return False
            
            # Откладываем инициализацию с проверкой доступности torch
            deferred_system.defer_command(
                _init_store,
                priority='critical',
                condition=check_torch_available,
                retries=3,
                delay=5,
                name='init_fractal_store'
            )
            self.logger.info("Отложенная инициализация фрактального хранилища запланирована")
        else:
            # Прямая синхронная инициализация (для обратной совместимости)
            return _init_store()
    
    def _setup_component(self) -> None:
        """Настраивает компонент после проверки зависимостей."""
        # Регистрируем модели в CoreBrain
        self.register_models_with_core_brain()
        self.logger.info("ModelManager готов к работе")
    
    def get_model_for_task(self, task_type: str, model_name: Optional[str] = None, **kwargs) -> Tuple[Any, Any, str]:
        """
        Получает модель и токенизатор для указанной задачи.
        
        Args:
            task_type: Тип задачи (например, 'text-generation', 'fractal-text-generation')
            model_name: Имя модели (опционально)
            **kwargs: Дополнительные параметры для инициализации модели
            
        Returns:
            Кортеж (модель, токенизатор, имя_модели)
        """
        model_name = model_name or str(task_type)

        if model_name in self._model_cache and not kwargs.get('force_reload', False):
            return self._model_cache[model_name]

        if AutoConfig is None or AutoModelForCausalLM is None:
            self.logger.error("transformers недоступен: невозможно собрать HF-модель из фрактального хранилища")
            return None, None, None

        # Ожидаем структуру:
        #   <fractal_models_dir>/<task_type>/index.json + (shards_manifest.jsonl|containers.jsonl|data/) + config.json + tokenizer/
        task_dir = os.path.join(self.fractal_models_dir, str(task_type))
        if not os.path.isdir(task_dir):
            self.logger.error(
                f"Фрактальная модель для task='{task_type}' не найдена: {task_dir}. "
                f"Сначала экспортируйте модель (ruGPT-small) в это хранилище."
            )
            return None, None, None

        # Check if this is a regular HuggingFace model directory
        # Look for .safetensors or .bin files
        has_hf_weights = False
        for ext in ['.safetensors', '.bin']:
            if any(f.endswith(ext) for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))):
                has_hf_weights = True
                break
        
        if has_hf_weights:
            # Load as regular HuggingFace model
            self.logger.info(f"Загрузка regular HuggingFace модели из {task_dir}")
            try:
                # Load config and model
                cfg = AutoConfig.from_pretrained(task_dir, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(task_dir, config=cfg, local_files_only=True)
                
                # Load tokenizer
                tokenizer = None
                tok_dir = os.path.join(task_dir, 'tokenizer')
                try:
                    if AutoTokenizer is not None:
                        if os.path.isdir(tok_dir):
                            tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, use_fast=True)
                        else:
                            tokenizer = AutoTokenizer.from_pretrained(task_dir, local_files_only=True, use_fast=True)
                except Exception as e:
                    self.logger.warning(f"Не удалось загрузить токенизатор: {e}")
                    tokenizer = None
                
                if tokenizer is None:
                    tokenizer = getattr(self.brain, 'tokenizer', None)
                
                # Move to device
                device = 'cpu'
                try:
                    if torch is not None and torch.cuda.is_available() and bool(kwargs.get('use_gpu', True)):
                        device = 'cuda'
                except Exception:
                    device = 'cpu'
                
                if device == 'cuda' and torch is not None and torch.cuda.is_available():
                    model.to('cuda')
                else:
                    model.to('cpu')
                model.eval()
                
                result = (model, tokenizer, model_name)
                self._model_cache[model_name] = result
                
                if self.brain is not None:
                    try:
                        self.brain.register_component(f"model_{task_type}", model)
                        if tokenizer is not None:
                            self.brain.register_component(f"tokenizer_{task_type}", tokenizer)
                    except Exception:
                        pass
                
                return result
                
            except Exception as e:
                self.logger.error(f"Ошибка загрузки HuggingFace модели: {e}", exc_info=True)
                return None, None, None
        
        # Otherwise, try to load as fractal storage
        self.logger.info(f"Попытка загрузки фрактальной модели из {task_dir}")
        try:
            from .storage.fractal_store import FractalWeightStore

            store_config = self.config.get('fractal_store', {})
            block_size = int(store_config.get('block_size', 64))
            fractal_levels = int(store_config.get('fractal_levels', 4))
            containers_per_group = int(store_config.get('containers_per_group', 4))

            device = 'cpu'
            try:
                if torch is not None and torch.cuda.is_available() and bool(kwargs.get('use_gpu', True)):
                    device = 'cuda'
            except Exception:
                device = 'cpu'

            fs = FractalWeightStore(
                block_size=block_size,
                fractal_levels=fractal_levels,
                containers_per_group=containers_per_group,
                device=device,
            )

            # Шардированный формат читаем лениво (экономия RAM)
            fs.load_from_disk(task_dir, lazy=True)

            state_dict = fs.reconstruct_state_dict(output_dtype='float32', device='cpu')

            cfg = AutoConfig.from_pretrained(task_dir, local_files_only=True)
            model = AutoModelForCausalLM.from_config(cfg)
            model.load_state_dict(state_dict, strict=False)

            if device == 'cuda' and torch is not None and torch.cuda.is_available():
                model.to('cuda')
            else:
                model.to('cpu')
            model.eval()

            tokenizer = None
            tok_dir = os.path.join(task_dir, 'tokenizer')
            try:
                if AutoTokenizer is not None and os.path.isdir(tok_dir):
                    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, use_fast=True)
            except Exception:
                tokenizer = None
            if tokenizer is None:
                tokenizer = getattr(self.brain, 'tokenizer', None)

            result = (model, tokenizer, model_name)
            self._model_cache[model_name] = result

            if self.brain is not None:
                try:
                    self.brain.register_component(f"model_{task_type}", model)
                    if tokenizer is not None:
                        self.brain.register_component(f"tokenizer_{task_type}", tokenizer)
                except Exception:
                    pass

            return result
        except Exception as e:
            self.logger.error(
                f"Критическая ошибка при загрузке модели из фрактального хранилища (task='{task_type}'): {e}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
            return None, None, None

    def export_task_model_to_fractal(self, task_type: str, hf_model_dir_or_id: str, model_id: str, **kwargs) -> bool:
        """Явный экспорт HF-модели (ruGPT-small) в фрактальное хранилище под task.

        Пишет в: <fractal_models_dir>/<task_type>/
        """
        try:
            from .storage.fractal_store import export_hf_model_to_fractal
            task_dir = os.path.join(self.fractal_models_dir, str(task_type))
            os.makedirs(task_dir, exist_ok=True)
            return bool(export_hf_model_to_fractal(
                hf_model_dir_or_id=hf_model_dir_or_id,
                output_path=task_dir,
                model_id=model_id,
                device=str(kwargs.get('device', 'cpu')),
                fractal_levels=int(kwargs.get('fractal_levels', 4)),
                block_size=int(kwargs.get('block_size', 64)),
                local_files_only=bool(kwargs.get('local_files_only', True)),
            ))
        except Exception as e:
            self.logger.error(f"Ошибка экспорта модели в фрактальное хранилище: {e}", exc_info=True)
            return False
            
    def scan_models_directory(self) -> Optional[int]:
        """Сканирует директорию моделей и возвращает количество найденных моделей.
        
        Returns:
            Optional[int]: Количество найденных моделей или None в случае ошибки
        """
        try:
            models_found = 0
            
            # Сканируем директорию кэша моделей
            cache_dir = getattr(self, 'cache_dir', None)
            if cache_dir and os.path.exists(cache_dir):
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        # Проверяем наличие файлов модели
                        model_files = [f for f in os.listdir(item_path) 
                                     if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
                        if model_files:
                            models_found += 1
                            self.logger.debug(f"Найдена модель в директории: {item}")
            
            # Сканируем фрактальную директорию моделей
            fractal_models_dir = getattr(self, 'fractal_models_dir', None)
            if fractal_models_dir and os.path.exists(fractal_models_dir):
                for item in os.listdir(fractal_models_dir):
                    item_path = os.path.join(fractal_models_dir, item)
                    if os.path.isdir(item_path):
                        # Проверяем наличие индексного файла
                        if os.path.exists(os.path.join(item_path, 'index.json')):
                            models_found += 1
                            self.logger.debug(f"Найдена фрактальная модель: {item}")
            
            self.logger.info(f"Сканирование завершено. Найдено моделей: {models_found}")
            return models_found
            
        except Exception as e:
            self.logger.error(f"Ошибка при сканировании директории моделей: {e}", exc_info=True)
            return None
            
    def _save_to_fractal_store(self, model_name: str, model: Any, tokenizer: Any) -> None:
        """Сохраняет модель и токенизатор в фрактальное хранилище.
        
        Args:
            model_name: Имя модели
            model: Загруженная модель
            tokenizer: Токенизатор для модели
        """
        if self.fractal_store is None:
            return

        if not hasattr(self.fractal_store, 'store'):
            return
            
        try:
            # Подготавливаем данные для сохранения
            model_data = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name,
                'timestamp': time.time()
            }
            
            # Сохраняем в хранилище
            self.fractal_store.store(f"model_{model_name}", model_data)
            self.logger.debug(f"Модель '{model_name}' сохранена в фрактальное хранилище")
            
        except Exception as e:
            self.logger.warning(
                f"Не удалось сохранить модель '{model_name}' в фрактальное хранилище: {e}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
    
    def register_models_with_core_brain(self) -> None:
        """Регистрирует модели в CoreBrain для ленивой загрузки."""
        if not hasattr(self.brain, 'register_model_initializer'):
            self.logger.warning("CoreBrain не поддерживает регистрацию моделей")
            return

        # В текущей архитектуре каноничный путь получения модели — через get_model_for_task,
        # поэтому здесь регистрируем только ленивые обёртки, которые вызывают get_model_for_task.
            
        # Регистрируем модели из конфигурации
        for model_name, config in self.model_configs.items():
            try:
                # Получаем зависимости модели из конфигурации
                dependencies = config.get('dependencies', [])
                model_class = config.get('model_class', 'AutoModel')
                tokenizer_class = config.get('tokenizer_class', 'AutoTokenizer')
                
                # Создаем замыкание для ленивой загрузки модели
                def make_model_initializer(name=model_name):
                    def initializer():
                        return self.get_model_for_task(name)

                    return initializer
                        
                # Регистрируем модель в CoreBrain
                self.brain.register_model_initializer(
                    model_name=model_name,
                    initializer=make_model_initializer(model_name),
                    dependencies=dependencies
                )
                
                self.logger.info(f"Модель '{model_name}' зарегистрирована в CoreBrain")
                
            except Exception as e:
                self.logger.error(f"Ошибка при регистрации модели '{model_name}': {e}", exc_info=True)
                
    def get_available_models(self) -> list:
        """Возвращает список доступных моделей.
        
        Returns:
            list: Список доступных моделей
        """
        models = []
        
        try:
            # Сканируем фрактальную директорию моделей
            fractal_models_dir = getattr(self, 'fractal_models_dir', None)
            if fractal_models_dir and os.path.exists(fractal_models_dir):
                for item in os.listdir(fractal_models_dir):
                    item_path = os.path.join(fractal_models_dir, item)
                    if os.path.isdir(item_path):
                        # Проверяем наличие индексного файла
                        if os.path.exists(os.path.join(item_path, 'index.json')):
                            models.append({
                                'name': item,
                                'type': 'fractal',
                                'path': item_path,
                                'status': 'available'
                            })
                            self.logger.debug(f"Найдена фрактальная модель: {item}")
            
            self.logger.info(f"Найдено доступных моделей: {len(models)}")
            return models
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка моделей: {e}", exc_info=True)
            return []
    
    @property
    def model_metadata(self) -> dict:
        """Возвращает метаданные доступных моделей для совместимости с MLUnit.
        
        Returns:
            dict: Словарь с метаданными моделей
        """
        models = self.get_available_models()
        metadata = {}
        
        for model in models:
            metadata[model['name']] = {
                'type': model['type'],
                'path': model['path'],
                'status': model['status']
            }
        
        return metadata
    
    def cleanup(self) -> None:
        """Очищает ресурсы, используемые менеджером моделей."""
        try:
            # Очищаем кэш моделей
            for model_name, (model, tokenizer, _) in self._model_cache.items():
                try:
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    del model
                    
                    if hasattr(tokenizer, 'save_pretrained'):
                        tokenizer.save_pretrained('.')
                    del tokenizer
                    
                    self.logger.debug(f"Модель {model_name} выгружена из памяти")
                except Exception as e:
                    self.logger.error(f"Ошибка при выгрузке модели {model_name}: {e}", exc_info=True)
            
            # Очищаем кэш
            self._model_cache.clear()
            
            # Вызываем cleanup родительского класса
            super().cleanup()
            
            self.logger.info("Ресурсы ModelManager успешно освобождены")
            
        except Exception as e:
            self.logger.error(f"Ошибка при очистке ресурсов: {e}", exc_info=True)
            raise
