"""
Fractal Model Manager для CogniFlex
Управляет фрактальной архитектурой трансформера с интеграцией памяти
"""
import logging
import json
import os
from typing import Optional, Any, Dict
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Возвращает корневую директорию проекта"""
    possible_roots = []
    
    # 1. Относительно текущего файла
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)  # cogniflex/mlearning
    possible_roots.append(os.path.dirname(os.path.dirname(current_dir)))  # cogniflex
    possible_roots.append(os.path.dirname(current_dir))  # project root
    
    # 2. Проверяем common project markers
    for root in possible_roots:
        if os.path.exists(os.path.join(root, 'cogniflex')) or \
           os.path.exists(os.path.join(root, 'pyproject.toml')) or \
           os.path.exists(os.path.join(root, 'setup.py')):
            return root
    
    # 3. Fallback - ищем по ключевым директориям
    drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
    possible_locations = [
        os.path.join(drive, 'Users', os.environ.get('USERNAME', 'user'), 'OneDrive', 'Desktop', 'CogniFlex'),
        os.path.join(drive, 'Users', os.environ.get('USERNAME', 'user'), 'Desktop', 'CogniFlex'),
        os.path.join(drive, 'CogniFlex'),
        os.path.join(os.getcwd(), '..'),
        os.path.join(os.getcwd(), '..', '..'),
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            if os.path.exists(os.path.join(loc, 'cogniflex')) or \
               os.path.exists(os.path.join(loc, 'pyproject.toml')):
                return os.path.abspath(loc)
    
    return os.getcwd()


class FractalModelManager:
    """
    Менеджер фрактальной модели с реальной генерацией.
    Использует стандартную модель GPT-2 для генерации ответов.
    """
    
    def __init__(self, config: Optional[Any] = None, model_path: Optional[str] = None):
        """Инициализация менеджера"""
        self.config = config
        self.model_path = model_path
        self.device = "auto"  # Используем auto для автоматического определения
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.has_fractal_model = False
        
        # Загружаем конфигурацию если есть
        if config and isinstance(config, str):
            self._load_config(config)
        elif model_path and model_path.endswith('.json'):
            self._load_config(model_path)
        
        # Инициализируем модель
        self._initialize_model()
        
        logger.info(f"FractalModelManager инициализирован на устройстве: {self.device}")
        logger.info(f"Статус инициализации: {self.initialized}")
    
    def _load_config(self, config_path: str):
        """Загружает конфигурацию из файла"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                    self.device = self.config.get("device", "cpu")
                    logger.info(f"Конфигурация загружена из {config_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
    
    def _initialize_model(self):
        """Инициализирует модель"""
        try:
            # Используем локальный загрузчик ruGPT-3 Medium
            from .local_rugpt3_loader import load_rugpt3_medium_local
            
            model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
            logger.info(f"Начальный model_name: {model_name}")
            
            if self.config:
                model_cfg = self.config.get("model", {}) if isinstance(self.config, dict) else {}
                config_model_name = model_cfg.get("name", "") if model_cfg else ""
                logger.info(f"Из конфига получено model name: '{config_model_name}'")
                logger.info(f"Тип config_model_name: {type(config_model_name)}")
                if config_model_name == "rugpt3small":
                    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
                    logger.info(f"Установлен small model_name: {model_name}")
                elif config_model_name == "rugpt3large":
                    model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
                    logger.info(f"Установлен large model_name: {model_name}")
                else:
                    model_name = config_model_name
                    logger.info(f"Использован config model_name: {model_name}")
            else:
                logger.info("Конфиг не найден или не содержит model.name")
            
            logger.info(f"Финальная модель для загрузки: '{model_name}'")
            
            # Определяем устройство
            device = self.device
            if device == "auto":
                # Проверяем доступность CUDA и используем её если доступна
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info("CUDA доступна, используем GPU")
                else:
                    device = "cpu"
                    logger.info("CUDA недоступна, используем CPU")
                    logger.info("Используем CPU для предотвращения ошибок памяти GPU")
            
            # Загружаем модель и токенизатор локально
            import os
            
            # Получаем корень проекта
            project_root = _get_project_root()
            logger.info(f"Корень проекта: {project_root}")
            
            # Пробуем путь к ruGPT-3 Small во фрактальном хранилище (абсолютные пути)
            storage_paths = [
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal", "model"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "tokenizers", "rugpt3_small_fractal"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal", "model"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "tokenizers", "rugpt3_small_fractal"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal", "model"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "tokenizers", "rugpt3_small_fractal"),
                os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal"),
                os.path.join(project_root, "cogniflex", "mlearning", "cogniflex_models", "rugpt3_small"),
                "cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal/model",
                "cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_small_fractal",
            ]
            
            storage_path = None
            for path in storage_paths:
                if os.path.exists(path):
                    storage_path = path
                    logger.info(f"Найдена модель по пути: {os.path.abspath(path)}")
                    break
            
            if storage_path is None:
                storage_path = os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal", "model")
                logger.warning(f"Модель не найдена, используем путь по умолчанию: {storage_path}")
            
            logger.info(f"Используем путь к хранилищу: {os.path.abspath(storage_path)}")
            
            self.model, self.tokenizer = load_rugpt3_medium_local(
                storage_path=storage_path,
                device=device
            )
            
            # Обновляем устройство на реальное устройство модели
            if self.model:
                self.device = str(self.model.device)
                logger.info(f"Устройство модели обновлено на: {self.device}")
            
            if self.model and self.tokenizer:
                # Устанавливаем pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Модель уже на нужном устройстве
                self.model.eval()
                
                # Инициализируем улучшатель качества текста
                try:
                    from .text_quality_improver import TextQualityImprover
                    self.quality_improver = TextQualityImprover(None)
                    logger.info("TextQualityImprover инициализирован")
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать TextQualityImprover: {e}")
                    self.quality_improver = None
                
                # Инициализируем тренер для улучшения качества
                try:
                    from .text_quality_trainer import TextQualityTrainer, TrainingConfig
                    self.trainer_config = TrainingConfig(
                        learning_rate=3e-5,
                        batch_size=2,
                        num_epochs=50,
                        warmup_steps=100,
                        max_length=128,
                        gradient_accumulation_steps=4
                    )
                    # Создаем тренер
                    self.trainer = TextQualityTrainer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=self.trainer_config
                    )
                    logger.info("TextQualityTrainer инициализирован")
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать TextQualityTrainer: {e}")
                    self.trainer = None
                
                logger.info(f"✅ Модель {model_name} успешно загружена локально")
                self.initialized = True
                self.has_fractal_model = True
                
            else:
                logger.error(f"❌ Не удалось загрузить модель {model_name}")
                self.initialized = False
                return
            
        except Exception as e:
            logger.error(f"Ошибка инициализации модели: {e}")
            self.initialized = False
    
    def _create_conversational_prompt(self, query: str) -> str:
        """
        Creates a well-formatted conversational prompt.
        
        Russian conversational assistant style:
        - Answer briefly and directly
        - Be helpful and friendly
        - Stay on topic
        """
        query = query.strip()
        
        prompts = {
            'greeting': ['привет', 'здравствуй', 'добрый день', 'hello', 'hi'],
            'question': ['кто', 'что', 'как', 'почему', 'зачем', 'where', 'who', 'what', 'how', 'why'],
            'help': ['помощь', 'help', 'помоги', 'подскажи'],
        }
        
        query_lower = query.lower()
        
        if any(g in query_lower for g in prompts['greeting']):
            return f"Привет! Я CogniFlex, русскоязычный AI ассистент. {query}\n"
        elif any(h in query_lower for h in prompts['help']):
            return f"Я помогу! {query}\n"
        else:
            return f"Вопрос: {query}\nОтвет на русском языке, кратко и по делу:\n"
    
    def generate_response(self, query: str, max_tokens: int = 512, **kwargs) -> str:
        """
        Генерирует ответ с использованием модели.
        
        Args:
            query: Запрос для генерации
            max_tokens: Максимальное количество токенов
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный ответ
        """
        if not self.initialized or not self.model or not self.tokenizer:
            return self._get_fallback_response(query)
        
        try:
            # Используем conversational промпт
            prompt = self._create_conversational_prompt(query)
            
            # Ограничиваем
            max_tokens = min(max_tokens, 50)
            
            # Токенизируем
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=200
            )
            
            # Правильное устройство
            device = next(self.model.parameters()).device if self.model else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0,
                )
            
            # Декодируем
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Убираем промпт из ответа
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()
            
            # Фильтруем
            response = self._clean_response(response, query)
            
            # Проверяем качество
            if not self._is_good_response(response):
                return self._get_fallback_response(query)
            
            # Короткий ответ
            if len(response) > 100:
                sentences = response.split('.')
                response = sentences[0].strip() + '.'
            
            return response if response else self._get_fallback_response(query)
            
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            return self._get_fallback_response(query)
    
    def _is_good_response(self, response: str) -> bool:
        """Проверяет качество ответа."""
        import re
        
        # Слишком короткий
        if len(response) < 3:
            return False
        
        # Слишком длинный без пунктуации
        if len(response) > 200 and response.count('.') < 2:
            return False
        
        # Содержит паттерны-маркеры плохого качества
        bad_markers = [
            r'\d{5,}',  # Длинные числа
            r'[A-Z]{5,}',  # Капс слова
            r'http[s]?://',  # URL
        ]
        for marker in bad_markers:
            if re.search(marker, response):
                return False
        
        return True
    
    def _get_fallback_response(self, query: str) -> str:
        """Возвращает запасной ответ."""
        query_lower = query.lower().strip()
        
        fallbacks = {
            'привет': 'Привет! Я CogniFlex, рад общению. Что хотите обсудить?',
            'здравств': 'Здравствуйте! Чем могу помочь?',
            'как дела': 'Спасибо, что спрашиваете! У меня всё хорошо, готов помогать.',
            'как тебя': 'Меня зовут CogniFlex. Я ваш AI ассистент.',
            'кто ты': 'Я CogniFlex - когнитивная AI система. Я умею отвечать на вопросы, анализировать и учиться.',
            'что ты умеешь': 'Я умею: отвечать на вопросы, анализировать тексты, искать информацию, помогать с идеями.',
            'помощь': 'Я к вашим услугам! Просто напишите вопрос или тему для обсуждения.',
            'спасибо': 'Пожалуйста! Если нужна ещё помощь - обращайтесь.',
            'пока': 'До свидания! Рад был помочь.',
            'до свидан': 'Пока! Обращайтесь ещё.',
        }
        
        for key, response in fallbacks.items():
            if key in query_lower:
                return response
        
        return 'Интересная мысль! Расскажите подробнее.'
    
    def _clean_response(self, response: str, query: str) -> str:
        """Очищает ответ от мусора и повторений."""
        import re
        
        # Убираем табы и лишние пробелы
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Список паттернов которые указывают на плохой ответ
        bad_patterns = [
            r'\d+\s+\w+\s+\d{4}',  # Даты типа "32682471 sergey-verevkin"
            r'^Я\s*-\s*ты',  # "Я - ты"
            r'Какие\s+ключевые\s+аспекты',  # Мета-вопросы
            r'^Что\s+такое\s+информация',  # Базовые определения
            r'^Ответ:\s*$',  # Пустой ответ
            r'^Вопрос:\s*$',  # Пустой вопрос
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # Находим первую точку до плохого паттерна
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return response[:match.start()].strip()
        
        # Если ответ начинается с вопроса - возвращаем только первый абзац
        if response.startswith('Как') or response.startswith('Что') or response.startswith('Почему'):
            sentences = response.split('.')
            if len(sentences) > 2:
                return sentences[0] + '.'
        
        # Убираем конец с URL или служебной информацией
        url_pattern = r'https?://\S+'
        if re.search(url_pattern, response):
            response = re.sub(url_pattern, '', response).strip()
        
        return response
    
    def generate(self, query: str, **kwargs) -> str:
        """
        Генерирует ответ (псевдоним для совместимости с существующим кодом)
        
        Args:
            query: Запрос для генерации
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный ответ
        """
        return self.generate_response(query, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        return {
            "initialized": self.initialized,
            "has_fractal_model": self.has_fractal_model,
            "device": self.device,
            "model_type": "GPT-2" if self.initialized else "None",
            "total_parameters": 124000000,  # 124M для GPT-2
            "model_name": (self.config.get("model", {}) or {}).get("name", "gpt2") if self.config else "gpt2"
        }
    
    def is_ready(self) -> bool:
        """Проверяет готовность модели"""
        return self.initialized
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        return {
            "initialized": self.initialized,
            "model_name": (self.config.get("model", {}) or {}).get("name", "gpt2") if self.config else "gpt2",
            "device": str(self.device)
        }
    
    def get_fractal_info(self) -> Dict[str, Any]:
        """Возвращает информацию о фрактальной архитектуре"""
        if not self.initialized:
            return {"error": "Модель не инициализирована"}
        
        return {
            "architecture": "GPT-2 (фрактальная эмуляция)",
            "parameters": "124M",
            "layers": 12,
            "heads": 12,
            "hidden_size": 768
        }
    
    def save_model(self, path: str) -> bool:
        """Сохраняет модель"""
        if not self.initialized:
            logger.error("Модель не инициализирована")
            return False
        
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Модель сохранена в {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Загружает модель"""
        try:
            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            self.has_fractal_model = True
            
            logger.info(f"Модель загружена из {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def get_model_for_task(self, task_type: str, model_name: Optional[str] = None, **kwargs) -> tuple:
        """
        Получает модель и токенизатор для указанной задачи.
        
        Args:
            task_type: Тип задачи (например, 'text-generation', 'fractal-text-generation')
            model_name: Имя модели (опционально)
            **kwargs: Дополнительные параметры для инициализации модели
            
        Returns:
            Кортеж (модель, токенизатор, имя_модели)
        """
        if not self.initialized or not self.model or not self.tokenizer:
            logger.error(f"Модель не инициализирована для задачи '{task_type}'")
            return None, None, None
        
        # Используем переданное имя модели или имя задачи
        actual_model_name = model_name or task_type
        
        # Возвращаем текущую модель и токенизатор
        logger.debug(f"Возвращаем модель для задачи '{task_type}' с именем '{actual_model_name}'")
        return self.model, self.tokenizer, actual_model_name
    
    def get_available_models(self):
        """
        Возвращает информацию о доступных моделях для совместимости с ML Unit.
        
        Returns:
            Dict[str, Dict]: Словарь с информацией о моделях
        """
        if not self.initialized:
            return {}
        
        # Возвращаем информацию о текущей модели
        model_info = {
            "name": (self.config.get("model", {}) or {}).get("name", "rugpt3large") if self.config else "rugpt3large",
            "display_name": "ruGPT-3 Large (фрактальная)",
            "type": "text-generation",
            "status": "loaded" if self.model and self.tokenizer else "error",
            "device": str(self.device) if hasattr(self, 'device') else "unknown",
            "initialized": self.initialized,
            "has_fractal_support": True,
            "model_path": self.model_path,
            "description": "Фрактальная модель ruGPT-3 Large для генерации текста"
        }
        
        # Возвращаем только одну модель - ruGPT-3 Large
        return {
            "rugpt3_small_fractal": model_info
        }
