"""
Fractal Model Manager для ЕВА
Управляет фрактальной архитектурой трансформера с интеграцией памяти
Текстовая генерация теперь через Qwen3.5-2B
"""
import logging
import json
import os
from typing import Optional, Any, Dict
import torch

logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Возвращает корневую директорию проекта"""
    possible_roots = []
    
    # 1. Относительно текущего файла
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)  # eva/mlearning
    possible_roots.append(os.path.dirname(os.path.dirname(current_dir)))  # eva
    possible_roots.append(os.path.dirname(current_dir))  # project root
    
    # 2. Проверяем common project markers
    for root in possible_roots:
        if os.path.exists(os.path.join(root, 'eva')) or \
           os.path.exists(os.path.join(root, 'pyproject.toml')) or \
           os.path.exists(os.path.join(root, 'setup.py')):
            return root
    
    # 3. Fallback - ищем по ключевым директориям
    drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
    possible_locations = [
        os.path.join(drive, 'Users', os.environ.get('USERNAME', 'user'), 'OneDrive', 'Desktop', 'ЕВА'),
        os.path.join(drive, 'Users', os.environ.get('USERNAME', 'user'), 'Desktop', 'ЕВА'),
        os.path.join(drive, 'ЕВА'),
        os.path.join(os.getcwd(), '..'),
        os.path.join(os.getcwd(), '..', '..'),
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            if os.path.exists(os.path.join(loc, 'eva')) or \
               os.path.exists(os.path.join(loc, 'pyproject.toml')):
                return os.path.abspath(loc)
    
    return os.getcwd()


class FractalModelManager:
    """
    Менеджер фрактальной модели с реальной генерацией.
    Теперь с поддержкой GGUF/LlamaCpp для быстрого CPU inference.
    """
    
    def __init__(self, config: Optional[Any] = None, model_path: Optional[str] = None):
        """Инициализация менеджера"""
        self.config = config
        self.model_path = model_path
        self.device = "auto"
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.has_fractal_model = False
        
        # GGUF/LlamaCpp support
        self.llama_cpp_deployment = None
        self.llama_cpp_ready = False
        
        # Загружаем конфигурацию если есть
        if config and isinstance(config, str):
            self._load_config(config)
        elif model_path and model_path.endswith('.json'):
            self._load_config(model_path)
        
        # Инициализируем GGUF (быстрее чем PyTorch)
        self._initialize_llama_cpp()
        
        # Инициализируем PyTorch модель (lazy loading)
        self._initialize_model()
        
        logger.info(f"FractalModelManager инициализирован на устройстве: {self.device}")
        logger.info(f"Статус инициализации: {self.initialized}")
        logger.info(f"GGUF ready: {self.llama_cpp_ready}")
    
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
    
    def _initialize_llama_cpp(self):
        """Инициализирует GGUF/LlamaCpp для быстрого CPU inference"""
        try:
            from eva_ai.mlearning.hot_deployment.llama_cpp_hot import LlamaCppHotDeployment
            
            project_root = _get_project_root()
            
            # Проверяем несколько путей для GGUF
            possible_paths = [
                os.path.join(project_root, "eva_ai", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-0.5b-instruct-q4_0.gguf"),
                os.path.join(project_root, "eva_ai", "models", "qwen2.5-0.5b-instruct-q4_0.gguf"),
                os.path.join(project_root, "models", "qwen2.5-0.5b-instruct-q4_0.gguf"),
            ]
            
            gguf_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    gguf_path = p
                    logger.info(f"Найден GGUF файл: {p}")
                    break
            
            if gguf_path is None:
                logger.warning("GGUF model not found in any path")
                return
            
            self.llama_cpp_deployment = LlamaCppHotDeployment(
                model_path=gguf_path,
                n_ctx=2048,
                n_threads=os.cpu_count() or 12
            )
            
            if self.llama_cpp_deployment.initialize(preload_root=True):
                self.llama_cpp_ready = True
                logger.info(f"FractalModelManager GGUF ready: {gguf_path}")
            else:
                logger.warning("FractalModelManager: LlamaCpp initialization failed")
                
        except Exception as e:
            logger.debug(f"FractalModelManager GGUF не инициализирован: {e}")
            self.llama_cpp_deployment = None
    
    def _initialize_model(self):
        """Инициализирует модель - теперь через Qwen3.5-2B"""
        try:
            project_root = _get_project_root()
            qwen_path = os.path.join(project_root, "eva_ai", "mlearning", "eva_models", "qwen2.5-0.5b")
            
            logger.info(f"Qwen3.5-2B path: {qwen_path}")
            logger.info("Текстовая генерация теперь через Qwen3.5-2B (lazy loading)")
            
            self.model = None
            self.tokenizer = None
            self.initialized = True
            self.has_fractal_model = False
            
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
            'greeting': ['привет', 'здравствуй', 'добрый день', 'hello', 'hi', 'хай', 'здорово'],
            'question': ['кто', 'что', 'как', 'почему', 'зачем', 'where', 'who', 'what', 'how', 'why'],
            'help': ['помощь', 'help', 'помоги', 'подскажи'],
        }
        
        query_lower = query.lower()
        
        # Для приветствий используем очень короткий промпт
        if any(g in query_lower for g in prompts['greeting']):
            return f"Привет! Ответь кратко: "
        elif any(h in query_lower for h in prompts['help']):
            return f"Помощь с: {query}. Краткий ответ:"
        else:
            return f"Вопрос: {query}\nКраткий ответ:"
    
    def generate_response(self, query: str, max_new_tokens: int = 2048, **kwargs) -> str:
        """
        Генерирует ответ с использованием модели.
        Приоритет: GGUF/LlamaCpp > PyTorch > Fallback
        """
        # Приоритет: GGUF/LlamaCpp
        if self.llama_cpp_ready and self.llama_cpp_deployment:
            try:
                prompt = self._create_conversational_prompt(query)
                response_text = self.llama_cpp_deployment.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens or 100,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1
                )
                
                if response_text and len(response_text) > 5:
                    return response_text
            except Exception as e:
                logger.debug(f"GGUF generation error: {e}")
        
        # Fallback to PyTorch if available
        if not self.initialized or not self.model or not self.tokenizer:
            return self._get_fallback_response(query)
        
        try:
            # Используем conversational промпт
            prompt = self._create_conversational_prompt(query)
            
            # Для приветствий используем только fallback (модель не обучена для диалогов)
            query_lower = query.lower()
            greeting_words = ['привет', 'здравствуй', 'добрый', 'hello', 'hi', 'хай', 'здорово']
            if any(g in query_lower for g in greeting_words):
                return self._get_fallback_response(query)
            
            # Ограничиваем генерацию
            max_new_tokens = min(max_new_tokens, 2048)  # Allow up to 2048 tokens per DESIGN.md
            
            # Токенизируем
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=32768
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
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0,
                    repetition_penalty=1.1,
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
        if len(response) < 10:
            return False
        
        # Слишком длинный без пунктуации
        if len(response) > 200 and response.count('.') < 2:
            return False
        
        # Слишком много точек подряд (признак плохой генерации)
        if '......' in response or '....' in response:
            return False
        
        # Слишком много повторений символов
        if re.search(r'(.)\1{4,}', response):  # 5+ повторений одного символа
            return False
        
        # Содержит паттерны-маркеры плохого качества
        bad_markers = [
            r'\d{5,}',  # Длинные числа
            r'[A-Z]{5,}',  # Капс слова
            r'http[s]?://',  # URL
            r'^\s*-{3,}',  # Много дефисов в начале
            r'\.{5,}',  # Много точек подряд
        ]
        for marker in bad_markers:
            if re.search(marker, response):
                return False
        
        # Проверка на бессмысленное повторение (более 30% текста - повторение)
        words = response.split()
        if len(words) > 10:
            unique_words = len(set(w.lower() for w in words))
            if unique_words / len(words) < 0.3:  # Менее 30% уникальных слов
                return False
        
        return True
    
    def _get_fallback_response(self, query: str) -> str:
        """Возвращает запасной ответ."""
        query_lower = query.lower().strip()
        
        fallbacks = {
            'привет': 'Привет! Я ЕВА, рада общению. Чем могу помочь?',
            'здравств': 'Здравствуйте! Чем могу помочь?',
            'как дела': 'Спасибо, что спрашиваете! У меня всё хорошо, готова помогать.',
            'как тебя': 'Меня зовут ЕВА. Я ваш AI ассистент.',
            'кто ты': 'Я ЕВА - когнитивная AI система. Я умею отвечать на вопросы, анализировать и учиться.',
            'что ты умеешь': 'Я умею: отвечать на вопросы, анализировать тексты, искать информацию, помогать с идеями.',
            'помощь': 'Я к вашим услугам! Просто напишите вопрос или тему для обсуждения.',
            'спасибо': 'Пожалуйста! Если нужна ещё помощь - обращайтесь.',
            'пока': 'До свидания! Рада была помочь.',
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
            "model_name": (self.config.get("model", {}) or {}).get("name", "qwen2.5-0.5b") if self.config else "qwen2.5-0.5b"
        }
    
    def is_ready(self) -> bool:
        """Проверяет готовность модели"""
        return self.initialized
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        return {
            "initialized": self.initialized,
            "model_name": (self.config.get("model", {}) or {}).get("name", "qwen2.5-0.5b") if self.config else "qwen2.5-0.5b",
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
        
        if self.model is None or self.tokenizer is None:
            logger.warning("Модель или токенизатор не загружены, пропускаем сохранение")
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
        """Загружает модель - теперь через QwenModelManager"""
        logger.info("Текстовая генерация теперь через QwenModelManager, не FractalModelManager")
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
            "name": (self.config.get("model", {}) or {}).get("name", "qwen2.5-0.5b") if self.config else "qwen2.5-0.5b",
            "display_name": "Qwen3.5-0.8B",
            "type": "text-generation",
            "status": "loaded" if self.model and self.tokenizer else "error",
            "device": str(self.device) if hasattr(self, 'device') else "unknown",
            "initialized": self.initialized,
            "has_fractal_support": True,
            "model_path": self.model_path,
            "description": "Qwen3.5-0.8B для генерации текста"
        }
        
        # Возвращаем только одну модель - Qwen
        return {
            "qwen2.5-0.5b_fractal": model_info
        }
    
    def unload(self) -> bool:
        """Активная выгрузка всех моделей из памяти."""
        try:
            # Выгружаем GGUF модель
            if self.llama_cpp_deployment is not None:
                if hasattr(self.llama_cpp_deployment, 'unload'):
                    self.llama_cpp_deployment.unload()
                self.llama_cpp_deployment = None
            self.llama_cpp_ready = False
            
            # Выгружаем PyTorch модель
            if self.model is not None:
                # Перемещаем на CPU перед удалением
                if hasattr(self.model, 'cpu'):
                    try:
                        self.model.cpu()
                    except Exception:
                        pass
                # Для мгновенного освобождения VRAM
                if hasattr(self.model, 'to'):
                    try:
                        self.model.to('meta')
                    except Exception:
                        pass
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            self.initialized = False
            self.has_fractal_model = False
            
            # Очистка VRAM и RAM
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            
            import gc
            gc.collect()
            
            logger.info("FractalModelManager: все модели выгружены из памяти")
            return True
        except Exception as e:
            logger.error(f"Ошибка выгрузки FractalModelManager: {e}")
            return False
    
    def __del__(self):
        """Деструктор — автоматическая выгрузка при удалении объекта."""
        try:
            if self.llama_cpp_deployment is not None:
                if hasattr(self.llama_cpp_deployment, 'unload'):
                    self.llama_cpp_deployment.unload()
            if self.model is not None:
                del self.model
        except Exception:
            pass
