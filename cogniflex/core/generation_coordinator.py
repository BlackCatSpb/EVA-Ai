#!/usr/bin/env python3
"""
Унифицированный координатор генерации текста CogniFlex
Обеспечивает единую точку входа/выхода для всех модулей
"""

import time
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class GenerationRequest:
    """Стандартизированный запрос на генерацию"""
    
    def __init__(self, text: str, **kwargs):
        self.text = text
        self.max_tokens = kwargs.get('max_tokens', 150)  # Уменьшено для скорости
        self.temperature = kwargs.get('temperature', 0.8)
        self.context = kwargs.get('context')
        self.user_context = kwargs.get('user_context')
        self.source = kwargs.get('source', 'unknown')
        self.priority = kwargs.get('priority', 'normal')
        self.metadata = kwargs.get('metadata', {})
        self.do_sample = kwargs.get('do_sample', False)  # Greedy для стабильности


class GenerationResponse:
    """Стандартизированный ответ от генерации"""
    
    def __init__(self, text: str, status: str = "ok", **kwargs):
        self.text = text
        self.status = status
        self.source = kwargs.get('source', 'unknown')
        self.model_name = kwargs.get('model_name', 'unknown')
        self.tokens_generated = kwargs.get('tokens_generated', 0)
        self.generation_time = kwargs.get('generation_time', 0.0)
        self.error_message = kwargs.get('error_message')
        self.metadata = kwargs.get('metadata', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует ответ в словарь"""
        result = {
            "text": self.text,
            "status": self.status,
            "source": self.source,
            "model_name": self.model_name,
            "tokens_generated": self.tokens_generated,
            "generation_time": self.generation_time,
            "timestamp": time.time(),
            "metadata": self.metadata
        }
        
        if self.error_message:
            result["error"] = self.error_message
            
        return result


class GenerationProvider(ABC):
    """Абстрактный провайдер генерации"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверяет доступность провайдера"""
        pass
    
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Генерирует ответ"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Возвращает приоритет провайдера (меньше = выше приоритет)"""
        pass


class HybridModelProvider(GenerationProvider):
    """Провайдер на основе гибридного менеджера моделей"""
    
    def __init__(self, hybrid_model_manager):
        self.hybrid_model_manager = hybrid_model_manager
    
    def is_available(self) -> bool:
        return (self.hybrid_model_manager is not None and 
                hasattr(self.hybrid_model_manager, 'get_available_models'))
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        
        try:
            # Используем первую доступную модель
            available_models = self.hybrid_model_manager.get_available_models()
            if not available_models:
                return GenerationResponse(
                    text="Ошибка: нет доступных моделей",
                    status="error",
                    error_message="Нет доступных моделей в гибридном менеджере",
                    generation_time=time.time() - start_time
                )
            
            model_name = list(available_models.keys())[0]
            
            # Генерация через гибридный менеджер
            response = self.hybrid_model_manager.generate_response(
                model_name=model_name,
                prompt=request.text,
                max_tokens=request.max_tokens,
                temperature=getattr(request, 'temperature', 0.7)
            )
            
            # Обрабатываем разные форматы ответа
            if isinstance(response, dict):
                response_text = response.get('text', str(response))
            else:
                response_text = str(response)
            
            generation_time = time.time() - start_time
            
            return GenerationResponse(
                text=response_text,
                status="ok",
                source="HybridModelProvider",
                model_name=model_name,
                tokens_generated=len(response_text.split()),
                generation_time=generation_time,
                metadata={
                    "provider": "HybridModelProvider",
                    "actual_model": model_name,
                    "window_type": available_models[model_name].get('window_type', 'unknown'),
                    "device": available_models[model_name].get('device', 'unknown')
                }
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Ошибка генерации в HybridModelProvider: {e}")
            
            return GenerationResponse(
                text=f"Ошибка генерации: {str(e)}",
                status="error",
                source="HybridModelProvider",
                generation_time=generation_time,
                error_message=str(e)
            )
    
    def get_priority(self) -> int:
        return 1  # Высший приоритет


class FractalModelProvider(GenerationProvider):
    """Провайдер на основе фрактальной модели"""
    
    def __init__(self, fractal_model_manager):
        self.fractal_model_manager = fractal_model_manager
    
    def is_available(self) -> bool:
        return (self.fractal_model_manager is not None and 
                hasattr(self.fractal_model_manager, 'initialized') and 
                self.fractal_model_manager.initialized)
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        
        try:
            if not self.is_available():
                return GenerationResponse(
                    text="",
                    status="error",
                    error_message="Фрактальная модель недоступна",
                    source="fractal_model"
                )
            
            # Генерация через модель (RuGPT-3 или Fractal)
            response_text = self.fractal_model_manager.generate_response(
                request.text, 
                max_tokens=request.max_tokens,
                temperature=getattr(request, 'temperature', 0.8),
                top_p=getattr(request, 'top_p', 0.95),
                top_k=getattr(request, 'top_k', 40),
                do_sample=False,  # Greedy для стабильности
                no_repeat_ngram_size=3
            )
            
            generation_time = time.time() - start_time
            
            # Определяем имя модели
            model_name = getattr(self.fractal_model_manager, 'model_name', 'unknown')
            if 'rugpt' in model_name.lower():
                model_display_name = "ruGPT-3"
            elif 'gpt2' in model_name.lower():
                model_display_name = "GPT-2"
            else:
                model_display_name = model_name
            
            return GenerationResponse(
                text=response_text,
                status="ok",
                source="fractal_model",
                model_name=model_display_name,
                generation_time=generation_time,
                metadata={
                    "provider": "FractalModelProvider",
                    "actual_model": model_name,
                    "device": getattr(self.fractal_model_manager, 'device', 'unknown')
                }
            )
            
        except Exception as e:
            logger.error(f"Ошибка генерации через фрактальную модель: {e}")
            return GenerationResponse(
                text="",
                status="error",
                error_message=str(e),
                source="fractal_model",
                generation_time=time.time() - start_time
            )
    
    def get_priority(self) -> int:
        return 1  # Высший приоритет


class ResponseGeneratorProvider(GenerationProvider):
    """Провайдер на основе ResponseGenerator"""
    
    def __init__(self, response_generator):
        self.response_generator = response_generator
    
    def is_available(self) -> bool:
        return self.response_generator is not None
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        
        try:
            if not self.is_available():
                return GenerationResponse(
                    text="",
                    status="error",
                    error_message="ResponseGenerator недоступен",
                    source="response_generator"
                )
            
            # Генерация через ResponseGenerator
            kwargs = {
                'max_tokens': request.max_tokens,
                'temperature': request.temperature,
                'context': request.context,
                'user_context': request.user_context
            }
            
            response_data = self.response_generator.generate_response(request.text, **kwargs)
            
            generation_time = time.time() - start_time
            
            # Извлекаем текст из ответа
            if isinstance(response_data, dict):
                text = response_data.get('text', str(response_data))
                model_name = response_data.get('model_name', 'unknown')
                status = response_data.get('status', 'ok')
                error = response_data.get('error')
            else:
                text = str(response_data)
                model_name = 'unknown'
                status = 'ok'
                error = None
            
            return GenerationResponse(
                text=text,
                status=status,
                source="response_generator",
                model_name=model_name,
                generation_time=generation_time,
                error_message=error,
                metadata={"provider": "ResponseGeneratorProvider"}
            )
            
        except Exception as e:
            logger.error(f"Ошибка генерации через ResponseGenerator: {e}")
            return GenerationResponse(
                text="",
                status="error",
                error_message=str(e),
                source="response_generator",
                generation_time=time.time() - start_time
            )
    
    def get_priority(self) -> int:
        return 2


class MLUnitProvider(GenerationProvider):
    """Провайдер на основе MLUnit"""
    
    def __init__(self, ml_unit):
        self.ml_unit = ml_unit
    
    def is_available(self) -> bool:
        return self.ml_unit is not None
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        
        try:
            if not self.is_available():
                return GenerationResponse(
                    text="",
                    status="error",
                    error_message="MLUnit недоступен",
                    source="ml_unit"
                )
            
            # Генерация через MLUnit
            kwargs = {
                'max_tokens': request.max_tokens,
                'temperature': request.temperature,
                'context': request.context,
                'user_context': request.user_context
            }
            
            response_data = self.ml_unit.generate_response(request.text, **kwargs)
            
            generation_time = time.time() - start_time
            
            # Извлекаем текст из ответа
            if isinstance(response_data, dict):
                text = response_data.get('text', response_data.get('response', str(response_data)))
                model_name = response_data.get('model_name', 'unknown')
                status = response_data.get('status', 'ok')
                error = response_data.get('error')
            else:
                text = str(response_data)
                model_name = 'unknown'
                status = 'ok'
                error = None
            
            return GenerationResponse(
                text=text,
                status=status,
                source="ml_unit",
                model_name=model_name,
                generation_time=generation_time,
                error_message=error,
                metadata={"provider": "MLUnitProvider"}
            )
            
        except Exception as e:
            logger.error(f"Ошибка генерации через MLUnit: {e}")
            return GenerationResponse(
                text="",
                status="error",
                error_message=str(e),
                source="ml_unit",
                generation_time=time.time() - start_time
            )
    
    def get_priority(self) -> int:
        return 3


class UnifiedGenerationCoordinator:
    """Унифицированный координатор генерации текста"""
    
    def __init__(self):
        self.providers = []
        self.default_provider = None
        self.fallback_response = "Извините, произошла ошибка при генерации ответа."
    
    def register_provider(self, provider: GenerationProvider):
        """Регистрирует провайдера генерации"""
        self.providers.append(provider)
        # Сортируем по приоритету
        self.providers.sort(key=lambda p: p.get_priority())
        logger.debug(f"Зарегистрирован провайдер: {provider.__class__.__name__}")
    
    def set_default_provider(self, provider: GenerationProvider):
        """Устанавливает провайдер по умолчанию"""
        self.default_provider = provider
        logger.debug(f"Провайдер по умолчанию: {provider.__class__.__name__}")
    
    def generate(self, text: str, **kwargs) -> GenerationResponse:
        """
        Основной метод генерации текста
        
        Args:
            text: Текст запроса
            **kwargs: Дополнительные параметры
            
        Returns:
            GenerationResponse: Стандартизированный ответ
        """
        request = GenerationRequest(text, **kwargs)
        
        logger.info(f"Запрос на генерацию: '{text[:50]}...' (провайдеров: {len(self.providers)})")
        
        # Пробуем провайдеры в порядке приоритета
        for provider in self.providers:
            try:
                if provider.is_available():
                    logger.debug(f"Используем провайдер: {provider.__class__.__name__}")
                    response = provider.generate(request)
                    
                    if response.status == "ok":
                        logger.info(f"Успешная генерация через {provider.__class__.__name__}")
                        return response
                    else:
                        logger.warning(f"Провайдер {provider.__class__.__name__} вернул ошибку: {response.error_message}")
                        continue
                else:
                    logger.debug(f"Провайдер {provider.__class__.__name__} недоступен")
                    
            except Exception as e:
                logger.error(f"Критическая ошибка провайдера {provider.__class__.__name__}: {e}")
                continue
        
        # Если все провайдеры недоступны или вернули ошибки
        logger.error("Все провайдеры генерации недоступны или вернули ошибки")
        
        # Пробуем провайдер по умолчанию
        if self.default_provider and self.default_provider.is_available():
            try:
                logger.info(f"Используем провайдер по умолчанию: {self.default_provider.__class__.__name__}")
                return self.default_provider.generate(request)
            except Exception as e:
                logger.error(f"Ошибка провайдера по умолчанию: {e}")
        
        # Возвращаем fallback ответ
        return GenerationResponse(
            text=self.fallback_response,
            status="error",
            error_message="Все провайдеры недоступны",
            source="fallback"
        )

    def generate_response(self, prompt: str, max_tokens: int = 200, **kwargs) -> str:
        """
        Упрощенный метод генерации текста для обратной совместимости.
        
        Args:
            prompt: Текст промпта
            max_tokens: Максимальное количество токенов (по умолчанию 200)
            **kwargs: Дополнительные параметры
            
        Returns:
            str: Сгенерированный текст
        """
        try:
            response = self.generate(prompt, max_tokens=max_tokens, **kwargs)
            if response and hasattr(response, 'text'):
                return response.text
            elif response and hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response) if response else ""
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус координатора и провайдеров"""
        providers_status = []
        
        for provider in self.providers:
            providers_status.append({
                "name": provider.__class__.__name__,
                "available": provider.is_available(),
                "priority": provider.get_priority()
            })
        
        return {
            "total_providers": len(self.providers),
            "available_providers": sum(1 for p in self.providers if p.is_available()),
            "default_provider": self.default_provider.__class__.__name__ if self.default_provider else None,
            "providers": providers_status
        }


# Глобальный экземпляр координатора
_generation_coordinator = None


def get_generation_coordinator() -> UnifiedGenerationCoordinator:
    """Возвращает глобальный координатор генерации"""
    global _generation_coordinator
    if _generation_coordinator is None:
        _generation_coordinator = UnifiedGenerationCoordinator()
    return _generation_coordinator


def initialize_generation_coordinator(brain):
    """Инициализирует координатор генерации с компонентами brain"""
    coordinator = get_generation_coordinator()
    
    # Регистрируем провайдеров в порядке приоритета
    
    # 1. HybridModelManager (высший приоритет)
    if hasattr(brain, 'model_manager') and brain.model_manager:
        # Проверяем является ли это HybridModelManager
        if hasattr(brain.model_manager, 'get_available_models') and hasattr(brain.model_manager, 'generate_response'):
            hybrid_provider = HybridModelProvider(brain.model_manager)
            coordinator.register_provider(hybrid_provider)
            logger.info("[OK] Зарегистрирован HybridModelProvider")
        else:
            # Fallback на старый провайдер
            model_provider = FractalModelProvider(brain.model_manager)
            coordinator.register_provider(model_provider)
            logger.info("[OK] Зарегистрирован FractalModelProvider (fallback)")
    
    # 2. Fallback на другие менеджеры (если доступны)
    elif hasattr(brain, 'fractal_model_manager') and brain.fractal_model_manager:
        if hasattr(brain.fractal_model_manager, 'initialized') and brain.fractal_model_manager.initialized:
            fractal_provider = FractalModelProvider(brain.fractal_model_manager)
            coordinator.register_provider(fractal_provider)
            logger.info("[OK] Зарегистрирован fractal_model_manager")
        else:
            logger.warning("[WARN] fractal_model_manager не инициализирован, пропускаем")
    
    # 3. ResponseGenerator
    if hasattr(brain, 'components') and 'response_generator' in brain.components:
        response_provider = ResponseGeneratorProvider(brain.components['response_generator'])
        coordinator.register_provider(response_provider)
    
    # 3. MLUnit
    if hasattr(brain, 'components') and 'ml_unit' in brain.components:
        ml_provider = MLUnitProvider(brain.components['ml_unit'])
        coordinator.register_provider(ml_provider)
    
    logger.info(f"Координатор генерации инициализирован с {len(coordinator.providers)} провайдерами")
    
    return coordinator
