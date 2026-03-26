"""
Qwen Cloud API Client для CogniFlex
Подключение к Alibaba Cloud Model Studio (OpenAI-совместимый API)
"""
import os
import logging
from typing import Optional, Dict, Any, List, Iterator

logger = logging.getLogger("cogniflex.mlearning.qwen_api")

class QwenAPIError(Exception):
    """Base exception for Qwen API errors"""
    pass

class QwenAPIConnectionError(QwenAPIError):
    """Connection error"""
    pass

class QwenAPIAuthenticationError(QwenAPIError):
    """Authentication error (invalid API key)"""
    pass

class QwenAPIRateLimitError(QwenAPIError):
    """Rate limit exceeded"""
    pass


class QwenAPIClient:
    """Клиент для работы с Qwen Cloud API через Alibaba Cloud Model Studio"""
    
    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen3.5-plus"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Инициализация клиента
        
        Args:
            api_key: API ключ (или из переменной окружения QWEN_API_KEY)
            base_url: Base URL API (по умолчанию Model Studio)
            model: Модель для использования (по умолчанию qwen3.5-plus)
            timeout: Таймаут запроса в секундах
        """
        self.api_key = api_key or os.environ.get('QWEN_API_KEY', '')
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("Qwen API key не предоставлен")
        
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Инициализирует OpenAI-совместимый клиент через requests"""
        self._client = True  # Marker that we're using direct requests
        logger.info(f"Qwen API клиент инициализирован (direct requests): {self.base_url}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Генерирует ответ от Qwen
        
        Args:
            prompt: Текстовый промпт
            system_prompt: Системный промпт
            temperature: Температура генерации (0.0-1.0)
            max_new_tokens: Максимальное количество токенов
            stream: Использовать потоковую генерацию
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict с ключами: text, usage, model
        """
        if not self.api_key:
            raise QwenAPIAuthenticationError("API key не предоставлен")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            import requests
            
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            data.update(kwargs)
            
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=self.timeout,
                stream=stream
            )
            
            if response.status_code == 401:
                raise QwenAPIAuthenticationError("Неверный API ключ")
            elif response.status_code == 429:
                raise QwenAPIRateLimitError("Превышен лимит запросов")
            elif response.status_code >= 400:
                raise QwenAPIError(f"API ошибка {response.status_code}: {response.text}")
            
            if stream:
                return {"stream": True, "response": response.iter_lines()}
            
            result_json = response.json()
            
            # Parse response
            choice = result_json.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            result = {
                "text": message.get("content", ""),
                "usage": result_json.get("usage", {}),
                "model": result_json.get("model", self.model),
                "finish_reason": choice.get("finish_reason")
            }
            
            logger.info(f"Qwen API: сгенерировано {result['usage'].get('total_tokens', 0)} токенов")
            return result
            
        except requests.exceptions.ConnectionError as e:
            raise QwenAPIConnectionError(f"Ошибка подключения: {e}")
        except requests.exceptions.Timeout as e:
            raise QwenAPIConnectionError(f"Таймаут: {e}")
        except QwenAPIError:
            raise
        except Exception as e:
            raise QwenAPIError(f"Ошибка API: {e}")
    
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Упрощённый синхронный вызов - возвращает только текст"""
        result = self.generate(prompt, stream=False, **kwargs)
        return result.get("text", "")
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Потоковая генерация - возвращает итератор"""
        result = self.generate(prompt, stream=True, **kwargs)
        for chunk in result.get("response"):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def is_available(self) -> bool:
        """Проверяет доступность API"""
        if not self.api_key:
            return False
        try:
            # Test with minimal request
            self.generate("Hi", max_new_tokens=5)
            return True
        except QwenAPIAuthenticationError:
            return False
        except Exception as e:
            logger.warning(f"Qwen API недоступен: {e}")
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Возвращает список доступных моделей"""
        # Список моделей Qwen (может быть устаревшим)
        return [
            {"id": "qwen3.5-plus", "name": "Qwen 3.5 Plus", "context": "32K"},
            {"id": "qwen3.5-27b", "name": "Qwen 3.5 27B", "context": "32K"},
            {"id": "qwen3.5-72b", "name": "Qwen 3.5 72B", "context": "32K"},
            {"id": "qwen3.5-plus-02-15", "name": "Qwen 3.5 Plus (Feb 15)", "context": "128K"},
            {"id": "qwen-turbo", "name": "Qwen Turbo", "context": "8K"},
            {"id": "qwen-plus", "name": "Qwen Plus", "context": "32K"},
            {"id": "qwen-max", "name": "Qwen Max", "context": "8K"},
        ]


__all__ = [
    'QwenAPIClient',
    'QwenAPIError',
    'QwenAPIConnectionError', 
    'QwenAPIAuthenticationError',
    'QwenAPIRateLimitError'
]
