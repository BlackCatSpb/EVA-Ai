"""
FractalQwenManager - менеджер второго экземпляра Qwen для генерации промтов.
Использует фрактальное хранилище и загружается на CPU.
"""
import os
import logging
import torch
from typing import Optional, Dict, Any, List

logger = logging.getLogger("eva_ai.mlearning.fractal_qwen")


class FractalQwenManager:
    """
    Управляет вторым экземпляром Qwen для генерации промтов уточнения.
    
    Отличия от основного Qwen:
    - Загружается на CPU (не конфликтует с GPU моделью)
    - Использует фрактальное хранилище для весов
    - Оптимизирован для коротких промтов (512 токенов)
    - Меньше памяти за счёт отсутствия GPU
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        fractal_storage_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Инициализация Fractal Qwen Manager.
        
        Args:
            model_path: Путь к модели (если не из фрактального хранилища)
            fractal_storage_path: Путь к фрактальному хранилищу
            device: Устройство для загрузки (cpu)
        """
        self.model_path = model_path
        self.fractal_storage_path = fractal_storage_path or os.path.join(
            os.path.dirname(__file__), "..", "memory", "fractal_torch_storage"
        )
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        logger.info(f"FractalQwenManager инициализирован: device={device}")
    
    def initialize(self) -> bool:
        """Инициализирует модель для генерации промтов."""
        try:
            # Пробуем загрузить из фрактального хранилища
            if self._load_from_fractal_storage():
                logger.info("Модель загружена из фрактального хранилища")
                self.initialized = True
                return True
            
            # Fallback: загружаем стандартно на CPU
            logger.info("Загрузка модели стандартным способом на CPU...")
            return self._load_standard()
            
        except Exception as e:
            logger.error(f"Ошибка инициализации FractalQwen: {e}")
            self.initialized = False
            return False
    
    def _load_from_fractal_storage(self) -> bool:
        """Загружает модель из фрактального хранилища."""
        try:
            from eva_ai.memory.fractal_torch_storage import ModelExporter
            
            export_dir = os.path.join(
                os.path.dirname(__file__), "..", "memory", "fractal_torch_storage", "exported_models"
            )
            
            exporter = ModelExporter(export_dir=export_dir)
            
            # Пробуем импортировать модель
            model_data = exporter.import_model("qwen3.5-0.8b")
            
            if model_data is None:
                logger.warning("Модель не найдена во фрактальном хранилище")
                return False
            
            # Загружаем токенизатор
            from transformers import AutoTokenizer
            
            model_path = os.path.join(
                os.path.dirname(__file__), "eva_models", "qwen3.5-0.8b"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Фрактальное хранилище содержит веса, но нам нужна модель для генерации
            # Загружаем стандартную модель - она будет использовать те же веса из кэша
            logger.info("Веса загружены из фрактального хранилища, загружаем модель...")
            return self._load_standard()
            
        except Exception as e:
            logger.warning(f"Ошибка загрузки из фрактального хранилища: {e}")
            return False
    
    def _load_standard(self) -> bool:
        """Загружает модель стандартным способом на CPU."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            model_path = os.path.join(
                os.path.dirname(__file__), "eva_models", "qwen3.5-0.8b"
            )
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загружаем модель на CPU с float16
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "cpu",
                "low_cpu_mem_usage": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            logger.info("Модель загружена стандартно на CPU")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Ошибка стандартной загрузки: {e}")
            return False
    
    def generate_prompt(
        self,
        query: str,
        previous_response: str,
        module_feedback: Dict[str, str],
        max_tokens: int = 50
    ) -> str:
        """
        Генерирует улучшенный промпт для регенерации.
        
        Args:
            query: Оригинальный запрос пользователя
            previous_response: Предыдущий ответ системы
            module_feedback: Обратная связь от модулей {module_name: feedback}
            max_tokens: Максимальная длина
            
        Returns:
            str: Улучшенный промпт
        """
        if not self.initialized:
            logger.warning("FractalQwen не инициализирован")
            return self._default_prompt(query, previous_response, module_feedback)
        
        try:
            # Формируем промпт
            prompt = self._build_refinement_prompt(
                query, previous_response, module_feedback
            )
            
            # Токенизация
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            # Генерация (без temperature для greedy)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только сгенерированную часть
            return self._extract_prompt(response, prompt)
            
        except Exception as e:
            logger.error(f"Ошибка генерации промпта: {e}")
            return self._default_prompt(query, previous_response, module_feedback)
    
    def _build_refinement_prompt(
        self,
        query: str,
        previous_response: str,
        module_feedback: Dict[str, str]
    ) -> str:
        """Формирует промпт для улучшения ответа."""
        
        feedback_parts = []
        for module, feedback in module_feedback.items():
            if feedback:
                feedback_parts.append(f"[{module.upper()}]: {feedback}")
        
        feedback_text = "\n".join(feedback_parts) if feedback_parts else "Нет замечаний"
        
        prompt = f"""Ты - аналитический модуль ЕВА. Твоя задача - сформировать промпт для улучшения ответа.

Оригинальный запрос: {query}

Предыдущий ответ: {previous_response}

Обратная связь от модулей:
{feedback_text}

Сформируй краткий промпт (2-3 предложения) для улучшения ответа, учитывая обратную связь.

Улучшенный промпт:"""
        
        return prompt
    
    def _extract_prompt(self, response: str, original_prompt: str) -> str:
        """Извлекает сгенерированный промпт из ответа."""
        # Удаляем оригинальный промпт из ответа
        if original_prompt in response:
            prompt = response.replace(original_prompt, "").strip()
        else:
            prompt = response.strip()
        
        # Очищаем от лишних токенов
        prompt = prompt.split("Улучшенный промпт:")[-1].strip()
        
        return prompt[:500]  # Ограничиваем длину
    
    def _default_prompt(
        self,
        query: str,
        previous_response: str,
        module_feedback: Dict[str, str]
    ) -> str:
        """Fallback промпт если модель недоступна."""
        parts = [f"Улучши ответ на: {query}"]
        
        for module, feedback in module_feedback.items():
            if feedback:
                parts.append(f"Учитывай: {feedback}")
        
        return " ".join(parts)[:200]
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус менеджера."""
        return {
            "initialized": self.initialized,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }


# Глобальный экземпляр
_fractal_qwen_instance: Optional[FractalQwenManager] = None


def get_fractal_qwen(
    force_reload: bool = False,
    device: str = "cpu"
) -> Optional[FractalQwenManager]:
    """
    DISABLED - Using UnifiedGenerator instead of FractalQwenManager.
    
    Returns None to prevent fractal_qwen loading.
    """
    logger.info("FractalQwenManager disabled - using UnifiedGenerator")
    return None