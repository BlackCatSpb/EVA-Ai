"""
FMF Model B Adapter - Подключение FMF в EVA как модели B
Позволяет использовать FMF (OpenVINO) вместо GGUF модели
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

class FMFModelBAdapter:
    """
    Адаптер для FMF, совместимый с интерфейсом llama-cpp модели
    Используется как model_b в DualGenerator
    """
    
    def __init__(
        self,
        model_path: str = None,
        graph_path: str = None,
        n_ctx: int = 2048,
        n_threads: int = 8,
        **kwargs
    ):
        self.model_path = model_path
        self.graph_path = graph_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._generator = None
        
        # Поиск путей по умолчанию
        if model_path is None:
            # Ищем в директории fmf_model
            default_path = Path(__file__).parent / "model.ov"
            if default_path.exists():
                self.model_path = str(default_path)
                
        if graph_path is None:
            default_graph = Path(__file__).parent / "data" / "graph.db"
            if default_graph.exists():
                self.graph_path = str(default_graph)
    
    def _ensure_generator(self):
        """Ленивая загрузка FMF генератора"""
        if self._generator is None:
            # Добавляем путь к FMF
            fmf_src_path = Path(__file__).parent / "src"
            if str(fmf_src_path) not in sys.path:
                sys.path.insert(0, str(fmf_src_path))
            
            from fmf_adapter import FMFAdapter
            
            self._generator = FMFAdapter(
                model_path=self.model_path,
                graph_path=self.graph_path,
                device="CPU",
                max_tokens=self.n_ctx
            )
            
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Интерфейс, совместимый с llama_cpp.Llama.create_chat_completion()
        """
        self._ensure_generator()
        
        # Преобразуем messages в текст
        prompt = self._messages_to_prompt(messages)
        
        result = self._generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": result.get("finish_reason", "stop")
            }],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": max_tokens,
                "total_tokens": len(prompt) + max_tokens
            }
        }
    
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Прямой вызов генерации"""
        self._ensure_generator()
        
        result = self._generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        
        return {
            "choices": [{
                "text": result["text"]
            }]
        }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Преобразовать chat messages в prompt"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    @property
    def tokenizer(self):
        """Вернуть токенизатор (для совместимости)"""
        return None
    
    def token_eos(self):
        """Вернуть EOS токен"""
        return 146215  # Qwen2 token
    
    def n_vocab(self):
        """Размер vocabulary"""
        return 146260


def create_fmf_model_b(
    model_path: str = None,
    graph_path: str = None,
    n_ctx: int = 2048,
    n_threads: int = 8
) -> FMFModelBAdapter:
    """Фабрика для создания FMF Model B"""
    return FMFModelBAdapter(
        model_path=model_path,
        graph_path=graph_path,
        n_ctx=n_ctx,
        n_threads=n_threads
    )


if __name__ == "__main__":
    # Тест
    print("=== FMF Model B Adapter Test ===")
    
    adapter = create_fmf_model_b()
    
    messages = [
        {"role": "user", "content": "Привет! Кто ты?"}
    ]
    
    result = adapter.create_chat_completion(messages, max_tokens=100)
    print(f"Response: {result['choices'][0]['message']['content'][:200]}...")
    print("OK!")
