"""
Universal FMF Adapter - Замена PyTorch/GGUF моделям
Позволяет использовать FMF EVA в любом проекте как стандартный генератор
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

class FMFAdapter:
    """
    Универсальный адаптер для FMF EVA
    Может использоваться вместо: transformers pipeline, llama-cpp, openvino-genai
    """
    
    def __init__(
        self,
        model_path: str = "model.ov",
        graph_path: str = "data/graph.db",
        device: str = "CPU",
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        self.model_path = Path(model_path)
        self.graph_path = Path(graph_path)
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.generator = None
        
    def _load(self):
        """Ленивая загрузка модели"""
        if self.generator is None:
            sys.path.insert(0, str(Path(__file__).parent))
            from fmf_interactive import FMFGeneratorInteractive
            
            self.generator = FMFGeneratorInteractive(
                str(self.model_path),
                str(self.graph_path),
                self.device,
                enable_embeddings=True
            )
            
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Основной метод генерации
        Аналог: model.generate() в transformers или llm.generate() в llama-cpp
        """
        self._load()
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        result = self.generator.generate(
            prompt,
            max_tokens=max_tokens,
            enable_thinking=kwargs.get("thinking", False)
        )
        
        return {
            "text": result.get("response", ""),
            "prompt": prompt,
            "latency_ms": result.get("latency_ms", 0),
            "finish_reason": "stop" if "<|response_end|>" in result.get("response", "") else "length"
        }
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Вызов как функции - возвращает только текст"""
        result = self.generate(prompt, **kwargs)
        return result["text"]
    
    @property
    def config(self) -> Dict[str, Any]:
        """Вернуть конфигурацию модели"""
        return {
            "model_path": str(self.model_path),
            "graph_path": str(self.graph_path),
            "device": self.device,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class FMFPipeline:
    """
    Аналог transformers.Pipeline для FMF EVA
    Простой интерфейс: pipeline("текст") -> результат
    """
    
    def __init__(
        self,
        model: str = "model.ov",
        device: str = "CPU",
        **kwargs
    ):
        self.adapter = FMFAdapter(
            model_path=model,
            device=device,
            **kwargs
        )
        
    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Простой вызов: pipeline("текст")"""
        return self.adapter.generate(prompt, **kwargs)
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Явный вызов generate"""
        return self.adapter.generate(prompt, **kwargs)


# === Функции совместимости с различными библиотеками ===

def from_pretrained(model_path: str, **kwargs) -> FMFAdapter:
    """
    Аналог: AutoModel.from_pretrained()
    from_pretrained("model.ov", device="CPU")
    """
    return FMFAdapter(model_path=model_path, **kwargs)


def create_pipeline(model: str = "model.ov", **kwargs) -> FMFPipeline:
    """
    Аналог: pipeline("text-generation", model="...")
    create_pipeline("model.ov")
    """
    return FMFPipeline(model=model, **kwargs)


# === Примеры использования ===

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FMF Universal Adapter")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--model", default="model.ov", help="Model path")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="CPU")
    
    args = parser.parse_args()
    
    print(f"[FMF] Loading model from {args.model}...")
    
    # Способ 1: Direct
    adapter = FMFAdapter(args.model, device=args.device)
    result = adapter.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(f"\n[Result]")
    print(result["text"])
    print(f"\nLatency: {result['latency_ms']:.0f}ms")