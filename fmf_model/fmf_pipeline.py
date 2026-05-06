"""
Simplified FMF-only Pipeline - заменяет DualGenerator
Использует только FMF (OpenVINO) для генерации
"""
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("eva_ai.core.fmf_pipeline")

class FMFPipeline:
    """
    Упрощённый пайплайн на базе FMF адаптера
    Заменяет DualGenerator для упрощения системы
    """
    
    def __init__(
        self,
        model_path: str = None,
        graph_path: str = None,
        n_ctx: int = 2048,
        n_threads: int = 8,
        brain=None,
        **kwargs
    ):
        self.model_path = model_path
        self.graph_path = graph_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.brain = brain
        self._generator = None
        
        # Автопоиск путей
        if model_path is None:
            default = Path(__file__).parent.parent / "fmf_model"
            self.model_path = str(default / "model.ov")
            self.graph_path = str(default / "data" / "graph.db")
        
    def _ensure_generator(self):
        """Ленивая загрузка"""
        if self._generator is None:
            # Добавляем путь к FMF
            fmf_src = Path(__file__).parent.parent / "fmf_model" / "src"
            if str(fmf_src) not in sys.path:
                sys.path.insert(0, str(fmf_src))
            
            from fmf_adapter import FMFAdapter
            
            self._generator = FMFAdapter(
                model_path=self.model_path,
                graph_path=self.graph_path,
                device="CPU",
                max_tokens=self.n_ctx
            )
            logger.info(f"FMF Pipeline инициализирован: {self.model_path}")
    
    def process_query(
        self,
        query: str,
        conversation_history: List[Dict] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Обработать запрос - основной метод
        """
        self._ensure_generator()
        
        start_time = time.time()
        
        # Формируем prompt с историей
        prompt = self._build_prompt(query, conversation_history)
        
        # Генерация
        result = self._generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "response": result.get("text", ""),
            "latency_ms": latency,
            "thought": "FMF Generation",
            "model": "FMF_OpenVINO",
            "success": True
        }
    
    def _build_prompt(self, query: str, history: List[Dict] = None) -> str:
        """Построить prompt с историей"""
        prompt = ""
        
        if history:
            for msg in history[-5:]:  # Последние 5 сообщений
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Прямая генерация"""
        return self.process_query(prompt, **kwargs)
    
    def generate_streaming(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, chunk_size: int = 5):
        """Streaming генерация - для совместимости с EVA GUI"""
        self._ensure_generator()
        
        result = self._generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        text = result.get("text", "")
        
        # Разбиваем на чанки для стриминга
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            yield {"type": "chunk", "text": chunk}
        
        yield {"type": "done", "latency_ms": result.get("latency_ms", 0)}


def create_fmf_pipeline(
    model_path: str = None,
    graph_path: str = None,
    n_ctx: int = 2048,
    n_threads: int = 8,
    brain=None,
    **kwargs
) -> FMFPipeline:
    """Фабрика для создания FMF Pipeline"""
    return FMFPipeline(
        model_path=model_path,
        graph_path=graph_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        brain=brain,
        **kwargs
    )


if __name__ == "__main__":
    print("=== FMF Pipeline Test ===")
    
    pipeline = create_fmf_pipeline()
    
    result = pipeline.process_query("Привет! Кто ты?", max_tokens=100)
    print(f"Response: {result['response'][:200]}...")
    print(f"Latency: {result['latency_ms']:.0f}ms")
