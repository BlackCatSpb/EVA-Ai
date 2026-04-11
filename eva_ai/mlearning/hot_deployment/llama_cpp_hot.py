"""
LlamaCpp генератор для горячего развертывания eva_ai.
Интегрирует GGUF модель в систему.
"""
import os
import sys
import time
import logging
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.mlearning.hot_deployment.llama_cpp_hot")

# Импорт родительского класса
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva_ai.mlearning.hot_deployment import HotDeploymentManager, GraphNode, NodeState


def load_ethics_prompt() -> str:
    """Загружает этический промпт из EVA_ethics.md"""
    try:
        # Ищем EVA_ethics.md в корне проекта
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        ethics_path = os.path.join(project_root, "EVA_ethics.md")
        
        if os.path.exists(ethics_path):
            with open(ethics_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Убираем markdown заголовок "# Системный промт для GGUF-модели"
                lines = content.split('\n')
                filtered_lines = []
                for line in lines:
                    if not line.startswith('# Системный промт'):
                        filtered_lines.append(line)
                return '\n'.join(filtered_lines).strip()
        else:
            logger.warning(f"EVA_ethics.md не найден")
            return ""
    except Exception as e:
        logger.error(f"Ошибка загрузки EVA_ethics.md: {e}")
        return ""


def format_prompt_with_ethics(prompt: str) -> str:
    """Форматирует промпт с этическим контекстом"""
    ethics = load_ethics_prompt()
    if ethics:
        return f"{ethics}\n\n---\n\n## Запрос пользователя\n\n{prompt}\n\nОтветь в соответствии с описанными выше правилами:"
    return prompt


class LlamaCppHotNode(GraphNode):
    """
    Узел графа с llama.cpp моделью в горячем состоянии.
    """
    
    def __init__(self, node_id: str = None, parent_id: str = None, 
                 address: str = "0", depth: int = 0):
        super().__init__(node_id, parent_id, address, depth)
        self.llama_instance = None
        
    def activate_with_llama(self, llama_instance) -> bool:
        """Активирует узел с llama.cpp инстансом"""
        with self._lock:
            self.llama_instance = llama_instance
            self.state = NodeState.HOT
            self.index.state = NodeState.HOT
            self.index.last_access = time.time()
            self.metadata.purpose = "llama_cpp"
            self.metadata.memory_footprint = 400_000_000  # ~400MB
            
            logger.info(f"Узел {self.node_id} активирован с llama.cpp")
            return True
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> Optional[str]:
        """Генерирует текст через llama.cpp"""
        with self._lock:
            if self.llama_instance is None:
                logger.warning(f"llama.cpp не инициализирован в узле {self.node_id}")
                return None
            
            try:
                start = time.time()
                
                response = self.llama_instance.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 40),
                    repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                    stop=kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"])
                )
                
                elapsed = time.time() - start
                
                text = response['choices'][0]['message']['content']
                tokens = response.get('usage', {}).get('completion_tokens', 0)
                speed = tokens / elapsed if elapsed > 0 else 0
                
                self.last_response = text
                self.index.last_access = time.time()
                self.index.access_count += 1
                
                logger.debug(f"Генерация: {tokens} токенов за {elapsed:.2f}s ({speed:.1f} tok/s)")
                
                return text
                
            except Exception as e:
                logger.error(f"Ошибка генерации llama.cpp: {e}")
                return None


class LlamaCppHotDeployment(HotDeploymentManager):
    """
    Горячее развертывание с llama.cpp (GGUF модель).
    Значительно быстрее чем PyTorch на CPU.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = None,  # По умолчанию: все ядра CPU
        n_gpu_layers: int = 0,
        system_prompt: str = None,
        purpose: str = "general",
        **kwargs
    ):
        # Определяем путь к GGUF модели
        if not model_path.endswith('.gguf'):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            possible_paths = [
                os.path.join(project_root, "models", "qwen2.5-0.5b-instruct-q4_0.gguf"),
                os.path.join(project_root, "eva_ai", "models", "qwen2.5-0.5b-instruct-q4_0.gguf"),
                os.path.join(project_root, "eva_ai", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-0.5b-instruct-q4_0.gguf"),
                model_path,
            ]
            
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break
        
        super().__init__(model_path, **kwargs)
        
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 12  # Все ядра CPU
        self.n_gpu_layers = n_gpu_layers
        self.system_prompt = system_prompt
        self.purpose = purpose
        
        self.llama = None
        
        logger.info(f"LlamaCppHotDeployment: model={model_path}, n_ctx={n_ctx}, threads={n_threads}, purpose={purpose}")
    
    def initialize(self, preload_root: bool = False) -> bool:
        """Инициализация с llama.cpp (без активации узла для скорости)"""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Загрузка GGUF модели: {self.model_path}")
            
            self.llama = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                chat_format="qwen",
                verbose=False
            )
            
            logger.info("llama.cpp модель загружена!")
            
            # Убираем тяжёлую активацию узла - она не нужна для генерации
            # Горячий узел активируется только при реальном использовании
            self.ready = True
            logger.info(f"LlamaCppHotDeployment [{self.purpose}] готов к работе (быстрая инициализация)!")
            
            return self.ready
        
        except Exception as e:
            logger.error(f"Ошибка инициализации llama.cpp: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        node_address: Optional[str] = None,
        use_best_node: bool = True,
        system_prompt: str = None,
        temperature: float = None,
        top_p: float = None,
        repeat_penalty: float = None,
        **kwargs
    ) -> Optional[str]:
        """Генерация через llama.cpp"""
        if not self.ready or self.llama is None:
            logger.error("LlamaCppHotDeployment не готов")
            return None
        
        try:
            start = time.time()
            
            # Определяем системный промпт: приоритет у переданного, потом у своего, потом этический
            final_system = system_prompt or self.system_prompt or ""
            
            # Формируем сообщения с системным промптом
            messages = []
            if final_system:
                messages.append({"role": "system", "content": final_system})
            messages.append({"role": "user", "content": prompt})
            
            # Используем параметры или значения по умолчанию
            temp = temperature if temperature is not None else kwargs.get("temperature", 0.7)
            tp = top_p if top_p is not None else kwargs.get("top_p", 0.9)
            rp = repeat_penalty if repeat_penalty is not None else kwargs.get("repeat_penalty", 1.1)
            
            # Используем llama.cpp напрямую (без узлов для скорости)
            response = self.llama.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temp,
                top_p=tp,
                top_k=kwargs.get("top_k", 40),
                repeat_penalty=rp,
                stop=kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"])
            )
            
            elapsed = time.time() - start
            
            text = response['choices'][0]['message']['content']
            tokens = response.get('usage', {}).get('completion_tokens', max_new_tokens)
            speed = tokens / elapsed if elapsed > 0 else 0
            
            logger.info(f"GGUF [{self.purpose}]: {tokens} токенов за {elapsed:.2f}s ({speed:.1f} tok/s)")
            
            # Проверяем на повторения и убираем
            if text:
                text = self._remove_repetitions(text)
            
            return text
        
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return None
    
    def _remove_repetitions(self, text: str) -> str:
        """Убирает повторяющиеся фразы из текста."""
        if not text or len(text) < 50:
            return text
        
        lines = text.split('\n')
        seen_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line not in seen_lines:
                seen_lines.append(line)
            else:
                break
        
        result = '\n'.join(seen_lines)
        
        words = result.split()
        unique_words = []
        for i, word in enumerate(words):
            if i > 0 and word == words[i-1]:
                continue
            unique_words.append(word)
        
        return ' '.join(unique_words)
    
    def get_status(self) -> Dict:
        """Статус системы"""
        return {
            "ready": self.ready,
            "engine": "llama.cpp",
            "model": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "speed_estimate": "~70 tok/s"
        }
    
    def unload(self) -> bool:
        """Активная выгрузка GGUF модели из памяти."""
        try:
            if self.llama is not None:
                del self.llama
                self.llama = None
            
            self.ready = False
            
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            
            logger.info(f"LlamaCppHotDeployment [{self.purpose}] выгружен из памяти")
            return True
        except Exception as e:
            logger.error(f"Ошибка выгрузки LlamaCppHotDeployment: {e}")
            return False
    
    def __del__(self):
        """Деструктор — автоматическая выгрузка при удалении объекта."""
        try:
            if self.llama is not None:
                del self.llama
                self.llama = None
        except Exception:
            pass


# ============================================================================
# Глобальный синглтон
# ============================================================================

_llama_cpp_instance: Optional[LlamaCppHotDeployment] = None
_init_lock = threading.Lock()


def get_llama_cpp_deployment(
    model_path: Optional[str] = None,
    force_reload: bool = False
) -> LlamaCppHotDeployment:
    """Возвращает синглтон LlamaCppHotDeployment"""
    global _llama_cpp_instance
    
    with _init_lock:
        if _llama_cpp_instance is None or force_reload:
            _llama_cpp_instance = LlamaCppHotDeployment(
                model_path=model_path or "",
                n_ctx=4096,
                n_threads=8
            )
            _llama_cpp_instance.initialize()
        
        return _llama_cpp_instance


def unload_llama_cpp_deployment() -> bool:
    """Выгружает глобальный экземпляр llama.cpp из памяти."""
    global _llama_cpp_instance
    with _init_lock:
        if _llama_cpp_instance is not None:
            result = _llama_cpp_instance.unload()
            _llama_cpp_instance = None
            return result
        return False


# ============================================================================
# Тест
# ============================================================================

def test_llama_cpp_hot():
    """Тест интегрированного llama.cpp в горячее развертывание"""
    logger.info("=== Тест LlamaCppHotDeployment ===")
    
    deployment = get_llama_cpp_deployment()
    
    if not deployment.ready:
        logger.error("Развертывание не готово")
        return
    
    status = deployment.get_status()
    logger.info(f"Статус: {status}")
    
    # Генерация
    logger.info("\nТест генерации:")
    
    response = deployment.generate("Привет! Как дела?", max_new_tokens=30)
    logger.info(f"Ответ: {response[:150] if response else 'None'}")
    
    # Ещё тесты
    logger.info("\nТест 2:")
    response2 = deployment.generate("Сколько будет 2+2?", max_new_tokens=20)
    logger.info(f"Ответ: {response2}")
    
    logger.info("\nТест 3:")
    response3 = deployment.generate("Расскажи анекдот", max_new_tokens=50)
    logger.info(f"Ответ: {response3[:200] if response3 else 'None'}")
    
    logger.info("\n=== Тест завершён ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_llama_cpp_hot()