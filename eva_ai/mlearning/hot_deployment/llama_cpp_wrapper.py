"""
Llama.cpp интеграция для eva_ai.
Использует llama-cpp-python для ускоренной генерации на CPU.
"""
import os
import sys
import time
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("eva_ai.mlearning.hot_deployment.llama_cpp_wrapper")

# Импорт HotDeploymentManager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva_ai.mlearning.hot_deployment import HotDeploymentManager

# Проверка наличия llama-cpp-python
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llama2ChatHandler
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    logger.warning("llama-cpp-python не установлен")


class LlamaCppGenerator:
    """
    Генератор на базе llama.cpp.
    Значительно быстрее чем transformers на CPU.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        
        self.model = None
        self.tokenizer = None
        
    def load(self) -> bool:
        """Загружает модель GGUF"""
        if not HAS_LLAMA_CPP:
            logger.error("llama-cpp-python не доступен")
            return False
        
        if not os.path.exists(self.model_path):
            logger.error(f"Модель не найдена: {self.model_path}")
            return False
        
        try:
            logger.info(f"Загрузка GGUF модели: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                use_mmap=True,
                use_mlock=False,
                chat_format="qwen"
            )
            
            logger.info("Модель llama.cpp загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Optional[str]:
        """Генерирует текст"""
        if self.model is None:
            logger.error("Модель не загружена")
            return None
        
        try:
            start = time.time()
            
            # Используем create_chat_completion для Qwen
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                **kwargs
            )
            
            elapsed = time.time() - start
            
            # Извлекаем текст
            if response and 'choices' in response:
                text = response['choices'][0]['message']['content']
                tokens_generated = response.get('usage', {}).get('completion_tokens', 0)
                speed = tokens_generated / elapsed if elapsed > 0 else 0
                
                logger.info(f"Сгенерировано {tokens_generated} токенов за {elapsed:.2f}s ({speed:.1f} tok/s)")
                
                return text
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return None
    
    def tokenize(self, text: str) -> List[int]:
        """Токенизирует текст"""
        if self.model is None:
            return []
        try:
            return self.model.tokenize(text)
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return []
    
    def get_status(self) -> Dict:
        """Возвращает статус"""
        return {
            "loaded": self.model is not None,
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers
        }


def download_qwen_gguf(
    output_dir: str = "./models",
    quant: str = "q5_k_m"
) -> Optional[str]:
    """
    Скачивает Qwen 0.8B в формате GGUF.
    
    Quantizations:
    - q2_k - наименьший размер, lowest quality
    - q3_k - маленький, low
    - q4_0 - средний, medium  
    - q4_k - medium
    - q5_k - хороший, high
    - q6_k - лучший, higher
    - q8_0 - почти без потерь, highest (no quant)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # URL для Qwen 0.8B GGUF
    model_name = f"qwen2.5-0.5b-instruct-{quant}.gguf"
    base_url = "https://huggingface.co/Qwen"
    
    # Пробуем разные источники
    # 1. TheBloke (готовые GGUF)
    urls = [
        f"https://huggingface.co/TheBloke/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/{model_name}",
        f"https://huggingface.co/Qwen/{model_name}/resolve/main/{model_name}"
    ]
    
    import urllib.request
    import ssl
    
    # Игнорируем SSL для huggingface
    ssl._create_default_https_context = ssl._create_unverified_context
    
    for url in urls:
        try:
            output_path = os.path.join(output_dir, model_name)
            if os.path.exists(output_path):
                logger.info(f"Модель уже существует: {output_path}")
                return output_path
            
            logger.info(f"Попытка скачать: {url}")
            
            # Скачиваем
            urllib.request.urlretrieve(url, output_path)
            
            logger.info(f"Модель скачана: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"Не удалось скачать с {url}: {e}")
            continue
    
    return None


def convert_to_gguf(
    model_path: str,
    output_dir: str,
    quantization: str = "q5_k_m"
) -> Optional[str]:
    """
    Конвертирует модель Transformers в GGUF формат.
    Требует llama.cpp установленный.
    """
    try:
        # Используем конвертер из llama.cpp
        # python -m llama_cpp.python.convert_hf_to_gguf ...
        
        output_path = os.path.join(
            output_dir,
            f"qwen3.5-0.8b-{quantization}.gguf"
        )
        
        logger.info(f"Конвертация в GGUF: {model_path} -> {output_path}")
        
        # Команда конвертации
        cmd = [
            sys.executable, "-m", "llama_cpp.llama_convert",
            model_path,
            output_path,
            "--quantize", quantization
        ]
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Конвертация завершена: {output_path}")
            return output_path
        else:
            logger.error(f"Ошибка конвертации: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Ошибка конвертации: {e}")
        return None


class HotDeploymentLlamaCpp(HotDeploymentManager):
    """
    Горячее развертывание с llama.cpp.
    """
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.llama_generator = None
        
    def initialize(self, preload_root: bool = True) -> bool:
        """Инициализация с llama.cpp"""
        if not HAS_LLAMA_CPP:
            logger.error("llama-cpp-python не доступен")
            return False
        
        try:
            # Определяем путь к GGUF модели
            if not model_path.endswith('.gguf'):
                # Пробуем найти GGUF в директории
                gguf_files = [
                    f for f in os.listdir(self.model_path)
                    if f.endswith('.gguf')
                ]
                if gguf_files:
                    self.model_path = os.path.join(self.model_path, gguf_files[0])
            
            # Создаём генератор
            self.llama_generator = LlamaCppGenerator(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=0
            )
            
            # Загружаем
            if not self.llama_generator.load():
                return False
            
            # Инициализируем родительский класс
            self.ready = True
            logger.info("HotDeployment (llama.cpp) готов!")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации llama.cpp: {e}")
            return False
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> Optional[str]:
        """Генерация через llama.cpp"""
        if self.llama_generator is None:
            return None
        
        return self.llama_generator.generate(
            prompt=prompt,
            max_tokens=max_new_tokens,
            **kwargs
        )


# ============================================================================
# Тесты
# ============================================================================

def test_llama_cpp():
    """Тест llama.cpp генерации"""
    # Путь к модели
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, "models")
    
    # Ищем GGUF модель
    gguf_path = None
    
    # Проверяем стандартные пути
    possible_paths = [
        os.path.join(models_dir, "qwen3.5-0.8b-instruct-q5_k_m.gguf"),
        os.path.join(project_root, "qwen3.5-0.8b-instruct-q5_k_m.gguf"),
        os.path.join(project_root, "eva", "models", "qwen3.5-0.8b-instruct-q5_k_m.gguf"),
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            gguf_path = p
            break
    
    if gguf_path is None:
        logger.error("GGUF модель не найдена")
        logger.info(f"Искали в: {possible_paths}")
        
        # Скачиваем модель
        logger.info("Пытаемся скачать GGUF модель...")
        gguf_path = download_qwen_gguf(models_dir, "q5_k_m")
        
        if gguf_path is None:
            logger.error("Не удалось скачать модель")
            return False
    
    logger.info(f"Используем модель: {gguf_path}")
    
    # Тест генерации
    logger.info("=== Тест llama.cpp ===")
    
    generator = LlamaCppGenerator(
        model_path=gguf_path,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=0
    )
    
    if not generator.load():
        logger.error("Не удалось загрузить модель")
        return False
    
    logger.info("Модель загружена!")
    
    # Генерация
    logger.info("Генерация...")
    response = generator.generate(
        prompt="Привет! Как дела?",
        max_tokens=50,
        temperature=0.1
    )
    
    logger.info(f"Ответ: {response[:200] if response else 'None'}")
    
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_llama_cpp()