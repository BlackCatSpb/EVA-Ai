"""
Параллельная токенизация через subprocess.
Избегаем проблем с pickle через IPC.
"""
import os
import sys
import time
import subprocess
import threading
import queue
import logging
import json
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("eva.mlearning.hot_deployment.tokenizer_pool")


class SubprocessTokenizerWorker:
    """
    Воркер токенизатора в отдельном процессе.
    Общается через stdin/stdout JSON.
    """
    
    def __init__(self, worker_id: int, model_path: str):
        self.worker_id = worker_id
        self.model_path = model_path
        self.process = None
        self.input_fd = None
        self.output_fd = None
        
    def start(self) -> bool:
        """Запускает subprocess воркер"""
        try:
            # Создаём скрипт воркера
            worker_script = self._create_worker_script()
            
            # Запускаем процесс
            self.process = subprocess.Popen(
                [sys.executable, "-c", worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.input_fd = self.process.stdin
            self.output_fd = self.process.stdout
            
            # Ждём готовности
            ready = self.output_fd.readline()
            if not ready.startswith("READY"):
                logger.error(f"Воркер не готов: {ready}")
                return False
            
            logger.info(f"Воркер {self.worker_id} запущен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска воркера: {e}")
            return False
    
    def _create_worker_script(self) -> str:
        """Создаёт скрипт воркера"""
        # Экранируем путь
        model_path_escaped = self.model_path.replace("\\", "\\\\")
        
        script = f'''
import sys
import json
from transformers import AutoTokenizer

model_path = r"{model_path_escaped}"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("READY", flush=True)

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        data = json.loads(line.strip())
        cmd = data.get("cmd")
        if cmd == "tokenize":
            text = data.get("text", "")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            print(json.dumps({{"tokens": tokens}}), flush=True)
        elif cmd == "stop":
            break
    except Exception as e:
        print(json.dumps({{"error": str(e)}}), flush=True)
'''
        return script
    
    def tokenize(self, text: str) -> List[int]:
        """Токенизирует текст"""
        if not self.process or not self.input_fd:
            return []
        
        try:
            request = json.dumps({"cmd": "tokenize", "text": text})
            self.input_fd.write(request + "\n")
            self.input_fd.flush()
            
            response = self.output_fd.readline()
            data = json.loads(response)
            return data.get("tokens", [])
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return []
    
    def stop(self):
        """Останавливает воркер"""
        if self.input_fd:
            try:
                self.input_fd.write(json.dumps({"cmd": "stop"}) + "\n")
                self.input_fd.flush()
            except:
                pass
        
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=2)


class SubprocessTokenizerPool:
    """
    Пул токенизаторов на subprocess.
    """
    
    def __init__(self, model_path: str, num_workers: int = 2):
        self.model_path = model_path
        self.num_workers = num_workers
        self.workers: List[SubprocessTokenizerWorker] = []
        self._running = False
        
    def start(self) -> bool:
        """Запускает пул"""
        self._running = True
        
        for i in range(self.num_workers):
            worker = SubprocessTokenizerWorker(i, self.model_path)
            if worker.start():
                self.workers.append(worker)
        
        logger.info(f"Запущено {len(self.workers)} воркеров")
        return len(self.workers) > 0
    
    def stop(self):
        """Останавливает пул"""
        self._running = False
        for w in self.workers:
            w.stop()
        self.workers.clear()
    
    def tokenize(self, text: str) -> List[int]:
        """Токенизирует через пул (round-robin)"""
        if not self.workers:
            return []
        
        # Round-robin
        idx = 0
        worker = self.workers[idx % len(self.workers)]
        return worker.tokenize(text)
    
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Токенизирует батч"""
        results = []
        for i, text in enumerate(texts):
            worker = self.workers[i % len(self.workers)]
            results.append(worker.tokenize(text))
        return results


class ThreadTokenizerPool:
    """
    Пул токенизаторов на потоках.
    Для CPU-bound задач менее эффективен, но проще.
    """
    
    def __init__(self, model_path: str, num_threads: int = 2):
        self.model_path = model_path
        self.num_threads = num_threads
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=num_threads)
        
    def tokenize(self, text: str) -> List[int]:
        """Синхронная токенизация"""
        with self._lock:
            return self.tokenizer.encode(text, add_special_tokens=False)
    
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Батч через потоки"""
        futures = [self._executor.submit(self.tokenize, t) for t in texts]
        return [f.result() for f in futures]
    
    def stop(self):
        self._executor.shutdown(wait=True)


class FastTokenizer:
    """
    Быстрый токенизатор с кэшированием и batch-обработкой.
    """
    
    def __init__(self, model_path: str, cache_size: int = 5000):
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # LRU кэш
        self._cache: Dict[str, List[int]] = {}
        self._cache_size = cache_size
        self._cache_order = []
        self._cache_lock = threading.Lock()
        
    def tokenize(self, text: str) -> List[int]:
        """Токенизация с кэшем"""
        key = text[:200]  # Хэшируем по первым 200 символам
        
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        with self._cache_lock:
            if len(self._cache) < self._cache_size:
                self._cache[key] = tokens
            elif self._cache_size > 0:
                # LRU вытеснение
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
                self._cache_order.append(key)
                self._cache[key] = tokens
        
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Батч токенизация"""
        return [self.tokenize(t) for t in texts]


def create_tokenizer_pool(
    model_path: str,
    strategy: str = "cached"
) -> Any:
    """
    Фабрика токенизаторов.
    """
    logger.info(f"Создание токенизатора: strategy={strategy}")
    
    if strategy == "subprocess":
        pool = SubprocessTokenizerPool(model_path, num_workers=2)
        pool.start()
        return pool
    elif strategy == "thread":
        return ThreadTokenizerPool(model_path, num_threads=2)
    elif strategy == "cached":
        return FastTokenizer(model_path)
    else:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# ============================================================================
# Тесты
# ============================================================================

def test_tokenizer_speed():
    """Тест скорости токенизации"""
    # Определяем путь
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    
    logger.info(f"Путь к модели: {model_path}")
    logger.info(f"Существует: {os.path.exists(model_path)}")
    
    logger.info("=== Тест скорости токенизации ===")
    
    # Тест простого токенизатора
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    text = "Привет! ЕVA - это продвинутая система искусственного интеллекта. Она способна анализировать информацию, обучаться и давать осмысленные ответы на вопросы пользователей."
    
    # Одиночная токенизация
    start = time.time()
    for _ in range(100):
        tokens = tokenizer.encode(text, add_special_tokens=False)
    single_time = time.time() - start
    
    logger.info(f"100x одиночная токенизация: {single_time:.3f}s ({100/single_time:.1f}/s)")
    logger.info(f"Длина токенов: {len(tokens)}")
    
    # Батч токенизация
    texts = [text] * 20
    
    start = time.time()
    batches = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    batch_time = time.time() - start
    
    logger.info(f"Батч 20 текстов: {batch_time:.3f}s")
    
    # Тест с кэшем
    fast_tok = FastTokenizer(model_path)
    
    start = time.time()
    for _ in range(100):
        tokens = fast_tok.tokenize(text)
    cached_time = time.time() - start
    
    logger.info(f"100x с кэшем: {cached_time:.3f}s ({100/cached_time:.1f}/s)")
    
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tokenizer_speed()