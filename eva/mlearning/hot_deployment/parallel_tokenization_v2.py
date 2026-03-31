"""
Параллельная токенизация с привязкой к ядрам CPU.
Использует multiprocessing для разделения по физическим ядрам.
"""
import os
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Array, Value
import threading
import queue
import logging
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

logger = logging.getLogger("eva.mlearning.parallel_tokenization_v2")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_physical_cores() -> int:
    """Получает количество физических ядер"""
    try:
        return psutil.cpu_count(logical=False) or os.cpu_count() or 2
    except Exception:
        return os.cpu_count() or 2


def set_process_affinity(pid: int, core_ids: List[int]) -> bool:
    """Устанавливает привязку процесса к ядрам"""
    if not HAS_PSUTIL:
        return False
    try:
        p = psutil.Process(pid)
        p.cpu_affinity(core_ids)
        return True
    except Exception as e:
        logger.warning(f"Не удалось установить affinity: {e}")
        return False


class TokenizerWorker:
    """
    Воркер токенизатора - отдельный процесс с токенизатором.
    """
    
    def __init__(self, model_path: str, core_id: int):
        self.model_path = model_path
        self.core_id = core_id
        self.tokenizer = None
        self.pid = os.getpid()
        
    def initialize(self) -> bool:
        """Инициализирует токенизатор в этом процессе"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Устанавливаем привязку к ядру
            set_process_affinity(os.getpid(), [self.core_id])
            
            logger.info(f"Воркер PID={self.pid} привязан к ядру {self.core_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации воркера: {e}")
            return False
    
    def tokenize(self, text: str) -> List[int]:
        """Токенизирует текст"""
        if self.tokenizer is None:
            return []
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return []
    
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Токенизирует батч"""
        return [self.tokenize(t) for t in texts]


class CoreBoundedTokenizerPool:
    """
    Пул токенизаторов с привязкой к конкретным ядрам CPU.
    
    Каждый воркер работает в отдельном процессе на своём ядре.
    """
    
    def __init__(
        self,
        model_path: str,
        num_workers: Optional[int] = None,
        cores: Optional[List[int]] = None
    ):
        self.model_path = model_path
        
        # Определяем количество воркеров
        phys_cores = get_physical_cores()
        if num_workers is None:
            num_workers = min(phys_cores, 4)  # Максимум 4 воркера
        self.num_workers = min(num_workers, phys_cores)
        
        # Определяем ядра
        if cores is None:
            # Используем первые N ядер
            self.cores = list(range(self.num_workers))
        else:
            self.cores = cores[:self.num_workers]
        
        # Очереди для коммуникации
        self._input_queue: Queue = Queue(maxsize=128)
        self._output_queue: Queue = Queue(maxsize=128)
        
        # Процессы
        self._workers: List[Process] = []
        self._running = Value('b', False)
        
        logger.info(f"Пул токенизаторов: {self.num_workers} воркеров на ядрах {self.cores}")
    
    def start(self):
        """Запускает воркеры"""
        if self._running.value:
            logger.warning("Пул уже запущен")
            return
        
        self._running.value = True
        
        for i in range(self.num_workers):
            p = Process(
                target=self._worker_loop,
                args=(self.model_path, self.cores[i], self._input_queue, self._output_queue),
                name=f"Tokenizer-{i}"
            )
            p.start()
            self._workers.append(p)
        
        logger.info(f"Запущено {len(self._workers)} воркеров")
    
    def stop(self):
        """Останавливает воркеры"""
        self._running.value = False
        
        # Посылаем сигнал остановки
        for _ in range(self.num_workers):
            try:
                self._input_queue.put_nowait((None, None, "STOP"))
            except queue.Full:
                pass
        
        # Ждём завершения
        for p in self._workers:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
        
        self._workers.clear()
        logger.info("Воркеры остановлены")
    
    def tokenize(self, text: str, timeout: float = 5.0) -> List[int]:
        """Токенизирует текст через пул"""
        if not self._running.value:
            return []
        
        result_queue = queue.Queue(maxsize=1)
        
        try:
            self._input_queue.put_nowait((text, result_queue, "TOKENIZE"))
            token_ids = result_queue.get(timeout=timeout)
            return token_ids
        except queue.Empty:
            logger.warning("Таймаут токенизации")
            return []
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return []
    
    def tokenize_async(self, text: str, callback=None):
        """Асинхронная токенизация с колбэком"""
        if not self._running.value:
            if callback:
                callback([])
            return
        
        def _worker():
            try:
                result = self.tokenize(text)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"Ошибка async токенизации: {e}")
                if callback:
                    callback([])
        
        threading.Thread(target=_worker, daemon=True).start()
    
    @staticmethod
    def _worker_loop(model_path: str, core_id: int, input_q: Queue, output_q: Queue):
        """Основной цикл воркера"""
        # Устанавливаем affinity
        set_process_affinity(os.getpid(), [core_id])
        
        # Инициализируем токенизатор
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Воркер на ядре {core_id} инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации воркера: {e}")
            return
        
        # Основной цикл
        while True:
            try:
                text, result_queue, cmd = input_q.get(timeout=1.0)
                
                if cmd == "STOP":
                    break
                
                if cmd == "TOKENIZE" and text is not None:
                    try:
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        if result_queue:
                            result_queue.put(tokens)
                    except Exception as e:
                        logger.error(f"Ошибка в воркере: {e}")
                        if result_queue:
                            result_queue.put([])
                            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в цикле воркера: {e}")


class ThreadPoolTokenizer:
    """
    Пул токенизаторов на потоках (для сравнения).
    Менее эффективен из-за GIL, но проще в реализации.
    """
    
    def __init__(self, model_path: str, num_threads: int = 2):
        self.model_path = model_path
        self.num_threads = num_threads
        
        # Единый токенизатор (потокобезопасный)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Пул потоков
        self._executor = ThreadPoolExecutor(max_workers=num_threads)
        
        logger.info(f"ThreadPoolTokenizer: {num_threads} потоков")
    
    def tokenize(self, text: str) -> List[int]:
        """Синхронная токенизация"""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Параллельная токенизация батча"""
        futures = [self._executor.submit(self.tokenize, t) for t in texts]
        return [f.result() for f in futures]
    
    def stop(self):
        """Остановка пула"""
        self._executor.shutdown(wait=True)


class CachedTokenizer:
    """
    Токенизатор с кэшированием результатов.
    Передовая токенизация для повторяющихся текстов.
    """
    
    def __init__(self, model_path: str, cache_size: int = 10000):
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
        self._cache_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
        logger.info(f"CachedTokenizer: cache_size={cache_size}")
    
    def tokenize(self, text: str) -> List[int]:
        """Токенизация с кэшированием"""
        # Хэш текста как ключ
        key = hash(text)
        
        with self._cache_lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
        
        # Токенизация
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        with self._cache_lock:
            self._misses += 1
            
            # Добавляем в кэш
            if len(self._cache) < self._cache_size:
                self._cache[key] = tokens
        
        return tokens
    
    def get_stats(self) -> Dict:
        """Статистика кэша"""
        with self._cache_lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache)
            }


# ============================================================================
# Фабрика для создания оптимального токенизатора
# ============================================================================

def create_optimal_tokenizer(
    model_path: str,
    strategy: str = "auto"
) -> Any:
    """
    Создаёт оптимальный токенизатор в зависимости от стратегии.
    
    Strategies:
    - "parallel" - Multiprocessing пул с привязкой к ядрам
    - "thread" - ThreadPoolExecutor
    - "cached" - LRU кэш
    - "hybrid" - Комбинация: кэш + пул потоков
    - "auto" - Автоматический выбор
    """
    phys_cores = get_physical_cores()
    
    if strategy == "auto":
        # Автоматический выбор
        if phys_cores >= 4:
            strategy = "parallel"
        else:
            strategy = "thread"
    
    logger.info(f"Создание токенизатора: strategy={strategy}, cores={phys_cores}")
    
    if strategy == "parallel":
        pool = CoreBoundedTokenizerPool(model_path, num_workers=min(phys_cores, 4))
        pool.start()
        return pool
    elif strategy == "thread":
        return ThreadPoolTokenizer(model_path, num_threads=min(phys_cores, 2))
    elif strategy == "cached":
        return CachedTokenizer(model_path)
    elif strategy == "hybrid":
        # Гибрид: кэш + потоки
        return CachedTokenizer(model_path)
    else:
        # Простой токенизатор
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# ============================================================================
# Тесты
# ============================================================================

def test_parallel_tokenizer():
    """Тест параллельного токенизатора"""
    import sys
    model_path = os.path.join(
        os.path.dirname(__file__), "eva_models", "qwen3.5-0.8b"
    )
    
    logger.info("=== Тест Parallel Tokenizer ===")
    
    # Создаём пул
    pool = create_optimal_tokenizer(model_path, strategy="parallel")
    
    # Токенизация
    texts = [
        "Привет, как дела?",
        "Что такое машинное обучение?",
        "ЕВА - искусственный интеллект",
        "Параллельная токенизация работает",
        "Тестируем скорость"
    ]
    
    start = time.time()
    for text in texts:
        tokens = pool.tokenize(text)
        logger.info(f"'{text[:20]}...' -> {len(tokens)} токенов")
    elapsed = time.time() - start
    
    logger.info(f"Время токенизации {len(texts)} текстов: {elapsed:.2f}s")
    
    # Остановка
    if hasattr(pool, 'stop'):
        pool.stop()
    
    return True


def test_cached_tokenizer():
    """Тест кэшированного токенизатора"""
    model_path = os.path.join(
        os.path.dirname(__file__), "eva_models", "qwen3.5-0.8b"
    )
    
    logger.info("=== Тест Cached Tokenizer ===")
    
    cached = create_optimal_tokenizer(model_path, strategy="cached")
    
    # Первая токенизация (miss)
    text = "Привет, как дела?"
    start = time.time()
    for _ in range(5):
        tokens = cached.tokenize(text)
    elapsed = time.time() - start
    
    logger.info(f"5x токенизация: {elapsed:.3f}s")
    logger.info(f"Статистика: {cached.get_stats()}")
    
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Тесты
    test_parallel_tokenizer()
    test_cached_tokenizer()