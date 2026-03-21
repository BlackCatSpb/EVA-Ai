import os
import time
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoConfig

# Relative imports
from ..memory.hybrid_token_cache import HybridTokenCache, get_shared_cache
from ..mlearning.parallel_tokenization import ParallelTokenizer
from ..memory.disk_cache import DiskCache

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationCoordinator:
    def __init__(
        self,
        model_name: str = "sberbank-ai/rugpt3large_based_on_gpt2",
        num_workers: int = 4,
        cache_dir: str = "./cache",
        max_cache_size_gb: int = 50,
        brain=None
    ):
        """
        Инициализация координатора генерации с использованием существующей инфраструктуры кеширования
        :param model_name: Название модели
        :param num_workers: Количество воркеров для обработки промптов
        :param cache_dir: Директория для кеша
        :param max_cache_size_gb: Максимальный размер кеша в ГБ
        :param brain: Ссылка на объект Brain (если есть)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = num_workers
        self.is_ready = False
        self.init_error = None
        
        # Инициализация гибридного кеша
        self.brain = brain or self._create_mock_brain(cache_dir, max_cache_size_gb)
        existing = getattr(self.brain, 'token_cache', None) or getattr(self.brain, 'hybrid_cache', None)
        if existing is not None:
            self.cache = existing
        else:
            self.cache = get_shared_cache(self.brain, "default")
        
        self.model = None
        self.model_config = None
        
        # Инициализация токенизатора с использованием ruGPT-3 Large из фрактального хранилища
        try:
            # Проверяем наличие ruGPT-3 Large во фрактальном хранилище
            # ВАЖНО: модель находится в models/rugpt3_small_fractal/model/
            possible_paths = [
                'cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal/model',
                'cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal',
                'cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_small_fractal',
                'cogniflex/mlearning/cogniflex_models/fractal_unified_text-generation',
            ]
            
            local_model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    local_model_path = path
                    logger.info(f"Найдена модель по пути: {os.path.abspath(path)}")
                    break
            
            if not local_model_path:
                raise FileNotFoundError(f"Локальная модель не найдена. Проверенные пути:\n" + 
                                    "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths))
            
            logger.info(f"Используем модель: {os.path.abspath(local_model_path)}")
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                local_files_only=True,
                trust_remote_code=False
            )
            logger.info(f"Токенизатор успешно загружен из: {local_model_path}")
            
            # Обновляем имя модели для соответствия ruGPT-3 Small
            if 'rugpt3_small' in local_model_path:
                self.model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'
                logger.info(f"Обновлено имя модели на: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки токенизатора: {e}")
            raise RuntimeError("Не удалось загрузить токенизатор. Проверьте наличие локальной модели.")
            
        # Инициализация параллельного токенизатора
        self.parallel_tokenizer = ParallelTokenizer(
            brain=self.brain,
            max_data_window_bytes=3 * 1024**3,  # 3GB
            worker_count=self.num_workers
        )
            
    def check_generation_ready(self) -> Tuple[bool, Optional[str]]:
        """
        Проверяет готовность системы к генерации
        Returns:
            Tuple[bool, Optional[str]]: (готовность, сообщение об ошибке)
        """
        try:
            # Проверяем состояние моделей
            if self.brain and hasattr(self.brain, 'model_manager'):
                for model_id in ['generator', 'ethics', 'knowledge']:
                    state = self.brain.model_manager.model_states.get(model_id)
                    if state == "loading":
                        return False, "Модели ещё загружаются. Пожалуйста, дождитесь готовности."
                    elif state == "error":
                        error = self.brain.model_manager.loading_errors.get(model_id, "недоступно")
                        return False, f"Ошибка загрузки модели {model_id}: {error}"
                    elif not state:
                        return False, f"Модель {model_id} не загружена"
                        
            # Проверяем состояние памяти
            if not hasattr(self.brain, 'memory_manager') or not self.brain.memory_manager.initialized:
                return False, "Система памяти не инициализирована"
                
            # Проверяем состояние графа знаний
            if not hasattr(self.brain, 'knowledge_graph') or not self.brain.knowledge_graph.is_initialized:
                return False, "Граф знаний не инициализирован"
                
            return True, None
        except Exception as e:
            return False, f"Ошибка проверки готовности: {str(e)}"
            
    def initialize(self):
        """Инициализация и запуск параллельного токенизатора"""
        self.parallel_tokenizer.start()
        logger.info(f"Инициализирован GenerationCoordinator с {self.num_workers} воркерами")
    
    def _create_mock_brain(self, cache_dir: str, max_cache_size_gb: int):
        """Создает минимальный объект Brain с необходимыми атрибутами"""
        class MockBrain:
            def __init__(self, cache_dir):
                self.cache_dir = cache_dir
                self.config = {
                    'hybrid_cache': {
                        'max_memory_size': 10000,
                        'disk_cache_threshold': 5000,
                        'eviction_policy': 'lru',
                        'cache_ttl': 86400,
                        'disk_cache_size': 120000,
                        'min_relevance_score': 0.3,
                        'max_context_tokens': 1000,
                        'target_memory_gb': max_cache_size_gb,
                        'dynamic_memory_limit': True
                    }
                }
                self.resource_queue = None  # Можно заменить на реальную очередь ресурсов
        
        return MockBrain(cache_dir)
        
    def load_model(self):
        """Загрузка модели ruGPT-3 Large из фрактального хранилища"""
        if self.model is not None:
            return True
            
        logger.info(f"Загрузка модели {self.model_name} на {self.device}...")
        start_time = time.time()
        
        try:
            # Сначала пробуем загрузить локальные веса ruGPT-3 Large
            local_model_paths = [
                'cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal/model',
                'cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal',
                'cogniflex/mlearning/cogniflex_models/rugpt3_small',
            ]
            
            local_model_path = None
            for path in local_model_paths:
                if os.path.exists(path):
                    # Проверяем наличие pytorch_model.bin
                    if os.path.exists(os.path.join(path, 'pytorch_model.bin')):
                        local_model_path = path
                        logger.info(f"Найдены локальные веса модели: {os.path.abspath(path)}")
                        break
            
            if local_model_path:
                # Загружаем из локальных весов
                logger.info("Загрузка модели из локальных весов...")
                self.model = GPT2LMHeadModel.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    torch_dtype=torch.float32
                ).to(self.device)
                
                logger.info(f"✅ Модель успешно загружена из локальных весов: {local_model_path}")
            else:
                # Fallback: загрузка из HuggingFace (только если локальные недоступны)
                logger.warning(f"Локальные веса не найдены, загрузка из HuggingFace: {self.model_name}")
                
                # Загружаем конфигурацию модели
                self.model_config = AutoConfig.from_pretrained(
                    self.model_name,
                    max_position_embeddings=2048,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                )
                
                # Загружаем модель с поддержкой ускорения
                if torch.cuda.is_available():
                    self.model = GPT2LMHeadModel.from_pretrained(
                        self.model_name,
                        config=self.model_config,
                        device_map="auto",
                        torch_dtype=torch.float16 if torch.cuda.is_bf16_supported() else torch.float32
                    ).to(self.device)
                else:
                    self.model = GPT2LMHeadModel.from_pretrained(
                        self.model_name,
                        config=self.model_config
                    ).to(self.device)
                
                logger.info(f"✅ Модель загружена из HuggingFace: {self.model_name}")
            
            self.model.eval()
            logger.info(f"Модель загружена за {time.time() - start_time:.2f} сек")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}", exc_info=True)
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 2.0,
        num_return_sequences: int = 1,
        use_cache: bool = True,
        do_sample: bool = False,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True
    ) -> Dict[str, Any]:
        """
        Генерация ответа на промпт с использованием гибридного кеширования
        
        :param prompt: Текст промпта для генерации
        :param max_new_tokens: Максимальное количество новых токенов для генерации
        :param temperature: Температура генерации (0.0-1.0, выше = более случайно)
        :param top_k: Количество топовых токенов для выборки (0 для отключения)
        :param top_p: Ядерная выборка (0.0-1.0), альтернатива top_k
        :param repetition_penalty: Штраф за повторения (1.0 = нет штрафа, >1.0 = меньше повторений)
        :param num_return_sequences: Количество возвращаемых вариантов ответа
        :param use_cache: Использовать ли кеширование для ускорения повторных запросов
        :param do_sample: Включить стохастическую выборку (True) или жадный поиск (False)
        :param num_beams: Количество лучей для beam search (работает только с do_sample=False)
        :param no_repeat_ngram_size: Запрещать n-граммы указанного размера (для уменьшения повторений)
        :param early_stopping: Останавливать генерацию, когда все лучи достигли конца последовательности
        
        :return: Словарь с результатами генерации, содержащий:
                 - status: 'success' или 'error'
                 - generated_text: сгенерированный текст (если успех)
                 - cached: был ли использован кешированный результат
                 - processing_time: время обработки в секундах
                 - model: имя использованной модели
                 - num_generated_tokens: количество сгенерированных токенов
                 - device: устройство, на котором выполнялась генерация
        """
        start_time = time.time()
        cache_key = f"prompt_{hash(prompt) & 0xFFFFFFFF}"
        
        # 1. Проверяем кеш, если включено кеширование
        if use_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                logger.info("Используем кешированный ответ")
                return {
                    'status': 'success',
                    'generated_text': cached_response,
                    'cached': True,
                    'processing_time': 0.0,
                    'model': self.model_name
                }
        
        # 2. Загружаем модель, если она еще не загружена
        if self.model is None and not self.load_model():
            return {
                'status': 'error',
                'message': 'Не удалось загрузить модель'
            }
        
        # 3. Токенизируем входные данные
        try:
            logger.info("Токенизация входных данных...")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,  # Отключаем padding для генерации
                truncation=True,
                max_length=4096,  # Увеличено с 1024 до 4096
                return_attention_mask=True
            ).to(self.device)
            
            # 4. Генерация текста
            logger.info("Запуск генерации...")
            
            # Подготавливаем параметры генерации
            generation_params = {
                **inputs,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature if do_sample else 1.0,
                'top_k': top_k if do_sample else 0,  # top_k=0 отключается при do_sample=False
                'top_p': top_p if do_sample else 1.0,
                'repetition_penalty': repetition_penalty,
                'num_return_sequences': num_return_sequences,
                'pad_token_id': self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                'use_cache': True,
                'do_sample': do_sample,
                'num_beams': num_beams if not do_sample else 1,  # beam search работает только с do_sample=False
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'early_stopping': early_stopping
            }
            
            # Удаляем None значения из параметров
            generation_params = {k: v for k, v in generation_params.items() if v is not None}
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_params)
            
            # 5. Декодируем сгенерированный текст
            generated_texts = []
            for i, output in enumerate(outputs):
                # Пропускаем промпт в выводе
                prompt_length = inputs['input_ids'].shape[1]
                generated_tokens = output[prompt_length:]
                
                # Декодируем токены в текст
                text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                generated_texts.append(text.strip())
            
            # Если сгенерирован только один вариант, возвращаем строку
            result_text = generated_texts[0] if num_return_sequences == 1 else generated_texts
            
            # Сохраняем в кеш, если включено кеширование
            if use_cache and result_text:
                self.cache.set(cache_key, result_text)
            
            return {
                'status': 'success',
                'generated_text': result_text,
                'processing_time': time.time() - start_time,
                'num_generated_tokens': len(generated_tokens) if num_return_sequences == 1 
                                    else [len(t) for t in generated_texts],
                'model': self.model_name,
                'device': self.device,
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Ошибка при генерации: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Ошибка при генерации: {str(e)}',
                'processing_time': time.time() - start_time
            }
    
    def clear_cache(self):
        """Очистка кеша и освобождение ресурсов"""
        try:
            if hasattr(self, 'cache') and self.cache is not None:
                # Проверяем наличие метода cleanup в HybridTokenCache
                if hasattr(self.cache, 'cleanup'):
                    self.cache.cleanup()
                elif hasattr(self.cache, 'clear'):
                    self.cache.clear()
                logger.info("Кеш успешно очищен")
            
            # Останавливаем параллельный токенизатор, если он был инициализирован
            if hasattr(self, 'parallel_tokenizer') and self.parallel_tokenizer is not None:
                if hasattr(self.parallel_tokenizer, 'stop'):
                    self.parallel_tokenizer.stop()
                    logger.info("Параллельный токенизатор остановлен")
                
            return True
        except Exception as e:
            logger.error(f"Ошибка при очистке кеша: {e}", exc_info=True)
            return False
    
    def cleanup(self):
        """Очистка ресурсов и освобождение памяти"""
        try:
            # Очищаем кеш
            self.clear_cache()
            
            # Выгружаем модель с GPU
            if hasattr(self, 'model') and self.model is not None:
                # Если модель поддерживает очистку, используем её
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')
                del self.model
                self.model = None
                
            # Очищаем токенизатор
            if hasattr(self, 'tokenizer'):
                if hasattr(self.tokenizer, 'save_pretrained'):
                    try:
                        self.tokenizer.save_pretrained(os.path.join(os.getcwd(), 'tokenizer_cache'))
                    except Exception as e:
                        logger.warning(f"Не удалось сохранить токенизатор: {e}")
                self.tokenizer = None
            
            # Очищаем кэш CUDA, если доступен
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Принудительный сбор мусора
            import gc
            gc.collect()
                
            logger.info("Ресурсы успешно освобождены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при освобождении ресурсов: {e}", exc_info=True)
            return False
        
    def __del__(self):
        """Деструктор - вызывается при удалении объекта"""
        self.cleanup()

def main():
    """Основная функция для демонстрации работы генератора"""
    coordinator = None
    try:
        # Инициализируем координатор
        print("Инициализация координатора генерации...")
        coordinator = GenerationCoordinator(
            model_name="sberbank-ai/rugpt3large_based_on_gpt2",
            num_workers=4,
            cache_dir="./cache",
            max_cache_size_gb=50
        )
        
        # Улучшенный промпт с вопросом о графе памяти
        prompt = """
        Искусственный интеллект и графы знаний:
        
        Графы памяти (Knowledge Graphs) играют важную роль в современных системах ИИ, 
        обеспечивая структурированное представление знаний и взаимосвязей между сущностями.
        
        Пожалуйста, напиши развернутый анализ на следующие темы:
        1. Как графы памяти используются в современных языковых моделях?
        2. Какие преимущества дает использование графов памяти в системах генерации текста?
        3. Какие существуют подходы к интеграции графов знаний в архитектуру нейронных сетей?
        4. Каковы перспективы развития графовых методов в ИИ?
        
        Проанализируй каждый аспект подробно, приведи конкретные примеры и технические детали.
        """
        
        # Генерируем ответ с увеличенным лимитом токенов
        print("\nГенерация развернутого ответа...")
        start_time = time.time()
        
        result = coordinator.generate_response(
            prompt=prompt,
            max_new_tokens=1000,  # Увеличили лимит токенов
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            use_cache=True,
            do_sample=True,
            num_beams=2  # Используем beam search для более качественной генерации
        )
        
        # Выводим результат
        if result['status'] == 'success':
            print("\n=== СГЕНЕРИРОВАННЫЙ ТЕКСТ ===")
            print(result['generated_text'])
            print("\n=== МЕТАДАННЫЕ ===")
            print(f"Статус: {result['status']}")
            print(f"Время обработки: {result['processing_time']:.2f} сек")
            print(f"Сгенерировано токенов: {result['num_generated_tokens']}")
            print(f"Устройство: {result['device']}")
            print(f"Использован кеш: {'да' if result.get('cached', False) else 'нет'}")
        else:
            print(f"\nОшибка: {result.get('message', 'Неизвестная ошибка')}")
            
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Всегда освобождаем ресурсы
        if coordinator is not None:
            print("\nОчистка ресурсов...")
            coordinator.cleanup()
            print("Ресурсы освобождены")

if __name__ == "__main__":
    main()
