"""
Обновленный FractalModelManager с оптимальной конфигурацией
"""
from __future__ import annotations

import os
import json
import time
import logging
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from safetensors.torch import load_file

# Импорты для генерации через transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger("eva.fractal_model_manager")
    logger.warning("Transformers не доступны, генерация будет ограничена")

logger = logging.getLogger("eva.fractal_model_manager_optimized")


class OptimizedFractalModelManager:
    """Оптимизированный FractalModelManager с оптимальной конфигурацией"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Инициализация с оптимальными параметрами"""
        
        # Загружаем оптимальную конфигурацию
        self.config = self._load_optimal_config(config_path)
        
        # Пути к модели
        self.model_path = self.config.get("model_path")
        self.config_path = self.config.get("config_path")
        
        # Устройство и память
        self.device = torch.device(self.config["device"])
        self.max_memory_tokens = self.config["max_memory_tokens"]
        self.target_memory_gb = self.config["target_memory_gb"]
        
        # Оптимизации
        self.cache_tokenization = self.config["cache_tokenization"]
        self.parallel_tokenization = self.config["parallel_tokenization"]
        self.tokenization_workers = self.config["tokenization_workers"]
        self.memory_optimization = self.config["memory_optimization"]
        self.use_uint16 = self.config["use_uint16"]
        self.tensor_pool_size = self.config["tensor_pool_size"]
        
        # Генерация
        self.batch_size = self.config["batch_size"]
        self.max_length = self.config["max_length"]
        self.overlap_tokens = self.config["overlap_tokens"]
        
        # Компоненты
        self.model = None
        self.tokenizer = None
        self.state_dict = None
        self.initialized = False
        self.model_name = "fractal_gpt2_optimized"
        
        # Пулы и кэши
        self.tensor_pool = []
        self.tokenization_cache = {}
        self.tokenization_executor = ThreadPoolExecutor(max_workers=self.tokenization_workers)
        
        # Метрики
        self.performance_stats = {
            "tokenization_time": 0.0,
            "generation_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_saved_mb": 0.0
        }
        
        # Инициализируем компоненты
        self._initialize_components()
        
        logger.info(f"OptimizedFractalModelManager инициализирован с {self.max_memory_tokens} токенов")
    
    def _load_optimal_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Загружает оптимальную конфигурацию"""
        
        # Приоритет: переданный путь > оптимальный > по умолчанию
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Пытаемся загрузить оптимальную конфигурацию
        optimal_config_path = os.path.join(
            os.getcwd(), "eva", "config", "fractal_model_config.json"
        )
        
        if os.path.exists(optimal_config_path):
            with open(optimal_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Конфигурация по умолчанию
        return {
            "model_path": None,
            "config_path": None,
            "device": "cpu",
            "max_memory_tokens": 50000,
            "target_memory_gb": 4.0,
            "cache_tokenization": True,
            "parallel_tokenization": True,
            "tokenization_workers": 4,
            "memory_optimization": True,
            "use_uint16": True,
            "tensor_pool_size": 1000,
            "batch_size": 4,
            "max_length": 32768,
            "overlap_tokens": 64,
            "auto_improvement": True,
            "quality_threshold": 0.7,
            "check_interval_seconds": 30
        }
    
    def _initialize_components(self):
        """Инициализирует компоненты с оптимизациями"""
        
        try:
            # Инициализируем улучшатель качества текста
            try:
                from .text_quality_improver import TextQualityImprover
                
                # Используем директорию модели, а не файл
                model_dir = None
                if self.model_path and os.path.exists(self.model_path):
                    model_dir = os.path.dirname(self.model_path)
                
                self.quality_improver = TextQualityImprover(model_dir)
                logger.info("TextQualityImprover инициализирован")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать TextQualityImprover: {e}")
                self.quality_improver = None
            
            # Инициализируем тренер
            try:
                from .text_quality_trainer import TextQualityTrainer, TrainingConfig
                training_config = TrainingConfig(
                    learning_rate=3e-5,
                    batch_size=2,
                    num_epochs=50,
                    max_length=32768,
                    warmup_steps=100,
                    weight_decay=0.01,
                    save_steps=200,
                    eval_steps=50,
                    gradient_accumulation_steps=4
                )
                # Будет инициализирован после загрузки модели
                self.trainer_config = training_config
                self.trainer = None
                logger.info("TextQualityTrainer готов к инициализации")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать TextQualityTrainer: {e}")
                self.trainer = None
                self.trainer_config = None
            
            # Инициализируем гибридное хранилище токенов
            try:
                from eva.memory.hybrid_token_cache import HybridTokenCache
                
                class TempBrain:
                    def __init__(self):
                        self.cache_dir = os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            'core', 'eva_cache'
                        )
                        self.config = {
                            'hybrid_cache': {
                                'max_memory_tokens': 50000,  # Фиксированное значение
                                'target_memory_gb': 4.0,
                                'dynamic_memory_limit': True,
                                'max_ram_usage_percent': 75.0,
                                'vram_threshold': 0.3,
                                'ram_threshold': 0.2,
                                'eviction_policy': 'lru',
                                'disk_cache_gb': 20.0
                            }
                        }
                
                temp_brain = TempBrain()
                self.token_cache = HybridTokenCache(
                    brain=temp_brain,
                    max_memory_tokens=self.max_memory_tokens,
                    disk_cache_dir="token_cache",
                    target_memory_gb=self.target_memory_gb,
                    dynamic_memory_limit=True,
                    max_ram_usage_percent=75.0
                )
                logger.info(f"Гибридное хранилище токенов инициализировано: {self.max_memory_tokens} токенов")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать гибридное хранилище: {e}")
                self.token_cache = None
            
            # Загружаем модель
            self._initialize_model()
            
        except Exception as e:
            logger.error(f"Критическая ошибка при инициализации компонентов: {e}", exc_info=True)
    
    def _initialize_model(self):
        """Инициализирует модель с оптимизациями"""
        
        try:
            logger.info(f"Загрузка модели из {self.model_path}...")
            
            # Определяем пути к модели
            if not self.model_path:
                module_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(module_dir))
                
                # Проверяем доступные модели
                possible_models = [
                    os.path.join(project_root, "out", "fractal_rugpt_full.safetensors"),
                    os.path.join(project_root, "text-generation", "model.safetensors"),
                    os.path.join(project_root, "fractal_unified_text-generation", "adapted_model", "model.safetensors"),
                    os.path.join(project_root, "out", "smoke.safetensors")
                ]
                
                # Находим существующий файл
                for model_file in possible_models:
                    if os.path.exists(model_file):
                        self.model_path = model_file
                        break
                
                if not self.model_path:
                    # Используем первый по умолчанию
                    self.model_path = possible_models[0]
                
                # Конфигурация
                self.config_path = self.model_path.replace('.safetensors', '.json')
                if not os.path.exists(self.config_path):
                    self.config_path = None
            
            # Проверяем существование файлов
            if not os.path.exists(self.model_path):
                logger.error(f"Файл модели не найден: {self.model_path}")
                return
            
            # Загружаем веса модели
            self.state_dict = load_file(self.model_path, device="cpu")
            
            if not self.state_dict:
                logger.error("Не удалось загрузить state_dict модели")
                return
            
            # Создаем модель
            if TRANSFORMERS_AVAILABLE:
                self._create_optimized_model()
            else:
                logger.warning("Transformers не доступны - генерация ограничена")
                return
            
            # Инициализируем тренер после загрузки модели
            if hasattr(self, 'trainer_config') and self.trainer_config:
                try:
                    from .text_quality_trainer import TextQualityTrainer
                    self.trainer = TextQualityTrainer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=self.trainer_config
                    )
                    logger.info("TextQualityTrainer инициализирован")
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать тренер: {e}")
                    self.trainer = None
            
            self.initialized = True
            logger.info(f"Оптимизированная модель успешно загружена: {len(self.state_dict)} параметров")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации модели: {e}", exc_info=True)
            self.initialized = False
    
    def _create_optimized_model(self):
        """Создает оптимизированную модель"""
        
        # Определяем параметры модели
        vocab_size = None
        n_embd = None
        n_layer = None
        n_head = None
        
        for key, tensor in self.state_dict.items():
            if 'wte.weight' in key:
                vocab_size = tensor.shape[0]
                n_embd = tensor.shape[1]
            elif key == 'transformer.h.0.attn.c_attn.weight':
                n_embd = tensor.shape[1] if n_embd is None else n_embd
        
        # Определяем количество слоев
        layer_keys = [k for k in self.state_dict.keys() if k.startswith('transformer.h.') and '.attn.c_attn.weight' in k]
        n_layer = len(layer_keys)
        
        # Определяем количество голов
        if 'transformer.h.0.attn.c_attn.weight' in self.state_dict:
            attn_weight = self.state_dict['transformer.h.0.attn.c_attn.weight']
            n_head = n_embd // 64 if n_embd and n_embd % 64 == 0 else 12
        
        # Устанавливаем значения по умолчанию
        vocab_size = vocab_size or 50264
        n_embd = n_embd or 768
        n_layer = n_layer or 12
        n_head = n_head or 12
        
        # Определяем количество позиций из весов
        n_positions = 1024  # По умолчанию
        for key, tensor in self.state_dict.items():
            if 'wpe.weight' in key:
                n_positions = tensor.shape[0]  # Используем реальный размер из весов
                break
        
        # Создаем конфигурацию
        self.config_obj = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions,  # Используем реальное значение
            n_ctx=n_positions,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=True,
        )
        
        # Создаем модель
        self.model = GPT2LMHeadModel(self.config_obj)
        
        # Загружаем веса
        missing_keys, unexpected_keys = self.model.load_state_dict(self.state_dict, strict=False)
        
        if missing_keys:
            bias_missing = [k for k in missing_keys if 'bias' in k]
            other_missing = [k for k in missing_keys if 'bias' not in k]
            
            if bias_missing and not other_missing:
                logger.info(f"Отсутствуют только bias параметры ({len(bias_missing)} шт.)")
            elif other_missing:
                logger.warning(f"Отсутствуют важные параметры: {other_missing}")
        
        if unexpected_keys:
            logger.warning(f"Неожиданные ключи: {unexpected_keys}")
        
        # Переносим на устройство
        self.model.to(self.device)
        self.model.eval()
        
        # Загружаем токенизатор
        self._load_optimized_tokenizer(vocab_size)
        
        logger.info(f"Оптимизированная модель создана: vocab_size={vocab_size}, n_embd={n_embd}, n_layer={n_layer}")
    
    def _load_optimized_tokenizer(self, vocab_size: int):
        """Загружает оптимизированный токенизатор"""
        
        # Ищем токенизатор в фрактальном хранилище
        tokenizer_path = self._find_tokenizer_in_fractal_storage()
        
        if tokenizer_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                
                # Проверяем совместимость vocab_size
                if hasattr(self.tokenizer, 'vocab_size'):
                    vocab_diff = abs(self.tokenizer.vocab_size - vocab_size)
                    if vocab_diff <= 10:
                        logger.info(f"Совместимость токенизатора: разница vocab_size={vocab_diff}")
                        
                        # Исправляем vocab_size если нужно
                        if self.tokenizer.vocab_size < vocab_size:
                            for i in range(self.tokenizer.vocab_size, vocab_size):
                                self.tokenizer.add_tokens([f"<extra_token_{i}>"])
                            logger.info(f"Добавлено {vocab_diff} токенов")
                    else:
                        logger.warning(f"Несовместимость vocab_size: разница={vocab_diff}")
                
                logger.info("Оптимизированный токенизатор загружен")
                
            except Exception as e:
                logger.warning(f"Не удалось загрузить токенизатор из фрактального хранилища: {e}")
                self.tokenizer = self._load_fallback_tokenizer()
        else:
            self.tokenizer = self._load_fallback_tokenizer()
    
    def _load_fallback_tokenizer(self):
        """Загружает запасной токенизатор"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3large_based_on_gpt2')
            logger.info("Запасной токенизатор ruGPT3 загружен")
        except Exception as e:
            logger.warning(f"Не удалось загрузить запасной токенизатор: {e}")
            return None
        return self.tokenizer
    
    def _find_tokenizer_in_fractal_storage(self) -> Optional[str]:
        """Ищет токенизатор во фрактальном хранилище"""
        base_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "text-generation"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "..", "text-generation"),
            os.path.join(os.getcwd(), "text-generation"),
        ]
        
        for path in base_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def optimized_tokenize(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Оптимизированная токенизация с кэшированием и параллелизацией"""
        
        start_time = time.time()
        
        if not self.cache_tokenization or not self.parallel_tokenization:
            # Стандартная токенизация
            results = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                results.append({
                    'input_ids': inputs['input_ids'].to(self.device),
                    'attention_mask': inputs['attention_mask'].to(self.device),
                    'text': text
                })
            
            self.performance_stats["tokenization_time"] += time.time() - start_time
            return results
        
        # Проверяем кэш
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.tokenization_cache:
                cached = self.tokenization_cache[text_hash]
                cached_results.append({
                    'input_ids': cached['input_ids'].to(self.device),
                    'attention_mask': cached['attention_mask'].to(self.device),
                    'text': text,
                    'cached': True
                })
                self.performance_stats["cache_hits"] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.performance_stats["cache_misses"] += 1
        
        # Параллельная токенизация
        if uncached_texts:
            def tokenize_single(text):
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                return {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'text': text,
                    'cached': False
                }
            
            futures = [self.tokenization_executor.submit(tokenize_single, text) for text in uncached_texts]
            new_results = [future.result() for future in futures]
            
            # Кэшируем результаты
            for text, result in zip(uncached_texts, new_results):
                text_hash = hash(text)
                self.tokenization_cache[text_hash] = {
                    'input_ids': result['input_ids'].cpu(),
                    'attention_mask': result['attention_mask'].cpu()
                }
                
                # Перемещаем на устройство
                result['input_ids'] = result['input_ids'].to(self.device)
                result['attention_mask'] = result['attention_mask'].to(self.device)
            
            # Объединяем результаты
            all_results = cached_results + new_results
            all_results.sort(key=lambda x: texts.index(x['text']))
        else:
            all_results = cached_results
        
        self.performance_stats["tokenization_time"] += time.time() - start_time
        
        return all_results
    
    def generate_response_optimized(self, query: str, max_new_tokens: int = 2048) -> str:
        """Оптимизированная генерация ответа"""
        
        if not self.initialized:
            return "Модель не инициализирована"
        
        start_time = time.time()
        
        try:
            # Оптимизированная токенизация
            tokenized = self.optimized_tokenize([query])[0]
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            # Генерируем ответ
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Декодируем ответ
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Очищаем ответ
            response = self._clean_response(query, generated_text)
            
            # Обновляем статистику
            self.performance_stats["generation_time"] += time.time() - start_time
            
            logger.info(f"Оптимизированная генерация завершена за {time.time() - start_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка оптимизированной генерации: {e}", exc_info=True)
            return f"Ошибка генерации: {str(e)}"
    
    def _clean_response(self, query: str, generated_text: str) -> str:
        """Очищает сгенерированный ответ"""
        
        # Убираем запрос из ответа
        if generated_text.lower().startswith(query.lower()):
            response = generated_text[len(query):].strip()
        else:
            response = generated_text
        
        # Базовая очистка
        import re
        response = re.sub(r'[^\w\s\.\,\!\?\;\:\-\—\nа-яА-ЯёЁ]', '', response)
        response = re.sub(r'\s+', ' ', response)
        
        return response.strip()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        
        stats = self.performance_stats.copy()
        
        # Добавляем текущее состояние
        stats.update({
            "initialized": self.initialized,
            "device": str(self.device),
            "max_memory_tokens": self.max_memory_tokens,
            "cache_size": len(self.tokenization_cache),
            "tensor_pool_size": len(self.tensor_pool),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / 
                (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
                if (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]) > 0 else 0
            )
        })
        
        return stats
    
    def optimize_memory(self):
        """Оптимизирует использование памяти"""
        
        if not self.memory_optimization:
            return
        
        # Очищаем кэш токенизации если он слишком большой
        if len(self.tokenization_cache) > 1000:
            # Оставляем только последние 500 записей
            items = list(self.tokenization_cache.items())
            self.tokenization_cache = dict(items[-500:])
            
            saved_memory = len(items) - 500
            self.performance_stats["memory_saved_mb"] += saved_memory * 0.001  # Приблизительно
            
            logger.info(f"Оптимизирован кэш токенизации: освобождено {saved_memory} записей")
        
        # Очищаем пул тензоров
        if len(self.tensor_pool) > self.tensor_pool_size:
            self.tensor_pool = self.tensor_pool[-self.tensor_pool_size//2:]
            logger.info(f"Оптимизирован пул тензоров: размер {len(self.tensor_pool)}")
    
    def __del__(self):
        """Очистка при удалении"""
        
        try:
            if hasattr(self, 'tokenization_executor'):
                self.tokenization_executor.shutdown(wait=True)
        except Exception:
            pass
        
        logger.info("OptimizedFractalModelManager очищен")
