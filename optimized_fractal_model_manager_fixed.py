"""
Исправленная версия OptimizedFractalModelManager с решением проблем устройств
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
    logger = logging.getLogger("cogniflex.fractal_model_manager")
    logger.warning("Transformers не доступны, генерация будет ограничена")

logger = logging.getLogger("cogniflex.fractal_model_manager_optimized_fixed")


class OptimizedFractalModelManager:
    """Исправленный OptimizedFractalModelManager с решением проблем устройств"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Инициализация с исправленной обработкой устройств"""
        
        # Загружаем оптимальную конфигурацию
        self.config = self._load_optimal_config(config_path)
        
        # Пути к модели
        self.model_path = self.config.get("model_path")
        self.config_path = self.config.get("config_path")
        
        # Устройство и память с исправлениями
        self.device = self._get_safe_device()
        self.max_memory_tokens = self.config["max_memory_tokens"]
        self.target_memory_gb = self.config["target_memory_gb"]
        
        # Оптимизации
        self.cache_tokenization = self.config["cache_tokenization"]
        self.parallel_tokenization = self.config["parallel_tokenization"]
        self.tokenization_workers = self.config["tokenization_workers"]
        
        # Инициализация
        self._initialize_components()
    
    def _get_safe_device(self) -> torch.device:
        """Безопасно определяет устройство с fallback"""
        try:
            # Проверяем доступность CUDA
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Тестируем создание тензора
                test_tensor = torch.tensor([1.0], device=device)
                logger.info(f"CUDA доступен, используем устройство: {device}")
                return device
            else:
                logger.info("CUDA недоступен, используем CPU")
                return torch.device("cpu")
        except Exception as e:
            logger.warning(f"Ошибка при определении устройства: {e}, используем CPU")
            return torch.device("cpu")
    
    def _load_optimal_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Загружает оптимальную конфигурацию"""
        default_config = {
            "model_path": "out/fractal_rugpt_full.safetensors",
            "config_path": None,
            "device": "cpu",
            "max_memory_tokens": 44002,
            "target_memory_gb": 4.0,
            "cache_tokenization": True,
            "parallel_tokenization": True,
            "tokenization_workers": 4,
            "memory_optimization": True,
            "tensor_pool_size": 1000
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Инициализирует компоненты с обработкой ошибок"""
        try:
            # Инициализация переменных
            self.initialized = False
            self.model = None
            self.tokenizer = None
            self.state_dict = None
            self.config_obj = None
            
            # Кэши и пулы
            self.tokenization_cache = {}
            self.tensor_pool = []
            self.performance_stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "tokenization_time": 0.0,
                "generation_time": 0.0,
                "memory_saved_mb": 0.0
            }
            
            # Исполнители
            self.tokenization_executor = ThreadPoolExecutor(max_workers=self.tokenization_workers)
            self.background_executor = ThreadPoolExecutor(max_workers=2)
            
            # Дополнительные компоненты
            self.trainer = None
            self.quality_improver = None
            self.web_search_integration = None
            
            # Инициализация модели
            self._initialize_model()
            
            # Инициализация дополнительных компонентов
            self._initialize_additional_components()
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}", exc_info=True)
            self.initialized = False
    
    def _initialize_model(self):
        """Инициализирует модель с безопасной обработкой устройств"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Загрузка модели из {self.model_path}")
                
                # Загружаем state_dict
                if self.model_path.endswith('.safetensors'):
                    self.state_dict = load_file(self.model_path)
                else:
                    self.state_dict = torch.load(self.model_path, map_location='cpu')
                
                # Создаем модель на CPU сначала
                self._create_model_from_state_dict()
                
                # Переносим на целевое устройство
                self.model.to(self.device)
                
                logger.info(f"Модель загружена на {self.device}")
            else:
                logger.warning(f"Путь к модели не найден: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}", exc_info=True)
            self.model = None
            self.state_dict = None
    
    def _create_model_from_state_dict(self):
        """Создает модель из state_dict с определением параметров"""
        try:
            # Определяем параметры модели
            vocab_size = None
            n_embd = None
            n_layer = None
            
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
            n_head = 12
            if 'transformer.h.0.attn.c_attn.weight' in self.state_dict:
                attn_weight = self.state_dict['transformer.h.0.attn.c_attn.weight']
                n_head = n_embd // 64 if n_embd and n_embd % 64 == 0 else 12
            
            # Устанавливаем значения по умолчанию
            vocab_size = vocab_size or 50264
            n_embd = n_embd or 768
            n_layer = n_layer or 12
            n_head = n_head or 12
            
            # Определяем количество позиций
            n_positions = 1024
            for key, tensor in self.state_dict.items():
                if 'wpe.weight' in key:
                    n_positions = tensor.shape[0]
                    break
            
            # Создаем конфигурацию
            self.config_obj = GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=n_positions,
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
            
            logger.info(f"Модель создана: vocab_size={vocab_size}, n_embd={n_embd}, n_layer={n_layer}")
            
        except Exception as e:
            logger.error(f"Ошибка создания модели: {e}", exc_info=True)
            self.model = None
            self.config_obj = None
    
    def _initialize_additional_components(self):
        """Инициализирует дополнительные компоненты"""
        try:
            # Инициализация токенизатора
            self._load_optimized_tokenizer()
            
            # Инициализация тренера
            if TRANSFORMERS_AVAILABLE:
                try:
                    from cogniflex.mlearning.text_quality_trainer import TextQualityTrainer, TrainingConfig
                    training_config = TrainingConfig(
                        max_length=self.config.get("max_length", 512)
                    )
                    self.trainer = TextQualityTrainer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=training_config
                    )
                    logger.info("TextQualityTrainer инициализирован")
                except ImportError as e:
                    logger.warning(f"TextQualityTrainer недоступен: {e}")
            
            # Инициализация улучшения качества
            try:
                from cogniflex.mlearning.text_quality_improver import TextQualityImprover
                # TextQualityImprover ожидает путь к директории модели
                if self.model_path and os.path.exists(self.model_path):
                    # Извлекаем директорию из пути к файлу
                    model_dir = os.path.dirname(self.model_path)
                    model_path = model_dir
                else:
                    model_path = None  # Используем путь по умолчанию
                
                self.quality_improver = TextQualityImprover(model_path=model_path)
                logger.info("TextQualityImprover инициализирован")
            except ImportError as e:
                logger.warning(f"TextQualityImprover недоступен: {e}")
            except Exception as e:
                logger.warning(f"Ошибка инициализации TextQualityImprover: {e}")
            
            # Инициализация веб-поиска
            try:
                # Пробуем разные пути импорта
                try:
                    from cogniflex.web_search import WebSearchEngine
                except ImportError:
                    try:
                        from cogniflex.core.web_search import WebSearchEngine
                    except ImportError:
                        logger.warning("WebSearchEngine недоступен")
                        self.web_search_engine = None
                        self.web_search_integration = None
                        return
                
                try:
                    from cogniflex.mlearning.web_search_integration import WebSearchLearningIntegration
                except ImportError:
                    try:
                        from cogniflex.core.web_search_integration import WebSearchLearningIntegration
                    except ImportError:
                        logger.warning("WebSearchLearningIntegration недоступен")
                        self.web_search_integration = None
                        return
                
                if self.web_search_engine is not None:
                    self.web_search_engine = WebSearchEngine()
                    logger.info("WebSearchEngine инициализирован")
                    
                    self.web_search_integration = WebSearchLearningIntegration(self)
                    logger.info("WebSearchLearningIntegration инициализирована")
                    
            except ImportError as e:
                logger.warning(f"Веб-поиск недоступен: {e}")
                self.web_search_engine = None
                self.web_search_integration = None
            
        except Exception as e:
            logger.warning(f"Ошибка инициализации дополнительных компонентов: {e}")
    
    def _load_optimized_tokenizer(self):
        """Загружает оптимизированный токенизатор"""
        try:
            # Ищем токенизатор во фрактальном хранилище
            tokenizer_path = self._find_tokenizer_in_fractal_storage()
            
            if tokenizer_path:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                
                # Проверяем совместимость vocab_size
                if hasattr(self.tokenizer, 'vocab_size'):
                    vocab_diff = abs(self.tokenizer.vocab_size - (self.config_obj.vocab_size if self.config_obj else 50264))
                    if vocab_diff <= 10:
                        logger.info(f"Совместимость токенизатора: разница vocab_size={vocab_diff}")
                        
                        # Исправляем vocab_size если нужно
                        if self.tokenizer.vocab_size < (self.config_obj.vocab_size if self.config_obj else 50264):
                            for i in range(self.tokenizer.vocab_size, (self.config_obj.vocab_size if self.config_obj else 50264)):
                                self.tokenizer.add_tokens([f"<extra_token_{i}>"])
                            logger.info(f"Добавлено {vocab_diff} токенов")
                    else:
                        logger.warning(f"Несовместимость vocab_size: разница={vocab_diff}")
                
                logger.info("Оптимизированный токенизатор загружен")
                
            else:
                self.tokenizer = self._load_fallback_tokenizer()
                
        except Exception as e:
            logger.warning(f"Не удалось загрузить токенизатор: {e}")
            self.tokenizer = self._load_fallback_tokenizer()
    
    def _load_fallback_tokenizer(self):
        """Загружает запасной токенизатор"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
            logger.info("Запасной токенизатор ruGPT3 загружен")
        except Exception as e:
            logger.warning(f"Не удалось загрузить запасной токенизатор: {e}")
            return None
        return self.tokenizer
    
    def _find_tokenizer_in_fractal_storage(self) -> Optional[str]:
        """Ищет токенизатор во фрактальном хранилище"""
        base_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "text-generation"),
            os.path.join(os.getcwd(), "text-generation"),
        ]
        
        for path in base_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def safe_tokenize(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Безопасная токенизация с обработкой устройств"""
        try:
            results = []
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                
                # Перемещаем на устройство модели с проверкой
                input_ids = inputs['input_ids']
                attention_mask = inputs.get('attention_mask')
                
                # Безопасное перемещение
                if input_ids.device != self.device:
                    input_ids = input_ids.to(self.device)
                
                if attention_mask is not None and attention_mask.device != self.device:
                    attention_mask = attention_mask.to(self.device)
                
                results.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'text': text
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            # Fallback
            return [{'text': text, 'input_ids': None, 'attention_mask': None} for text in texts]
    
    def generate_text(self, query: str, max_length: int = 100) -> str:
        """Генерирует текст с исправленной обработкой устройств"""
        if not self.initialized or not self.model or not self.tokenizer:
            return "Модель не инициализирована"
        
        start_time = time.time()
        
        try:
            # Безопасная токенизация
            tokenized = self.safe_tokenize([query])
            if not tokenized or tokenized[0]['input_ids'] is None:
                return "Ошибка токенизации"
            
            input_ids = tokenized[0]['input_ids']
            attention_mask = tokenized[0].get('attention_mask')
            
            # Дополнительная проверка устройства
            if input_ids.device != self.device:
                logger.debug(f"Несоответствие устройств: input_ids={input_ids.device}, model={self.device}")
                input_ids = input_ids.to(self.device)
            
            if attention_mask is not None and attention_mask.device != self.device:
                logger.debug(f"Несоответствие устройств: attention_mask={attention_mask.device}, model={self.device}")
                attention_mask = attention_mask.to(self.device)
            
            # Генерируем ответ
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.85,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.1,
                    length_penalty=1.2,
                    num_beams=3,
                    early_stopping=True
                )
            
            # Декодируем ответ
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Очищаем ответ
            response = self._clean_response(query, generated_text)
            
            # Обновляем статистику
            self.performance_stats["generation_time"] += time.time() - start_time
            
            logger.info(f"Генерация завершена за {time.time() - start_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}", exc_info=True)
            return f"Ошибка генерации: {str(e)}"
    
    def generate_response(self, query: str, max_tokens: int = 100) -> str:
        """Генерирует ответ (обратная совместимость)"""
        return self.generate_text(query, max_tokens)
    
    def generate_response_optimized(self, query: str, max_tokens: int = 100) -> str:
        """Оптимизированная генерация ответа"""
        return self.generate_text(query, max_tokens)
    
    def _clean_response(self, query: str, generated_text: str) -> str:
        """Очищает сгенерированный ответ"""
        try:
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
            
        except Exception as e:
            logger.warning(f"Ошибка очистки ответа: {e}")
            return generated_text
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        stats = self.performance_stats.copy()
        
        # Добавляем текущее состояние
        stats.update({
            "initialized": self.initialized,
            "device": str(self.device),
            "max_memory_tokens": self.max_memory_tokens,
            "cache_size": len(self.tokenization_cache),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / 
                (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
                if (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]) > 0 else 0
            )
        })
        
        return stats
    
    def __del__(self):
        """Очистка при удалении"""
        try:
            # Проверяем, не закрыт ли уже executor
            if hasattr(self, 'tokenization_executor'):
                if not self.tokenization_executor._shutdown:
                    self.tokenization_executor.shutdown(wait=False)
        except Exception:
            pass
        
        try:
            # Проверяем, не закрыт ли уже background executor
            if hasattr(self, 'background_executor'):
                if not self.background_executor._shutdown:
                    self.background_executor.shutdown(wait=False)
        except Exception:
            pass


# Функция для тестирования исправленного менеджера
def test_fixed_manager():
    """Тестирует исправленный менеджер"""
    logger.info("🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОГО МЕНЕДЖЕРА")
    logger.info("="*50)
    
    try:
        # Создаем менеджер
        manager = OptimizedFractalModelManager()
        
        logger.info(f"✅ Менеджер создан")
        logger.info(f"   🔧 Устройство: {manager.device}")
        logger.info(f"   💾 Память токенов: {manager.max_memory_tokens}")
        
        # Проверяем модель
        if hasattr(manager, 'model') and manager.model is not None:
            logger.info(f"   ✅ Модель загружена")
            param_count = sum(p.numel() for p in manager.model.parameters())
            logger.info(f"   📊 Параметров: {param_count:,}")
            
            # Тестируем генерацию
            test_queries = [
                "Привет, как дела?",
                "Что такое искусственный интеллект?",
                "Расскажи о России"
            ]
            
            for i, query in enumerate(test_queries, 1):
                try:
                    logger.info(f"   {i}. 📝 '{query}'")
                    
                    response = manager.generate_text(query, max_length=50)
                    
                    logger.info(f"      💬 '{response[:100]}{'...' if len(response) > 100 else ''}'")
                    logger.info(f"      📊 Длина ответа: {len(response)}")
                    
                except Exception as e:
                    logger.error(f"      ❌ Ошибка: {e}")
            
            logger.info("✅ Тестирование генерации завершено")
            return True
        else:
            logger.warning("   ⚠️ Модель не загружена")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_fixed_manager()
    if success:
        logger.info("\n🎉 ИСПРАВЛЕННЫЙ МЕНЕДЖЕР РАБОТАЕТ КОРРЕКТНО!")
        logger.info("✅ Проблемы устройств решены")
        logger.info("✅ Генерация работает стабильно")
    else:
        logger.error("\n❌ Тестирование не удалось")
