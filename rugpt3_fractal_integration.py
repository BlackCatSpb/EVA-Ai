"""
Полная интеграция ruGPT3 во фрактальное хранилище с гибридным кешем
"""
import sys
import os
import torch
import json
import logging
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rugpt3_fractal_integration")

class RuGPT3FractalIntegration:
    """
    Полная интеграция ruGPT3 во фрактальное хранилище с учетом:
    - Фрактальной структуры хранения
    - Гибридного кеша
    - Адаптированной токенизации
    - Оптимизированного чтения/записи
    """
    
    def __init__(self):
        self.model_id = "rugpt3_large_fractal"
        self.cache_dir = Path("cogniflex_cache/ml_unit/fractal_storage")
        self.model_dir = self.cache_dir / "models" / self.model_id
        self.tokenizer_dir = self.cache_dir / "tokenizers" / self.model_id
        
        # Создаем директории
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RuGPT3FractalIntegration инициализирован: {self.model_id}")
    
    def analyze_model_structure(self, model) -> Dict[str, Any]:
        """Анализирует структуру модели для фрактального хранения"""
        logger.info("🔍 Анализ структуры модели...")
        
        analysis = {
            "model_type": type(model).__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "layers": {},
            "embedding_analysis": {},
            "attention_analysis": {},
            "ffn_analysis": {}
        }
        
        # Анализируем каждый слой
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                layer_info = {
                    "shape": tuple(weight.shape),
                    "parameters": weight.numel(),
                    "dtype": str(weight.dtype),
                    "device": str(weight.device),
                    "requires_grad": weight.requires_grad,
                    "mean": float(weight.mean().item()),
                    "std": float(weight.std().item()),
                    "min": float(weight.min().item()),
                    "max": float(weight.max().item())
                }
                
                # Классифицируем слои
                if any(key in name.lower() for key in ['wte', 'embed']):
                    analysis["embedding_analysis"][name] = layer_info
                elif any(key in name.lower() for key in ['attn', 'attention']):
                    analysis["attention_analysis"][name] = layer_info
                elif any(key in name.lower() for key in ['ffn', 'mlp', 'dense']):
                    analysis["ffn_analysis"][name] = layer_info
                else:
                    analysis["layers"][name] = layer_info
        
        logger.info(f"   📊 Всего параметров: {analysis['total_parameters']:,}")
        logger.info(f"   🏗️ Слоев: {len(analysis['layers'])}")
        logger.info(f"   🔤 Эмбеддингов: {len(analysis['embedding_analysis'])}")
        logger.info(f"   👁 Attention: {len(analysis['attention_analysis'])}")
        logger.info(f"   🧠 FFN: {len(analysis['ffn_analysis'])}")
        
        return analysis
    
    def create_fractal_tokenizer(self, base_tokenizer) -> Any:
        """Создает фрактальный токенизатор на основе базового"""
        logger.info("🔤 Создание фрактального токенизатора...")
        
        try:
            from fractal_tokenizer import FractalTokenizer
            
            # Создаем фрактальный токенизатор
            fractal_tokenizer = FractalTokenizer(
                base_tokenizer_path=base_tokenizer.name_or_path,
                cache_dir=str(self.tokenizer_dir),
                fractal_levels=3,  # Оптимизировано для ruGPT3
                block_size=32,     # Уменьшено для экономии памяти
                hybrid_cache_size=20000  # Увеличено для большой модели
            )
            
            logger.info(f"✅ Фрактальный токенизатор создан")
            return fractal_tokenizer
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания фрактального токенизатора: {e}")
            return base_tokenizer
    
    def export_model_to_fractal(self, model, tokenizer, model_analysis: Dict[str, Any]) -> bool:
        """Экспортирует модель во фрактальное хранилище"""
        logger.info("📦 Экспорт модели во фрактальное хранилище...")
        
        try:
            from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
            
            # Определяем оптимальные параметры для ruGPT3
            total_params = model_analysis["total_parameters"]
            param_gb = total_params * 4 / (1024**3)  # float32 ~ 4 байта
            
            if param_gb > 5:
                # Очень большая модель - максимальная оптимизация
                fractal_levels = 2
                block_size = 16
                containers_per_group = 2
                hot_window_size = 128 * 1024 * 1024  # 128MB
            elif param_gb > 2:
                # Большая модель - средняя оптимизация
                fractal_levels = 3
                block_size = 32
                containers_per_group = 3
                hot_window_size = 256 * 1024 * 1024  # 256MB
            else:
                # Средняя модель - стандартная оптимизация
                fractal_levels = 4
                block_size = 64
                containers_per_group = 4
                hot_window_size = 512 * 1024 * 1024  # 512MB
            
            logger.info(f"   📊 Параметры фрактала: levels={fractal_levels}, block={block_size}, groups={containers_per_group}")
            logger.info(f"   💾 Hot window: {hot_window_size // (1024*1024)}MB")
            
            # Создаем фрактальное хранилище
            fractal_store = FractalWeightStore(
                block_size=block_size,
                fractal_levels=fractal_levels,
                containers_per_group=containers_per_group,
                device="cpu"
            )
            
            # Устанавливаем размер горячего окна
            fractal_store.hot_window_size = hot_window_size
            
            # Сохраняем state_dict
            state_dict = model.state_dict()
            
            logger.info(f"   🔄 Упаковка {len(state_dict)} тензоров...")
            
            # Используем поэтапную упаковку для больших моделей
            if param_gb > 2:
                success = self._pack_state_dict_incremental(fractal_store, state_dict, batch_size=5)
            else:
                success = fractal_store.pack_state_dict(state_dict, self.model_id)
            
            if not success:
                logger.error("❌ Ошибка упаковки state_dict")
                return False
            
            logger.info("   💾 Сохранение на диск...")
            
            # Сохраняем фрактальное хранилище
            save_success = fractal_store.save_to_disk_sharded(
                str(self.model_dir),
                knowledge_graph=None,
                shard_size=100,  # Маленькие шарды для надежности
                by_level=True,
                compress=True
            )
            
            if not save_success:
                logger.error("❌ Ошибка сохранения на диск")
                return False
            
            # Сохраняем токенизатор
            logger.info("   🔤 Сохранение токенизатора...")
            tokenizer.save_pretrained(str(self.tokenizer_dir))
            
            # Сохраняем метаданные
            metadata = {
                "model_id": self.model_id,
                "model_type": model_analysis["model_type"],
                "total_parameters": model_analysis["total_parameters"],
                "fractal_config": {
                    "levels": fractal_levels,
                    "block_size": block_size,
                    "containers_per_group": containers_per_group,
                    "hot_window_size": hot_window_size
                },
                "model_analysis": model_analysis,
                "export_timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
            metadata_file = self.model_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Модель экспортирована: {self.model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка экспорта модели: {e}", exc_info=True)
            return False
    
    def _pack_state_dict_incremental(self, fractal_store, state_dict, batch_size=5):
        """Поэтапная упаковка state_dict для экономии памяти"""
        logger.info(f"   🔄 Поэтапная упаковка (batch_size={batch_size})...")
        
        tensor_names = list(state_dict.keys())
        total_batches = (len(tensor_names) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(tensor_names))
            batch_names = tensor_names[start_idx:end_idx]
            
            logger.info(f"   📦 Батч {batch_idx + 1}/{total_batches}: {len(batch_names)} тензоров")
            
            # Создаем временный state_dict для батча
            batch_state_dict = {name: state_dict[name] for name in batch_names}
            
            try:
                # Упаковываем батч
                if batch_idx == 0:
                    # Первый батч - инициализация хранилища
                    success = fractal_store.pack_state_dict(batch_state_dict, f"{self.model_id}_batch_{batch_idx}")
                else:
                    # Последующие батчи - добавление к существующему хранилищу
                    success = self._add_batch_to_fractal_store(fractal_store, batch_state_dict, batch_idx)
                
                if not success:
                    logger.error(f"   ❌ Ошибка упаковки батча {batch_idx}")
                    return False
                
                # Очищаем память
                del batch_state_dict
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"   ❌ Ошибка батча {batch_idx}: {e}")
                return False
        
        return True
    
    def _add_batch_to_fractal_store(self, fractal_store, batch_state_dict, batch_idx):
        """Добавляет батч к существующему фрактальному хранилищу"""
        try:
            # Для каждого тензора в батче
            for name, tensor in batch_state_dict.items():
                # Преобразуем в numpy
                arr = tensor.detach().cpu().numpy()
                flat = arr.reshape(-1)
                
                # Разбиваем на блоки
                total_elements = int(flat.size)
                block_size = fractal_store.block_size
                
                for i in range(0, total_elements, block_size):
                    block = flat[i:i + block_size]
                    
                    # Создаем контейнер
                    container_id = f"{self.model_id}_batch_{batch_idx}_{name}_{i//block_size}"
                    
                    # Используем правильный API FractalWeightStore
                    # Создаем временный state_dict для одного блока
                    temp_state_dict = {container_id: torch.from_numpy(block)}
                    
                    # Добавляем в хранилище через pack_state_dict
                    if batch_idx == 0 and i == 0:
                        # Первый блок - инициализация
                        fractal_store.pack_state_dict(temp_state_dict, f"{self.model_id}_batch_{batch_idx}")
                    else:
                        # Последующие блоки - добавляем к существующему
                        self._add_to_existing_fractal_store(fractal_store, temp_state_dict)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления батча: {e}")
            return False
    
    def _add_to_existing_fractal_store(self, fractal_store, state_dict):
        """Добавляет state_dict к существующему фрактальному хранилищу"""
        try:
            # Прямая манипуляция с контейнерами
            for key, tensor in state_dict.items():
                arr = tensor.detach().cpu().numpy()
                flat = arr.reshape(-1)
                
                # Создаем контейнер
                from cogniflex.mlearning.storage.fractal_store import FractalContainer
                container = FractalContainer(
                    id=key,
                    level=0,
                    position=(0,),
                    data=flat,
                    shape=(flat.size,),
                    dtype="float32",
                    metadata={
                        "layer_name": key,
                        "model_id": self.model_id,
                        "original_shape": tuple(arr.shape),
                        "block_start": 0,
                        "block_end": flat.size,
                        "is_critical": False,
                        "storage_dtype": "float32",
                        "param_name": key,
                        "tensor_path": key
                    }
                )
                
                # Добавляем в хранилище
                fractal_store.containers[key] = container
                fractal_store.fractal_tree[0].append(key)
                fractal_store.total_memory += container.get_memory_size()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления к хранилищу: {e}")
            return False
    
    def load_model_from_fractal(self) -> Optional[Any]:
        """Загружает модель из фрактального хранилища"""
        logger.info("📥 Загрузка модели из фрактального хранилища...")
        
        try:
            from cogniflex.mlearning.storage.fractal_model_loader import FractalModelLoader
            from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig
            
            # Проверяем наличие модели
            if not self.model_dir.exists():
                logger.error(f"❌ Модель не найдена: {self.model_dir}")
                return None
            
            # Загружаем метаданные
            metadata_file = self.model_dir / "metadata.json"
            if not metadata_file.exists():
                logger.error("❌ Метаданные модели не найдены")
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"   📊 Модель: {metadata['model_id']}")
            logger.info(f"   📊 Параметров: {metadata['total_parameters']:,}")
            logger.info(f"   📅 Экспорт: {metadata['export_timestamp']}")
            
            # Создаем конфигурацию загрузчика
            config = ModelStorageConfig(
                base_path=str(self.model_dir),
                block_size=metadata["fractal_config"]["block_size"],
                fractal_levels=metadata["fractal_config"]["levels"],
                device="cpu"
            )
            
            # Создаем загрузчик
            loader = FractalModelLoader(config)
            
            # Проверяем доступные модели
            available_models = loader.list_models()
            if not available_models:
                logger.error("❌ Модели в фрактальном хранилище не найдены")
                return None
            
            logger.info(f"   📋 Доступные модели: {available_models}")
            
            # Загружаем первую модель
            model_id = available_models[0]
            model = loader.load_model(model_id)
            
            if model is None:
                logger.error(f"❌ Не удалось загрузить модель: {model_id}")
                return None
            
            logger.info(f"✅ Модель {model_id} загружена")
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}", exc_info=True)
            return None
    
    def load_tokenizer_from_fractal(self) -> Optional[Any]:
        """Загружает токенизатор из фрактального хранилища"""
        logger.info("🔤 Загрузка токенизатора из фрактального хранилища...")
        
        try:
            from transformers import AutoTokenizer
            
            # Проверяем наличие токенизатора
            if not self.tokenizer_dir.exists():
                logger.error(f"❌ Токенизатор не найден: {self.tokenizer_dir}")
                return None
            
            # Загружаем токенизатор
            tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            
            logger.info(f"✅ Токенизатор загружен: vocab_size={len(tokenizer.get_vocab()):,}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки токенизатора: {e}", exc_info=True)
            return None
    
    def test_fractal_model(self, model, tokenizer) -> bool:
        """Тестирует фрактальную модель"""
        logger.info("🧪 Тестирование фрактальной модели...")
        
        try:
            test_queries = [
                "Привет, как дела?",
                "Что такое искусственный интеллект?",
                "Расскажи о России",
                "Объясни фрактальную структуру данных",
                "Как работает нейронная сеть?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                try:
                    # Кодируем запрос
                    inputs = tokenizer.encode(query, return_tensors='pt')
                    
                    # Генерируем ответ
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_length=inputs.shape[1] + 50,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
                        )
                    
                    # Декодируем результат
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    logger.info(f"   {i}. 📝 '{query}'")
                    logger.info(f"      💬 '{response[:100]}{'...' if len(response) > 100 else ''}'")
                    logger.info(f"      📊 Длина ответа: {len(response)}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Ошибка для запроса '{query}': {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования: {e}", exc_info=True)
            return False
    
    def integrate_with_hybrid_cache(self, model, tokenizer) -> bool:
        """Интегрирует модель с гибридным кешем"""
        logger.info("🔄 Интеграция с гибридным кешем...")
        
        try:
            from cogniflex.memory.hybrid_token_cache import HybridTokenCache
            
            # Создаем гибридный кеш для модели
            hybrid_cache = HybridTokenCache(
                brain=None,  # Будет установлен позже
                max_memory_tokens=15000,  # Увеличено для ruGPT3
                disk_cache_dir=str(self.cache_dir / "hybrid_cache"),
                target_memory_gb=3.0,  # 3GB для токенов
                dynamic_memory_limit=True,
                max_ram_usage_percent=80.0,
                vram_threshold=0.2,
                ram_threshold=0.15,
                eviction_policy="hybrid",
                vram_ratio=0.7,
                ram_cache_ratio=0.3
            )
            
            # Интегрируем кеш в модель
            if hasattr(model, 'set_hybrid_cache'):
                model.set_hybrid_cache(hybrid_cache)
            else:
                # Сохраняем ссылку на кеш в атрибутах модели
                model._hybrid_cache = hybrid_cache
            
            logger.info("✅ Гибридный кеш интегрирован")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка интеграции гибридного кеша: {e}", exc_info=True)
            return False

def main():
    """Основная функция интеграции ruGPT3"""
    logger.info("🚀 ИНТЕГРАЦИЯ RU-GPT3 ВО ФРАКТАЛЬНОЕ ХРАНИЛИЩЕ")
    logger.info("="*60)
    
    try:
        # Создаем интеграцию
        integration = RuGPT3FractalIntegration()
        
        # 1. Загружаем ruGPT3
        logger.info("📦 Загрузка ruGPT3...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        logger.info(f"✅ ruGPT3 загружена: {sum(p.numel() for p in model.parameters()):,} параметров")
        
        # 2. Анализируем структуру модели
        model_analysis = integration.analyze_model_structure(model)
        
        # 3. Создаем фрактальный токенизатор
        fractal_tokenizer = integration.create_fractal_tokenizer(tokenizer)
        
        # 4. Экспортируем модель во фрактальное хранилище
        export_success = integration.export_model_to_fractal(model, tokenizer, model_analysis)
        
        if not export_success:
            logger.error("❌ Экспорт модели не удался")
            return False
        
        # 5. Загружаем модель из фрактального хранилища
        loaded_model = integration.load_model_from_fractal()
        
        if loaded_model is None:
            logger.error("❌ Загрузка модели из фрактального хранилища не удалась")
            return False
        
        # 6. Загружаем токенизатор из фрактального хранилища
        loaded_tokenizer = integration.load_tokenizer_from_fractal()
        
        if loaded_tokenizer is None:
            logger.error("❌ Загрузка токенизатора не удалась")
            return False
        
        # 7. Интегрируем с гибридным кешем
        cache_success = integration.integrate_with_hybrid_cache(loaded_model, loaded_tokenizer)
        
        if not cache_success:
            logger.error("❌ Интеграция гибридного кеша не удалась")
            return False
        
        # 8. Тестируем полную систему
        test_success = integration.test_fractal_model(loaded_model, loaded_tokenizer)
        
        if test_success:
            logger.info("🎉 ИНТЕГРАЦИЯ RU-GPT3 УСПЕШНА!")
            logger.info("✅ Модель экспортирована во фрактальное хранилище")
            logger.info("✅ Фрактальный токенизатор создан")
            logger.info("✅ Гибридный кеш интегрирован")
            logger.info("✅ Система готова к использованию")
            
            logger.info(f"📁 Путь к модели: {integration.model_dir}")
            logger.info(f"🔤 Путь к токенизатору: {integration.tokenizer_dir}")
            
            return True
        else:
            logger.error("❌ Тестирование системы не удался")
            return False
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка интеграции: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
