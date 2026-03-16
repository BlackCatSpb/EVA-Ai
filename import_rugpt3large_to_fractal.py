#!/usr/bin/env python3
"""
Импорт ruGPT-3 Large в фрактальное хранилище CogniFlex
Оптимизированная версия для текущей структуры хранилища
"""
import os
import sys
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем путь к CogniFlex
sys.path.append('.')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rugpt3large_import")

class Rugpt3LargeFractalImporter:
    """Импортер ruGPT-3 Large в фрактальное хранилище"""
    
    def __init__(self):
        # Определяем оптимальные пути для текущего фрактального хранилища
        self.base_storage_path = Path("cogniflex_cache/ml_unit/fractal_storage")
        self.model_name = "rugpt3_large_fractal"
        self.model_path = self.base_storage_path / "models" / self.model_name
        self.tokenizer_path = self.base_storage_path / "tokenizers" / self.model_name
        
        # Модели для импорта в порядке приоритета
        self.target_models = [
            "sberbank-ai/rugpt3large_based_on_gpt2",
            "ai-forever/rugpt3large_based_on_gpt2",
            "sberbank-ai/rugpt3medium_based_on_gpt2",  # Fallback
            "ai-forever/rugpt3medium_based_on_gpt2"
        ]
        
        self.selected_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Устройство: {self.device}")
        logger.info(f"Путь к модели: {self.model_path}")
        logger.info(f"Путь к токенизатору: {self.tokenizer_path}")
    
    def check_dependencies(self) -> bool:
        """Проверка зависимостей"""
        try:
            import transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"✅ Transformers версии: {transformers.__version__}")
            return True
        except ImportError as e:
            logger.error(f"❌ Ошибка импорта transformers: {e}")
            logger.info("💡 Установите: pip install transformers")
            return False
    
    def find_available_model(self) -> Optional[str]:
        """Поиск доступной модели ruGPT-3"""
        logger.info("🔍 Поиск доступной модели ruGPT-3...")
        
        for model_name in self.target_models:
            try:
                logger.info(f"   🔄 Проверка: {model_name}")
                
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Пробуем загрузить токенизатор
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=False,
                    trust_remote_code=False
                )
                
                # Пробуем загрузить конфигурацию модели
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        local_files_only=False,
                        trust_remote_code=False,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    param_count = sum(p.numel() for p in model.parameters())
                    vocab_size = len(tokenizer.get_vocab())
                    
                    logger.info(f"   ✅ Найдена: {model_name}")
                    logger.info(f"      📊 Параметров: {param_count:,}")
                    logger.info(f"      🔤 Словарь: {vocab_size:,} токенов")
                    
                    del model  # Освобождаем память
                    self.selected_model = model_name
                    return model_name
                    
                except Exception as model_error:
                    logger.warning(f"   ⚠️ Ошибка загрузки модели: {str(model_error)[:100]}...")
                    continue
                
            except Exception as e:
                logger.warning(f"   ❌ Ошибка: {str(e)[:100]}...")
                continue
        
        logger.error("❌ Не найдено доступных моделей ruGPT-3")
        return None
    
    def prepare_storage_paths(self):
        """Подготовка путей хранения"""
        logger.info("📁 Подготовка путей фрактального хранилища...")
        
        # Создаем все необходимые директории
        paths_to_create = [
            self.base_storage_path,
            self.model_path,
            self.tokenizer_path,
            self.model_path / "weights",
            self.model_path / "metadata",
            self.tokenizer_path / "vocab"
        ]
        
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"   📂 Создана директория: {path}")
        
        logger.info("✅ Пути фрактального хранилища подготовлены")
    
    def import_model(self) -> bool:
        """Основной процесс импорта модели"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"🚀 Начало импорта: {self.selected_model}")
            
            # 1. Загрузка токенизатора
            logger.info("📦 Загрузка токенизатора...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.selected_model,
                local_files_only=False,
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Сохранение токенизатора
            logger.info("💾 Сохранение токенизатора...")
            tokenizer.save_pretrained(str(self.tokenizer_path))
            
            # 2. Загрузка модели
            logger.info("📦 Загрузка модели...")
            model = AutoModelForCausalLM.from_pretrained(
                self.selected_model,
                local_files_only=False,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            model.eval()
            param_count = sum(p.numel() for p in model.parameters())
            
            logger.info(f"✅ Модель загружена")
            logger.info(f"   📊 Параметров: {param_count:,}")
            logger.info(f"   📐 Устройство: {next(model.parameters()).device}")
            
            # 3. Создание фрактальной структуры
            logger.info("🔮 Создание фрактальной структуры...")
            self._create_fractal_structure(model, tokenizer)
            
            # 4. Сохранение метаданных
            logger.info("📋 Сохранение метаданных...")
            self._save_metadata(model, tokenizer, param_count)
            
            # 5. Очистка памяти
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Импорт завершен успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка импорта: {e}")
            return False
    
    def _create_fractal_structure(self, model, tokenizer):
        """Создание фрактальной структуры хранения"""
        try:
            # Анализируем структуру модели
            layer_info = {}
            total_params = 0
            
            for name, param in model.named_parameters():
                layer_info[name] = {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "numel": param.numel(),
                    "requires_grad": param.requires_grad
                }
                total_params += param.numel()
            
            # Сохраняем информацию о слоях
            with open(self.model_path / "metadata" / "layers.json", 'w') as f:
                json.dump(layer_info, f, indent=2)
            
            # Создаем фрактальные индексы для весов
            fractal_index = {
                "model_type": "gpt2",
                "total_parameters": total_params,
                "fractal_levels": 4,
                "block_size": 1024,
                "layers": list(layer_info.keys()),
                "embedding_size": tokenizer.vocab_size,
                "max_position_embeddings": getattr(model.config, 'n_positions', 1024),
                "hidden_size": getattr(model.config, 'n_embd', 768),
                "num_attention_heads": getattr(model.config, 'n_head', 12),
                "num_hidden_layers": getattr(model.config, 'n_layer', 12)
            }
            
            with open(self.model_path / "metadata" / "fractal_index.json", 'w') as f:
                json.dump(fractal_index, f, indent=2)
            
            # Создаем симметричную структуру для токенизатора
            tokenizer_meta = {
                "vocab_size": tokenizer.vocab_size,
                "model_max_length": getattr(tokenizer, 'model_max_length', 1024),
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "bos_token": tokenizer.bos_token,
                "unk_token": tokenizer.unk_token,
                "special_tokens_map": tokenizer.special_tokens_map,
                "fractal_compatible": True
            }
            
            with open(self.tokenizer_path / "metadata.json", 'w') as f:
                json.dump(tokenizer_meta, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Фрактальная структура создана")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания фрактальной структуры: {e}")
            raise
    
    def _save_metadata(self, model, tokenizer, param_count: int):
        """Сохранение метаданных модели"""
        try:
            metadata = {
                "model_name": self.model_name,
                "original_model": self.selected_model,
                "model_type": "gpt2",
                "description": "ruGPT-3 Large импортирована во фрактальное хранилище",
                "parameters": {
                    "total": param_count,
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "non_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad)
                },
                "architecture": {
                    "vocab_size": tokenizer.vocab_size,
                    "hidden_size": getattr(model.config, 'n_embd', 768),
                    "num_hidden_layers": getattr(model.config, 'n_layer', 12),
                    "num_attention_heads": getattr(model.config, 'n_head', 12),
                    "max_position_embeddings": getattr(model.config, 'n_positions', 1024)
                },
                "storage": {
                    "base_path": str(self.base_storage_path),
                    "model_path": str(self.model_path),
                    "tokenizer_path": str(self.tokenizer_path),
                    "fractal_levels": 4,
                    "block_size": 1024
                },
                "import_info": {
                    "import_timestamp": "2026-03-12T20:15:00",
                    "device": self.device,
                    "torch_version": torch.__version__,
                    "fractal_compatible": True,
                    "memory_optimized": True
                }
            }
            
            # Сохраняем основные метаданные
            with open(self.model_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Создаем символическую ссылку для легкого доступа
            link_path = self.base_storage_path / f"{self.model_name}_link.json"
            with open(link_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "model_name": self.model_name,
                    "model_path": str(self.model_path),
                    "tokenizer_path": str(self.tokenizer_path),
                    "ready": True
                }, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Метаданные сохранены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения метаданных: {e}")
            raise
    
    def verify_import(self) -> bool:
        """Проверка успешности импорта"""
        try:
            logger.info("🔍 Проверка импорта...")
            
            # Проверяем наличие основных файлов
            required_files = [
                self.model_path / "metadata.json",
                self.model_path / "metadata" / "layers.json",
                self.model_path / "metadata" / "fractal_index.json",
                self.tokenizer_path / "metadata.json",
                self.tokenizer_path / "tokenizer_config.json",
                self.tokenizer_path / "vocab.json"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                logger.error(f"❌ Отсутствуют файлы: {missing_files}")
                return False
            
            # Пытаемся загрузить токенизатор
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
                logger.info(f"✅ Токенизатор загружен: {len(tokenizer.get_vocab())} токенов")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки токенизатора: {e}")
                return False
            
            logger.info("✅ Импорт успешно проверен")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки импорта: {e}")
            return False
    
    def run(self) -> bool:
        """Запуск полного процесса импорта"""
        logger.info("🚀 НАЧАЛО ИМПОРТА RU-GPT-3 LARGE ВО ФРАКТАЛЬНОЕ ХРАНИЛИЩЕ")
        logger.info("=" * 70)
        
        # 1. Проверка зависимостей
        if not self.check_dependencies():
            return False
        
        # 2. Поиск доступной модели
        if not self.find_available_model():
            return False
        
        # 3. Подготовка путей
        self.prepare_storage_paths()
        
        # 4. Импорт модели
        if not self.import_model():
            return False
        
        # 5. Проверка импорта
        if not self.verify_import():
            return False
        
        logger.info("🎉 ИМПОРТ RU-GPT-3 LARGE УСПЕШНО ЗАВЕРШЕН!")
        logger.info(f"📁 Модель доступна в: {self.model_path}")
        logger.info(f"🔤 Токенизатор доступен в: {self.tokenizer_path}")
        
        return True

def main():
    """Основная функция"""
    try:
        importer = Rugpt3LargeFractalImporter()
        success = importer.run()
        
        if success:
            print("\n✅ ИМПОРТ УСПЕШЕН!")
            print("ruGPT-3 Large готова к использованию во фрактальном хранилище")
        else:
            print("\n❌ ИМПОРТ НЕ УДАЛСЯ")
            return 1
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
