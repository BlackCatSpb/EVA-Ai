#!/usr/bin/env python3
"""
Создание полной конфигурации ruGPT-3 Medium для фрактального хранилища
"""
import os
import json
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_rugpt3_medium_fractal():
    """Создает полную конфигурацию ruGPT-3 Medium для фрактального хранилища"""
    
    # Путь к фрактальному хранилищу
    fractal_path = "./cogniflex_cache/ml_unit/fractal_storage/rugpt3large"
    
    # Создаем директории
    os.makedirs(fractal_path, exist_ok=True)
    os.makedirs(os.path.join(fractal_path, "model"), exist_ok=True)
    os.makedirs(os.path.join(fractal_path, "tokenizer"), exist_ok=True)
    
    # Конфигурация модели ruGPT-3 Medium
    model_config = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.1,
        "bos_token_id": 1,
        "embd_pdrop": 0.1,
        "eos_token_id": 2,
        "id2label": {"0": "LABEL_0"},
        "initializer_range": 0.02,
        "label2id": {"LABEL_0": 0},
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 2048,
        "n_embd": 1024,
        "n_head": 16,
        "n_inner": None,
        "n_layer": 24,
        "n_positions": 2048,
        "n_special": 0,
        "output_past": True,
        "pad_token_id": 0,
        "predict_special_tokens": True,
        "reorder_and_upcast_attn": False,
        "resid_pdrop": 0.1,
        "scale_attn_by_inverse_layer_idx": False,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "torch_dtype": "float32",
        "transformers_version": "4.55.4",
        "use_cache": True,
        "vocab_size": 50257,
        "_name_or_path": "sberbank-ai/rugpt3large_based_on_gpt2",
        "task_specific_params": {
            "text-generation": {
                "do_sample": True,
                "max_length": 50
            }
        }
    }
    
    # Сохраняем конфигурацию модели
    with open(os.path.join(fractal_path, "model", "config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    
    # Конфигурация генерации
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "do_sample": True,
        "max_length": 2048,
        "pad_token_id": 0,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "num_return_sequences": 1
    }
    
    # Сохраняем конфигурацию генерации
    with open(os.path.join(fractal_path, "model", "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2, ensure_ascii=False)
    
    # Конфигурация токенизатора
    tokenizer_config = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "bos_token": "<s>",
        "cls_token": "<s>",
        "eos_token": "</s>",
        "errors": "replace",
        "mask_token": "<mask>",
        "model_max_length": 1024,
        "name_or_path": "sberbank-ai/rugpt3large_based_on_gpt2",
        "pad_token": "<pad>",
        "sep_token": "</s>",
        "special_tokens_map_file": None,
        "tokenizer_class": "GPT2Tokenizer",
        "trim_offsets": True,
        "unk_token": "<unk>",
        "use_fast": True,
        "vocab_size": 50257,
        "wordpieces_prefix": "##"
    }
    
    # Сохраняем конфигурацию токенизатора
    with open(os.path.join(fractal_path, "tokenizer", "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    # Карта специальных токенов
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "cls_token": "<s>",
        "mask_token": "<mask>"
    }
    
    # Сохраняем карту специальных токенов
    with open(os.path.join(fractal_path, "tokenizer", "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    
    # Метаданные фрактального хранилища
    metadata = {
        "model_name": "rugpt3large",
        "model_type": "gpt2",
        "description": "ruGPT-3 Medium (355M параметров) - фрактальное хранилище",
        "vocab_size": 50257,
        "max_length": 2048,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "intermediate_size": 4096,
        "activation": "gelu_new",
        "dropout": 0.1,
        "version": "1.0",
        "export_date": "2026-03-09",
        "source": "fractal_setup",
        "fractal_compatible": True,
        "memory_optimized": True,
        "local_only": True,
        "no_hf_dependency": True,
        "model_files": {
            "config": "model/config.json",
            "generation_config": "model/generation_config.json",
            "tokenizer_config": "tokenizer/tokenizer_config.json",
            "special_tokens": "tokenizer/special_tokens_map.json"
        }
    }
    
    # Сохраняем метаданные
    with open(os.path.join(fractal_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Обновляем конфигурацию системы
    system_config = {
        "model_path": "./cogniflex_cache/ml_unit/fractal_storage/rugpt3large",
        "config_path": "./cogniflex_cache/ml_unit/fractal_storage/rugpt3large",
        "model_name": "rugpt3large",
        "model_type": "fractal",
        "device": "auto",
        "max_memory_tokens": 25000,
        "target_memory_gb": 1.0,
        "vram_limit_gb": 1.5,
        "ram_limit_gb": 1.0,
        "ssd_limit_gb": 50.0,
        "cache_tokenization": True,
        "parallel_tokenization": True,
        "tokenization_workers": 4,
        "memory_optimization": True,
        "use_uint16": True,
        "tensor_pool_size": 1000,
        "batch_size": 4,
        "max_length": 256,
        "overlap_tokens": 64,
        "auto_improvement": True,
        "quality_threshold": 0.7,
        "check_interval_seconds": 30,
        "local_only": True,
        "no_hf_download": True,
        "fractal_storage": {
            "enabled": True,
            "path": "./cogniflex_cache/ml_unit/fractal_storage/rugpt3large",
            "compression": True,
            "memory_mapping": True
        },
        "hybrid_cache": {
            "target_memory_gb": 1.0,
            "vram_threshold": 0.15,
            "ram_threshold": 0.12,
            "max_ram_usage_percent": 70.0,
            "hot_threshold": 5,
            "eviction_policy": "hybrid",
            "cache_ttl": 86400,
            "min_relevance_score": 0.3,
            "max_context_tokens": 1000,
            "system_free_mem_threshold": 0.1,
            "memory_pressure_interval_s": 2.0,
            "pressure_offload_batch": 32
        },
        "hot_window": {
            "size_gb": 1.5,
            "strategy": "adaptive",
            "threshold": 5
        }
    }
    
    # Обновляем конфигурацию системы
    system_config_path = "./cogniflex/config/fractal_model_config.json"
    with open(system_config_path, "w", encoding="utf-8") as f:
        json.dump(system_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Конфигурация ruGPT-3 Medium создана в {fractal_path}")
    logger.info(f"✅ Системная конфигурация обновлена в {system_config_path}")
    
    return True

if __name__ == "__main__":
    success = setup_rugpt3_medium_fractal()
    if success:
        print("Конфигурация ruGPT-3 Medium для фрактального хранилища создана успешно")
    else:
        print("Ошибка создания конфигурации")
