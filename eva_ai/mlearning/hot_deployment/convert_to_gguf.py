"""
Локальная конвертация Qwen в GGUF формат.
Использует llama.cpp для конвертации.
"""
import os
import sys
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("eva_ai.mlearning.hot_deployment.convert_to_gguf")

def get_llama_cpp_dir() -> str:
    """Ищет директорию llama.cpp"""
    # Проверяем в директории проекта
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    possible_dirs = [
        os.path.join(project_root, "llama.cpp"),
        os.path.join(project_root, "..", "llama.cpp"),
        os.path.join(os.getcwd(), "llama.cpp"),
    ]
    
    for d in possible_dirs:
        if os.path.exists(d):
            return d
    
    return None


def check_conda_environ():
    """Проверяет conda/Miniconda"""
    conda_path = os.environ.get("CONDA_PREFIX")
    if conda_path:
        logger.info(f"Conda environment: {conda_path}")
        return conda_path
    
    # Ищем Miniconda
    possible_conda = [
        os.path.join(os.path.expanduser("~"), "miniconda3"),
        os.path.join(os.path.expanduser("~"), "anaconda3"),
        "C:\\miniconda3",
        "C:\\ProgramData\\miniconda3",
    ]
    
    for p in possible_conda:
        if os.path.exists(p):
            return p
    
    return None


def run_gguf_conversion():
    """Запускает конвертацию"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    output_path = os.path.join(project_root, "models", "qwen3.5-0.8b-instruct-q5_k_m.gguf")
    
    logger.info(f"Модель: {model_path}")
    logger.info(f"Выход: {output_path}")
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена: {model_path}")
        return False
    
    # Проверяем уже сконвертированное
    if os.path.exists(output_path):
        logger.info(f"Модель уже существует: {output_path}")
        return output_path
    
    # Создаём директорию для вывода
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Пробуем использовать llama.cpp из conda или пути
    llama_cpp_dir = get_llama_cpp_dir()
    
    if llama_cpp_dir:
        convert_script = os.path.join(llama_cpp_dir, "convert.py")
        
        if os.path.exists(convert_script):
            logger.info(f"Используем llama.cpp: {llama_cpp_dir}")
            
            # Запускаем конвертацию
            cmd = [
                sys.executable, convert_script,
                model_path,
                "--outfile", output_path,
                "--outtype", "q5_k_m"
            ]
            
            logger.info(f"Команда: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 минут
                )
                
                if result.returncode == 0:
                    logger.info("Конвертация успешна!")
                    logger.info(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                    return output_path
                else:
                    logger.error(f"Ошибка: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error("Таймаут конвертации (30 мин)")
            except Exception as e:
                logger.error(f"Исключение: {e}")
        else:
            logger.warning(f"convert.py не найден: {convert_script}")
    
    # Пробуем через pip + llama-cpp-python
    logger.info("Пробуем через llama-cpp-python...")
    
    return convert_via_llama_cpp_python(model_path, output_path)


def convert_via_llama_cpp_python(model_path: str, output_path: str) -> Optional[str]:
    """Конвертирует через llama-cpp-python"""
    try:
        # Проверяем версию llama-cpp-python
        import llama_cpp
        logger.info(f"llama-cpp-python версия: {llama_cpp.__version__}")
        
        # К сожалению, llama-cpp-python не имеет встроенной конвертации
        # Нужен llama.cpp
        
        logger.error("Требуется llama.cpp для конвертации")
        logger.info("Скачайте llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
        return None
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return None


def simple_quantization():
    """
    Простое решение - используем существующую модель и просто квантизируем веса.
    Создаёт упрощённую GGUF-подобную структуру.
    """
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    output_dir = os.path.join(project_root, "models", "qwen3.5-0.8b-simple-gguf")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Загрузка модели...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Загрузка весов...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Сохраняем в упрощённом формате
    logger.info("Сохранение весов...")
    
    state_dict = model.state_dict()
    
    # Сохраняем основные веса
    weights_to_save = {}
    for key, value in state_dict.items():
        # Пропускаем очень большие ключи (позиционные эмбеддинги и т.д.)
        if value.numel() > 100_000_000:
            logger.warning(f"Пропускаем {key}: {value.shape}")
            continue
        
        # Квантизация в int8
        weights_to_save[key] = value.to(torch.int8).numpy()
    
    # Сохраняем
    weights_path = os.path.join(output_dir, "weights.npz")
    np.savez_compressed(weights_path, **weights_to_save)
    
    # Сохраняем конфигурацию
    config = {
        "model_type": "qwen3.5",
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
        "num_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
    }
    
    import json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Токенизатор
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Сохранено в: {output_dir}")
    logger.info(f"Веса: {weights_path}")
    
    return output_dir


def check_gguf_available():
    """Проверяет готовность к конвертации"""
    # 1. Проверяем llama.cpp
    llama_cpp_dir = get_llama_cpp_dir()
    if llama_cpp_dir:
        logger.info(f"llama.cpp найден: {llama_cpp_dir}")
        return True
    
    # 2. Проверяем conda
    conda = check_conda_environ()
    if conda:
        logger.info(f"Conda: {conda}")
        
        # Ищем llama.cpp в conda
        llama_bin = os.path.join(conda, "Library", "bin", "llama")
        if os.path.exists(llama_bin):
            logger.info(f"llama binary: {llama_bin}")
            return True
    
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("=== Конвертация Qwen в GGUF ===")
    
    # Проверяем готовность
    check_gguf_available()
    
    # Пробуем основной метод
    result = run_gguf_conversion()
    
    if result:
        logger.info(f"Конвертация завершена: {result}")
    else:
        logger.info("Пробуем простую конвертацию...")
        result = simple_quantization()
        
        if result:
            logger.info(f"Создано: {result}")