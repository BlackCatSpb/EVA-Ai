"""
Skript konvertatsii GGUF modeli v OpenVINO format.
Ispolzuet llama.cpp dlya konvertatsii GGUF → FP16, zatem optimum-cli dlya OpenVINO.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Proverka kodirovki dlya Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Puti
GGUF_DIR = Path(r"C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_pie_architecture\models\gguf_models")
OPENVINO_DIR = GGUF_DIR / "ruadapt_qwen3_4b_extended"

# Model dlya konvertatsii (berem osnovnuyu)
GGUF_FILE = GGUF_DIR / "Q4_K_M.gguf"

def check_requirements():
    """Proverka neobkhodimykh komponentov."""
    print("=" * 60)
    print("PROVERKA TREBOVANII")
    print("=" * 60)
    
    # Python пакеты
    packages = {
        "optimum": "optimum[openvino]",
        "transformers": "transformers",
        "openvino": "openvino",
    }
    
    missing = []
    for pkg, install_cmd in packages.items():
        try:
            __import__(pkg.replace("-", "_").replace("[", "_").split("_")[0])
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - НЕ УСТАНОВЛЕН")
            missing.append(install_cmd)
    
    if missing:
        print("\n⚠️  Установите недостающие пакеты:")
        for cmd in missing:
            print(f"   pip install {cmd}")
        return False
    
    # llama.cpp для конвертации
    try:
        result = subprocess.run(["llama-cl.exe", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ llama.cpp: {result.stdout.strip()}")
        else:
            print("⚠️  llama.cpp не найден (нужен для конвертации)")
    except FileNotFoundError:
        print("⚠️  llama.cpp не найден (нужен для конвертации)")
        print("   Установите: https://github.com/ggerganov/llama.cpp/releases")
        return False
    
    return True

def convert_gguf_to_fp16():
    """Конвертация GGUF → FP16 через llama.cpp."""
    if not GGUF_FILE.exists():
        print(f"❌ GGUF файл не найден: {GGUF_FILE}")
        return False
    
    fp16_path = GGUF_DIR / "Q4_K_M-FP16.gguf"
    if fp16_path.exists():
        print(f"✅ FP16 модель уже существует: {fp16_path}")
        return fp16_path
    
    print("\n" + "=" * 60)
    print("КОНВЕРТАЦИЯ GGUF → FP16")
    print("=" * 60)
    print(f"Источник: {GGUF_FILE}")
    print(f"Цель: {fp16_path}")
    
    # Команда llama.cpp для конвертации
    cmd = [
        "llama-cl.exe",  # или llama-c-server.exe
        "--convert",
        str(GGUF_FILE),
        "-o", str(fp16_path),
        "--outtype", "f16"
    ]
    
    try:
        print(f"\nВыполняю: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ Конвертация завершена: {fp16_path}")
            return fp16_path
        else:
            print(f"❌ Ошибка конвертации: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Таймаут конвертации (>10 минут)")
        return False
    except FileNotFoundError as e:
        print(f"❌ llama.cpp не найден: {e}")
        return False

def convert_fp16_to_openvino(fp16_path):
    """Конвертация FP16 → OpenVINO через optimum-cli."""
    if OPENVINO_DIR.exists() and any(OPENVINO_DIR.iterdir()):
        print(f"\n✅ OpenVINO модель уже существует: {OPENVINO_DIR}")
        return OPENVINO_DIR
    
    print("\n" + "=" * 60)
    print("КОНВЕРТАЦИЯ FP16 → OPENVINO")
    print("=" * 60)
    
    # Проверяем, откуда модель (GGUF конвертированная или HuggingFace)
    # optimum-cli работает с HuggingFace моделями, но может и с директориями
    cmd = [
        sys.executable, "-m", "optimum-cli",
        "export", "openvino",
        "--model", str(fp16_path.parent),  # Директория с моделью
        "--task", "text-generation-with-past",
        "--trust-remote-code",
        "--weight-format", "fp16",
        "--output", str(OPENVINO_DIR)
    ]
    
    try:
        print(f"\nВыполняю: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print(f"✅ Конвертация завершена: {OPENVINO_DIR}")
            return OPENVINO_DIR
        else:
            print(f"❌ Ошибка: {result.stderr}")
            # Пробуем альтернативный путь - копируем FP16 как есть
            print("\n⚠️  Пробую альтернативный путь...")
            return convert_alternative(fp16_path)
    except subprocess.TimeoutExpired:
        print("❌ Таймаут конвертации (>30 минут)")
        return False

def convert_alternative(fp16_path):
    """Альтернативный путь - используем openvino_genai напрямую."""
    print("\n" + "=" * 60)
    print("АЛЬТЕРНАТИВНЫЙ ПУТЬ")
    print("=" * 60)
    print("Используем llama.cpp для конвертации напрямую в safetensors,")
    print("затем экспортируем в OpenVINO.")
    
    # llama.cpp может экспортировать в format= safetensors
    safetensors_path = GGUF_DIR / "model.safetensors"
    
    if safetensors_path.exists():
        print(f"✅ safetensors уже есть: {safetensors_path}")
    else:
        # Пробуем конвертировать через llama.cpp-cli
        cmd = [
            "python", "-m", "llama_cpp.tools.convert",
            "--model", str(GGUF_FILE),
            "--output", str(safetensors_path)
        ]
        print(f"\nКонвертация в safetensors...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"⚠️  {result.stderr}")
        except:
            pass
    
    # Проверяем можно ли использовать GGUF напрямую
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ПРЯМОГО ИСПОЛЬЗОВАНИЯ GGUF")
    print("=" * 60)
    
    try:
        import openvino_genai as ov_genai
        print("✅ openvino_genai доступен")
        
        # Пробуем загрузить GGUF напрямую
        print(f"Пробую загрузить: {GGUF_FILE}")
        pipeline = ov_genai.LLMPipeline(str(GGUF_FILE), "CPU")
        print("✅ GGUF загружен напрямую!")
        print(f"   Тип модели: {type(pipeline)}")
        
        # Если работает - создаем символическую ссылку или копируем
        if not OPENVINO_DIR.exists():
            OPENVINO_DIR.mkdir(parents=True)
        
        # Копируем конфиг
        config_example = OPENVINO_DIR / "README.txt"
        with open(config_example, "w") as f:
            f.write(f"""OpenVINO GenAI - Директория для модели
=====================================

Модель: {GGUF_FILE.name}
Формат: GGUF (загружается напрямую через openvino_genai)
Дата: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}

Использование:
```python
from openvino_genai import LLMPipeline
pipeline = LLMPipeline("{GGUF_FILE}", "CPU")
result = pipeline.generate("Привет")
```
""")
        print(f"✅ Создан конфиг: {config_example}")
        return OPENVINO_DIR
        
    except Exception as e:
        print(f"❌ Не удалось: {e}")
        return None

def update_brain_config(openvino_path):
    """Обновление brain_config.json для использования OpenVINO модели."""
    import json
    
    config_path = Path(r"C:\Users\black\OneDrive\Desktop\EVA-Ai\brain_config.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Добавляем путь к OpenVINO модели
    config["model"]["openvino_model_path"] = str(openvino_path)
    config["model"]["use_openvino_direct"] = True
    
    # Создаем резервную копию
    backup_path = config_path.with_suffix(".json.backup")
    shutil.copy(config_path, backup_path)
    print(f"✅ Резервная копия: {backup_path}")
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Конфиг обновлён: {config_path}")

def main():
    print("\n" + "=" * 60)
    print("КОНВЕРТАЦИЯ GGUF → OPENVINO")
    print("=" * 60)
    print(f"GGUF файл: {GGUF_FILE}")
    print(f"Целевая директория: {OPENVINO_DIR}")
    print()
    
    # Шаг 1: Проверка требований
    if not check_requirements():
        print("\n❌ Проверка требований не пройдена")
        return 1
    
    # Шаг 2: Проверяем есть ли уже готовая модель
    if OPENVINO_DIR.exists() and any(OPENVINO_DIR.iterdir()):
        print(f"\n✅ OpenVINO модель уже существует!")
        result = OPENVINO_DIR
    else:
        # Шаг 3: Конвертация
        # Пробуем прямой путь с llama.cpp
        result = convert_alternative(GGUF_FILE)
        
        if not result:
            # Шаг 3a: Конвертация через FP16
            fp16_path = convert_gguf_to_fp16()
            if fp16_path:
                result = convert_fp16_to_openvino(fp16_path)
    
    # Шаг 4: Обновление конфига
    if result and result.exists():
        update_brain_config(result)
        print("\n" + "=" * 60)
        print("✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
        print("=" * 60)
        print(f"Модель доступна в: {result}")
        return 0
    else:
        print("\n❌ Конвертация не удалась")
        return 1

if __name__ == "__main__":
    sys.exit(main())
