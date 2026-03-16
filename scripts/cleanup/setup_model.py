"""
Скрипт для загрузки и настройки предобученной модели.
"""
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(model_name: str = "sberbank-ai/rugpt3small_based_on_gpt2"):
    """
    Загружает предобученную модель и токенизатор, сохраняет их локально.
    
    Args:
        model_name: Название модели из Hugging Face Hub
    """
    # Определяем пути
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "cogniflex" / "mlearning" / "cogniflex_models" / "text-generation"
    tokenizer_dir = project_root / "cogniflex" / "mlearning" / "tokenizers" / "fractal_unified_tokenizer"
    
    # Создаем директории, если их нет
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка модели {model_name}...")
    
    try:
        # Загружаем модель и токенизатор
        print("Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Загрузка модели...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Сохраняем модель и токенизатор
        print(f"Сохранение модели в {model_dir}...")
        model.save_pretrained(model_dir)
        
        print(f"Сохранение токенизатора в {tokenizer_dir}...")
        tokenizer.save_pretrained(tokenizer_dir)
        
        print("\n✅ Модель и токенизатор успешно загружены и сохранены!")
        print(f"Модель: {model_dir}")
        print(f"Токенизатор: {tokenizer_dir}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке модели: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Используем модель по умолчанию или можно передать другое название
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sberbank-ai/rugpt3small_based_on_gpt2"
    setup_model(model_name)
