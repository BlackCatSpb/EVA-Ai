"""
Тест генерации обеих моделей Qwen.
Запустить: python -c "from eva.scripts.test_qwen_generation import test; test()"
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test():
    print("=" * 60)
    print("ТЕСТ ГЕНЕРАЦИИ QWEN")
    print("=" * 60)
    
    # Тест 1: Основная модель (qwen3.5-0.8b)
    print("\n[1] Тестирование основной модели qwen3.5-0.8b...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva\mlearning\eva_models\qwen3.5-0.8b"
        
        print("  - Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  - Токенизатор загружен (vocab_size={len(tokenizer)})")
        
        print("  - Загрузка модели на CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        print(f"  - Модель загружена, device={next(model.parameters()).device}")
        
        print("  - Генерация...")
        prompt = "Привет! Как дела?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  - Ответ: {response[:100]}...")
        print("  ✓ Основная модель работает!")
        
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
    
    # Тест 2: Fractal Qwen (для промтов)
    print("\n[2] Тестирование FractalQwenManager...")
    try:
        from eva.mlearning.fractal_qwen_manager import get_fractal_qwen
        
        fractal_qwen = get_fractal_qwen(device="cpu")
        status = fractal_qwen.get_status()
        print(f"  - Status: {status}")
        
        if fractal_qwen.initialized:
            prompt = fractal_qwen.generate_prompt(
                query="Тестовый вопрос",
                previous_response="Тестовый ответ",
                module_feedback={"test": "ok"}
            )
            print(f"  - Prompt: {prompt[:50]}...")
            print("  ✓ FractalQwen работает!")
        else:
            print("  ✗ FractalQwen не инициализирован")
            
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
    
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЁН")
    print("=" * 60)

if __name__ == "__main__":
    test()