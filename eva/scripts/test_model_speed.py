"""
Тест скорости загрузки и генерации моделей Qwen.
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}с"
    else:
        return f"{seconds/60:.1f}мин"


def test_main_model():
    """Тест основной модели qwen3.5-0.8b"""
    print("\n" + "="*60)
    print("ТЕСТ: Основная модель qwen3.5-0.8b")
    print("="*60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva\mlearning\eva_models\qwen3.5-0.8b"
    
    # Загрузка токенизатора
    print("\n[1] Загрузка токенизатора...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tok_time = time.time() - start
    print(f"    Время: {format_time(tok_time)}")
    print(f"    Vocab size: {len(tokenizer)}")
    
    # Загрузка модели
    print("\n[2] Загрузка модели на CPU...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    load_time = time.time() - start
    print(f"    Время: {format_time(load_time)}")
    print(f"    Устройство: {next(model.parameters()).device}")
    
    # Генерация (короткий тест)
    print("\n[3] Генерация (50 токенов)...")
    prompt = "Привет! Как дела?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    gen_time = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = len(outputs[0]) - len(inputs[0])
    
    print(f"    Время: {format_time(gen_time)}")
    print(f"    Токенов сгенерировано: {tokens_generated}")
    print(f"    Скорость: {tokens_generated/gen_time:.1f} токенов/сек")
    print(f"    Ответ: {response[:80]}...")
    
    # Генерация (длинный тест)
    print("\n[4] Генерация (200 токенов)...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    gen_time2 = time.time() - start
    
    tokens_gen2 = len(outputs[0]) - len(inputs[0])
    print(f"    Время: {format_time(gen_time2)}")
    print(f"    Токенов сгенерировано: {tokens_gen2}")
    print(f"    Скорость: {tokens_gen2/gen_time2:.1f} токенов/сек")
    
    # Очистка памяти
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "tok_time": tok_time,
        "load_time": load_time,
        "gen_time_50": gen_time,
        "gen_time_200": gen_time2,
        "speed_50": tokens_generated/gen_time,
        "speed_200": tokens_gen2/gen_time2
    }


def test_fractal_model():
    """Тест модели во фрактальном хранилище"""
    print("\n" + "="*60)
    print("ТЕСТ: Fractal Qwen (фрактальное хранилище)")
    print("="*60)
    
    from eva.mlearning.fractal_qwen_manager import get_fractal_qwen
    
    # Загрузка через FractalQwenManager
    print("\n[1] Загрузка через FractalQwenManager...")
    start = time.time()
    fractal_qwen = get_fractal_qwen(device="cpu", force_reload=True)
    load_time = time.time() - start
    
    status = fractal_qwen.get_status()
    print(f"    Время: {format_time(load_time)}")
    print(f"    Status: {status}")
    
    if not fractal_qwen.initialized:
        print("    ОШИБКА: Модель не инициализирована!")
        return None
    
    # Генерация промта
    print("\n[2] Генерация промта...")
    start = time.time()
    prompt = fractal_qwen.generate_prompt(
        query="Тестовый вопрос",
        previous_response="Тестовый ответ",
        module_feedback={"test": "ok"}
    )
    gen_time = time.time() - start
    
    print(f"    Время: {format_time(gen_time)}")
    print(f"    Prompt: {prompt[:100]}...")
    
    # Тест полной генерации с моделью
    print("\n[3] Прямая генерация (50 токенов)...")
    if fractal_qwen.model and fractal_qwen.tokenizer:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Используем ту же модель что загружена во fractal
        model = fractal_qwen.model
        tokenizer = fractal_qwen.tokenizer
        
        prompt = "Привет! Как дела?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        gen_time2 = time.time() - start
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = len(outputs[0]) - len(inputs[0])
        
        print(f"    Время: {format_time(gen_time2)}")
        print(f"    Токенов: {tokens}")
        print(f"    Скорость: {tokens/gen_time2:.1f} токенов/сек")
        print(f"    Ответ: {response[:80]}...")
        
        return {
            "load_time": load_time,
            "gen_prompt": gen_time,
            "gen_time_50": gen_time2,
            "speed_50": tokens/gen_time2
        }
    else:
        print("    Модель недоступна для генерации")
        return {"load_time": load_time, "gen_prompt": gen_time}


def main():
    print("="*60)
    print("СРАВНЕНИЕ СКОРОСТЕЙ МОДЕЛЕЙ QWEN")
    print("="*60)
    
    # Тест основной модели
    main_results = test_main_model()
    
    # Тест фрактальной модели
    fractal_results = test_fractal_model()
    
    # Итоговое сравнение
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*60)
    
    if main_results:
        print(f"\nОсновная модель (qwen3.5-0.8b):")
        print(f"  Загрузка: {format_time(main_results['load_time'])}")
        print(f"  Генерация 50 токенов: {format_time(main_results['gen_time_50'])} ({main_results['speed_50']:.1f} ток/сек)")
        print(f"  Генерация 200 токенов: {format_time(main_results['gen_time_200'])} ({main_results['speed_200']:.1f} ток/сек)")
    
    if fractal_results:
        print(f"\nFractal Qwen (фрактальное хранилище):")
        print(f"  Загрузка: {format_time(fractal_results['load_time'])}")
        if 'gen_time_50' in fractal_results:
            print(f"  Генерация 50 токенов: {format_time(fractal_results['gen_time_50'])} ({fractal_results['speed_50']:.1f} ток/сек)")
        else:
            print(f"  Генерация промта: {format_time(fractal_results.get('gen_prompt', 0))}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()