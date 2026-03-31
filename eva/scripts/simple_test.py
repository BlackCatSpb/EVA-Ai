"""
Простой тест генерации - запускать по частям.
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_generation():
    """Базовая генерация без сложных зависимостей"""
    print("="*50)
    print("ТЕСТ ГЕНЕРАЦИИ QWEN")
    print("="*50)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva\mlearning\eva_models\qwen3.5-0.8b"
    
    print("\n1. Загрузка токенизатора...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Готово за {time.time()-t0:.1f}с")
    
    print("\n2. Загрузка модели...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    load_time = time.time() - t0
    print(f"   Готово за {load_time:.1f}с")
    print(f"   Device: {next(model.parameters()).device}")
    
    print("\n3. Генерация (100 токенов)...")
    prompt = "Привет! Как дела?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    gen_time = time.time() - t0
    
    tokens = len(outputs[0]) - len(inputs[0])
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    speed = tokens / gen_time
    
    print(f"   Время: {gen_time:.1f}с")
    print(f"   Токенов: {tokens}")
    print(f"   Скорость: {speed:.1f} ток/сек")
    print(f"   Ответ: {response[:150]}...")
    
    print("\n" + "="*50)
    print("ТЕСТ УСПЕШЕН")
    print("="*50)

if __name__ == "__main__":
    test_basic_generation()