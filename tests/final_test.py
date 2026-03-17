import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    print("=== Тестирование загрузки модели и генерации ===\n")
    
    # Проверка устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    
    try:
        # 1. Загрузка токенизатора
        print("\n1. Загрузка токенизатора...")
        tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Токенизатор загружен успешно")
        
        # 2. Загрузка модели
        print("\n2. Загрузка модели...")
        model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(device)
        model.eval()
        print("Модель загружена успешно")
        
        # 3. Простая генерация
        print("\n3. Тест генерации...")
        prompt = "Привет! Как твои дела?"
        print(f"Промпт: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nСгенерированный текст:")
        print(generated_text)
        
        print("\n=== Тест завершен успешно! ===")
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
