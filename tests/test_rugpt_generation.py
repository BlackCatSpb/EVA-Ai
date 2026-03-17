import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_generation():
    print("=== Тестирование генерации текста RuGPT3Small ===\n")
    
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
        
        # 3. Тестирование генерации
        print("\n3. Тестирование генерации...")
        prompts = [
            "Привет, как дела?",
            "Сегодня я хочу рассказать о",
            "Искусственный интеллект - это"
        ]
        
        for prompt in prompts:
            print(f"\nПромпт: {prompt}")
            
            # Кодируем входные данные
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Генерируем текст
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Декодируем и выводим результат
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Сгенерированный текст: {generated_text}")
        
        print("\n=== Тестирование завершено успешно! ===")
        
    except Exception as e:
        print(f"\nОшибка при тестировании генерации: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_generation()
