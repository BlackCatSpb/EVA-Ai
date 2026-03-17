print("=== Начало теста ===\n")

# Проверка базового Python
print("1. Проверка базового Python...")
print(f"Версия Python: {__import__('sys').version}")
print(f"Текущая директория: {__import__('os').getcwd()}")

# Проверка PyTorch
try:
    import torch
    print("\n2. Проверка PyTorch...")
    print(f"Версия PyTorch: {torch.__version__}")
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Устройство CUDA: {torch.cuda.get_device_name(0)}")
    
    # Простой тест с тензором
    x = torch.rand(2, 3)
    print(f"\nТестовый тензор:\n{x}")
    print(f"Умножение тензора на себя:\n{x @ x.T}")
    
except Exception as e:
    print(f"Ошибка при проверке PyTorch: {e}")
    import traceback
    traceback.print_exc()

# Проверка загрузки модели
try:
    print("\n3. Попытка загрузки модели...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print("Загрузка токенизатора...")
    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Токенизатор загружен успешно")
    
    # Проверка кодирования текста
    text = "Привет, как дела?"
    encoded = tokenizer.encode(text, return_tensors="pt")
    print(f"\nТекст: {text}")
    print(f"Закодированный текст: {encoded.tolist()}")
    print(f"Декодированный текст: {tokenizer.decode(encoded[0])}")
    
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Тест завершен ===")
