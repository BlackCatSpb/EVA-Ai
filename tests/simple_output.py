with open('test_output.txt', 'w', encoding='utf-8') as f:
    f.write("Это тестовая строка для проверки вывода в файл\n")
    f.write(f"Python работает корректно!\n")
    
    # Проверка PyTorch
    try:
        import torch
        f.write(f"PyTorch версия: {torch.__version__}\n")
        f.write(f"CUDA доступен: {torch.cuda.is_available()}\n")
    except Exception as e:
        f.write(f"Ошибка при импорте PyTorch: {e}\n")

print("Тестовый файл создан: test_output.txt")
