with open('output_log.txt', 'w', encoding='utf-8') as f:
    f.write("Начало теста\n")
    
    try:
        import torch
        f.write(f"PyTorch версия: {torch.__version__}\n")
        f.write(f"CUDA доступен: {torch.cuda.is_available()}\n")
        
        if torch.cuda.is_available():
            f.write(f"Устройство CUDA: {torch.cuda.get_device_name(0)}\n")
            
        # Простой тест с тензором
        x = torch.rand(2, 2)
        f.write(f"Тестовый тензор:\n{x}\n")
        f.write("Тест завершен успешно!\n")
        
    except Exception as e:
        f.write(f"Ошибка: {str(e)}\n")
        import traceback
        f.write(traceback.format_exc())

print("Файл output_log.txt создан")
