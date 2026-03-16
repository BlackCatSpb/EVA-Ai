import torch

print('🔍 Анализ доступности GPU и токенизации...')
print(f'CUDA доступна: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA устройств: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Устройство {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f'  Память: {memory_gb:.1f} GB')
else:
    print('CUDA недоступна, используется CPU')

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Текущее устройство по умолчанию: {default_device}')
print(f'PyTorch версия: {torch.__version__}')

try:
    cuda_version = torch.version.cuda if torch.cuda.is_available() else 'N/A'
    print(f'CUDA версия: {cuda_version}')
except:
    print('CUDA версия: N/A')

# Анализ токенизации
print('\n🔍 Анализ токенизации...')
device = torch.device(default_device)
print(f'Устройство для токенизации: {device}')

# Тест токенизации на разных устройствах
try:
    from transformers import AutoTokenizer
    
    # Загружаем простой токенизатор для теста
    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    
    test_text = "Привет, как дела?"
    
    # Токенизация на CPU
    print('\n📊 Токенизация на CPU:')
    inputs_cpu = tokenizer(test_text, return_tensors='pt')
    print(f'  Форма input_ids: {inputs_cpu["input_ids"].shape}')
    print(f'  Устройство тензора: {inputs_cpu["input_ids"].device}')
    
    # Токенизация на GPU если доступно
    if torch.cuda.is_available():
        print('\n📊 Токенизация на GPU:')
        inputs_gpu = tokenizer(test_text, return_tensors='pt')
        inputs_gpu = {k: v.to('cuda') for k, v in inputs_gpu.items()}
        print(f'  Форма input_ids: {inputs_gpu["input_ids"].shape}')
        print(f'  Устройство тензора: {inputs_gpu["input_ids"].device}')
        
except Exception as e:
    print(f'Ошибка при анализе токенизации: {e}')
