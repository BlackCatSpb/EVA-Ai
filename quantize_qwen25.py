import os
from llama_cpp import Llama

project_root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_root, "eva", "memory", "fractal_torch_storage", "gguf_models")

input_file = os.path.join(output_dir, "qwen2.5-0.5b-f16.gguf")
output_file = os.path.join(output_dir, "qwen2.5-0.5b-instruct-q4_0-new.gguf")

print(f"Квантизация {input_file} в q4_0...")
print(f"Выход: {output_file}")

if not os.path.exists(input_file):
    print("Входной файл не найден!")
    exit(1)

if os.path.exists(output_file):
    print("Выходной файл уже существует, удаляем...")
    os.remove(output_file)

# Используем llama_cpp для квантизации
llm = Llama(model_path=input_file, verbose=True)

# Квантизация через quantize
llm.quantize(
    fname_out=output_file,
    ftype="q4_0"  # 4-bit quantization
)

print(f"Готово! Квантизированный GGUF: {output_file}")
