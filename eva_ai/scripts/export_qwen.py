"""
Скрипт экспорта модели Qwen во фрактальное хранилище.
Запустить: python -c "from eva_ai.scripts.export_qwen import export; export()"
"""
import os
import sys

def export():
    # Добавляем путь для импорта eva модулей
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from eva_ai.memory.fractal_torch_storage.model_exporter import ModelExporter
    
    model_path = r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva\mlearning\eva_models\qwen3.5-0.8b"
    model_name = "qwen3.5-0.8b"
    
    print(f"Экспорт модели {model_name} из {model_path}...")
    
    exporter = ModelExporter()
    result = exporter.export_model(
        model_path=model_path,
        model_name=model_name,
        quantization=None,  # Без квантизации - full precision
        device="cpu"
    )
    
    print(f"Статус: {result['status']}")
    if result['status'] == 'success':
        print(f"Слоёв: {result['layers_exported']}")
        print(f"Весов: {result['total_weights']}")
        print(f"Размер: {result['total_bytes'] / (1024**2):.1f} MB")
        print(f"Метаданные: {result['metadata_path']}")
    else:
        print(f"Ошибки: {result['errors']}")

if __name__ == "__main__":
    export()