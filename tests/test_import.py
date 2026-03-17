import sys
print("Python version:", sys.version)
print("\nPython path:", sys.path)

try:
    import cogniflex
    print("\nПакет cogniflex успешно импортирован!")
    if hasattr(cogniflex, '__version__'):
        print(f"Версия: {cogniflex.__version__}")
    else:
        print("Атрибут __version__ не найден в пакете cogniflex")
    
    # Проверка доступности подмодулей
    print("\nДоступные подмодули:")
    for name in dir(cogniflex):
        if not name.startswith('_'):
            print(f"- {name}")
            
except Exception as e:
    print(f"\nОшибка при импорте пакета cogniflex: {e}")
