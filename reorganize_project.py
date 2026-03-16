# Скрипт реорганизации проекта CogniFlex
# Запускать из директории C:\Users\black\OneDrive\Desktop\CogniFlex

import os
import shutil
from pathlib import Path

def create_directories():
    """Создание новой структуры директорий"""
    directories = [
        'debug',           # Отладочные файлы
        'logs',            # Логи
        'cache/all_caches', # Все кэши
        'tests',           # Тестовые файлы
        'scripts/cleanup', # Очистные скрипты
        'docs',            # Документация
        'config'           # Конфигурационные файлы
    ]

    for dir_path in directories:
        full_path = Path(dir_path)
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Создана директория: {full_path}")

def move_debug_files():
    """Перемещение отладочных и тестовых файлов"""
    debug_files = [
        'system_selftest.py',
        'system_selftest.log',
        'system_selftest copy.log',
        'test_fractal_store.py',
        'nlp_fallbacks.py',
        'dependency_report.log',
        'output.txt',
        'output_log.txt',
        'test_generation_integration.log',
        'test_output.txt'
    ]

    for file in debug_files:
        if os.path.exists(file):
            shutil.move(file, f'debug/{file}')
            print(f"Перемещен в debug/: {file}")

def move_cache_files():
    """Перемещение кэш-файлов"""
    cache_files = [
        'cogniflex_cache',
        'ethics_cache',
        'hybrid_cache',
        'ml_cache',
        'tokenizer_cache',
        'search_cache.json',
        'base_knowledge_v2.json',
        'initial_knowledge.json',
        'cogniflex_knowledge.db',
        'cogniflex_dialog_history.json',
        'dialog_history.json',
        'Dialog history.json',
        'bench_ctx_wd_safe.json',
        'gui_simulation_results.json',
        'heavy_token_bench_synth.json',
        'test_model.pt',
        'test_safetensors.safetensors'
    ]

    for file in cache_files:
        if os.path.exists(file):
            shutil.move(file, f'cache/all_caches/{file}')
            print(f"Перемещен в cache/all_caches/: {file}")

def move_log_files():
    """Перемещение лог-файлов"""
    # Пропускаем директорию logs, так как она уже создана
    # Если есть другие лог-файлы в корне, добавьте их сюда
    log_files = [
        # Добавьте сюда другие лог-файлы, если они есть
    ]
    
    for file in log_files:
        if os.path.exists(file):
            if os.path.isdir(file):
                shutil.move(file, f'logs/{file}')
            else:
                shutil.move(file, f'logs/{file}')
            print(f"Перемещен в logs/: {file}")
    
    print("Директория logs уже существует и готова для использования")

def move_scripts():
    """Перемещение скриптов"""
    if os.path.exists('scripts'):
        # Перемещаем содержимое папки scripts в scripts/cleanup
        scripts_path = Path('scripts')
        cleanup_path = Path('scripts/cleanup')

        # Перемещаем все файлы и подпапки, кроме cleanup
        for item in scripts_path.iterdir():
            if item.name != 'cleanup':  # Пропускаем папку cleanup
                destination = cleanup_path / item.name
                if item.is_file():
                    shutil.move(str(item), str(destination))
                    print(f"Перемещен скрипт: {item.name}")
                elif item.is_dir():
                    shutil.move(str(item), str(destination))
                    print(f"Перемещена папка скриптов: {item.name}")

        # Удаляем пустую папку scripts
        try:
            scripts_path.rmdir()
            print("Удалена пустая папка scripts/")
        except OSError:
            print("Не удалось удалить папку scripts/ - она не пустая")

def move_documentation():
    """Перемещение документации"""
    if os.path.exists('documentation'):
        shutil.move('documentation', 'docs/cogniflex_docs')
        print("Перемещена папка documentation в docs/cogniflex_docs/")

def move_config_files():
    """Перемещение конфигурационных файлов"""
    config_files = [
        'universal_analyzer_config.ini',
        'pytest.ini'
    ]

    for file in config_files:
        if os.path.exists(file):
            shutil.move(file, f'config/{file}')
            print(f"Перемещен в config/: {file}")

def clean_root_directory():
    """Очистка корневой директории от ненужных файлов"""
    files_to_remove = [
        'dependency_graph.dot',
        'dependency_graph.json',
        'CogniFlex.spec',
        'nonexistent'  # Папка с deltas.jsonl
    ]

    for file in files_to_remove:
        if os.path.exists(file):
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)
            print(f"Удален: {file}")

def main():
    """Основная функция реорганизации"""
    print("=== НАЧАЛО РЕОРГАНИЗАЦИИ ПРОЕКТА COGNIFLEX ===")
    print("Текущая директория:", os.getcwd())

    try:
        # Создание новой структуры
        print("\n1. Создание новой структуры директорий...")
        create_directories()

        # Перемещение файлов по категориям
        print("\n2. Перемещение отладочных файлов...")
        move_debug_files()

        print("\n3. Перемещение кэш-файлов...")
        move_cache_files()

        print("\n4. Перемещение лог-файлов...")
        move_log_files()

        print("\n5. Перемещение скриптов...")
        move_scripts()

        print("\n6. Перемещение документации...")
        move_documentation()

        print("\n7. Перемещение конфигурационных файлов...")
        move_config_files()

        print("\n8. Очистка корневой директории...")
        clean_root_directory()

        print("\n=== РЕОРГАНИЗАЦИЯ ЗАВЕРШЕНА ===")
        print("Новая структура:")
        print("- debug/          # Отладочные файлы")
        print("- logs/           # Логи")
        print("- cache/          # Кэши")
        print("- tests/          # Тесты")
        print("- scripts/        # Скрипты")
        print("- docs/           # Документация")
        print("- config/         # Конфигурация")
        print("- cogniflex/      # Основной код (сохраняется)")

    except Exception as e:
        print(f"Ошибка при реорганизации: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Реорганизация завершена успешно!")
        print("Проверьте новую структуру проекта.")
    else:
        print("\n❌ Произошла ошибка при реорганизации.")
