#!/usr/bin/env python3
"""
Проверка гибридного кэша в системе
"""
import sys
sys.path.append('.')

def main():
    from cogniflex.core.core_brain import CoreBrain
    brain = CoreBrain()
    brain.initialize()

    # Проверяем все компоненты
    print('📋 ДОСТУПНЫЕ КОМПОНЕНТЫ:')
    for name, component in brain.components.items():
        print(f'   {name}: {type(component).__name__}')

    # Проверяем гибридный кэш напрямую
    if 'hybrid_cache' in brain.components:
        cache = brain.components['hybrid_cache']
        print(f'✅ Гибридный кэш найден: {type(cache).__name__}')
        
        if hasattr(cache, 'get_cache_stats'):
            stats = cache.get_cache_stats()
            print(f'📊 Статистика кэша:')
            print(f'   VRAM токенов: {stats.get("vram_tokens", 0)}')
            print(f'   RAM токенов: {stats.get("ram_tokens", 0)}')
            print(f'   Disk токенов: {stats.get("disk_tokens", 0)}')
        
        if hasattr(cache, 'disk_cache') and hasattr(cache.disk_cache, 'get_stats'):
            disk_stats = cache.disk_cache.get_stats()
            print(f'💾 Дисковый кэш:')
            print(f'   Файлов: {disk_stats.get("total_files", 0)}')
            print(f'   Размер: {disk_stats.get("total_size_mb", 0):.2f} MB')
            print(f'   Лимит: {disk_stats.get("max_size_gb", 0):.1f} GB')
    else:
        print('❌ Гибридный кэш не найден в brain.components')

if __name__ == "__main__":
    main()
