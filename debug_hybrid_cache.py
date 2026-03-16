#!/usr/bin/env python3
"""
Отладка создания HybridTokenCache
"""
import sys
import os
sys.path.append('.')

def debug_hybrid_cache():
    """Отладка создания HybridTokenCache"""
    print('🔍 ОТЛАДКА HYBRID TOKEN CACHE')
    print('=' * 50)
    
    try:
        from cogniflex.memory.hybrid_token_cache import HybridTokenCache
        print('✅ HybridTokenCache импортирован')
        
        # Создаем mock brain
        class MockBrain:
            def __init__(self):
                self.components = {}
                self.cache_dir = './test_cache'
        
        brain = MockBrain()
        print('✅ Mock brain создан')
        
        # Пробуем создать HybridTokenCache
        try:
            print('🔄 Создание HybridTokenCache...')
            hybrid_cache = HybridTokenCache(brain=brain)
            print('✅ HybridTokenCache создан')
            print(f'   Тип: {type(hybrid_cache).__name__}')
            
            # Проверяем методы
            methods = ['put', 'get', 'get_cache_stats']
            for method in methods:
                if hasattr(hybrid_cache, method):
                    print(f'   ✅ Есть метод: {method}')
                else:
                    print(f'   ❌ Нет метода: {method}')
            
            return hybrid_cache
            
        except Exception as e:
            print(f'❌ Ошибка создания HybridTokenCache: {e}')
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f'❌ Ошибка импорта: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    cache = debug_hybrid_cache()
    if cache:
        print('\n✅ Тест успешно завершен')
    else:
        print('\n❌ Тест не удался')
