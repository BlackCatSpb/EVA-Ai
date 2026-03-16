"""
Упрощенный тест гибридного кэша токенов
"""

import os
import sys
import time

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def main():
    print("=== Простой тест гибридного кэша ===")
    
    try:
        # Создаем минимальный brain
        class TestBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                print(f"Cache dir: {self.cache_dir}")
        
        brain = TestBrain()
        
        # Импортируем кэш
        print("Импортируем HybridTokenCache...")
        from memory.hybrid_token_cache import HybridTokenCache
        
        print("Создаем кэш...")
        cache = HybridTokenCache(brain, max_memory_tokens=10)
        
        print("Тестируем базовые операции...")
        
        # Добавляем токены
        for i in range(5):
            token_id = f"test_token_{i}"
            token_data = {"text": f"Token {i}", "value": i}
            cache.add_token(token_id, token_data)
            print(f"Добавлен токен: {token_id}")
        
        # Получаем токены
        for i in range(5):
            token_id = f"test_token_{i}"
            result = cache.get_token(token_id)
            if result:
                print(f"Получен токен: {token_id} -> {result}")
            else:
                print(f"Токен не найден: {token_id}")
        
        # Статистика
        if hasattr(cache, 'get_cache_stats'):
            stats = cache.get_cache_stats()
            print(f"Статистика: {stats}")
        else:
            print("Статистика недоступна")
        
        print("✅ Тест успешно завершен!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
