"""
Тест интеграции генерации текста в ЕВА.

Этот скрипт демонстрирует, как использовать интегрированную функциональность
генерации текста в CoreBrain.
"""
import os
import sys
import logging
import time
from typing import Dict, Any, Optional

# Настройка логирования в файл
log_file = 'test_generation_integration.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger('test_generation')

# Добавляем путь к корню проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем CoreBrain
from eva.core.core_brain import CoreBrain

class TestEventSystem:
    """Простая реализация системы событий для тестирования."""
    
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """Подписывает обработчик на событие."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, data):
        """Публикует событие."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Ошибка в обработчике события {event_type}: {e}")

def print_header(text: str, width: int = 80):
    """Печатает заголовок с рамкой."""
    print("\n" + "=" * width)
    print(f"{text:^{width}}")
    print("=" * width + "\n")

def test_text_generation():
    """Тестирует генерацию текста через CoreBrain."""
    print_header("ТЕСТ ИНТЕГРАЦИИ ГЕНЕРАЦИИ ТЕКСТА В COGNIFLEX")
    
    # Очищаем предыдущий лог-файл
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception as e:
            print(f"Не удалось удалить старый лог-файл: {e}")
    
    # Перенаправляем stdout и stderr в лог-файл
    sys.stdout = sys.stderr = open(log_file, 'w', encoding='utf-8')
    
    # Дублируем вывод в консоль
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Создаем объект для дублирования вывода
    tee = Tee(sys.stdout, sys.stderr)
    sys.stdout = sys.stderr = tee
    
    # Создаем тестовую систему событий
    event_system = TestEventSystem()
    
    # Обработчики событий
    def on_generation_ready(data):
        logger.info(f"Событие generation_ready: модель {data.get('model')} готова")
        
    def on_generation_complete(data):
        logger.info(f"Событие generation_complete: сгенерировано {data.get('num_tokens', 0)} токенов")
        
    def on_generation_error(data):
        logger.error(f"Событие generation_error: {data.get('error')}")
    
    # Подписываемся на события
    event_system.subscribe('generation_ready', on_generation_ready)
    event_system.subscribe('generation_complete', on_generation_complete)
    event_system.subscribe('generation_error', on_generation_error)
    
    # Конфигурация для CoreBrain
    config = {
        'debug_minimal_mode': True,  # Используем минимальный режим для тестов
        'generation': {
            'enabled': True,
            'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
            'max_tokens': 200,  # Уменьшаем для ускорения тестов
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.2,
            'cache_dir': './test_generation_cache',
            'num_workers': 2
        }
    }
    
    # Инициализируем CoreBrain
    logger.info("Инициализация CoreBrain...")
    try:
        brain = CoreBrain(config=config)
        # Подменяем систему событий на нашу тестовую
        brain.events = event_system
    except Exception as e:
        logger.error(f"Ошибка при инициализации CoreBrain: {e}")
        return
    
    try:
        # Проверяем статус генерации
        status = brain.get_generation_status()
        print("\n=== Статус генерации текста ===")
        for key, value in status.items():
            if key != 'config':  # Выводим конфиг отдельно
                print(f"  {key}: {value}")
        
        if 'config' in status:
            print("\n=== Конфигурация генерации ===")
            for key, value in status['config'].items():
                print(f"  {key}: {value}")
        
        # Тестируем генерацию текста
        test_prompts = [
            "Расскажи о возможностях искусственного интеллекта в современном мире",
            "Какие существуют подходы к машинному обучению?",
            "Объясни, как работает трансформерная архитектура в NLP"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n=== Тест генерации {i}/{len(test_prompts)} ===")
            print(f"Промпт: {prompt}")
            
            try:
                # Замеряем время выполнения
                start_time = time.time()
                
                # Генерируем ответ
                logger.info(f"Генерация ответа на промпт: {prompt}")
                response = brain.generate_text(
                    prompt=prompt,
                    max_tokens=150,  # Ограничиваем для ускорения тестов
                    temperature=0.7
                )
                
                # Выводим результат
                if isinstance(response, dict) and 'generated_text' in response:
                    print("\nСгенерированный ответ:")
                    print("-" * 80)
                    print(response['generated_text'])
                    print("-" * 80)
                    
                    if 'num_tokens' in response:
                        print(f"\nИспользовано токенов: {response['num_tokens']}")
                    
                    # Выводим метрики
                    elapsed = time.time() - start_time
                    tokens_per_sec = response.get('num_tokens', 0) / elapsed if elapsed > 0 else 0
                    print(f"Время генерации: {elapsed:.2f} сек. (~{tokens_per_sec:.1f} токенов/сек)")
                    
                else:
                    logger.warning(f"Неожиданный формат ответа: {type(response)}")
                    print(f"Неожиданный формат ответа: {response}")
                    
            except Exception as e:
                logger.error(f"Ошибка при генерации текста: {e}", exc_info=True)
                print(f"Ошибка при генерации текста: {e}")
        
        # Тестируем очистку кеша
        print("\n=== Тестирование очистки кеша ===")
        try:
            brain.clear_generation_cache()
            print("Кеш успешно очищен")
        except Exception as e:
            logger.error(f"Ошибка при очистке кеша: {e}", exc_info=True)
            print(f"Ошибка при очистке кеша: {e}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка при тестировании: {e}", exc_info=True)
        print(f"\n!!! Критическая ошибка: {e}")
    
    finally:
        # Завершаем работу
        print("\n=== Завершение тестирования ===")
        try:
            # Очищаем ресурсы
            if 'brain' in locals() and hasattr(brain, 'cleanup'):
                brain.cleanup()
            print("Ресурсы освобождены")
        except Exception as e:
            logger.error(f"Ошибка при освобождении ресурсов: {e}", exc_info=True)
        
        print("\nТестирование завершено. Подробности в лог-файле:", os.path.abspath(log_file))

if __name__ == "__main__":
    test_text_generation()
