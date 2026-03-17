import os
import time
import argparse
from generation_coordinator import GenerationCoordinator

def print_banner():
    """Вывод приветственного баннера"""
    banner = """
    ╔══════════════════════════════════════════════╗
    ║          ПАРАЛЛЕЛЬНЫЙ ГЕНЕРАТОР ТЕКСТА       ║
    ║  с гибридным кешированием и обработкой CPU/GPU║
    ╚══════════════════════════════════════════════╝
    """
    print(banner)

def main():
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='Параллельный генератор текста с гибридным кешированием')
    parser.add_argument('--prompt', type=str, help='Текст промпта для генерации')
    parser.add_argument('--workers', type=int, default=4, help='Количество воркеров (по умолчанию: 4)')
    parser.add_argument('--max-tokens', type=int, default=500, help='Максимальное количество токенов (по умолчанию: 500)')
    parser.add_argument('--temp', type=float, default=0.7, help='Температура генерации (по умолчанию: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.9, help='Параметр top-p (по умолчанию: 0.9)')
    parser.add_argument('--interactive', action='store_true', help='Интерактивный режим')
    parser.add_argument('--clear-cache', action='store_true', help='Очистить кеш перед запуском')
    
    args = parser.parse_args()
    
    # Выводим баннер
    print_banner()
    
    # Инициализируем координатор
    print("Инициализация системы...")
    coordinator = GenerationCoordinator(num_workers=args.workers)
    
    # Очищаем кеш, если нужно
    if args.clear_cache:
        print("Очистка кеша...")
        coordinator.clear_cache()
    
    # Загружаем модель
    print("Загрузка модели...")
    if not coordinator.load_model():
        print("Ошибка: не удалось загрузить модель")
        return
    
    # Функция для генерации текста
    def generate_text(prompt):
        print(f"\n{'='*80}")
        print(f"ПРОМПТ: {prompt}")
        print(f"{'='*80}")
        
        start_time = time.time()
        print("Обработка запроса...")
        
        # Генерируем ответ
        result = coordinator.generate_response(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p
        )
        
        # Выводим результат
        if result['status'] == 'success':
            print("\n=== РЕЗУЛЬТАТ ===")
            print(result['generated_text'])
            print("\n=== ИНФОРМАЦИЯ ===")
            print(f"Время обработки: {result['processing_time']:.2f} сек")
            print(f"Сгенерировано токенов: {result['num_generated_tokens']}")
            print(f"Использовано устройство: {result['device']}")
        else:
            print(f"\nОШИБКА: {result.get('message', 'Неизвестная ошибка')}")
    
    # Режим интерактивной командной строки
    if args.interactive or not args.prompt:
        print("\nРежим интерактивной генерации (для выхода введите 'exit' или 'quit')")
        print("Настройки:")
        print(f"- Воркеров: {args.workers}")
        print(f"- Макс. токенов: {args.max_tokens}")
        print(f"- Температура: {args.temp}")
        print(f"- Top-p: {args.top_p}")
        print("\nВведите промпт:")
        
        while True:
            try:
                prompt = input(">>> ").strip()
                if prompt.lower() in ['exit', 'quit']:
                    break
                if prompt:
                    generate_text(prompt)
                    print("\nВведите следующий промпт (или 'exit' для выхода):")
            except KeyboardInterrupt:
                print("\nЗавершение работы...")
                break
            except Exception as e:
                print(f"Ошибка: {str(e)}")
    # Пакетный режим с одним промптом
    elif args.prompt:
        generate_text(args.prompt)
    
    # Закрываем соединения и освобождаем ресурсы
    print("\nЗавершение работы...")
    coordinator.clear_cache()

if __name__ == "__main__":
    main()
