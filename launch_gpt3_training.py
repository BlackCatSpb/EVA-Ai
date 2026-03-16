"""
Скрипт запуска и мониторинга самообучения до уровня GPT3
"""
import sys
import os
import time
import json
import argparse
from datetime import datetime
sys.path.append('.')

def main():
    """Основная функция запуска"""
    
    parser = argparse.ArgumentParser(description='GPT3 Self-Training Launcher')
    parser.add_argument('--mode', choices=['train', 'monitor', 'test'], default='train',
                       help='Режим работы: train - обучение, monitor - мониторинг, test - тесты')
    parser.add_argument('--config', type=str, default='gpt3_config.json',
                       help='Файл конфигурации')
    parser.add_argument('--resume', action='store_true',
                       help='Продолжить с предыдущего прогресса')
    parser.add_argument('--test-only', action='store_true',
                       help='Только запустить тесты')
    
    args = parser.parse_args()
    
    print("🚀 GPT3 SELF-TRAINING LAUNCHER")
    print("="*60)
    
    if args.mode == 'train':
        print("🎓 Режим: Обучение до уровня GPT3")
        print("🔄 Авто-запуск: Включен")
        print("📊 Мониторинг: Включен")
        print("⏱️ 24/7 Работа: Включена")
        
        # Импортируем и запускаем
        try:
            from gpt3_self_training import GPT3TrainingOrchestrator
            
            orchestrator = GPT3TrainingOrchestrator()
            
            # Восстановление прогресса если нужно
            if args.resume:
                orchestrator._load_progress()
            
            # Запуск
            success = orchestrator.run()
            
            if success:
                print("✅ Обучение завершено успешно")
            else:
                print("❌ Ошибка в процессе обучения")
                
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            return 1
    
    elif args.mode == 'monitor':
        print("📊 Режим: Мониторинг прогресса")
        
        try:
            # Загружаем прогресс
            with open('gpt3_training_progress.json', 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            progress = progress_data.get('progress', {})
            metrics = progress_data.get('current_metrics', {})
            stats = progress_data.get('training_stats', {})
            
            print(f"\n📈 Текущий прогресс: {progress.get('overall_completion', 0):.2%}")
            print(f"🤖 Размер модели: {progress.get('model_size_completion', 0):.2%}")
            print(f"⚡ Производительность: {progress.get('performance_completion', 0):.2%}")
            print(f"📚 Данные: {progress.get('data_completion', 0):.2%}")
            
            print(f"\n📊 Метрики:")
            print(f"  Качество: {metrics.get('performance', {}).get('quality_score', 0):.3f}")
            print(f"  Текстов: {metrics.get('data', {}).get('training_texts', 0):,}")
            print(f"  Поисков: {metrics.get('data', {}).get('total_web_searches', 0):,}")
            
            print(f"\n⏱️ Статистика:")
            print(f"  Сессий: {stats.get('total_sessions', 0)}")
            print(f"  Успешных: {stats.get('successful_sessions', 0)}")
            print(f"  Время: {stats.get('total_training_time', 0):.1f}s")
            
            if progress.get('estimated_completion_time'):
                eta = datetime.fromtimestamp(progress['estimated_completion_time'])
                print(f"  ETA: {eta.strftime('%Y-%m-%d %H:%M')}")
            
        except FileNotFoundError:
            print("❌ Файл прогресса не найден")
        except Exception as e:
            print(f"❌ Ошибка загрузки прогресса: {e}")
    
    elif args.mode == 'test':
        print("🧪 Режим: Тестирование генерации")
        
        try:
            from gpt3_self_training import GPT3TrainingOrchestrator
            
            orchestrator = GPT3TrainingOrchestrator()
            
            if not orchestrator.initialize_manager():
                print("❌ Ошибка инициализации")
                return 1
            
            # Запуск тестов
            print("🧪 Запуск тестов генерации...")
            test_results = orchestrator.run_generation_tests()
            
            print(f"\n📊 Результаты тестов:")
            print(f"  Статус: {test_results.get('status', 'unknown')}")
            print(f"  Общий балл: {test_results.get('overall_score', 0):.3f}")
            
            comparison = test_results.get('gpt3_comparison', {})
            print(f"  GPT3 эквивалент: {comparison.get('gpt3_equivalent', False)}")
            print(f"  Разница в баллах: {comparison.get('score_difference', 0):.3f}")
            
            if comparison.get('strengths'):
                print(f"\n✅ Сильные стороны:")
                for strength in comparison['strengths']:
                    print(f"  • {strength}")
            
            if comparison.get('weaknesses'):
                print(f"\n❌ Слабые стороны:")
                for weakness in comparison['weaknesses']:
                    print(f"  • {weakness}")
            
            if comparison.get('recommendations'):
                print(f"\n💡 Рекомендации:")
                for rec in comparison['recommendations']:
                    print(f"  • {rec}")
            
            # Детальные результаты
            tests = test_results.get('tests', [])
            if tests:
                print(f"\n📋 Детальные результаты:")
                for test in tests:
                    print(f"  📝 {test.get('query', 'Unknown')[:50]}...")
                    print(f"     Сложность: {test.get('complexity', 'unknown')}")
                    print(f"     Балл: {test.get('score', 0):.3f}")
                    print(f"     Время: {test.get('generation_time', 0):.3f}s")
                    print(f"     Поиск: {'Да' if test.get('web_search_used') else 'Нет'}")
                    print()
            
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
