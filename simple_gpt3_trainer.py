"""
Упрощенный скрипт самообучения до уровня GPT3
"""
import sys
import os
import time
import json
import logging
from datetime import datetime
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpt3_simple")

class SimpleGPT3Trainer:
    """Упрощенный тренер до уровня GPT3"""
    
    def __init__(self):
        """Инициализация тренера"""
        self.manager = None
        self.start_time = time.time()
        self.target_quality = 0.85  # Целевое качество GPT3
        self.current_quality = 0.0
        
        # Метрики
        self.metrics = {
            "total_sessions": 0,
            "successful_sessions": 0,
            "total_texts": 0,
            "total_searches": 0,
            "quality_history": []
        }
        
        print("🚀 Simple GPT3 Trainer инициализирован")
    
    def initialize(self):
        """Инициализация менеджера"""
        try:
            from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
            
            self.manager = UnifiedFractalManager()
            print(f"✅ Менеджер: {type(self.manager.manager).__name__}")
            
            return True
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            return False
    
    def get_current_quality(self):
        """Получает текущее качество"""
        try:
            quality_metrics = self.manager.get_quality_metrics()
            self.current_quality = quality_metrics.get('overall', 0.0)
            return self.current_quality
        except Exception as e:
            print(f"❌ Ошибка получения качества: {e}")
            return 0.0
    
    def should_train(self):
        """Определяет, нужно ли обучать"""
        current_quality = self.get_current_quality()
        
        # Обучаем если качество ниже целевого
        if current_quality < self.target_quality:
            print(f"📊 Качество {current_quality:.3f} < {self.target_quality:.3f} - нужно обучать")
            return True
        
        print(f"✅ Качество {current_quality:.3f} >= {self.target_quality:.3f} - цель достигнута")
        return False
    
    def training_session(self, topics=None):
        """Выполняет сессию обучения"""
        try:
            if topics is None:
                topics = [
                    "машинное обучение",
                    "нейронные сети", 
                    "искусственный интеллект",
                    "глубокое обучение",
                    "трансформеры"
                ]
            
            print(f"🎓 Запуск сессии обучения с темами: {topics}")
            
            # Запускаем сессию
            session_id = self.manager.start_enhanced_learning_session(
                topics=topics,
                session_name=f"gpt3_session_{int(time.time())}"
            )
            
            self.metrics["total_sessions"] += 1
            
            # Ждем завершения сессии с таймаутом
            max_wait_time = 120  # 2 минуты
            wait_interval = 5    # Проверяем каждые 5 секунд
            waited_time = 0
            
            print(f"⏳ Ожидание завершения сессии {session_id}...")
            
            while waited_time < max_wait_time:
                # Проверяем статус
                status = self.manager.get_enhanced_system_status()
                
                if 'sessions' in status and session_id in status['sessions']:
                    session_info = status['sessions'][session_id]
                    current_status = session_info.get('status', 'unknown')
                    
                    print(f"  📊 Статус: {current_status}")
                    
                    if current_status == 'completed':
                        print(f"✅ Сессия {session_id} завершена успешно")
                        self.metrics["successful_sessions"] += 1
                        
                        # Обновляем метрики
                        self.metrics["total_texts"] += session_info.get('training_texts', 0)
                        self.metrics["total_searches"] += session_info.get('web_searches', 0)
                        
                        # Сохраняем качество
                        quality_after = self.get_current_quality()
                        self.metrics["quality_history"].append(quality_after)
                        
                        return True
                    
                    elif current_status == 'failed':
                        print(f"❌ Сессия {session_id} завершилась с ошибкой")
                        return False
                    
                    elif current_status == 'active':
                        print(f"  🔄 Сессия активна, ждем...")
                
                time.sleep(wait_interval)
                waited_time += wait_interval
            
            # Если таймаут
            print(f"⏰ Сессия {session_id} не завершилась за {max_wait_time} секунд")
            
            # Принудительно проверяем финальный статус
            final_status = self.manager.get_enhanced_system_status()
            if 'sessions' in final_status and session_id in final_status['sessions']:
                session_info = final_status['sessions'][session_id]
                final_session_status = session_info.get('status', 'unknown')
                
                if final_session_status == 'completed':
                    print(f"✅ Сессия {session_id} завершена (проверка после таймаута)")
                    self.metrics["successful_sessions"] += 1
                    return True
                else:
                    print(f"❌ Сессия {session_id} не завершена. Статус: {final_session_status}")
                    return False
            else:
                print(f"❌ Сессия {session_id} не найдена в финальной проверке")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка сессии обучения: {e}")
            return False
    
    def test_generation(self):
        """Тестирует генерацию"""
        try:
            test_queries = [
                "Что такое машинное обучение?",
                "Как работают нейронные сети?",
                "Объясни концепцию искусственного интеллекта"
            ]
            
            results = []
            
            for query in test_queries:
                print(f"🧪 Тест: {query}")
                
                # Генерируем ответ
                response = self.manager.generate_enhanced_response(
                    query, 
                    max_tokens=100, 
                    use_web_search=True
                )
                
                # Оцениваем качество
                quality_score = self.evaluate_response(response.get('response', ''))
                
                result = {
                    "query": query,
                    "response": response.get('response', ''),
                    "quality": quality_score,
                    "web_search_used": response.get('web_search_used', False)
                }
                
                results.append(result)
                
                print(f"  📊 Качество: {quality_score:.3f}")
                print(f"  🔍 Поиск: {'Да' if response.get('web_search_used') else 'Нет'}")
                print(f"  📝 Ответ: {response.get('response', '')[:50]}...")
                print()
            
            # Средний балл
            avg_quality = sum(r['quality'] for r in results) / len(results)
            print(f"📈 Среднее качество: {avg_quality:.3f}")
            
            return avg_quality
            
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")
            return 0.0
    
    def evaluate_response(self, response):
        """Оценивает качество ответа"""
        score = 0.0
        
        # Длина
        if len(response) > 50:
            score += 0.2
        elif len(response) > 100:
            score += 0.3
        
        # Структура
        if any(word in response.lower() for word in ['потому что', 'так как', 'например']):
            score += 0.2
        
        # Информативность
        if len(response.split()) > 10:
            score += 0.2
        
        # Грамматика (простая проверка)
        if response.count('?') <= 1 and response.count('!') <= 1:
            score += 0.1
        
        # Релевантность (простая проверка)
        if any(word in response.lower() for word in ['машинное', 'нейрон', 'интеллект', 'обучение']):
            score += 0.2
        
        return min(1.0, score)
    
    def run_training_loop(self, max_sessions=10):
        """Запускает цикл обучения"""
        print("🚀 Запуск цикла обучения до уровня GPT3")
        print(f"🎯 Целевое качество: {self.target_quality:.3f}")
        
        for session_num in range(max_sessions):
            print(f"\n📊 Сессия {session_num + 1}/{max_sessions}")
            print("="*50)
            
            # Проверяем текущее качество
            current_quality = self.get_current_quality()
            print(f"📊 Текущее качество: {current_quality:.3f}")
            
            # Проверяем нужно ли обучать
            if not self.should_train():
                print("🎉 Цель достигнута!")
                break
            
            # Выполняем обучение
            success = self.training_session()
            
            if success:
                print(f"✅ Сессия {session_num + 1} успешна")
            else:
                print(f"❌ Сессия {session_num + 1} не удалась")
                continue
            
            # Тестирование
            print("\n🧪 Тестирование генерации...")
            test_quality = self.test_generation()
            
            # Статистика
            elapsed_time = time.time() - self.start_time
            print(f"\n📈 Статистика:")
            print(f"  ⏱️ Время: {elapsed_time:.1f}s")
            print(f"  🎓 Сессий: {self.metrics['total_sessions']}")
            print(f"  ✅ Успешных: {self.metrics['successful_sessions']}")
            print(f"  📚 Текстов: {self.metrics['total_texts']}")
            print(f"  🔍 Поисков: {self.metrics['total_searches']}")
            
            # Проверяем прогресс
            if test_quality >= self.target_quality:
                print(f"\n🎉 Целевое качество достигнуто: {test_quality:.3f}")
                break
            
            # Пауза между сессиями
            print("⏳ Пауза 30 секунд...")
            time.sleep(30)
        
        # Финальный отчет
        self.generate_final_report()
    
    def generate_final_report(self):
        """Генерирует финальный отчет"""
        print("\n" + "="*60)
        print("🎉 ФИНАЛЬНЫЙ ОТЧЕТ")
        print("="*60)
        
        final_quality = self.get_current_quality()
        elapsed_time = time.time() - self.start_time
        
        print(f"📊 Финальное качество: {final_quality:.3f}")
        print(f"🎯 Целевое качество: {self.target_quality:.3f}")
        print(f"⏱️ Общее время: {elapsed_time:.1f}s")
        print(f"🎓 Всего сессий: {self.metrics['total_sessions']}")
        print(f"✅ Успешных сессий: {self.metrics['successful_sessions']}")
        print(f"📚 Обучающих текстов: {self.metrics['total_texts']}")
        print(f"🔍 Веб-поисков: {self.metrics['total_searches']}")
        
        # Оценка результата
        if final_quality >= self.target_quality:
            print(f"\n🎉 УРОВЕНЬ GPT3 ДОСТИГНУТ!")
            print("✅ Модель готова к использованию")
        else:
            print(f"\n⚠️ Уровень GPT3 не достигнут")
            print(f"📊 Требуется дополнительное обучение")
            print(f"🎯 Разница: {self.target_quality - final_quality:.3f}")
        
        # История качества
        if self.metrics['quality_history']:
            print(f"\n📈 История качества:")
            for i, quality in enumerate(self.metrics['quality_history'], 1):
                print(f"  Сессия {i}: {quality:.3f}")
        
        # Сохраняем отчет
        report = {
            "final_quality": final_quality,
            "target_quality": self.target_quality,
            "elapsed_time": elapsed_time,
            "metrics": self.metrics,
            "gpt3_achieved": final_quality >= self.target_quality,
            "timestamp": time.time()
        }
        
        with open('gpt3_simple_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Отчет сохранен: gpt3_simple_report.json")
        print("="*60)

def main():
    """Основная функция"""
    print("🚀 Simple GPT3 Self-Training")
    print("="*50)
    print("Цель: Обучить модель до уровня GPT3")
    print("="*50)
    
    trainer = SimpleGPT3Trainer()
    
    # Инициализация
    if not trainer.initialize():
        print("❌ Инициализация не удалась")
        return 1
    
    # Запуск обучения
    trainer.run_training_loop(max_sessions=5)
    
    return 0

if __name__ == "__main__":
    main()
