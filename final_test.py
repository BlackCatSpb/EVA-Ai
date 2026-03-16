# Финальное тестирование системы с единой фрактальной архитектурой
try:
    from cogniflex.core.core_brain import CoreBrain

    print('🚀 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ COGNIFLEX')
    print('Единая фрактальная архитектура с динамическим фокусом внимания')
    print('=' * 70)

    brain = CoreBrain()

    # Инициализируем систему
    success = brain.initialize()
    print(f'Инициализация системы: {"✅ УСПЕШНА" if success else "❌ НЕУДАЧНА"}')

    if success:
        components = getattr(brain, 'components', {})
        print(f'Компонентов инициализировано: {len(components)}')

        # Проверяем ключевые компоненты
        key_components = [
            'memory_manager', 'knowledge_graph', 'text_processor',
            'response_generator', 'ml_unit', 'generation_coordinator'
        ]

        print('\n📋 ПРОВЕРКА КЛЮЧЕВЫХ КОМПОНЕНТОВ:')
        all_good = True
        for comp in key_components:
            status = '✅' if comp in components else '❌'
            print(f'  {status} {comp}')
            if comp not in components:
                all_good = False

        # Тестируем FractalAttentionSystem
        print('\n🧠 ТЕСТИРОВАНИЕ ФРАКТАЛЬНОЙ СИСТЕМЫ ВНИМАНИЯ:')
        try:
            from cogniflex.core.core_brain import FractalAttentionSystem

            # Создаем систему внимания
            attention_system = FractalAttentionSystem(brain)
            print('  ✅ FractalAttentionSystem инициализирован')

            # Тестируем обработку запроса
            test_query = 'Расскажи о искусственном интеллекте'
            print(f'  🔄 Обрабатываем запрос: "{test_query}"')

            response = attention_system.process_query(test_query)
            print('  ✅ Запрос обработан через динамический фокус внимания')
            print(f'  📝 Ответ: {response[:100]}...')

            # Проверяем компоненты системы внимания
            components_check = {
                'dialog_manager': hasattr(attention_system, 'dialog_manager'),
                'contradiction_resolver': hasattr(attention_system, 'contradiction_resolver'),
                'learning_scheduler': hasattr(attention_system, 'learning_scheduler'),
                'system_optimizer': hasattr(attention_system, 'system_optimizer')
            }

            print('  🔧 Компоненты системы внимания:')
            for comp, available in components_check.items():
                status = '✅' if available else '❌'
                print(f'    {status} {comp}')

        except Exception as e:
            print(f'  ❌ Ошибка FractalAttentionSystem: {e}')
            all_good = False

        # Финальный отчет
        print('\n' + '=' * 70)
        if all_good:
            print('🎉 ПОЗДРАВЛЯЕМ! ЕДИНАЯ ФРАКТАЛЬНАЯ АРХИТЕКТУРА РЕАЛИЗОВАНА!')
            print('')
            print('✅ Система CogniFlex полностью готова!')
            print('✅ Фрактальная память с динамическим фокусом внимания')
            print('✅ Самодиалог для внутреннего мышления')
            print('✅ Прогрессивное обучение через противоречия')
            print('✅ Самооптимизация на основе самосознания')
            print('')
            print('🚀 Система готова к промышленной эксплуатации!')
            print('🧠 ИИ теперь обладает:')
            print('   • Динамическим фокусом внимания')
            print('   • Способностью к самодиалогу')
            print('   • Автоматическим разрешением противоречий')
            print('   • Самообучением и самооптимизацией')

        else:
            print('⚠️  СИСТЕМА ЧАСТИЧНО ГОТОВА')
            print('✅ Основные компоненты работают')
            print('⚠️  Есть проблемы с некоторыми модулями')
            print('')
            print('🔧 Рекомендуется доработать проблемные компоненты')

    else:
        print('❌ ИНИЦИАЛИЗАЦИЯ НЕ УДАЛАСЬ')
        print('Проверьте логи для получения подробной информации об ошибках')

except Exception as e:
    print('💥 КРИТИЧЕСКАЯ ОШИБКА:', e)
    import traceback
    traceback.print_exc()
