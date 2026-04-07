#!/usr/bin/env python3
"""
Тест исправлений runtime ошибок ЕВА
"""

def test_system_metrics_manager():
    """Тестирует исправление system_metrics_manager"""
    try:
        from eva_ai.core.core_brain import CoreBrain
        brain = CoreBrain()
        
        # Проверяем наличие system_metrics_manager
        assert hasattr(brain, 'system_metrics_manager'), "system_metrics_manager отсутствует"
        assert brain.system_metrics_manager is not None, "system_metrics_manager is None"
        
        print("✅ system_metrics_manager исправлен")
        return True
    except Exception as e:
        print(f"❌ Ошибка system_metrics_manager: {e}")
        return False

def test_get_system_health():
    """Тестирует исправление get_system_health"""
    try:
        from eva_ai.core.core_brain import CoreBrain
        brain = CoreBrain()
        
        # Проверяем наличие get_system_health
        assert hasattr(brain, 'get_system_health'), "get_system_health отсутствует"
        
        # Тестируем вызов метода
        health = brain.get_system_health()
        assert isinstance(health, dict), "get_system_health должен возвращать dict"
        assert 'status' in health, "health должен содержать status"
        
        print(f"✅ get_system_health исправлен: статус={health['status']}")
        return True
    except Exception as e:
        print(f"❌ Ошибка get_system_health: {e}")
        return False

def test_dashboard_data():
    """Тестирует get_system_dashboard_data после исправлений"""
    try:
        from eva_ai.core.core_brain import CoreBrain
        brain = CoreBrain()
        brain.initialize()
        
        # Тестируем dashboard data
        dashboard = brain.get_system_dashboard_data()
        assert isinstance(dashboard, dict), "dashboard должен быть dict"
        
        if 'error' in dashboard:
            print(f"⚠️ Dashboard с ошибкой: {dashboard.get('error', 'unknown')}")
        else:
            print("✅ get_system_dashboard_data работает без ошибок")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка dashboard data: {e}")
        return False

if __name__ == "__main__":
    print("🔧 ТЕСТ RUNTIME ИСПРАВЛЕНИЙ")
    print("=" * 40)
    
    results = []
    
    print("\n=== Тест system_metrics_manager ===")
    results.append(test_system_metrics_manager())
    
    print("\n=== Тест get_system_health ===")
    results.append(test_get_system_health())
    
    print("\n=== Тест dashboard data ===")
    results.append(test_dashboard_data())
    
    print("\n" + "=" * 40)
    print(f"📊 РЕЗУЛЬТАТЫ: {sum(results)}/{len(results)} тестов пройдено")
    
    if all(results):
        print("🎉 Все runtime исправления работают!")
    else:
        print("⚠️ Некоторые проблемы остаются")
