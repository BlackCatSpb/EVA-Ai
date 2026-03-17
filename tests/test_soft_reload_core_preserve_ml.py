import pytest

from cogniflex.core.core_brain import CoreBrain


def test_soft_reload_preserves_ml_where_present():
    core = CoreBrain()
    assert core.initialize(), "Инициализация ядра должна завершиться успешно"

    # Запуск
    assert core.start(), "Запуск ядра должен завершиться успешно"

    # Захватываем ссылки на ML-компоненты (если они есть в сборке)
    ml_before = core.components.get('ml_unit')
    mm_before = getattr(core, 'model_manager', None)

    # Выполняем soft-reload
    ok = core.soft_reload(reload_gui=False)
    assert ok, "soft_reload должен вернуть True"

    # Ядро должно быть в рабочем состоянии
    assert core.initialized is True
    assert core.running is True

    # Если ML был, он должен сохраниться (та же ссылка)
    if ml_before is not None:
        assert core.components.get('ml_unit') is ml_before, "ml_unit должен сохраняться при soft-reload"
    if mm_before is not None:
        assert getattr(core, 'model_manager', None) is mm_before, "model_manager должен сохраняться при soft-reload"

    # Корректное завершение
    core.stop()
