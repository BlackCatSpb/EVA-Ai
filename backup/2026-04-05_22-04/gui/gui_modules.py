"""
Модуль для управления модулями графического интерфейса
"""
import logging
import tkinter as tk
from tkinter import ttk
from .widgets import create_rounded_button

logger = logging.getLogger("eva.gui.modules")

def init_modules(gui):
    """Инициализирует модули GUI."""
    gui.active_modules = []
    
    # Инициализируем модули с обработкой ошибок
    try:
        from .chat_module import ChatModule
        gui.chat_module = ChatModule(gui)
        gui.active_modules.append(gui.chat_module)
        logger.info("Модуль чата инициализирован")
    except Exception as e:
        logger.warning(f"Модуль чата недоступен: {e}")
        gui.chat_module = None
    
    try:
        from .analytics_module import AnalyticsModule
        gui.analytics_module = AnalyticsModule(gui)
        gui.active_modules.append(gui.analytics_module)
        logger.info("Модуль аналитики инициализирован")
    except Exception as e:
        logger.warning(f"Модуль аналитики недоступен: {e}")
        gui.analytics_module = None
    
    try:
        from .knowledge_graph_module import KnowledgeGraphModule
        gui.knowledge_module = KnowledgeGraphModule(gui)
        gui.active_modules.append(gui.knowledge_module)
        logger.info("Модуль графа знаний инициализирован")
    except Exception as e:
        logger.warning(f"Модуль графа знаний недоступен: {e}")
        gui.knowledge_module = None
    
    try:
        from .contradiction_module import ContradictionModule
        gui.contradiction_module = ContradictionModule(gui)
        gui.active_modules.append(gui.contradiction_module)
        logger.info("Модуль противоречий инициализирован")
    except Exception as e:
        logger.warning(f"Модуль противоречий недоступен: {e}")
        gui.contradiction_module = None
    
    try:
        from .memory_module import MemoryModule
        gui.memory_module = MemoryModule(gui)
        gui.active_modules.append(gui.memory_module)
        logger.info("Модуль памяти инициализирован")
    except Exception as e:
        logger.warning(f"Модуль памяти недоступен: {e}")
        gui.memory_module = None
    
    try:
        from .learning_module import LearningModule
        gui.learning_module = LearningModule(gui)
        gui.active_modules.append(gui.learning_module)
        logger.info("Модуль обучения инициализирован")
    except Exception as e:
        logger.warning(f"Модуль обучения недоступен: {e}")
        gui.learning_module = None
    
    try:
        from .settings_module import SettingsModule
        gui.settings_module = SettingsModule(gui)
        gui.active_modules.append(gui.settings_module)
        logger.info("Модуль настроек инициализирован")
    except Exception as e:
        logger.warning(f"Модуль настроек недоступен: {e}")
        gui.settings_module = None

    try:
        from .neuromorphic_module import NeuromorphicModule
        gui.neuromorphic_module = NeuromorphicModule(gui)
        gui.active_modules.append(gui.neuromorphic_module)
        logger.info("Модуль нейроморфики инициализирован")
    except Exception as e:
        logger.warning(f"Модуль нейроморфики недоступен: {e}")
        gui.neuromorphic_module = None
    
    logger.info(f"Модули GUI инициализированы: {len(gui.active_modules)} активных")

def switch_view(gui, view_id):
    """Переключает между различными представлениями."""
    modules = {
        "chat": gui.chat_module,
        "knowledge": gui.knowledge_module,
        "contradictions": gui.contradiction_module,
        "memory": gui.memory_module,
        "learning": gui.learning_module,
        "analytics": gui.analytics_module,
        "settings": gui.settings_module,
        "neuromorphic": gui.neuromorphic_module,
    }
    
    # Деактивируем текущий модуль
    if gui.current_view in modules and modules[gui.current_view]:
        try:
            modules[gui.current_view].deactivate()
        except Exception as e:
            logger.error(f"Ошибка деактивации модуля {gui.current_view}: {e}")
    
    # Очищаем область контента
    if hasattr(gui, 'content_area') and gui.content_area:
        for widget in gui.content_area.winfo_children():
            widget.destroy()
    
    # Активируем новый модуль
    gui.current_view = view_id
    module = modules.get(view_id)
    if module:
        try:
            module.activate()
            gui.system_status = "active"
        except Exception as e:
            logger.error(f"Ошибка активации модуля {view_id}: {e}")
            gui.system_status = "error"
            gui.show_notification(f"Ошибка загрузки модуля: {str(e)}", "error")
            # Возвращаемся к чату
            if view_id != "chat":
                switch_view(gui, "chat")
    else:
        # Создаем сообщение об ошибке
        error_frame = ttk.Frame(gui.content_area)
        error_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        error_label = ttk.Label(
            error_frame,
            text=f"Модуль '{view_id}' недоступен",
            font=("Arial", 12, "bold"),
            foreground=gui.colors["danger"]
        )
        error_label.pack(pady=10)
        
        retry_button = create_rounded_button(
            error_frame,
            "Попробовать снова",
            lambda: switch_view(gui, view_id),
            bg=gui.colors["primary"],
            fg="white"
        )
        retry_button.pack(pady=5)
        
        home_button = create_rounded_button(
            error_frame,
            "Вернуться в чат",
            lambda: switch_view(gui, "chat"),
            bg=gui.colors["secondary"],
            fg="white"
        )
        home_button.pack(pady=5)
    
    # Обновляем выделение кнопки навигации
    update_nav_button_selection(gui, view_id)
    logger.debug(f"Переключено на представление: {view_id}")

def update_nav_button_selection(gui, selected_view):
    """Обновляет визуальное выделение кнопки навигации."""
    if not hasattr(gui, 'nav_buttons'):
        return
        
    for view_id, button in gui.nav_buttons.items():
        if view_id == selected_view:
            button.configure(style='Nav.TButton')
        else:
            button.configure(style='')