import logging

logger = logging.getLogger(__name__)


def _init_modules(self):
    """Инициализирует модули GUI с безопасной загрузкой."""

    if not hasattr(self, 'content_area') or self.content_area is None:
        logger.error("Попытка инициализации модулей до создания интерфейса")
        return

    self.active_modules = []

    # Список модулей в формате: (атрибут, модуль, класс)
    modules = [
        ("chat_module", "chat_module", "ChatModule"),
        ("analytics_module", "analytics_module", "AnalyticsModule"),
        ("knowledge_module", "knowledge_graph_module", "KnowledgeGraphModule"),
        ("contradiction_module", "contradiction_module", "ContradictionModule"),
        ("memory_module", "memory_module", "MemoryModule"),
        ("learning_module", "learning_module", "LearningModule"),
        ("settings_module", "settings_module", "SettingsModule"),
    ]

    for attr_name, module_name, class_name in modules:
        try:
            # Относительный импорт из текущей папки (где лежит core_gui.py)
            mod = __import__(f".{module_name}", fromlist=[class_name], package=__package__)
            cls = getattr(mod, class_name)
            instance = cls(self)
            setattr(self, attr_name, instance)
            self.active_modules.append(instance)
            logger.info(f"{class_name} инициализирован")
        except ImportError as e:
            logger.warning(f"{class_name} недоступен (ImportError): {e}")
            setattr(self, attr_name, None)
        except Exception as e:
            logger.error(f"Ошибка инициализации {class_name}: {e}", exc_info=True)
            setattr(self, attr_name, None)

    # После загрузки переключаем на чат, если он есть
    if self.chat_module:
        self._switch_view("chat")
    else:
        logger.warning("Чат-модуль не загружен, невозможно переключиться на чат")
