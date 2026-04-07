"""
Модуль принципов этической рамки для ЕВА - определение, загрузка и управление принципами
"""
import os
import logging
import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.ethics")

@dataclass
class EthicalPrinciple:
    """Представляет этический принцип."""
    name: str
    description: str
    phrase: str = ""
    weight: float = 1.0
    threshold: float = 0.8
    category: str = "general"
    priority: int = 5
    last_updated: float = field(default_factory=time.time)
    active: bool = True


class EthicsPrinciplesMixin:
    """Миксин для управления этическими принципами."""

    def _load_configuration(self):
        """Загружает конфигурацию этической рамки из файла."""
        try:
            config_path = os.path.join(self.cache_dir, "ethics_config.json")

            if not os.path.exists(config_path):
                logger.info(f"Конфигурационный файл не найден, создаем по умолчанию: {config_path}")
                self._init_default_configuration()
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            config_version = config_data.get("version", "1.0")

            if config_version == "1.0":
                self._load_configuration_v1(config_data)
            elif config_version == "2.0":
                self._load_configuration_v2(config_data)
            else:
                logger.warning(f"Неизвестная версия конфигурации: {config_version}")
                self._init_default_configuration()

        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации этической рамки: {e}", exc_info=True)
            self._init_default_configuration()

    def _load_configuration_v1(self, config_data: Dict[str, Any]):
        """Загружает конфигурацию версии 1.0 (список принципов)."""
        if "principles" not in config_data:
            logger.warning("Конфигурация не содержит раздела principles")
            self._init_default_principles()
            return

        principles_data = config_data["principles"]

        if isinstance(principles_data, list):
            logger.info("Конфигурация principles представлена в виде списка, преобразуем в словарь")
            principles_dict = {}
            for item in principles_data:
                if isinstance(item, dict) and "name" in item:
                    principles_dict[item["name"]] = item
                elif isinstance(item, dict) and "id" in item:
                    principles_dict[item["id"]] = item
                else:
                    logger.warning(f"Пропущен некорректный элемент принципа: {item}")
            principles_data = principles_dict

        if not isinstance(principles_data, dict):
            logger.error("Данные принципов не являются ни списком, ни словарем")
            self._init_default_principles()
            return

        for name, data in principles_data.items():
            if not isinstance(data, dict):
                logger.warning(f"Пропущен принцип '{name}' с некорректными данными")
                continue

            principle = EthicalPrinciple(
                name=name,
                description=data.get("description", ""),
                weight=data.get("weight", 1.0),
                threshold=data.get("threshold", 0.8),
                category=data.get("category", "general"),
                priority=data.get("priority", 5)
            )
            self.principles[name] = principle

        logger.info(f"Загружено {len(self.principles)} этических принципов (v1.0)")
        self._load_violations_and_stats()

    def _load_configuration_v2(self, config_data: Dict[str, Any]):
        """Загружает конфигурацию версии 2.0 (словарь принципов)."""
        if "principles" not in config_data:
            logger.warning("Конфигурация не содержит раздела principles")
            self._init_default_principles()
            return

        principles_data = config_data["principles"]

        if not isinstance(principles_data, dict):
            logger.error("В версии 2.0 принципы должны быть словарем")
            self._init_default_principles()
            return

        for name, data in principles_data.items():
            if not isinstance(data, dict):
                continue

            principle = EthicalPrinciple(
                name=name,
                description=data.get("description", ""),
                weight=data.get("weight", 1.0),
                threshold=data.get("threshold", 0.8),
                category=data.get("category", "general"),
                priority=data.get("priority", 5)
            )
            self.principles[name] = principle

        logger.info(f"Загружено {len(self.principles)} этических принципов (v2.0)")
        self._load_violations_and_stats()

    def _init_default_configuration(self):
        """Инициализирует конфигурацию по умолчанию."""
        self._init_default_principles()

    def _init_default_principles(self):
        """Инициализирует принципы по умолчанию (согласно brain_config.json)."""
        default_principles = {
            "no_violence": EthicalPrinciple(
                name="no_violence",
                phrase="Без насилия",
                description="Нет насилию, оружию, наркотикам, травле",
                weight=1.5,
                threshold=0.6,
                category="safety",
                priority=10
            ),
            "honesty": EthicalPrinciple(
                name="honesty",
                phrase="Честность",
                description="Если не уверена — скажи, спроси",
                weight=1.2,
                threshold=0.7,
                category="integrity",
                priority=9
            ),
            "fact_verification": EthicalPrinciple(
                name="fact_verification",
                phrase="Проверка фактов",
                description="Проверяй факты, исправляй ошибки",
                weight=1.2,
                threshold=0.7,
                category="accuracy",
                priority=9
            ),
            "safe_code": EthicalPrinciple(
                name="safe_code",
                phrase="Безопасный код",
                description="Нет вредоносному коду",
                weight=1.5,
                threshold=0.6,
                category="security",
                priority=10
            ),
            "risk_blocking": EthicalPrinciple(
                name="risk_blocking",
                phrase="Блокировка рисков",
                description="Блокируй высокорисковые запросы",
                weight=1.3,
                threshold=0.65,
                category="safety",
                priority=10
            ),
            "output_control": EthicalPrinciple(
                name="output_control",
                phrase="Контроль вывода",
                description="Проверяй ответ перед отправкой",
                weight=1.0,
                threshold=0.75,
                category="quality",
                priority=8
            )
        }
        self.principles.update(default_principles)
        logger.info(f"Инициализировано {len(default_principles)} принципов по умолчанию")

    def add_ethical_principle(self, principle: EthicalPrinciple) -> bool:
        """
        Добавляет новый этический принцип.

        Args:
            principle: Новый принцип

        Returns:
            bool: Успешно ли добавлено
        """
        with self.lock:
            if principle.name in self.principles:
                logger.warning(f"Принцип {principle.name} уже существует")
                return False

            self.principles[principle.name] = principle
            self._save_principles()
            logger.info(f"Добавлен новый этический принцип: {principle.name}")
            return True

    def update_ethical_principle(self, name: str, **kwargs) -> bool:
        """
        Обновляет существующий этический принцип.

        Args:
            name: Имя принципа
            **kwargs: Параметры для обновления

        Returns:
            bool: Успешно ли обновлено
        """
        with self.lock:
            if name not in self.principles:
                logger.warning(f"Принцип {name} не найден")
                return False

            principle = self.principles[name]

            if "description" in kwargs:
                principle.description = kwargs["description"]
            if "weight" in kwargs:
                principle.weight = kwargs["weight"]
            if "threshold" in kwargs:
                principle.threshold = kwargs["threshold"]
            if "category" in kwargs:
                principle.category = kwargs["category"]
            if "priority" in kwargs:
                principle.priority = kwargs["priority"]

            self._save_principles()
            logger.info(f"Обновлен этический принцип: {name}")
            return True

    def get_principle(self, name: str) -> Optional[EthicalPrinciple]:
        """Возвращает принцип по имени."""
        return self.principles.get(name)

    def get_all_principles(self) -> Dict[str, EthicalPrinciple]:
        """Возвращает все принципы."""
        return self.principles.copy()
