"""
Модуль управления этическими принципами для CogniFlex
"""
import os
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import threading
import time

from .violation_id_manager import generate_violation_id
from .ethics_framework import EthicalPrinciple

logger = logging.getLogger("cogniflex.ethics.principles")

class PrinciplesManager:
    """Управляет этическими принципами и их состоянием."""
    
    def __init__(self, ethics_framework, cache_dir: Optional[str] = None):
        """
        Инициализирует менеджер этических принципов.
        
        Args:
            ethics_framework: Ссылка на основной модуль этической рамки
            cache_dir: Путь к директории кэша
        """
        self.ethics_framework = ethics_framework
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "ethics_cache", "principles")
        self.initialized = False
        
        # Создаем директорию кэша если её нет
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Пути к файлам
        self.principles_file = os.path.join(self.cache_dir, "principles.json")
        self.history_file = os.path.join(self.cache_dir, "principles_history.json")
        
        # Инициализация данных
        self.principles: Dict[str, EthicalPrinciple] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
        
        # Загружаем конфигурацию
        self._load_configuration()
        
        logger.info("Менеджер этических принципов инициализирован")
        self.initialized = True

    def _load_configuration(self):
        """Загружает конфигурацию менеджера принципов."""
        try:
            # Загружаем принципы
            if os.path.exists(self.principles_file):
                with open(self.principles_file, 'r', encoding='utf-8') as f:
                    principles_data = json.load(f)
                
                for name, data in principles_data.items():
                    self.principles[name] = EthicalPrinciple(
                        name=data["name"],
                        description=data["description"],
                        weight=data.get("weight", 1.0),
                        threshold=data.get("threshold", 0.8),
                        category=data.get("category", "general"),
                        priority=data.get("priority", 5)
                    )
                logger.info(f"Загружено {len(self.principles)} этических принципов в менеджер")
            
            # Загружаем историю
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Загружена история изменений принципов ({len(self.history)} записей)")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации менеджера принципов: {e}", exc_info=True)
            # Инициализируем стандартные принципы при ошибке
            self._init_default_principles()

    def _init_default_principles(self):
        """Инициализирует стандартные этические принципы."""
        default_principles = [
            EthicalPrinciple(
                name="Autonomy",
                description="Уважение к автономии и свободе выбора пользователя",
                weight=1.0,
                threshold=0.7,
                category="autonomy",
                priority=8
            ),
            EthicalPrinciple(
                name="Beneficence",
                description="Стремление к благу и пользе для пользователя",
                weight=1.2,
                threshold=0.75,
                category="beneficence",
                priority=9
            ),
            EthicalPrinciple(
                name="Non-maleficence",
                description="Отсутствие вреда пользователю и обществу",
                weight=1.5,
                threshold=0.6,
                category="safety",
                priority=10
            ),
            EthicalPrinciple(
                name="Justice",
                description="Справедливое и беспристрастное отношение",
                weight=0.9,
                threshold=0.65,
                category="fairness",
                priority=7
            ),
            EthicalPrinciple(
                name="Privacy",
                description="Защита личной информации и конфиденциальности",
                weight=1.3,
                threshold=0.6,
                category="privacy",
                priority=9
            ),
            EthicalPrinciple(
                name="Transparency",
                description="Ясность и открытость в работе системы",
                weight=0.8,
                threshold=0.7,
                category="transparency",
                priority=6
            ),
            EthicalPrinciple(
                name="Accountability",
                description="Ответственность за действия и решения",
                weight=1.1,
                threshold=0.7,
                category="accountability",
                priority=8
            )
        ]
        
        for principle in default_principles:
            self.principles[principle.name] = principle
        
        # Сохраняем
        self._save_principles()
        logger.info("Инициализированы стандартные этические принципы в менеджере")

    def _save_principles(self):
        """Сохраняет принципы в файл."""
        try:
            principles_data = {}
            for name, principle in self.principles.items():
                principles_data[name] = {
                    "name": principle.name,
                    "description": principle.description,
                    "weight": principle.weight,
                    "threshold": principle.threshold,
                    "category": principle.category,
                    "priority": principle.priority
                }
            
            with open(self.principles_file, 'w', encoding='utf-8') as f:
                json.dump(principles_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения принципов: {e}", exc_info=True)

    def _save_history(self):
        """Сохраняет историю изменений в файл."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения истории изменений принципов: {e}", exc_info=True)

    def add_principle(self, principle: EthicalPrinciple) -> bool:
        """
        Добавляет новый этический принцип.
        
        Args:
            principle: Новый принцип
            
        Returns:
            bool: Успешно ли добавлено
        """
        with self.lock:
            if principle.name in self.principles:
                logger.warning(f"Принцип {principle.name} уже существует в менеджере")
                return False
            
            # Добавляем в менеджер
            self.principles[principle.name] = principle
            
            # Добавляем в основную рамку
            self.ethics_framework.principles[principle.name] = principle
            
            # Сохраняем историю
            self.history.append({
                "action": "add",
                "principle": principle.name,
                "data": principle.__dict__,
                "timestamp": time.time(),
                "violation_id": generate_violation_id("System")
            })
            
            # Сохраняем изменения
            self._save_principles()
            self._save_history()
            
            logger.info(f"Добавлен новый этический принцип через менеджер: {principle.name}")
            return True

    def update_principle(self, name: str, **kwargs) -> bool:
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
                logger.warning(f"Принцип {name} не найден в менеджере")
                return False
            
            principle = self.principles[name]
            old_data = principle.__dict__.copy()
            
            # Обновляем параметры
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
            
            # Обновляем в основной рамке
            if name in self.ethics_framework.principles:
                self.ethics_framework.principles[name] = principle
            
            # Сохраняем историю
            self.history.append({
                "action": "update",
                "principle": name,
                "old_data": old_data,
                "new_data": principle.__dict__,
                "timestamp": time.time(),
                "violation_id": generate_violation_id("System")
            })
            
            # Сохраняем изменения
            self._save_principles()
            self._save_history()
            
            logger.info(f"Обновлен этический принцип через менеджер: {name}")
            return True

    def remove_principle(self, name: str) -> bool:
        """
        Удаляет этический принцип.
        
        Args:
            name: Имя принципа
            
        Returns:
            bool: Успешно ли удалено
        """
        with self.lock:
            if name not in self.principles:
                logger.warning(f"Принцип {name} не найден в менеджере")
                return False
            
            principle = self.principles[name]
            
            # Удаляем из менеджера
            del self.principles[name]
            
            # Удаляем из основной рамки
            if name in self.ethics_framework.principles:
                del self.ethics_framework.principles[name]
            
            # Сохраняем историю
            self.history.append({
                "action": "remove",
                "principle": name,
                "data": principle.__dict__,
                "timestamp": time.time(),
                "violation_id": generate_violation_id("System")
            })
            
            # Сохраняем изменения
            self._save_principles()
            self._save_history()
            
            logger.info(f"Удален этический принцип через менеджер: {name}")
            return True

    def get_principle(self, name: str) -> Optional[EthicalPrinciple]:
        """Возвращает принцип по имени."""
        return self.principles.get(name)

    def get_all_principles(self) -> Dict[str, EthicalPrinciple]:
        """Возвращает все принципы."""
        return self.principles.copy()

    def get_principle_history(self, name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает историю изменений для конкретного принципа."""
        principle_history = [
            entry for entry in self.history 
            if entry["principle"] == name
        ]
        return principle_history[-limit:]

    def get_compliance_score(self) -> float:
        """
        Возвращает общий уровень соблюдения этических принципов.
        
        Returns:
            float: Уровень соблюдения (0.0-1.0)
        """
        if not self.ethics_framework.violations:
            return 1.0
        
        # Анализируем последние 30 дней
        now = time.time()
        thirty_days_ago = now - 30 * 24 * 3600
        recent_violations = [
            v for v in self.ethics_framework.violations 
            if v.timestamp >= thirty_days_ago
        ]
        
        if not recent_violations:
            return 1.0
        
        # Вычисляем средний уровень серьезности
        total_severity = sum(v.severity for v in recent_violations)
        avg_severity = total_severity / len(recent_violations)
        
        # Нормализуем в диапазон 0.0-1.0 (1.0 = нет нарушений)
        return max(0.0, 1.0 - avg_severity)

    def generate_ethical_visualization(self, view_type: str = "compliance") -> str:
        """
        Генерирует визуализацию данных этических принципов.
        
        Args:
            view_type: Тип визуализации
            
        Returns:
            str: Изображение в формате base64
        """
        try:
            # Здесь будет код для генерации визуализации
            # Временная заглушка
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64
            
            # Создаем график
            plt.figure(figsize=(10, 6))
            
            if view_type == "compliance":
                # График соблюдения принципов
                principles = list(self.principles.keys())
                compliance = [1.0] * len(principles)  # В реальной системе здесь будут реальные данные
                
                plt.bar(principles, compliance)
                plt.title('Соблюдение этических принципов')
                plt.xlabel('Принципы')
                plt.ylabel('Уровень соблюдения')
                plt.xticks(rotation=45)
            
            elif view_type == "violations":
                # График нарушений
                principles = list(self.principles.keys())
                violations = [0] * len(principles)
                
                for violation in self.ethics_framework.violations:
                    if violation.principle in principles:
                        idx = principles.index(violation.principle)
                        violations[idx] += 1
                
                plt.bar(principles, violations)
                plt.title('Количество нарушений по принципам')
                plt.xlabel('Принципы')
                plt.ylabel('Количество нарушений')
                plt.xticks(rotation=45)
            
            else:
                # График временных тенденций
                plt.plot([0.9, 0.85, 0.8, 0.75, 0.8, 0.85, 0.9])
                plt.title('Динамика соблюдения этических стандартов')
                plt.xlabel('Время')
                plt.ylabel('Уровень соблюдения')
            
            plt.tight_layout()
            
            # Сохраняем в буфер
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{img_data}"
        
        except Exception as e:
            logger.error(f"Ошибка генерации визуализации этических принципов: {e}")
            return ""

    def close(self):
        """Закрывает менеджер принципов и освобождает ресурсы."""
        logger.info("Закрытие менеджера этических принципов...")
        self._save_principles()
        self._save_history()
        logger.info("Менеджер этических принципов закрыт")