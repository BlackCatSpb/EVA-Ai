#!/usr/bin/env python3
"""
Продолжение интеграции модулей CogniFlex
Интеграция contradiction и ethics модулей
"""

import os
import sys
import shutil
from datetime import datetime

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_integrated_contradiction():
    """Создает интегрированную версию модуля противоречий"""
    print("🔧 Создание интегрированного модуля противоречий...")
    
    template = '''"""
Интегрированный менеджер противоречий CogniFlex
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("cogniflex.contradiction")

from cogniflex.core.base_component import BaseComponent, ComponentState
from cogniflex.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальные менеджеры
try:
    from cogniflex.contradiction.contradiction_manager import ContradictionManager
    ORIGINAL_MANAGER_AVAILABLE = True
except ImportError:
    ORIGINAL_MANAGER_AVAILABLE = False
    logger.warning("Оригинальный ContradictionManager недоступен")

try:
    from cogniflex.contradiction.contradiction_resolver import ContradictionResolver
    ORIGINAL_RESOLVER_AVAILABLE = True
except ImportError:
    ORIGINAL_RESOLVER_AVAILABLE = False
    logger.warning("Оригинальный ContradictionResolver недоступен")


class IntegratedContradictionManager(BaseComponent):
    """Интегрированный менеджер противоречий с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("contradiction_manager", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'contradiction_cache')
        
        # Инициализируем оригинальные компоненты если доступны
        self._original_manager = None
        self._original_resolver = None
        
        if ORIGINAL_MANAGER_AVAILABLE:
            try:
                self._original_manager = ContradictionManager(brain, cache_dir)
                logger.info("Оригинальный ContradictionManager инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального менеджера: {e}")
        
        if ORIGINAL_RESOLVER_AVAILABLE:
            try:
                self._original_resolver = ContradictionResolver(brain, cache_dir)
                logger.info("Оригинальный ContradictionResolver инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального резолвера: {e}")
        
        # Статистика
        self.stats = {
            "contradictions_detected": 0,
            "contradictions_resolved": 0,
            "checks_performed": 0,
            "errors": 0
        }
        
        # База данных противоречий
        self.contradictions_db = []
        
        logger.info(f"IntegratedContradictionManager {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация менеджера противоречий...")
            
            # Инициализируем оригинальные компоненты
            if self._original_manager and hasattr(self._original_manager, 'initialize'):
                self._original_manager.initialize()
            
            if self._original_resolver and hasattr(self._original_resolver, 'initialize'):
                self._original_resolver.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Загружаем базу данных противоречий
            self._load_contradictions_db()
            
            # Публикуем событие инициализации
            self._emit_event("contradiction_manager.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir,
                'contradictions_count': len(self.contradictions_db)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации менеджера противоречий: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск менеджера противоречий...")
            
            # Запускаем оригинальные компоненты
            if self._original_manager and hasattr(self._original_manager, 'start'):
                self._original_manager.start()
            
            if self._original_resolver and hasattr(self._original_resolver, 'start'):
                self._original_resolver.start()
            
            # Публикуем событие запуска
            self._emit_event("contradiction_manager.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска менеджера противоречий: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка менеджера противоречий...")
            
            # Останавливаем оригинальные компоненты
            if self._original_manager and hasattr(self._original_manager, 'stop'):
                self._original_manager.stop()
            
            if self._original_resolver and hasattr(self._original_resolver, 'stop'):
                self._original_resolver.stop()
            
            # Сохраняем базу данных противоречий
            self._save_contradictions_db()
            
            # Публикуем событие остановки
            self._emit_event("contradiction_manager.stopped", {
                'component': self.name,
                'stats': self.stats,
                'contradictions_count': len(self.contradictions_db)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки менеджера противоречий: {e}")
            return False
    
    def detect_contradiction(self, statement1: str, statement2: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обнаруживает противоречия между двумя утверждениями"""
        start_time = time.time()
        
        try:
            # Используем оригинальный менеджер если доступен
            if self._original_manager and hasattr(self._original_manager, 'detect_contradiction'):
                result = self._original_manager.detect_contradiction(statement1, statement2, context)
            else:
                # Базовое обнаружение противоречий
                result = self._basic_contradiction_detection(statement1, statement2, context)
            
            # Обновляем статистику
            self.stats["checks_performed"] += 1
            if result.get("is_contradiction", False):
                self.stats["contradictions_detected"] += 1
                
                # Сохраняем в базу данных
                contradiction_entry = {
                    "id": len(self.contradictions_db) + 1,
                    "statement1": statement1,
                    "statement2": statement2,
                    "context": context or {},
                    "detection_time": datetime.now().isoformat(),
                    "confidence": result.get("confidence", 0.0),
                    "contradiction_type": result.get("type", "unknown")
                }
                self.contradictions_db.append(contradiction_entry)
            
            # Публикуем событие обнаружения
            self._emit_event("contradiction_manager.contradiction_detected", {
                'statement1_length': len(statement1),
                'statement2_length': len(statement2),
                'is_contradiction': result.get("is_contradiction", False),
                'confidence': result.get("confidence", 0.0),
                'processing_time': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения противоречия: {e}")
            self.stats["errors"] += 1
            return {"is_contradiction": False, "error": str(e)}
    
    def _basic_contradiction_detection(self, statement1: str, statement2: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Базовое обнаружение противоречий"""
        # Простая эвристика для обнаружения противоречий
        statement1_lower = statement1.lower()
        statement2_lower = statement2.lower()
        
        # Проверяем на антонимы
        antonym_pairs = [
            ("да", "нет"), ("истинно", "ложно"), ("правильно", "неправильно"),
            ("включен", "выключен"), ("активен", "неактивен"), ("работает", "не работает")
        ]
        
        for word1, word2 in antonym_pairs:
            if word1 in statement1_lower and word2 in statement2_lower:
                return {
                    "is_contradiction": True,
                    "confidence": 0.8,
                    "type": "antonym",
                    "details": f"Обнаружены антонимы: {word1}/{word2}"
                }
        
        # Проверяем на отрицание
        negation_words = ["не", "нет", "никогда", "никак", "ни"]
        has_negation1 = any(neg in statement1_lower for neg in negation_words)
        has_negation2 = any(neg in statement2_lower for neg in negation_words)
        
        # Если одно утверждение содержит отрицание, а другое нет, и они похожи
        if has_negation1 != has_negation2:
            # Упрощенная проверка схожести (убираем отрицания)
            clean1 = statement1_lower
            clean2 = statement2_lower
            
            for neg in negation_words:
                clean1 = clean1.replace(neg, "").strip()
                clean2 = clean2.replace(neg, "").strip()
            
            # Если после очистки утверждения похожи
            if clean1 == clean2 or abs(len(clean1) - len(clean2)) < 5:
                return {
                    "is_contradiction": True,
                    "confidence": 0.7,
                    "type": "negation",
                    "details": "Обнаружено противоречие через отрицание"
                }
        
        return {
            "is_contradiction": False,
            "confidence": 0.0,
            "type": "none",
            "details": "Противоречие не обнаружено"
        }
    
    def resolve_contradiction(self, contradiction_id: int, resolution_strategy: str = "auto") -> Dict[str, Any]:
        """Разрешает противоречие"""
        try:
            # Находим противоречие в базе данных
            contradiction = None
            for entry in self.contradictions_db:
                if entry["id"] == contradiction_id:
                    contradiction = entry
                    break
            
            if not contradiction:
                return {"success": False, "error": "Противоречие не найдено"}
            
            # Используем оригинальный резолвер если доступен
            if self._original_resolver and hasattr(self._original_resolver, 'resolve_contradiction'):
                result = self._original_resolver.resolve_contradiction(
                    contradiction["statement1"],
                    contradiction["statement2"],
                    contradiction["context"],
                    resolution_strategy
                )
            else:
                # Базовое разрешение противоречия
                result = self._basic_contradiction_resolution(contradiction, resolution_strategy)
            
            if result.get("success", False):
                # Обновляем статистику
                self.stats["contradictions_resolved"] += 1
                
                # Помечаем противоречие как разрешенное
                contradiction["resolved"] = True
                contradiction["resolution"] = result
                contradiction["resolution_time"] = datetime.now().isoformat()
                
                # Публикуем событие разрешения
                self._emit_event("contradiction_manager.contradiction_resolved", {
                    'contradiction_id': contradiction_id,
                    'resolution_strategy': resolution_strategy,
                    'success': True
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречия: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}
    
    def _basic_contradiction_resolution(self, contradiction: Dict, strategy: str) -> Dict[str, Any]:
        """Базовое разрешение противоречия"""
        if strategy == "prefer_first":
            return {
                "success": True,
                "resolution": contradiction["statement1"],
                "strategy": strategy,
                "reason": "Выбрано первое утверждение"
            }
        elif strategy == "prefer_second":
            return {
                "success": True,
                "resolution": contradiction["statement2"],
                "strategy": strategy,
                "reason": "Выбрано второе утверждение"
            }
        elif strategy == "merge":
            # Попытка объединить утверждения
            return {
                "success": True,
                "resolution": f"Компромисс: {contradiction['statement1']} и {contradiction['statement2']}",
                "strategy": strategy,
                "reason": "Утверждения объединены"
            }
        else:
            return {
                "success": False,
                "error": f"Неизвестная стратегия разрешения: {strategy}"
            }
    
    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику противоречий"""
        stats = self.stats.copy()
        
        # Добавляем детальную статистику
        stats.update({
            "total_contradictions": len(self.contradictions_db),
            "resolved_contradictions": sum(1 for c in self.contradictions_db if c.get("resolved", False)),
            "unresolved_contradictions": sum(1 for c in self.contradictions_db if not c.get("resolved", False)),
            "contradiction_types": list(set(c.get("contradiction_type", "unknown") for c in self.contradictions_db))
        })
        
        # Добавляем статистику из оригинальных компонентов
        if self._original_manager and hasattr(self._original_manager, 'get_statistics'):
            original_stats = self._original_manager.get_statistics()
            stats.update(original_stats)
        
        return stats
    
    def _load_contradictions_db(self):
        """Загружает базу данных противоречий"""
        try:
            db_file = os.path.join(self.cache_dir, 'contradictions_db.json')
            if os.path.exists(db_file):
                import json
                with open(db_file, 'r', encoding='utf-8') as f:
                    self.contradictions_db = json.load(f)
                logger.info(f"Загружено {len(self.contradictions_db)} противоречий")
        except Exception as e:
            logger.error(f"Ошибка загрузки базы данных противоречий: {e}")
            self.contradictions_db = []
    
    def _save_contradictions_db(self):
        """Сохраняет базу данных противоречий"""
        try:
            db_file = os.path.join(self.cache_dir, 'contradictions_db.json')
            import json
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump(self.contradictions_db, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено {len(self.contradictions_db)} противоречий")
        except Exception as e:
            logger.error(f"Ошибка сохранения базы данных противоречий: {e}")
'''
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "contradiction", 
        "contradiction_integrated.py"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"   ✅ Создан файл: {output_path}")
    return output_path

def create_integrated_ethics():
    """Создает интегрированную версию этического модуля"""
    print("🔧 Создание интегрированного этического модуля...")
    
    template = '''"""
Интегрированный этический фреймворк CogniFlex
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("cogniflex.ethics")

from cogniflex.core.base_component import BaseComponent, ComponentState
from cogniflex.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный этический фреймворк
try:
    from cogniflex.ethics.ethics_core import EthicsFramework
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный EthicsFramework недоступен")


class IntegratedEthicsFramework(BaseComponent):
    """Интегрированный этический фреймворк с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("ethics_framework", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'ethics_cache')
        
        # Инициализируем оригинальный фреймворк если доступен
        self._original_framework = None
        if ORIGINAL_AVAILABLE:
            try:
                self._original_framework = EthicsFramework()
                logger.info("Оригинальный EthicsFramework инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального фреймворка: {e}")
        
        # Этические принципы и правила
        self.ethical_principles = {
            "beneficence": "Благодеяние - действовать во благо",
            "non_maleficence": "Не навреди - избегать вреда",
            "autonomy": "Автономия - уважать выбор и решения",
            "justice": "Справедливость - обеспечивать равное отношение",
            "privacy": "Конфиденциальность - защищать личные данные",
            "transparency": "Прозрачность - быть открытым и честным"
        }
        
        # Статистика
        self.stats = {
            "ethical_checks": 0,
            "violations_detected": 0,
            "warnings_issued": 0,
            "approvals_granted": 0,
            "errors": 0
        }
        
        # История этических оценок
        self.ethical_history = []
        
        logger.info(f"IntegratedEthicsFramework {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация этического фреймворка...")
            
            # Инициализируем оригинальный фреймворк
            if self._original_framework and hasattr(self._original_framework, 'initialize'):
                self._original_framework.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Загружаем историю этических оценок
            self._load_ethical_history()
            
            # Публикуем событие инициализации
            self._emit_event("ethics_framework.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir,
                'principles_count': len(self.ethical_principles)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации этического фреймворка: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск этического фреймворка...")
            
            # Запускаем оригинальный фреймворк
            if self._original_framework and hasattr(self._original_framework, 'start'):
                self._original_framework.start()
            
            # Публикуем событие запуска
            self._emit_event("ethics_framework.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска этического фреймворка: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка этического фреймворка...")
            
            # Останавливаем оригинальный фреймворк
            if self._original_framework and hasattr(self._original_framework, 'stop'):
                self._original_framework.stop()
            
            # Сохраняем историю этических оценок
            self._save_ethical_history()
            
            # Публикуем событие остановки
            self._emit_event("ethics_framework.stopped", {
                'component': self.name,
                'stats': self.stats,
                'history_count': len(self.ethical_history)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки этического фреймворка: {e}")
            return False
    
    def evaluate_action(self, action: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Оценивает этичность действия"""
        start_time = time.time()
        
        try:
            # Используем оригинальный фреймворк если доступен
            if self._original_framework and hasattr(self._original_framework, 'evaluate_action'):
                result = self._original_framework.evaluate_action(action, context)
            else:
                # Базовая этическая оценка
                result = self._basic_ethical_evaluation(action, context)
            
            # Обновляем статистику
            self.stats["ethical_checks"] += 1
            
            if result.get("is_violation", False):
                self.stats["violations_detected"] += 1
            elif result.get("warning", False):
                self.stats["warnings_issued"] += 1
            else:
                self.stats["approvals_granted"] += 1
            
            # Сохраняем в историю
            evaluation_entry = {
                "id": len(self.ethical_history) + 1,
                "action": action,
                "context": context or {},
                "evaluation_time": datetime.now().isoformat(),
                "result": result,
                "processing_time": time.time() - start_time
            }
            self.ethical_history.append(evaluation_entry)
            
            # Публикуем событие оценки
            self._emit_event("ethics_framework.action_evaluated", {
                'action_length': len(action),
                'is_violation': result.get("is_violation", False),
                'warning': result.get("warning", False),
                'score': result.get("ethical_score", 0.0),
                'processing_time': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка этической оценки: {e}")
            self.stats["errors"] += 1
            return {"is_violation": False, "error": str(e), "ethical_score": 0.0}
    
    def _basic_ethical_evaluation(self, action: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Базовая этическая оценка"""
        action_lower = action.lower()
        
        # Список потенциально проблемных слов и фраз
        problematic_keywords = [
            ("вред", 0.8), ("убить", 0.9), ("нарушить", 0.6),
            ("взломать", 0.8), ("украсть", 0.9), ("обмануть", 0.7),
            ("дискриминировать", 0.8), ("оскорбить", 0.6),
            ("шантажировать", 0.9), ("мошенничество", 0.8)
        ]
        
        # Положительные ключевые слова
        positive_keywords = [
            ("помочь", -0.3), ("защитить", -0.2), ("поддержать", -0.2),
            ("спасти", -0.4), ("лечить", -0.3), ("обучить", -0.1),
            ("советовать", -0.1), ("предупредить", -0.2)
        ]
        
        ethical_score = 0.0
        violations = []
        warnings = []
        
        # Проверяем проблемные ключевые слова
        for keyword, penalty in problematic_keywords:
            if keyword in action_lower:
                ethical_score += penalty
                if penalty >= 0.8:
                    violations.append(f"Обнаружена потенциально вредная активность: {keyword}")
                else:
                    warnings.append(f"Требуется внимание: {keyword}")
        
        # Проверяем положительные ключевые слова
        for keyword, bonus in positive_keywords:
            if keyword in action_lower:
                ethical_score += bonus
        
        # Нормализуем оценку
        ethical_score = max(-1.0, min(1.0, ethical_score))
        
        # Определяем результат
        is_violation = ethical_score > 0.7
        warning = 0.3 < ethical_score <= 0.7
        
        # Генерируем рекомендации
        recommendations = []
        if is_violation:
            recommendations.append("Действие не рекомендуется из-за этических соображений")
        elif warning:
            recommendations.append("Требуется дополнительная оценка этических последствий")
        elif ethical_score < -0.3:
            recommendations.append("Действие соответствует этическим принципам")
        
        return {
            "ethical_score": ethical_score,
            "is_violation": is_violation,
            "warning": warning,
            "violations": violations,
            "warnings": warnings,
            "recommendations": recommendations,
            "principles_violated": self._check_principles_violations(action_lower),
            "confidence": min(0.9, abs(ethical_score) + 0.1)
        }
    
    def _check_principles_violations(self, action_lower: str) -> List[str]:
        """Проверяет нарушение этических принципов"""
        violations = []
        
        principle_keywords = {
            "non_maleficence": ["вред", "повредить", "навредить", "ущерб"],
            "privacy": ["личные данные", "конфиденциальность", "тайна", "приватность"],
            "autonomy": ["заставить", "принудить", "нарушить выбор", "ограничить"],
            "justice": ["дискриминировать", "неравенство", "несправедливость"],
            "transparency": ["скрыть", "обмануть", "ввести в заблуждение"]
        }
        
        for principle, keywords in principle_keywords.items():
            for keyword in keywords:
                if keyword in action_lower:
                    violations.append(principle)
                    break
        
        return violations
    
    def get_ethical_guidance(self, topic: str) -> Dict[str, Any]:
        """Предоставляет этические рекомендации по теме"""
        try:
            # Используем оригинальный фреймворк если доступен
            if self._original_framework and hasattr(self._original_framework, 'get_ethical_guidance'):
                return self._original_framework.get_ethical_guidance(topic)
            else:
                # Базовые рекомендации
                guidance = {
                    "topic": topic,
                    "principles": [],
                    "recommendations": [],
                    "warnings": []
                }
                
                # Добавляем релевантные принципы
                topic_lower = topic.lower()
                if "данные" in topic_lower or "информация" in topic_lower:
                    guidance["principles"].append("privacy")
                    guidance["recommendations"].append("Защищайте личные данные пользователей")
                
                if "решение" in topic_lower or "выбор" in topic_lower:
                    guidance["principles"].append("autonomy")
                    guidance["recommendations"].append("Уважайте автономию и свободу выбора")
                
                if "воздействие" in topic_lower or "последствия" in topic_lower:
                    guidance["principles"].append("non_maleficence")
                    guidance["recommendations"].append("Избегайте причинения вреда")
                
                return guidance
                
        except Exception as e:
            logger.error(f"Ошибка получения этических рекомендаций: {e}")
            return {"error": str(e)}
    
    def get_ethical_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику этических оценок"""
        stats = self.stats.copy()
        
        # Добавляем детальную статистику
        stats.update({
            "total_evaluations": len(self.ethical_history),
            "recent_violations": sum(1 for h in self.ethical_history[-10:] if h["result"].get("is_violation", False)),
            "average_processing_time": sum(h["processing_time"] for h in self.ethical_history) / max(1, len(self.ethical_history)),
            "most_common_violations": self._get_most_common_violations()
        })
        
        # Добавляем статистику из оригинального фреймворка
        if self._original_framework and hasattr(self._original_framework, 'get_statistics'):
            original_stats = self._original_framework.get_statistics()
            stats.update(original_stats)
        
        return stats
    
    def _get_most_common_violations(self) -> List[str]:
        """Возвращает наиболее частые нарушения"""
        violations_count = {}
        for entry in self.ethical_history:
            for violation in entry["result"].get("violations", []):
                violations_count[violation] = violations_count.get(violation, 0) + 1
        
        # Сортируем по частоте
        sorted_violations = sorted(violations_count.items(), key=lambda x: x[1], reverse=True)
        return [v[0] for v in sorted_violations[:5]]
    
    def _load_ethical_history(self):
        """Загружает историю этических оценок"""
        try:
            history_file = os.path.join(self.cache_dir, 'ethical_history.json')
            if os.path.exists(history_file):
                import json
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.ethical_history = json.load(f)
                logger.info(f"Загружено {len(self.ethical_history)} этических оценок")
        except Exception as e:
            logger.error(f"Ошибка загрузки истории этических оценок: {e}")
            self.ethical_history = []
    
    def _save_ethical_history(self):
        """Сохраняет историю этических оценок"""
        try:
            history_file = os.path.join(self.cache_dir, 'ethical_history.json')
            import json
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.ethical_history, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено {len(self.ethical_history)} этических оценок")
        except Exception as e:
            logger.error(f"Ошибка сохранения истории этических оценок: {e}")
'''
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "ethics", 
        "ethics_integrated.py"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"   ✅ Создан файл: {output_path}")
    return output_path

def update_component_initializer_with_new_modules():
    """Обновляет ComponentInitializer с новыми модулями"""
    print("🔧 Обновление ComponentInitializer с новыми модулями...")
    
    # Читаем текущий файл
    initializer_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "core", 
        "component_initializer.py"
    )
    
    try:
        with open(initializer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Добавляем импорты новых модулей
        new_imports = '''# Импорты новых интегрированных модулей
from ..contradiction.contradiction_integrated import IntegratedContradictionManager
from ..ethics.ethics_integrated import IntegratedEthicsFramework
'''
        
        # Находим место для вставки импортов
        import_pos = content.find("# Импорты интегрированных модулей")
        if import_pos != -1:
            # Вставляем после существующих импортов
            insert_pos = content.find("\n", import_pos) + 1
            content = content[:insert_pos] + new_imports + content[insert_pos:]
        
        # Добавляем фабрики в конец файла
        new_factories = '''
    def create_contradiction_manager(self) -> IntegratedContradictionManager:
        """Создает интегрированный менеджер противоречий."""
        try:
            logger.debug("Создание интегрированного менеджера противоречий...")
            event_bus = self.brain.get_event_bus() if hasattr(self.brain, 'get_event_bus') else self.event_bus
            component = IntegratedContradictionManager(
                event_bus=event_bus,
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, "contradiction")
            )
            logger.debug("Интегрированный менеджер противоречий создан")
            return component
        except Exception as e:
            logger.error(f"Ошибка создания интегрированного менеджера противоречий: {e}")
            raise
    
    def create_ethics_framework(self) -> IntegratedEthicsFramework:
        """Создает интегрированный этический фреймворк."""
        try:
            logger.debug("Создание интегрированного этического фреймворка...")
            event_bus = self.brain.get_event_bus() if hasattr(self.brain, 'get_event_bus') else self.event_bus
            component = IntegratedEthicsFramework(
                event_bus=event_bus,
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, "ethics")
            )
            logger.debug("Интегрированный этический фреймворк создан")
            return component
        except Exception as e:
            logger.error(f"Ошибка создания интегрированного этического фреймворка: {e}")
            raise
'''
        
        # Находим конец файла перед последним методом
        last_method_pos = content.rfind("def ")
        if last_method_pos != -1:
            # Находим конец этого метода
            method_end = content.find("\n\n", last_method_pos)
            if method_end == -1:
                method_end = len(content)
            
            content = content[:method_end] + new_factories + content[method_end:]
        
        # Сохраняем обновленный файл
        with open(initializer_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"   ✅ Обновлен файл: {initializer_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка обновления ComponentInitializer: {e}")
        return False

def main():
    """Основная функция интеграции"""
    print("🚀 Продолжение интеграции модулей CogniFlex")
    print("=" * 50)
    
    results = []
    
    # 1. Создаем интегрированный модуль противоречий
    try:
        contradiction_path = create_integrated_contradiction()
        results.append(("contradiction", True, contradiction_path))
    except Exception as e:
        results.append(("contradiction", False, str(e)))
    
    # 2. Создаем интегрированный этический модуль
    try:
        ethics_path = create_integrated_ethics()
        results.append(("ethics", True, ethics_path))
    except Exception as e:
        results.append(("ethics", False, str(e)))
    
    # 3. Обновляем ComponentInitializer
    try:
        initializer_success = update_component_initializer_with_new_modules()
        results.append(("component_initializer", initializer_success, ""))
    except Exception as e:
        results.append(("component_initializer", False, str(e)))
    
    # 4. Итоги
    print(f"\n📊 ИТОГИ ИНТЕГРАЦИИ:")
    
    success_count = 0
    for module, success, path in results:
        status = "✅ УСПЕХ" if success else "❌ НЕУДАЧА"
        print(f"   {module}: {status}")
        if success and path:
            print(f"      📁 {path}")
        elif not success:
            print(f"      ⚠️ {path}")
        
        if success:
            success_count += 1
    
    print(f"\n🎯 Результат: {success_count}/{len(results)} модулей интегрировано")
    
    if success_count == len(results):
        print("🎉 Интеграция успешно завершена!")
        return True
    else:
        print("⚠️ Некоторые модули требуют внимания.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
