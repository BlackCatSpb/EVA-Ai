"""
Интегрированный этический фреймворк ЕВА
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("eva_ai.ethics")

from eva_ai.core.base_component import BaseComponent, ComponentState
from eva_ai.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный этический фреймворк
try:
    from eva_ai.ethics.ethics_core import EthicsFramework
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
