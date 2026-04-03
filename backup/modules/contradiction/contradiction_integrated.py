"""
Интегрированный менеджер противоречий ЕВА
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("eva.contradiction")

from eva.core.base_component import BaseComponent, ComponentState
from eva.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальные менеджеры
try:
    from eva.contradiction.contradiction_manager import ContradictionManager
    ORIGINAL_MANAGER_AVAILABLE = True
except ImportError:
    ORIGINAL_MANAGER_AVAILABLE = False
    logger.warning("Оригинальный ContradictionManager недоступен")

try:
    from eva.contradiction.contradiction_resolver import ContradictionResolver
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
