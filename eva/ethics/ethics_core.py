"""Основной модуль этической рамки для ЕВА"""
import os
import logging
import time
import threading
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from eva.ethics.ethics_framework import EthicalDecision, EthicalIssue
from eva.ethics.principles_manager import PrinciplesManager
from eva.ethics.risk_assessment import RiskAssessor
from eva.ethics.ethical_situations import EthicalSituationHandler


logger = logging.getLogger("eva.ethics.core")

class EthicsFramework:
    """Основной модуль этической рамки ЕВА."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует этическую рамку.
        
        Args:
            brain: Ссылка на ядро ЕВА (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "eva_ethics_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Блокировка ресурсов
        self.lock = threading.Lock()
        
        # Инициализируем компоненты
        self.principles_manager = PrinciplesManager(self, self.cache_dir)
        self.risk_assessor = RiskAssessor(self.principles_manager, self.brain)
        self.situation_handler = EthicalSituationHandler(
            self.principles_manager, 
            self.risk_assessor, 
            self.brain
        )
        
        # Статистика
        self.stats = {
            "total_assessments": 0,
            "high_risk_situations": 0,
            "deferred_to_human": 0,
            "auto_decisions": 0,
            "last_assessment": 0
        }
        
        # Флаг работы системы
        self.running = False
        
        logger.info("Этическая рамка ЕВА инициализирована")
    
    def start(self):
        """Запускает фоновые процессы этической рамки."""
        if self.running:
            return
        
        self.running = True
        logger.info("Этическая рамка запущена")
    
    def stop(self):
        """Останавливает фоновые процессы этической рамки."""
        if not self.running:
            return
        
        self.running = False
        
        # Закрываем компоненты
        if self.situation_handler:
            self.situation_handler.close()
        if self.risk_assessor:
            self.risk_assessor = None
        if self.principles_manager:
            self.principles_manager.close()
        
        logger.info("Этическая рамка остановлена")
    
    def assess_ethics(self, context: Dict[str, Any]) -> EthicalDecision:
        """
        Оценивает этическую ситуацию и принимает решение.
        
        Args:
            context: Контекст для оценки
            
        Returns:
            EthicalDecision: Принятое решение
        """
        start_time = time.time()
        logger.debug(f"Начата оценка этической ситуации: {context.get('query', '')[:50]}...")
        
        try:
            # Оцениваем этические риски
            decision = self.situation_handler.handle_situation(context)
            
            # Обновляем статистику
            with self.lock:
                self.stats["total_assessments"] += 1
                self.stats["last_assessment"] = time.time()
                
                if decision.requires_human_review:
                    self.stats["deferred_to_human"] += 1
                else:
                    self.stats["auto_decisions"] += 1
                
                # Определяем высокий риск
                high_risk = any(
                    a.violation_detected and a.severity == "high" 
                    for a in decision.assessment
                )
                if high_risk:
                    self.stats["high_risk_situations"] += 1
            
            logger.info(f"Этическая оценка завершена. Решение: {decision.decision}")
            return decision
            
        except Exception as e:
            logger.error(f"Ошибка оценки этической ситуации: {str(e)}")
            with self.lock:
                self.stats["total_assessments"] += 1
                self.stats["last_assessment"] = time.time()
            return self._get_default_decision(context, str(e))
    
    def _get_default_decision(self, context: Dict[str, Any], error: str) -> EthicalDecision:
        """Возвращает решение по умолчанию в случае ошибки."""
        return EthicalDecision(
            decision="error",
            confidence=0.1,
            justification=f"Ошибка обработки этической ситуации: {error}",
            alternatives=[],
            assessment=[],
            requires_human_review=True
        )
    
    def needs_ethical_review(self, context: Dict[str, Any]) -> bool:
        """
        Определяет, требуется ли этический обзор для контекста.
        
        Args:
            context: Контекст для проверки
            
        Returns:
            bool: Требуется ли обзор
        """
        try:
            if not self.risk_assessor:
                return True
            # Оцениваем риски
            assessments = self.risk_assessor.assess_risk(context)
            
            # Проверяем, есть ли серьезные нарушения
            for assessment in assessments:
                if assessment.violation_detected and assessment.severity in ["high", "medium"]:
                    return True
            
            # Проверяем, требуется ли человеческое вмешательство
            return self.situation_handler._requires_human_review(assessments)
            
        except Exception as e:
            logger.error(f"Ошибка определения необходимости этического обзора: {e}")
            return True  # В случае ошибки требуем обзор для безопасности
    
    def get_ethical_issues(self, limit: int = 10, min_priority: float = 0.5) -> List[Dict[str, Any]]:
        """
        Возвращает список этических проблем.
        
        Args:
            limit: Максимальное количество проблем
            min_priority: Минимальный приоритет
            
        Returns:
            List[Dict[str, Any]]: Список этических проблем
        """
        return [
            {
                "name": issue.name,
                "description": issue.description,
                "type": issue.type,
                "priority": issue.priority,
                "evidence": issue.evidence,
                "timestamp": issue.timestamp,
                "resolved": issue.resolved,
                "resolution": issue.resolution
            } for issue in self.situation_handler.get_ethical_issues(limit, min_priority)
        ]
    
    def add_ethical_issue(self, issue_data: Dict[str, Any]):
        """
        Добавляет новую этическую проблему.
        
        Args:
            issue_data: Данные этической проблемы
        """
        issue = EthicalIssue(
            name=issue_data.get("name", ""),
            description=issue_data.get("description", ""),
            type=issue_data.get("type", "general"),
            priority=issue_data.get("priority", 0.5),
            evidence=issue_data.get("evidence", []),
            timestamp=issue_data.get("timestamp", time.time()),
            resolved=issue_data.get("resolved", False),
            resolution=issue_data.get("resolution")
        )
        self.situation_handler.add_ethical_issue(issue)
    
    def resolve_ethical_issue(self, issue_name: str, resolution: Dict[str, Any]):
        """
        Помечает этическую проблему как решенную.
        
        Args:
            issue_name: Название проблемы
            resolution: Описание решения
        """
        self.situation_handler.resolve_ethical_issue(issue_name, resolution)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье этической рамки.
        
        Returns:
            Dict: Отчет о здоровье
        """
        # Получаем данные от компонентов
        try:
            principles_health = getattr(self.principles_manager, 'get_principles_dashboard_data', lambda: {})()
        except Exception:
            principles_health = {}
        try:
            risk_health = getattr(self.risk_assessor, 'get_risk_dashboard_data', lambda: {})()
        except Exception:
            risk_health = {}
        try:
            situation_health = getattr(self.situation_handler, 'get_situation_dashboard_data', lambda: {})()
        except Exception:
            situation_health = {}
        
        # Получаем общий отчет о здоровье
        overall_health = self.situation_handler.get_system_health()
        
        return {
            "principles": principles_health,
            "risk_assessment": risk_health,
            "situations": situation_health,
            "overall": overall_health,
            "stats": self.stats,
            "timestamp": time.time()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Возвращает данные для дашборда этической рамки.
        
        Returns:
            Dict: Данные для дашборда
        """
        # Получаем данные от компонентов
        try:
            principles_data = getattr(self.principles_manager, 'get_principles_dashboard_data', lambda: {})()
        except Exception:
            principles_data = {}
        try:
            risk_data = getattr(self.risk_assessor, 'get_risk_dashboard_data', lambda: {})()
        except Exception:
            risk_data = {}
        try:
            situation_data = getattr(self.situation_handler, 'get_situation_dashboard_data', lambda: {})()
        except Exception:
            situation_data = {}
        
        return {
            "principles": principles_data,
            "risk": risk_data,
            "situations": situation_data,
            "stats": self.stats,
            "timestamp": time.time()
        }
    
    def generate_visualization(self, view_type: str = "compliance", component: str = "all") -> str:
        """
        Генерирует визуализацию данных этической рамки.
        
        Args:
            view_type: Тип визуализации
            component: Компонент для визуализации (principles, risk, situations, all)
            
        Returns:
            str: Изображение в формате base64
        """
        try:
            if component == "principles" or component == "all":
                vis_method = getattr(self.principles_manager, 'generate_ethical_visualization', None)
                if vis_method:
                    return vis_method(view_type)
            elif component == "risk":
                vis_method = getattr(self.risk_assessor, 'generate_risk_visualization', None)
                if vis_method:
                    return vis_method(view_type)
            elif component == "situations":
                vis_method = getattr(self.situation_handler, 'generate_situation_visualization', None)
                if vis_method:
                    return vis_method(view_type)
            
            # По умолчанию используем обзор
            vis_method = getattr(self.principles_manager, 'generate_ethical_visualization', None)
            if vis_method:
                return vis_method("compliance")
            
        except Exception as e:
            logger.error(f"Ошибка генерации визуализации этической рамки: {e}")
        return ""
    
    def export_data(self, file_path: str) -> bool:
        """
        Экспортирует данные этической рамки в файл.
        
        Args:
            file_path: Путь к файлу для экспорта
            
        Returns:
            bool: Успешно ли экспортировано
        """
        return self.situation_handler.export_ethics_data(file_path)
    
    def import_data(self, file_path: str) -> bool:
        """
        Импортирует данные этической рамки из файла.
        
        Args:
            file_path: Путь к файлу для импорта
            
        Returns:
            bool: Успешно ли импортировано
        """
        return self.situation_handler.import_ethics_data(file_path)
    
    def get_system_summary(self) -> str:
        """
        Возвращает краткую сводку о состоянии этической рамки.
        
        Returns:
            str: Сводка о состоянии
        """
        health = self.get_system_health()["overall"]
        
        summary = (
            f"Этическая рамка\n"
            f"{'=' * 30}\n\n"
            f"Оценка: {health['health_score']:.2f}/100\n"
            f"Принципы: {health['total_principles']}\n"
            f"Низкое соблюдение: {health['low_compliance_count']}\n"
            f"Открытые проблемы: {health['open_issues_count']}\n"
        )
        
        if health["health_score"] < 70:
            summary += "\nРекомендации:\n"
            for i, rec in enumerate(health["recommendations"][:2], 1):
                summary += f"{i}. {rec}\n"
        
        return summary
    
    def close(self):
        """Закрывает этическую рамку и освобождает ресурсы."""
        logger.info("Закрытие этической рамки...")
        self.stop()
        #logger.info("Этическая рамка закрыта")