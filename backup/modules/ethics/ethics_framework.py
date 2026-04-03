"""
Модуль этической рамки для ЕВА - обеспечение этических стандартов в работе системы
"""
import os
import logging
import time
import threading
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict

# Импортируем менеджер идентификаторов
from .violation_id_manager import (
    generate_violation_id,
    parse_violation_id,
    is_valid_violation_id,
    get_violation_principle,
    get_violation_timestamp
)

logger = logging.getLogger("eva.ethics")

@dataclass
class EthicalPrinciple:
    """Представляет этический принцип."""
    name: str
    description: str
    weight: float = 1.0  # Вес принципа в общей оценке
    threshold: float = 0.8  # Порог для нарушения принципа
    category: str = "general"  # Категория принципа (безопасность, приватность и т.д.)
    priority: int = 5  # Приоритет (1-10, где 10 - самый высокий)
    last_updated: float = field(default_factory=time.time)  # Время последнего обновления
    active: bool = True  # Активен ли принцип

@dataclass
class EthicalDecision:
    """Представляет этическое решение."""
    approved: bool
    principle: str
    severity: float
    description: str
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None
    resolution_timestamp: Optional[float] = None
    source: str = "system"  # Источник решения (система, человек и т.д.)
    violation_id: str = field(init=False)  # Добавляем поле для уникального ID

    def __post_init__(self):
        """Генерируем уникальный ID при создании объекта."""
        self.violation_id = generate_violation_id(self.principle, self.timestamp)

@dataclass
class EthicalAssessment:
    """Результат этической оценки."""
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    principle_scores: Dict[str, float]
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class EthicalReview:
    """Результат этического обзора контента."""
    content: str
    review_type: str
    reviewer: str
    decision: EthicalDecision
    timestamp: float = field(default_factory=time.time)

@dataclass
class EthicalIssue:
    name: str
    description: str
    type: str
    priority: float
    evidence: List[str]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[Dict[str, Any]] = None

@dataclass
class EthicsAnalysisResult:
    overall_score: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    principle_scores: Dict[str, float]

class EthicsFramework:
    """
    Этическая рамка для ЕВА - управление этическими решениями и проверками.
    
    Основные функции:
    - Оценка запросов на соответствие этическим принципам
    - Выявление потенциальных этических проблем
    - Генерация рекомендаций по разрешению этических дилемм
    - Отслеживание и анализ этических решений
    """
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует этическую рамку.
        
        Args:
            brain: Ссылка на ядро ЕВА (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "ethics_cache")
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        
        # Создаем директорию кэша если её нет
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Файлы для хранения данных
        self.principles_file = os.path.join(self.cache_dir, "principles.json")
        self.violations_file = os.path.join(self.cache_dir, "violations.json")
        self.stats_file = os.path.join(self.cache_dir, "stats.json")
        
        # Инициализация компонентов
        self.principles: Dict[str, EthicalPrinciple] = {}
        self.violations: List[EthicalDecision] = []
        self.stats = {
            "total_assessments": 0,
            "violations_detected": 0,
            "high_severity_violations": 0,
            "resolved_violations": 0,
            "pending_reviews": 0,
            "last_assessment": 0
        }
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
        
        # Загружаем конфигурацию
        self._load_configuration()
        
        # Инициализируем фоновые процессы
        self._init_background_services()
        
        logger.info("Этическая рамка ЕВА инициализирована")
        self.initialized = True
    
    def is_ready(self) -> bool:
        """Проверяет готовность этической рамки к работе."""
        return self.initialized and len(self.principles) > 0

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
            
            # Проверяем версию конфигурации
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
        
        # Обрабатываем список принципов
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
        
        # Загружаем принципы
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
        
        # Загружаем принципы
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
        """Инициализирует принципы по умолчанию."""
        default_principles = {
            "safety": EthicalPrinciple(
                name="safety",
                description="Обеспечение безопасности пользователей",
                weight=1.0,
                threshold=0.8,
                category="safety",
                priority=10
            ),
            "privacy": EthicalPrinciple(
                name="privacy",
                description="Защита приватности данных",
                weight=0.9,
                threshold=0.7,
                category="privacy",
                priority=9
            )
        }
        self.principles.update(default_principles)
        logger.info(f"Инициализировано {len(default_principles)} принципов по умолчанию")
    
    def _init_background_services(self):
        """Инициализирует фоновые службы для мониторинга этики."""
        try:
            import threading
            
            # Запускаем поток мониторинга нарушений
            self._violation_monitor_thread = threading.Thread(
                target=self._monitor_violations,
                daemon=True,
                name="EthicsViolationMonitor"
            )
            self._violation_monitor_thread.start()
            
            # Запускаем периодическую проверку принципов
            self._principle_check_thread = threading.Thread(
                target=self._periodic_principle_check,
                daemon=True,
                name="EthicsPrincipleCheck"
            )
            self._principle_check_thread.start()
            
            logger.debug("Фоновые службы этической рамки инициализированы")
        except Exception as e:
            logger.error(f"Ошибка инициализации фоновых служб: {e}")
    
    def _monitor_violations(self):
        """Мониторит нарушения в фоне."""
        while self.running:
            try:
                time.sleep(60)  # Проверка каждую минуту
                if not self.running:
                    break
                    
                # Проверяем старые нарушения на разрешение
                self._check_resolved_violations()
            except Exception as e:
                logger.error(f"Ошибка мониторинга нарушений: {e}")
    
    def _check_resolved_violations(self):
        """Проверяет разрешенные нарушения."""
        try:
            current_time = time.time()
            for violation in list(self.violations):
                if not isinstance(violation, EthicalDecision):
                    continue
                if not violation.resolved and (current_time - violation.timestamp) > 7 * 24 * 3600:
                    violation.resolved = True
                    violation.resolution_timestamp = current_time
                    logger.info(f"Автоматически разрешено старое нарушение: {violation.violation_id}")
        except Exception as e:
            logger.error(f"Ошибка проверки разрешенных нарушений: {e}")
    
    def _periodic_principle_check(self):
        """Периодически проверяет актуальность принципов."""
        while self.running:
            try:
                time.sleep(3600)  # Проверка каждый час
                if not self.running:
                    break
                    
                # Обновляем статистику принципов
                self._update_principle_stats()
            except Exception as e:
                logger.error(f"Ошибка периодической проверки принципов: {e}")
    
    def _update_principle_stats(self):
        """Обновляет статистику использования принципов."""
        try:
            for principle_name, principle in self.principles.items():
                # Обновляем время последнего использования
                principle.last_updated = time.time()
        except Exception as e:
            logger.error(f"Ошибка обновления статистики принципов: {e}")
    
    def _load_violations_and_stats(self):
        """Загружает нарушения и статистику из хранилища."""
        try:
            cache_dir = self.cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            
            # Загружаем нарушения
            violations_file = os.path.join(cache_dir, 'violations.json')
            if os.path.exists(violations_file):
                with open(violations_file, 'r', encoding='utf-8') as f:
                    loaded_violations = json.load(f)
                    for vdata in loaded_violations:
                        try:
                            self.violations.append(EthicalDecision(
                                approved=vdata.get("approved", False),
                                principle=vdata.get("principle", "unknown"),
                                severity=vdata.get("severity", 0.0),
                                description=vdata.get("description", ""),
                                context=vdata.get("context", {}),
                                timestamp=vdata.get("timestamp", time.time()),
                                resolved=vdata.get("resolved", False),
                                resolution=vdata.get("resolution"),
                                resolution_timestamp=vdata.get("resolution_timestamp"),
                                source=vdata.get("source", "system")
                            ))
                        except Exception:
                            pass
                    logger.debug(f"Загружено {len(loaded_violations)} нарушений")
            
            # Загружаем статистику
            stats_file = os.path.join(cache_dir, 'ethics_stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
                    logger.debug("Статистика этики загружена")
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки нарушений и статистики: {e}")
    
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
            
        self.stop_event.set()
        self.running = False
        logger.info("Этическая рамка остановлена")

    def analyze_content(self, content: str, context: Optional[Dict[str, Any]] = None) -> EthicsAnalysisResult:
        result = self.analyze_request(content, context=context)
        try:
            principle_scores = result.get('principle_scores', {}) if isinstance(result, dict) else {}
            violations = result.get('violations', []) if isinstance(result, dict) else []
            recommendations = result.get('recommendations', []) if isinstance(result, dict) else []
            max_score = 0.0
            try:
                max_score = float(max(principle_scores.values())) if principle_scores else 0.0
            except Exception:
                max_score = 0.0
            overall_score = max(0.0, min(1.0, 1.0 - max_score))
            return EthicsAnalysisResult(
                overall_score=overall_score,
                violations=violations,
                recommendations=recommendations,
                principle_scores=principle_scores
            )
        except Exception:
            return EthicsAnalysisResult(
                overall_score=1.0,
                violations=[],
                recommendations=[],
                principle_scores={}
            )

    def analyze_response(self, query: str, response: str) -> Dict[str, Any]:
        analysis = self.analyze_content(response, context={"query": query})
        return {
            "overall_score": analysis.overall_score,
            "violations": analysis.violations,
            "recommendations": analysis.recommendations,
            "principle_scores": analysis.principle_scores
        }

    def get_system_health(self) -> Dict[str, Any]:
        try:
            score = 1.0 if self.is_ready() else 0.3
        except Exception:
            score = 0.0
        return {
            "health_score": score,
            "status": "healthy" if score > 0.7 else "warning" if score > 0.3 else "critical",
            "initialized": bool(getattr(self, 'initialized', False)),
            "running": bool(getattr(self, 'running', False)),
            "principles_count": len(getattr(self, 'principles', {}) or {}),
            "timestamp": time.time()
        }

    def get_system_status(self) -> Dict[str, Any]:
        return self.get_system_health()

    def analyze_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Анализирует запрос на соответствие этическим принципам.
        
        Args:
            request: Текст запроса пользователя
            context: Дополнительный контекст
            
        Returns:
            Dict: Результат анализа
        """
        start_time = time.time()
        
        with self.lock:
            self.stats["total_assessments"] += 1
            self.stats["last_assessment"] = time.time()
            
            # Анализируем запрос
            assessment = self._assess_request(request, context)
            
            # Проверяем на нарушения
            violations = []
            for principle_name, score in assessment.principle_scores.items():
                principle = self.principles.get(principle_name)
                if principle and score > principle.threshold:
                    violation = EthicalDecision(
                        approved=False,
                        principle=principle_name,
                        severity=score,
                        description=f"Нарушение принципа {principle_name}",
                        context=context or {}
                    )
                    violations.append(violation)
                    self.violations.append(violation)
                    
                    # Обновляем статистику
                    self.stats["violations_detected"] += 1
                    if score > 0.9:
                        self.stats["high_severity_violations"] += 1
            
            # Сохраняем нарушения
            self._save_violations()
            
            # Формируем результат
            result = {
                "approved": len(violations) == 0,
                "violations": [v.__dict__ for v in violations],
                "principle_scores": assessment.principle_scores,
                "confidence": assessment.confidence,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time
            }
            
            if not result["approved"]:
                result["response"] = self._generate_rejection_response(violations)
            
            # Логируем результат
            if violations:
                logger.warning(f"Обнаружено {len(violations)} этических нарушений в запросе")
                for violation in violations:
                    logger.warning(f"Нарушение: {violation.principle} (серьезность: {violation.severity:.2f})")
            else:
                logger.info("Запрос прошел этическую проверку успешно")
            
            return result

    def _assess_request(self, request: str, context: Optional[Dict[str, Any]]) -> EthicalAssessment:
        """
        Оценивает запрос на соответствие этическим принципам.
        
        Args:
            request: Текст запроса
            context: Дополнительный контекст
            
        Returns:
            EthicalAssessment: Результат оценки
        """
        principle_scores = {}
        recommendations = []
        
        # Анализируем запрос по каждому принципу
        for name, principle in self.principles.items():
            score = self._evaluate_principle(request, context, principle)
            principle_scores[name] = score
            
            # Генерируем рекомендации для нарушений
            if score > principle.threshold:
                recommendations.append(
                    f"Внимание: возможное нарушение принципа '{name}' (уровень: {score:.2f}). "
                    "Рекомендуется пересмотреть запрос или предоставить дополнительные разъяснения."
                )
        
        # Вычисляем общий уровень доверия
        confidence = self._calculate_confidence(principle_scores)
        
        return EthicalAssessment(
            violations=[],
            recommendations=recommendations,
            principle_scores=principle_scores,
            confidence=confidence
        )

    def _evaluate_principle(self, request: str, context: Optional[Dict[str, Any]], 
                           principle: EthicalPrinciple) -> float:
        """
        Оценивает запрос на соответствие конкретному этическому принципу.
        
        Args:
            request: Текст запроса
            context: Дополнительный контекст
            principle: Этический принцип
            
        Returns:
            float: Уровень нарушения (0.0-1.0)
        """
        # Стандартная оценка
        score = 0.0
        
        # Анализируем в зависимости от категории принципа
        if principle.category == "privacy":
            score = self._evaluate_privacy(request, context)
        elif principle.category == "safety":
            score = self._evaluate_safety(request, context)
        elif principle.category == "fairness":
            score = self._evaluate_fairness(request, context)
        elif principle.category == "transparency":
            score = self._evaluate_transparency(request, context)
        elif principle.category == "autonomy":
            score = self._evaluate_autonomy(request, context)
        elif principle.category == "beneficence":
            score = self._evaluate_beneficence(request, context)
        elif principle.category == "accountability":
            score = self._evaluate_accountability(request, context)
        
        # Нормализуем оценку
        return min(max(score, 0.0), 1.0)

    def _evaluate_privacy(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу приватности."""
        score = 0.0
        
        # Проверка на запрос личной информации
        personal_info_keywords = [
            "имя", "фамилия", "адрес", "телефон", "email", "почта", "номер", 
            "паспорт", "документ", "личные данные", "конфиденциальная", "приватная"
        ]
        
        for keyword in personal_info_keywords:
            if keyword in request.lower():
                score += 0.15
        
        # Проверка на упоминание конкретных людей
        if re.search(r"\b[А-Я][а-я]+\s+[А-Я][а-я]+\b", request):
            score += 0.1
        
        # Проверка на запрос информации о других людях
        if "о ком-то" in request.lower() or "человеке" in request.lower():
            score += 0.1
        
        # Проверка на запрос конфиденциальной информации
        if "конфиденциальная" in request.lower() or "секрет" in request.lower():
            score += 0.2
        
        return min(score, 1.0)

    def _evaluate_safety(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу безопасности."""
        score = 0.0
        
        # Проверка на опасный контент
        dangerous_keywords = [
            "убить", "навредить", "повредить", "опасно", "опасность", "вред", "взрыв", 
            "оружие", "наркотик", "наркотики", "взлом", "взломать", "взломать", "обмануть"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in request.lower():
                score += 0.2
        
        # Проверка на призывы к насилию
        if "призыв" in request.lower() and ("насилие" in request.lower() or "агрессия" in request.lower()):
            score += 0.3
        
        # Проверка на запрос инструкций по опасным действиям
        if "как" in request.lower() and ("сделать взрыв" in request.lower() or "создать оружие" in request.lower()):
            score += 0.4
        
        return min(score, 1.0)

    def _evaluate_fairness(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу справедливости."""
        score = 0.0
        
        # Проверка на дискриминационный язык
        discriminatory_keywords = [
            "расист", "расизм", "расистский", "националист", "национализм", 
            "ксенофоб", "ксенофобия", "сексист", "сексизм", "гомофоб", "гомофобия"
        ]
        
        for keyword in discriminatory_keywords:
            if keyword in request.lower():
                score += 0.2
        
        # Проверка на предвзятые утверждения
        if "все [A-ZА-Я][a-zа-я]+" in request.lower() and "плохие" in request.lower():
            score += 0.15
        
        # Проверка на запрос предвзятой информации
        if "лучшая раса" in request.lower() or "лучшая национальность" in request.lower():
            score += 0.25
        
        return min(score, 1.0)

    def _evaluate_transparency(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу прозрачности."""
        score = 0.0
        
        # Проверка на запрос скрытой информации
        if "секрет" in request.lower() and "алгоритм" in request.lower():
            score += 0.2
        
        # Проверка на запрос обмана
        if "как обмануть" in request.lower() or "как скрыть" in request.lower():
            score += 0.3
        
        # Проверка на запрос манипулятивной информации
        if "манипулировать" in request.lower() or "ввести в заблуждение" in request.lower():
            score += 0.25
        
        return min(score, 1.0)

    def _evaluate_autonomy(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу автономии."""
        score = 0.0
        
        # Проверка на запрос ограничения свободы
        if "контролировать" in request.lower() and "мысли" in request.lower():
            score += 0.3
        
        # Проверка на запрос манипуляции выбором
        if "заставить выбрать" in request.lower() or "манипулировать выбором" in request.lower():
            score += 0.25
        
        # Проверка на запрос отмены прав пользователя
        if "отменить права" in request.lower() or "ограничить возможности" in request.lower():
            score += 0.2
        
        return min(score, 1.0)

    def _evaluate_beneficence(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу пользы."""
        score = 0.0
        
        # Проверка на запрос вреда
        if "вред" in request.lower() and ("пользователю" in request.lower() or "людям" in request.lower()):
            score += 0.4
        
        # Проверка на запрос бесполезной информации
        if "бесполезная информация" in request.lower() or "пустая трата времени" in request.lower():
            score += 0.15
        
        # Проверка на запрос дезинформации
        if "ложная информация" in request.lower() or "дезинформация" in request.lower():
            score += 0.25
        
        return min(score, 1.0)

    def _evaluate_accountability(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу подотчетности."""
        score = 0.0
        
        # Проверка на запрос анонимных действий
        if "анонимно" in request.lower() and ("действие" in request.lower() or "операция" in request.lower()):
            score += 0.25
        
        # Проверка на запрос скрытия ответственности
        if "скрыть ответственность" in request.lower() or "избежать ответственности" in request.lower():
            score += 0.3
        
        # Проверка на запрос уклонения от последствий
        if "уйти от последствий" in request.lower() or "избежать наказания" in request.lower():
            score += 0.2
        
        return min(score, 1.0)

    def _calculate_confidence(self, principle_scores: Dict[str, float]) -> float:
        """Вычисляет общий уровень доверия к оценке."""
        # Берем среднее значение с учетом весов
        total_weight = 0
        weighted_sum = 0
        
        for name, score in principle_scores.items():
            principle = self.principles.get(name)
            weight = principle.weight if principle else 1.0
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _generate_rejection_response(self, violations: List[EthicalDecision]) -> str:
        """
        Генерирует ответ при отклонении запроса по этическим соображениям.
        
        Args:
            violations: Список обнаруженных нарушений
            
        Returns:
            str: Ответ для пользователя
        """
        # Определяем самый серьезный тип нарушения
        highest_severity = max(v.severity for v in violations)
        primary_violation = next(v for v in violations if v.severity == highest_severity)
        
        principle_key = primary_violation.principle.lower()
        
        if principle_key == "privacy":
            return (
                "Извините, но ваш запрос затрагивает вопросы приватности и конфиденциальности. "
                "Я не могу обрабатывать запросы, связанные с личной информацией других людей или "
                "конфиденциальными данными. Пожалуйста, переформулируйте ваш запрос так, чтобы он "
                "не нарушал принципы приватности."
            )
        elif principle_key == "non-maleficence" or principle_key == "non_maleficence":
            return (
                "Извините, но ваш запрос может привести к потенциальному вреду или опасности. "
                "Я не могу участвовать в обсуждении или предоставлении информации, которая может "
                "нанести вред людям или нарушить безопасность. Пожалуйста, переформулируйте ваш запрос."
            )
        elif principle_key == "justice":
            return (
                "Извините, но ваш запрос содержит элементы дискриминации или несправедливости. "
                "Я не могу поддерживать или распространять информацию, которая нарушает принципы "
                "справедливости и равенства. Пожалуйста, переформулируйте ваш запрос более нейтрально."
            )
        else:
            return (
                "Извините, но ваш запрос не соответствует этическим стандартам системы. "
                "Пожалуйста, переформулируйте запрос так, чтобы он соответствовал принципам "
                "уважения, безопасности и пользы для всех пользователей."
            )

    def get_violation_history(self, limit: int = 50, 
                              principle: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Возвращает историю нарушений.
        
        Args:
            limit: Максимальное количество записей
            principle: Фильтр по принципу
            
        Returns:
            List[Dict]: Список нарушений
        """
        with self.lock:
            filtered = self.violations
            if principle:
                filtered = [v for v in filtered if v.principle == principle]
            
            # Сортируем по времени (новые сначала)
            filtered = sorted(filtered, key=lambda x: x.timestamp, reverse=True)
            
            # Ограничиваем количество
            result = [v.__dict__ for v in filtered[:limit]]
            return result

    def resolve_violation(self, violation_id: str, resolution: str, 
                         reviewer: str = "system") -> bool:
        """
        Разрешает нарушение.
        
        Args:
            violation_id: ID нарушения
            resolution: Описание решения
            reviewer: Кто разрешил нарушение
            
        Returns:
            bool: Успешно ли разрешено
        """
        with self.lock:
            # Проверяем валидность ID
            if not is_valid_violation_id(violation_id):
                logger.warning(f"Недействительный ID нарушения: {violation_id}")
                return False
            
            # Проверяем, что ID соответствует ожидаемому принципу
            expected_principle = get_violation_principle(violation_id)
            if expected_principle and expected_principle not in self.principles:
                logger.warning(f"Принцип '{expected_principle}' из ID не найден в системе")
                return False
            
            # Ищем нарушение по ID
            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    violation.resolution = resolution
                    violation.resolution_timestamp = time.time()
                    
                    # Обновляем статистику
                    self.stats["resolved_violations"] += 1
                    self.stats["pending_reviews"] = max(0, self.stats["pending_reviews"] - 1)
                    
                    # Сохраняем изменения
                    self._save_violations()
                    self._save_stats()
                    
                    logger.info(f"Нарушение {violation_id} разрешено")
                    return True
            
            logger.warning(f"Нарушение {violation_id} не найдено для разрешения")
            return False

    def _save_violations(self):
        """Сохраняет нарушения в файл."""
        try:
            violations_data = []
            for violation in self.violations:
                violations_data.append({
                    "approved": violation.approved,
                    "principle": violation.principle,
                    "severity": violation.severity,
                    "description": violation.description,
                    "context": violation.context,
                    "timestamp": violation.timestamp,
                    "resolved": violation.resolved,
                    "resolution": violation.resolution,
                    "resolution_timestamp": violation.resolution_timestamp,
                    "source": violation.source,
                    "violation_id": violation.violation_id  # Сохраняем уникальный ID
                })
            
            with open(self.violations_file, 'w', encoding='utf-8') as f:
                json.dump(violations_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения нарушений: {e}", exc_info=True)

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

    def _save_stats(self):
        """Сохраняет статистику в файл."""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения статистики: {e}", exc_info=True)

    def get_ethics_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику этической рамки."""
        with self.lock:
            return {
                "total_assessments": self.stats["total_assessments"],
                "violations_detected": self.stats["violations_detected"],
                "high_severity_violations": self.stats["high_severity_violations"],
                "resolved_violations": self.stats["resolved_violations"],
                "pending_reviews": self.stats["pending_reviews"],
                "last_assessment": self.stats["last_assessment"],
                "violation_rate": (self.stats["violations_detected"] / self.stats["total_assessments"]) \
                    if self.stats["total_assessments"] > 0 else 0
            }

    def export_ethics_data(self, file_path: str) -> bool:
        """
        Экспортирует данные этической рамки в файл.
        
        Args:
            file_path: Путь к файлу для экспорта
            
        Returns:
            bool: Успешно ли экспортировано
        """
        try:
            export_data = {
                "metadata": {
                    "format_version": "1.0",
                    "exported_at": time.time(),
                    "system": "ЕВА"
                },
                "principles": {
                    name: {
                        "name": principle.name,
                        "description": principle.description,
                        "weight": principle.weight,
                        "threshold": principle.threshold,
                        "category": principle.category,
                        "priority": principle.priority
                    } for name, principle in self.principles.items()
                },
                "violations": [{
                    "approved": v.approved,
                    "principle": v.principle,
                    "severity": v.severity,
                    "description": v.description,
                    "context": v.context,
                    "timestamp": v.timestamp,
                    "resolved": v.resolved,
                    "resolution": v.resolution,
                    "resolution_timestamp": v.resolution_timestamp,
                    "source": v.source,
                    "violation_id": v.violation_id  # Экспортируем уникальный ID
                } for v in self.violations],
                "statistics": self.stats
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Данные этической рамки экспортированы в {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта данных этической рамки: {e}", exc_info=True)
            return False

    def import_ethics_data(self, file_path: str) -> bool:
        """
        Импортирует данные этической рамки из файла.
        
        Args:
            file_path: Путь к файлу для импорта
            
        Returns:
            bool: Успешно ли импортировано
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Импортируем принципы
            self.principles = {}
            for name, data in import_data["principles"].items():
                self.principles[name] = EthicalPrinciple(
                    name=data["name"],
                    description=data["description"],
                    weight=data["weight"],
                    threshold=data["threshold"],
                    category=data["category"],
                    priority=data["priority"]
                )
            
            # Импортируем нарушения
            self.violations = []
            for data in import_data["violations"]:
                # Проверяем валидность violation_id
                if "violation_id" in data and is_valid_violation_id(data["violation_id"]):
                    self.violations.append(EthicalDecision(
                        approved=data["approved"],
                        principle=data["principle"],
                        severity=data["severity"],
                        description=data["description"],
                        context=data["context"],
                        timestamp=data["timestamp"],
                        resolved=data["resolved"],
                        resolution=data.get("resolution"),
                        resolution_timestamp=data.get("resolution_timestamp"),
                        source=data.get("source", "system")
                    ))
                else:
                    # Генерируем новый ID, если старый недействителен
                    violation = EthicalDecision(
                        approved=data["approved"],
                        principle=data["principle"],
                        severity=data["severity"],
                        description=data["description"],
                        context=data["context"],
                        timestamp=data["timestamp"],
                        resolved=data["resolved"],
                        resolution=data.get("resolution"),
                        resolution_timestamp=data.get("resolution_timestamp"),
                        source=data.get("source", "system")
                    )
                    self.violations.append(violation)
            
            # Импортируем статистику
            self.stats = import_data["statistics"]
            
            # Сохраняем импортированные данные
            self._save_principles()
            self._save_violations()
            self._save_stats()
            
            logger.info(f"Данные этической рамки импортированы из {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка импорта данных этической рамки: {e}", exc_info=True)
            return False

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
            
            self._save_principles()
            logger.info(f"Обновлен этический принцип: {name}")
            return True

    def get_active_violations(self) -> List[Dict[str, Any]]:
        """Возвращает список активных (неразрешенных) нарушений."""
        with self.lock:
            active = [v for v in self.violations if not v.resolved]
            # Сортируем по серьезности (сначала самые серьезные)
            active = sorted(active, key=lambda x: x.severity, reverse=True)
            return [v.__dict__ for v in active]

    def get_principle(self, name: str) -> Optional[EthicalPrinciple]:
        """Возвращает принцип по имени."""
        return self.principles.get(name)

    def get_all_principles(self) -> Dict[str, EthicalPrinciple]:
        """Возвращает все принципы."""
        return self.principles.copy()

    def _analyze_ethical_trends(self) -> Dict[str, Any]:
        """
        Анализирует тенденции в этических нарушениях.
        
        Returns:
            Dict: Результат анализа тенденций
        """
        trends = {
            "by_principle": defaultdict(int),
            "by_severity": {
                "low": 0,
                "medium": 0,
                "high": 0
            },
            "by_category": defaultdict(int),
            "time_trends": []
        }
        
        # Анализируем последние 100 нарушений
        recent_violations = self.violations[-100:]
        
        for violation in recent_violations:
            # По принципам
            trends["by_principle"][violation.principle] += 1
            
            # По категориям
            principle = self.principles.get(violation.principle)
            if principle:
                trends["by_category"][principle.category] += 1
            
            # По серьезности
            if violation.severity > 0.8:
                trends["by_severity"]["high"] += 1
            elif violation.severity > 0.6:
                trends["by_severity"]["medium"] += 1
            else:
                trends["by_severity"]["low"] += 1
        
        # Анализ временных тенденций (последние 7 дней)
        now = time.time()
        week_ago = now - 7 * 24 * 3600
        daily_counts = [0] * 7
        
        for violation in self.violations:
            if violation.timestamp >= week_ago:
                day = int((violation.timestamp - week_ago) / (24 * 3600))
                if 0 <= day < 7:
                    daily_counts[day] += 1
        
        trends["time_trends"] = daily_counts
        
        return dict(trends)

    def generate_ethics_report(self) -> Dict[str, Any]:
        """
        Генерирует отчет по этической активности.
        
        Returns:
            Dict: Отчет
        """
        with self.lock:
            trends = self._analyze_ethical_trends()
            
            # Определяем основные проблемы
            main_issues = []
            if trends["by_severity"]["high"] > 0:
                high_principle = max(trends["by_principle"], key=trends["by_principle"].get)
                main_issues.append(f"Высокая частота нарушений принципа '{high_principle}'")
            
            if trends["by_severity"]["high"] > 5:
                main_issues.append("Критически высокий уровень серьезных нарушений")
            
            # Формируем рекомендации
            recommendations = []
            if trends["by_severity"]["high"] > 0:
                recommendations.append(
                    "Рассмотрите возможность усиления проверок для принципа с наибольшим "
                    "количеством нарушений"
                )
            if trends["time_trends"][-1] > trends["time_trends"][-2] * 1.5:
                recommendations.append(
                    "Наблюдается рост числа нарушений, рекомендуется провести анализ причин"
                )
            
            return {
                "timestamp": time.time(),
                "statistics": self.get_ethics_statistics(),
                "trends": trends,
                "main_issues": main_issues,
                "recommendations": recommendations
            }

    def check_with_context(
        self,
        text: str,
        query: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Проверяет текст на этические нарушения с учётом контекста
        
        Args:
            text: Текст для проверки
            query: Оригинальный запрос
            context: Дополнительный контекст
            
        Returns:
            Dict с результатами проверки и рекомендациями
        """
        logger.info("Проверка этики с контекстом...")
        
        # Используем существующий метод analyze_content если доступен
        try:
            # Используем analyze_content если есть
            if hasattr(self, 'analyze_content'):
                analysis = self.analyze_content(text, context)
                # analysis - это EthicsAnalysisResult dataclass
                if hasattr(analysis, 'violations'):
                    violations = analysis.violations if isinstance(analysis.violations, list) else []
                else:
                    violations = []
            else:
                violations = []
            
            # Классифицируем нарушения по серьёзности
            critical = [v for v in violations if v.get('severity', 0) > 0.8]
            warnings = [v for v in violations if 0.5 < v.get('severity', 0) <= 0.8]
            minor = [v for v in violations if v.get('severity', 0) <= 0.5]
            
            # Рассчитываем оценку
            overall_score = 1.0 - (len(critical) * 0.3 + len(warnings) * 0.1)
            overall_score = max(0.0, min(1.0, overall_score))
            
            return {
                'violations': violations,
                'critical_violations': critical,
                'warnings': warnings,
                'minor_violations': minor,
                'overall_score': overall_score,
                'has_violations': len(critical) > 0 or len(warnings) > 0,
                'violation_count': len(violations)
            }
            
        except Exception as e:
            logger.warning(f"Ошибка при check_with_context: {e}")
            return {
                'violations': [],
                'overall_score': 0.5,
                'has_violations': False,
                'violation_count': 0
            }
    
    def generate_regeneration_prompt(
        self,
        ethics_result: Dict[str, Any],
        query: str = "",
        response: str = ""
    ) -> str:
        """
        Генерирует промпт для регенерации при этических нарушениях
        
        Args:
            ethics_result: Результат check_with_context()
            query: Оригинальный запрос
            response: Текущий ответ
            
        Returns:
            str: Промпт для Qwen
        """
        critical = ethics_result.get('critical_violations', [])
        warnings = ethics_result.get('warnings', [])
        
        if not critical and not warnings:
            return ""  # Нет нарушений
        
        parts = []
        
        # Критические нарушения
        if critical:
            parts.append("КРИТИЧЕСКИЕ НАРУШЕНИЯ:")
            for i, v in enumerate(critical[:2], 1):
                principle = v.get('principle', 'unknown')
                desc = v.get('description', v.get('message', ''))
                parts.append(f"{i}. [{principle}] {desc}")
        
        # Предупреждения
        if warnings:
            parts.append("\nПРЕДУПРЕЖДЕНИЯ:")
            for i, v in enumerate(warnings[:2], 1):
                principle = v.get('principle', 'unknown')
                desc = v.get('description', v.get('message', ''))
                parts.append(f"{i}. [{principle}] {desc}")
        
        prompt = """Обнаружены этические нарушения в ответе:
"
        prompt += "\n".join(parts)
        prompt += "\n\nПереформулируй ответ, устранив нарушения. Будь этичной."""

        return prompt