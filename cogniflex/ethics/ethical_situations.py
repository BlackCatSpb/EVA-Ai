"""Модуль обработки этических ситуаций для CogniFlex"""
import os
import logging
import json
import time
import hashlib
import base64
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from io import BytesIO

# Добавляем недостающие импорты
try:
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None

try:
    from cogniflex.ethics.ethics_framework import EthicalDecision, EthicalIssue
    from cogniflex.ethics.principles_manager import PrinciplesManager
    from cogniflex.ethics.risk_assessment import RiskAssessor
except ImportError as e:
    logging.warning(f"Import error for ethics modules: {e}")
    # Create placeholder classes if imports fail
    class EthicalDecision:
        def __init__(self, decision, confidence, justification, alternatives, assessment, requires_human_review):
            self.decision = decision
            self.confidence = confidence
            self.justification = justification
            self.alternatives = alternatives
            self.assessment = assessment
            self.requires_human_review = requires_human_review
    
    class EthicalIssue:
        def __init__(self, name, description, type, priority, evidence, timestamp=None, resolved=False, resolution=None):
            self.name = name
            self.description = description
            self.type = type
            self.priority = priority
            self.evidence = evidence
            self.timestamp = timestamp or time.time()
            self.resolved = resolved
            self.resolution = resolution
    
    class PrinciplesManager:
        def __init__(self):
            pass
        def get_principle_by_name(self, name):
            return None
        def get_all_principles(self):
            return {}
        def add_principle(self, principle):
            pass
        def get_assessment_history(self, principle_id, days=7):
            return []
    
    class RiskAssessor:
        def __init__(self):
            pass
        def assess_risk(self, context):
            return []
        def get_risk_dashboard_data(self):
            return {}

logger = logging.getLogger("cogniflex.ethics.situations")

class EthicalAssessment:
    """Класс для представления этической оценки."""
    def __init__(self, principle_name: str, score: float, confidence: float, 
                 violation_detected: bool, severity: str):
        self.principle_name = principle_name
        self.score = score
        self.confidence = confidence
        self.violation_detected = violation_detected
        self.severity = severity

class EthicalPrinciple:
    """Класс для представления этического принципа."""
    def __init__(self, name: str, description: str, weight: float, 
                 threshold: float, category: str, last_updated: float, active: bool):
        self.name = name
        self.description = description
        self.weight = weight
        self.threshold = threshold
        self.category = category
        self.last_updated = last_updated
        self.active = active

class EthicalSituationHandler:
    """Обрабатывает этически сложные ситуации и принимает решения."""
    
    def __init__(self, principles_manager: PrinciplesManager = None, risk_assessor: RiskAssessor = None, brain=None):
        """
        Инициализирует обработчик этических ситуаций.
        
        Args:
            principles_manager: Менеджер этических принципов
            risk_assessor: Оценщик этических рисков
            brain: Ссылка на ядро CogniFlex
        """
        self.principles_manager = principles_manager or PrinciplesManager()
        self.risk_assessor = risk_assessor or RiskAssessor()
        self.brain = brain
        
        try:
            self.cache_dir = os.path.join(os.path.dirname(__file__), "cogniflex_ethics_cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Путь к файлу кэша решений
            self.solutions_cache_path = os.path.join(self.cache_dir, "ethical_solutions.json")
            self.review_cache_path = os.path.join(self.cache_dir, "ethical_reviews.json")
            
            # Загружаем кэш
            self.solutions_cache = self._load_cache(self.solutions_cache_path)
            self.review_cache = self._load_cache(self.review_cache_path)
            
            # База данных этических проблем
            self.ethical_issues = []
            self._load_ethical_issues()
            
            logger.info(f"Обработчик этических ситуаций инициализирован. Загружено {len(self.ethical_issues)} этических проблем")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации обработчика этических ситуаций: {e}")
            raise
    
    def _load_cache(self, file_path: str) -> Dict[str, Any]:
        """Загружает кэш из файла."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"solutions": [], "reviews": []}
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")
            return {"solutions": [], "reviews": []}
    
    def _save_cache(self):
        """Сохраняет кэш в файл."""
        try:
            with open(self.solutions_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.solutions_cache, f, ensure_ascii=False, indent=2)
            
            with open(self.review_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.review_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")
    
    def _load_ethical_issues(self):
        """Загружает этические проблемы из файла."""
        try:
            issues_file = os.path.join(self.cache_dir, "ethical_issues.json")
            
            if os.path.exists(issues_file):
                with open(issues_file, 'r', encoding='utf-8') as f:
                    issues_data = json.load(f)
                
                self.ethical_issues = [
                    EthicalIssue(
                        name=issue["name"],
                        description=issue["description"],
                        type=issue["type"],
                        priority=issue["priority"],
                        evidence=issue["evidence"],
                        timestamp=issue.get("timestamp", time.time()),
                        resolved=issue.get("resolved", False),
                        resolution=issue.get("resolution")
                    ) for issue in issues_data
                ]
                
                logger.debug(f"Загружено {len(self.ethical_issues)} этических проблем")
            else:
                # Создаем базовые этические проблемы
                self.ethical_issues = [
                    EthicalIssue(
                        name="нейроэстетика",
                        description="Отсутствие знаний о взаимодействии нейронауки и эстетики",
                        type="incomplete",
                        priority=0.6,
                        evidence=["Запросы о нейроэстетике не могут быть полноценно обработаны"]
                    ),
                    EthicalIssue(
                        name="этика_искусственного_интеллекта",
                        description="Противоречивые подходы к этике ИИ в разных источниках",
                        type="contradictory",
                        priority=0.7,
                        evidence=["Разные источники дают противоречивые рекомендации"]
                    ),
                    EthicalIssue(
                        name="автономия_человека",
                        description="Недостаток знаний о балансе автономии человека и ИИ",
                        type="missing",
                        priority=0.65,
                        evidence=["Частые запросы о контроле над ИИ"]
                    )
                ]
                
                # Сохраняем для будущего использования
                self._save_ethical_issues()
                
                logger.info("Созданы базовые этические проблемы")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки этических проблем: {e}")
            self.ethical_issues = []
    
    def _save_ethical_issues(self):
        """Сохраняет этические проблемы в файл."""
        try:
            issues_file = os.path.join(self.cache_dir, "ethical_issues.json")
            issues_data = [
                {
                    "name": issue.name,
                    "description": issue.description,
                    "type": issue.type,
                    "priority": issue.priority,
                    "evidence": issue.evidence,
                    "timestamp": issue.timestamp,
                    "resolved": issue.resolved,
                    "resolution": issue.resolution
                } for issue in self.ethical_issues
            ]
            
            with open(issues_file, 'w', encoding='utf-8') as f:
                json.dump(issues_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения этических проблем: {e}")
    
    def handle_situation(self, context: Dict[str, Any]) -> EthicalDecision:
        """
        Обрабатывает этическую ситуацию и принимает решение.
        
        Args:
            context: Контекст ситуации
            
        Returns:
            EthicalDecision: Принятое решение
        """
        if not context:
            return self._get_default_decision({}, "Пустой контекст")
            
        try:
            # Оцениваем риски
            assessments = self.risk_assessor.assess_risk(context)
            
            # Анализируем, требуется ли человеческое вмешательство
            requires_human_review = self._requires_human_review(assessments)
            
            # Генерируем решение
            decision, justification, alternatives = self._generate_decision(
                assessments, context, requires_human_review
            )
            
            # Создаем объект решения
            ethical_decision = EthicalDecision(
                decision=decision,
                confidence=self._calculate_confidence(assessments),
                justification=justification,
                alternatives=alternatives,
                assessment=assessments,
                requires_human_review=requires_human_review
            )
            
            # Сохраняем решение в кэш
            self._cache_solution(context, ethical_decision)
            
            # Обновляем статистику
            self._update_statistics(ethical_decision)
            
            logger.info(f"Принято этическое решение: {decision} (требует человеческого вмешательства: {requires_human_review})")
            return ethical_decision
            
        except Exception as e:
            logger.error(f"Ошибка обработки этической ситуации: {e}")
            return self._get_default_decision(context, str(e))
    
    def _requires_human_review(self, assessments: List[EthicalAssessment]) -> bool:
        """Определяет, требуется ли человеческое вмешательство."""
        if not assessments:
            return True
            
        for assessment in assessments:
            if assessment.violation_detected and assessment.severity == "high":
                return True
            if assessment.confidence < 0.4:
                return True
        
        # Проверяем, есть ли противоречия между принципами
        high_risk_principles = [
            a for a in assessments 
            if a.violation_detected and a.severity in ["high", "medium"]
        ]
        
        if len(high_risk_principles) > 1:
            # Проверяем, не противоречат ли принципы друг другу
            principle_weights = {
                a.principle_name: a.score * (1 - a.score) 
                for a in high_risk_principles
            }
            
            if len(principle_weights) > 1:
                max_weight = max(principle_weights.values())
                min_weight = min(principle_weights.values())
                if max_weight - min_weight > 0.3:
                    return True
        
        return False
    
    def _generate_decision(self, assessments: List[EthicalAssessment], 
                          context: Dict[str, Any], 
                          requires_human_review: bool) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Генерирует решение на основе оценок.
        
        Returns:
            Tuple[str, str, List[Dict[str, Any]]]: (решение, обоснование, альтернативы)
        """
        # Если требуется человеческое вмешательство
        if requires_human_review:
            return (
                "defer_to_human",
                "Этическая ситуация требует человеческого вмешательства из-за высокого риска или неопределенности",
                []
            )
        
        if not assessments:
            return (
                "proceed_with_caution",
                "Недостаточно данных для полной оценки, действие разрешено с осторожностью",
                []
            )
        
        # Определяем приоритетные принципы
        principle_scores = {
            a.principle_name: a.score * self._get_principle_weight(a.principle_name)
            for a in assessments
        }
        
        # Находим принцип с наименьшей оценкой (наибольший риск)
        highest_risk_principle = min(principle_scores, key=principle_scores.get)
        risk_score = principle_scores[highest_risk_principle]
        
        # Генерируем решение на основе риска
        if risk_score < 0.3:
            return (
                "reject_action",
                f"Отклонение действия из-за серьезного нарушения этического принципа {highest_risk_principle}",
                self._generate_alternatives(highest_risk_principle, context)
            )
        elif risk_score < 0.6:
            return (
                "modify_action",
                f"Модификация действия для уменьшения риска по принципу {highest_risk_principle}",
                self._generate_alternatives(highest_risk_principle, context)
            )
        else:
            return (
                "proceed",
                "Действие соответствует этическим стандартам",
                []
            )
    
    def _get_principle_weight(self, principle_name: str) -> float:
        """Возвращает вес принципа с учетом текущего контекста."""
        try:
            principle = self.principles_manager.get_principle_by_name(principle_name)
            if principle:
                return principle.weight
        except Exception as e:
            logger.warning(f"Ошибка получения веса принципа {principle_name}: {e}")
        return 1.0
    
    def _generate_alternatives(self, highest_risk_principle: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерирует альтернативные решения."""
        alternatives = []
        
        # Базовые альтернативы в зависимости от принципа
        if highest_risk_principle == "non_harm":
            alternatives.append({
                "action": "provide_warning",
                "description": "Предоставить предупреждение о потенциальном вреде",
                "expected_outcome": "Пользователь будет информирован о рисках"
            })
            alternatives.append({
                "action": "suggest_alternative",
                "description": "Предложить менее рискованный вариант",
                "expected_outcome": "Снижение потенциального вреда"
            })
        
        elif highest_risk_principle == "autonomy":
            alternatives.append({
                "action": "provide_more_information",
                "description": "Предоставить больше информации для принятия решения",
                "expected_outcome": "Повышение информированности пользователя"
            })
            alternatives.append({
                "action": "offer_choice",
                "description": "Предложить несколько вариантов действий",
                "expected_outcome": "Увеличение автономии пользователя"
            })
        
        elif highest_risk_principle == "justice":
            alternatives.append({
                "action": "check_for_bias",
                "description": "Проверить ответ на наличие предвзятости",
                "expected_outcome": "Устранение потенциальной дискриминации"
            })
            alternatives.append({
                "action": "provide_context",
                "description": "Добавить контекст для справедливой оценки",
                "expected_outcome": "Более сбалансированный ответ"
            })
        
        # Дополнительные альтернативы из кэша
        if self.solutions_cache.get("solutions"):
            similar_solutions = [
                s for s in self.solutions_cache["solutions"]
                if highest_risk_principle in s.get("relevant_principles", [])
            ]
            
            for solution in similar_solutions[:2]:
                alternatives.append({
                    "action": solution["decision"],
                    "description": f"Адаптировано из предыдущего решения: {solution['justification']}",
                    "expected_outcome": "Аналогичная ситуация была успешно разрешена ранее"
                })
        
        return alternatives
    
    def _calculate_confidence(self, assessments: List[EthicalAssessment]) -> float:
        """Рассчитывает общую уверенность в решении."""
        if not assessments:
            return 0.5
        
        # Учитываем уверенность в оценках и их количество
        total_confidence = sum(a.confidence for a in assessments)
        avg_confidence = total_confidence / len(assessments)
        
        # Учитываем разногласия между оценками
        scores = [a.score for a in assessments]
        if len(scores) > 1 and np:
            variance = np.var(scores)
            # Чем выше дисперсия, тем ниже уверенность
            confidence = avg_confidence * (1 - min(0.5, variance))
        else:
            confidence = avg_confidence
        
        return max(0.1, min(1.0, confidence))
    
    def _cache_solution(self, context: Dict[str, Any], decision: EthicalDecision):
        """Кэширует решение для будущих ссылок."""
        try:
            # Создаем ключ для кэша
            cache_key = self._generate_cache_key(context)
            
            # Формируем запись
            solution_entry = {
                "key": cache_key,
                "timestamp": time.time(),
                "context_summary": self._summarize_context(context),
                "relevant_principles": [a.principle_name for a in decision.assessment],
                "decision": decision.decision,
                "justification": decision.justification,
                "confidence": decision.confidence,
                "requires_human_review": decision.requires_human_review
            }
            
            # Добавляем в кэш
            self.solutions_cache["solutions"].insert(0, solution_entry)
            
            # Ограничиваем размер кэша
            if len(self.solutions_cache["solutions"]) > 100:
                self.solutions_cache["solutions"] = self.solutions_cache["solutions"][:100]
            
            # Сохраняем
            self._save_cache()
            
        except Exception as e:
            logger.error(f"Ошибка кэширования решения: {e}")
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Генерирует уникальный ключ для кэша на основе контекста."""
        # Создаем строку для хеширования
        context_str = ""
        if "query" in context:
            context_str += str(context["query"]) + " "
        if "response" in context:
            context_str += str(context["response"])
        
        # Генерируем хеш
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Создает краткое описание контекста."""
        summary = []
        
        if "query" in context:
            query_str = str(context["query"])
            summary.append(f"Запрос: {query_str[:100]}{'...' if len(query_str) > 100 else ''}")
        
        if "response" in context:
            response_str = str(context["response"])
            summary.append(f"Ответ: {response_str[:100]}{'...' if len(response_str) > 100 else ''}")
        
        if "user_profile" in context and isinstance(context["user_profile"], dict):
            if "preferences" in context["user_profile"]:
                prefs = context["user_profile"]["preferences"]
                if isinstance(prefs, list):
                    summary.append(f"Преференции: {', '.join(str(p) for p in prefs[:3])}")
        
        return " | ".join(summary)
    
    def _update_statistics(self, decision: EthicalDecision):
        """Обновляет статистику по этическим решениям."""
        # Здесь можно добавить логику обновления статистики
        pass
    
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
    
    def get_ethical_issues(self, limit: int = 10, min_priority: float = 0.5) -> List[EthicalIssue]:
        """
        Возвращает список этических проблем.
        
        Args:
            limit: Максимальное количество проблем
            min_priority: Минимальный приоритет
            
        Returns:
            List[EthicalIssue]: Список этических проблем
        """
        # Сортируем по приоритету и фильтруем
        issues = [
            issue for issue in self.ethical_issues 
            if not issue.resolved and issue.priority >= min_priority
        ]
        issues.sort(key=lambda x: x.priority, reverse=True)
        
        return issues[:limit]
    
    def add_ethical_issue(self, issue: EthicalIssue):
        """
        Добавляет новую этическую проблему.
        
        Args:
            issue: Этическая проблема
        """
        if not issue:
            return
            
        self.ethical_issues.append(issue)
        self._save_ethical_issues()
        logger.info(f"Добавлена новая этическая проблема: {issue.name} (приоритет: {issue.priority})")
    
    def resolve_ethical_issue(self, issue_name: str, resolution: Dict[str, Any]):
        """
        Помечает этическую проблему как решенную.
        
        Args:
            issue_name: Название проблемы
            resolution: Описание решения
        """
        if not issue_name:
            return
            
        for issue in self.ethical_issues:
            if issue.name == issue_name and not issue.resolved:
                issue.resolved = True
                issue.resolution = resolution
                self._save_ethical_issues()
                logger.info(f"Этическая проблема решена: {issue_name}")
                return
    
    def get_situation_dashboard_data(self) -> Dict[str, Any]:
        """
        Возвращает данные для дашборда этических ситуаций.
        
        Returns:
            Dict[str, Any]: Данные для дашборда
        """
        try:
            # Получаем данные о решениях
            recent_solutions = self.solutions_cache.get("solutions", [])[:10]
            
            # Анализируем распределение решений
            decision_counts = defaultdict(int)
            for solution in recent_solutions:
                decision_counts[solution.get("decision", "unknown")] += 1
            
            # Получаем данные о проблемах
            open_issues = [i for i in self.ethical_issues if not i.resolved]
            high_priority_issues = [i for i in open_issues if i.priority >= 0.7]
            
            # Получаем данные о рисках
            try:
                risk_data = self.risk_assessor.get_risk_dashboard_data()
            except Exception as e:
                logger.warning(f"Ошибка получения данных о рисках: {e}")
                risk_data = {}
            
            return {
                "total_solutions": len(self.solutions_cache.get("solutions", [])),
                "recent_solutions": recent_solutions,
                "decision_counts": dict(decision_counts),
                "open_issues_count": len(open_issues),
                "high_priority_issues_count": len(high_priority_issues),
                "high_priority_issues": [
                    {
                        "name": i.name,
                        "description": i.description,
                        "priority": i.priority,
                        "type": i.type
                    } for i in high_priority_issues[:5]
                ],
                "risk_data": risk_data,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка получения данных дашборда: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def generate_situation_visualization(self, view_type: str = "issues") -> str:
        """
        Генерирует визуализацию данных о ситуациях.
        
        Args:
            view_type: Тип визуализации
            
        Returns:
            str: Изображение в формате base64
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib недоступен для визуализации")
            return ""
            
        try:
            # Получаем данные для визуализации
            dashboard_data = self.get_situation_dashboard_data()
            
            if "error" in dashboard_data:
                return ""
            
            # Создаем фигуру
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            if view_type == "issues":
                # Визуализация этических проблем
                issues = [i["name"] for i in dashboard_data.get("high_priority_issues", [])]
                priorities = [i["priority"] for i in dashboard_data.get("high_priority_issues", [])]
                
                if issues and priorities:
                    y_pos = range(len(issues))
                    ax.barh(y_pos, priorities, align='center', color='salmon')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(issues)
                    ax.invert_yaxis()
                    ax.set_xlabel('Приоритет (0-1)')
                    ax.set_title('Высокоприоритетные этические проблемы')
                    ax.set_xlim(0, 1)
                else:
                    ax.text(0.5, 0.5, 'Нет данных для отображения', 
                           ha='center', va='center', transform=ax.transAxes)
            
            elif view_type == "decisions":
                # Визуализация принятых решений
                decision_counts = dashboard_data.get("decision_counts", {})
                if decision_counts:
                    decisions = list(decision_counts.keys())
                    counts = list(decision_counts.values())
                    
                    ax.pie(counts, labels=decisions, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    ax.set_title('Распределение этических решений')
                else:
                    ax.text(0.5, 0.5, 'Нет данных для отображения', 
                           ha='center', va='center', transform=ax.transAxes)
            
            # Сохраняем в буфер
            buf = BytesIO()
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.print_png(buf)
            
            # Преобразуем в base64
            buf.seek(0)
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            logger.error(f"Ошибка генерации визуализации ситуаций: {e}")
            return ""
    
    def export_ethics_data(self, file_path: str) -> bool:
        """
        Экспортирует данные этической рамки в файл.
        
        Args:
            file_path: Путь к файлу для экспорта
            
        Returns:
            bool: Успешно ли экспортировано
        """
        if not file_path:
            return False
            
        try:
            # Собираем данные для экспорта
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "format_version": "1.0"
                },
                "solutions": self.solutions_cache.get("solutions", []),
                "reviews": self.review_cache.get("reviews", []),
                "ethical_issues": [
                    {
                        "name": issue.name,
                        "description": issue.description,
                        "type": issue.type,
                        "priority": issue.priority,
                        "evidence": issue.evidence,
                        "timestamp": issue.timestamp,
                        "resolved": issue.resolved,
                        "resolution": issue.resolution
                    } for issue in self.ethical_issues
                ],
                "system_health": self.get_system_health()
            }
            
            # Добавляем принципы если доступны
            try:
                principles = self.principles_manager.get_all_principles()
                export_data["principles"] = [
                    {
                        "id": pid,
                        **principle.__dict__
                    } for pid, principle in principles.items()
                ]
            except Exception as e:
                logger.warning(f"Ошибка экспорта принципов: {e}")
                export_data["principles"] = []
            
            # Сохраняем в JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Данные этической рамки экспортированы в {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта данных этической рамки: {e}")
            return False
    
    def import_ethics_data(self, file_path: str) -> bool:
        """
        Импортирует данные этической рамки из файла.
        
        Args:
            file_path: Путь к файлу для импорта
            
        Returns:
            bool: Успешно ли импортировано
        """
        if not file_path or not os.path.exists(file_path):
            return False
            
        try:
            # Загружаем данные из JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Импортируем принципы
            if "principles" in data:
                try:
                    for principle_data in data["principles"]:
                        principle = EthicalPrinciple(
                            name=principle_data["name"],
                            description=principle_data["description"],
                            weight=principle_data["weight"],
                            threshold=principle_data["threshold"],
                            category=principle_data["category"],
                            last_updated=principle_data["last_updated"],
                            active=principle_data["active"]
                        )
                        self.principles_manager.add_principle(principle)
                except Exception as e:
                    logger.error(f"Ошибка импорта принципов: {e}")
            
            # Импортируем решения
            if "solutions" in data:
                self.solutions_cache["solutions"] = data["solutions"]
                self._save_cache()
            
            # Импортируем обзоры
            if "reviews" in data:
                self.review_cache["reviews"] = data["reviews"]
                self._save_cache()
            
            # Импортируем этические проблемы
            if "ethical_issues" in data:
                self.ethical_issues = [
                    EthicalIssue(
                        name=issue["name"],
                        description=issue["description"],
                        type=issue["type"],
                        priority=issue["priority"],
                        evidence=issue["evidence"],
                        timestamp=issue.get("timestamp", time.time()),
                        resolved=issue.get("resolved", False),
                        resolution=issue.get("resolution")
                    ) for issue in data["ethical_issues"]
                ]
                self._save_ethical_issues()
            
            logger.info(f"Данные этической рамки импортированы из {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка импорта данных этической рамки: {e}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье системы этической рамки.
        
        Returns:
            Dict: Отчет о здоровье
        """
        try:
            # Получаем данные о принципах
            try:
                principles = self.principles_manager.get_all_principles()
                total_principles = len(principles)
                
                # Анализируем оценки
                low_compliance_count = 0
                for principle_id, principle in principles.items():
                    try:
                        history = self.principles_manager.get_assessment_history(principle_id, days=7)
                        if history:
                            avg_score = sum(item["score"] for item in history) / len(history)
                            if avg_score < principle.threshold * 0.8:
                                low_compliance_count += 1
                    except Exception as e:
                        logger.warning(f"Ошибка анализа принципа {principle_id}: {e}")
            except Exception as e:
                logger.warning(f"Ошибка получения данных о принципах: {e}")
                total_principles = 0
                low_compliance_count = 0
            
            # Рассчитываем общий показатель здоровья
            health_score = 100.0
            
            # Учитываем соблюдение принципов
            if low_compliance_count > 0:
                health_score -= min(40, low_compliance_count * 15)
            
            # Учитываем количество нерешенных проблем
            open_issues = len([i for i in self.ethical_issues if not i.resolved])
            if open_issues > 5:
                health_score -= min(30, (open_issues - 2) * 5)
            
            # Формируем рекомендации
            recommendations = []
            if low_compliance_count > 0:
                recommendations.append(
                    f"Обнаружено {low_compliance_count} принципов с низким уровнем соблюдения. "
                    "Рассмотрите возможность улучшения обработки соответствующих сценариев."
                )
            
            if open_issues > 0:
                recommendations.append(
                    f"Есть {open_issues} нерешенных этических проблем. "
                    "Рекомендуется сосредоточиться на их решении."
                )
            
            if not recommendations:
                recommendations.append(
                    "Этическая рамка работает стабильно. Продолжайте мониторинг для "
                    "раннего выявления потенциальных проблем."
                )
            
            return {
                "health_score": max(0, min(100, health_score)),
                "total_principles": total_principles,
                "low_compliance_count": low_compliance_count,
                "open_issues_count": open_issues,
                "recommendations": recommendations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения отчета о здоровье системы: {e}")
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def close(self):
        """Закрывает обработчик этических ситуаций и освобождает ресурсы."""
        logger.info("Закрытие обработчика этических ситуаций...")
        try:
            self._save_cache()
            self._save_ethical_issues()
        except Exception as e:
            logger.error(f"Ошибка при закрытии: {e}")
        logger.info("Обработчик этических ситуаций закрыт")