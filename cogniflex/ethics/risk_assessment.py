"""Модуль оценки этических рисков для CogniFlex"""
import os
import logging
import json
import re
import time
import base64
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from io import BytesIO
from cogniflex.ethics.ethics_framework import EthicalAssessment, EthicalPrinciple
from .principles_manager import PrinciplesManager

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # Scikit-learn не установлен, используем базовую функциональность
    np = None
    TfidfVectorizer = None
    cosine_similarity = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    # Matplotlib не установлен, визуализация будет недоступна
    Figure = None
    FigureCanvasAgg = None
    plt = None

logger = logging.getLogger("cogniflex.ethics.risk")

class RiskAssessor:
    """Оценивает этические риски в различных сценариях."""
    
    def __init__(self, principles_manager: PrinciplesManager, brain=None):
        """
        Инициализирует оценщик этических рисков.
        
        Args:
            principles_manager: Менеджер этических принципов
            brain: Ссылка на ядро CogniFlex
        """
        self.principles_manager = principles_manager
        self.brain = brain
        self.vectorizer = TfidfVectorizer() if TfidfVectorizer else None
        
        # Загружаем эталонные сценарии
        self.load_reference_scenarios()
        
        logger.info("Оценщик этических рисков инициализирован")
    
    def load_reference_scenarios(self):
        """Загружает эталонные сценарии для оценки рисков."""
        try:
            # Путь к файлу эталонных сценариев
            scenarios_file = os.path.join(os.path.dirname(__file__), "reference_scenarios.json")
            
            if os.path.exists(scenarios_file):
                with open(scenarios_file, 'r', encoding='utf-8') as f:
                    self.reference_scenarios = json.load(f)
                logger.debug(f"Загружено {len(self.reference_scenarios)} эталонных сценариев")
            else:
                # Создаем базовые эталонные сценарии
                self.reference_scenarios = [
                    {
                        "id": "scenario_001",
                        "description": "Распространение ложной информации",
                        "risks": [
                            {"principle": "transparency", "score": 0.95},
                            {"principle": "non_harm", "score": 0.9},
                            {"principle": "accountability", "score": 0.85}
                        ],
                        "keywords": ["ложь", "дезинформация", "неправда", "обман"]
                    },
                    {
                        "id": "scenario_002",
                        "description": "Принуждение к действиям",
                        "risks": [
                            {"principle": "autonomy", "score": 0.95},
                            {"principle": "non_harm", "score": 0.8},
                            {"principle": "justice", "score": 0.7}
                        ],
                        "keywords": ["принуждать", "заставлять", "угрожать", "шантаж"]
                    },
                    {
                        "id": "scenario_003",
                        "description": "Дискриминация на основе характеристик",
                        "risks": [
                            {"principle": "justice", "score": 0.95},
                            {"principle": "non_harm", "score": 0.9},
                            {"principle": "beneficence", "score": 0.75}
                        ],
                        "keywords": ["раса", "пол", "возраст", "национальность", "дискриминация"]
                    },
                    {
                        "id": "scenario_004",
                        "description": "Нарушение конфиденциальности",
                        "risks": [
                            {"principle": "autonomy", "score": 0.9},
                            {"principle": "non_harm", "score": 0.85},
                            {"principle": "transparency", "score": 0.7}
                        ],
                        "keywords": ["конфиденциальность", "приватность", "данные", "секрет"]
                    }
                ]
                
                # Сохраняем для будущего использования
                with open(scenarios_file, 'w', encoding='utf-8') as f:
                    json.dump(self.reference_scenarios, f, ensure_ascii=False, indent=2)
                
                logger.info("Созданы базовые эталонные сценарии")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки эталонных сценариев: {e}")
            # Создаем минимальный набор эталонных сценариев
            self.reference_scenarios = [
                {
                    "id": "basic_scenario_001",
                    "description": "Общая этическая ситуация",
                    "risks": [
                        {"principle": "non_harm", "score": 0.5},
                        {"principle": "transparency", "score": 0.5}
                    ],
                    "keywords": []
                }
            ]
    
    def assess_risk(self, context: Dict[str, Any]) -> List[EthicalAssessment]:
        """
        Оценивает этические риски в заданном контексте.
        
        Args:
            context: Контекст для оценки рисков
            
        Returns:
            List[EthicalAssessment]: Список этических оценок
        """
        try:
            # Извлекаем текст для анализа
            text = self._extract_text_from_context(context)
            
            if not text:
                logger.warning("Пустой контекст для оценки этических рисков")
                return []
            
            # Определяем сценарий
            scenario = self._identify_scenario(text)
            
            # Оцениваем риски по каждому принципу
            assessments = []
            principles = self.principles_manager.get_all_principles()
            
            for principle_id, principle in principles.items():
                # Оцениваем риск для этого принципа
                score, confidence, explanation = self._assess_principle_risk(
                    principle, text, scenario
                )
                
                # Определяем, есть ли нарушение
                violation_detected = score < principle.threshold * 0.9
                severity = "low"
                if violation_detected:
                    if score < principle.threshold * 0.7:
                        severity = "high"
                    elif score < principle.threshold * 0.85:
                        severity = "medium"
                    else:
                        severity = "low"
                
                assessment = EthicalAssessment(
                    principle_name=principle.name,
                    score=score,
                    confidence=confidence,
                    explanation=explanation,
                    context=context,
                    violation_detected=violation_detected,
                    severity=severity
                )
                assessments.append(assessment)
                
                # Сохраняем оценку
                self.principles_manager.record_assessment(
                    principle_id,
                    score,
                    confidence,
                    context
                )
            
            return assessments
            
        except Exception as e:
            logger.error(f"Ошибка оценки этических рисков: {e}")
            return []
    
    def _extract_text_from_context(self, context: Dict[str, Any]) -> str:
        """Извлекает текст из контекста для анализа."""
        text_parts = []
        
        # Извлекаем текст из различных полей контекста
        if "query" in context:
            text_parts.append(context["query"])
        
        if "response" in context:
            text_parts.append(context["response"])
        
        if "conversation_history" in context:
            for turn in context["conversation_history"][-3:]:  # Последние 3 обмена
                if "user" in turn:
                    text_parts.append(turn["user"])
                if "system" in turn:
                    text_parts.append(turn["system"])
        
        if "content" in context:
            text_parts.append(context["content"])
        
        return " ".join(text_parts)
    
    def _identify_scenario(self, text: str) -> Optional[Dict[str, Any]]:
        """Определяет, к какому эталонному сценарию ближе всего данный текст."""
        try:
            # Подготавливаем текст
            text = text.lower()
            
            # Ищем совпадения по ключевым словам
            best_match = None
            max_matches = 0
            
            for scenario in self.reference_scenarios:
                matches = sum(1 for keyword in scenario["keywords"] if keyword in text)
                if matches > max_matches:
                    max_matches = matches
                    best_match = scenario
            
            # Если нет явного совпадения, используем TF-IDF для определения близости
            if max_matches < 2 and len(self.reference_scenarios) > 1:
                # Создаем корпус из описаний сценариев
                scenario_descriptions = [s["description"] for s in self.reference_scenarios]
                scenario_descriptions.append(text)
                
                # Векторизуем
                tfidf_matrix = self.vectorizer.fit_transform(scenario_descriptions)
                
                # Вычисляем косинусное сходство
                similarities = cosine_similarity(
                    tfidf_matrix[-1],  # Наш текст
                    tfidf_matrix[:-1]  # Описания сценариев
                )[0]
                
                # Находим наиболее похожий сценарий
                max_index = np.argmax(similarities)
                if similarities[max_index] > 0.3:  # Порог сходства
                    best_match = self.reference_scenarios[max_index]
            
            return best_match
            
        except Exception as e:
            logger.error(f"Ошибка определения сценария: {e}")
            return None
    
    def _assess_principle_risk(self, principle: EthicalPrinciple, text: str, 
                             scenario: Optional[Dict[str, Any]]) -> Tuple[float, float, str]:
        """
        Оценивает риск нарушения конкретного этического принципа.
        
        Returns:
            Tuple[float, float, str]: (оценка, уверенность, объяснение)
        """
        # Базовая оценка на основе ключевых слов
        risk_score = 0.7
        confidence = 0.6
        explanation = f"Базовая оценка для принципа {principle.name}"
        
        try:
            # Анализируем текст на предмет ключевых слов, связанных с принципом
            principle_keywords = self._get_principle_keywords(principle.name)
            keyword_matches = sum(1 for keyword in principle_keywords if keyword in text.lower())
            
            # Корректируем оценку на основе совпадений
            if keyword_matches > 0:
                risk_score -= 0.1 * keyword_matches
                confidence = min(1.0, confidence + 0.1 * keyword_matches)
            
            # Учитываем сценарий, если он определен
            if scenario:
                for risk in scenario["risks"]:
                    if risk["principle"] == principle.name:
                        # Снижаем оценку на величину риска
                        risk_score *= (1 - risk["score"] * 0.5)
                        confidence = min(1.0, confidence + 0.2)
                        explanation = f"Снижение оценки из-за сценария: {scenario['description']}"
                        break
            
            # Дополнительный анализ с использованием NLP, если доступен
            if self.brain and hasattr(self.brain, 'nlp_processor'):
                try:
                    # Анализируем тональность и содержание
                    analysis = self.brain.nlp_processor.process_text(text)
                    
                    # Проверяем на наличие противоречий
                    if analysis.contradictions:
                        risk_score -= 0.1
                        explanation += ", обнаружены противоречия"
                    
                    # Проверяем связность аргументации
                    if analysis.coherence_score < 0.5:
                        risk_score -= 0.05
                        explanation += ", низкая связность аргументации"
                    
                    confidence = min(1.0, confidence + 0.15)
                except Exception as e:
                    logger.debug(f"Ошибка при NLP-анализе для оценки риска: {e}")
            
            # Ограничиваем оценку в пределах 0-1
            risk_score = max(0.0, min(1.0, risk_score))
            
            return risk_score, confidence, explanation
            
        except Exception as e:
            logger.error(f"Ошибка оценки риска для принципа {principle.name}: {e}")
            return risk_score, confidence, explanation
    
    def _get_principle_keywords(self, principle_name: str) -> List[str]:
        """Возвращает ключевые слова для конкретного этического принципа."""
        keywords = {
            "non_harm": ["вред", "опасность", "риск", "травма", "ущерб", "боль"],
            "beneficence": ["польза", "выгода", "улучшение", "помощь", "поддержка"],
            "autonomy": ["автономия", "свобода", "выбор", "решение", "контроль", "самостоятельность"],
            "justice": ["справедливость", "равенство", "дискриминация", "предвзятость", "несправедливость"],
            "transparency": ["прозрачность", "ясность", "открытость", "скрытый", "тайный", "непонятный"],
            "accountability": ["ответственность", "отчетность", "виновный", "обязанность", "подотчетность"]
        }
        
        return keywords.get(principle_name, [])
    
    def analyze_ethical_gaps(self, assessments: List[EthicalAssessment]) -> List[Dict[str, Any]]:
        """
        Анализирует пробелы в этических знаниях на основе оценок.
        
        Args:
            assessments: Список этических оценок
            
        Returns:
            List[Dict[str, Any]]: Выявленные пробелы
        """
        gaps = []
        
        for assessment in assessments:
            # Если оценка ниже порога и уверенность высока
            if assessment.score < assessment.context.get("threshold", 0.8) * 0.85 and assessment.confidence > 0.7:
                gap_type = "incomplete"
                if "contradiction" in assessment.explanation.lower():
                    gap_type = "contradictory"
                
                gaps.append({
                    "name": f"этика_{assessment.principle_name}",
                    "description": f"Пробел в знаниях по принципу {assessment.principle_name}",
                    "type": gap_type,
                    "priority": 1.0 - assessment.score,
                    "evidence": [assessment.explanation]
                })
        
        return gaps
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """
        Возвращает данные для дашборда оценки этических рисков.
        
        Returns:
            Dict[str, Any]: Данные для дашборда
        """
        # Получаем данные о принципах
        principles_data = self.principles_manager.get_principles_dashboard_data()
        
        # Анализируем текущие риски
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        for principle_id, principle in self.principles_manager.get_all_principles().items():
            history = self.principles_manager.get_assessment_history(principle_id, days=1)
            if history:
                avg_score = sum(item["score"] for item in history) / len(history)
                if avg_score < principle.threshold * 0.7:
                    high_risk_count += 1
                elif avg_score < principle.threshold * 0.85:
                    medium_risk_count += 1
                else:
                    low_risk_count += 1
        
        # Получаем последние оценки
        recent_assessments = []
        for principle_id, principle in self.principles_manager.get_all_principles().items():
            history = self.principles_manager.get_assessment_history(principle_id, days=1)
            if history:
                recent_assessments.append({
                    "principle": principle.name,
                    "score": history[0]["score"],
                    "timestamp": history[0]["timestamp"]
                })
        
        # Сортируем по времени
        recent_assessments.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_assessments = recent_assessments[:10]  # Берем последние 10
        
        return {
            "principles_count": principles_data["principles_count"],
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "low_risk_count": low_risk_count,
            "recent_assessments": recent_assessments,
            "problematic_principles": principles_data["problematic_principles"],
            "timestamp": time.time()
        }
    
    def generate_risk_visualization(self, view_type: str = "overview") -> str:
        """
        Генерирует визуализацию данных о рисках.
        
        Args:
            view_type: Тип визуализации
            
        Returns:
            str: Изображение в формате base64
        """
        try:
            # Проверяем доступность matplotlib
            if Figure is None or FigureCanvasAgg is None:
                logger.warning("Модуль matplotlib не установлен, визуализация недоступна")
                return ""
            
            # Получаем данные для визуализации
            dashboard_data = self.get_risk_dashboard_data()
            
            # Создаем фигуру
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            if view_type == "overview":
                # Обзор рисков
                labels = ['Высокий', 'Средний', 'Низкий']
                sizes = [
                    dashboard_data["high_risk_count"],
                    dashboard_data["medium_risk_count"],
                    dashboard_data["low_risk_count"]
                ]
                colors = ['red', 'orange', 'green']
                
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90, shadow=True)
                ax.axis('equal')
                ax.set_title('Распределение этических рисков')
            
            elif view_type == "principles":
                # Риски по принципам
                principles = [p["name"] for p in dashboard_data["problematic_principles"]]
                scores = [p["current_compliance"] for p in dashboard_data["problematic_principles"]]
                
                y_pos = range(len(principles))
                ax.barh(y_pos, scores, align='center', color='salmon')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(principles)
                ax.invert_yaxis()
                ax.set_xlabel('Соблюдение (0-1)')
                ax.set_title('Проблемные этические принципы')
            
            elif view_type == "trends":
                # Тренды рисков
                principles_data = self.principles_manager.get_principles_dashboard_data()
                trends = principles_data["trends"]
                
                # Группируем данные по принципам
                principle_trends = defaultdict(list)
                dates = set()
                
                for trend in trends:
                    principle_trends[trend["category"]].append((trend["date"], trend["compliance"]))
                    dates.add(trend["date"])
                
                dates = sorted(dates)
                for category, values in principle_trends.items():
                    values_dict = {date: compliance for date, compliance in values}
                    compliance = [values_dict.get(date, 0) for date in dates]
                    ax.plot(dates, compliance, marker='o', label=category)
                
                ax.set_xlabel('Дата')
                ax.set_ylabel('Соблюдение (0-1)')
                ax.set_title('Тренды соблюдения этических принципов')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
            
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
            logger.error(f"Ошибка генерации визуализации рисков: {e}")
            return ""