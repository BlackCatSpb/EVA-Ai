"""Модуль обработки этических ситуаций для ЕВА (Рефактор: основной модуль импортирует из разделённых модулей)"""
import os
import logging
import json
import time
import hashlib
import base64
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from io import BytesIO

try:
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None

from eva_ai.ethics.ethics_framework import EthicalDecision as FrameworkEthicalDecision, EthicalIssue as FrameworkEthicalIssue
from eva_ai.ethics.principles_manager import PrinciplesManager
from eva_ai.ethics.risk_assessment import RiskAssessor

from .situations_db import SituationsDBMixin, EthicalIssue
from .situations_scenarios import SituationsScenariosMixin, EthicalAssessment, EthicalPrinciple, EthicalDecision
from .situations_evaluation import SituationsEvaluationMixin

logger = logging.getLogger("eva_ai.ethics.situations")

class EthicalSituationHandler(SituationsDBMixin, SituationsScenariosMixin, SituationsEvaluationMixin):
    """Обрабатывает этически сложные ситуации и принимает решения."""
    
    def __init__(self, principles_manager: PrinciplesManager = None, risk_assessor: RiskAssessor = None, brain=None):
        """
        Инициализирует обработчик этических ситуаций.
        
        Args:
            principles_manager: Менеджер этических принципов
            risk_assessor: Оценщик этических рисков
            brain: Ссылка на ядро ЕВА
        """
        self.principles_manager = principles_manager or PrinciplesManager()
        self.risk_assessor = risk_assessor or RiskAssessor()
        self.brain = brain
        
        try:
            self.cache_dir = os.path.join(os.path.dirname(__file__), "eva_ethics_cache")
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

__all__ = [
    'EthicalSituationHandler',
    'EthicalIssue',
    'EthicalAssessment',
    'EthicalPrinciple',
    'EthicalDecision',
    'SituationsDBMixin',
    'SituationsScenariosMixin',
    'SituationsEvaluationMixin',
]
