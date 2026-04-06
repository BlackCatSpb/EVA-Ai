"""
Модуль этической рамки для ЕВА - обеспечение этических стандартов в работе системы
(Рефактор: основной модуль импортирует из разделённых модулей)
"""
from .framework_core import EthicsFramework
from .framework_principles import EthicalPrinciple, EthicsPrinciplesMixin
from .framework_checks import EthicalDecision, EthicalAssessment, EthicsAnalysisResult, EthicsChecksMixin
from .framework_violations import EthicsViolationsMixin
from .situations_db import EthicalIssue

__all__ = [
    'EthicsFramework',
    'EthicalPrinciple',
    'EthicalDecision',
    'EthicalAssessment',
    'EthicsAnalysisResult',
    'EthicalIssue',
    'EthicsPrinciplesMixin',
    'EthicsChecksMixin',
    'EthicsViolationsMixin',
]
