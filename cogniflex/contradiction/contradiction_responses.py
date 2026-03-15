"""Модуль для генерации сбалансированных ответов на противоречия"""
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("cogniflex.contradiction.responses")

class ContradictionResponseGenerator:
    """Генератор сбалансированных ответов на противоречия."""
    
    def __init__(self):
        """Инициализирует генератор ответов."""
        self.supported_languages = ["ru", "en"]
        logger.info("Генератор ответов на противоречия инициализирован")
    
    def generate_balanced_response(self, contradiction: Dict[str, Any], language: str = "ru") -> str:
        """
        Генерирует сбалансированный ответ на основе противоречия.
        
        Args:
            contradiction: Экземпляр противоречия
            language: Язык ответа
            
        Returns:
            str: Сбалансированный ответ
        """
        if language not in self.supported_languages:
            logger.warning(f"Неподдерживаемый язык '{language}'. Используется русский по умолчанию.")
            language = "ru"
        
        # Определяем тип противоречия
        c_type = self._get_contradiction_type(contradiction)
        
        # Определяем серьезность
        severity = self._get_severity(contradiction)
        
        # Генерируем ответ в зависимости от типа
        if c_type == "numeric_conflict":
            return self._format_numeric_conflict_response(contradiction, severity, language)
        elif c_type == "boolean_conflict":
            return self._format_boolean_conflict_response(contradiction, severity, language)
        elif c_type == "response_conflict":
            return self._format_response_conflict_response(contradiction, severity, language)
        else:
            return self._format_general_conflict_response(contradiction, severity, language)
    
    def _get_contradiction_type(self, contradiction: Dict[str, Any]) -> str:
        """Определяет тип противоречия."""
        if "type" in contradiction.get("metadata", {}):
            return contradiction["metadata"]["type"]
        
        # Проверяем числовые противоречия
        if self._is_numeric_conflict(contradiction):
            return "numeric_conflict"
        
        # Проверяем булевы противоречия
        if self._is_boolean_conflict(contradiction):
            return "boolean_conflict"
        
        # Проверяем противоречия в ответах
        if self._is_response_conflict(contradiction):
            return "response_conflict"
        
        # Проверяем типы отношений
        if self._is_exclusivity_conflict(contradiction):
            return "exclusivity_conflict"
        
        if self._is_hierarchy_conflict(contradiction):
            return "hierarchy_conflict"
        
        return "general_conflict"
    
    def _is_numeric_conflict(self, contradiction: Dict[str, Any]) -> bool:
        """Проверяет, является ли противоречие числовым."""
        metadata = contradiction.get("metadata", {})
        conflicting_facts = metadata.get("conflicting_facts", [])
        
        if len(conflicting_facts) < 2:
            return False
        
        # Проверяем, что оба значения являются числами
        value1 = conflicting_facts[0].get("value")
        value2 = conflicting_facts[1].get("value")
        
        return (isinstance(value1, (int, float)) and isinstance(value2, (int, float)))
    
    def _is_boolean_conflict(self, contradiction: Dict[str, Any]) -> bool:
        """Проверяет, является ли противоречие булевым."""
        metadata = contradiction.get("metadata", {})
        conflicting_facts = metadata.get("conflicting_facts", [])
        
        if len(conflicting_facts) < 2:
            return False
        
        # Проверяем, что оба значения являются булевыми
        value1 = conflicting_facts[0].get("value")
        value2 = conflicting_facts[1].get("value")
        
        return (isinstance(value1, bool) and isinstance(value2, bool))
    
    def _is_response_conflict(self, contradiction: Dict[str, Any]) -> bool:
        """Проверяет, является ли противоречие противоречием в ответах."""
        metadata = contradiction.get("metadata", {})
        return "response" in metadata.get("relation_type", "")
    
    def _is_exclusivity_conflict(self, contradiction: Dict[str, Any]) -> bool:
        """Проверяет, является ли противоречие противоречием эксклюзивности."""
        metadata = contradiction.get("metadata", {})
        relation_type = metadata.get("relation_type", "")
        return relation_type.startswith("only_") or relation_type.startswith("not_only_")
    
    def _is_hierarchy_conflict(self, contradiction: Dict[str, Any]) -> bool:
        """Проверяет, является ли противоречие иерархическим."""
        metadata = contradiction.get("metadata", {})
        return metadata.get("relation_type") in ["is_a", "part_of", "member_of"]
    
    def _get_severity(self, contradiction: Dict[str, Any]) -> str:
        """Определяет серьезность противоречия."""
        divergence = contradiction.get("divergence", 0.0)
        
        if divergence >= 0.7:
            return "high"
        elif divergence >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _format_numeric_conflict_response(self, contradiction: Dict[str, Any], severity: str, language: str) -> str:
        """
        Форматирует ответ для числового противоречия.
        
        Args:
            contradiction: Противоречие
            severity: Серьезность
            language: Язык
            
        Returns:
            str: Отформатированный ответ
        """
        if language == "ru":
            return self._format_numeric_conflict_response_ru(contradiction, severity)
        else:
            return self._format_numeric_conflict_response_en(contradiction, severity)
    
    def _format_boolean_conflict_response(self, contradiction: Dict[str, Any], severity: str, language: str) -> str:
        """
        Форматирует ответ для булева противоречия.
        
        Args:
            contradiction: Противоречие
            severity: Серьезность
            language: Язык
            
        Returns:
            str: Отформатированный ответ
        """
        if language == "ru":
            return self._format_boolean_conflict_response_ru(contradiction, severity)
        else:
            return self._format_boolean_conflict_response_en(contradiction, severity)
    
    def _format_response_conflict_response(self, contradiction: Dict[str, Any], severity: str, language: str) -> str:
        """
        Форматирует ответ для противоречия в ответах.
        
        Args:
            contradiction: Противоречие
            severity: Серьезность
            language: Язык
            
        Returns:
            str: Отформатированный ответ
        """
        if language == "ru":
            return self._format_response_conflict_response_ru(contradiction, severity)
        else:
            return self._format_response_conflict_response_en(contradiction, severity)
    
    def _format_general_conflict_response(self, contradiction: Dict[str, Any], severity: str, language: str) -> str:
        """
        Форматирует общий ответ на противоречие.
        
        Args:
            contradiction: Противоречие
            severity: Серьезность
            language: Язык
            
        Returns:
            str: Отформатированный ответ
        """
        if language == "ru":
            return self._format_general_conflict_response_ru(contradiction, severity)
        else:
            return self._format_general_conflict_response_en(contradiction, severity)
    
    def _format_numeric_conflict_response_ru(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует ответ для числового противоречия на русском языке."""
        fact1 = contradiction["metadata"]["conflicting_facts"][0]
        fact2 = contradiction["metadata"]["conflicting_facts"][1]
        
        value1 = fact1.get("value", "")
        value2 = fact2.get("value", "")
        source1 = fact1.get("source", "источник 1")
        source2 = fact2.get("source", "источник 2")
        
        if severity == "high":
            return (f"Обнаружены значительные различия в данных по '{contradiction['concept']}': {value1} согласно {source1} "
                    f"и {value2} согласно {source2}. Эти различия, вероятно, обусловлены разными методами измерения или "
                    f"контекстными условиями. Рекомендуется уточнить условия применения каждого значения.")
        elif severity == "medium":
            return (f"Существуют заметные различия в данных по '{contradiction['concept']}': {value1} согласно {source1} "
                    f"и {value2} согласно {source2}. Эти различия могут быть объяснены "
                    f"небольшими отличиями в контексте.")
        else:
            return (f"Незначительные различия в данных по '{contradiction['concept']}': {value1} согласно {source1} "
                    f"и {value2} согласно {source2}. Эти различия находятся в пределах нормальной вариативности "
                    f"и могут быть объяснены различиями в методах измерения.")
    
    def _format_boolean_conflict_response_ru(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует ответ для булева противоречия на русском языке."""
        fact1 = contradiction["metadata"]["conflicting_facts"][0]
        fact2 = contradiction["metadata"]["conflicting_facts"][1]
        
        value1 = "да" if fact1.get("value", False) else "нет"
        value2 = "да" if fact2.get("value", False) else "нет"
        source1 = fact1.get("source", "источник 1")
        source2 = fact2.get("source", "источник 2")
        
        if severity == "high":
            return (f"Противоречивые утверждения об '{contradiction['concept']}': {source1} утверждает, что это {value1}, "
                    f"а {source2} утверждает, что это {value2}. Это серьезное противоречие, требующее дополнительного анализа "
                    f"контекстных условий и источников информации.")
        elif severity == "medium":
            return (f"Имеются расхождения в утверждениях об '{contradiction['concept']}': {source1} утверждает, что это {value1}, "
                    f"а {source2} утверждает, что это {value2}. Эти расхождения могут быть объяснены "
                    f"разными интерпретациями концепта.")
        else:
            return (f"Незначительные различия в ответах на вопрос о '{contradiction['concept']}': "
                    f"1. {source1} утверждает, что это {value1} "
                    f"2. {source2} утверждает, что это {value2} "
                    f"Эти ответы дополняют друг друга, отражая разные грани одного явления. "
                    f"Оба ответа могут быть верны в разных контекстах.")
    
    def _format_response_conflict_response_ru(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует ответ для противоречия в ответах на русском языке."""
        fact1 = contradiction["metadata"]["conflicting_facts"][0]
        fact2 = contradiction["metadata"]["conflicting_facts"][1]
        snippet1 = fact1.get("value", "")[:100] + "..." if len(str(fact1.get("value", ""))) > 100 else fact1.get("value", "")
        snippet2 = fact2.get("value", "")[:100] + "..." if len(str(fact2.get("value", ""))) > 100 else fact2.get("value", "")
        
        if severity == "high":
            return (f"Существуют радикально разные ответы на вопрос о '{contradiction['concept']}':"
                    f"1. {snippet1}"
                    f"2. {snippet2}"
                    f"Эти ответы значительно расходятся и требуют детального анализа для определения условий, "
                    f"при которых каждый из них верен. Рекомендуется уточнить контекст вопроса и цели использования информации.")
        elif severity == "medium":
            return (f"Существуют различные ответы на вопрос о '{contradiction['concept']}':"
                    f"1. {snippet1}"
                    f"2. {snippet2}"
                    f"Эти ответы отражают разные аспекты концепта. Рекомендуется уточнить контекст вопроса "
                    f"и определить, какие условия делают каждый ответ верным.")
        else:
            return (f"Незначительные различия в ответах на вопрос о '{contradiction['concept']}':"
                    f"1. {snippet1}"
                    f"2. {snippet2}"
                    f"Эти ответы дополняют друг друга, отражая разные грани одного явления. "
                    f"Оба ответа могут быть верны в разных контекстах.")
    
    def _format_general_conflict_response_ru(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует общий ответ на противоречие на русском языке."""
        if severity == "high":
            return (f"Обнаружено серьезное противоречие в знаниях по '{contradiction['concept']}'. "
                    f"Существуют несовместимые утверждения, требующие анализа. "
                    f"Рекомендуется провести детальный анализ источников и контекста для определения "
                    f"наиболее достоверной информации.")
        elif severity == "medium":
            return (f"Обнаружено противоречие средней серьезности по '{contradiction['concept']}'. "
                    f"Существуют различные интерпретации, которые могут быть объяснены "
                    f"контекстными различиями. Рекомендуется уточнить условия применения знаний.")
        else:
            return (f"Незначительное противоречие в знаниях по '{contradiction['concept']}'. "
                    f"Существуют небольшие расхождения в информации, которые могут быть объяснены "
                    f"нормальной вариативностью источников. Оба варианта могут быть верны в разных контекстах.")
    
    def _format_numeric_conflict_response_en(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует ответ для числового противоречия на английском языке."""
        fact1 = contradiction["metadata"]["conflicting_facts"][0]
        fact2 = contradiction["metadata"]["conflicting_facts"][1]
        
        value1 = fact1.get("value", "")
        value2 = fact2.get("value", "")
        source1 = fact1.get("source", "source 1")
        source2 = fact2.get("source", "source 2")
        
        if severity == "high":
            return (f"Significant differences in data for '{contradiction['concept']}': {value1} according to {source1} "
                    f"and {value2} according to {source2}. These differences are likely due to different measurement methods or "
                    f"contextual conditions. It is recommended to clarify the application context for each value.")
        elif severity == "medium":
            return (f"Notable differences in data for '{contradiction['concept']}': {value1} according to {source1} "
                    f"and {value2} according to {source2}. These differences may be explained by "
                    f"minor contextual variations.")
        else:
            return (f"Minor differences in data for '{contradiction['concept']}': {value1} according to {source1} "
                    f"and {value2} according to {source2}. These discrepancies are within normal variability "
                    f"and may be explained by differences in measurement methods.")
    
    def _format_boolean_conflict_response_en(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует ответ для булева противоречия на английском языке."""
        fact1 = contradiction["metadata"]["conflicting_facts"][0]
        fact2 = contradiction["metadata"]["conflicting_facts"][1]
        
        value1 = "yes" if fact1.get("value", False) else "no"
        value2 = "yes" if fact2.get("value", False) else "no"
        source1 = fact1.get("source", "source 1")
        source2 = fact2.get("source", "source 2")
        
        if severity == "high":
            return (f"Conflicting statements about '{contradiction['concept']}': {source1} states that it is {value1}, "
                    f"while {source2} states that it is {value2}. This is a serious contradiction requiring additional analysis "
                    f"of contextual conditions and information sources.")
        elif severity == "medium":
            return (f"There are discrepancies in statements about '{contradiction['concept']}': {source1} states that it is {value1}, "
                    f"while {source2} states that it is {value2}. These discrepancies may be explained by "
                    f"different interpretations of the concept.")
        else:
            return (f"Minor differences in responses to the question about '{contradiction['concept']}':"
                    f"1. {source1} states that it is {value1}"
                    f"2. {source2} states that it is {value2}"
                    f"These responses complement each other, reflecting different facets of the same phenomenon. "
                    f"Both responses may be valid in different contexts.")
    
    def _format_response_conflict_response_en(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует ответ для противоречия в ответах на английском языке."""
        fact1 = contradiction["metadata"]["conflicting_facts"][0]
        fact2 = contradiction["metadata"]["conflicting_facts"][1]
        snippet1 = str(fact1.get("value", ""))[:100] + "..." if len(str(fact1.get("value", ""))) > 100 else fact1.get("value", "")
        snippet2 = str(fact2.get("value", ""))[:100] + "..." if len(str(fact2.get("value", ""))) > 100 else fact2.get("value", "")
        
        if severity == "high":
            return (f"There are radically different responses to the question about '{contradiction['concept']}':"
                    f"1. {snippet1}"
                    f"2. {snippet2}"
                    f"These responses significantly diverge and require detailed analysis to determine the conditions "
                    f"under which each is valid. It is recommended to clarify the context of the question and the purpose of the information.")
        elif severity == "medium":
            return (f"There are different responses to the question about '{contradiction['concept']}':"
                    f"1. {snippet1}"
                    f"2. {snippet2}"
                    f"These responses reflect different aspects of the concept. It is recommended to clarify the context of the question "
                    f"and determine which conditions make each response valid.")
        else:
            return (f"Minor differences in responses to the question about '{contradiction['concept']}':"
                    f"1. {snippet1}"
                    f"2. {snippet2}"
                    f"These responses complement each other, reflecting different facets of the same phenomenon. "
                    f"Both responses may be valid in different contexts.")
    
    def _format_general_conflict_response_en(self, contradiction: Dict[str, Any], severity: str) -> str:
        """Форматирует общий ответ на противоречие на английском языке."""
        if severity == "high":
            return (f"A serious contradiction in knowledge about '{contradiction['concept']}' has been detected. "
                    f"Incompatible statements require analysis. It is recommended to conduct a detailed analysis of sources and context "
                    f"to determine the most reliable information.")
        elif severity == "medium":
            return (f"A moderate contradiction in knowledge about '{contradiction['concept']}' has been detected. "
                    f"Different interpretations may be explained by contextual differences. It is recommended to clarify "
                    f"the conditions for applying this knowledge.")
        else:
            return (f"A minor contradiction in knowledge about '{contradiction['concept']}' has been detected. "
                    f"Minor discrepancies in information may be explained by normal source variability. "
                    f"Both variants may be valid in different contexts.")