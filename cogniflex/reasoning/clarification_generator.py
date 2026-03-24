"""
Clarification Generator - генерация контекстных вопросов для Self-Reasoning
Генерирует вопросы на основе анализа, а НЕ случайные/рандомные
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ClarificationGenerator:
    """
    Генератор контекстных вопросов для уточнения
    Вопросы должны быть связаны с запросом, а не рандомные
    """
    
    def __init__(self):
        # Шаблоны для разных типов информации
        self.entity_patterns = {
            'person': ['Кто?', 'Кому?', 'Кем?'],
            'place': ['Где?', 'Куда?', 'Откуда?'],
            'time': ['Когда?', 'Сколько времени?'],
            'object': ['Что?', 'Чем?', 'Какой?'],
            'action': ['Что делал?', 'Как?', 'Почему?'],
            'reason': ['Почему?', 'Зачем?'],
        }
    
    def generate_clarification(
        self, 
        analysis_result: Dict[str, Any], 
        query: str
    ) -> List[str]:
        """
        Генерация уточняющих вопросов на основе анализа
        
        Args:
            analysis_result: Результат анализа (ethics, contradiction, knowledge gaps)
            query: Оригинальный запрос пользователя
            
        Returns:
            Список контекстно связанных вопросов
        """
        questions = []
        
        # Анализируем что именно не хватает
        knowledge_gaps = analysis_result.get('gaps_found', [])
        missing_entities = analysis_result.get('missing_entities', [])
        
        # Генерируем вопросы на основе пробелов в знаниях
        if knowledge_gaps:
            for gap in knowledge_gaps[:3]:  # Максимум 3 вопроса
                gap_type = gap.get('type', '')
                entity = gap.get('entity', '')
                
                if entity:
                    question = self._generate_question_for_entity(entity, gap_type, query)
                    if question:
                        questions.append(question)
        
        # Вопросы о недостающих сущностях
        if missing_entities and len(questions) < 3:
            for entity in missing_entities[:3 - len(questions)]:
                question = self._generate_entity_question(entity, query)
                if question:
                    questions.append(question)
        
        # Если вопросов всё ещё мало - анализируем сам запрос
        if len(questions) < 2:
            extra = self._generate_from_query_analysis(query)
            questions.extend(extra)
        
        # Убираем дубликаты
        questions = list(dict.fromkeys(questions))
        
        return questions[:5]  # Максимум 5 вопросов
    
    def _generate_question_for_entity(self, entity: str, gap_type: str, original_query: str) -> Optional[str]:
        """Генерация вопроса для конкретной сущности"""
        entity_lower = entity.lower()
        
        # Анализируем тип сущности и генерируем соответствующий вопрос
        if any(w in entity_lower for w in ['человек', 'человека', 'человек', 'мужчина', 'женщина', 'ребёнок']):
            return f"Кто именно {entity}?"
        
        elif any(w in entity_lower for w in ['машин', 'автомобил', 'транспорт']):
            return f"Какая именно {entity}?"
        
        elif any(w in entity_lower for w in ['дорог', 'улиц', 'путь', 'местност']):
            return f"По какой {entity}?"
        
        elif any(w in entity_lower for w in ['город', 'деревня', 'село', 'посёлок', 'страна']):
            return f"В каком {entity}?"
        
        elif any(w in entity_lower for w in ['врем', 'день', 'ночь', 'год', 'месяц']):
            return f"Когда именно?"
        
        else:
            # Универсальный вопрос
            return f"Что вы имеете в виду под '{entity}'?"
    
    def _generate_entity_question(self, entity: str, query: str) -> Optional[str]:
        """Генерация вопроса о сущности"""
        # Простой анализ
        if len(entity) < 2:
            return None
        
        return f"Что такое '{entity}' в контексте вашего запроса?"
    
    def _generate_from_query_analysis(self, query: str) -> List[str]:
        """Анализ самого запроса для генерации вопросов"""
        questions = []
        query_lower = query.lower()
        
        # Разбиваем на слова
        words = query_lower.split()
        
        # Анализируем существительные (упрощённо - слова более 4 букв)
        important_words = [w for w in words if len(w) > 4]
        
        for word in important_words[:3]:
            # Пропускаем распространённые слова
            if word in ['котор', 'потому', 'потом', 'потому', 'поэтому', 'значит', 'потом', 'потом']:
                continue
            
            question = f"Что вы имеете в виду под '{word}'?"
            questions.append(question)
        
        return questions
    
    def generate_simple_clarification(self, query: str) -> List[str]:
        """
        Упрощённая генерация вопросов - для случая когда нет анализа
        Всё равно НЕ рандомные, а связанные с запросом
        """
        questions = []
        query_lower = query.lower()
        
        # Типы вопросов на основе ключевых слов в запросе
        question_patterns = [
            # Количество/числа
            (['сколько', 'много', 'мало', 'число', 'количество'], "Сколько именно?"),
            # Время
            (['когда', 'время', 'дата', 'день', 'год', 'месяц'], "Когда именно это произошло?"),
            # Место
            (['где', 'куда', 'откуда', 'место', 'город', 'страна'], "В каком месте?"),
            # Причина
            (['почему', 'причина', 'зачем', 'цель'], "По какой причине?"),
            # Способ
            (['как', 'способ', 'метод', 'путь'], "Каким способом?"),
            # Личность
            (['кто', 'человек', 'лицо', 'автор'], "Кто именно?"),
            # Предмет
            (['что', 'предмет', 'вещь', 'объект'], "Что именно имеете в виду?"),
            # Сравнение
            (['лучше', 'хуже', 'отличие', 'разница', 'сравни'], "По каким критериям сравниваем?"),
        ]
        
        for keywords, question in question_patterns:
            if any(kw in query_lower for kw in keywords):
                if question not in questions:
                    questions.append(question)
        
        # Извлекаем существительные и задаём вопросы по ним
        nouns = self._extract_key_nouns(query)
        for noun in nouns[:3]:
            if noun not in questions and len(questions) < 4:
                questions.append(f"Что именно о '{noun}' вы хотите узнать?")
        
        # Если мало вопросов - задаём общие
        if len(questions) < 2:
            questions.append("Что именно вы хотите узнать?")
            questions.append("Можете уточнить детали?")
        
        return questions[:4]
    
    def _extract_key_nouns(self, text: str) -> List[str]:
        """Упрощённое извлечение ключевых существительных"""
        # Русские существительные (упрощённо - слова с определёнными окончаниями)
        words = text.split()
        nouns = []
        
        for word in words:
            # Пропускаем глаголы и предлоги
            if word in ['по', 'на', 'в', 'с', 'из', 'к', 'за', 'ехал', 'ехала', 'ехало', 'ехали', 'шла', 'шли', 'шёл']:
                continue
            
            # Добавляем слова которые могут быть существительными
            if len(word) > 3 and word.isalpha():
                nouns.append(word)
        
        return nouns


# Примеры использования
if __name__ == "__main__":
    gen = ClarificationGenerator()
    
    # Тест с запросом "По дороге ехала машина"
    query = "По дороге ехала машина"
    questions = gen.generate_simple_clarification(query)
    
    print(f"Запрос: {query}")
    print(f"Вопросы: {questions}")
