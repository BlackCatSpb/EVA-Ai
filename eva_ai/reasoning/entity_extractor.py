"""
Entity Extractor - извлечение сущностей для самообучения
Сохраняет сущности из противоречий (даже отклонённых) в knowledge graph
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Извлечённая сущность для сохранения в knowledge graph"""
    entity_type: str  # concept, fact, person, event, etc.
    name: str
    content: str
    source: str  # query, response, contradiction
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Результат извлечения сущностей"""
    entities: List[ExtractedEntity] = field(default_factory=list)
    concepts: Set[str] = field(default_factory=set)
    facts: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)


class EntityExtractor:
    """
    Извлекатель сущностей для самообучения
    
    Функции:
    - extract_from_query(): извлечение из запроса пользователя
    - extract_from_response(): извлечение из ответа Qwen
    - extract_from_contradiction(): извлечение из противоречий
    - save_to_knowledge_graph(): сохранение в knowledge graph
    
    Важно: даже "отклонённые" противоречия сохраняются для самообучения
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Паттерны для извлечения
        self.entity_patterns = {
            'concept': [
                r'это\s+([A-ZА-Я][a-zа-я]+)',
                r'называется\s+([A-ZА-Я][a-zа-я]+)',
                r'термин\s+([A-ZА-Я][a-zа-я]+)',
                r'понятие\s+"([^"]+)"',
            ],
            'fact': [
                r'факт:\s*([^.\n]+)',
                r'известно,\s+что\s+([^.\n]+)',
                r'правда,\s+что\s+([^.\n]+)',
            ],
            'person': [
                r'([A-ZА-Я][a-zа-я]+)\s+([A-ZА-Я][a-zа-я]+)\s+[A-ZА-Я][a-zа-я]+',
                r'(учёный|исследователь|эксперт|автор|ученик)\s+([A-ZА-Я][a-zа-я]+)',
            ],
            'event': [
                r'произошло\s+([A-ZА-Я][a-zа-я]+)',
                r'случилось\s+([A-ZА-Я][a-zа-я]+)',
                r'в\s+(\d{4})\s+году',
            ],
            'value': [
                r'(\d+(?:\.\d+)?)\s*(%|процент|градус|метр|километр)',
                r'(высокий|низкий|средний)\s+уровень',
            ],
        }
        
        logger.info("EntityExtractor инициализирован")
    
    def extract_from_query(self, query: str) -> ExtractionResult:
        """
        Извлекает сущности из запроса пользователя
        
        Args:
            query: Запрос пользователя
            
        Returns:
            ExtractionResult с извлечёнными сущностями
        """
        result = ExtractionResult()
        
        if not query:
            return result
        
        # Извлекаем ключевые слова как потенциальные концепты
        words = query.split()
        keywords = [w for w in words if len(w) > 4 and w.lower() not in {
            'это', 'что', 'как', 'почему', 'когда', 'где', 'кто', 'какой', 'какая'
        }]
        
        for keyword in keywords[:5]:
            result.concepts.add(keyword)
            result.entities.append(ExtractedEntity(
                entity_type='concept',
                name=keyword,
                content=f"Ключевое понятие из запроса: {keyword}",
                source='query',
                confidence=0.6
            ))
        
        # Извлекаем именованные сущности по паттернам
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, query, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group(1) if match.groups() else match.group(0)
                        entity_text = entity_text.strip()
                        
                        if len(entity_text) > 2:
                            result.entities.append(ExtractedEntity(
                                entity_type=entity_type,
                                name=entity_text,
                                content=f"Извлечено из запроса: {entity_text}",
                                source='query',
                                confidence=0.8
                            ))
                except Exception as e:
                    logger.debug(f"Pattern error for {entity_type}: {e}")
        
        logger.debug(f"Извлечено {len(result.entities)} сущностей из запроса")
        return result
    
    def extract_from_response(self, response: str) -> ExtractionResult:
        """
        Извлекает сущности из ответа Qwen
        
        Args:
            response: Ответ модели
            
        Returns:
            ExtractionResult с извлечёнными сущностями
        """
        result = ExtractionResult()
        
        if not response:
            return result
        
        # Извлекаем факты (предложения с утверждениями)
        sentences = re.split(r'[.!?]+', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                # Проверяем что это утверждение
                if any(w in sentence.lower() for w in ['это', 'является', 'означает', 'состоит']):
                    result.facts.append(sentence)
        
        # Извлекаем концепты
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, response, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group(1) if match.groups() else match.group(0)
                        entity_text = entity_text.strip()
                        
                        if len(entity_text) > 2:
                            result.entities.append(ExtractedEntity(
                                entity_type=entity_type,
                                name=entity_text,
                                content=f"Извлечено из ответа: {entity_text}",
                                source='response',
                                confidence=0.7
                            ))
                            result.concepts.add(entity_text)
                except Exception as e:
                    logger.debug(f"Pattern error for {entity_type}: {e}")
        
        logger.debug(f"Извлечено {len(result.entities)} сущностей из ответа")
        return result
    
    def extract_from_contradiction(
        self,
        contradiction: Dict[str, Any],
        weight: float = 0.5,
        is_rejected: bool = False
    ) -> ExtractionResult:
        """
        Извлекает сущности из противоречия
        
        Даже отклонённые противоречия (is_rejected=True) сохраняются для самообучения!
        
        Args:
            contradiction: Словарь с данными о противоречии
            weight: Вес противоречия (0-1)
            is_rejected: Было ли противоречие отклонено весами
            
        Returns:
            ExtractionResult с извлечёнными сущностями
        """
        result = ExtractionResult()
        
        if not contradiction:
            return result
        
        # Извлекаем концепт противоречия
        concept = contradiction.get('concept', '')
        if concept:
            result.concepts.add(concept)
            result.entities.append(ExtractedEntity(
                entity_type='concept',
                name=concept,
                content=f"Концепт противоречия: {concept}",
                source='contradiction',
                confidence=weight,
                properties={
                    'weight': weight,
                    'is_rejected': is_rejected,
                    'contradiction_id': contradiction.get('id', '')
                }
            ))
        
        # Извлекаем противоречивые факты
        conflicting_facts = contradiction.get('conflicting_facts', [])
        if isinstance(conflicting_facts, list):
            for fact in conflicting_facts:
                if isinstance(fact, dict):
                    fact_text = fact.get('fact', fact.get('content', str(fact)))
                else:
                    fact_text = str(fact)
                
                if fact_text:
                    result.facts.append(fact_text)
                    result.entities.append(ExtractedEntity(
                        entity_type='fact',
                        name=fact_text[:50],
                        content=fact_text,
                        source='contradiction',
                        confidence=weight,
                        properties={
                            'weight': weight,
                            'is_rejected': is_rejected
                        }
                    ))
        
        # Извлекаем отношения между концептами
        if concept and conflicting_facts:
            for fact in conflicting_facts[:2]:
                if isinstance(fact, dict):
                    related = fact.get('fact', '')
                else:
                    related = str(fact)[:50]
                
                if related:
                    result.relationships.append({
                        'from': concept,
                        'to': related,
                        'type': 'contradiction',
                        'weight': weight,
                        'is_rejected': is_rejected
                    })
        
        logger.debug(f"Извлечено {len(result.entities)} сущностей из противоречия (rejected={is_rejected})")
        return result
    
    def extract_all(
        self,
        query: str,
        response: str,
        contradictions: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[ExtractedEntity]:
        """
        Извлекает все сущности из запроса, ответа и противоречий
        
        Args:
            query: Запрос пользователя
            response: Ответ Qwen
            contradictions: Список противоречий
            weights: Словарь весов для противоречий (id -> weight)
            
        Returns:
            List[ExtractedEntity]: Все извлечённые сущности
        """
        all_entities = []
        
        # Сущности из запроса
        query_result = self.extract_from_query(query)
        all_entities.extend(query_result.entities)
        
        # Сущности из ответа
        response_result = self.extract_from_response(response)
        all_entities.extend(response_result.entities)
        
        # Сущности из противоречий (включая отклонённые!)
        weights = weights or {}
        
        for contr in contradictions:
            contr_id = contr.get('id', '')
            weight = weights.get(contr_id, 0.5)
            
            # Извлекаем даже если weight низкий (для самообучения!)
            is_rejected = weight < 0.3
            
            contr_result = self.extract_from_contradiction(contr, weight, is_rejected)
            all_entities.extend(contr_result.entities)
        
        # Удаляем дубликаты
        unique_entities = []
        seen = set()
        for entity in all_entities:
            key = f"{entity.entity_type}:{entity.name}"
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        logger.info(f"Всего извлечено {len(unique_entities)} уникальных сущностей")
        return unique_entities
    
    def save_to_knowledge_graph(
        self,
        entities: List[ExtractedEntity],
        knowledge_graph: Optional[Any] = None
    ) -> int:
        """
        Сохраняет сущности в knowledge graph
        
        Args:
            entities: Список сущностей для сохранения
            knowledge_graph: Knowledge graph (optional, использует brain.knowledge_graph)
            
        Returns:
            int: Количество сохранённых сущностей
        """
        if not entities:
            return 0
        
        # Получаем knowledge graph
        kg = knowledge_graph
        if kg is None and self.brain is not None:
            kg = getattr(self.brain, 'knowledge_graph', None)
        
        if kg is None:
            logger.warning("Knowledge graph недоступен, сущности не сохранены")
            return 0
        
        saved_count = 0
        
        for entity in entities:
            try:
                # Используем метод add_node если доступен
                if hasattr(kg, 'add_node'):
                    kg.add_node(
                        node_id=entity.name,
                        node_type=entity.entity_type,
                        content=entity.content,
                        properties=entity.properties
                    )
                    saved_count += 1
                elif hasattr(kg, 'add_entity'):
                    kg.add_entity(
                        entity_type=entity.entity_type,
                        name=entity.name,
                        content=entity.content,
                        properties=entity.properties
                    )
                    saved_count += 1
            except Exception as e:
                logger.debug(f"Не удалось сохранить сущность {entity.name}: {e}")
        
        logger.info(f"Сохранено {saved_count} сущностей в knowledge graph")
        return saved_count
    
    def format_for_self_learning(self, entities: List[ExtractedEntity]) -> str:
        """
        Форматирует сущности для использования в самообучении
        
        Returns:
            str: Форматированная строка для промпта
        """
        if not entities:
            return "Новых сущностей не обнаружено."
        
        parts = ["Сущности для самообучения:"]
        
        for entity in entities[:10]:
            parts.append(f"\n- [{entity.entity_type}] {entity.name}")
            parts.append(f"  Контент: {entity.content[:100]}")
            parts.append(f"  Источник: {entity.source}, confidence: {entity.confidence:.2f}")
            
            if entity.properties.get('is_rejected'):
                parts.append(f"  (Отклонено весами, но сохранено для самообучения)")
        
        return "\n".join(parts)


__all__ = ['EntityExtractor', 'ExtractedEntity', 'ExtractionResult']