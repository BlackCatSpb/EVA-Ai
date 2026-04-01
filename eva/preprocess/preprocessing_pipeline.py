"""
Pre-Processing Layer - GGUF-based entity extraction и clarification
Использует GGUF модель для извлечения сущностей и генерации уточнений до основной генерации
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger("eva.preprocess")


@dataclass
class ExtractedEntity:
    """Сущность, извлеченная через GGUF"""
    name: str
    entity_type: str  # person, place, object, concept, action, time, etc.
    confidence: float
    context: str
    relationships: List[str] = field(default_factory=list)


@dataclass  
class PreprocessedQuery:
    """Результат предобработки запроса"""
    original_query: str
    entities: List[ExtractedEntity]
    raw_context: str  # Контекст для hybrid cache
    clarification_needed: bool
    clarification_question: Optional[str] = None
    missing_info: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class GGUFEntityExtractor:
    """
    Извлекает сущности из запроса используя GGUF модель.
    Работает ДО основной генерации для лучшего понимания контекста.
    """
    
    def __init__(self, llama_instance=None):
        self.llama = llama_instance
        
        # Промпт для извлечения сущностей
        self.entity_extraction_prompt = """Проанализируй запрос и извлеки семантические сущности.
Запрос: {query}

Извлеки сущности в формате JSON:
{{
  "entities": [
    {{"name": "название", "type": "тип", "confidence": 0.9, "context": "контекст где найдено", "relationships": ["связанные сущности"]}}
  ],
  "keywords": ["ключевые слова запроса"],
  "ambiguous": ["неоднозначные термины требующие уточнения"]
}}

Типы сущностей: person, place, object, concept, action, time, organization, event, value
Ответь только JSON:"""

        # Промпт для определения необходимости уточнения
        self.clarification_prompt = """Проанализируй запрос и определи, нужно ли уточнение.
Запрос: {query}

Требуется уточнение ТОЛЬКО если:
- Запрос содержит неоднозначные местоимения без контекста (он, она, это без предыдущего упоминания)
- Запрос содержит расплывчатые формулировки (что-то, кто-то, любой)
- Не хватает критически важной информации для ответа

Для простых запросов (привет, как дела, расскажи о X) уточнение НЕ требуется.
Формат JSON:
{{
  "clarification_needed": true/false,
  "question": "уточняющий вопрос",
  "missing_info": ["что нужно уточнить"]
}}

Ответь только JSON:"""
    
    def extract_entities(self, query: str, session_context: str = "") -> List[ExtractedEntity]:
        """Извлекает сущности из запроса через GGUF"""
        if not self.llama:
            logger.warning("llama not initialized, using fallback")
            return self._fallback_extraction(query)
        
        try:
            full_query = query
            if session_context:
                full_query = f"Контекст: {session_context}\n\nЗапрос: {query}"
            
            prompt = self.entity_extraction_prompt.format(query=full_query)
            
            response = self.llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
                top_p=0.8,
                stop=["<|endoftext|>", "<|im_end|>", "```"]
            )
            
            text = response['choices'][0]['message']['content']
            
            # Парсим JSON из ответа
            entities = self._parse_entities_response(text, query)
            
            logger.info(f"GGUF извлек {len(entities)} сущностей из запроса")
            return entities
            
        except Exception as e:
            logger.error(f"Ошибка GGUF entity extraction: {e}")
            return self._fallback_extraction(query)
    
    def _parse_entities_response(self, text: str, original_query: str) -> List[ExtractedEntity]:
        """Парсит JSON ответ от GGUF"""
        entities = []
        
        # Ищем JSON в ответе
        try:
            # Убираем markdown если есть
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            for e in data.get("entities", []):
                entities.append(ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "concept"),
                    confidence=e.get("confidence", 0.8),
                    context=e.get("context", ""),
                    relationships=e.get("relationships", [])
                ))
                
        except json.JSONDecodeError:
            # Fallback - простой парсинг
            entities = self._fallback_extraction(original_query)
        
        return entities
    
    def _fallback_extraction(self, query: str) -> List[ExtractedEntity]:
        """Простой fallback без GGUF"""
        entities = []
        
        # Простое извлечение ключевых слов
        words = query.lower().split()
        
        # Ищем capitalized слова (potential names/places)
        capitalized = re.findall(r'[A-ZА-Я][a-zа-я]+', query)
        for word in capitalized[:5]:
            entities.append(ExtractedEntity(
                name=word,
                entity_type="unknown",
                confidence=0.5,
                context=query,
                relationships=[]
            ))
        
        return entities
    
    def check_clarification_needed(self, query: str, entities: List[ExtractedEntity], 
                                   session_context: str = "") -> tuple[bool, Optional[str], List[str]]:
        """Определяет нужно ли уточнение через GGUF"""
        if not self.llama:
            return False, None, []
        
        # Проверяем только если есть сильно неоднозначные сущности (confidence < 0.5)
        # Also skip for short/simple queries
        if len(query.split()) < 4:
            return False, None, []
        ambiguous = [e for e in entities if e.confidence < 0.5]
        if not ambiguous:
            return False, None, []
        
        try:
            context_info = ""
            if session_context:
                context_info = f"\nКонтекст разговора: {session_context[-500:]}"
            
            prompt = self.clarification_prompt.format(query=query + context_info)
            
            response = self.llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
                stop=["<|endoftext|>", "<|im_end|>", "```"]
            )
            
            text = response['choices'][0]['message']['content']
            
            # Парсим ответ
            return self._parse_clarification_response(text)
            
        except Exception as e:
            logger.error(f"Ошибка clarification: {e}")
            return False, None, []
    
    def _parse_clarification_response(self, text: str) -> tuple[bool, Optional[str], List[str]]:
        """Парсит JSON ответ для уточнения"""
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            return (
                data.get("clarification_needed", False),
                data.get("question"),
                data.get("missing_info", [])
            )
            
        except json.JSONDecodeError:
            return False, None, []


class PreprocessingPipeline:
    """
    Полный пайплайн предобработки:
    1. Извлечение сущностей через GGUF
    2. Проверка на уточнение
    3. Сохранение в hybrid cache
    """
    
    def __init__(self, llama_instance=None, hybrid_cache=None):
        self.llama = llama_instance
        self.hybrid_cache = hybrid_cache
        self.entity_extractor = GGUFEntityExtractor(llama_instance)
        
        logger.info("PreprocessingPipeline инициализирован")
    
    def process(self, query: str, session_context: str = "", 
                session_id: str = None) -> PreprocessedQuery:
        """
        Обрабатывает запрос и возвращает структурированный результат
        """
        logger.debug(f"Preprocessing: query_len={len(query)}, session={session_id}")
        
        # 1. Извлекаем сущности
        entities = self.entity_extractor.extract_entities(query, session_context)
        logger.debug(f"Extracted {len(entities)} entities")
        
        # 2. Проверяем нужно ли уточнение
        clarification_needed, clarification_question, missing_info = \
            self.entity_extractor.check_clarification_needed(query, entities, session_context)
        
        # 3. Извлекаем ключевые слова
        keywords = self._extract_keywords(query, entities)
        
        # 4. Формируем raw_context для hybrid cache
        raw_context = self._build_raw_context(query, entities, session_context)
        
        # 5. Сохраняем в hybrid cache если есть
        if self.hybrid_cache and session_id:
            self._save_to_cache(session_id, query, entities, raw_context)
            logger.debug(f"Context cached for session {session_id}")
        
        return PreprocessedQuery(
            original_query=query,
            entities=entities,
            raw_context=raw_context,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            missing_info=missing_info,
            keywords=keywords
        )
    
    def _extract_keywords(self, query: str, entities: List[ExtractedEntity]) -> List[str]:
        """Извлекает ключевые слова из запроса"""
        keywords = []
        
        # Добавляем имена сущностей
        for e in entities:
            if e.entity_type in ["concept", "object", "action"]:
                keywords.append(e.name)
        
        # Добавляем значимые слова
        stop_words = {"и", "в", "на", "по", "что", "как", "это", "который", "а", "но", "или"}
        words = query.lower().split()
        for w in words:
            if w not in stop_words and len(w) > 3:
                keywords.append(w)
        
        return list(set(keywords))[:10]
    
    def _build_raw_context(self, query: str, entities: List[ExtractedEntity],
                          session_context: str) -> str:
        """Строит raw контекст для hybrid cache"""
        entities_str = ", ".join([
            f"{e.name} ({e.entity_type})" for e in entities
        ])
        
        context_parts = [
            f"Запрос: {query}",
        ]
        
        if entities_str:
            context_parts.append(f"Сущности: {entities_str}")
        
        if session_context:
            context_parts.append(f"История: {session_context[-300:]}")
        
        return "\n".join(context_parts)
    
    def _save_to_cache(self, session_id: str, query: str, 
                      entities: List[ExtractedEntity], raw_context: str):
        """Сохраняет обработанный контекст в hybrid cache"""
        try:
            if hasattr(self.hybrid_cache, 'add_context'):
                self.hybrid_cache.add_context(
                    session_id=session_id,
                    query=query,
                    entities=[e.name for e in entities],
                    raw_text=raw_context
                )
        except Exception as e:
            logger.debug(f"Не удалось сохранить в cache: {e}")
    
    def get_cached_context(self, session_id: str) -> Optional[str]:
        """Получает закэшированный контекст для сессии"""
        try:
            if hasattr(self.hybrid_cache, 'get_context'):
                return self.hybrid_cache.get_context(session_id)
        except Exception as e:
            logger.debug(f"Не удалось получить из cache: {e}")
        return None
