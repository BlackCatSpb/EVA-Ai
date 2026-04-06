"""
Analytics Module - извлечение логических компонентов из ответа Qwen
Анализирует текст ответа и разбивает на семантические сущности и логические блоки
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SemanticEntity:
    """Семантическая сущность, извлечённая из текста"""
    text: str
    entity_type: str  # person, concept, fact, action, value, etc.
    confidence: float = 1.0
    start_pos: int = 0
    end_pos: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicalBlock:
    """Логический блок в тексте"""
    block_type: str  # premise, conclusion, example, explanation, definition
    content: str
    position: int
    confidence: float = 1.0
    connections: List[str] = field(default_factory=list)


@dataclass
class AnalyticsResult:
    """Результат анализа текста"""
    entities: List[SemanticEntity] = field(default_factory=list)
    logical_blocks: List[LogicalBlock] = field(default_factory=list)
    coherence_score: float = 1.0
    complexity_score: float = 0.5


class AnalyticsModule:
    """
    Модуль аналитики для извлечения логических компонентов из текста
    
    Функции:
    - extract_entities(): извлечение семантических сущностей
    - decompose_into_logical_blocks(): разбиение текста на логические блоки
    - analyze_text(): полный анализ текста
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Паттерны для извлечения сущностей
        self.entity_patterns = {
            'fact': [
                r'(это|является|означает|что|состоит|включает)\s+([^,\.]+)',
                r'([A-ZА-Я][a-zа-я]+)\s+—\s+([^,\.]+)',
            ],
            'concept': [
                r'понятие\s+([A-ZА-Я][a-zа-я]+)',
                r'термин\s+([A-ZА-Я][a-zа-я]+)',
                r'определение\s+([A-ZА-Я][a-zа-я]+)',
            ],
            'value': [
                r'(\d+(?:\.\d+)?)\s*(%|процент|градус|метр|километр|секунда|минута)',
                r'(высокий|низкий|средний|минимальный|максимальный)\s+уровень',
            ],
            'action': [
                r'(делает|создаёт|изменяет|управляет|влияет|проводит|выполняет)',
                r'\b(глагол|процесс|действие)\b',
            ],
            'person': [
                r'([A-ZА-Я][a-zа-я]+)\s+([A-ZА-Я][a-zа-я]+)\s+[A-ZА-Я]',
                r'(учёный|исследователь|эксперт|специалист)\s+([A-ZА-Я][a-zа-я]+)',
            ],
        }
        
        # Паттерны для логических блоков
        self.block_patterns = {
            'definition': [
                r'это\s+[A-ZА-Я]',
                r'называется\s+',
                r'представляет\s+собой\s+',
                r'является\s+',
            ],
            'premise': [
                r'потому\s+что',
                r'так\s+как',
                r'поскольку',
                r'в\s+связи\s+с',
                r'в\s+результате',
                r'вследствие',
                r'из-за',
            ],
            'conclusion': [
                r'поэтому',
                r'следовательно',
                r'таким\s+образом',
                r'значит',
                r'в\s+итоге',
                r'в\s+результате',
            ],
            'example': [
                r'например',
                r'к примеру',
                r'к примеру',
                r'когда\s+',
                r'в\s+случае\s+',
            ],
            'explanation': [
                r'объясняется\s+',
                r'связано\s+с',
                r'заключается\s+в',
                r'состоит\s+в',
            ],
        }
        
        logger.info("AnalyticsModule инициализирован")
    
    def extract_entities(self, text: str) -> List[SemanticEntity]:
        """
        Извлекает семантические сущности из текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            List[SemanticEntity]: Список извлечённых сущностей
        """
        if not text:
            return []
        
        entities = []
        text_lower = text.lower()
        
        # Извлечение фактов
        for pattern in self.entity_patterns.get('fact', []):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(0).strip()
                    if len(entity_text) > 3:
                        entities.append(SemanticEntity(
                            text=entity_text,
                            entity_type='fact',
                            confidence=0.8,
                            start_pos=match.start(),
                            end_pos=match.end()
                        ))
            except Exception as e:
                logger.debug(f"Pattern error for fact: {e}")
        
        # Извлечение концептов
        for pattern in self.entity_patterns.get('concept', []):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(1).strip()
                    if len(entity_text) > 2:
                        entities.append(SemanticEntity(
                            text=entity_text,
                            entity_type='concept',
                            confidence=0.9,
                            start_pos=match.start(),
                            end_pos=match.end()
                        ))
            except Exception as e:
                logger.debug(f"Pattern error for concept: {e}")
        
        # Извлечение чисел и значений
        for pattern in self.entity_patterns.get('value', []):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(0).strip()
                    if len(entity_text) > 1:
                        entities.append(SemanticEntity(
                            text=entity_text,
                            entity_type='value',
                            confidence=0.9,
                            start_pos=match.start(),
                            end_pos=match.end()
                        ))
            except Exception as e:
                logger.debug(f"Pattern error for value: {e}")
        
        # Извлечение действий
        for pattern in self.entity_patterns.get('action', []):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(0).strip()
                    if len(entity_text) > 3:
                        entities.append(SemanticEntity(
                            text=entity_text,
                            entity_type='action',
                            confidence=0.7,
                            start_pos=match.start(),
                            end_pos=match.end()
                        ))
            except Exception as e:
                logger.debug(f"Pattern error for action: {e}")
        
        # Удаление дубликатов
        unique_entities = []
        seen_texts = set()
        for entity in entities:
            if entity.text.lower() not in seen_texts:
                seen_texts.add(entity.text.lower())
                unique_entities.append(entity)
        
        logger.debug(f"Извлечено {len(unique_entities)} уникальных сущностей")
        return unique_entities
    
    def decompose_into_logical_blocks(self, text: str) -> List[LogicalBlock]:
        """
        Разбивает текст на логические блоки
        
        Args:
            text: Текст для разбиения
            
        Returns:
            List[LogicalBlock]: Список логических блоков
        """
        if not text:
            return []
        
        blocks = []
        sentences = self._split_into_sentences(text)
        
        current_block_type = 'explanation'
        current_block_content = []
        current_position = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Определяем тип блока для предложения
            detected_type = self._detect_block_type(sentence)
            
            # Если тип изменился - создаём новый блок
            if detected_type != current_block_type and current_block_content:
                content = ' '.join(current_block_content)
                if content.strip():
                    blocks.append(LogicalBlock(
                        block_type=current_block_type,
                        content=content.strip(),
                        position=current_position,
                        confidence=0.8
                    ))
                current_block_content = []
                current_position = i
            
            current_block_type = detected_type
            current_block_content.append(sentence)
        
        # Добавляем последний блок
        if current_block_content:
            content = ' '.join(current_block_content)
            if content.strip():
                blocks.append(LogicalBlock(
                    block_type=current_block_type,
                    content=content.strip(),
                    position=current_position,
                    confidence=0.8
                ))
        
        # Если блоков мало - разбиваем на равные части
        if len(blocks) < 2 and len(sentences) > 1:
            blocks = self._fallback_split(sentences)
        
        logger.debug(f"Разбито на {len(blocks)} логических блоков")
        return blocks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        # Разбиваем по знакам препинания
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _detect_block_type(self, sentence: str) -> str:
        """Определяет тип логического блока для предложения"""
        sentence_lower = sentence.lower()
        
        for block_type, patterns in self.block_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    return block_type
        
        # По умолчанию - объяснение
        return 'explanation'
    
    def _fallback_split(self, sentences: List[str]) -> List[LogicalBlock]:
        """Разбивает на блоки если определение типа не сработало"""
        blocks = []
        block_size = max(1, len(sentences) // 3)
        
        for i in range(0, len(sentences), block_size):
            chunk = sentences[i:i + block_size]
            if chunk:
                blocks.append(LogicalBlock(
                    block_type='explanation',
                    content=' '.join(chunk),
                    position=i,
                    confidence=0.6
                ))
        
        return blocks
    
    def analyze_text(self, text: str) -> AnalyticsResult:
        """
        Полный анализ текста - извлечение сущностей и логических блоков
        
        Args:
            text: Текст для анализа
            
        Returns:
            AnalyticsResult: Результат анализа
        """
        entities = self.extract_entities(text)
        logical_blocks = self.decompose_into_logical_blocks(text)
        
        # Рассчитываем оценки
        coherence_score = self._calculate_coherence(entities, logical_blocks)
        complexity_score = len(text) / 1000.0  # Нормализуем по длине
        
        return AnalyticsResult(
            entities=entities,
            logical_blocks=logical_blocks,
            coherence_score=coherence_score,
            complexity_score=min(1.0, complexity_score)
        )
    
    def _calculate_coherence(self, entities: List[SemanticEntity], 
                              blocks: List[LogicalBlock]) -> float:
        """Рассчитывает оценку связности текста"""
        if not entities or not blocks:
            return 0.5
        
        # Базовая оценка
        score = 0.5
        
        # Чем больше сущностей - тем лучше
        if len(entities) >= 5:
            score += 0.2
        elif len(entities) >= 3:
            score += 0.1
        
        # Чем больше блоков - тем лучше структура
        if len(blocks) >= 4:
            score += 0.2
        elif len(blocks) >= 2:
            score += 0.1
        
        return min(1.0, score)
    
    def get_entity_summary(self, result: AnalyticsResult) -> Dict[str, Any]:
        """
        Получает сводку по извлечённым сущностям
        
        Returns:
            Dict с агрегированной информацией
        """
        entity_counts = {}
        for entity in result.entities:
            entity_type = entity.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return {
            'total_entities': len(result.entities),
            'by_type': entity_counts,
            'coherence_score': result.coherence_score,
            'complexity_score': result.complexity_score,
            'block_count': len(result.logical_blocks)
        }
    
    def format_for_prompt(self, result: AnalyticsResult) -> str:
        """
        Форматирует результат анализа для включения в промпт Qwen
        
        Returns:
            str: Форматированная строка для промпта
        """
        parts = []
        
        if result.entities:
            parts.append("Извлечённые сущности:")
            for entity in result.entities[:10]:  # Ограничиваем 10 сущностями
                parts.append(f"  - [{entity.entity_type}] {entity.text}")
        
        if result.logical_blocks:
            parts.append("\nЛогическая структура:")
            for block in result.logical_blocks:
                parts.append(f"  [{block.block_type}] {block.content[:100]}...")
        
        parts.append(f"\nОценка связности: {result.coherence_score:.2f}")
        
        return "\n".join(parts)