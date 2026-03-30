"""
Ambiguity Resolution Module for ЕВА
Resolves ambiguous entities by generating clarification requests and refinement queries.
"""
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from .context_entity import (
    AmbiguousEntity,
    AmbiguityType,
    ClarificationRequest,
    RefinementQuery,
    EntityExtractor
)


class AmbiguityResolver:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.entity_extractor = EntityExtractor(config)
        self.min_clarification_priority = self.config.get("min_clarification_priority", 1)
        self.max_clarification_options = self.config.get("max_clarification_options", 4)
        self.context_window_size = self.config.get("context_window_size", 200)

    def generate_clarification(self, entity: AmbiguousEntity, context: str = "") -> ClarificationRequest:
        question = self._generate_question(entity)
        options = self._generate_options(entity)
        priority = self._calculate_priority(entity)
        return ClarificationRequest(
            entity=entity,
            question=question,
            options=options,
            context=context or entity.context,
            priority=priority
        )

    def rank_possible_meanings(self, entity: AmbiguousEntity) -> List[str]:
        if not entity.possible_meanings:
            return []
        ranked = []
        for i, meaning in enumerate(entity.possible_meanings):
            score = self._calculate_meaning_score(meaning, entity)
            ranked.append((meaning, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in ranked]

    def create_refinement_query(self, entity: AmbiguousEntity, answer: str) -> RefinementQuery:
        refined_query = self._apply_refinement(entity, answer)
        resolved_entity = self._clone_entity_with_resolution(entity, answer)
        return RefinementQuery(
            original_query=entity.context,
            refined_query=refined_query,
            resolved_entities=[resolved_entity],
            confidence_score=self._calculate_refinement_confidence(entity, answer)
        )

    def resolve_query(self, query: str) -> Tuple[List[ClarificationRequest], Optional[RefinementQuery]]:
        entities = self.entity_extractor.extract_ambiguous_terms(query)
        clarifications = []
        for entity in entities:
            if entity.needs_clarification:
                clarification = self.generate_clarification(entity, query)
                if clarification.priority >= self.min_clarification_priority:
                    clarifications.append(clarification)
        clarifications.sort(key=lambda c: c.priority, reverse=True)
        if clarifications:
            return clarifications, None
        refinement = self._create_auto_refinement(query, entities)
        return [], refinement

    def _generate_question(self, entity: AmbiguousEntity) -> str:
        type_to_question = {
            AmbiguityType.VAGUE_ADJECTIVE: [
                "How {adjective} exactly?",
                "What degree of {adjective} are you referring to?",
                "Can you be more specific about how {adjective}?"
            ],
            AmbiguityType.VAGUE_QUANTIFIER: [
                "How many exactly?",
                "What specific quantity do you mean?",
                "Can you give me an approximate number?"
            ],
            AmbiguityType.PRONOUN_REFERENCE: [
                "What are you referring to?",
                "Which specific entity do you mean?",
                "Can you clarify what 'it' refers to?"
            ],
            AmbiguityType.DEMONSTRATIVE_REFERENCE: [
                "Which specific item do you mean by '{text}'?",
                "What is '{text}' referring to?",
                "Can you name the specific entity?"
            ],
            AmbiguityType.COMPARATIVE_TERM: [
                "What specifically is it being compared to?",
                "What is the baseline or reference point?",
                "Can you provide a specific comparison value?"
            ],
            AmbiguityType.IMPLICIT_SUBJECT: [
                "What specifically are you referring to?",
                "Which subject are you talking about?",
                "Can you name the entity explicitly?"
            ],
            AmbiguityType.TEMPORAL_VAGUENESS: [
                "At what specific time?",
                "What date/time range are you referring to?",
                "Can you be more specific about the timing?"
            ],
            AmbiguityType.SPATIAL_VAGUENESS: [
                "Where specifically?",
                "What is the location or position?",
                "Can you provide specific coordinates or place?"
            ]
        }
        templates = type_to_question.get(entity.ambiguity_type, [
            "What specifically do you mean by '{text}'?",
            "Can you clarify '{text}'?",
            "Could you be more specific about '{text}'?"
        ])
        template = templates[0]
        question = template.format(adjective=entity.text, text=entity.text)
        return question

    def _generate_options(self, entity: AmbiguousEntity) -> List[str]:
        if not entity.possible_meanings:
            return []
        return entity.possible_meanings[:self.max_clarification_options]

    def _calculate_priority(self, entity: AmbiguousEntity) -> int:
        base_priority = int(entity.confidence * 10)
        type_weights = {
            AmbiguityType.PRONOUN_REFERENCE: 2,
            AmbiguityType.DEMONSTRATIVE_REFERENCE: 2,
            AmbiguityType.VAGUE_QUANTIFIER: 3,
            AmbiguityType.COMPARATIVE_TERM: 2,
            AmbiguityType.VAGUE_ADJECTIVE: 1,
            AmbiguityType.IMPLICIT_SUBJECT: 2,
            AmbiguityType.TEMPORAL_VAGUENESS: 1,
            AmbiguityType.SPATIAL_VAGUENESS: 1
        }
        weight = type_weights.get(entity.ambiguity_type, 1)
        priority = base_priority + weight
        return min(priority, 5)

    def _calculate_meaning_score(self, meaning: str, entity: AmbiguousEntity) -> float:
        score = 0.5
        if meaning and meaning[0].isupper():
            score += 0.1
        numeric_patterns = [r'\d+', r'>', r'<', r'%', r'°', r'km', r'mg', r'kg', r'cm']
        for pattern in numeric_patterns:
            if re.search(pattern, meaning):
                score += 0.2
                break
        specificity_indicators = ['specific', 'exact', 'precise', 'particular']
        for indicator in specificity_indicators:
            if indicator in meaning.lower():
                score += 0.1
        if len(meaning.split()) <= 5:
            score += 0.05
        return min(score, 1.0)

    def _apply_refinement(self, entity: AmbiguousEntity, answer: str) -> str:
        return f"{entity.text} [{answer}]"

    def _clone_entity_with_resolution(self, entity: AmbiguousEntity, answer: str) -> AmbiguousEntity:
        return AmbiguousEntity(
            text=f"{entity.text} [{answer}]",
            ambiguity_type=entity.ambiguity_type,
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            possible_meanings=[answer],
            confidence=0.95,
            context=entity.context,
            needs_clarification=False,
            refinement_suggestion=f"Resolved to: {answer}"
        )

    def _calculate_refinement_confidence(self, entity: AmbiguousEntity, answer: str) -> float:
        if answer in entity.possible_meanings:
            return 0.9
        if any(x in answer.lower() for x in ['specific', 'exact', 'precise']):
            return 0.85
        return 0.7

    def _create_auto_refinement(self, query: str, entities: List[AmbiguousEntity]) -> Optional[RefinementQuery]:
        if not entities:
            return None
        ranked_entities = []
        for entity in entities:
            meanings = self.rank_possible_meanings(entity)
            if meanings:
                ranked_entities.append((entity, meanings[0]))
        if not ranked_entities:
            return None
        resolved = [self._clone_entity_with_resolution(e, m) for e, m in ranked_entities]
        refined = query
        for entity, meaning in ranked_entities:
            refined = refined.replace(entity.text, f"{entity.text} [{meaning}]")
        avg_confidence = sum(e.confidence for e in resolved) / len(resolved)
        return RefinementQuery(
            original_query=query,
            refined_query=refined,
            resolved_entities=resolved,
            confidence_score=avg_confidence
        )
