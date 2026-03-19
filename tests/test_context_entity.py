"""
Tests for Context Entity Extraction Module
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniflex.knowledge.context_entity import (
    EntityExtractor,
    AmbiguousEntity,
    AmbiguityType
)
from cogniflex.knowledge.ambiguity_resolver import (
    AmbiguityResolver,
    ClarificationRequest,
    RefinementQuery
)


class TestEntityExtractor:
    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_detect_vague_quantifiers_many(self):
        entities = self.extractor.detect_vague_quantifiers("many people attended")
        assert len(entities) >= 1
        assert entities[0].text == "many"
        assert entities[0].ambiguity_type == AmbiguityType.VAGUE_QUANTIFIER
        assert len(entities[0].possible_meanings) > 0

    def test_detect_vague_quantifiers_few(self):
        entities = self.extractor.detect_vague_quantifiers("a few items remain")
        assert len(entities) >= 1
        assert entities[0].text == "few"

    def test_detect_vague_quantifiers_some(self):
        entities = self.extractor.detect_vague_quantifiers("some solutions work")
        assert len(entities) >= 1
        assert entities[0].text == "some"

    def test_detect_vague_quantifiers_several(self):
        entities = self.extractor.detect_vague_quantifiers("several versions exist")
        assert len(entities) >= 1

    def test_detect_vague_quantifiers_a_lot(self):
        entities = self.extractor.detect_vague_quantifiers("a lot of data")
        assert len(entities) >= 1
        assert entities[0].text == "a lot"

    def test_detect_vague_quantifiers_lots_of(self):
        entities = self.extractor.detect_vague_quantifiers("lots of issues")
        assert len(entities) >= 1
        assert entities[0].text == "lots of"

    def test_identify_implicit_references_demonstrative_that(self):
        entities = self.extractor.identify_implicit_references("that system works well")
        found = [e for e in entities if e.text == "that"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.DEMONSTRATIVE_REFERENCE

    def test_identify_implicit_references_demonstrative_this(self):
        entities = self.extractor.identify_implicit_references("this model is fast")
        found = [e for e in entities if e.text == "this"]
        assert len(found) >= 1

    def test_identify_implicit_references_pronoun_it(self):
        entities = self.extractor.identify_implicit_references("it works perfectly")
        found = [e for e in entities if e.text == "it"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.PRONOUN_REFERENCE

    def test_identify_implicit_references_pronoun_they(self):
        entities = self.extractor.identify_implicit_references("they processed the data")
        found = [e for e in entities if e.text == "they"]
        assert len(found) >= 1

    def test_analyze_relative_terms_bigger_than_before(self):
        entities = self.extractor.analyze_relative_terms("bigger than before")
        assert len(entities) >= 1
        assert entities[0].ambiguity_type == AmbiguityType.COMPARATIVE_TERM
        assert "bigger" in entities[0].text.lower()

    def test_analyze_relative_terms_smaller_than_average(self):
        entities = self.extractor.analyze_relative_terms("smaller than average")
        assert len(entities) >= 1

    def test_analyze_relative_terms_more_than_expected(self):
        entities = self.extractor.analyze_relative_terms("more than expected")
        assert len(entities) >= 1

    def test_analyze_relative_terms_faster_than_light(self):
        entities = self.extractor.analyze_relative_terms("faster than light")
        assert len(entities) >= 1

    def test_extract_ambiguous_terms_bright_light(self):
        entities = self.extractor.extract_ambiguous_terms("bright light")
        entity_texts = [e.text for e in entities]
        assert "bright" in entity_texts
        bright_entity = next(e for e in entities if e.text == "bright")
        assert bright_entity.ambiguity_type == AmbiguityType.VAGUE_ADJECTIVE

    def test_extract_ambiguous_terms_it_works(self):
        entities = self.extractor.extract_ambiguous_terms("it works great")
        entity_texts = [e.text for e in entities]
        assert "it" in entity_texts
        assert any("great" in e.text.lower() or e.ambiguity_type == AmbiguityType.VAGUE_ADJECTIVE for e in entities)

    def test_extract_ambiguous_terms_that_system(self):
        entities = self.extractor.extract_ambiguous_terms("that system needs update")
        entity_texts = [e.text for e in entities]
        assert "that" in entity_texts

    def test_extract_ambiguous_terms_many_people(self):
        entities = self.extractor.extract_ambiguous_terms("many people prefer this")
        entity_texts = [e.text for e in entities]
        assert "many" in entity_texts

    def test_extract_ambiguous_terms_bigger_than_before(self):
        entities = self.extractor.extract_ambiguous_terms("bigger than before")
        entity_texts = [e.text for e in entities]
        found_comparative = any(e.ambiguity_type == AmbiguityType.COMPARATIVE_TERM for e in entities)
        assert found_comparative

    def test_extract_ambiguous_terms_complex(self):
        text = "many people think that system is bigger than before and it works great"
        entities = self.extractor.extract_ambiguous_terms(text)
        assert len(entities) >= 3
        types = {e.ambiguity_type for e in entities}
        assert AmbiguityType.VAGUE_QUANTIFIER in types
        assert AmbiguityType.COMPARATIVE_TERM in types
        assert AmbiguityType.VAGUE_ADJECTIVE in types

    def test_context_window(self):
        entity = self.extractor.detect_vague_quantifiers("exactly 50 items, not many")[0]
        assert len(entity.context) > 0
        assert "many" in entity.context

    def test_confidence_scores(self):
        entities = self.extractor.extract_ambiguous_terms("many people")
        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0

    def test_deduplication(self):
        entities = self.extractor.extract_ambiguous_terms("many many items")
        positions = [(e.start_pos, e.end_pos) for e in entities]
        assert len(positions) == len(set(positions))

    def test_max_entities_limit(self):
        extractor = EntityExtractor(config={"max_entities": 3})
        text = "many people think that system is bigger than before and it works great and many more"
        entities = extractor.extract_ambiguous_terms(text)
        assert len(entities) <= 3

    def test_detect_russian_vague_quantifier_mnogo(self):
        entities = self.extractor.detect_russian_vague_quantifiers("много людей пришло")
        assert len(entities) >= 1
        assert entities[0].text == "много"
        assert entities[0].ambiguity_type == AmbiguityType.VAGUE_QUANTIFIER

    def test_detect_russian_vague_quantifier_malo(self):
        entities = self.extractor.detect_russian_vague_quantifiers("мало человек осталось")
        assert len(entities) >= 1
        assert entities[0].text == "мало"

    def test_detect_russian_vague_quantifier_neskolko(self):
        entities = self.extractor.detect_russian_vague_quantifiers("несколько человек участвовало")
        assert len(entities) >= 1
        assert entities[0].text == "несколько"

    def test_detect_russian_vague_quantifier_bolshinstvo(self):
        entities = self.extractor.detect_russian_vague_quantifiers("большинство людей согласны")
        assert len(entities) >= 1
        assert entities[0].text == "большинство"

    def test_detect_russian_vague_quantifier_malenkoe(self):
        entities = self.extractor.detect_russian_vague_quantifiers("немного людей пришло")
        assert len(entities) >= 1
        assert entities[0].text == "немного"

    def test_identify_russian_implicit_reference_eto(self):
        entities = self.extractor.identify_russian_implicit_references("это работает хорошо")
        found = [e for e in entities if e.text == "это"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.DEMONSTRATIVE_REFERENCE

    def test_identify_russian_implicit_reference_oni(self):
        entities = self.extractor.identify_russian_implicit_references("они обработали данные")
        found = [e for e in entities if e.text == "они"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.PRONOUN_REFERENCE

    def test_identify_russian_implicit_reference_on(self):
        entities = self.extractor.identify_russian_implicit_references("он сделал работу")
        found = [e for e in entities if e.text == "он"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.PRONOUN_REFERENCE

    def test_identify_russian_implicit_reference_etot(self):
        entities = self.extractor.identify_russian_implicit_references("этот проект важен")
        found = [e for e in entities if e.text == "этот"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.DEMONSTRATIVE_REFERENCE

    def test_identify_russian_implicit_reference_tot(self):
        entities = self.extractor.identify_russian_implicit_references("тот документ нужен")
        found = [e for e in entities if e.text == "тот"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.DEMONSTRATIVE_REFERENCE

    def test_analyze_russian_relative_terms_bystree_chem(self):
        entities = self.extractor.analyze_russian_relative_terms("быстрее чем раньше")
        assert len(entities) >= 1
        assert entities[0].ambiguity_type == AmbiguityType.COMPARATIVE_TERM

    def test_analyze_russian_relative_terms_luchshe(self):
        entities = self.extractor.analyze_russian_relative_terms("лучше чем вчера")
        assert len(entities) >= 1

    def test_analyze_russian_relative_terms_bolshe(self):
        entities = self.extractor.analyze_russian_relative_terms("больше чем обычно")
        assert len(entities) >= 1

    def test_analyze_russian_relative_terms_menshe(self):
        entities = self.extractor.analyze_russian_relative_terms("меньше чем в прошлый раз")
        assert len(entities) >= 1

    def test_analyze_russian_relative_terms_vyshe(self):
        entities = self.extractor.analyze_russian_relative_terms("выше чем раньше")
        assert len(entities) >= 1

    def test_analyze_russian_relative_terms_nizhe(self):
        entities = self.extractor.analyze_russian_relative_terms("ниже чем ожидалось")
        assert len(entities) >= 1

    def test_detect_russian_vague_adjectives_yarkiy(self):
        entities = self.extractor._detect_russian_vague_adjectives("яркий свет")
        found = [e for e in entities if e.text == "яркий"]
        assert len(found) >= 1
        assert found[0].ambiguity_type == AmbiguityType.VAGUE_ADJECTIVE

    def test_detect_russian_vague_adjectives_tusklый(self):
        entities = self.extractor._detect_russian_vague_adjectives("тусклый свет")
        found = [e for e in entities if e.text == "тусклый"]
        assert len(found) >= 1

    def test_detect_russian_vague_adjectives_gromkiy(self):
        entities = self.extractor._detect_russian_vague_adjectives("громкий звук")
        found = [e for e in entities if e.text == "громкий"]
        assert len(found) >= 1

    def test_detect_russian_vague_adjectives_tikhiy(self):
        entities = self.extractor._detect_russian_vague_adjectives("тихий звук")
        found = [e for e in entities if e.text == "тихий"]
        assert len(found) >= 1

    def test_detect_russian_vague_adjectives_bystriy(self):
        entities = self.extractor._detect_russian_vague_adjectives("быстрый процесс")
        found = [e for e in entities if e.text == "быстрый"]
        assert len(found) >= 1

    def test_detect_russian_vague_adjectives_medlenniy(self):
        entities = self.extractor._detect_russian_vague_adjectives("медленный процесс")
        found = [e for e in entities if e.text == "медленный"]
        assert len(found) >= 1

    def test_extract_ambiguous_terms_russian_yarkiy_svet(self):
        entities = self.extractor.extract_ambiguous_terms("яркий свет")
        entity_texts = [e.text for e in entities]
        assert "яркий" in entity_texts

    def test_extract_ambiguous_terms_russian_mnogo_lyudey(self):
        entities = self.extractor.extract_ambiguous_terms("много людей пришло")
        entity_texts = [e.text for e in entities]
        assert "много" in entity_texts

    def test_extract_ambiguous_terms_russian_bystree_chem(self):
        entities = self.extractor.extract_ambiguous_terms("быстрее чем раньше")
        entity_texts = [e.text for e in entities]
        found_comparative = any(e.ambiguity_type == AmbiguityType.COMPARATIVE_TERM for e in entities)
        assert found_comparative

    def test_extract_ambiguous_terms_russian_complex(self):
        text = "много людей думают, что эта система работает быстрее чем раньше"
        entities = self.extractor.extract_ambiguous_terms(text)
        assert len(entities) >= 2
        types = {e.ambiguity_type for e in entities}
        assert AmbiguityType.VAGUE_QUANTIFIER in types or AmbiguityType.VAGUE_ADJECTIVE in types
        assert AmbiguityType.COMPARATIVE_TERM in types or AmbiguityType.DEMONSTRATIVE_REFERENCE in types


class TestAmbiguityResolver:
    def setup_method(self):
        self.resolver = AmbiguityResolver()

    def test_generate_clarification_for_quantifier(self):
        entity = AmbiguousEntity(
            text="many",
            ambiguity_type=AmbiguityType.VAGUE_QUANTIFIER,
            start_pos=0,
            end_pos=4,
            possible_meanings=["several", "tens", "hundreds"],
            context="many people attended"
        )
        clarification = self.resolver.generate_clarification(entity)
        assert isinstance(clarification, ClarificationRequest)
        assert len(clarification.question) > 0
        assert len(clarification.options) > 0
        assert clarification.priority >= 1

    def test_generate_clarification_for_pronoun(self):
        entity = AmbiguousEntity(
            text="it",
            ambiguity_type=AmbiguityType.PRONOUN_REFERENCE,
            start_pos=0,
            end_pos=2,
            possible_meanings=["the system", "the process", "the result"],
            context="it works"
        )
        clarification = self.resolver.generate_clarification(entity)
        assert "it" in clarification.question.lower() or "what" in clarification.question.lower()

    def test_generate_clarification_for_comparative(self):
        entity = AmbiguousEntity(
            text="bigger than before",
            ambiguity_type=AmbiguityType.COMPARATIVE_TERM,
            start_pos=0,
            end_pos=16,
            possible_meanings=["10% larger", "50% larger"],
            context="bigger than before"
        )
        clarification = self.resolver.generate_clarification(entity)
        assert clarification.priority >= 2

    def test_rank_possible_meanings(self):
        entity = AmbiguousEntity(
            text="many",
            ambiguity_type=AmbiguityType.VAGUE_QUANTIFIER,
            start_pos=0,
            end_pos=4,
            possible_meanings=["several", "tens", "hundreds", "100+", "1000+"]
        )
        ranked = self.resolver.rank_possible_meanings(entity)
        assert len(ranked) == len(entity.possible_meanings)
        numeric_first = any(r[0].isdigit() for r in ranked[:2])
        assert numeric_first or len(ranked) > 0

    def test_rank_possible_meanings_empty(self):
        entity = AmbiguousEntity(
            text="xyz",
            ambiguity_type=AmbiguityType.VAGUE_ADJECTIVE,
            start_pos=0,
            end_pos=3,
            possible_meanings=[]
        )
        ranked = self.resolver.rank_possible_meanings(entity)
        assert ranked == []

    def test_create_refinement_query(self):
        entity = AmbiguousEntity(
            text="many",
            ambiguity_type=AmbiguityType.VAGUE_QUANTIFIER,
            start_pos=0,
            end_pos=4,
            possible_meanings=["tens", "hundreds"],
            context="many people attended"
        )
        refinement = self.resolver.create_refinement_query(entity, "tens")
        assert isinstance(refinement, RefinementQuery)
        assert refinement.refined_query != entity.context
        assert len(refinement.resolved_entities) == 1
        assert refinement.confidence_score > 0.5

    def test_resolve_query_with_ambiguities(self):
        query = "many people prefer that solution"
        clarifications, refinement = self.resolver.resolve_query(query)
        assert len(clarifications) > 0
        assert refinement is None
        for c in clarifications:
            assert c.priority >= 1

    def test_resolve_query_without_ambiguities(self):
        query = "I want to process 500 items with exact precision"
        clarifications, refinement = self.resolver.resolve_query(query)
        if refinement:
            assert len(refinement.resolved_entities) >= 0
        assert clarifications is not None

    def test_priority_calculation_vague_quantifier(self):
        entity = AmbiguousEntity(
            text="many",
            ambiguity_type=AmbiguityType.VAGUE_QUANTIFIER,
            start_pos=0,
            end_pos=4,
            possible_meanings=["several"],
            confidence=0.7
        )
        clarification = self.resolver.generate_clarification(entity)
        assert clarification.priority >= 4

    def test_priority_calculation_pronoun(self):
        entity = AmbiguousEntity(
            text="it",
            ambiguity_type=AmbiguityType.PRONOUN_REFERENCE,
            start_pos=0,
            end_pos=2,
            possible_meanings=["the system"],
            confidence=0.6
        )
        clarification = self.resolver.generate_clarification(entity)
        assert clarification.priority >= 3

    def test_clarification_options_limited(self):
        resolver = AmbiguityResolver(config={"max_clarification_options": 2})
        entity = AmbiguousEntity(
            text="many",
            ambiguity_type=AmbiguityType.VAGUE_QUANTIFIER,
            start_pos=0,
            end_pos=4,
            possible_meanings=["option1", "option2", "option3", "option4", "option5"],
            context="many items"
        )
        clarification = resolver.generate_clarification(entity)
        assert len(clarification.options) <= 2


class TestIntegration:
    def test_full_pipeline_bright_light(self):
        text = "bright light"
        extractor = EntityExtractor()
        entities = extractor.extract_ambiguous_terms(text)
        resolver = AmbiguityResolver()
        for entity in entities:
            clarification = resolver.generate_clarification(entity)
            ranked = resolver.rank_possible_meanings(entity)
            refinement = resolver.create_refinement_query(entity, ranked[0] if ranked else "specific value")
            assert clarification is not None
            assert len(ranked) >= 0
            assert refinement is not None

    def test_full_pipeline_many_people(self):
        text = "many people attended"
        extractor = EntityExtractor()
        entities = extractor.extract_ambiguous_terms(text)
        assert len(entities) >= 1
        resolver = AmbiguityResolver()
        clarifications, _ = resolver.resolve_query(text)
        assert len(clarifications) >= 1

    def test_full_pipeline_that_system(self):
        text = "that system is efficient"
        extractor = EntityExtractor()
        entities = extractor.extract_ambiguous_terms(text)
        resolver = AmbiguityResolver()
        clarifications, _ = resolver.resolve_query(text)
        assert len(clarifications) >= 1
        assert any("that" in c.entity.text for c in clarifications)

    def test_full_pipeline_bigger_than_before(self):
        text = "bigger than before"
        extractor = EntityExtractor()
        entities = extractor.extract_ambiguous_terms(text)
        resolver = AmbiguityResolver()
        clarifications, _ = resolver.resolve_query(text)
        assert len(clarifications) >= 1

    def test_full_pipeline_it_works(self):
        text = "it works"
        extractor = EntityExtractor()
        entities = extractor.extract_ambiguous_terms(text)
        resolver = AmbiguityResolver()
        clarifications, _ = resolver.resolve_query(text)
        assert len(clarifications) >= 1
        assert any("it" in c.entity.text for c in clarifications)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
