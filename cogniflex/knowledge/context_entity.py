"""
Context Entity Extraction Module for CogniFlex
Extracts ambiguous and context-dependent entities from natural language queries.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class AmbiguityType(Enum):
    VAGUE_ADJECTIVE = "vague_adjective"
    PRONOUN_REFERENCE = "pronoun_reference"
    DEMONSTRATIVE_REFERENCE = "demonstrative_reference"
    VAGUE_QUANTIFIER = "vague_quantifier"
    COMPARATIVE_TERM = "comparative_term"
    IMPLICIT_SUBJECT = "implicit_subject"
    TEMPORAL_VAGUENESS = "temporal_vagueness"
    SPATIAL_VAGUENESS = "spatial_vagueness"


@dataclass
class AmbiguousEntity:
    text: str
    ambiguity_type: AmbiguityType
    start_pos: int
    end_pos: int
    possible_meanings: List[str] = field(default_factory=list)
    confidence: float = 0.5
    context: str = ""
    needs_clarification: bool = True
    refinement_suggestion: str = ""

    def __hash__(self):
        return hash((self.text, self.start_pos, self.end_pos))


@dataclass
class ClarificationRequest:
    entity: AmbiguousEntity
    question: str
    options: List[str] = field(default_factory=list)
    context: str = ""
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_text": self.entity.text,
            "ambiguity_type": self.entity.ambiguity_type.value,
            "question": self.question,
            "options": self.options,
            "context": self.context,
            "priority": self.priority
        }


@dataclass
class RefinementQuery:
    original_query: str
    refined_query: str
    resolved_entities: List[AmbiguousEntity] = field(default_factory=list)
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "refined_query": self.refined_query,
            "resolved_entities": [
                {
                    "text": e.text,
                    "type": e.ambiguity_type.value,
                    "resolved_meaning": e.possible_meanings[0] if e.possible_meanings else ""
                }
                for e in self.resolved_entities
            ],
            "confidence_score": self.confidence_score
        }


class EntityExtractor:
    AMBIGUOUS_PATTERNS = {
        'vague_quantifier': [
            r'\b(много|мало|несколько|немного|большинство|меньшинство)\b.*?\b(людей|человек|раз|случаев|раз)?',
            r'\b(много|мало|несколько)\b',
        ],
        'relative_comparison': [
            r'(быстрее|медленнее|лучше|хуже|больше|меньше|выше|ниже)\s*чем\s*',
            r'(так|более|менее)\s+(быстро|медленно|хорошо|плохо)',
        ],
        'implicit_reference': [
            r'\b(это|то|этот|тот|оно|они|их|его|её)\b',
        ],
        'ambiguous_term': [
            r'\b(яркий|тусклый|громкий|тихий|быстрый|медленный)\s+\w+',
            r'\b(\w+)\s+(свет|звук|система|процесс|работа)\b',
        ],
    }

    RUSSIAN_VAGUE_ADJECTIVES = {
        "яркий", "тусклый", "громкий", "тихий", "быстрый", "медленный",
        "большой", "малый", "маленький", "огромный", "крошечный",
        "хороший", "плохой", "отличный", "ужасный", "прекрасный",
        "сильный", "слабый", "мощный",
        "важный", "значительный", "незначительный", "главный",
        "тяжёлый", "лёгкий", "тёмный", "светлый",
        "высокий", "низкий", "длинный", "короткий",
        "чистый", "грязный", "ясный", "туманный",
        "богатый", "бедный", "состоятельный",
        "лёгкий", "трудный", "простой", "сложный",
        "старый", "новый", "молодой", "древний", "современный",
        "похожий", "разный", "одинаковый", "подобный",
        "близкий", "далёкий", "далеко", "близко",
        "мокрый", "сухой", "влажный", "сухой",
        "спелый", "зелёный", "зрелый",
        "безопасный", "опасный", "рискованный", "надёжный"
    }

    RUSSIAN_VAGUE_QUANTIFIERS = {
        "много": ["десятки", "сотни", "тысячи", "конкретное число"],
        "мало": ["несколько", "единицы", "конкретное число"],
        "несколько": ["три", "четыре", "пять", "конкретное число"],
        "немного": ["чуть-чуть", "незначительное количество", "конкретное число"],
        "большинство": ["более 50%", "60%", "70%", "конкретный процент"],
        "меньшинство": ["менее 50%", "40%", "30%", "конкретный процент"],
        "некоторые": ["несколько", "часть", "конкретное количество"],
        "многие": ["много", "значительная часть", "конкретное число"],
        "немногие": ["мало", "небольшая часть", "конкретное число"],
    }

    RUSSIAN_DEMONSTRATIVES = {
        "это", "то", "этот", "тот", "та", "такой", "такая", "такое", "такие"
    }

    RUSSIAN_PRONOUNS = {
        "он", "она", "оно", "они", "их", "его", "её", "ей", "им",
        "я", "мы", "вы", "ты",
        "который", "которая", "которое", "которые", "кто", "что", "чей"
    }

    RUSSIAN_COMPARATIVE_PATTERNS = [
        (r"\b(быстрее|медленнее|лучше|хуже|больше|меньше|выше|ниже|сильнее|слабее)\s+чем\s+(\w+)", "comparative_value"),
        (r"\b(более|менее)\s+(быстро|медленно|хорошо|плохо|сильно|слабо)", "comparative_adverb"),
        (r"\b(такой|такая|такое)\s+как\s+(\w+)", "comparison_like"),
    ]

    VAGUE_ADJECTIVES = {
        "bright", "dark", "big", "small", "large", "huge", "tiny",
        "fast", "slow", "quick", "rapid", "gradual", "sudden",
        "hot", "cold", "warm", "cool", "lukewarm",
        "good", "bad", "great", "terrible", "awesome", "awful",
        "strong", "weak", "powerful", "mighty",
        "beautiful", "ugly", "pretty", "handsome",
        "important", "significant", "minor", "major",
        "heavy", "light", "dense", "sparse",
        "deep", "shallow", "high", "low", "tall", "short",
        "clean", "dirty", "clear", "cloudy", "fuzzy", "blurry",
        "loud", "quiet", "noisy", "silent",
        "rich", "poor", "wealthy", "affluent",
        "easy", "hard", "difficult", "simple", "complex",
        "old", "new", "young", "ancient", "modern",
        "long", "brief", "lengthy", "short",
        "similar", "different", "identical", "alike",
        "close", "far", "near", "distant",
        "tight", "loose", "neat", "messy",
        "wet", "dry", "humid", "arid",
        "ripe", "raw", "mature", "green",
        "safe", "dangerous", "risky", "secure"
    }

    VAGUE_QUANTIFIERS = {
        "many": ["several", "numerous", "a lot", "tens", "hundreds", "thousands"],
        "few": ["two", "three", "four", "five", "a couple"],
        "some": ["several", "a few", "a couple", "one or two"],
        "several": ["five", "six", "seven", "eight", "nine", "ten"],
        "a lot": ["dozens", "hundreds", "many", "numerous"],
        "lots of": ["many", "numerous", "a large number of"],
        "most": ["more than half", "majority", "vast majority"],
        "somewhat": ["slightly", "a bit", "moderately", "partially"],
        "quite": ["fairly", "rather", "pretty", "somewhat"],
        "very": ["extremely", "highly", "incredibly", "really"],
        "too": ["excessively", "overly", "extremely", "way too"],
        "any": ["at least one", "some", "a single"],
        "all": ["every", "each", "the entire", "the whole"],
        "every": ["all", "each", "every single"],
        "enough": ["sufficient", "adequate", "the required amount"],
        "less": ["fewer", "a smaller amount", "not as many"],
        "more": ["additional", "extra", "a greater amount"],
        "much": ["a great deal", "a lot", "substantial"],
        "little": ["a small amount", "barely any", "sparse"],
        "various": ["several", "different", "diverse", "multiple"],
        "numerous": ["many", "countless", "innumerable"],
        "multiple": ["several", "various", "multiple distinct"],
        "plenty": ["enough", "sufficient", "ample", "abundant"],
        "a bunch of": ["many", "several", "numerous", "lots of"],
        "a number of": ["several", "many", "numerous", "various"],
        "a variety of": ["various", "different", "diverse", "multiple"]
    }

    DEMONSTRATIVES = {
        "that", "this", "these", "those",
        "such", "the", "it", "they", "them"
    }

    PRONOUNS = {
        "it", "he", "she", "they", "them", "its",
        "his", "her", "their", "hers", "theirs",
        "you", "your", "yours",
        "we", "us", "our", "ours",
        "I", "me", "my", "mine",
        "which", "who", "whom", "whose"
    }

    IMPLICIT_SUBJECTS = {
        "it", "this", "that", "thing", "stuff", "matter",
        "what", "something", "anything", "everything", "nothing"
    }

    COMPARATIVE_PATTERNS = [
        (r"\b(bigger|larger|greater|higher|better|worse|smaller|less|faster|slower|more|less|earlier|later|older|younger)\s+than\s+(\w+)", "comparative_value"),
        (r"\b(more|less|fewer|most|least)\s+than\s+(\w+)", "comparative_value"),
        (r"\b(same|similar|different)\s+as\s+(\w+)", "comparative_value"),
        (r"\b(before|after|previously|earlier|later)\s+than\s+(\w+)", "temporal_comparison"),
        (r"\b(bigger|larger|smaller|taller|shorter|heavier|lighter|faster|slower)\s+than\s+(before|beforehand|earlier|yesterday|last time|before)", "temporal_comparative"),
        (r"\b(most|least|best|worst|biggest|smallest)\s+(of|among)", "superlative"),
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.3)
        self.max_entities = self.config.get("max_entities", 10)

    def extract_ambiguous_terms(self, query: str) -> List[AmbiguousEntity]:
        entities = []
        entities.extend(self.detect_vague_quantifiers(query))
        entities.extend(self.identify_implicit_references(query))
        entities.extend(self.analyze_relative_terms(query))
        entities.extend(self._detect_vague_adjectives(query))
        entities.extend(self.detect_russian_vague_quantifiers(query))
        entities.extend(self.identify_russian_implicit_references(query))
        entities.extend(self.analyze_russian_relative_terms(query))
        entities.extend(self._detect_russian_vague_adjectives(query))
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: (e.confidence, e.start_pos), reverse=True)
        return entities[:self.max_entities]

    def detect_vague_quantifiers(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        words = text_lower.split()
        for quantifier, expansions in self.VAGUE_QUANTIFIERS.items():
            pattern = r'\b' + re.escape(quantifier) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entity = AmbiguousEntity(
                    text=match.group(),
                    ambiguity_type=AmbiguityType.VAGUE_QUANTIFIER,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    possible_meanings=expansions,
                    confidence=0.7,
                    context=self._get_context_window(text, match.start(), match.end()),
                    needs_clarification=True,
                    refinement_suggestion=f"Specify exact quantity instead of '{match.group()}'"
                )
                entities.append(entity)
        return entities

    def identify_implicit_references(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        for demonstrative in self.DEMONSTRATIVES:
            pattern = r'\b' + re.escape(demonstrative) + r'\b'
            for match in re.finditer(pattern, text_lower):
                if self._is_referring_expression(text, match.start(), match.end()):
                    ambiguity_type = AmbiguityType.DEMONSTRATIVE_REFERENCE
                    if demonstrative in {"he", "she", "it", "they", "them", "his", "her", "their"}:
                        ambiguity_type = AmbiguityType.PRONOUN_REFERENCE
                    entity = AmbiguousEntity(
                        text=match.group(),
                        ambiguity_type=ambiguity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        possible_meanings=self._get_referent_suggestions(text, match.group(), match.start()),
                        confidence=0.6,
                        context=self._get_context_window(text, match.start(), match.end()),
                        needs_clarification=True,
                        refinement_suggestion=f"Replace '{match.group()}' with specific entity name"
                    )
                    entities.append(entity)
        return entities

    def analyze_relative_terms(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        for pattern, term_type in self.COMPARATIVE_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                comparative_word = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group()
                entity = AmbiguousEntity(
                    text=match.group(),
                    ambiguity_type=AmbiguityType.COMPARATIVE_TERM,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    possible_meanings=self._expand_comparative(comparative_word),
                    confidence=0.65,
                    context=self._get_context_window(text, match.start(), match.end()),
                    needs_clarification=True,
                    refinement_suggestion=f"Provide specific baseline for '{match.group()}' comparison"
                )
                entities.append(entity)
        return entities

    def _detect_vague_adjectives(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        for adj in self.VAGUE_ADJECTIVES:
            pattern = r'\b' + re.escape(adj) + r'\b'
            for match in re.finditer(pattern, text_lower):
                if not self._is_intensified(text, match.start(), match.end()):
                    entity = AmbiguousEntity(
                        text=match.group(),
                        ambiguity_type=AmbiguityType.VAGUE_ADJECTIVE,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        possible_meanings=self._get_adjective_degrees(adj),
                        confidence=0.5,
                        context=self._get_context_window(text, match.start(), match.end()),
                        needs_clarification=True,
                        refinement_suggestion=f"Specify degree or measurement for '{match.group()}'"
                    )
                    entities.append(entity)
        return entities

    def _is_referring_expression(self, text: str, start: int, end: int) -> bool:
        if end >= len(text):
            return True
        remaining = text[end:].strip()
        if remaining and remaining[0] in '.,!?;:)':
            return False
        return True

    def _get_referent_suggestions(self, text: str, pronoun: str, pos: int) -> List[str]:
        suggestions = []
        if pronoun.lower() in {"that", "this"}:
            suggestions = ["the mentioned item", "the previous topic", "the current subject"]
        elif pronoun.lower() in {"it", "this", "that"}:
            suggestions = ["the system", "the process", "the result", "the output", "the component"]
        elif pronoun.lower() in {"they", "them", "these", "those"}:
            suggestions = ["the mentioned items", "the related components", "the previous elements"]
        elif pronoun.lower() in {"he", "she"}:
            suggestions = ["the person", "the user", "the client", "the mentioned individual"]
        return suggestions

    def _expand_comparative(self, comparative: str) -> List[str]:
        expansions = {
            "bigger": ["10% larger", "50% larger", "twice as large", "10 units larger"],
            "smaller": ["10% smaller", "half the size", "20 units smaller"],
            "better": ["10% improvement", "significant improvement", "qualitatively improved"],
            "worse": ["10% degradation", "significantly worse", "qualitatively worse"],
            "more": ["additional quantity", "increased amount", "extra units"],
            "less": ["reduced quantity", "decreased amount", "fewer units"],
            "faster": ["2x speed", "50% faster", "specific time reduction"],
            "slower": ["2x slower", "50% slower", "specific time increase"],
            "larger": ["bigger size", "greater capacity", "expanded scope"],
            "greater": ["higher value", "increased magnitude", "larger quantity"],
            "higher": ["elevated level", "increased height", "greater value"],
            "earlier": ["specific datetime", "previous time point", "antecedent event"],
            "later": ["future datetime", "subsequent time point", "later event"],
            "older": ["specific age", "prior version", "preceding time"],
            "younger": ["specific age", "later version", "subsequent time"],
            "similar": ["with 90% similarity", "nearly identical", "comparable in aspect X"],
            "different": ["different in specific way", "divergent from baseline", "altered parameter"],
            "same": ["identical to X", "equivalent to Y", "matching specific criteria"],
            "before": ["prior to specific time", "preceding event", "earlier than X"],
            "after": ["following specific time", "subsequent to event", "later than X"]
        }
        return expansions.get(comparative.lower(), [f"specific value relative to {comparative}"])

    def _get_adjective_degrees(self, adj: str) -> List[str]:
        degrees = {
            "bright": ["intensity 80-100%", "intensity 50-80%", "specific lumen value"],
            "dark": ["intensity 0-20%", "intensity 20-40%", "specific darkness level"],
            "big": ["dimensions >1m", "dimensions >10m", "specific size in units"],
            "small": ["dimensions <10cm", "dimensions <1m", "specific size in units"],
            "hot": [">80°C", "40-60°C", "specific temperature value"],
            "cold": ["<10°C", "0-5°C", "specific temperature value"],
            "fast": [">100 km/h", ">10x baseline", "specific speed value"],
            "slow": ["<10 km/h", "<50% baseline", "specific speed value"],
            "good": ["rating 8-10/10", "meets all criteria", "specific quality metric"],
            "bad": ["rating 0-3/10", "fails critical criteria", "specific deficiency"],
            "strong": [">80% capacity", "specific force value", "high correlation"],
            "weak": ["<20% capacity", "specific force value", "low correlation"],
            "old": [">50 years", ">10 years", "specific age or version"],
            "new": ["<1 year", "<6 months", "specific release date"],
            "long": [">1 hour", ">10 km", "specific length value"],
            "short": ["<1 minute", "<100 m", "specific length value"],
            "heavy": [">100 kg", ">50 kg", "specific weight value"],
            "light": ["<5 kg", "<1 kg", "specific weight value"],
            "high": [">1000 m", ">100 units", "specific elevation/value"],
            "low": ["<100 m", "<10 units", "specific elevation/value"],
            "rich": [">$1M assets", "specific net worth", "income >X"],
            "poor": ["<$10K assets", "specific financial state", "income <X"],
        }
        return degrees.get(adj, [f"degree of {adj}", f"specific measurement of {adj}"])

    def _is_intensified(self, text: str, start: int, end: int) -> bool:
        intensifiers = {"very", "really", "extremely", "incredibly", "absolutely", "quite", "rather", "somewhat", "fairly", "pretty", "super", "highly", "too", "so"}
        text_lower = text.lower()
        before_start = max(0, start - 10)
        before = text_lower[before_start:start]
        for intensifier in intensifiers:
            if before.strip().endswith(intensifier):
                return True
        return False

    def _get_context_window(self, text: str, start: int, end: int, window: int = 50) -> str:
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        prefix = "..." if ctx_start > 0 else ""
        suffix = "..." if ctx_end < len(text) else ""
        return prefix + text[ctx_start:ctx_end] + suffix

    def _deduplicate_entities(self, entities: List[AmbiguousEntity]) -> List[AmbiguousEntity]:
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.start_pos, e.end_pos)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def detect_russian_vague_quantifiers(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        for quantifier, expansions in self.RUSSIAN_VAGUE_QUANTIFIERS.items():
            pattern = r'\b' + re.escape(quantifier) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entity = AmbiguousEntity(
                    text=match.group(),
                    ambiguity_type=AmbiguityType.VAGUE_QUANTIFIER,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    possible_meanings=expansions,
                    confidence=0.7,
                    context=self._get_context_window(text, match.start(), match.end()),
                    needs_clarification=True,
                    refinement_suggestion=f"Уточните точное количество вместо '{match.group()}'"
                )
                entities.append(entity)
        return entities

    def identify_russian_implicit_references(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        all_demonstratives = set(self.RUSSIAN_DEMONSTRATIVES) | set(self.RUSSIAN_PRONOUNS)
        for demonstrative in all_demonstratives:
            pattern = r'\b' + re.escape(demonstrative) + r'\b'
            for match in re.finditer(pattern, text_lower):
                if self._is_referring_expression(text, match.start(), match.end()):
                    ambiguity_type = AmbiguityType.DEMONSTRATIVE_REFERENCE
                    if demonstrative in self.RUSSIAN_PRONOUNS:
                        ambiguity_type = AmbiguityType.PRONOUN_REFERENCE
                    entity = AmbiguousEntity(
                        text=match.group(),
                        ambiguity_type=ambiguity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        possible_meanings=self._get_russian_referent_suggestions(match.group()),
                        confidence=0.6,
                        context=self._get_context_window(text, match.start(), match.end()),
                        needs_clarification=True,
                        refinement_suggestion=f"Замените '{match.group()}' конкретным названием"
                    )
                    entities.append(entity)
        return entities

    def analyze_russian_relative_terms(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        for pattern, term_type in self.RUSSIAN_COMPARATIVE_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                comparative_word = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group()
                entity = AmbiguousEntity(
                    text=match.group(),
                    ambiguity_type=AmbiguityType.COMPARATIVE_TERM,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    possible_meanings=self._expand_russian_comparative(comparative_word),
                    confidence=0.65,
                    context=self._get_context_window(text, match.start(), match.end()),
                    needs_clarification=True,
                    refinement_suggestion=f"Укажите конкретную базу для сравнения '{match.group()}'"
                )
                entities.append(entity)
        return entities

    def _detect_russian_vague_adjectives(self, text: str) -> List[AmbiguousEntity]:
        entities = []
        text_lower = text.lower()
        for adj in self.RUSSIAN_VAGUE_ADJECTIVES:
            pattern = r'\b' + re.escape(adj) + r'\b'
            for match in re.finditer(pattern, text_lower):
                if not self._is_intensified(text, match.start(), match.end()):
                    entity = AmbiguousEntity(
                        text=match.group(),
                        ambiguity_type=AmbiguityType.VAGUE_ADJECTIVE,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        possible_meanings=self._get_russian_adjective_degrees(adj),
                        confidence=0.5,
                        context=self._get_context_window(text, match.start(), match.end()),
                        needs_clarification=True,
                        refinement_suggestion=f"Уточните степень для '{match.group()}'"
                    )
                    entities.append(entity)
        return entities

    def _get_russian_referent_suggestions(self, pronoun: str) -> List[str]:
        suggestions = {
            "это": ["упомянутый объект", "предыдущая тема", "текущий предмет"],
            "то": ["тот объект", "другая тема", "упомянутое"],
            "этот": ["этот объект", "данный элемент", "текущий предмет"],
            "тот": ["тот объект", "другой элемент", "упомянутый предмет"],
            "он": ["человек", "пользователь", "упомянутый человек"],
            "она": ["человек", "пользователь", "упомянутый человек"],
            "оно": ["система", "процесс", "результат"],
            "они": ["люди", "пользователи", "упомянутые элементы"],
            "его": ["упомянутый объект", "его часть", "его атрибут"],
            "её": ["упомянутый объект", "её часть", "её атрибут"],
            "их": ["упомянутые объекты", "их части", "их атрибуты"],
        }
        return suggestions.get(pronoun.lower(), ["конкретный объект", "уточните предмет"])

    def _expand_russian_comparative(self, comparative: str) -> List[str]:
        expansions = {
            "быстрее": ["в 2 раза быстрее", "на 50% быстрее", "конкретное уменьшение времени"],
            "медленнее": ["в 2 раза медленнее", "на 50% медленнее", "конкретное увеличение времени"],
            "лучше": ["улучшение на 10%", "значительное улучшение", "качественно лучше"],
            "хуже": ["ухудшение на 10%", "значительно хуже", "качественно хуже"],
            "больше": ["на 10% больше", "в 2 раза больше", "конкретное количество"],
            "меньше": ["на 10% меньше", "в 2 раза меньше", "конкретное количество"],
            "выше": ["выше на X единиц", "значительно выше", "конкретное значение"],
            "ниже": ["ниже на X единиц", "значительно ниже", "конкретное значение"],
            "сильнее": ["на 20% сильнее", "значительно сильнее", "конкретное значение силы"],
            "слабее": ["на 20% слабее", "значительно слабее", "конкретное значение силы"],
        }
        return expansions.get(comparative.lower(), [f"конкретное значение для {comparative}"])

    def _get_russian_adjective_degrees(self, adj: str) -> List[str]:
        degrees = {
            "яркий": ["интенсивность 80-100%", "интенсивность 50-80%", "конкретное значение яркости"],
            "тусклый": ["интенсивность 0-20%", "интенсивность 20-40%", "конкретное значение яркости"],
            "громкий": ["уровень >80 дБ", "уровень 60-80 дБ", "конкретное значение громкости"],
            "тихий": ["уровень <30 дБ", "уровень 30-50 дБ", "конкретное значение громкости"],
            "быстрый": [">100 км/ч", ">10x от базы", "конкретное значение скорости"],
            "медленный": ["<10 км/ч", "<50% от базы", "конкретное значение скорости"],
            "большой": ["размеры >1м", "размеры >10м", "конкретный размер"],
            "малый": ["размеры <10см", "размеры <1м", "конкретный размер"],
            "маленький": ["размеры <10см", "размеры <1м", "конкретный размер"],
            "огромный": ["размеры >10м", "значительно больше", "конкретный размер"],
            "хороший": ["оценка 8-10/10", "соответствует критериям", "конкретный показатель качества"],
            "плохой": ["оценка 0-3/10", "не соответствует критериям", "конкретный недостаток"],
            "сильный": [">80% мощности", "конкретное значение силы", "высокая корреляция"],
            "слабый": ["<20% мощности", "конкретное значение силы", "низкая корреляция"],
            "высокий": [">1000 м", ">100 единиц", "конкретное значение"],
            "низкий": ["<100 м", "<10 единиц", "конкретное значение"],
            "длинный": [">1 час", ">10 км", "конкретная длина"],
            "короткий": ["<1 минута", "<100 м", "конкретная длина"],
            "старый": [">50 лет", ">10 лет", "конкретный возраст или версия"],
            "новый": ["<1 год", "<6 месяцев", "конкретная дата выпуска"],
            "молодой": ["<30 лет", "конкретный возраст", "моложе X лет"],
            "тёмный": ["яркость 0-20%", "яркость 20-40%", "конкретный уровень темноты"],
            "светлый": ["яркость 80-100%", "яркость 60-80%", "конкретный уровень освещения"],
            "чистый": ["100% чистота", "конкретный показатель чистоты", "класс чистоты"],
            "грязный": ["требует очистки", "конкретный уровень загрязнения", "класс загрязнения"],
        }
        return degrees.get(adj, [f"степень {adj}", f"конкретное измерение {adj}"])
