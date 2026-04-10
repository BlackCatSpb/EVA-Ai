# CogniFlex Context Extraction System - Design Document

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTEXT EXTRACTION SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   INPUT      │    │   ENTITY     │    │   ENTITY     │                   │
│  │   HANDLER    │───▶│   DETECTOR   │───▶│   CLASSIFIER │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                          │
│         │                   │                   ▼                          │
│         │                   │            ┌──────────────┐                  │
│         │                   │            │  CONTEXT     │                  │
│         │                   │            │  RESOLVER    │                  │
│         │                   │            └──────────────┘                  │
│         │                   │                   │                          │
│         │                   ▼                   ▼                          │
│         │            ┌────────────────────────────────┐                   │
│         │            │      INTEGRATION LAYER         │                   │
│         │            │  ┌────────┐ ┌────────┐ ┌─────┐ │                   │
│         │            │  │Memory  │ │Knowledge│ │Fractal│ │                   │
│         │            │  │Graph   │ │Graph   │ │Store │ │                   │
│         │            │  └────────┘ └────────┘ └─────┘ │                   │
│         │            └────────────────────────────────┘                   │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │              OUTPUT GENERATOR                         │                 │
│  │  ┌────────────────┐  ┌────────────────┐               │                 │
│  │  │ClarificationReq│  │RefinementQuery │               │                 │
│  │  └────────────────┘  └────────────────┘               │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Core Data Classes

### 2.1 AmbiguousEntity

```python
@dataclass
class AmbiguousEntity:
    term: str
    entity_type: EntityType  # Enum: AMBIGUOUS_TERM, VAGUE_QUANTIFIER, 
                              #       UNSPECIFIED_REFERENCE, RELATIVE_COMPARISON, 
                              #       IMPLICIT_SUBJECT
    context: str
    position: Tuple[int, int]  # Start, end character positions
    possible_meanings: List[MeaningCandidate]
    confidence: float  # 0.0 - 1.0
    grounding_required: bool
    resolution_source: Optional[ResolutionSource] = None
```

### 2.2 MeaningCandidate

```python
@dataclass
class MeaningCandidate:
    meaning: str
    probability: float
    source: str  # "knowledge_graph", "memory", "context", "inference"
    supporting_evidence: Optional[str] = None
```

### 2.3 ClarificationRequest

```python
@dataclass
class ClarificationRequest:
    request_id: str
    original_entity: AmbiguousEntity
    question: str
    options: Optional[List[str]] = None  # Multiple choice if available
    priority: int  # 1-5, higher = more impactful
    context_summary: str
    generated_at: datetime
```

### 2.4 RefinementQuery

```python
@dataclass
class RefinementQuery:
    query_id: str
    original_query: str
    extracted_entities: List[AmbiguousEntity]
    refinement_type: RefinementType  # Enum: DISAMBIGUATE, SPECIFY, ELABORATE, CORRECT
    context_requirements: List[str]
    priority: int
```

## 3. Processing Pipeline

### Stage 1: Input Processing
```
Input Text → Tokenization → POS Tagging → Dependency Parsing → Named Entity Recognition
```

### Stage 2: Entity Detection
```
Token Stream → Pattern Matcher → Rule Engine → Candidate Generator
```

Detection Patterns:
- Vague quantifiers: regex `(many|few|some|several|several|lots|many|a lot)`
- Relative comparisons: regex `(er|est)` comparative/superlative forms
- Pronouns/References: spaCy POS tags (PRON, DET)
- Implicit subjects: dependency parsing (root without subject)

### Stage 3: Context Collection
```
Entity → Memory Graph Lookup → Knowledge Graph Lookup → Context Aggregation
```

### Stage 4: Disambiguation
```
Context + Candidates → LLM Inference → Probability Scoring → Best Candidate Selection
```

### Stage 5: Output Generation
```
Resolved Entities → ClarificationRequest Generator OR RefinementQuery Generator
```

## 4. Entity Type Handlers

| Entity Type | Detection Strategy | Resolution Strategy |
|-------------|---------------------|---------------------|
| AMBIGUOUS_TERM | Lexicon + Context window | Knowledge graph hyponyms |
| VAGUE_QUANTIFIER | Regex patterns | Question: "How many/amount?" |
| UNSPECIFIED_REFERENCE | Pronoun resolution | Memory graph antecedent search |
| RELATIVE_COMPARISON | Comparative adjective detection | Question: "Compared to what?" |
| IMPLICIT_SUBJECT | Dependency analysis | Question: "What are you referring to?" |

## 5. Integration Points

### 5.1 Memory Graph Integration

```python
class MemoryGraphContextProvider:
    def get_recent_entities(self, entity_type: EntityType, limit: int = 10) -> List[AmbiguousEntity]
    def get_antecedents(self, pronoun: str, context_window: int) -> List[str]
    def store_extracted_entity(self, entity: AmbiguousEntity) -> None
    def get_entity_history(self, term: str) -> List[AmbiguousEntity]
```

### 5.2 Knowledge Graph Integration

```python
class KnowledgeGraphResolver:
    def get_concept_definitions(self, term: str) -> List[str]
    def get_hyponyms(self, hypernym: str) -> List[str]
    def get_related_terms(self, term: str) -> List[str]
    def resolve_semantic_meaning(self, term: str, context: str) -> MeaningCandidate
```

### 5.3 Fractal Storage Integration

```python
class FractalEntityStore:
    def store_at_level(self, entity: AmbiguousEntity, level: int) -> None
    def retrieve_by_level(self, level: int, entity_type: EntityType) -> List[AmbiguousEntity]
    def aggregate_across_levels(self, entity: AmbiguousEntity) -> AmbiguousEntity
```

## 6. Confidence Scoring

```python
def calculate_confidence(entity: AmbiguousEntity, context: Dict) -> float:
    factors = {
        "context_clarity": context.get("sentence_clarity", 0.0),
        "memory_match": 1.0 if context.get("in_memory") else 0.0,
        "knowledge_match": 1.0 if context.get("in_knowledge_graph") else 0.0,
        "linguistic_clarity": context.get("dependency_depth", 0.0),
        "entity_type_weight": ENTITY_TYPE_WEIGHTS.get(entity.entity_type, 0.5)
    }
    return sum(factors.values()) / len(factors)
```

## 7. File Structure

```
cogniflex/knowledge/
├── context_extraction_design.md    # This document
├── context_extractor.py            # Main extraction engine
├── entity_classes.py               # Data classes
├── detectors/
│   ├── __init__.py
│   ├── quantifier_detector.py
│   ├── reference_detector.py
│   ├── comparison_detector.py
│   └── implicit_subject_detector.py
├── resolvers/
│   ├── __init__.py
│   ├── memory_resolver.py
│   ├── knowledge_resolver.py
│   └── context_resolver.py
├── generators/
│   ├── __init__.py
│   ├── clarification_generator.py
│   └── refinement_generator.py
└── integration/
    ├── __init__.py
    ├── memory_graph_integration.py
    ├── knowledge_graph_integration.py
    └── fractal_storage_integration.py
```

## 8. API Interface

```python
class ContextExtractor:
    def extract(self, text: str, source: str = "query") -> List[AmbiguousEntity]
    def generate_clarification(self, entity: AmbiguousEntity) -> ClarificationRequest
    def generate_refinement_query(self, text: str, entities: List[AmbiguousEntity]) -> RefinementQuery
    def resolve_from_history(self, entity: AmbiguousEntity) -> Optional[AmbiguousEntity]
    def integrate_with_pipeline(self, pipeline_output: str) -> List[AmbiguousEntity]
```
