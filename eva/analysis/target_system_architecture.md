# CogniFlex TARGET System Architecture

**Document Version:** 1.0  
**Date:** March 21, 2026  
**Purpose:** Blueprint for CogniFlex Self-Learning Cognitive System

---

## PART 1: CORE PRINCIPLES

### 1.1 Design Philosophy

CogniFlex TARGET is a **Self-Learning Cognitive System** that:
1. **Knows what it knows** - Tracks verified vs generated knowledge
2. **Knows what it doesn't know** - Identifies knowledge gaps
3. **Curious by design** - Actively seeks new information
4. **Transparent** - Shows users confidence levels and knowledge sources

### 1.2 Key Design Decisions

| Current Problem | Target Solution |
|-----------------|-----------------|
| Fake training loops | Real curiosity-driven learning |
| No knowledge awareness | Knowledge status tracking |
| Complex GUI (8 tabs) | Minimal 3-tab GUI |
| Multiple stub methods | Complete implementations |
| Race conditions | Dependency-ordered initialization |
| Duplicate managers | Single source of truth |

---

## PART 2: CORE MODULE SPECIFICATIONS

### 2.1 CuriosityEngine

**File:** `cogniflex/core/curiosity_engine.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
from datetime import datetime
import threading
import queue

class CuriosityType(Enum):
    UNKNOWN_ENTITY = "unknown_entity"           # Entity not in knowledge graph
    KNOWLEDGE_GAP = "knowledge_gap"              # Weak area detected
    NEW_PATTERN = "new_pattern"                  # Novel pattern discovered
    CONTRADICTION_TRIGGER = "contradiction"     # Contradiction found
    USER_INTEREST = "user_interest"              # Topic user cares about

@dataclass
class CuriosityTrigger:
    trigger_id: str
    trigger_type: CuriosityType
    source_text: str
    entities: List[str] = field(default_factory=list)
    topic: str = ""
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0
    status: str = "pending"  # pending, in_progress, completed, dismissed
    
@dataclass  
class SelfQuestion:
    question_id: str
    topic: str
    question_text: str
    expected_outcome: str
    search_queries: List[str]
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)

class CuriosityEngine:
    """
    Drives autonomous learning by detecting knowledge gaps and
    generating self-directed questions.
    """
    
    def __init__(
        self,
        knowledge_graph: 'KnowledgeGraph',
        entity_learner: 'EntityLearner',
        web_search: 'WebSearchEngine',
        memory_manager: 'MemoryManager',
        config: Optional[Dict] = None
    ):
        self._kg = knowledge_graph
        self._entity_learner = entity_learner
        self._web_search = web_search
        self._memory = memory_manager
        self._config = config or {}
        
        # State
        self._triggers: Dict[str, CuriosityTrigger] = {}
        self._questions: Dict[str, SelfQuestion] = {}
        self._pending_learnings: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Configuration
        self._max_concurrent = self._config.get("max_concurrent_learning", 3)
        self._trigger_threshold = self._config.get("trigger_threshold", 0.5)
        self._gap_check_interval = self._config.get("gap_check_interval", 300)
        
    # === Trigger Detection ===
    
    def detect_curiosity_triggers(self, text: str) -> List[CuriosityTrigger]:
        """
        Analyze text and detect potential curiosity triggers.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of CuriosityTrigger objects
        """
        triggers = []
        
        # 1. Extract entities and check if unknown
        entities = self._entity_learner.extract_entities(text)
        for entity in entities:
            if not self._is_entity_known(entity.name):
                trigger = self._create_trigger(
                    CuriosityType.UNKNOWN_ENTITY,
                    text,
                    entities=[e.name for e in entities],
                    topic=entity.name,
                    confidence=0.8
                )
                triggers.append(trigger)
        
        # 2. Check for knowledge gaps in related topics
        gap_score = self.assess_knowledge_gap(text)
        if gap_score < self._trigger_threshold:
            trigger = self._create_trigger(
                CuriosityType.KNOWLEDGE_GAP,
                text,
                topic=self._extract_topic(text),
                confidence=1.0 - gap_score
            )
            triggers.append(trigger)
        
        # 3. Detect unknown terms/patterns
        unknown_terms = self._detect_unknown_terms(text)
        for term in unknown_terms:
            trigger = self._create_trigger(
                CuriosityType.NEW_PATTERN,
                text,
                topic=term,
                confidence=0.6
            )
            triggers.append(trigger)
        
        return triggers
    
    def _is_entity_known(self, entity_name: str) -> bool:
        """Check if entity exists in knowledge graph."""
        if not self._kg:
            return False
        node = self._kg.get_node(entity_name)
        return node is not None
    
    def _detect_unknown_terms(self, text: str) -> List[str]:
        """Detect terms not in vocabulary or knowledge base."""
        # Tokenize and check against known terms
        words = text.lower().split()
        unknown = []
        for word in words:
            if len(word) > 5 and not self._kg.has_term(word):
                unknown.append(word)
        return unknown[:5]  # Limit to 5 per query
    
    def _extract_topic(self, text: str) -> str:
        """Extract primary topic from text."""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.lower().split()
        topics = [w for w in words if len(w) > 4]
        return topics[0] if topics else text[:50]
    
    def _create_trigger(
        self,
        trigger_type: CuriosityType,
        source_text: str,
        entities: List[str] = None,
        topic: str = "",
        confidence: float = 0.5
    ) -> CuriosityTrigger:
        """Factory method for creating triggers."""
        import uuid
        return CuriosityTrigger(
            trigger_id=str(uuid.uuid4())[:8],
            trigger_type=trigger_type,
            source_text=source_text[:200],
            entities=entities or [],
            topic=topic,
            confidence=confidence,
            priority=self._calculate_priority(trigger_type, confidence)
        )
    
    def _calculate_priority(self, trigger_type: CuriosityType, confidence: float) -> int:
        """Calculate priority (higher = more urgent)."""
        type_weights = {
            CuriosityType.UNKNOWN_ENTITY: 3,
            CuriosityType.KNOWLEDGE_GAP: 2,
            CuriosityType.CONTRADICTION_TRIGGER: 4,
            CuriosityType.NEW_PATTERN: 1,
            CuriosityType.USER_INTEREST: 2
        }
        return int(type_weights.get(trigger_type, 1) * confidence * 10)
    
    # === Self-Question Generation ===
    
    def generate_self_questions(self, entity: str) -> List[SelfQuestion]:
        """
        Generate self-directed questions about an entity.
        
        Args:
            entity: Entity name to generate questions about
            
        Returns:
            List of SelfQuestion objects
        """
        questions = []
        import uuid
        
        # Question templates
        templates = [
            {
                "text": f"What is {entity}?",
                "expected": f"Definition and key characteristics of {entity}",
                "queries": [entity, f"{entity} definition", f"what is {entity}"]
            },
            {
                "text": f"Where is {entity} located?",
                "expected": f"Geographic or conceptual location of {entity}",
                "queries": [f"{entity} location", f"{entity} where"]
            },
            {
                "text": f"Who created {entity}?",
                "expected": f"Origin and creator of {entity}",
                "queries": [f"{entity} history", f"{entity} origin", f"{entity} created by"]
            },
            {
                "text": f"How does {entity} work?",
                "expected": f"Functioning and mechanism of {entity}",
                "queries": [f"{entity} how works", f"{entity} mechanism"]
            }
        ]
        
        for i, template in enumerate(templates):
            questions.append(SelfQuestion(
                question_id=f"{entity[:8]}_{i}_{str(uuid.uuid4())[:4]}",
                topic=entity,
                question_text=template["text"],
                expected_outcome=template["expected"],
                search_queries=template["queries"],
                priority=10 - i  # First questions higher priority
            ))
        
        return questions
    
    # === Knowledge Gap Assessment ===
    
    def assess_knowledge_gap(self, topic: str) -> float:
        """
        Assess how much the system knows about a topic.
        
        Args:
            topic: Topic to assess
            
        Returns:
            Float 0.0-1.0 where 1.0 = fully known, 0.0 = unknown
        """
        if not self._kg:
            return 0.5  # Neutral if no KG
        
        # Count connections and depth
        node = self._kg.get_node(topic)
        if not node:
            return 0.0  # Unknown
        
        connections = self._kg.get_connections(topic)
        depth = self._kg.get_depth(topic)
        
        # Normalize scores
        connection_score = min(1.0, len(connections) / 10)
        depth_score = min(1.0, depth / 5)
        recency_score = self._get_recency_score(node)
        
        return (connection_score * 0.4 + depth_score * 0.4 + recency_score * 0.2)
    
    def _get_recency_score(self, node) -> float:
        """Check how recently knowledge was updated."""
        if hasattr(node, 'updated_at'):
            age_days = (datetime.now() - node.updated_at).days
            return max(0.0, 1.0 - (age_days / 90))  # Decay over 90 days
        return 0.5  # Neutral if no timestamp
    
    # === Self-Learning Trigger ===
    
    def trigger_self_learning(self, topic: str, urgency: str = "normal") -> str:
        """
        Queue a topic for self-directed learning.
        
        Args:
            topic: Topic to learn about
            urgency: "low", "normal", "high"
            
        Returns:
            Learning task ID
        """
        import uuid
        task_id = f"learn_{topic[:20]}_{str(uuid.uuid4())[:8]}"
        
        learning_task = {
            "task_id": task_id,
            "topic": topic,
            "urgency": urgency,
            "questions": self.generate_self_questions(topic),
            "created_at": datetime.now().isoformat(),
            "status": "queued"
        }
        
        self._pending_learnings.put(learning_task)
        
        # Start worker if not running
        if not self._running:
            self._start_worker()
        
        return task_id
    
    def _start_worker(self):
        """Start background learning worker."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._learning_worker,
            daemon=True
        )
        self._worker_thread.start()
    
    def _learning_worker(self):
        """Background worker that processes learning tasks."""
        while self._running:
            try:
                task = self._pending_learnings.get(timeout=5)
                self._process_learning_task(task)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Learning worker error: {e}")
    
    def _process_learning_task(self, task: Dict):
        """Process a single learning task."""
        topic = task["topic"]
        questions = task.get("questions", [])
        
        for question in questions[:2]:  # Process top 2 questions
            # Search web
            for query in question.search_queries[:2]:  # Top 2 search queries
                results = self._web_search.search(query, max_results=3)
                if results:
                    # Store verified knowledge
                    self._store_learned_knowledge(
                        topic,
                        question.question_text,
                        results
                    )
                    break  # Got results, move to next question
        
        # Update knowledge graph
        self._kg.refresh_entity(topic)
    
    def _store_learned_knowledge(
        self,
        topic: str,
        question: str,
        search_results: List[Dict]
    ):
        """Store learned knowledge to memory and knowledge graph."""
        if not search_results:
            return
        
        # Extract key facts from results
        facts = []
        for result in search_results[:3]:
            if "snippet" in result:
                facts.append(result["snippet"])
            elif "content" in result:
                facts.append(result["content"])
        
        # Store to memory
        if self._memory:
            self._memory.store_insight(
                topic=topic,
                insight="\n".join(facts),
                source="web_search",
                confidence=0.85
            )
        
        # Update knowledge graph
        if self._kg:
            self._kg.add_verified_knowledge(topic, facts)
    
    # === Status and Control ===
    
    def get_pending_triggers(self) -> List[CuriosityTrigger]:
        """Get all pending curiosity triggers."""
        with self._lock:
            return [t for t in self._triggers.values() 
                   if t.status == "pending"]
    
    def get_learning_stats(self) -> Dict:
        """Get current learning statistics."""
        with self._lock:
            return {
                "pending_triggers": len([t for t in self._triggers.values() 
                                        if t.status == "pending"]),
                "in_progress": len([t for t in self._triggers.values() 
                                  if t.status == "in_progress"]),
                "completed": len([t for t in self._triggers.values() 
                                if t.status == "completed"]),
                "queued_tasks": self._pending_learnings.qsize(),
                "running": self._running
            }
    
    def stop(self):
        """Stop the curiosity engine."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
```

---

### 2.2 KnowledgeAwareness

**File:** `cogniflex/core/knowledge_awareness.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from datetime import datetime
import threading
import hashlib

class KnowledgeSource(Enum):
    VERIFIED_WEB = "verified_web"           # From web search
    VERIFIED_WIKIPEDIA = "verified_wikipedia" # From Wikipedia
    VERIFIED_MEMORY = "verified_memory"     # From stored memories
    VERIFIED_KG = "verified_kg"             # From knowledge graph
    GENERATED = "generated"                  # Model inference
    UNKNOWN = "unknown"                       # Source unclear

class KnowledgeStatus(Enum):
    VERIFIED = "verified"     # Source confirmed
    GENERATED = "generated"  # Model created
    UNCERTAIN = "uncertain"  # Low confidence

@dataclass
class KnowledgeMark:
    """Marks a piece of knowledge with its source and confidence."""
    text_hash: str
    original_text: str
    source: KnowledgeSource
    status: KnowledgeStatus
    confidence: float  # 0.0 - 1.0
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    
@dataclass
class ResponseAnnotation:
    """Annotates a response with knowledge markers."""
    full_text: str
    segments: List[Dict] = field(default_factory=list)  # List of (text, mark)
    verified_count: int = 0
    generated_count: int = 0
    
    @property
    def verified_ratio(self) -> float:
        total = self.verified_count + self.generated_count
        return self.verified_count / total if total > 0 else 0.0

class KnowledgeAwareness:
    """
    Tracks which knowledge is verified vs generated.
    Provides transparency to users about knowledge sources.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        
        # Knowledge marks storage (in-memory for speed)
        self._marks: Dict[str, KnowledgeMark] = {}
        
        # Confidence thresholds
        self._verified_threshold = self._config.get("verified_threshold", 0.7)
        self._generated_threshold = self._config.get("generated_threshold", 0.5)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
    # === Marking Knowledge ===
    
    def mark_as_generated(
        self,
        text: str,
        confidence: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Mark text as generated by the model.
        
        Args:
            text: Text to mark
            confidence: Model confidence in generation (0.0-1.0)
            metadata: Optional additional metadata
            
        Returns:
            Text hash for reference
        """
        text_hash = self._compute_hash(text)
        
        with self._lock:
            self._marks[text_hash] = KnowledgeMark(
                text_hash=text_hash,
                original_text=text[:500],  # Store truncated
                source=KnowledgeSource.GENERATED,
                status=KnowledgeStatus.GENERATED,
                confidence=confidence,
                verified_at=datetime.now()
            )
        
        return text_hash
    
    def mark_as_verified(
        self,
        text: str,
        source: str,
        citations: Optional[List[str]] = None,
        confidence: float = 0.9
    ) -> str:
        """
        Mark text as verified from a source.
        
        Args:
            text: Verified text
            source: Source identifier (e.g., "wikipedia", "web_search")
            citations: List of source URLs/references
            confidence: Verification confidence (0.0-1.0)
            
        Returns:
            Text hash for reference
        """
        text_hash = self._compute_hash(text)
        
        # Determine source type
        source_type = self._map_source_type(source)
        
        with self._lock:
            self._marks[text_hash] = KnowledgeMark(
                text_hash=text_hash,
                original_text=text[:500],
                source=source_type,
                status=KnowledgeStatus.VERIFIED,
                confidence=confidence,
                verified_at=datetime.now(),
                verified_by=source,
                citations=citations or []
            )
        
        return text_hash
    
    def _map_source_type(self, source: str) -> KnowledgeSource:
        """Map source string to KnowledgeSource enum."""
        source_lower = source.lower()
        if "wikipedia" in source_lower:
            return KnowledgeSource.VERIFIED_WIKIPEDIA
        elif "web" in source_lower or "search" in source_lower:
            return KnowledgeSource.VERIFIED_WEB
        elif "memory" in source_lower:
            return KnowledgeSource.VERIFIED_MEMORY
        elif "kg" in source_lower or "graph" in source_lower:
            return KnowledgeSource.VERIFIED_KG
        return KnowledgeSource.UNKNOWN
    
    def _compute_hash(self, text: str) -> str:
        """Compute short hash for text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    # === Querying Knowledge Status ===
    
    def get_knowledge_status(self, text: str) -> Dict:
        """
        Get status information for a piece of text.
        
        Args:
            text: Text to check
            
        Returns:
            Dict with status, source, confidence, citations
        """
        text_hash = self._compute_hash(text)
        
        with self._lock:
            if text_hash in self._marks:
                mark = self._marks[text_hash]
                return {
                    "status": mark.status.value,
                    "source": mark.source.value,
                    "confidence": mark.confidence,
                    "verified_at": mark.verified_at.isoformat() if mark.verified_at else None,
                    "citations": mark.citations
                }
        
        return {
            "status": "unknown",
            "source": "unknown",
            "confidence": 0.0,
            "verified_at": None,
            "citations": []
        }
    
    def get_confidence_level(self, text: str) -> float:
        """
        Get confidence level for text.
        
        Args:
            text: Text to check
            
        Returns:
            Confidence 0.0-1.0
        """
        text_hash = self._compute_hash(text)
        
        with self._lock:
            if text_hash in self._marks:
                return self._marks[text_hash].confidence
        
        return 0.0
    
    def get_source(self, text: str) -> Optional[str]:
        """Get source of text if known."""
        text_hash = self._compute_hash(text)
        
        with self._lock:
            if text_hash in self._marks:
                return self._marks[text_hash].source.value
        
        return None
    
    # === Response Annotation ===
    
    def annotate_response(
        self,
        response_text: str,
        web_results: Optional[List[Dict]] = None,
        memory_results: Optional[List[Dict]] = None,
        kg_results: Optional[List[Dict]] = None,
        generated_segments: Optional[List[str]] = None
    ) -> ResponseAnnotation:
        """
        Annotate a full response with knowledge markers.
        
        Args:
            response_text: Full response text
            web_results: Web search results used
            memory_results: Memory retrieval results
            kg_results: Knowledge graph results
            generated_segments: Model-generated segments
            
        Returns:
            ResponseAnnotation with segment markers
        """
        segments = []
        verified_count = 0
        generated_count = 0
        
        # Build known verified sources set
        verified_sources = set()
        
        if web_results:
            for result in web_results:
                snippet = result.get("snippet", "")
                if snippet:
                    hash_key = self._compute_hash(snippet)
                    verified_sources.add(hash_key)
                    segments.append({
                        "text": snippet,
                        "status": "verified",
                        "source": "web",
                        "hash": hash_key
                    })
                    verified_count += 1
        
        if kg_results:
            for result in kg_results:
                fact = result.get("fact", result.get("text", ""))
                if fact:
                    hash_key = self._compute_hash(fact)
                    verified_sources.add(hash_key)
                    segments.append({
                        "text": fact,
                        "status": "verified",
                        "source": "knowledge_graph",
                        "hash": hash_key
                    })
                    verified_count += 1
        
        if generated_segments:
            for seg in generated_segments:
                hash_key = self._compute_hash(seg)
                if hash_key not in verified_sources:
                    segments.append({
                        "text": seg,
                        "status": "generated",
                        "source": "model",
                        "hash": hash_key
                    })
                    generated_count += 1
                    # Mark as generated
                    self.mark_as_generated(seg, 0.7)
        
        return ResponseAnnotation(
            full_text=response_text,
            segments=segments,
            verified_count=verified_count,
            generated_count=generated_count
        )
    
    def get_knowledge_stats(self) -> Dict:
        """Get overall knowledge statistics."""
        with self._lock:
            total = len(self._marks)
            verified = sum(1 for m in self._marks.values() 
                          if m.status == KnowledgeStatus.VERIFIED)
            generated = sum(1 for m in self._marks.values() 
                          if m.status == KnowledgeStatus.GENERATED)
            
            avg_confidence = sum(m.confidence for m in self._marks.values()) / total if total > 0 else 0
            
            return {
                "total_marks": total,
                "verified": verified,
                "generated": generated,
                "verified_ratio": verified / total if total > 0 else 0,
                "average_confidence": avg_confidence,
                "by_source": self._get_source_breakdown()
            }
    
    def _get_source_breakdown(self) -> Dict[str, int]:
        """Get breakdown of sources."""
        breakdown = {}
        for mark in self._marks.values():
            source = mark.source.value
            breakdown[source] = breakdown.get(source, 0) + 1
        return breakdown
    
    def clear_old_marks(self, max_age_days: int = 30):
        """Remove old marks to prevent memory bloat."""
        import time
        
        with self._lock:
            cutoff = datetime.now().timestamp() - (max_age_days * 86400)
            to_remove = []
            
            for text_hash, mark in self._marks.items():
                if mark.verified_at and mark.verified_at.timestamp() < cutoff:
                    to_remove.append(text_hash)
            
            for text_hash in to_remove:
                del self._marks[text_hash]
            
            return len(to_remove)
```

---

### 2.3 EntityLearner

**File:** `cogniflex/learning/entity_learner.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum
import threading

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    EVENT = "event"
    UNKNOWN = "unknown"

@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    knowledge_level: float = 0.0  # 0.0-1.0 how much we know
    sources: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)

@dataclass
class EntityKnowledge:
    """Knowledge about an entity."""
    entity: Entity
    facts: List[Dict] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    learning_history: List[Dict] = field(default_factory=list)

class EntityLearner:
    """
    Extracts entities from queries and autonomously learns about them.
    """
    
    def __init__(
        self,
        knowledge_graph: 'KnowledgeGraph',
        web_search: 'WebSearchEngine',
        online_access: 'OnlineKnowledgeAccess',
        memory_manager: 'MemoryManager',
        config: Optional[Dict] = None
    ):
        self._kg = knowledge_graph
        self._web_search = web_search
        self._online = online_access
        self._memory = memory_manager
        self._config = config or {}
        
        # Entity cache
        self._entities: Dict[str, Entity] = {}
        self._learning_queue: List[str] = []
        self._lock = threading.RLock()
        
        # Settings
        self._min_confidence = self._config.get("min_entity_confidence", 0.6)
        self._knowledge_threshold = self._config.get("knowledge_threshold", 0.5)
    
    # === Entity Extraction ===
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted Entity objects
        """
        entities = []
        
        # Method 1: Knowledge Graph lookup
        kg_entities = self._kg.extract_entities(text) if self._kg else []
        for name in kg_entities:
            entity = self._get_or_create_entity(name)
            entities.append(entity)
        
        # Method 2: Pattern-based extraction (capitalized words, numbers, etc.)
        pattern_entities = self._extract_pattern_based(text)
        for name in pattern_entities:
            if not any(e.name.lower() == name.lower() for e in entities):
                entity = self._get_or_create_entity(name, EntityType.UNKNOWN)
                entities.append(entity)
        
        # Method 3: Known entity database
        known_entities = self._extract_known_entities(text)
        for name in known_entities:
            if not any(e.name.lower() == name.lower() for e in entities):
                entity = self._get_or_create_entity(name)
                entity.knowledge_level = self._assess_entity_knowledge(entity.name)
                entities.append(entity)
        
        return entities
    
    def _extract_pattern_based(self, text: str) -> List[str]:
        """Extract entities using simple patterns."""
        import re
        
        entities = []
        
        # Capitalized word patterns
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend([c for c in capitalized if len(c) > 2][:10])
        
        # Numbers with units or context
        numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:km|mb|gb|years?|people)\b', text)
        entities.extend(numbers[:5])
        
        return entities
    
    def _extract_known_entities(self, text: str) -> List[str]:
        """Extract from known entity list."""
        with self._lock:
            found = []
            text_lower = text.lower()
            
            for name, entity in self._entities.items():
                if name.lower() in text_lower:
                    found.append(name)
            
            return found
    
    def _get_or_create_entity(self, name: str, 
                             entity_type: EntityType = EntityType.UNKNOWN) -> Entity:
        """Get existing entity or create new one."""
        with self._lock:
            name_lower = name.lower()
            
            if name_lower in self._entities:
                return self._entities[name_lower]
            
            entity = Entity(
                name=name,
                entity_type=entity_type
            )
            self._entities[name_lower] = entity
            return entity
    
    # === Learning About Entities ===
    
    def learn_about_entity(self, entity: Entity, depth: str = "basic") -> EntityKnowledge:
        """
        Autonomously learn about an entity via web search.
        
        Args:
            entity: Entity to learn about
            depth: "basic", "detailed", "comprehensive"
            
        Returns:
            EntityKnowledge with learned information
        """
        facts = []
        relationships = []
        
        # Search Wikipedia
        if self._online:
            wiki_result = self._online.search_wikipedia(entity.name)
            if wiki_result:
                facts.extend(self._extract_facts_from_wiki(wiki_result))
                entity.sources.append("wikipedia")
        
        # Search general web
        web_results = self._web_search.search(
            f"{entity.name} information facts",
            max_results=10 if depth == "comprehensive" else 5
        )
        for result in web_results:
            if "snippet" in result:
                facts.append({
                    "text": result["snippet"],
                    "source": result.get("url", "web"),
                    "relevance": result.get("score", 0.5)
                })
            entity.sources.append("web_search")
        
        # Extract and store relationships
        if self._kg:
            relationships = self._kg.get_entity_relationships(entity.name)
        
        # Update entity knowledge level
        entity.knowledge_level = min(1.0, len(facts) / 10)
        entity.last_updated = datetime.now()
        
        # Store to knowledge graph
        self.update_entity_in_graph(entity, {"facts": facts, "relationships": relationships})
        
        # Store to memory
        if self._memory:
            for fact in facts[:5]:
                self._memory.store_insight(
                    topic=entity.name,
                    insight=fact.get("text", ""),
                    source=fact.get("source", "entity_learner"),
                    confidence=fact.get("relevance", 0.7)
                )
        
        return EntityKnowledge(
            entity=entity,
            facts=facts,
            relationships=relationships,
            learning_history=[{
                "timestamp": datetime.now().isoformat(),
                "depth": depth,
                "facts_found": len(facts)
            }]
        )
    
    def _extract_facts_from_wiki(self, wiki_result: Dict) -> List[Dict]:
        """Extract structured facts from Wikipedia result."""
        facts = []
        
        if "summary" in wiki_result:
            facts.append({
                "text": wiki_result["summary"],
                "source": "wikipedia",
                "type": "summary",
                "relevance": 0.9
            })
        
        if "infobox" in wiki_result:
            for key, value in wiki_result["infobox"].items():
                facts.append({
                    "text": f"{key}: {value}",
                    "source": "wikipedia_infobox",
                    "type": "property",
                    "relevance": 0.85
                })
        
        if "sections" in wiki_result:
            for section in wiki_result["sections"][:3]:
                facts.append({
                    "text": f"{section.get('title', '')}: {section.get('content', '')[:200]}",
                    "source": "wikipedia_section",
                    "type": "detail",
                    "relevance": 0.7
                })
        
        return facts
    
    # === Entity Knowledge Status ===
    
    def check_entity_knowledge(self, entity_name: str) -> Dict:
        """
        Check how much knowledge we have about an entity.
        
        Args:
            entity_name: Entity to check
            
        Returns:
            Dict with knowledge status
        """
        with self._lock:
            entity_lower = entity_name.lower()
            
            if entity_lower not in self._entities:
                return {
                    "known": False,
                    "knowledge_level": 0.0,
                    "sources": [],
                    "needs_learning": True
                }
            
            entity = self._entities[entity_lower]
            
            return {
                "known": True,
                "name": entity.name,
                "entity_type": entity.entity_type.value,
                "knowledge_level": entity.knowledge_level,
                "sources": entity.sources,
                "needs_learning": entity.knowledge_level < self._knowledge_threshold,
                "first_seen": entity.first_seen.isoformat(),
                "last_updated": entity.last_updated.isoformat()
            }
    
    def update_entity_in_graph(self, entity: Entity, knowledge: Dict):
        """Update entity in knowledge graph."""
        if not self._kg:
            return
        
        # Add/update node
        self._kg.add_entity(
            name=entity.name,
            entity_type=entity.entity_type.value,
            properties={
                "knowledge_level": entity.knowledge_level,
                "first_seen": entity.first_seen.isoformat(),
                "last_updated": entity.last_updated.isoformat(),
                "sources": entity.sources
            }
        )
        
        # Add facts as properties or related nodes
        facts = knowledge.get("facts", [])
        for i, fact in enumerate(facts[:20]):  # Limit to 20 facts
            fact_text = fact.get("text", "")
            if fact_text:
                self._kg.add_fact(
                    entity_name=entity.name,
                    fact=fact_text,
                    source=fact.get("source", "entity_learner"),
                    confidence=fact.get("relevance", 0.7)
                )
        
        # Add relationships
        relationships = knowledge.get("relationships", [])
        for rel in relationships:
            if "related_to" in rel and "relationship_type" in rel:
                self._kg.add_relationship(
                    from_entity=entity.name,
                    to_entity=rel["related_to"],
                    relationship_type=rel["relationship_type"]
                )
    
    def _assess_entity_knowledge(self, entity_name: str) -> float:
        """Assess current knowledge level about entity."""
        if not self._kg:
            return 0.0
        
        connections = self._kg.get_connections(entity_name)
        facts = self._kg.get_facts(entity_name)
        
        # Normalize to 0-1
        conn_score = min(1.0, len(connections) / 10)
        fact_score = min(1.0, len(facts) / 20)
        
        return (conn_score * 0.4 + fact_score * 0.6)
    
    # === Batch Operations ===
    
    def queue_learning(self, entity_name: str, depth: str = "basic"):
        """Queue entity for learning."""
        with self._lock:
            if entity_name not in self._learning_queue:
                self._learning_queue.append(entity_name)
    
    def process_learning_queue(self, max_items: int = 5) -> int:
        """Process queued learning items."""
        processed = 0
        
        with self._lock:
            items = self._learning_queue[:max_items]
            self._learning_queue = self._learning_queue[max_items:]
        
        for entity_name in items:
            entity = self._get_or_create_entity(entity_name)
            self.learn_about_entity(entity, depth="basic")
            processed += 1
        
        return processed
    
    def get_entity_stats(self) -> Dict:
        """Get entity learning statistics."""
        with self._lock:
            total = len(self._entities)
            by_type = {}
            total_knowledge = 0.0
            
            for entity in self._entities.values():
                by_type[entity.entity_type.value] = by_type.get(
                    entity.entity_type.value, 0) + 1
                total_knowledge += entity.knowledge_level
            
            return {
                "total_entities": total,
                "by_type": by_type,
                "average_knowledge_level": total_knowledge / total if total > 0 else 0,
                "queued_for_learning": len(self._learning_queue)
            }
```

---

### 2.4 OnlineKnowledgeAccess

**File:** `cogniflex/websearch/online_knowledge_access.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

@dataclass
class FactVerification:
    """Result of fact verification."""
    fact: str
    verified: bool
    confidence: float
    sources: List[str]
    conflicting_info: List[str] = None

@dataclass
class WikipediaResult:
    """Wikipedia search result."""
    title: str
    page_id: str
    summary: str
    url: str
    infobox: Dict = None
    sections: List[Dict] = None
    last_updated: datetime = None

class OnlineKnowledgeAccess:
    """
    Provides structured access to online knowledge sources.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = self._config.get("cache_ttl", 3600)  # 1 hour
        
        # Rate limiting
        self._request_times: List[datetime] = []
        self._max_requests_per_minute = 30
        
    # === Wikipedia Access ===
    
    def search_wikipedia(self, query: str) -> Optional[WikipediaResult]:
        """
        Search Wikipedia for a query.
        
        Args:
            query: Search query
            
        Returns:
            WikipediaResult or None if not found
        """
        cache_key = f"wiki:{query}"
        if cache_key in self._cache:
            return WikipediaResult(**self._cache[cache_key])
        
        # Use Wikipedia API
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        
        try:
            import requests
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                result = WikipediaResult(
                    title=data.get("title", query),
                    page_id=str(data.get("pageid", "")),
                    summary=data.get("extract", ""),
                    url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    infobox={},
                    sections=[],
                    last_updated=datetime.now()
                )
                
                self._cache[cache_key] = {
                    "title": result.title,
                    "page_id": result.page_id,
                    "summary": result.summary,
                    "url": result.url,
                    "infobox": {},
                    "sections": [],
                    "last_updated": result.last_updated.isoformat()
                }
                
                return result
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return None
    
    def search_wikipedia_full(self, query: str) -> Optional[WikipediaResult]:
        """Get full Wikipedia article."""
        cache_key = f"wiki_full:{query}"
        if cache_key in self._cache:
            return WikipediaResult(**self._cache[cache_key])
        
        wiki_result = self.search_wikipedia(query)
        if not wiki_result:
            return None
        
        # Try to get more sections
        url = f"https://en.wikipedia.org/api/rest_v1/page/html/{query.replace(' ', '_')}"
        
        try:
            import requests
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                # Parse HTML for sections (simplified)
                sections = self._parse_wiki_html(response.text)
                wiki_result.sections = sections
                
        except Exception as e:
            print(f"Wikipedia full fetch error: {e}")
        
        return wiki_result
    
    def _parse_wiki_html(self, html: str) -> List[Dict]:
        """Parse Wikipedia HTML for sections."""
        import re
        
        sections = []
        
        # Simple h2/h3 extraction
        headings = re.findall(r'<h[23][^>]*>([^<]+)</h[23]>', html)
        for i, heading in enumerate(headings[:10]):
            sections.append({
                "title": heading.strip(),
                "content": ""  # Would need more parsing
            })
        
        return sections
    
    # === General Online Search ===
    
    def search_online_db(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search multiple online databases/sources.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        # 1. Wikipedia
        wiki = self.search_wikipedia(query)
        if wiki and wiki.summary:
            results.append({
                "source": "wikipedia",
                "title": wiki.title,
                "snippet": wiki.summary,
                "url": wiki.url,
                "relevance": 0.9
            })
        
        # 2. DuckDuckGo (via web search engine if available)
        # This would integrate with existing WebSearchEngine
        
        return results[:max_results]
    
    def fetch_article_summary(self, url: str) -> str:
        """
        Fetch summary from article URL.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted summary text
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for tag in soup(["script", "style"]):
                tag.decompose()
            
            # Get text
            text = soup.get_text(separator="\n", strip=True)
            
            # Return first 500 chars
            return text[:500]
            
        except Exception as e:
            return f"Error fetching article: {e}"
    
    # === Fact Verification ===
    
    def verify_fact(self, fact: str, context: Optional[str] = None) -> FactVerification:
        """
        Verify a factual claim.
        
        Args:
            fact: Fact to verify
            context: Optional context
            
        Returns:
            FactVerification with results
        """
        # Extract key claim
        claim = self._extract_claim(fact)
        
        # Search for contradicting information
        search_results = self.search_online_db(claim, max_results=5)
        
        # Check for contradictions
        conflicting = []
        verified = False
        confidence = 0.5
        
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            claim_lower = claim.lower()
            
            # Simple contradiction detection
            negations = ["not", "never", "no", "doesn't", "isn't", "wasn't", "won't"]
            if any(neg in snippet[:50] for neg in negations):
                if claim_lower[:30] in snippet:
                    conflicting.append(result["snippet"])
            else:
                # Supporting evidence
                if claim_lower[:20] in snippet:
                    confidence = max(confidence, result.get("relevance", 0.7))
                    verified = True
        
        return FactVerification(
            fact=fact,
            verified=verified,
            confidence=confidence,
            sources=[r.get("url", "") for r in search_results],
            conflicting_info=conflicting if conflicting else None
        )
    
    def _extract_claim(self, fact: str) -> str:
        """Extract the core claim from text."""
        import re
        
        # Remove punctuation
        claim = re.sub(r'[^\w\s]', '', fact)
        
        # Take first 100 chars
        return claim[:100]
    
    # === Cache Management ===
    
    def clear_cache(self):
        """Clear the knowledge cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "ttl": self._cache_ttl
        }
```

---

## PART 3: INTEGRATED DATA FLOW

### 3.1 Complete Query Processing Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     QueryProcessor                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Entity Extraction ──────► EntityLearner                    │
│  2. Knowledge Gap Check ────► CuriosityEngine                   │
│  3. Trigger Detection ───────► CuriosityEngine                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CuriosityEngine                             │
├─────────────────────────────────────────────────────────────────┤
│  If triggers found:                                              │
│    ├─► Generate self-questions                                  │
│    ├─► Queue for background learning                            │
│    └─► EntityLearner.learn_about_entity()                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KnowledgeAwareness                           │
├─────────────────────────────────────────────────────────────────┤
│  Mark response segments:                                        │
│    ├─► VERIFIED: From web/memory/kg                              │
│    └─► GENERATED: From model inference                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ResponseGenerator                            │
├─────────────────────────────────────────────────────────────────┤
│  Generate response with:                                        │
│    ├─► Verified knowledge (high confidence)                     │
│    ├─► Generated knowledge (medium confidence)                   │
│    └─► Knowledge status annotations                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
User Response (with knowledge indicators)
```

### 3.2 Module Dependencies

```
                    ┌──────────────┐
                    │   CoreBrain  │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Query       │ │  Curiosity    │ │  Knowledge    │
│   Processor   │ │   Engine      │ │  Awareness    │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ EntityLearner │ │ OnlineKnowledge│ │ MemoryManager │
│   (NEW)        │ │    Access      │ │               │
└───────┬───────┘ └───────┬───────┘ └───────────────┘
        │                  │
        ▼                  ▼
┌───────────────┐ ┌───────────────┐
│ KnowledgeGraph│ │ WebSearchEngine│
│               │ │               │
└───────────────┘ └───────────────┘
```

---

## PART 4: MINIMAL GUI ARCHITECTURE

### 4.1 Tab Structure (3 Tabs)

```
┌────────────────────────────────────────────────────────┐
│  CogniFlex                                      [─][□][×]│
├────────────────────────────────────────────────────────┤
│  [ Chat ]  [ Memory ]  [ Status ]                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│                   TAB CONTENT                          │
│                                                        │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 4.2 Tab Specifications

#### Tab 1: Chat (PRIMARY)
**File:** `cogniflex/gui/tabs/chat_tab.py`

| Component | Purpose |
|-----------|---------|
| Message list | Display chat history |
| Input field | User message input |
| Send button | Submit message |
| **Knowledge indicators** | Show verified/generated status |
| Context indicator | Show active context |

**Knowledge Indicator Format:**
```
┌─────────────────────────────────────────────────────┐
│ 📚 [Verified: 85%] - From Wikipedia + Web Search   │
│ 💭 [Generated: 60%] - Model inference              │
└─────────────────────────────────────────────────────┘
```

#### Tab 2: Memory (IMPORTANT)
**File:** `cogniflex/gui/tabs/memory_tab.py`

| Component | Purpose |
|-----------|---------|
| Learned entities list | Show all learned entities |
| Entity details panel | Show entity knowledge |
| Learning queue | Show pending learning tasks |
| Stats summary | Learning statistics |

#### Tab 3: Status (NECESSARY)
**File:** `cogniflex/gui/tabs/status_tab.py`

| Component | Purpose |
|-----------|---------|
| System health | Overall system status |
| Component status | Individual component health |
| Learning stats | Curiosity engine stats |
| Knowledge stats | Knowledge awareness stats |

### 4.3 Deleted Tabs

| Old Tab | New Location |
|---------|--------------|
| Analytics | → Merged into Status tab |
| Knowledge Graph | → Integrated into Memory tab |
| Learning | → Integrated into Chat tab |
| Contradictions | → Removed (behind-the-scenes) |
| Neuromorphic | → Removed (visual only) |
| Settings | → Simplified into Status tab |

### 4.4 GUI File Structure

```
cogniflex/gui/
├── core_gui.py              # Main orchestrator (REDUCED)
├── base_tab.py              # Base class for tabs
├── tabs/
│   ├── __init__.py
│   ├── chat_tab.py          # NEW - consolidated chat
│   ├── memory_tab.py         # NEW - consolidated memory
│   └── status_tab.py        # NEW - system status
├── widgets/
│   ├── __init__.py
│   ├── chat_widgets.py      # Chat-specific widgets
│   ├── knowledge_indicator.py # NEW - knowledge status
│   └── common.py            # Shared widgets
└── themes/
    └── themes.py            # Theme definitions
```

---

## PART 5: IMPLEMENTATION ROADMAP

### Phase 1: Core Modules (Week 1)
- [ ] Implement `CuriosityEngine`
- [ ] Implement `KnowledgeAwareness`
- [ ] Implement `EntityLearner`
- [ ] Implement `OnlineKnowledgeAccess`
- [ ] Add to `CoreBrain` initialization

### Phase 2: Integration (Week 2)
- [ ] Integrate `CuriosityEngine` into `QueryProcessor`
- [ ] Integrate `KnowledgeAwareness` into `ResponseGenerator`
- [ ] Integrate `EntityLearner` into query pipeline
- [ ] Connect all modules to `CoreBrain`

### Phase 3: GUI Simplification (Week 3)
- [ ] Create new tab structure
- [ ] Implement `chat_tab.py`
- [ ] Implement `memory_tab.py`
- [ ] Implement `status_tab.py`
- [ ] Delete old tab files
- [ ] Add knowledge indicators to chat

### Phase 4: Cleanup (Week 4)
- [ ] Remove stub methods
- [ ] Fix race conditions
- [ ] Unify duplicate managers
- [ ] Remove unused files
- [ ] Update documentation

---

## PART 6: API REFERENCE

### 6.1 CoreBrain Interface Changes

```python
# New methods to add to CoreBrain
class CoreBrain:
    def get_curiosity_engine(self) -> CuriosityEngine:
        """Get the curiosity engine instance."""
        
    def get_knowledge_awareness(self) -> KnowledgeAwareness:
        """Get the knowledge awareness instance."""
        
    def get_entity_learner(self) -> EntityLearner:
        """Get the entity learner instance."""
        
    def trigger_learning(self, topic: str, urgency: str = "normal") -> str:
        """Manually trigger learning about a topic."""
        
    def get_learning_stats(self) -> Dict:
        """Get overall learning statistics."""
```

### 6.2 Event System Integration

```python
# New events
CURIOSITY_TRIGGERED = "curiosity_triggered"
LEARNING_STARTED = "learning_started"
LEARNING_COMPLETED = "learning_completed"
KNOWLEDGE_VERIFIED = "knowledge_verified"
KNOWLEDGE_GENERATED = "knowledge_generated"

# Event payloads
{
    "event": CURIOSITY_TRIGGERED,
    "data": {
        "trigger_id": "...",
        "trigger_type": "unknown_entity",
        "topic": "...",
        "confidence": 0.8
    }
}
```

### 6.3 Configuration Schema

```json
{
  "curiosity_engine": {
    "enabled": true,
    "max_concurrent_learning": 3,
    "trigger_threshold": 0.5,
    "gap_check_interval": 300
  },
  "knowledge_awareness": {
    "enabled": true,
    "verified_threshold": 0.7,
    "generated_threshold": 0.5,
    "show_indicators": true
  },
  "entity_learner": {
    "enabled": true,
    "min_entity_confidence": 0.6,
    "knowledge_threshold": 0.5,
    "auto_learn": true
  },
  "online_knowledge": {
    "enabled": true,
    "cache_ttl": 3600,
    "max_requests_per_minute": 30,
    "wikipedia_enabled": true
  },
  "gui": {
    "show_knowledge_indicators": true,
    "simplified_mode": true
  }
}
```

---

## PART 7: TESTING STRATEGY

### Unit Tests
- `test_curiosity_engine.py` - Trigger detection, question generation
- `test_knowledge_awareness.py` - Marking, annotation, status
- `test_entity_learner.py` - Extraction, learning, updates
- `test_online_knowledge.py` - Wikipedia access, fact verification

### Integration Tests
- `test_query_with_curiosity.py` - Full pipeline with learning
- `test_knowledge_indicators.py` - Response annotation

### GUI Tests
- `test_chat_tab.py` - Knowledge indicator display
- `test_memory_tab.py` - Entity list and details
- `test_status_tab.py` - Statistics display

---

*End of Target Architecture Document*
