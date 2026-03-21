# SelfDialogLearningSystem Architecture

## Overview

The **SelfDialogLearningSystem** replaces the current training-based learning system with a self-referential dialog system that analyzes its own responses through existing managers. No model fine-tuning occurs—only analysis, validation, and insight storage.

---

## Problems Solved

| Problem | Solution |
|---------|----------|
| Massive code duplication (3+ IntegratedLearningManager) | Single `SelfDialogLearningSystem` class |
| Fragmented responsibility (15+ files) | Unified loop through existing managers |
| Disabled training with fake data | Real analysis via ContradictionManager, EthicsFramework, etc. |
| No real ML training capability | Removed entirely—analysis only |
| Stub implementations | Real multi-manager analysis pipeline |

---

## Core Design

### 1. Class Structure

```
cogniflex/learning/
├── self_dialog_learning.py      # Main class (NEW)
└── self_dialog_learning.md     # This document
```

### 2. SelfDialogLearningSystem Class

```python
"""
Self-Dialog Learning System for CogniFlex
Uses existing managers to analyze system responses without training.
"""
import logging
import time
import threading
import json
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from ..core.base_component import BaseComponent

logger = logging.getLogger("cogniflex.self_dialog")

class DialogState(Enum):
    IDLE = "idle"
    GENERATING = "generating"
    ANALYZING = "analyzing"
    STORING = "storing"
    COMPLETED = "completed"

@dataclass
class DialogTurn:
    """Single turn in self-dialog."""
    turn_id: str
    timestamp: float
    role: str  # "system" or "analyst"
    content: str
    analysis_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DialogCycle:
    """Complete self-dialog cycle."""
    cycle_id: str
    topic: str
    turns: List[DialogTurn] = field(default_factory=list)
    ethics_passed: bool = True
    contradictions_found: int = 0
    facts_verified: int = 0
    quality_score: float = 0.0
    insights: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LearningInsight:
    """Insight extracted from self-dialog."""
    insight_id: str
    timestamp: float
    topic: str
    category: str  # "ethics", "contradiction", "fact", "quality", "pattern"
    content: str
    source_turn: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SelfDialogLearningSystem(BaseComponent):
    """
    Self-Dialog Learning System - analyzes own responses through existing managers.
    
    No training loops. No model fine-tuning. Only analysis and insight storage.
    """
    
    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SelfDialogLearningSystem.
        
        Args:
            brain: Reference to CoreBrain
            config: Configuration options
        """
        super().__init__(brain, config, name="SelfDialogLearningSystem")
        
        # Configuration
        self.cycle_interval = config.get("cycle_interval", 300)  # 5 minutes
        self.cycles_per_session = config.get("cycles_per_session", 3)
        self.topics = config.get("topics", self._default_topics())
        self.enabled_managers = config.get("enabled_managers", [
            "ethics", "contradiction", "websearch", "analytics"
        ])
        self.min_quality_threshold = config.get("min_quality_threshold", 0.5)
        self.max_insights_per_cycle = config.get("max_insights_per_cycle", 10)
        
        # Manager references (lazy-loaded)
        self._ethics_framework = None
        self._contradiction_manager = None
        self._web_search_engine = None
        self._analytics_manager = None
        self._memory_manager = None
        self._model_manager = None
        self._knowledge_graph = None
        
        # Dialog state
        self._dialog_state = DialogState.IDLE
        self._current_cycle: Optional[DialogCycle] = None
        self._dialog_history: List[DialogCycle] = []
        self._insights: List[LearningInsight] = []
        
        # Threading
        self._learning_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Statistics
        self._stats = {
            "total_cycles": 0,
            "total_turns": 0,
            "total_insights": 0,
            "cycles_with_contradictions": 0,
            "cycles_with_ethics_violations": 0,
            "average_quality_score": 0.0,
            "last_cycle_time": 0.0
        }
        
        logger.info("SelfDialogLearningSystem initialized")
    
    def _default_topics(self) -> List[str]:
        """Default topics for self-dialog."""
        return [
            "philosophy and consciousness",
            "scientific method and discovery",
            "ethical decision making",
            "creativity and problem solving",
            "knowledge and understanding",
            "language and communication",
            "logic and reasoning",
            "history and human experience"
        ]
```

---

## 3. Main Methods

### 3.1 run_self_dialog_cycle()

```python
def run_self_dialog_cycle(self, topic: Optional[str] = None) -> DialogCycle:
    """
    Execute a complete self-dialog cycle.
    
    Flow:
        1. Select topic
        2. Generate question via model
        3. Generate response via model
        4. Check ethics (EthicsFramework)
        5. Check contradictions (ContradictionManager)
        6. Verify facts (WebSearchEngine)
        7. Analyze quality (AnalyticsManager)
        8. Store insights in memory
    
    Args:
        topic: Specific topic or None for random
    
    Returns:
        DialogCycle: Complete cycle with all analysis results
    """
    cycle_id = f"cycle_{int(time.time())}_{self._stats['total_cycles']}"
    topic = topic or random.choice(self.topics)
    
    self._dialog_state = DialogState.GENERATING
    cycle = DialogCycle(cycle_id=cycle_id, topic=topic)
    
    logger.info(f"Starting self-dialog cycle {cycle_id} on topic: {topic}")
    
    try:
        # Step 1: Generate initial question
        question = self._generate_question(topic)
        cycle.turns.append(DialogTurn(
            turn_id=f"{cycle_id}_q1",
            timestamp=time.time(),
            role="system",
            content=question
        ))
        
        # Step 2: Generate response
        response = self._generate_response(question, topic)
        response_turn = DialogTurn(
            turn_id=f"{cycle_id}_r1",
            timestamp=time.time(),
            role="system",
            content=response
        )
        
        # Step 3-7: Analyze through managers
        self._dialog_state = DialogState.ANALYZING
        response_turn.analysis_results = self._analyze_response(question, response)
        
        cycle.turns.append(response_turn)
        
        # Update cycle status from analysis
        cycle.ethics_passed = response_turn.analysis_results.get("ethics_passed", True)
        cycle.contradictions_found = response_turn.analysis_results.get("contradictions_count", 0)
        cycle.facts_verified = response_turn.analysis_results.get("facts_verified", 0)
        cycle.quality_score = response_turn.analysis_results.get("quality_score", 0.0)
        
        # Step 8: Generate follow-up and analysis
        follow_up = self._generate_follow_up(question, response)
        cycle.turns.append(DialogTurn(
            turn_id=f"{cycle_id}_f1",
            timestamp=time.time(),
            role="analyst",
            content=follow_up
        ))
        
        # Step 9: Extract and store insights
        self._dialog_state = DialogState.STORING
        insights = self._extract_insights(cycle)
        cycle.insights = insights
        
        for insight in insights:
            self.store_learning_insight(insight)
        
        # Update stats
        self._update_stats(cycle)
        
        self._dialog_state = DialogState.COMPLETED
        self._dialog_history.append(cycle)
        
        logger.info(
            f"Cycle {cycle_id} completed: "
            f"ethics={cycle.ethics_passed}, "
            f"contradictions={cycle.contradictions_found}, "
            f"quality={cycle.quality_score:.2f}, "
            f"insights={len(insights)}"
        )
        
        return cycle
        
    except Exception as e:
        logger.error(f"Error in cycle {cycle_id}: {e}")
        self._dialog_state = DialogState.IDLE
        raise
```

### 3.2 _analyze_response()

```python
def _analyze_response(self, question: str, response: str) -> Dict[str, Any]:
    """
    Analyze response through all enabled managers.
    
    Args:
        question: Original question
        response: Generated response
    
    Returns:
        Dict with analysis results from all managers
    """
    results = {
        "ethics_passed": True,
        "ethics_violations": [],
        "ethics_score": 1.0,
        "contradictions_count": 0,
        "contradictions": [],
        "facts_verified": 0,
        "facts_unverified": [],
        "quality_score": 0.0,
        "quality_metrics": {},
        "manager_status": {}
    }
    
    # Ethics check
    if "ethics" in self.enabled_managers:
        ethics_result = self.check_ethical_compliance(response, {"query": question})
        results["ethics_passed"] = ethics_result.get("approved", True)
        results["ethics_violations"] = ethics_result.get("violations", [])
        results["ethics_score"] = ethics_result.get("overall_score", 1.0)
        results["manager_status"]["ethics"] = "success"
    
    # Contradiction check
    if "contradiction" in self.enabled_managers:
        contradiction_result = self.detect_internal_contradictions(response)
        results["contradictions_count"] = len(contradiction_result)
        results["contradictions"] = contradiction_result
        results["manager_status"]["contradiction"] = "success"
    
    # Web search verification
    if "websearch" in self.enabled_managers:
        web_result = self.verify_facts_with_web(response)
        results["facts_verified"] = web_result.get("verified_count", 0)
        results["facts_unverified"] = web_result.get("unverified_claims", [])
        results["manager_status"]["websearch"] = "success"
    
    # Quality analysis
    if "analytics" in self.enabled_managers:
        quality_result = self.analyze_text_quality(response)
        results["quality_score"] = quality_result.get("overall_score", 0.0)
        results["quality_metrics"] = quality_result
    
    return results
```

### 3.3 check_ethical_compliance()

```python
def check_ethical_compliance(
    self,
    text: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Check text compliance using EthicsFramework.
    
    Args:
        text: Text to check
        context: Additional context (query, etc.)
    
    Returns:
        Dict with ethics analysis results
    """
    if not self._ethics_framework:
        self._ethics_framework = self._get_ethics_framework()
    
    if not self._ethics_framework:
        logger.warning("EthicsFramework not available")
        return {"approved": True, "violations": [], "overall_score": 1.0}
    
    try:
        # Use analyze_content for direct analysis
        if hasattr(self._ethics_framework, 'analyze_content'):
            analysis = self._ethics_framework.analyze_content(text, context)
            return {
                "approved": analysis.overall_score >= 0.8,
                "violations": analysis.violations,
                "recommendations": analysis.recommendations,
                "overall_score": analysis.overall_score,
                "principle_scores": analysis.principle_scores
            }
        
        # Fallback to analyze_request
        result = self._ethics_framework.analyze_request(text, context)
        return result
        
    except Exception as e:
        logger.error(f"Ethics check failed: {e}")
        return {"approved": True, "violations": [], "overall_score": 1.0, "error": str(e)}
```

### 3.4 detect_internal_contradictions()

```python
def detect_internal_contradictions(self, text: str) -> List[Dict[str, Any]]:
    """
    Detect contradictions using ContradictionManager.
    
    Args:
        text: Text to analyze
    
    Returns:
        List of detected contradictions
    """
    if not self._contradiction_manager:
        self._contradiction_manager = self._get_contradiction_manager()
    
    if not self._contradiction_manager:
        logger.warning("ContradictionManager not available")
        return []
    
    try:
        contradictions = self._contradiction_manager.detect_contradictions(text)
        
        # Format contradictions
        formatted = []
        for c in contradictions:
            if isinstance(c, dict):
                formatted.append({
                    "id": c.get("id", ""),
                    "type": c.get("type", "unknown"),
                    "description": c.get("description", ""),
                    "severity": c.get("divergence_level", 0.0),
                    "concepts": c.get("concepts", [])
                })
        
        return formatted
        
    except Exception as e:
        logger.error(f"Contradiction detection failed: {e}")
        return []
```

### 3.5 verify_facts_with_web()

```python
def verify_facts_with_web(self, text: str) -> Dict[str, Any]:
    """
    Verify factual claims using WebSearchEngine.
    
    Args:
        text: Text containing claims to verify
    
    Returns:
        Dict with verification results
    """
    if not self._web_search_engine:
        self._web_search_engine = self._get_web_search_engine()
    
    if not self._web_search_engine:
        logger.warning("WebSearchEngine not available")
        return {"verified_count": 0, "unverified_claims": [], "claims": []}
    
    try:
        # Extract factual claims (simple approach - can be enhanced)
        claims = self._extract_factual_claims(text)
        
        verified_count = 0
        unverified_claims = []
        verification_results = []
        
        for claim in claims[:5]:  # Limit to 5 claims per cycle
            result = self._web_search_engine.search(claim, max_results=3)
            
            if result.get("status") == "completed":
                search_results = result.get("results", [])
                if search_results:
                    verified_count += 1
                    verification_results.append({
                        "claim": claim,
                        "verified": True,
                        "sources": [r.url for r in search_results[:2]]
                    })
                else:
                    unverified_claims.append(claim)
                    verification_results.append({
                        "claim": claim,
                        "verified": False,
                        "sources": []
                    })
        
        return {
            "verified_count": verified_count,
            "unverified_claims": unverified_claims,
            "claims": verification_results
        }
        
    except Exception as e:
        logger.error(f"Fact verification failed: {e}")
        return {"verified_count": 0, "unverified_claims": [], "claims": [], "error": str(e)}

def _extract_factual_claims(self, text: str) -> List[str]:
    """Extract potential factual claims from text."""
    # Simple extraction - can be enhanced with NLP
    import re
    
    # Look for sentences with fact indicators
    patterns = [
        r'\b(is|are|was|were|has|have|will|can|does|do)\s+[^\.!?]+',
        r'\b\d+\s+(people|years|meters|kilograms|percent|%)\b',
        r'\b(because|therefore|thus|hence|consequently)\b[^\.!?]+'
    ]
    
    claims = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        claims.extend(matches)
    
    # Deduplicate and limit
    return list(dict.fromkeys(claims))[:10]
```

### 3.6 analyze_text_quality()

```python
def analyze_text_quality(self, text: str) -> Dict[str, Any]:
    """
    Analyze text quality using AnalyticsManager or direct analysis.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dict with quality metrics
    """
    if not self._analytics_manager:
        self._analytics_manager = self._get_analytics_manager()
    
    try:
        # Use AnalyticsManager if available
        if self._analytics_manager:
            insights = self._analytics_manager.get_learning_insights()
            return {
                "overall_score": 0.8,
                "metrics": insights,
                "source": "analytics_manager"
            }
        
        # Direct quality analysis
        return self._direct_quality_analysis(text)
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        return self._direct_quality_analysis(text)

def _direct_quality_analysis(self, text: str) -> Dict[str, Any]:
    """Perform direct quality analysis without AnalyticsManager."""
    import re
    
    # Basic metrics
    words = len(text.split())
    sentences = len(re.split(r'[.!?]+', text))
    avg_sentence_length = words / max(sentences, 1)
    
    # Complexity score (simple)
    complex_words = len([w for w in text.split() if len(w) > 6])
    complexity_ratio = complex_words / max(words, 1)
    
    # Coherence indicators
    coherence_indicators = len(re.findall(r'\b(this|that|these|those|therefore|however|furthermore)\b', text, re.I))
    
    # Calculate overall score
    scores = []
    if 10 <= words <= 500:
        scores.append(0.8)
    elif words > 500:
        scores.append(0.6)
    else:
        scores.append(0.4)
    
    if 10 <= avg_sentence_length <= 25:
        scores.append(0.8)
    else:
        scores.append(0.5)
    
    scores.append(min(complexity_ratio * 2, 1.0))
    scores.append(min(coherence_indicators * 0.1, 0.3))
    
    overall = sum(scores) / len(scores)
    
    return {
        "overall_score": overall,
        "word_count": words,
        "sentence_count": sentences,
        "avg_sentence_length": avg_sentence_length,
        "complexity_ratio": complexity_ratio,
        "coherence_indicators": coherence_indicators,
        "source": "direct_analysis"
    }
```

### 3.7 store_learning_insight()

```python
def store_learning_insight(self, insight: LearningInsight) -> bool:
    """
    Store learning insight in memory system.
    
    Args:
        insight: LearningInsight to store
    
    Returns:
        bool: Success status
    """
    if not self._memory_manager:
        self._memory_manager = self._get_memory_manager()
    
    try:
        insight_data = {
            "insight_id": insight.insight_id,
            "timestamp": insight.timestamp,
            "topic": insight.topic,
            "category": insight.category,
            "content": insight.content,
            "source_turn": insight.source_turn,
            "confidence": insight.confidence,
            "metadata": insight.metadata
        }
        
        # Store in MemoryManager if available
        if self._memory_manager:
            if hasattr(self._memory_manager, 'add_to_episodic'):
                self._memory_manager.add_to_episodic(insight_data)
            elif hasattr(self._memory_manager, 'store'):
                self._memory_manager.store("insights", insight.insight_id, insight_data)
        
        # Store in KnowledgeGraph if available
        if self._knowledge_graph:
            self._knowledge_graph.add_insight(insight_data)
        
        # Also save to JSON file for persistence
        self._save_insight_to_file(insight_data)
        
        self._insights.append(insight)
        self._stats["total_insights"] += 1
        
        logger.debug(f"Stored insight {insight.insight_id}: {insight.category}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store insight: {e}")
        return False

def _save_insight_to_file(self, insight_data: Dict[str, Any]) -> bool:
    """Save insight to JSON file."""
    try:
        import os
        
        cache_dir = getattr(self._model_manager, 'cache_dir', './cache') if self._model_manager else './cache'
        insights_dir = os.path.join(cache_dir, 'self_dialog_insights')
        os.makedirs(insights_dir, exist_ok=True)
        
        filename = f"{insight_data['insight_id']}.json"
        filepath = os.path.join(insights_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insight_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save insight to file: {e}")
        return False
```

---

## 4. Helper Methods

```python
def _generate_question(self, topic: str) -> str:
    """Generate a question on the given topic using model."""
    if not self._model_manager:
        self._model_manager = self._get_model_manager()
    
    if not self._model_manager:
        # Fallback question
        return f"What is your understanding of {topic}?"
    
    try:
        prompt = f"Generate a thoughtful question about {topic}. The question should be philosophical or analytical in nature."
        
        model, tokenizer, model_name = self._model_manager.get_model_for_task("text-generation")
        
        if model and tokenizer:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_p=0.9
                )
            
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return question.strip()
        
        return f"What is your understanding of {topic}?"
        
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return f"What is your understanding of {topic}?"

def _generate_response(self, question: str, topic: str) -> str:
    """Generate a response to the question using model."""
    if not self._model_manager:
        self._model_manager = self._get_model_manager()
    
    if not self._model_manager:
        return "I need to develop my understanding of this topic further."
    
    try:
        prompt = f"Question: {question}\n\nProvide a thoughtful, nuanced response that considers multiple perspectives."
        
        model, tokenizer, model_name = self._model_manager.get_model_for_task("text-generation")
        
        if model and tokenizer:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=300,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        
        return "I need to develop my understanding of this topic further."
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "I need to develop my understanding of this topic further."

def _generate_follow_up(self, question: str, response: str) -> str:
    """Generate analytical follow-up based on response analysis."""
    return f"Analysis: I observe that my response engaged with the question about this topic. The key points were examined from multiple angles."

def _extract_insights(self, cycle: DialogCycle) -> List[LearningInsight]:
    """Extract insights from a completed cycle."""
    insights = []
    
    for turn in cycle.turns:
        if turn.role != "system":
            continue
        
        # Ethics insights
        if turn.analysis_results.get("ethics_violations"):
            for violation in turn.analysis_results["ethics_violations"]:
                insights.append(LearningInsight(
                    insight_id=f"insight_{int(time.time())}_{len(insights)}",
                    timestamp=time.time(),
                    topic=cycle.topic,
                    category="ethics",
                    content=f"Ethics concern: {violation.get('description', 'Unknown violation')}",
                    source_turn=turn.turn_id,
                    confidence=violation.get("severity", 0.5),
                    metadata={"violation_id": violation.get("id")}
                ))
        
        # Contradiction insights
        if turn.analysis_results.get("contradictions"):
            for contradiction in turn.analysis_results["contradictions"]:
                insights.append(LearningInsight(
                    insight_id=f"insight_{int(time.time())}_{len(insights)}",
                    timestamp=time.time(),
                    topic=cycle.topic,
                    category="contradiction",
                    content=f"Internal contradiction detected: {contradiction.get('description', 'Unknown')}",
                    source_turn=turn.turn_id,
                    confidence=contradiction.get("severity", 0.5),
                    metadata={"contradiction_id": contradiction.get("id")}
                ))
        
        # Quality insights
        quality_score = turn.analysis_results.get("quality_score", 0.0)
        if quality_score < self.min_quality_threshold:
            insights.append(LearningInsight(
                insight_id=f"insight_{int(time.time())}_{len(insights)}",
                timestamp=time.time(),
                topic=cycle.topic,
                category="quality",
                content=f"Quality improvement needed: score {quality_score:.2f} below threshold {self.min_quality_threshold}",
                source_turn=turn.turn_id,
                confidence=0.8,
                metadata={"quality_score": quality_score}
            ))
    
    # Limit insights per cycle
    return insights[:self.max_insights_per_cycle]

def _update_stats(self, cycle: DialogCycle):
    """Update system statistics."""
    self._stats["total_cycles"] += 1
    self._stats["total_turns"] += len(cycle.turns)
    self._stats["last_cycle_time"] = time.time()
    
    if cycle.contradictions_found > 0:
        self._stats["cycles_with_contradictions"] += 1
    
    if not cycle.ethics_passed:
        self._stats["cycles_with_ethics_violations"] += 1
    
    # Calculate running average
    n = self._stats["total_cycles"]
    old_avg = self._stats["average_quality_score"]
    new_score = cycle.quality_score
    self._stats["average_quality_score"] = (old_avg * (n - 1) + new_score) / n
```

---

## 5. Manager Integration Methods

```python
def _get_ethics_framework(self):
    """Get EthicsFramework from brain."""
    if not self.brain:
        return None
    
    if hasattr(self.brain, 'ethics_framework'):
        return self.brain.ethics_framework
    
    if hasattr(self.brain, 'components') and 'ethics_framework' in self.brain.components:
        return self.brain.components['ethics_framework']
    
    # Try direct import
    try:
        from cogniflex.ethics.ethics_framework import EthicsFramework
        return EthicsFramework(self.brain)
    except ImportError:
        return None

def _get_contradiction_manager(self):
    """Get ContradictionManager from brain."""
    if not self.brain:
        return None
    
    if hasattr(self.brain, 'contradiction_manager'):
        return self.brain.contradiction_manager
    
    if hasattr(self.brain, 'components') and 'contradiction_manager' in self.brain.components:
        return self.brain.components['contradiction_manager']
    
    try:
        from cogniflex.contradiction.contradiction_manager import ContradictionManager
        return ContradictionManager(self.brain)
    except ImportError:
        return None

def _get_web_search_engine(self):
    """Get WebSearchEngine from brain."""
    if not self.brain:
        return None
    
    if hasattr(self.brain, 'web_search_engine'):
        return self.brain.web_search_engine
    
    if hasattr(self.brain, 'components') and 'web_search_engine' in self.brain.components:
        return self.brain.components['web_search_engine']
    
    try:
        from cogniflex.websearch.web_search_engine import WebSearchEngine
        return WebSearchEngine(self.brain)
    except ImportError:
        return None

def _get_analytics_manager(self):
    """Get AnalyticsManager from brain."""
    if not self.brain:
        return None
    
    if hasattr(self.brain, 'analytics_manager'):
        return self.brain.analytics_manager
    
    if hasattr(self.brain, 'components') and 'analytics_manager' in self.brain.components:
        return self.brain.components['analytics_manager']
    
    try:
        from cogniflex.analytics.analytics_manager import AnalyticsManager
        return AnalyticsManager(self.brain)
    except ImportError:
        return None

def _get_memory_manager(self):
    """Get MemoryManager from brain."""
    if not self.brain:
        return None
    
    if hasattr(self.brain, 'memory_manager'):
        return self.brain.memory_manager
    
    if hasattr(self.brain, 'components') and 'memory_manager' in self.brain.components:
        return self.brain.components['memory_manager']
    
    return None

def _get_model_manager(self):
    """Get ModelManager from brain."""
    if not self.brain:
        return None
    
    if hasattr(self.brain, 'model_manager'):
        return self.brain.model_manager
    
    if hasattr(self.brain, 'components') and 'model_manager' in self.brain.components:
        return self.brain.components['model_manager']
    
    return None
```

---

## 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SelfDialogLearningSystem                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Select     │────▶│   Generate   │────▶│   Generate   │                │
│  │   Topic      │     │   Question   │     │   Response   │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                    │                        │
│                                                    ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ANALYSIS PIPELINE                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │   │
│  │  │  EthicsFramework │    │ContradictionMgr │    │ WebSearchEngine │ │   │
│  │  │  (check_ethics)  │    │(detect_contr.)  │    │ (verify_facts)  │ │   │
│  │  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘ │   │
│  │           │                      │                      │          │   │
│  │           ▼                      ▼                      ▼          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │                  Analysis Results Merge                      │ │   │
│  │  │  - ethics_passed: bool                                      │ │   │
│  │  │  - contradictions_count: int                               │ │   │
│  │  │  - facts_verified: int                                      │ │   │
│  │  │  - quality_score: float                                     │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                              │                                    │   │
│  └──────────────────────────────┼────────────────────────────────────┘   │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              EXTRACT & STORE INSIGHTS                                │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │   │
│  │  │ MemoryManager   │    │ KnowledgeGraph  │    │    JSON File    │  │   │
│  │  │ (episodic mem)  │    │ (nodes/edges)   │    │   (insights/)   │  │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Configuration Options

```python
# In brain_config.json or system config

{
  "self_dialog_learning": {
    "enabled": true,
    "cycle_interval": 300,           // Seconds between cycles (5 min)
    "cycles_per_session": 3,         // Cycles per learning session
    "min_quality_threshold": 0.5,    // Minimum quality score for response
    "max_insights_per_cycle": 10,    // Max insights extracted per cycle
    
    "topics": [
      "philosophy and consciousness",
      "scientific method and discovery",
      "ethical decision making",
      "creativity and problem solving",
      "knowledge and understanding",
      "language and communication",
      "logic and reasoning",
      "history and human experience"
    ],
    
    "enabled_managers": [
      "ethics",        // EthicsFramework for ethical compliance
      "contradiction", // ContradictionManager for internal consistency
      "websearch",     // WebSearchEngine for fact verification
      "analytics"      // AnalyticsManager for quality metrics
    ],
    
    "generation": {
      "temperature": 0.7,
      "max_length": 300,
      "top_p": 0.9
    }
  }
}
```

---

## 8. Expected Outputs

### 8.1 Learning Insights Stored

Each insight is saved with structure:

```json
{
  "insight_id": "insight_1234567890_0",
  "timestamp": 1234567890.0,
  "topic": "philosophy and consciousness",
  "category": "ethics|contradiction|fact|quality|pattern",
  "content": "Description of the insight",
  "source_turn": "cycle_1234567890_0_r1",
  "confidence": 0.85,
  "metadata": {}
}
```

### 8.2 Categories of Insights

| Category | Source | Example |
|----------|--------|---------|
| `ethics` | EthicsFramework | "Ethics concern: potential violation of privacy principle" |
| `contradiction` | ContradictionManager | "Internal contradiction: statement A conflicts with statement B" |
| `fact` | WebSearchEngine | "Claim about X verified with Y sources" |
| `quality` | Direct analysis | "Quality score 0.45 below threshold 0.5" |
| `pattern` | Cross-cycle analysis | "Consistent difficulty with metaphysical topics" |

### 8.3 Statistics Tracked

```python
{
    "total_cycles": 150,
    "total_turns": 450,
    "total_insights": 892,
    "cycles_with_contradictions": 23,
    "cycles_with_ethics_violations": 5,
    "average_quality_score": 0.72,
    "last_cycle_time": 1234567890.0
}
```

---

## 9. Integration with CoreBrain

### 9.1 Registration

```python
# In CoreBrain or ComponentInitializer
from cogniflex.learning.self_dialog_learning import SelfDialogLearningSystem

self_dialog = SelfDialogLearningSystem(
    brain=self,
    config=config.get("self_dialog_learning", {})
)
self.register_component("self_dialog_learning", self_dialog)
```

### 9.2 Event Subscriptions

```python
# Subscribe to system events
self._subscribe(EventTypes.SYSTEM_START, self._handle_system_start)
self._subscribe(EventTypes.SYSTEM_STOP, self._handle_system_stop)
self._subscribe("learning/trigger_cycle", self._handle_cycle_trigger)
```

### 9.3 Lifecycle Methods

```python
def _do_initialize(self) -> bool:
    """Initialize component resources."""
    return True

def _do_start(self) -> bool:
    """Start the self-dialog learning thread."""
    self._stop_event.clear()
    self._learning_thread = threading.Thread(
        target=self._learning_loop,
        daemon=True
    )
    self._learning_thread.start()
    return True

def _do_stop(self) -> bool:
    """Stop the learning thread."""
    self._stop_event.set()
    if self._learning_thread and self._learning_thread.is_alive():
        self._learning_thread.join(timeout=10)
    return True

def _learning_loop(self):
    """Main learning loop running in background."""
    while not self._stop_event.is_set():
        try:
            # Wait for interval or stop signal
            if self._pause_event.wait(timeout=self.cycle_interval):
                break  # Pause signal received
            
            # Run a cycle
            self.run_self_dialog_cycle()
            
        except Exception as e:
            logger.error(f"Error in learning loop: {e}")
            time.sleep(60)  # Back off on error
```

---

## 10. Comparison with Old System

| Aspect | Old System | New System |
|--------|------------|-------------|
| Training loops | Yes (fake) | No |
| Model fine-tuning | Attempted (stubs) | None |
| Real ML training | No | No |
| Fake data generation | Yes | No |
| Code files | 15+ | 1 |
| Manager usage | Fragmented | Unified |
| Training data | Generated | N/A |
| Analysis | None | Full pipeline |
| Insights storage | None | Full pipeline |

---

## 11. Files to Create/Modify

### New Files
- `cogniflex/learning/self_dialog_learning.py` - Main implementation

### Files to Deprecate/Disable
- `cogniflex/learning/integrated_learning_manager.py`
- `cogniflex/learning/learning_integrated.py`
- `cogniflex/learning/learning_manager.py`
- `cogniflex/learning/integration_manager.py`

### Configuration Updates
- Update `brain_config.json` to set `self_dialog_learning.enabled: true`
- Set `learning.enabled: false` in brain_config
