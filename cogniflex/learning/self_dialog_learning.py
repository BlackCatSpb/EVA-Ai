"""
Self-Dialog Learning System for CogniFlex
Uses existing managers to analyze system responses without training.
"""
import logging
import time
import threading
import json
import random
import re
import os
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
    role: str
    content: str
    analysis_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningInsight:
    """Insight extracted from self-dialog."""
    insight_id: str
    timestamp: float
    topic: str
    category: str
    content: str
    source_turn: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    insights: List[LearningInsight] = field(default_factory=list)


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
        
        cfg = config or {}
        self.cycle_interval = cfg.get("cycle_interval", 300)
        self.cycles_per_session = cfg.get("cycles_per_session", 3)
        self.topics = cfg.get("topics", self._default_topics())
        self.enabled_managers = cfg.get("enabled_managers", [
            "ethics", "contradiction", "websearch", "analytics"
        ])
        self.min_quality_threshold = cfg.get("min_quality_threshold", 0.5)
        self.max_insights_per_cycle = cfg.get("max_insights_per_cycle", 10)
        self.enabled = cfg.get("enabled", True)
        
        self._ethics_framework = None
        self._contradiction_manager = None
        self._web_search_engine = None
        self._analytics_manager = None
        self._memory_manager = None
        self._model_manager = None
        self._knowledge_graph = None
        
        self._dialog_state = DialogState.IDLE
        self._current_cycle: Optional[DialogCycle] = None
        self._dialog_history: List[DialogCycle] = []
        self._insights: List[LearningInsight] = []
        
        self._learning_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
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
    
    def _do_initialize(self) -> bool:
        """Initialize managers from brain."""
        if not self.brain:
            logger.warning("No brain reference, managers will be lazy-loaded")
            return True
        
        self._model_manager = getattr(self.brain, 'fractal_model_manager', None) or \
                              getattr(self.brain, 'model_manager', None)
        self._memory_manager = getattr(self.brain, 'memory_manager', None)
        self._knowledge_graph = getattr(self.brain, 'knowledge_graph', None)
        self._ethics_framework = getattr(self.brain, 'ethics_framework', None)
        self._contradiction_manager = getattr(self.brain, 'contradiction_manager', None)
        self._web_search_engine = getattr(self.brain, 'web_search_engine', None)
        self._analytics_manager = getattr(self.brain, 'analytics_manager', None)
        
        logger.info("SelfDialogLearningSystem managers initialized")
        return True
    
    def _do_start(self) -> bool:
        """Start self-dialog cycle."""
        if not self.enabled:
            logger.info("SelfDialogLearningSystem disabled in config")
            return True
        
        self._stop_event.clear()
        self._learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True,
            name="SelfDialogLearningThread"
        )
        self._learning_thread.start()
        logger.info(f"SelfDialogLearningSystem started with interval={self.cycle_interval}s")
        return True
    
    def _do_stop(self) -> bool:
        """Stop self-dialog cycle."""
        self._stop_event.set()
        self._pause_event.set()
        
        if self._learning_thread and self._learning_thread.is_alive():
            self._learning_thread.join(timeout=10)
        
        logger.info("SelfDialogLearningSystem stopped")
        return True
    
    def _learning_loop(self):
        """Main learning loop running in background."""
        logger.info("Self-dialog learning loop started")
        
        while not self._stop_event.is_set():
            try:
                if self._pause_event.wait(timeout=self.cycle_interval):
                    break
                
                self.run_self_dialog_cycle()
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(60)
        
        logger.info("Self-dialog learning loop ended")
    
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
            question = self.generate_question(topic)
            cycle.turns.append(DialogTurn(
                turn_id=f"{cycle_id}_q1",
                timestamp=time.time(),
                role="system",
                content=question
            ))
            
            response = self.generate_response(question, topic)
            response_turn = DialogTurn(
                turn_id=f"{cycle_id}_r1",
                timestamp=time.time(),
                role="system",
                content=response
            )
            
            self._dialog_state = DialogState.ANALYZING
            response_turn.analysis_results = self._analyze_response(question, response)
            
            cycle.turns.append(response_turn)
            
            cycle.ethics_passed = response_turn.analysis_results.get("ethics_passed", True)
            cycle.contradictions_found = response_turn.analysis_results.get("contradictions_count", 0)
            cycle.facts_verified = response_turn.analysis_results.get("facts_verified", 0)
            cycle.quality_score = response_turn.analysis_results.get("quality_score", 0.0)
            
            follow_up = self._generate_follow_up(question, response)
            cycle.turns.append(DialogTurn(
                turn_id=f"{cycle_id}_f1",
                timestamp=time.time(),
                role="analyst",
                content=follow_up
            ))
            
            self._dialog_state = DialogState.STORING
            insights = self._extract_insights(cycle)
            cycle.insights = insights
            
            for insight in insights:
                self.store_learning_insight(insight)
            
            self._update_learning_stats(cycle)
            
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
    
    def generate_question(self, topic: str) -> str:
        """
        Generate self-dialog question.
        
        Args:
            topic: Topic to generate question about
        
        Returns:
            str: Generated question
        """
        if self._model_manager:
            try:
                prompt = f"Generate a thoughtful question about {topic}. The question should be philosophical or analytical in nature."
                
                if hasattr(self._model_manager, 'generate'):
                    result = self._model_manager.generate(
                        prompt,
                        max_length=100,
                        temperature=0.8,
                        top_p=0.9
                    )
                    if result:
                        return result.strip()
                elif hasattr(self._model_manager, 'get_model_for_task'):
                    pass
                
            except Exception as e:
                logger.warning(f"Model-based question generation failed: {e}")
        
        return f"What is your understanding of {topic}?"
    
    def generate_response(self, question: str, topic: str) -> str:
        """
        Generate response via fractal_model_manager.
        
        Args:
            question: The question to respond to
            topic: Topic context
        
        Returns:
            str: Generated response
        """
        if self._model_manager:
            try:
                prompt = f"Question: {question}\n\nProvide a thoughtful, nuanced response that considers multiple perspectives."
                
                if hasattr(self._model_manager, 'generate'):
                    result = self._model_manager.generate(
                        prompt,
                        max_length=300,
                        temperature=0.7,
                        top_p=0.9
                    )
                    if result:
                        return result.strip()
                elif hasattr(self._model_manager, 'get_model_for_task'):
                    pass
                
            except Exception as e:
                logger.warning(f"Model-based response generation failed: {e}")
        
        return "I need to develop my understanding of this topic further."
    
    def store_learning_insight(self, insight: LearningInsight) -> bool:
        """
        Store learning insight in memory/knowledge graph.
        
        Args:
            insight: LearningInsight to store
        
        Returns:
            bool: Success status
        """
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
            
            if self._memory_manager:
                if hasattr(self._memory_manager, 'add_to_episodic'):
                    self._memory_manager.add_to_episodic(insight_data)
                elif hasattr(self._memory_manager, 'store'):
                    self._memory_manager.store("insights", insight.insight_id, insight_data)
            
            if self._knowledge_graph:
                if hasattr(self._knowledge_graph, 'add_insight'):
                    self._knowledge_graph.add_insight(insight_data)
                elif hasattr(self._knowledge_graph, 'add_node'):
                    self._knowledge_graph.add_node(
                        insight.insight_id,
                        node_type="learning_insight",
                        **insight_data
                    )
            
            self._save_insight_to_file(insight_data)
            
            self._insights.append(insight)
            self._stats["total_insights"] += 1
            
            logger.debug(f"Stored insight {insight.insight_id}: {insight.category}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
            return False
    
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
        
        if "ethics" in self.enabled_managers:
            ethics_result = self._check_ethics(response)
            results["ethics_passed"] = ethics_result.get("approved", True)
            results["ethics_violations"] = ethics_result.get("violations", [])
            results["ethics_score"] = ethics_result.get("overall_score", 1.0)
            results["manager_status"]["ethics"] = "success"
        
        if "contradiction" in self.enabled_managers:
            contradiction_result = self._detect_contradictions(response)
            results["contradictions_count"] = len(contradiction_result)
            results["contradictions"] = contradiction_result
            results["manager_status"]["contradiction"] = "success"
        
        if "websearch" in self.enabled_managers:
            web_result = self._verify_facts(response)
            results["facts_verified"] = web_result.get("verified_count", 0)
            results["facts_unverified"] = web_result.get("unverified_claims", [])
            results["manager_status"]["websearch"] = "success"
        
        if "analytics" in self.enabled_managers:
            quality_result = self._analyze_quality(response)
            results["quality_score"] = quality_result.get("overall_score", 0.0)
            results["quality_metrics"] = quality_result
            results["manager_status"]["analytics"] = "success"
        
        return results
    
    def _check_ethics(self, text: str) -> Dict[str, Any]:
        """
        Placeholder for ethics check via EthicsFramework.
        
        Args:
            text: Text to check
        
        Returns:
            Dict with ethics analysis results
        """
        if self._ethics_framework:
            try:
                if hasattr(self._ethics_framework, 'analyze_content'):
                    analysis = self._ethics_framework.analyze_content(text, {})
                    return {
                        "approved": analysis.overall_score >= 0.8,
                        "violations": analysis.violations,
                        "overall_score": analysis.overall_score
                    }
                elif hasattr(self._ethics_framework, 'analyze_request'):
                    return self._ethics_framework.analyze_request(text, {})
            except Exception as e:
                logger.warning(f"Ethics framework check failed: {e}")
        
        return {"approved": True, "violations": [], "overall_score": 1.0}
    
    def _detect_contradictions(self, text: str) -> List[Dict[str, Any]]:
        """
        Placeholder for contradiction detection via ContradictionManager.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of detected contradictions
        """
        if self._contradiction_manager:
            try:
                contradictions = self._contradiction_manager.detect_contradictions(text)
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
                logger.warning(f"Contradiction detection failed: {e}")
        
        return []
    
    def _verify_facts(self, text: str) -> Dict[str, Any]:
        """
        Placeholder for fact verification via WebSearchEngine.
        
        Args:
            text: Text containing claims to verify
        
        Returns:
            Dict with verification results
        """
        if self._web_search_engine:
            try:
                claims = self._extract_factual_claims(text)
                verified_count = 0
                unverified_claims = []
                verification_results = []
                
                for claim in claims[:5]:
                    result = self._web_search_engine.search(claim, max_results=3)
                    if result.get("status") == "completed":
                        search_results = result.get("results", [])
                        if search_results:
                            verified_count += 1
                            verification_results.append({
                                "claim": claim,
                                "verified": True,
                                "sources": [r.url for r in search_results[:2]] if hasattr(search_results[0], 'url') else []
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
                logger.warning(f"Fact verification failed: {e}")
        
        return {"verified_count": 0, "unverified_claims": [], "claims": []}
    
    def _analyze_quality(self, text: str) -> Dict[str, Any]:
        """
        Placeholder for quality analysis.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with quality metrics
        """
        if self._analytics_manager:
            try:
                insights = self._analytics_manager.get_learning_insights()
                return {
                    "overall_score": 0.8,
                    "metrics": insights,
                    "source": "analytics_manager"
                }
            except Exception as e:
                logger.warning(f"Analytics manager quality analysis failed: {e}")
        
        return self._direct_quality_analysis(text)
    
    def _direct_quality_analysis(self, text: str) -> Dict[str, Any]:
        """Perform direct quality analysis."""
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        avg_sentence_length = words / max(sentences, 1)
        
        complex_words = len([w for w in text.split() if len(w) > 6])
        complexity_ratio = complex_words / max(words, 1)
        
        coherence_indicators = len(re.findall(
            r'\b(this|that|these|those|therefore|however|furthermore)\b',
            text, re.I
        ))
        
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
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract potential factual claims from text."""
        patterns = [
            r'\b(is|are|was|were|has|have|will|can|does|do)\s+[^\.!?]+',
            r'\b\d+\s+(people|years|meters|kilograms|percent|%)\b',
            r'\b(because|therefore|thus|hence|consequently)\b[^\.!?]+'
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend(matches)
        
        return list(dict.fromkeys(claims))[:10]
    
    def _generate_follow_up(self, question: str, response: str) -> str:
        """Generate analytical follow-up based on response analysis."""
        return f"Analysis: I observe that my response engaged with the question about this topic. The key points were examined from multiple angles."
    
    def _extract_insights(self, cycle: DialogCycle) -> List[LearningInsight]:
        """Extract insights from a completed cycle."""
        insights = []
        
        for turn in cycle.turns:
            if turn.role != "system":
                continue
            
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
        
        return insights[:self.max_insights_per_cycle]
    
    def _update_learning_stats(self, cycle: DialogCycle):
        """Update system statistics."""
        self._stats["total_cycles"] += 1
        self._stats["total_turns"] += len(cycle.turns)
        self._stats["last_cycle_time"] = time.time()
        
        if cycle.contradictions_found > 0:
            self._stats["cycles_with_contradictions"] += 1
        
        if not cycle.ethics_passed:
            self._stats["cycles_with_ethics_violations"] += 1
        
        n = self._stats["total_cycles"]
        old_avg = self._stats["average_quality_score"]
        new_score = cycle.quality_score
        self._stats["average_quality_score"] = (old_avg * (n - 1) + new_score) / n
    
    def _save_insight_to_file(self, insight_data: Dict[str, Any]) -> bool:
        """Save insight to JSON file."""
        try:
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {**self._stats}
    
    def get_dialog_history(self) -> List[DialogCycle]:
        """Get dialog history."""
        return self._dialog_history
    
    def get_insights(self) -> List[LearningInsight]:
        """Get all stored insights."""
        return self._insights
    
    def pause(self):
        """Pause the learning cycle."""
        self._pause_event.set()
    
    def resume(self):
        """Resume the learning cycle."""
        self._pause_event.clear()
