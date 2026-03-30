"""
KnowledgeAwareness Module for ЕВА
Tracks whether knowledge is VERIFIED (from web/memory) or GENERATED (model inference).
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class KnowledgeSource(Enum):
    VERIFIED = "verified"
    GENERATED = "generated"
    UNKNOWN = "unknown"


@dataclass
class KnowledgeEntry:
    text: str
    source_type: KnowledgeSource
    confidence: float
    external_source: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeAwareness:
    """
    Tracks whether knowledge is VERIFIED (from web/memory) 
    or GENERATED (model inference).
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        self.verified_knowledge: Dict[str, Dict[str, Any]] = {}
        self.generated_knowledge: Dict[str, Dict[str, Any]] = {}
    
    def mark_verified(self, text: str, source: str, metadata: Optional[Dict] = None):
        """Mark text as verified from external source."""
        self.verified_knowledge[text] = {
            'source': source,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
    
    def mark_generated(self, text: str, confidence: float, metadata: Optional[Dict] = None):
        """Mark text as model-generated with confidence."""
        self.generated_knowledge[text] = {
            'confidence': confidence,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
    
    def get_status(self, text: str) -> str:
        """Return 'verified', 'generated', or 'unknown'."""
        if text in self.verified_knowledge:
            return 'verified'
        if text in self.generated_knowledge:
            return 'generated'
        return 'unknown'
    
    def get_source_type(self, text: str) -> KnowledgeSource:
        """Return KnowledgeSource enum value."""
        status = self.get_status(text)
        if status == 'verified':
            return KnowledgeSource.VERIFIED
        elif status == 'generated':
            return KnowledgeSource.GENERATED
        return KnowledgeSource.UNKNOWN
    
    def get_verified_sources(self, text: str) -> Optional[str]:
        """Get the external source for verified text."""
        if text in self.verified_knowledge:
            return self.verified_knowledge[text].get('source')
        return None
    
    def get_generated_confidence(self, text: str) -> Optional[float]:
        """Get the confidence score for generated text."""
        if text in self.generated_knowledge:
            return self.generated_knowledge[text].get('confidence')
        return None
    
    def get_knowledge_report(self) -> Dict[str, Any]:
        """Return summary of known vs generated knowledge."""
        verified_sources = set()
        for v in self.verified_knowledge.values():
            src = v.get('source', '')
            if src:
                verified_sources.add(src)
        
        return {
            'verified_count': len(self.verified_knowledge),
            'generated_count': len(self.generated_knowledge),
            'total_tracked': len(self.verified_knowledge) + len(self.generated_knowledge),
            'verified_sources': list(verified_sources),
            'verified_ratio': len(self.verified_knowledge) / max(
                len(self.verified_knowledge) + len(self.generated_knowledge), 1
            )
        }
    
    def get_recent_knowledge(self, limit: int = 10, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent knowledge entries sorted by timestamp."""
        entries = []
        
        if source_type is None or source_type == 'verified':
            for text, data in self.verified_knowledge.items():
                entries.append({
                    'text': text,
                    'source': 'verified',
                    'external_source': data.get('source'),
                    'timestamp': data.get('timestamp', 0),
                    'metadata': data.get('metadata', {})
                })
        
        if source_type is None or source_type == 'generated':
            for text, data in self.generated_knowledge.items():
                entries.append({
                    'text': text,
                    'source': 'generated',
                    'confidence': data.get('confidence'),
                    'timestamp': data.get('timestamp', 0),
                    'metadata': data.get('metadata', {})
                })
        
        entries.sort(key=lambda x: x['timestamp'], reverse=True)
        return entries[:limit]
    
    def merge_knowledge(self, other: 'KnowledgeAwareness'):
        """Merge knowledge from another KnowledgeAwareness instance."""
        for text, data in other.verified_knowledge.items():
            if text not in self.verified_knowledge:
                self.verified_knowledge[text] = data.copy()
        
        for text, data in other.generated_knowledge.items():
            if text not in self.generated_knowledge:
                self.generated_knowledge[text] = data.copy()
    
    def clear_generated(self):
        """Clear all generated knowledge (e.g., after verification)."""
        self.generated_knowledge.clear()
    
    def clear_verified(self):
        """Clear all verified knowledge."""
        self.verified_knowledge.clear()
    
    def clear_all(self):
        """Clear all tracked knowledge."""
        self.verified_knowledge.clear()
        self.generated_knowledge.clear()
