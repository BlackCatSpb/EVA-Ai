"""Модели данных для веб-поиска CogniFlex"""
import time
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class SearchResult:
    """Результат поисковой выдачи."""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 1.0
    content: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    query: str = ""

@dataclass
class SearchQuery:
    """Поисковый запрос и его результаты."""
    query: str
    timestamp: float = field(default_factory=time.time)
    results: List[SearchResult] = field(default_factory=list)
    status: str = "pending"  # pending, completed, failed
    processing_time: float = 0.0
    cache_hit: bool = False