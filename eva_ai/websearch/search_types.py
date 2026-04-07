"""
Типы для Web Search
Часть модулей web_search_engine.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class SearchEngine(Enum):
    """Поисковые системы."""
    GOOGLE = "google"
    BING = "bing"
    YANDEX = "yandex"
    WIKIPEDIA = "wikipedia"
    CUSTOM = "custom"


class ContentType(Enum):
    """Типы контента."""
    WEB_PAGE = "web_page"
    PDF = "pdf"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"


@dataclass
class SearchResult:
    """Результат поиска."""
    url: str
    title: str
    snippet: str
    source: SearchEngine
    content_type: ContentType
    relevance_score: float = 0.0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source.value if isinstance(self.source, SearchEngine) else self.source,
            "content_type": self.content_type.value if isinstance(self.content_type, ContentType) else self.content_type,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp
        }


@dataclass
class SearchQuery:
    """Поисковый запрос."""
    query: str
    engines: List[SearchEngine] = field(default_factory=list)
    max_results: int = 10
    language: Optional[str] = None
    region: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "engines": [e.value if isinstance(e, SearchEngine) else e for e in self.engines],
            "max_results": self.max_results,
            "language": self.language,
            "region": self.region
        }
