"""
Online Knowledge Access for CogniFlex
Provides access to Wikipedia and other online knowledge sources.
"""
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger("cogniflex.online_knowledge")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

class FactVerification:
    """Result of fact verification."""
    def __init__(self, fact: str, verified: bool, source: str, 
                 summary: str = "", url: str = "", confidence: float = 0.0):
        self.fact = fact
        self.verified = verified
        self.source = source
        self.summary = summary
        self.url = url
        self.confidence = confidence
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        return {
            'fact': self.fact,
            'verified': self.verified,
            'source': self.source,
            'summary': self.summary,
            'url': self.url,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }

class OnlineKnowledgeAccess:
    """
    Access online knowledge sources for verification and learning.
    
    Sources:
    - Wikipedia API
    - Other public APIs
    - Web search fallback
    """
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        
        # Cache
        self.cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # Wikipedia API
        self.wikipedia_api = "https://en.wikipedia.org/api/rest_v1/page/summary"
        self.wikipedia_ru_api = "https://ru.wikipedia.org/api/rest_v1/page/summary"
        
        # Rate limiting
        self.last_request = 0
        self.min_request_interval = 1.0  # seconds
    
    def search_wikipedia(self, query: str, lang: str = 'ru') -> Optional[Dict]:
        """
        Search Wikipedia for a query.
        
        Args:
            query: Search query
            lang: Language ('en' or 'ru')
            
        Returns:
            Dict with article summary or None
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available")
            return None
        
        # Check cache
        cache_key = f"wiki_{lang}_{query}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['data']
        
        # Rate limiting
        if time.time() - self.last_request < self.min_request_interval:
            time.sleep(self.min_request_interval)
        
        try:
            api_url = self.wikipedia_ru_api if lang == 'ru' else self.wikipedia_api
            url = f"{api_url}/{query.replace(' ', '_')}"
            
            headers = {
                'User-Agent': 'CogniFlex/1.0 (Cognitive AI System)'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    'title': data.get('title', ''),
                    'summary': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'source': f'Wikipedia ({lang})',
                    'timestamp': time.time()
                }
                
                # Cache result
                self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
                self.last_request = time.time()
                
                return result
            else:
                logger.debug(f"Wikipedia returned {response.status_code} for {query}")
                return None
                
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
            return None
    
    def search_wikipedia_multi(self, queries: List[str], lang: str = 'ru') -> List[Dict]:
        """
        Search Wikipedia for multiple queries.
        
        Args:
            queries: List of search queries
            lang: Language
            
        Returns:
            List of results (None for failed searches)
        """
        results = []
        for query in queries[:5]:  # Limit to 5
            result = self.search_wikipedia(query, lang)
            results.append(result)
            time.sleep(0.5)  # Be nice to Wikipedia API
        
        return results
    
    def verify_fact(self, fact: str) -> FactVerification:
        """
        Verify a fact against online sources.
        
        Args:
            fact: Fact statement to verify
            
        Returns:
            FactVerification object
        """
        # Extract potential entity from fact
        entity = self._extract_entity_from_fact(fact)
        
        if entity:
            wiki_result = self.search_wikipedia(entity)
            
            if wiki_result and wiki_result.get('summary'):
                # Check if fact relates to summary
                relevance = self._calculate_relevance(fact, wiki_result['summary'])
                
                return FactVerification(
                    fact=fact,
                    verified=relevance > 0.3,
                    source=wiki_result['source'],
                    summary=wiki_result['summary'][:200],
                    url=wiki_result.get('url', ''),
                    confidence=relevance
                )
        
        return FactVerification(
            fact=fact,
            verified=False,
            source='unknown',
            confidence=0.0
        )
    
    def _extract_entity_from_fact(self, fact: str) -> Optional[str]:
        """Extract main entity from fact statement."""
        # Remove common patterns
        patterns_to_remove = [
            r'^(?:Это|This is|This|That is|Это|Это\s+)',
            r'\b(?:является|is|represents|means)\b.*$',
            r'\b(?:состоит|consists|includes)\b.*$',
        ]
        
        entity = fact
        for pattern in patterns_to_remove:
            entity = re.sub(pattern, '', entity, flags=re.IGNORECASE).strip()
        
        # Get first significant noun phrase
        words = entity.split()
        if len(words) > 5:
            entity = ' '.join(words[:5])
        
        return entity if entity else None
    
    def _calculate_relevance(self, fact: str, summary: str) -> float:
        """Calculate relevance between fact and summary."""
        fact_words = set(fact.lower().split())
        summary_words = set(summary.lower().split())
        
        # Simple word overlap
        overlap = len(fact_words & summary_words)
        total = len(fact_words)
        
        if total == 0:
            return 0.0
        
        return min(1.0, overlap / total)
    
    def learn_about_entity(self, entity: str) -> Dict[str, Any]:
        """
        Learn everything about an entity from online sources.
        
        Args:
            entity: Entity name
            
        Returns:
            Dict with learning results
        """
        result = {
            'entity': entity,
            'wikipedia': None,
            'verified': False,
            'knowledge': '',
            'sources': []
        }
        
        # Try Russian Wikipedia first
        wiki = self.search_wikipedia(entity, lang='ru')
        if wiki:
            result['wikipedia'] = wiki
            result['knowledge'] = wiki.get('summary', '')
            result['sources'].append(wiki.get('source', 'Wikipedia'))
            result['verified'] = True
        
        # Fallback to English
        if not wiki:
            wiki = self.search_wikipedia(entity, lang='en')
            if wiki:
                result['wikipedia'] = wiki
                result['knowledge'] = wiki.get('summary', '')
                result['sources'].append(wiki.get('source', 'Wikipedia'))
                result['verified'] = True
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_ttl': self.cache_ttl,
            'last_request': self.last_request
        }
