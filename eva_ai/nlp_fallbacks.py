"""Reusable NLP fallbacks and guards for optional dependencies.

This module provides safe fallback implementations for common NLP operations
when optional libraries are not available. It includes:

- Text similarity calculations using multiple algorithms
- Sentiment analysis with VADER fallback
- Text preprocessing utilities
- Batch processing for large datasets
- Safe import guards for sklearn, nltk, spacy

All functions are designed to gracefully handle missing dependencies and
provide reasonable fallback behavior.
"""

from __future__ import annotations
import re
import logging
import concurrent.futures as cf
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
import hashlib
import time

logger = logging.getLogger(__name__)


def _safe_import_sklearn():
    """Safely import sklearn components."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        return TfidfVectorizer, cosine_similarity
    except ImportError:
        logger.debug("sklearn not available for text similarity")
        return None, None


def _safe_import_nltk_vader():
    """Safely import NLTK VADER sentiment analyzer."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        return SentimentIntensityAnalyzer
    except ImportError:
        logger.debug("NLTK VADER not available for sentiment analysis")
        return None


def _safe_import_spacy():
    """Safely import spaCy."""
    try:
        import spacy  # type: ignore
        return spacy
    except ImportError:
        logger.debug("spaCy not available for advanced NLP")
        return None


def clean_text(text: str, lowercase: bool = True, remove_punctuation: bool = True) -> str:
    """Basic text cleaning utility.
    
    Args:
        text: Input text to clean
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text


def tokenize(text: str, method: str = 'simple') -> List[str]:
    """Tokenize text using various methods.
    
    Args:
        text: Input text to tokenize
        method: Tokenization method ('simple', 'whitespace', 'nltk', 'spacy')
        
    Returns:
        List of tokens
    """
    text = clean_text(text, lowercase=True, remove_punctuation=False)
    
    if method == 'simple':
        # Basic word tokenization
        return re.findall(r'\b\w+\b', text.lower())
    elif method == 'whitespace':
        return text.split()
    elif method == 'nltk':
        try:
            from nltk.tokenize import word_tokenize  # type: ignore
            return word_tokenize(text)
        except ImportError:
            logger.debug("NLTK tokenizer not available, using simple tokenization")
            return re.findall(r'\b\w+\b', text.lower())
    elif method == 'spacy':
        spacy = _safe_import_spacy()
        if spacy:
            try:
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(text)
                return [token.text for token in doc]
            except Exception:
                logger.debug("spaCy model not available, using simple tokenization")
                return re.findall(r'\b\w+\b', text.lower())
        else:
            return re.findall(r'\b\w+\b', text.lower())
    else:
        return re.findall(r'\b\w+\b', text.lower())


def jaccard_similarity(text1: str, text2: str, preprocess: bool = True) -> float:
    """Calculate Jaccard similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        preprocess: Whether to preprocess texts
        
    Returns:
        Jaccard similarity score (0-1)
    """
    if preprocess:
        text1 = clean_text(text1)
        text2 = clean_text(text2)
    
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    
    if not tokens1 and not tokens2:
        return 1.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union) if union else 0.0


def cosine_similarity_texts(text1: str, text2: str, method: str = 'tfidf') -> float:
    """Calculate cosine similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity calculation method ('tfidf', 'simple')
        
    Returns:
        Cosine similarity score (0-1)
    """
    if method == 'tfidf':
        TfidfVectorizer, cosine_sim = _safe_import_sklearn()
        if TfidfVectorizer and cosine_sim:
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                similarity = cosine_sim(tfidf_matrix[0], tfidf_matrix[1])
                return float(similarity[0][0])
            except Exception as e:
                logger.debug(f"TF-IDF similarity failed: {e}")
                return jaccard_similarity(text1, text2)
        else:
            logger.debug("sklearn not available, using Jaccard similarity")
            return jaccard_similarity(text1, text2)
    else:
        # Simple word overlap similarity
        tokens1 = Counter(tokenize(text1))
        tokens2 = Counter(tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(tokens1[token] * tokens2.get(token, 0) for token in tokens1)
        
        # Calculate magnitudes
        magnitude1 = sum(count ** 2 for count in tokens1.values()) ** 0.5
        magnitude2 = sum(count ** 2 for count in tokens2.values()) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


def get_sentiment_analyzer(cached: Optional[Any] = None) -> Optional[Any]:
    """Return a VADER sentiment analyzer or None if unavailable."""
    if cached is not None:
        return cached
    
    SIA = _safe_import_nltk_vader()
    if SIA is None:
        return None
    
    try:
        return SIA()
    except Exception as e:
        logger.debug(f"Failed to initialize VADER sentiment analyzer: {e}")
        return None


def polarity_scores(text: str, analyzer: Optional[Any] = None) -> Dict[str, float]:
    """Analyze sentiment of text using VADER or fallback method.
    
    Args:
        text: Text to analyze
        analyzer: Pre-initialized sentiment analyzer
        
    Returns:
        Dictionary with sentiment scores
    """
    if not text or not isinstance(text, str):
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    
    # Use VADER if available
    if analyzer is None:
        analyzer = get_sentiment_analyzer()
    
    if analyzer:
        try:
            return analyzer.polarity_scores(text)
        except Exception as e:
            logger.debug(f"VADER sentiment analysis failed: {e}")
    
    # Fallback: simple keyword-based sentiment
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'enjoy', 'happy', 'pleased'}
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'sad', 'disappointed', 'poor'}
    
    tokens = tokenize(text.lower())
    pos_count = sum(1 for token in tokens if token in positive_words)
    neg_count = sum(1 for token in tokens if token in negative_words)
    neu_count = len(tokens) - pos_count - neg_count
    
    total = max(1, len(tokens))
    
    # Simple compound score calculation
    compound = (pos_count - neg_count) / total
    
    return {
        'neg': neg_count / total,
        'neu': neu_count / total,
        'pos': pos_count / total,
        'compound': compound
    }


class BatchProcessor:
    """Batch processor for handling large datasets efficiently."""
    
    def __init__(self, max_workers: int = 4, timeout: Optional[float] = None):
        """Initialize batch processor.
        
        Args:
            max_workers: Maximum number of worker threads
            timeout: Timeout for batch processing
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = None
    
    def __enter__(self):
        self.executor = cf.ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def process_batch(self, items: List[Any], process_func: callable, 
                     batch_size: int = 100) -> List[Any]:
        """Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            batch_size: Size of each batch
            
        Returns:
            List of processed results
        """
        if not self.executor:
            raise RuntimeError("BatchProcessor must be used as context manager")
        
        futures = []
        results = []
        
        # Create batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            future = self.executor.submit(self._process_batch, batch, process_func)
            futures.append(future)
        
        # Collect results
        for future in cf.as_completed(futures, timeout=self.timeout):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results.extend([None] * batch_size)  # Fallback
        
        return results
    
    def _process_batch(self, batch: List[Any], process_func: callable) -> List[Any]:
        """Process a single batch."""
        results = []
        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                results.append(None)
        return results


def analyze_sentiment_batch(texts: List[str]) -> List[Dict[str, float]]:
    """Analyze sentiment for multiple texts."""
    analyzer = get_sentiment_analyzer()
    results = []
    for text in texts:
        sentiment = polarity_scores(text, analyzer)
        results.append(sentiment)
    return results


def process_batch(metas: List[Any]) -> List[Any]:
    """Process metadata batch with sentiment analysis."""
    results = []
    for meta in metas:
        if isinstance(meta, str):
            tokens = tokenize(meta)
            analyzer = get_sentiment_analyzer()
            sentiment = polarity_scores(meta, analyzer)
            result = {'text': meta, 'tokens': tokens, 'sentiment': sentiment}
            results.append(result)
            logger.debug(f"Обработан текст: {len(tokens)} токенов, sentiment {sentiment}")
        else:
            results.append(meta)
    return results


def compute_semantic_similarity(texts, model=None):
    """
    Вычисляет семантическое сходство между двумя текстами.
    
    Args:
        texts: список из двух строк [text1, text2]
        model: игнорируется (для совместимости с API внешних модулей)
    
    Returns:
        float: значение сходства от 0.0 до 1.0
    """
    if not isinstance(texts, (list, tuple)) or len(texts) != 2:
        raise ValueError("compute_semantic_similarity ожидает список из двух текстов")
    text1, text2 = texts
    return cosine_similarity_texts(text1, text2)


def polarity_scores(text: str) -> Dict[str, float]:
    """
    Возвращает словарь с оценками тональности в формате, совместимом с VADER.
    Ключи: 'pos', 'neu', 'neg', 'compound'.
    """
    result = _processor.analyze_sentiment(text, detailed=True)
    return {
        'pos': result['pos'],
        'neu': result['neu'],
        'neg': result['neg'],
        'compound': result['compound']
    }


def tokenize(text: str) -> List[str]:
    """Токенизирует текст (обёртка над tokenize_text с методом 'advanced')."""
    return _processor.tokenize_text(text, method='advanced')


def get_stopwords(languages=("english", "russian")) -> Set[str]:
    """
    Возвращает множество стоп-слов для указанных языков.
    В текущей реализации стоп-слова едины для всех языков.
    """
    # Возвращаем копию, чтобы избежать случайного изменения оригинального множества
    return set(_processor.stop_words)


__all__ = [
    'tokenize',
    'get_sentiment_analyzer',
    'polarity_scores',
    'cosine_similarity_texts',
    'jaccard_similarity',
    'compute_semantic_similarity',
    'get_stopwords',
    'BatchProcessor',
    'process_batch',
    'analyze_sentiment_batch'
]
