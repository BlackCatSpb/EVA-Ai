"""
Text chunking for handling large prompts that exceed model context window.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

MAX_INPUT_TOKENS_MODEL_A = 1800
MAX_INPUT_TOKENS_MODEL_B = 1500
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 50


class TextChunker:
    """Разбиение текста на чанки с учётом токенов"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._token_cache = {}
    
    def estimate_tokens(self, text: str) -> int:
        """Оценка количества токенов в тексте"""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        return len(text) // 4
    
    def chunk_text_by_tokens(
        self, 
        text: str, 
        max_tokens: int = CHUNK_SIZE_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS
    ) -> List[str]:
        """Разбивает текст на чанки с заданным лимитом токенов"""
        if not text:
            return []
        
        estimated = self.estimate_tokens(text)
        if estimated <= max_tokens:
            return [text]
        
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.estimate_tokens(line)
            
            if current_tokens + line_tokens > max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                if overlap > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]
                    current_tokens = self.estimate_tokens('\n'.join(current_chunk))
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        if not chunks:
            chars_per_token = len(text) / max(estimated, 1)
            chunk_size_chars = int(max_tokens * chars_per_token)
            for i in range(0, len(text), chunk_size_chars):
                chunks.append(text[i:i + chunk_size_chars])
        
        logger.info(f"Text chunked into {len(chunks)} parts (estimated {estimated} tokens)")
        return chunks
    
    def chunk_by_sentences(
        self, 
        text: str, 
        max_tokens: int = CHUNK_SIZE_TOKENS
    ) -> List[str]:
        """Разбивает текст по предложениям"""
        if not text:
            return []
        
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = self.estimate_tokens(sent)
            
            if current_tokens + sent_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class ChunkedGenerator:
    """Генерация с разбиением больших запросов"""
    
    def __init__(self, text_chunker: TextChunker, model_generator):
        self.chunker = text_chunker
        self.model_generator = model_generator
    
    def generate_with_chunking(
        self,
        query: str,
        max_input_tokens: int = MAX_INPUT_TOKENS_MODEL_A,
        system_prompt_template: str = "Извлеки ключевые факты из текста. Отвечай кратко.",
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Генерация с автоматическим разбиением если нужно"""
        estimated_input = self.chunker.estimate_tokens(query)
        
        if estimated_input <= max_input_tokens:
            return self.model_generator(query, **generation_kwargs)
        
        logger.info(f"Query too large ({estimated_input} tokens), chunking...")
        
        chunks = self.chunker.chunk_by_sentences(
            query, 
            max_tokens=max_input_tokens - 200
        )
        
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_prompt = f"{system_prompt_template}\n\nТекст: {chunk}"
            result = self.model_generator(chunk_prompt, **generation_kwargs)
            
            if result.get('status') == 'generating':
                return result
            
            if result.get('raw_response'):
                results.append(result['raw_response'])
        
        if not results:
            return {
                'raw_response': '',
                'natural_response': '',
                'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['No results from chunks']},
                'tokens': 0,
                'chunks': len(chunks)
            }
        
        merged = self.merge_chunk_results(results)
        
        return {
            'raw_response': merged,
            'natural_response': merged,
            'quality': {'is_gibberish': False, 'score': 0.8, 'reasons': ['OK']},
            'tokens': self.chunker.estimate_tokens(merged),
            'chunks': len(chunks),
            'merged': True
        }
    
    def merge_chunk_results(
        self, 
        chunks: List[str], 
        strategy: str = "sequential"
    ) -> str:
        """Объединяет результаты чанков"""
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]
        
        if strategy == "sequential":
            merged = ""
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue
                
                if i > 0:
                    if not merged.endswith(('.', '!', '?')):
                        merged += "."
                    merged += " "
                
                merged += chunk
            
            return merged.strip()
        
        elif strategy == "facts":
            facts = []
            for chunk in chunks:
                for line in chunk.split('\n'):
                    line = line.strip()
                    if line and len(line) > 10:
                        facts.append(line)
            return "; ".join(facts)
        
        return " ".join(chunks)


def create_text_chunker(tokenizer=None) -> TextChunker:
    """Фабрика для создания TextChunker"""
    return TextChunker(tokenizer)


def chunk_query_if_needed(
    query: str,
    max_tokens: int,
    chunker: Optional[TextChunker] = None
) -> List[str]:
    """Утилита для проверки и разбиения запроса"""
    if chunker is None:
        chunker = TextChunker()
    
    estimated = chunker.estimate_tokens(query)
    
    if estimated <= max_tokens:
        return [query]
    
    return chunker.chunk_by_sentences(query, max_tokens=max_tokens - 200)
