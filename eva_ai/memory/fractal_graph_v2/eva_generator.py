"""
EVAGenerator - Единый генератор для EVA с гибридными токенами

Интегрирует:
- HybridTokenizer (BPE + виртуальные токены)
- RecursiveModelPipeline (модели A/B/C)
- GGUFShadowProfiler (маршрутизация)
- SemanticContextCache (необработанный контекст)
- Quality checks (оценка качества)

Принцип работы:
1. Входной текст → HybridTokenizer → BPE + виртуальные токены
2. SemanticContextCache → семантический поиск по необработанному контексту
3. Виртуальные токены → маршрутизация через GGUFShadowProfiler
4. Генерация через RecursiveModelPipeline
5. Постобработка: замена виртуальных токенов на контент узлов
"""

import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .hybrid_tokenizer import HybridTokenizer, Token
from .gguf_shadow import GGUFShadowProfiler
from .semantic_context_cache import SemanticContextCache

logger = logging.getLogger("eva_ai.fractal_graph_v2.eva_generator")

QUERY_TYPE_KEYWORDS = {
    'кратко': ['кратко', 'вкратце', 'суть', 'что такое', 'кто такой', 'дай определение', 'назови', 'перечисли'],
    'подробно': ['подробно', 'детально', 'развернуто', 'расскажи', 'объясни', 'опиши', 'проанализируй']
}

GENERATION_PARAMS = {
    'кратко': {
        'model_a': {'temperature': 0.1, 'max_tokens': 128, 'top_p': 0.8, 'repeat_penalty': 1.5},
        'model_b': {'temperature': 0.5, 'max_tokens': 256, 'top_p': 0.8, 'repeat_penalty': 1.5}
    },
    'подробно': {
        'model_a': {'temperature': 0.2, 'max_tokens': 256, 'top_p': 0.85, 'repeat_penalty': 1.5},
        'model_b': {'temperature': 0.6, 'max_tokens': 512, 'top_p': 0.85, 'repeat_penalty': 1.5}
    }
}


@dataclass
class GenerationRequest:
    """Запрос на генерацию."""
    text: str
    query_type: str = 'подробно'
    conversation_history: List[Dict] = None
    max_tokens: int = 512
    temperature: float = 0.5
    session_id: str = None


@dataclass
class GenerationResult:
    """Результат генерации."""
    response: str
    confidence: float
    quality_score: float
    virtual_tokens_used: List[str]
    reasoning_steps: List[Dict]
    processing_time: float
    query_type: str


class EVAGenerator:
    """
    Единый генератор EVA с поддержкой виртуальных токенов.
    
    Использует:
    - HybridTokenizer для токенизации с виртуальными токенами
    - SemanticContextCache для семантического поиска по необработанному контексту
    - RecursiveModelPipeline для генерации (опционально)
    - GGUFShadowProfiler для маршрутизации
    """
    
    def __init__(
        self,
        fractal_graph,
        model_pipeline=None,
        gguf_shadow=None,
        base_tokenizer=None,
        semantic_cache: SemanticContextCache = None,
        max_semantic_contexts: int = 500
    ):
        """
        Args:
            fractal_graph: FractalGraphV2 instance
            model_pipeline: RecursiveModelPipeline (опционально)
            gguf_shadow: GGUFShadowProfiler (опционально)
            base_tokenizer: Базовый BPE токенизатор (опционально)
            semantic_cache: SemanticContextCache (опционально)
            max_semantic_contexts: Максимум контекстов в SemanticContextCache
        """
        self.graph = fractal_graph
        self.pipeline = model_pipeline
        self.gguf_shadow = gguf_shadow
        
        self.tokenizer = HybridTokenizer(
            fractal_graph=fractal_graph,
            base_tokenizer=base_tokenizer
        )
        
        self.semantic_cache = semantic_cache
        if self.semantic_cache is None:
            self.semantic_cache = SemanticContextCache(
                max_contexts=max_semantic_contexts,
                embedding_dim=384,
                use_faiss=True
            )
        
        self.total_generations = 0
        self.total_quality_checks = 0
        
        logger.info("EVAGenerator инициализирован с SemanticContextCache")
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Генерация ответа.
        
        Args:
            request: GenerationRequest с текстом и параметрами
            
        Returns:
            GenerationResult
        """
        start_time = time.time()
        self.total_generations += 1
        
        reasoning_steps = []
        
        reasoning_steps.append({
            'step': 1,
            'phase': 'tokenization',
            'action': 'Токенизация с виртуальными токенами'
        })
        
        tokens = self.tokenizer.encode(request.text)
        virtual_tokens = [t for t in tokens if t.is_virtual]
        virtual_node_ids = [t.node_id for t in virtual_tokens if t.node_id]
        
        reasoning_steps.append({
            'step': 2,
            'phase': 'entity_extraction',
            'action': f'Извлечено {len(virtual_node_ids)} сущностей из графа'
        })
        
        reasoning_steps.append({
            'step': 3,
            'phase': 'semantic_search',
            'action': 'Поиск в SemanticContextCache'
        })
        
        semantic_results = self.semantic_cache.search(
            request.text, 
            top_k=3, 
            min_similarity=0.4,
            session_filter=getattr(request, 'session_id', None)
        )
        
        reasoning_steps.append({
            'step': 3.5,
            'phase': 'semantic_results',
            'action': f'Найдено {len(semantic_results)} релевантных контекстов'
        })
        
        query_type = self._determine_query_type(request.text)
        reasoning_steps.append({
            'step': 4,
            'phase': 'query_type_detection',
            'action': f'Тип запроса: {query_type}'
        })
        
        params = self._get_generation_params(query_type)
        
        routing_config = None
        if self.gguf_shadow and virtual_node_ids:
            routing_config = self._get_routing_for_entities(virtual_node_ids)
            if routing_config:
                params = self._merge_params(params, routing_config)
                reasoning_steps.append({
                    'step': 5,
                    'phase': 'routing_applied',
                    'action': 'Применены параметры маршрутизации из графа'
                })
        
        prompt = self._build_prompt(
            request.text, 
            request.conversation_history, 
            virtual_node_ids,
            semantic_results
        )
        reasoning_steps.append({
            'step': 6,
            'phase': 'prompt_building',
            'action': f'Промпт сформирован ({len(prompt)} символов)'
        })
        
        response = self._generate_response(prompt, params)
        reasoning_steps.append({
            'step': 6,
            'phase': 'generation',
            'action': f'Сгенерировано {len(response)} символов'
        })
        
        quality = self._check_quality(response, request.text)
        self.total_quality_checks += 1
        reasoning_steps.append({
            'step': 7,
            'phase': 'quality_check',
            'action': f'Качество: {quality["score"]:.2f}'
        })
        
        if not quality.get('is_gibberish', False):
            response = self._sanitize_response(response)
        
        response = self._postprocess_virtual_tokens(response)
        reasoning_steps.append({
            'step': 8,
            'phase': 'postprocessing',
            'action': 'Заменены виртуальные токены на контент'
        })
        
        return GenerationResult(
            response=response,
            confidence=quality.get('score', 0.7),
            quality_score=quality.get('score', 0.7),
            virtual_tokens_used=virtual_node_ids,
            reasoning_steps=reasoning_steps,
            processing_time=time.time() - start_time,
            query_type=query_type
        )
    
    def _determine_query_type(self, text: str) -> str:
        """Определить тип запроса."""
        text_lower = text.lower()
        
        for kw in QUERY_TYPE_KEYWORDS['кратко']:
            if kw in text_lower:
                return 'кратко'
        
        for kw in QUERY_TYPE_KEYWORDS['подробно']:
            if kw in text_lower:
                return 'подробно'
        
        return 'подробно'
    
    def _get_generation_params(self, query_type: str) -> Dict[str, Any]:
        """Получить параметры генерации."""
        return GENERATION_PARAMS.get(query_type, GENERATION_PARAMS['подробно'])
    
    def _get_routing_for_entities(self, entity_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Получить параметры маршрутизации для сущностей."""
        if not self.gguf_shadow:
            return None
        
        try:
            for entity_id in entity_ids:
                if entity_id in self.graph.nodes:
                    node = self.graph.nodes[entity_id]
                    metadata = getattr(node, 'metadata', {})
                    
                    routing = metadata.get('action', {})
                    if routing:
                        return routing.get('parameters')
            
            return None
        except Exception as e:
            logger.debug(f"Routing error: {e}")
            return None
    
    def _merge_params(self, base_params: Dict, routing_params: Dict) -> Dict:
        """Слить параметры базовые и маршрутизации."""
        merged = {}
        for model_key, model_params in base_params.items():
            if isinstance(model_params, dict):
                merged[model_key] = {**model_params}
                if isinstance(routing_params, dict):
                    for k, v in routing_params.items():
                        if k in ['temperature', 'max_tokens', 'top_p', 'repeat_penalty']:
                            merged[model_key][k] = v
        return merged
    
    def _build_prompt(
        self, 
        text: str, 
        history: Optional[List[Dict]],
        entity_ids: List[str],
        semantic_results: List[Dict] = None
    ) -> str:
        """Построить промпт с контекстом."""
        parts = []
        semantic_results = semantic_results or []
        
        if semantic_results:
            semantic_contexts = [r['text'] for r in semantic_results[:3]]
            parts.append(f"Релевантные контексты: {' | '.join(semantic_contexts)}")
        
        if entity_ids:
            entity_contents = []
            for eid in entity_ids[:5]:
                if eid in self.graph.nodes:
                    node = self.graph.nodes[eid]
                    content = getattr(node, 'content', '')
                    if content:
                        entity_contents.append(content)
            
            if entity_contents:
                parts.append(f"Контекст из графа: {', '.join(entity_contents[:3])}")
        
        if history:
            recent = history[-5:]
            history_parts = []
            for msg in recent:
                role = 'Пользователь' if msg.get('role') == 'user' else 'Ассистент'
                content = msg.get('content', '')[:200]
                if content:
                    history_parts.append(f"{role}: {content}")
            
            if history_parts:
                parts.append(f"История: {' | '.join(history_parts)}")
        
        parts.append(f"Вопрос: {text}")
        
        return "\n\n".join(parts)
    
    def _generate_response(self, prompt: str, params: Dict) -> str:
        """Сгенерировать ответ."""
        if self.pipeline:
            try:
                result = self.pipeline.process_query(
                    query=prompt,
                    gen_params=params
                )
                return result.get('model_b_result', {}).get('natural_response', '') or result.get('response', '')
            except Exception as e:
                logger.warning(f"Pipeline generation failed: {e}")
        
        return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Fallback генерация без модели."""
        if 'контекст из графа' in prompt.lower():
            parts = prompt.split('\n\n')
            for part in parts:
                if part.startswith('Контекст из графа:'):
                    context = part.replace('Контекст из графа:', '').strip()
                    if context:
                        return f"Основываясь на имеющихся данных: {context}"
        
        return "Информация обрабатывается. Пожалуйста, уточните вопрос."
    
    def _check_quality(self, response: str, query: str) -> Dict[str, Any]:
        """Проверить качество ответа."""
        if not response or len(response.strip()) < 5:
            return {'score': 0.1, 'is_gibberish': True, 'reasons': ['Пустой ответ']}
        
        is_gibberish = False
        reasons = []
        words = response.split()
        
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                is_gibberish = True
                reasons.append('Много повторений')
        
        vowels = set('аеёиоуыэюяaeiou')
        if not any(v in response.lower() for v in vowels):
            is_gibberish = True
            reasons.append('Нет гласных (мусор)')
        
        score = 0.8
        if is_gibberish:
            score = 0.2
        elif len(response) < 50:
            score = 0.5
        
        return {'score': score, 'is_gibberish': is_gibberish, 'reasons': reasons or ['OK']}
    
    def _sanitize_response(self, response: str) -> str:
        """Очистить ответ от артефактов."""
        filler_prefixes = [
            'хорошо,', 'давайте', 'начнём', 'итак,', 'что ж,',
            'ok,', 'okay,', 'well,', 'sure,', 'of course,'
        ]
        
        lines = response.split('\n')
        if lines:
            first_line_lower = lines[0].lower().strip()
            for prefix in filler_prefixes:
                if first_line_lower.startswith(prefix):
                    lines[0] = lines[0][len(prefix):].strip()
                    break
        
        response = '\n'.join(line for line in lines if line.strip())
        
        sentences = response.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        seen = {}
        for i, sent in enumerate(sentences):
            sent_norm = ' '.join(sent.lower().split())
            if sent_norm in seen and i - seen[sent_norm] >= 2:
                sentences = sentences[:i]
                break
            else:
                seen[sent_norm] = i
        
        return '. '.join(sentences) + ('.' if sentences and not response.endswith('.') else '')
    
    def _postprocess_virtual_tokens(self, response: str) -> str:
        """Заменить виртуальные токены на контент узлов."""
        virtual_pattern = r'<virtual_(\d+)>'
        
        def replace_virtual(match):
            token_id = int(match.group(1))
            info = self.tokenizer.get_virtual_token_info(token_id)
            if info:
                return f"[{info['content']}]"
            return match.group(0)
        
        return re.sub(virtual_pattern, replace_virtual, response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            'total_generations': self.total_generations,
            'total_quality_checks': self.total_quality_checks,
            'virtual_token_range': self.tokenizer.get_stats(),
            'pipeline_available': self.pipeline is not None,
            'gguf_shadow_available': self.gguf_shadow is not None
        }


def create_eva_generator(
    fractal_graph,
    model_pipeline=None,
    gguf_shadow=None,
    base_tokenizer=None
) -> EVAGenerator:
    """Фабричная функция."""
    return EVAGenerator(
        fractal_graph=fractal_graph,
        model_pipeline=model_pipeline,
        gguf_shadow=gguf_shadow,
        base_tokenizer=base_tokenizer
    )