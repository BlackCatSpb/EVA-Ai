"""
Graph-Based Generator - Тестовая архитектура генерации ответа из графа памяти

Это ОТДЕЛЬНАЯ тестовая система для валидации подхода.
Не связана с текущей EVA системой.

Цель: Проверить, может ли граф памяти полностью заменить GGUF модель для генерации ответов.
"""

import os
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.fractal_graph_v2.test_generation")


@dataclass
class GenerationResult:
    """Результат генерации ответа."""
    response: str
    confidence: float
    sources: List[str]           # ID узлов-источников
    reasoning: str                # Цепочка рассуждений
    fallback_used: bool = False   # Использовался ли fallback


@dataclass
class QueryContext:
    """Контекст запроса для генерации."""
    query: str
    tokens: List[str]
    key_concepts: List[str]       # Извлечённые концепты из запроса
    required_level: int           # Уровень детализации


class GraphBasedGenerator:
    """
    Генератор ответов на основе графа памяти.
    """
    
    def __init__(self, graph, tokenizer, fallback_llm=None):
        self.graph = graph
        self.tokenizer = tokenizer
        self.fallback_llm = fallback_llm
    
    def parse_query(self, query: str) -> QueryContext:
        """Разбор запроса на компоненты."""
        tokens = self.tokenizer._tokenize_text(query)
        
        stop_words = {'это', 'что', 'как', 'почему', 'когда', 'где', 'кто', 'какой', 'какая', 'какое', 'какие'}
        key_concepts = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        required_level = 1
        if 'подробно' in query.lower() or 'детально' in query.lower() or 'объясни' in query.lower():
            required_level = 2
        elif 'кратко' in query.lower() or 'вкратце' in query.lower():
            required_level = 0
        
        return QueryContext(
            query=query,
            tokens=tokens,
            key_concepts=key_concepts,
            required_level=required_level
        )
    
    def retrieve_context(self, context: QueryContext) -> Dict[str, Any]:
        """Извлечь контекст из графа."""
        
        # Получаем семантические результаты
        semantic_results = self.graph.semantic_search(context.query, top_k=10, min_level=1)
        
        # Обрабатываем разные форматы результатов
        processed_semantic = []
        for item in semantic_results:
            if isinstance(item, dict):
                processed_semantic.append(item)
            elif isinstance(item, tuple) and len(item) >= 2:
                # (node_id, similarity, group_id) format from storage
                processed_semantic.append({'id': item[0], 'similarity': item[1], 'group_id': item[2] if len(item) > 2 else None})
        
        # Keyword поиск
        keyword_nodes = []
        for concept in context.key_concepts:
            nodes = self.graph.keyword_search(concept, top_k=5)
            if nodes:
                for node in nodes:
                    if hasattr(node, 'id'):
                        keyword_nodes.append({'id': node.id, 'content': node.content})
                    elif isinstance(node, dict):
                        keyword_nodes.append(node)
        
        # Объединяем уникальные узлы
        all_nodes = {}
        for node in processed_semantic + keyword_nodes:
            if isinstance(node, dict):
                node_id = node.get('id')
            else:
                node_id = getattr(node, 'id', None)
            if node_id and node_id not in all_nodes:
                all_nodes[node_id] = node
        
        # Получаем полный контекст для каждого узла
        contexts = []
        for node_id, node_data in all_nodes.items():
            full_context = self.graph.get_context(node_id)
            if full_context:
                contexts.append(full_context)
        
        return {
            'semantic_results': processed_semantic,
            'keyword_nodes': list(all_nodes.values()),
            'full_contexts': contexts,
            'total_sources': len(contexts)
        }
    
    def build_response_from_graph(self, context: QueryContext, graph_context: Dict[str, Any]) -> GenerationResult:
        """Построить ответ из графа."""
        
        if not graph_context['full_contexts']:
            return GenerationResult(
                response="",
                confidence=0.0,
                sources=[],
                reasoning="Граф памяти пуст",
                fallback_used=True
            )
        
        query_lower = context.query.lower()
        response_parts = []
        sources = []
        confidence_scores = []
        
        if 'что такое' in query_lower or 'кто такой' in query_lower:
            for fc in graph_context['full_contexts']:
                node = fc.get('node', {})
                if node.get('node_type') in ['concept', 'fact']:
                    response_parts.append(node.get('content', ''))
                    sources.append(node.get('id', ''))
                    confidence_scores.append(node.get('confidence', 0.5))
        
        elif 'как' in query_lower or 'какой' in query_lower or 'какая' in query_lower:
            for fc in graph_context['full_contexts']:
                related = fc.get('related', [])
                attrs = [r for r in related if r.get('relation') in ['has_property', 'attribute_of']]
                if attrs:
                    subject = fc.get('node', {}).get('content', '')
                    attr_list = ', '.join([r['node']['content'] for r in attrs[:5]])
                    response_parts.append(f"{subject}: {attr_list}")
                    confidence_scores.extend([r['node'].get('confidence', 0.5) for r in attrs])
        
        elif 'связан' in query_lower or 'связь' in query_lower or 'отно' in query_lower:
            for fc in graph_context['full_contexts']:
                related = fc.get('related', [])
                if related:
                    subject = fc.get('node', {}).get('content', '')
                    rels = [f"{subject} - {r['relation']} - {r['node']['content']}" for r in related[:5]]
                    response_parts.extend(rels)
                    confidence_scores.extend([r['node'].get('confidence', 0.5) for r in related[:5]])
        
        if not response_parts:
            for fc in graph_context['full_contexts'][:5]:
                node = fc.get('node', {})
                response_parts.append(node.get('content', ''))
                sources.append(node.get('id', ''))
                confidence_scores.append(node.get('confidence', 0.5))
        
        if response_parts:
            unique_parts = []
            seen = set()
            for part in response_parts:
                key = part[:50].lower()
                if key not in seen:
                    seen.add(key)
                    unique_parts.append(part)
            
            response = '. '.join(unique_parts[:5])
            confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            return GenerationResult(
                response=response,
                confidence=confidence,
                sources=sources[:10],
                reasoning=f"Извлечено из {graph_context['total_sources']} источников",
                fallback_used=False
            )
        
        return GenerationResult(
            response="Недостаточно информации в графе.",
            confidence=0.0,
            sources=[],
            reasoning="Запрос не найден",
            fallback_used=True
        )
    
    def generate(self, query: str) -> GenerationResult:
        """Основной метод генерации."""
        logger.info(f"Generation: '{query[:50]}...'")
        
        context = self.parse_query(query)
        graph_context = self.retrieve_context(context)
        result = self.build_response_from_graph(context, graph_context)
        
        if result.fallback_used:
            result.response = (
                "Граф памяти содержит недостаточно знаний.\n"
                "Для загрузки используйте load_knowledge_from_gguf()"
            )
        
        return result


def create_graph_memory_system(storage_dir=None, gguf_model_path=None, embedding_device="cuda"):
    """Фабричная функция."""
    # Import here to avoid circular imports
    from . import create_fractal_memory_graph
    from .tokenizer import create_graph_tokenizer
    
    class GraphMemorySystem:
        def __init__(self):
            logger.info("Initializing GraphMemorySystem...")
            self.graph = create_fractal_memory_graph(
                storage_dir=storage_dir,
                embedding_device=embedding_device
            )
            self.tokenizer = create_graph_tokenizer(self.graph)
            self.generator = GraphBasedGenerator(self.graph, self.tokenizer)
            self.gguf_model_path = gguf_model_path
            self.extractor = None
        
        def add_knowledge(self, subject, relation, obj, subj_level=1, obj_level=1):
            return self.graph.add_knowledge(subject, relation, obj, subj_level, obj_level)
        
        def generate_response(self, query):
            return self.generator.generate(query)
        
        def semantic_search(self, query, top_k=5):
            return self.graph.semantic_search(query, top_k=top_k)
        
        def vectorize_all(self):
            self.graph.vectorize_all()
        
        def get_stats(self):
            return self.graph.get_stats()
    
    return GraphMemorySystem()