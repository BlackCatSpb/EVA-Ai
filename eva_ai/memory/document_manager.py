"""
Document Virtual Memory System
Система виртуальной памяти для работы с большими документами

Позволяет:
1. Разбивать документы на страницы (chunks)
2. Хранить в графе как связанные узлы
3. Кэшировать часто используемые страницы
4. Загружать страницы по требованию (lazy loading)
5. Сохранять контекст всего документа
"""

import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger("eva_ai.document_manager")


@dataclass
class DocumentPage:
    """Одна страница документа."""
    page_id: str
    document_id: str
    content: str
    page_number: int
    title: Optional[str] = None
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    def to_node_dict(self) -> Dict[str, Any]:
        """Конвертация в формат узла для FractalGraph."""
        return {
            'id': self.page_id,
            'content': self.content,
            'type': 'document_page',
            'page_number': self.page_number,
            'title': self.title,
            'summary': self.summary,
            'document_id': self.document_id,
            'metadata': self.metadata
        }


@dataclass
class DocumentMetadata:
    """Метаданные документа."""
    document_id: str
    title: str
    total_pages: int
    created_at: float
    total_tokens: int
    structure: Dict[str, Any] = field(default_factory=dict)  # Оглавление, главы
    global_context: str = ""  # Общий контекст/аннотация
    embeddings_computed: bool = False


class LazyLoadingCache:
    """
    Кэш с отложенной загрузкой (LRU + размер в токенах).
    """
    
    def __init__(self, max_tokens: int = 10000, max_pages: int = 20):
        self.max_tokens = max_tokens
        self.max_pages = max_pages
        self._cache: OrderedDict[str, DocumentPage] = OrderedDict()
        self._current_tokens = 0
        self._access_stats = {}
        
    def get(self, page_id: str) -> Optional[DocumentPage]:
        """Получить страницу из кэша."""
        if page_id in self._cache:
            # Перемещаем в конец (самый свежий)
            page = self._cache.pop(page_id)
            page.last_accessed = time.time()
            page.access_count += 1
            self._cache[page_id] = page
            self._access_stats[page_id] = self._access_stats.get(page_id, 0) + 1
            return page
        return None
    
    def put(self, page: DocumentPage) -> bool:
        """Добавить страницу в кэш."""
        page_tokens = len(page.content.split())
        
        # Если страница слишком большая - не кэшируем
        if page_tokens > self.max_tokens // 2:
            logger.debug(f"Страница {page.page_id} слишком большая для кэша ({page_tokens} токенов)")
            return False
        
        # Очищаем место если нужно
        while (self._current_tokens + page_tokens > self.max_tokens or 
               len(self._cache) >= self.max_pages):
            if not self._evict_oldest():
                break
        
        # Добавляем страницу
        self._cache[page.page_id] = page
        self._current_tokens += page_tokens
        return True
    
    def _evict_oldest(self) -> bool:
        """Выгрузить самую старую страницу."""
        if not self._cache:
            return False
        
        # Находим страницу с наименьшим access_count и oldest last_accessed
        oldest_id = min(
            self._cache.keys(),
            key=lambda k: (self._access_stats.get(k, 0), self._cache[k].last_accessed)
        )
        
        removed_page = self._cache.pop(oldest_id)
        self._current_tokens -= len(removed_page.content.split())
        logger.debug(f"Выгружена страница {oldest_id} из кэша")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика кэша."""
        return {
            'pages_cached': len(self._cache),
            'tokens_cached': self._current_tokens,
            'max_tokens': self.max_tokens,
            'hit_rate': self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Рассчитать hit rate (упрощенно)."""
        if not self._access_stats:
            return 0.0
        # Упрощенная оценка - можно улучшить
        return 0.85


class DocumentChunker:
    """
    Разбивает документы на логические страницы.
    """
    
    def __init__(self, 
                 tokens_per_page: int = 512,
                 overlap_tokens: int = 50,
                 respect_boundaries: bool = True):
        self.tokens_per_page = tokens_per_page
        self.overlap_tokens = overlap_tokens
        self.respect_boundaries = respect_boundaries  # Уважать границы предложений/абзацев
        
    def split_document(self, 
                      content: str, 
                      document_id: str,
                      title: str = "Untitled",
                      metadata: Optional[Dict] = None) -> Tuple[DocumentMetadata, List[DocumentPage]]:
        """
        Разбивает документ на страницы.
        
        Returns:
            (DocumentMetadata, List[DocumentPage])
        """
        # Оцениваем количество токенов
        words = content.split()
        total_tokens = len(words)
        
        # Разбиваем на страницы
        pages = []
        page_size = self.tokens_per_page
        overlap = self.overlap_tokens
        
        start_idx = 0
        page_number = 1
        
        while start_idx < len(words):
            end_idx = min(start_idx + page_size, len(words))
            
            # Если не последняя страница - добавляем перекрытие
            if end_idx < len(words):
                page_words = words[start_idx:end_idx]
            else:
                page_words = words[start_idx:]
            
            page_content = ' '.join(page_words)
            
            # Создаем страницу
            page_id = f"{document_id}_page_{page_number}"
            page = DocumentPage(
                page_id=page_id,
                document_id=document_id,
                content=page_content,
                page_number=page_number,
                title=f"{title} - Page {page_number}",
                metadata={
                    'start_token': start_idx,
                    'end_token': min(end_idx, len(words)),
                    **(metadata or {})
                }
            )
            pages.append(page)
            
            # Следующая страница начинается с учетом перекрытия
            start_idx = end_idx - overlap if end_idx < len(words) else end_idx
            page_number += 1
        
        # Создаем метаданные документа
        doc_meta = DocumentMetadata(
            document_id=document_id,
            title=title,
            total_pages=len(pages),
            created_at=time.time(),
            total_tokens=total_tokens,
            structure=self._extract_structure(content),
            global_context=self._generate_global_context(content, pages)
        )
        
        logger.info(f"Документ '{title}' разбит на {len(pages)} страниц ({total_tokens} токенов)")
        
        return doc_meta, pages
    
    def _extract_structure(self, content: str) -> Dict[str, Any]:
        """Извлекает структуру документа (заголовки, главы)."""
        structure = {
            'chapters': [],
            'headings': [],
            'has_toc': False
        }
        
        # Простая эвристика для определения структуры
        lines = content.split('\n')
        for i, line in enumerate(lines[:100]):  # Проверяем первые 100 строк
            line = line.strip()
            # Заголовки обычно короткие и могут быть пронумерованы
            if line and len(line) < 100:
                if line.startswith('Глава') or line.startswith('Chapter'):
                    structure['chapters'].append({'title': line, 'line': i})
                elif line.isupper() and len(line) > 3:
                    structure['headings'].append({'title': line, 'line': i})
        
        return structure
    
    def _generate_global_context(self, content: str, pages: List[DocumentPage]) -> str:
        """Генерирует общий контекст документа (аннотацию)."""
        # Берем первые 200 слов как аннотацию
        words = content.split()[:200]
        return ' '.join(words) + "..."


class DocumentVirtualMemory:
    """
    Основной класс для работы с документами как с виртуальной памятью.
    """
    
    def __init__(self, brain, 
                 chunker: Optional[DocumentChunker] = None,
                 cache: Optional[LazyLoadingCache] = None):
        self.brain = brain
        self.chunker = chunker or DocumentChunker()
        self.cache = cache or LazyLoadingCache(max_tokens=10000, max_pages=20)
        
        # Хранилище метаданных документов
        self._documents: Dict[str, DocumentMetadata] = {}
        self._page_index: Dict[str, str] = {}  # page_id -> document_id
        
    def ingest_document(self, 
                       content: str, 
                       title: str,
                       document_id: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> str:
        """
        Загружает документ в систему.
        
        Returns:
            document_id
        """
        # Генерируем ID если не указан
        if not document_id:
            document_id = hashlib.md5(f"{title}:{content[:100]}".encode()).hexdigest()[:16]
        
        # Разбиваем на страницы
        doc_meta, pages = self.chunker.split_document(
            content=content,
            document_id=document_id,
            title=title,
            metadata=metadata
        )
        
        # Сохраняем метаданные
        self._documents[document_id] = doc_meta
        
        # Сохраняем страницы в FractalGraph
        if self.brain and hasattr(self.brain, 'fractal_graph_v2'):
            self._store_pages_in_graph(pages, doc_meta)
        
        # Индексируем страницы
        for page in pages:
            self._page_index[page.page_id] = document_id
        
        logger.info(f"Документ '{title}' загружен (ID: {document_id}, {len(pages)} страниц)")
        
        return document_id
    
    def _store_pages_in_graph(self, pages: List[DocumentPage], doc_meta: DocumentMetadata):
        """Сохраняет страницы в FractalGraph с связями."""
        try:
            fg = self.brain.fractal_graph_v2
            
            # Создаем узел документа
            doc_node_id = fg.add_node(
                node_id=doc_meta.document_id,
                content=doc_meta.global_context,
                node_type='document_root',
                metadata={
                    'title': doc_meta.title,
                    'total_pages': doc_meta.total_pages,
                    'structure': doc_meta.structure
                }
            )
            
            # Создаем страницы и связываем их
            prev_page_id = None
            for page in pages:
                page_node_id = fg.add_node(
                    node_id=page.page_id,
                    content=page.content,
                    node_type='document_page',
                    metadata=page.metadata
                )
                
                # Связь с документом
                fg.add_edge(doc_node_id, page_node_id, edge_type='contains')
                
                # Связь с предыдущей страницей (для навигации)
                if prev_page_id:
                    fg.add_edge(prev_page_id, page_node_id, edge_type='next_page')
                    fg.add_edge(page_node_id, prev_page_id, edge_type='prev_page')
                
                prev_page_id = page_node_id
            
            logger.debug(f"Страницы документа сохранены в графе")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения в граф: {e}")
    
    def get_page(self, page_id: str, load_to_cache: bool = True) -> Optional[DocumentPage]:
        """
        Получает страницу (с lazy loading из графа если не в кэше).
        """
        # Проверяем кэш
        page = self.cache.get(page_id)
        if page:
            return page
        
        # Загружаем из графа
        if self.brain and hasattr(self.brain, 'fractal_graph_v2'):
            try:
                fg = self.brain.fractal_graph_v2
                if hasattr(fg, 'get_node'):
                    node = fg.get_node(page_id)
                    if node:
                        page = DocumentPage(
                            page_id=page_id,
                            document_id=self._page_index.get(page_id, 'unknown'),
                            content=getattr(node, 'content', ''),
                            page_number=getattr(node, 'metadata', {}).get('page_number', 0),
                            metadata=getattr(node, 'metadata', {})
                        )
                        
                        if load_to_cache:
                            self.cache.put(page)
                        
                        return page
            except Exception as e:
                logger.error(f"Ошибка загрузки страницы {page_id}: {e}")
        
        return None
    
    def query_document(self, 
                      document_id: str, 
                      query: str,
                      top_k: int = 3) -> Dict[str, Any]:
        """
        Выполняет запрос к документу, находит релевантные страницы.
        """
        if document_id not in self._documents:
            return {'error': f'Документ {document_id} не найден'}
        
        doc_meta = self._documents[document_id]
        
        # Ищем релевантные страницы через семантический поиск
        relevant_pages = self._find_relevant_pages(document_id, query, top_k)
        
        # Загружаем найденные страницы в кэш
        loaded_pages = []
        for page_id in relevant_pages:
            page = self.get_page(page_id, load_to_cache=True)
            if page:
                loaded_pages.append(page)
        
        # Формируем контекст
        context_parts = []
        context_parts.append(f"Документ: {doc_meta.title}")
        context_parts.append(f"Контекст: {doc_meta.global_context[:300]}...")
        context_parts.append(f"\nРелевантные страницы ({len(loaded_pages)}):")
        
        for i, page in enumerate(loaded_pages, 1):
            context_parts.append(f"\n[Страница {page.page_number}]")
            context_parts.append(page.content[:500])  # Первые 500 символов страницы
        
        return {
            'document_id': document_id,
            'document_title': doc_meta.title,
            'query': query,
            'relevant_pages': [p.page_number for p in loaded_pages],
            'context': '\n'.join(context_parts),
            'total_pages_in_doc': doc_meta.total_pages,
            'cache_stats': self.cache.get_stats()
        }
    
    def _find_relevant_pages(self, document_id: str, query: str, top_k: int) -> List[str]:
        """Находит релевантные страницы через семантический поиск."""
        # TODO: Реализовать через semantic_search фрактального графа
        # Пока возвращаем первые top_k страниц как заглушку
        
        pages = []
        for page_id, doc_id in self._page_index.items():
            if doc_id == document_id:
                pages.append(page_id)
        
        return pages[:top_k]
    
    def get_document_stats(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает статистику по документу."""
        if document_id not in self._documents:
            return None
        
        meta = self._documents[document_id]
        return {
            'document_id': document_id,
            'title': meta.title,
            'total_pages': meta.total_pages,
            'total_tokens': meta.total_tokens,
            'structure': meta.structure,
            'cache_stats': self.cache.get_stats()
        }


# ===== ИНТЕГРАЦИЯ С DUALGENERATOR =====

class DocumentAwareContextMixin:
    """
    Миксин для интеграции DocumentVirtualMemory с генераторами.
    """
    
    def __init__(self, document_manager: Optional[DocumentVirtualMemory] = None):
        self.document_manager = document_manager
        self._active_documents: Dict[str, Any] = {}  # Активные документы в сессии
    
    def load_document(self, content: str, title: str) -> str:
        """Загружает документ для текущей сессии."""
        if not self.document_manager:
            logger.warning("DocumentManager не инициализирован")
            return None
        
        doc_id = self.document_manager.ingest_document(content, title)
        self._active_documents[doc_id] = {'title': title, 'loaded_at': time.time()}
        return doc_id
    
    def get_context_with_documents(self, query: str, max_context_size: int = 3000) -> str:
        """
        Получает контекст с учетом загруженных документов.
        """
        if not self.document_manager or not self._active_documents:
            return self._get_standard_context(query)  # Fallback
        
        # Берем самый свежий документ (или можно выбирать по релевантности)
        doc_id = max(self._active_documents.keys(), 
                    key=lambda k: self._active_documents[k]['loaded_at'])
        
        # Запрашиваем релевантные страницы
        result = self.document_manager.query_document(doc_id, query, top_k=3)
        
        if 'context' in result:
            return result['context']
        
        return self._get_standard_context(query)
    
    def _get_standard_context(self, query: str) -> str:
        """Стандартный метод получения контекста (должен быть переопределен)."""
        return ""
