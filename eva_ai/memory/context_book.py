"""
ContextBook - Превращает контекст FractalGraph в интерактивную книгу с индексацией

Позволяет:
1. Организовать узлы графа как страницы книги
2. Создать оглавление (Table of Contents) с индексацией
3. Навигировать по страницам с сохранением контекста
4. Быстро находить нужную информацию через индекс
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re

logger = logging.getLogger("eva_ai.context_book")


@dataclass
class BookIndex:
    """Индекс книги для быстрого поиска."""
    keywords: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))  # слово -> страницы
    topics: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))    # тема -> страницы
    concepts: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))  # концепт -> страницы
    
    def add_entry(self, term: str, page_number: int, entry_type: str = "keyword"):
        """Добавляет запись в индекс."""
        term_lower = term.lower()
        if entry_type == "topic":
            self.topics[term_lower].append(page_number)
        elif entry_type == "concept":
            self.concepts[term_lower].append(page_number)
        else:
            self.keywords[term_lower].append(page_number)
    
    def search(self, query: str) -> List[Tuple[int, float]]:
        """Поиск по индексу, возвращает страницы с релевантностью."""
        query_terms = query.lower().split()
        page_scores = defaultdict(float)
        
        for term in query_terms:
            # Ищем в keywords (вес 1.0)
            for page in self.keywords.get(term, []):
                page_scores[page] += 1.0
            
            # Ищем в topics (вес 2.0)
            for page in self.topics.get(term, []):
                page_scores[page] += 2.0
            
            # Ищем в concepts (вес 1.5)
            for page in self.concepts.get(term, []):
                page_scores[page] += 1.5
        
        # Сортируем по релевантности
        results = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return results
    
    def get_toc(self) -> Dict[str, List[int]]:
        """Получить оглавление (Table of Contents)."""
        toc = {}
        # Группируем по первой букве для алфавитного указателя
        for concept in sorted(self.concepts.keys()):
            first_letter = concept[0].upper() if concept else "#"
            if first_letter not in toc:
                toc[first_letter] = []
            toc[first_letter].append({
                'term': concept,
                'pages': sorted(set(self.concepts[concept]))
            })
        return toc


@dataclass
class BookPage:
    """Страница виртуальной книги контекста."""
    page_number: int
    title: str
    content: str
    source_nodes: List[str]  # ID узлов графа
    summary: str = ""  # Краткое содержание страницы
    topics: List[str] = field(default_factory=list)
    related_pages: List[int] = field(default_factory=list)  # Связанные страницы
    
    def to_text(self, include_metadata: bool = True) -> str:
        """Конвертация в текстовый формат."""
        text = f"""
{'='*60}
Страница {self.page_number}: {self.title}
{'='*60}

{self.content}

"""
        if include_metadata:
            if self.summary:
                text += f"\n[Краткое содержание: {self.summary}]\n"
            if self.related_pages:
                text += f"[См. также: страницы {', '.join(map(str, self.related_pages))}]\n"
        
        return text


@dataclass
class ContextBook:
    """Виртуальная книга контекста из FractalGraph."""
    book_id: str
    title: str = "Книга контекста"
    pages: List[BookPage] = field(default_factory=list)
    index: BookIndex = field(default_factory=BookIndex)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def get_page(self, page_number: int) -> Optional[BookPage]:
        """Получить страницу по номеру."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]
        return None
    
    def find_pages(self, query: str, top_k: int = 5) -> List[BookPage]:
        """Найти страницы по запросу через индекс."""
        results = self.index.search(query)
        pages = []
        for page_num, score in results[:top_k]:
            page = self.get_page(page_num)
            if page:
                page.metadata = {'relevance_score': score}
                pages.append(page)
        return pages
    
    def get_table_of_contents(self) -> str:
        """Получить текстовое оглавление."""
        toc = self.index.get_toc()
        
        text = f"""
{'='*60}
ОГЛАВЛЕНИЕ: {self.title}
{'='*60}

Всего страниц: {len(self.pages)}

Алфавитный указатель концептов:
"""
        for letter in sorted(toc.keys()):
            text += f"\n{letter}:\n"
            for entry in toc[letter]:
                pages_str = ', '.join(map(str, entry['pages']))
                text += f"  • {entry['term']} ... стр. {pages_str}\n"
        
        return text
    
    def get_context_for_generation(self, query: str, max_pages: int = 3) -> str:
        """Получить контекст для генерации ответа."""
        # Находим релевантные страницы
        relevant_pages = self.find_pages(query, top_k=max_pages)
        
        if not relevant_pages:
            # Если ничего не нашли - берем первые страницы как введение
            relevant_pages = self.pages[:2]
        
        # Формируем контекст
        context_parts = [f"Контекст из книги '{self.title}':"]
        context_parts.append(f"(Найдено {len(relevant_pages)} релевантных страниц)\n")
        
        for page in relevant_pages:
            context_parts.append(page.to_text(include_metadata=False))
        
        return "\n---\n".join(context_parts)


class ContextBookBuilder:
    """
    Строитель книги контекста из FractalGraph.
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.fractal_graph = getattr(brain, 'fractal_graph_v2', None)
        
    def build_from_graph(
        self,
        query: Optional[str] = None,
        title: str = "Книга контекста",
        pages_per_topic: int = 3,
        tokens_per_page: int = 500
    ) -> ContextBook:
        """
        Создает книгу контекста из узлов FractalGraph.
        
        Args:
            query: Тематический запрос для организации (опционально)
            title: Название книги
            pages_per_topic: Сколько страниц на тему
            tokens_per_page: Токенов на страницу
            
        Returns:
            ContextBook - готовая книга с индексом
        """
        if not self.fractal_graph:
            logger.error("FractalGraph не доступен")
            return ContextBook(book_id="empty", title="Empty Book")
        
        # Получаем узлы из графа
        nodes = self._collect_nodes(query)
        
        # Группируем по темам
        topic_groups = self._group_by_topic(nodes)
        
        # Создаем книгу
        book = ContextBook(
            book_id=f"context_book_{int(time.time())}",
            title=title,
            metadata={
                'source_query': query,
                'total_nodes': len(nodes),
                'topics_count': len(topic_groups)
            }
        )
        
        page_number = 1
        
        # Для каждой темы создаем страницы
        for topic, topic_nodes in topic_groups.items():
            # Создаем страницы для темы
            for i in range(0, min(len(topic_nodes), pages_per_topic)):
                node = topic_nodes[i] if i < len(topic_nodes) else None
                if not node:
                    continue
                
                content = self._extract_content(node)
                summary = self._generate_summary(content)
                
                page = BookPage(
                    page_number=page_number,
                    title=f"{topic} (часть {i+1})" if len(topic_nodes) > 1 else topic,
                    content=content[:tokens_per_page * 4],  # ~4 символа на токен
                    source_nodes=[str(node.get('id', 'unknown'))],
                    summary=summary,
                    topics=[topic],
                    related_pages=[]  # Заполним позже
                )
                
                book.pages.append(page)
                
                # Добавляем в индекс
                book.index.add_entry(topic, page_number, "topic")
                
                # Извлекаем концепты для индекса
                concepts = self._extract_concepts(content)
                for concept in concepts:
                    book.index.add_entry(concept, page_number, "concept")
                
                page_number += 1
        
        # Добавляем связи между страницами
        self._add_page_links(book)
        
        logger.info(f"Создана книга контекста: {len(book.pages)} страниц, "
                   f"{len(book.index.concepts)} концептов в индексе")
        
        return book
    
    def _collect_nodes(self, query: Optional[str] = None) -> List[Dict]:
        """Собирает узлы из графа."""
        nodes = []
        
        try:
            # Пробуем семантический поиск если есть запрос
            if query and hasattr(self.fractal_graph, 'semantic_search'):
                results = self.fractal_graph.semantic_search(query, top_k=50)
                for r in results:
                    nodes.append({
                        'id': r.get('id'),
                        'content': r.get('content', ''),
                        'type': r.get('type', 'unknown'),
                        'score': r.get('score', 0)
                    })
            
            # Если мало результатов - добавляем все узлы
            if len(nodes) < 10:
                if hasattr(self.fractal_graph, 'storage'):
                    fg = self.fractal_graph.storage
                    for node_id, node in getattr(fg, 'nodes', {}).items():
                        nodes.append({
                            'id': node_id,
                            'content': getattr(node, 'content', ''),
                            'type': getattr(node, 'node_type', 'unknown'),
                            'level': getattr(node, 'level', 0)
                        })
                        if len(nodes) >= 100:  # Ограничиваем для производительности
                            break
        
        except Exception as e:
            logger.error(f"Ошибка сбора узлов: {e}")
        
        return nodes
    
    def _group_by_topic(self, nodes: List[Dict]) -> Dict[str, List[Dict]]:
        """Группирует узлы по темам."""
        groups = defaultdict(list)
        
        for node in nodes:
            node_type = node.get('type', 'unknown')
            content = node.get('content', '')
            
            # Определяем тему по типу узла и содержимому
            if node_type == 'concept':
                topic = self._extract_topic_from_content(content) or "Концепты"
            elif node_type == 'fact':
                topic = "Факты"
            elif node_type == 'query':
                topic = "Вопросы"
            elif node_type == 'response':
                topic = "Ответы"
            else:
                topic = node_type.capitalize()
            
            groups[topic].append(node)
        
        # Сортируем группы по количеству узлов
        return dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))
    
    def _extract_topic_from_content(self, content: str) -> Optional[str]:
        """Извлекает тему из содержимого."""
        # Берем первые 2-3 слова как тему
        words = content.split()[:3]
        if words:
            return ' '.join(words).capitalize()
        return None
    
    def _extract_content(self, node: Dict) -> str:
        """Извлекает текстовое содержимое из узла."""
        content = node.get('content', '')
        if not content and 'text' in node:
            content = node['text']
        return content
    
    def _generate_summary(self, content: str, max_length: int = 100) -> str:
        """Генерирует краткое содержание."""
        sentences = re.split(r'[.!?]', content)
        if sentences:
            first = sentences[0].strip()
            if len(first) > max_length:
                return first[:max_length] + "..."
            return first
        return ""
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Извлекает концепты из текста для индексации."""
        concepts = []
        
        # Простая эвристика: длинные слова (возможно термины)
        words = re.findall(r'\b[A-Za-zА-Яа-я]{5,}\b', content)
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Берем слова которые встречаются 1-3 раза (специфичные термины)
        for word, freq in word_freq.items():
            if 1 <= freq <= 3:
                concepts.append(word)
        
        return concepts[:10]  # Ограничиваем количество
    
    def _add_page_links(self, book: ContextBook):
        """Добавляет связи между страницами."""
        for i, page in enumerate(book.pages):
            # Связь с предыдущей страницей
            if i > 0:
                page.related_pages.append(book.pages[i-1].page_number)
            
            # Связь со следующей страницей
            if i < len(book.pages) - 1:
                page.related_pages.append(book.pages[i+1].page_number)
            
            # Связь с тематически близкими страницами (те же темы)
            for other_page in book.pages:
                if other_page.page_number != page.page_number:
                    if set(page.topics) & set(other_page.topics):
                        page.related_pages.append(other_page.page_number)
            
            # Убираем дубликаты и ограничиваем
            page.related_pages = sorted(set(page.related_pages))[:5]


# ===== ИНТЕГРАЦИЯ С DUALGENERATOR =====

class ContextBookMixin:
    """Миксин для интеграции ContextBook с генераторами."""
    
    def __init__(self, brain):
        self.brain = brain
        self.context_book_builder = ContextBookBuilder(brain)
        self._active_book: Optional[ContextBook] = None
        self._book_cache: Dict[str, ContextBook] = {}
    
    def build_context_book(
        self,
        query: Optional[str] = None,
        title: str = "Книга знаний системы",
        use_cache: bool = True
    ) -> ContextBook:
        """
        Создает книгу контекста из текущего состояния графа.
        
        Args:
            query: Тематический запрос для фильтрации
            title: Название книги
            use_cache: Использовать кэш если книга уже создавалась
            
        Returns:
            ContextBook - индексированная книга контекста
        """
        cache_key = f"{title}:{query or 'all'}"
        
        if use_cache and cache_key in self._book_cache:
            logger.info(f"Использована кэшированная книга: {title}")
            return self._book_cache[cache_key]
        
        # Строим новую книгу
        book = self.context_book_builder.build_from_graph(
            query=query,
            title=title,
            pages_per_topic=3,
            tokens_per_page=500
        )
        
        self._active_book = book
        self._book_cache[cache_key] = book
        
        return book
    
    def query_book(self, query: str, top_k: int = 3) -> str:
        """
        Выполняет запрос к активной книге контекста.
        
        Args:
            query: Вопрос
            top_k: Количество страниц для контекста
            
        Returns:
            Отформатированный контекст для генерации
        """
        if not self._active_book:
            # Создаем книгу если еще нет
            self._active_book = self.build_context_book()
        
        return self._active_book.get_context_for_generation(query, max_pages=top_k)
    
    def get_book_toc(self) -> str:
        """Получить оглавление активной книги."""
        if not self._active_book:
            return "Книга не создана"
        return self._active_book.get_table_of_contents()
    
    def get_book_stats(self) -> Dict[str, Any]:
        """Статистика книги."""
        if not self._active_book:
            return {'error': 'No active book'}
        
        return {
            'title': self._active_book.title,
            'pages': len(self._active_book.pages),
            'concepts_indexed': len(self._active_book.index.concepts),
            'topics': len(self._active_book.index.topics),
            'created_at': self._active_book.created_at
        }
