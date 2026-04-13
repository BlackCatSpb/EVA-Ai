"""
ConceptExtractor - извлечение концептов из запросов и ответов
Создаёт узлы типа 'concept' в FGv2 с фактами
"""
import re
import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.knowledge.concept_extractor")


@dataclass
class Concept:
    """Представление концепта с метаданными."""
    name: str
    description: str
    domain: str
    source: str
    confidence: float
    related_terms: List[str]
    facts: List[Dict[str, Any]]  # Факты о концепте для создания противоречий


class ConceptExtractor:
    """
    Извлекает концепты из текста и сохраняет в FractalGraph v2.
    
    Флоу:
    1. Извлечь ключевые термины из запроса + ответа
    2. Для каждого термина создать Concept с фактами
    3. Сохранить Concept как узел типа 'concept' в FGv2
    4. Создать связи с запросом/ответом
    """
    
    def __init__(self, fractal_graph=None, brain=None):
        self.brain = brain
        self._fg = fractal_graph
        self._stop_words = self._load_stop_words()
        self._known_concepts: Set[str] = set()  # Кэш известных концептов
        
    def _load_stop_words(self) -> Set[str]:
        """Загружает стоп-слова для русского и английского."""
        return {
            'это', 'что', 'как', 'где', 'когда', 'почему', 'потому', 'для', 
            'от', 'до', 'при', 'над', 'под', 'между', 'который', 'которая', 
            'которое', 'свой', 'своя', 'своё', 'быть', 'был', 'была', 'было', 
            'были', 'есть', 'will', 'are', 'was', 'were', 'have', 'has', 
            'the', 'a', 'an', 'is', 'been', 'being', 'and', 'or', 'but',
            'чем', 'такой', 'такая', 'такое', 'такие', 'все', 'весь',
            'можно', 'нужно', 'надо', 'должен', 'должна', 'должно'
        }
    
    def extract_concepts(self, query: str, response: str, 
                        context: Optional[Dict] = None) -> List[Concept]:
        """
        Извлекает концепты из запроса и ответа.
        
        Args:
            query: Запрос пользователя
            response: Ответ системы
            context: Дополнительный контекст
            
        Returns:
            Список извлечённых концептов
        """
        # Объединяем текст для анализа
        full_text = f"{query} {response}".lower()
        
        # Извлекаем ключевые термины
        terms = self._extract_terms(full_text)
        
        # Фильтруем уже известные концепты
        new_terms = [t for t in terms if t not in self._known_concepts]
        
        concepts = []
        for term in new_terms[:5]:  # Максимум 5 новых концептов за раз
            concept = self._create_concept(term, query, response, context)
            if concept:
                concepts.append(concept)
                self._known_concepts.add(term)
        
        logger.info(f"Извлечено {len(concepts)} новых концептов из {len(terms)} терминов")
        return concepts
    
    def _extract_terms(self, text: str) -> List[str]:
        """
        Извлекает ключевые термины из текста.
        
        Стратегия:
        1. Находим слова 4+ символов
        2. Подсчитываем частоту
        3. Убираем стоп-слова
        4. Возвращаем топ по частоте
        """
        # Находим слова (русские и английские)
        words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())
        
        # Подсчитываем частоту
        freq = {}
        for word in words:
            if word not in self._stop_words:
                freq[word] = freq.get(word, 0) + 1
        
        # Сортируем по частоте и возвращаем топ-15
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:15]]
    
    def _create_concept(self, term: str, query: str, response: str,
                       context: Optional[Dict]) -> Optional[Concept]:
        """
        Создаёт концепт с фактами на основе термина.
        
        Генерирует различные факты о концепте для потенциальных противоречий:
        - is_a: определение (что это)
        - has_property: свойства
        - related_to: связанные понятия
        - can: возможности/действия
        """
        # Определяем домен из контекста или запроса
        domain = self._detect_domain(query, response)
        
        # Генерируем факты о концепте
        facts = self._generate_facts(term, query, response)
        
        # Находим связанные термины
        related = self._find_related_terms(term, query, response)
        
        concept = Concept(
            name=term,
            description=f"Концепт извлечён из: {query[:50]}...",
            domain=domain,
            source="extraction",
            confidence=0.7,
            related_terms=related,
            facts=facts
        )
        
        return concept
    
    def _detect_domain(self, query: str, response: str) -> str:
        """Определяет домен концепта из текста."""
        text = (query + " " + response).lower()
        
        # Простая эвристика по ключевым словам
        domains = {
            'science': ['наука', 'физика', 'химия', 'биология', 'математика', 'science', 'physics'],
            'technology': ['технология', 'компьютер', 'программа', 'код', 'technology', 'computer'],
            'philosophy': ['философия', 'мышление', 'сознание', 'philosophy', 'mind'],
            'general': []
        }
        
        for domain, keywords in domains.items():
            if any(kw in text for kw in keywords):
                return domain
        
        return 'general'
    
    def _generate_facts(self, term: str, query: str, response: str) -> List[Dict[str, Any]]:
        """
        Генерирует факты о концепте из текста query/response.
        
        Извлекает реальные факты, а не шаблоны.
        """
        facts = []
        text = f"{query} {response}".lower()
        
        # Извлекаем предложения содержащие term
        sentences = [s.strip() for s in response.split('.') if term.lower() in s.lower()]
        
        # Факт 1: Определение (ищем "это", "является", "называется")
        definition_patterns = [
            f'{term.lower()} - это', f'{term.lower()} является',
            f'{term.lower()} называется', f'{term.lower()} — это',
            f'это {term.lower()}', f'называют {term.lower()}'
        ]
        
        for sent in sentences:
            for pat in definition_patterns:
                if pat in sent:
                    facts.append({
                        'relation_type': 'is_a',
                        'value': sent.strip()[:200],
                        'confidence': 0.8,
                        'source': 'extraction'
                    })
                    break
            if len(facts) >= 2:
                break
        
        # Факт 2: Свойства (ищем прилагательные и признаки)
        property_words = ['большой', 'маленький', 'новый', 'старый', 'важный', 'основной',
                         'ключевой', 'простой', 'сложный', 'быстрый', 'медленный',
                         'important', 'big', 'small', 'new', 'old', 'main', 'key']
        
        for sent in sentences[:5]:
            for pw in property_words:
                if pw in sent:
                    facts.append({
                        'relation_type': 'has_property',
                        'value': sent.strip()[:200],
                        'confidence': 0.7,
                        'source': 'extraction'
                    })
                    break
            if len(facts) >= 4:
                break
        
        # Факт 3: Возможности (ищем "может", "умеет", "способен", "allows", "can")
        capability_patterns = ['может', 'умеет', 'способен', 'позволяет', 'allows', 'can', 'able to']
        
        for sent in sentences[:5]:
            for pat in capability_patterns:
                if pat in sent:
                    facts.append({
                        'relation_type': 'can',
                        'value': sent.strip()[:200],
                        'confidence': 0.7,
                        'source': 'extraction'
                    })
                    break
            if len(facts) >= 6:
                break
        
        # Факт 4: Связи (ищем "связан", "относится", "связан с", "related to")
        relation_patterns = ['связан с', 'относится к', 'влияет на', 'связано с',
                             'related to', 'connected to', 'associated with']
        
        for sent in sentences[:5]:
            for pat in relation_patterns:
                if pat in sent:
                    facts.append({
                        'relation_type': 'related_to',
                        'value': sent.strip()[:200],
                        'confidence': 0.6,
                        'source': 'extraction'
                    })
                    break
            if len(facts) >= 8:
                break
        
        # Fallback: если ничего не найдено - используем общий факт из контекста
        if not facts:
            if len(response) > 50:
                facts.append({
                    'relation_type': 'description',
                    'value': response[:200],
                    'confidence': 0.5,
                    'source': 'extraction'
                })
        
        return facts[:8]  # Максимум 8 фактов
    
    def _find_related_terms(self, term: str, query: str, response: str) -> List[str]:
        """Находит термины, связанные с данным в тексте."""
        text = query + " " + response
        words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())
        
        # Простая эвристика: слова в том же предложении
        sentences = re.split(r'[.!?]+', text)
        related = []
        
        for sent in sentences:
            if term in sent.lower():
                sent_words = re.findall(r'\b[а-яёa-z]{4,}\b', sent.lower())
                for word in sent_words:
                    if word != term and word not in self._stop_words and len(word) > 3:
                        related.append(word)
        
        return list(set(related))[:5]  # Максимум 5 связанных
    
    def save_concept_to_graph(self, concept: Concept) -> Optional[str]:
        """
        Сохраняет концепт в FractalGraph v2.
        
        Args:
            concept: Концепт для сохранения
            
        Returns:
            ID узла или None
        """
        if not self._fg:
            logger.warning("FractalGraph не доступен для сохранения концепта")
            return None
        
        try:
            # Создаём узел типа 'concept'
            node = self._fg.add_node(
                content=concept.name,
                node_type='concept',
                metadata={
                    'description': concept.description,
                    'domain': concept.domain,
                    'source': concept.source,
                    'confidence': concept.confidence,
                    'related_terms': concept.related_terms,
                    'facts': concept.facts,
                    'extracted_at': time.time()
                }
            )
            
            if node:
                # Сохраняем факты как отдельные связи/атрибуты
                self._save_concept_facts(node.id, concept.facts)
                
                logger.debug(f"Концепт '{concept.name}' сохранён с ID {node.id}")
                return node.id
            
        except Exception as e:
            logger.error(f"Ошибка сохранения концепта '{concept.name}': {e}")
        
        return None
    
    def _save_concept_facts(self, concept_id: str, facts: List[Dict[str, Any]]):
        """Сохраняет факты о концепте как свойства узла."""
        if not self._fg or not hasattr(self._fg.storage, 'nodes'):
            return
        
        try:
            node = self._fg.storage.nodes.get(concept_id)
            if node and hasattr(node, 'metadata'):
                if not hasattr(node.metadata, 'facts'):
                    node.metadata['facts'] = []
                node.metadata['facts'].extend(facts)
        except Exception as e:
            logger.debug(f"Ошибка сохранения фактов: {e}")
    
    def get_concept_facts(self, concept_name: str) -> List[Dict[str, Any]]:
        """
        Получает все факты о концепте из графа.
        
        Args:
            concept_name: Имя концепта
            
        Returns:
            Список фактов
        """
        if not self._fg:
            return []
        
        try:
            # Ищем узел по имени
            for node_id, node in self._fg.storage.nodes.items():
                if hasattr(node, 'content') and node.content == concept_name:
                    if hasattr(node, 'metadata') and 'facts' in node.metadata:
                        return node.metadata['facts']
        except Exception as e:
            logger.debug(f"Ошибка получения фактов: {e}")
        
        return []
    
    def get_concepts_for_prompt(self, query: str, max_concepts: int = 3) -> str:
        """
        Извлекает концепты из запроса и формирует промпт для генерации.
        
        Args:
            query: Запрос пользователя
            max_concepts: Максимум концептов
            
        Returns:
            Строка с контекстом концептов для промпта
        """
        try:
            # Извлекаем термины из запроса
            terms = self._extract_terms(query.lower())
            
            if not terms:
                return ""
            
            prompt_parts = []
            found_count = 0
            
            for term in terms[:max_concepts * 2]:  # Берём с запасом
                if found_count >= max_concepts:
                    break
                    
                # Ищем концепт в графе
                facts = self.get_concept_facts(term)
                if facts:
                    fact_texts = [f.get('value', '') for f in facts[:2]]
                    if fact_texts:
                        prompt_parts.append(f"[{term}]: {', '.join(fact_texts)}")
                        found_count += 1
            
            if prompt_parts:
                return "\n".join(["Известные концепты:", *prompt_parts, ""])
            
            return ""
            
        except Exception as e:
            logger.debug(f"Ошибка формирования промпта концептов: {e}")
            return ""
    
    def save_learned_concept(self, concept_name: str, new_facts: List[str], source: str = "dialog"):
        """
        Сохраняет новые знания о концепте после обработки.
        
        Args:
            concept_name: Имя концепта
            new_facts: Новые факты
            source: Источник (dialog, web, etc.)
        """
        if not self._fg:
            return
        
        try:
            # Ищем существующий концепт
            concept_id = None
            for node_id, node in self._fg.storage.nodes.items():
                if hasattr(node, 'content') and node.content == concept_name:
                    concept_id = node_id
                    break
            
            if concept_id:
                # Добавляем факты к существующему
                node = self._fg.storage.nodes.get(concept_id)
                if node and hasattr(node, 'metadata'):
                    if 'facts' not in node.metadata:
                        node.metadata['facts'] = []
                    
                    for fact_text in new_facts:
                        node.metadata['facts'].append({
                            'relation_type': 'learned',
                            'value': fact_text,
                            'source': source,
                            'timestamp': time.time()
                        })
                    
                    logger.debug(f"Добавлено {len(new_facts)} фактов к '{concept_name}'")
            else:
                # Создаём новый концепт
                concept = Concept(
                    name=concept_name,
                    description=f"Узнано из {source}",
                    domain="general",
                    source=source,
                    confidence=0.6,
                    related_terms=[],
                    facts=[{
                        'relation_type': 'learned',
                        'value': fact,
                        'source': source
                    } for fact in new_facts]
                )
                self.save_concept_to_graph(concept)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения знаний о концепте: {e}")
    
    def format_concept_for_dialog(self, concept: Concept) -> str:
        """
        Форматирует концепт в текст для самодиалога.
        
        Пример:
        "Концепт: [term]
         Домен: [domain]
         Факты:
         1. [fact1]
         2. [fact2]"
        """
        lines = [
            f"Концепт: {concept.name}",
            f"Домен: {concept.domain}",
            f"Описание: {concept.description}",
            "Факты:"
        ]
        
        for i, fact in enumerate(concept.facts[:4], 1):
            value = fact.get('value', '')
            relation = fact.get('relation_type', 'unknown')
            lines.append(f"  {i}. [{relation}] {value}")
        
        if concept.related_terms:
            lines.append(f"Связанные термины: {', '.join(concept.related_terms)}")
        
        return '\n'.join(lines)


def create_concept_extractor(fractal_graph=None, brain=None) -> ConceptExtractor:
    """Factory function для создания ConceptExtractor."""
    return ConceptExtractor(fractal_graph=fractal_graph, brain=brain)
