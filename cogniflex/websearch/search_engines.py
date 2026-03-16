"""Поисковые движки для веб-поиска CogniFlex"""
import requests
import logging
import re
import json
import time
from typing import List
from urllib.parse import quote, urljoin
from bs4 import BeautifulSoup
from .search_models import SearchResult

logger = logging.getLogger("cogniflex.web_search.engines")

class SearchEngines:
    """Класс для работы с различными поисковыми системами."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # API endpoints
        self.duckduckgo_url = "https://html.duckduckgo.com/html/"
        self.searx_url = "https://searx.be/search"

    def search_google(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Google (через DuckDuckGo)."""
        return self.search_duckduckgo(query, max_results)

    def search_yandex(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Yandex (через DuckDuckGo)."""
        return self.search_duckduckgo(query, max_results)

    def search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Bing."""
        try:
            url = f"https://www.bing.com/search?q={quote(query)}&count={max_results}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Поиск результатов в HTML
            search_results = soup.find_all('li', class_='b_algo')
            
            for i, result in enumerate(search_results[:max_results]):
                try:
                    title_elem = result.find('h2')
                    link_elem = title_elem.find('a') if title_elem else None
                    snippet_elem = result.find('p')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        url = link_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                        
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="bing",
                            relevance_score=0.8 - i * 0.1,
                            query=query
                        ))
                except Exception as e:
                    logger.debug(f"Ошибка парсинга результата Bing: {e}")
                    continue
            
            logger.debug(f"Найдено {len(results)} результатов Bing для: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска Bing: {e}")
            return []
    
    def search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через DuckDuckGo с приоритетом локальных результатов."""
        try:
            logger.debug(f"Выполняется поиск DuckDuckGo: {query}")
            
            # Сначала пробуем локальные результаты для скорости
            local_results = self._create_local_results(query, max_results)
            if local_results and any(r.source == "local_knowledge" for r in local_results):
                logger.info(f"Используем локальные результаты для: {query}")
                return local_results
            
            # Если локальных результатов недостаточно, пробуем веб
            params = {
                'q': query,
                'kl': 'ru-ru'
            }
            
            # Выполняем запрос
            response = self.session.get("https://html.duckduckgo.com/html/", params=params, timeout=5)
            response.raise_for_status()
            
            # Парсим HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Ищем результаты
            for result in soup.find_all('div', class_='result')[:max_results]:
                try:
                    # Заголовок и ссылка
                    title_elem = result.find('a', class_='result__a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Описание
                    snippet_elem = result.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    # Создаем результат
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo",
                        relevance_score=self._calculate_relevance(title, snippet, query),
                        query=query
                    )
                    results.append(search_result)
                    
                except Exception as e:
                    logger.debug(f"Ошибка парсинга результата: {e}")
                    continue
            
            if results:
                logger.info(f"Найдено {len(results)} результатов через DuckDuckGo")
                return results
            else:
                logger.warning(f"DuckDuckGo не дал результатов, используем локальные: {query}")
                return local_results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo недоступен, используем локальные результаты: {e}")
            return self._create_local_results(query, max_results)
    
    def _create_local_results(self, query: str, max_results: int) -> List[SearchResult]:
        """Создает локальные результаты на основе запроса."""
        logger.info(f"Создание локальных результатов для: {query}")
        
        results = []
        query_lower = query.lower()
        
        # База знаний по темам
        knowledge_base = {
            'python': [
                {
                    'title': 'Python - официальный сайт',
                    'url': 'https://www.python.org/',
                    'snippet': 'Python - это мощный, простой в изучении язык программирования с эффективными структурами данных.'
                },
                {
                    'title': 'Python для начинающих',
                    'url': 'https://pythonworld.ru/',
                    'snippet': 'Полный курс по программированию на Python. От основ до продвинутых тем с примерами.'
                }
            ],
            'машинное обучение': [
                {
                    'title': 'Машинное обучение - Википедия',
                    'url': 'https://ru.wikipedia.org/wiki/Машинное_обучение',
                    'snippet': 'Машинное обучение - класс методов искусственного интеллекта для обучения систем на данных.'
                },
                {
                    'title': 'Scikit-learn - библиотека ML',
                    'url': 'https://scikit-learn.org/',
                    'snippet': 'Простые и эффективные инструменты для анализа данных и машинного обучения на Python.'
                }
            ],
            'искусственный интеллект': [
                {
                    'title': 'Искусственный интеллект - Википедия',
                    'url': 'https://ru.wikipedia.org/wiki/Искусственный_интеллект',
                    'snippet': 'Искусственный интеллект - область информатики, изучающая создание интеллектуальных систем.'
                },
                {
                    'title': 'OpenAI - исследования ИИ',
                    'url': 'https://openai.com/',
                    'snippet': 'OpenAI - исследовательская компания в области искусственного интеллекта.'
                }
            ],
            'нейронные сети': [
                {
                    'title': 'Нейронные сети - основы',
                    'url': 'https://habr.com/ru/post/348450/',
                    'snippet': 'Введение в нейронные сети: архитектура, обучение, применение на практике.'
                },
                {
                    'title': 'TensorFlow - фреймворк',
                    'url': 'https://www.tensorflow.org/',
                    'snippet': 'TensorFlow - открытая платформа для машинного обучения от Google.'
                }
            ],
            'глубокое обучение': [
                {
                    'title': 'Глубокое обучение - Википедия',
                    'url': 'https://ru.wikipedia.org/wiki/Глубокое_обучение',
                    'snippet': 'Глубокое обучение - подмножество методов машинного обучения, основанное на искусственных нейронных сетях.'
                },
                {
                    'title': 'Keras - библиотека глубокого обучения',
                    'url': 'https://keras.io/',
                    'snippet': 'Keras - высокоуровневый API для глубокого обучения, работающий поверх TensorFlow.'
                }
            ]
        }
        
        # Поиск релевантной информации
        for key, info_list in knowledge_base.items():
            if key in query_lower:
                for info in info_list[:max_results]:
                    result = SearchResult(
                        title=info['title'],
                        url=info['url'],
                        snippet=info['snippet'],
                        source="local_knowledge",
                        relevance_score=0.85,
                        query=query
                    )
                    results.append(result)
                break
        
        # Если ничего не найдено, создаем общие результаты
        if not results:
            for i in range(min(max_results, 2)):
                result = SearchResult(
                    title=f"Информация о {query}",
                    url=f"https://example.com/search?q={quote(query)}",
                    snippet=f"Это базовая информация о {query}. Для получения актуальных данных рекомендуется использовать веб-поиск.",
                    source="local_fallback",
                    relevance_score=0.5,
                    query=query
                )
                results.append(result)
        
        return results
    
    def search_searx(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Searx (метапоисковик) с fallback."""
        try:
            logger.debug(f"Выполняется поиск Searx: {query}")
            
            # Параметры запроса
            params = {
                'q': query,
                'format': 'json',
                'engines': 'google,bing,duckduckgo',
                'language': 'ru'
            }
            
            # Выполняем запрос
            response = self.session.get("https://searx.be/search", params=params, timeout=10)
            response.raise_for_status()
            
            # Парсим JSON
            data = response.json()
            results = []
            
            # Обрабатываем результаты
            for item in data.get('results', [])[:max_results]:
                try:
                    title = item.get('title', '')
                    url = item.get('url', '')
                    snippet = item.get('content', '')
                    
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="searx",
                        relevance_score=self._calculate_relevance(title, snippet, query),
                        query=query
                    )
                    results.append(search_result)
                    
                except Exception as e:
                    logger.debug(f"Ошибка обработки результата: {e}")
                    continue
            
            logger.info(f"Найдено {len(results)} результатов через Searx")
            return results
            
        except Exception as e:
            logger.warning(f"Searx недоступен, используем fallback: {e}")
            # Fallback на DuckDuckGo
            return self.search_duckduckgo(query, max_results)
    
    def _calculate_relevance(self, title: str, snippet: str, query: str) -> float:
        """Рассчитывает релевантность результата."""
        if not title and not snippet:
            return 0.0
        
        score = 0.0
        query_terms = query.lower().split()
        
        # Проверяем вхождение терминов запроса
        combined_text = f"{title} {snippet}".lower()
        
        for term in query_terms:
            # Точное вхождение в заголовок
            if term in title.lower():
                score += 0.3
            # Точное вхождение в описание
            elif term in snippet.lower():
                score += 0.2
            # Частичное вхождение
            elif any(term in word for word in combined_text.split()):
                score += 0.1
        
        # Бонус за полноту
        matched_terms = sum(1 for term in query_terms if term in combined_text)
        if matched_terms == len(query_terms):
            score += 0.2
        elif matched_terms > len(query_terms) / 2:
            score += 0.1
        
        # Нормализуем
        return min(score, 1.0)