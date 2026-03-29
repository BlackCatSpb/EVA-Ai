"""Модуль расширения знаний для CogniFlex - добавление новых знаний в систему"""
import os
import logging
import time
import re
import json
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

from .knowledge_graph import KnowledgeGraph, KnowledgeNode
from .knowledge_analyzer import KnowledgeAnalyzer

logger = logging.getLogger("cogniflex.knowledge_expander")

class KnowledgeExpander:
    """Модуль расширения знаний для CogniFlex - добавление новых знаний в систему."""
    
    def __init__(self, brain=None, knowledge_graph=None, knowledge_analyzer=None):
        """
        Инициализирует расширитель знаний.
        
        Args:
            brain: Ссылка на ядро CogniFlex (опционально)
            knowledge_graph: Ссылка на граф знаний (опционально)
            knowledge_analyzer: Ссылка на анализатор знаний (опционально)
        """
        self.brain = brain
        # Создаем/принимаем KnowledgeGraph по актуальной сигнатуре
        if knowledge_graph is not None:
            self.knowledge_graph = knowledge_graph
        else:
            cache_dir = getattr(self.brain, "cache_dir", None)
            self.knowledge_graph = KnowledgeGraph(brain=self.brain, cache_dir=cache_dir)
        
        # Анализатор знаний принимает (knowledge_graph, brain)
        self.knowledge_analyzer = knowledge_analyzer or KnowledgeAnalyzer(self.knowledge_graph, brain=self.brain)
        
        # Инициализируем внутренние структуры
        self._init_domain_authority()
        
        logger.info("KnowledgeExpander инициализирован")
    
    def _init_domain_authority(self):
        """Инициализирует базовые значения авторитетности доменов."""
        # Авторитетность доменов по умолчанию
        self.domain_authority = {
            "science": 0.95,
            "medicine": 0.95,
            "technology": 0.9,
            "mathematics": 0.9,
            "engineering": 0.85,
            "economics": 0.8,
            "psychology": 0.75,
            "sociology": 0.7,
            "art": 0.65,
            "entertainment": 0.6,
            "general": 0.5
        }
    
    def get_domain_authority(self, domain: str) -> float:
        """
        Возвращает авторитетность домена.
        
        Args:
            domain: Домен знаний
            
        Returns:
            float: Авторитетность (0.0-1.0)
        """
        return self.domain_authority.get(domain, 0.5)
    
    def _update_domain_authority(self, domain: str, change: float):
        """
        Обновляет авторитетность домена.
        
        Args:
            domain: Домен знаний
            change: Изменение авторитетности
        """
        current_authority = self.get_domain_authority(domain)
        new_authority = current_authority + change
        new_authority = max(0.1, min(0.99, new_authority))
        self.domain_authority[domain] = new_authority
        
        logger.debug(f"Авторитетность домена '{domain}' обновлена: {current_authority:.2f} -> {new_authority:.2f}")
    
    def _determine_concept_domain(self, concept: str, description: str) -> str:
        """
        Определяет домен для концепта на основе его содержимого и связей.
        
        Args:
            concept: Концепт
            description: Описание концепта
            
        Returns:
            str: Определенный домен
        """
        # 1. Проверяем ключевые слова в описании
        domain_keywords = {
            "science": ["наука", "физика", "химия", "биология", "эксперимент", "теория", "гипотеза"],
            "medicine": ["медицина", "болезнь", "лечение", "симптом", "диагноз", "лекарство"],
            "technology": ["технология", "программа", "алгоритм", "компьютер", "инженерия", "робот"],
            "mathematics": ["математика", "число", "уравнение", "теорема", "функция", "алгебра"],
            "economics": ["экономика", "рынок", "финансы", "бизнес", "инвестиция", "валюта"],
            "psychology": ["психология", "мышление", "эмоция", "поведение", "сознание", "личность"],
            "sociology": ["социология", "общество", "культура", "социальный", "группа", "институт"],
            "art": ["искусство", "музыка", "живопись", "литература", "театр", "кино"],
            "entertainment": ["развлечение", "игра", "фильм", "сериал", "музыка", "шоу"]
        }
        
        # Считаем совпадения
        description_lower = description.lower()
        domain_scores = defaultdict(float)
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    domain_scores[domain] += 1.0
        
        # 2. Проверяем существующие связи
        existing_nodes = self.knowledge_graph.search_nodes(concept, limit=5)
        for node in existing_nodes:
            if node.domain in self.domain_authority:
                domain_scores[node.domain] += 2.0  # Больший вес для существующих связей
        
        # 3. Если доступен ML, используем его для определения домена
        if self.brain and hasattr(self.brain, 'text_processor'):
            try:
                ml_domain = self.brain.text_processor.classify_domain(concept, description)
                if ml_domain in self.domain_authority:
                    domain_scores[ml_domain] += 3.0
            except Exception as e:
                logger.warning(f"Ошибка классификации домена с помощью ML: {e}")
        
        # 4. Определяем домен с наибольшим счетом
        if domain_scores:
            determined_domain = max(domain_scores, key=domain_scores.get)
            # Если максимальный счет низкий, используем общий домен
            if domain_scores[determined_domain] < 2.0:
                return "general"
            return determined_domain
        
        # 5. Если не удалось определить, используем анализ связей
        if existing_nodes:
            related_nodes = self.knowledge_graph.get_related_nodes(existing_nodes[0].id)
            if related_nodes:
                related_domains = [node.domain for node in related_nodes]
                most_common_domain = Counter(related_domains).most_common(1)[0][0]
                return most_common_domain
        
        # 6. По умолчанию - общий домен
        return "general"
    
    def expand_knowledge(self, concept: str, depth: int = 1, 
                        num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Расширяет знания по указанному концепту.
        
        Args:
            concept: Концепт для расширения
            depth: Глубина расширения
            num_results: Количество результатов
            
        Returns:
            List[Dict[str, Any]]: Расширенные знания
        """
        try:
            logger.info(f"Расширение знаний по концепту '{concept}' (глубина: {depth}, результатов: {num_results})")
            
            # Проверяем, доступен ли WebSearchEngine
            if not self.brain or not hasattr(self.brain, 'web_search_engine'):
                logger.warning("WebSearchEngine недоступен для расширения знаний")
                return []
            
            # Выполняем веб-поиск
            knowledge = self.brain.web_search_engine.web_search_and_learn(
                concept, 
                num_results=num_results
            )
            
            # Если нет результатов, пытаемся использовать MLUnit
            if not knowledge and hasattr(self.brain, 'response_generator'):
                logger.info("Использование response_generator для генерации знаний")
                knowledge = self._generate_knowledge_with_ml(concept, depth)
            
            # Интегрируем новые знания
            if knowledge:
                self._integrate_knowledge(knowledge, concept, depth)
            
            logger.info(f"Расширение знаний завершено. Добавлено {len(knowledge)} новых знаний")
            return knowledge
            
        except Exception as e:
            logger.error(f"Ошибка расширения знаний по концепту '{concept}': {e}")
            
            # Уведомляем SelfAnalyzer об ошибке
            if self.brain and hasattr(self.brain, 'self_analyzer'):
                self.brain.self_analyzer.add_learning_opportunity(
                    concept=f"knowledge_expansion_{concept}",
                    opportunity_type="expansion",
                    priority=0.9,
                    domain="system",
                    evidence=[f"Ошибка расширения знаний: {str(e)}"],
                    suggested_actions=[
                        "Проверить подключение к интернету",
                        "Обновить API ключи для поисковых сервисов"
                    ]
                )
            
            return []
    
    def _generate_knowledge_with_ml(self, concept: str, depth: int) -> List[Dict[str, Any]]:
        """Генерирует знания с помощью MLUnit."""
        try:
            if not self.brain or not hasattr(self.brain, 'response_generator'):
                return []
            
            # Генерируем граф знаний
            graph = self.brain.response_generator.generate_knowledge_graph(
                concept,
                depth=depth
            )
            
            # Преобразуем в формат для интеграции
            knowledge = []
            for node in graph["nodes"]:
                # Определяем домен для каждого узла
                domain = self._determine_concept_domain(node["label"], node.get("description", ""))
                
                knowledge.append({
                    "concept": node["label"],
                    "content": node.get("description", ""),
                    "domain": domain,
                    "source": "ml_generated",
                    "relevance": node.get("strength", 0.7)
                })
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Ошибка генерации знаний с помощью ML: {e}")
            return []
    
    def _integrate_knowledge(self, knowledge: List[Dict[str, Any]], 
                            base_concept: str, depth: int):
        """Интегрирует новые знания в граф знаний."""
        for item in knowledge:
            # Определяем домен, если он не указан
            domain = item.get("domain", "general")
            if not domain or domain == "general":
                domain = self._determine_concept_domain(item["concept"], item["content"])
            
            # Добавляем концепт
            concept_id = self.knowledge_graph.add_concept(
                item["concept"],
                item["content"],
                domain=domain,
                strength=item.get("relevance", 0.7),
                source=item.get("source", "unknown")
            )
            
            # Добавляем связь с базовым концептом
            if depth > 0:
                # Ищем базовый концепт
                base_nodes = self.knowledge_graph.search_nodes(base_concept, limit=1)
                if base_nodes:
                    # Определяем тип связи
                    relation_type = self._determine_relation_type(base_concept, item["concept"])
                    
                    self.knowledge_graph.add_connection(
                        base_nodes[0].id,
                        concept_id,
                        relation_type,
                        strength=item.get("relevance", 0.7)
                    )
    
    def _determine_relation_type(self, concept1: str, concept2: str) -> str:
        """
        Определяет тип связи между концептами.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            
        Returns:
            str: Тип связи
        """
        # Если доступен ML, используем его для определения типа связи
        if self.brain and hasattr(self.brain, 'text_processor'):
            try:
                return self.brain.text_processor.determine_relation_type(concept1, concept2)
            except Exception as e:
                logger.warning(f"Ошибка определения типа связи с помощью ML: {e}")
        
        # Анализируем ключевые слова
        common_relations = {
            "cause": ["причина", "вызывает", "приводит к", "является причиной"],
            "effect": ["следствие", "результат", "является результатом", "вызывает"],
            "part_of": ["часть", "составная часть", "входит в"],
            "is_a": ["является", "тип", "разновидность", "подкатегория"],
            "related_to": ["связан с", "относится к", "имеет отношение к"],
            "opposite_of": ["противоположность", "противоположный", "антоним"]
        }
        
        # Проверяем описания концептов
        node1 = self.knowledge_graph.search_nodes(concept1, limit=1)
        node2 = self.knowledge_graph.search_nodes(concept2, limit=1)
        
        if node1 and node2:
            meta1 = getattr(node1[0], 'meta', None) or {}
            meta2 = getattr(node2[0], 'meta', None) or {}
            desc1 = meta1.get("description", "").lower()
            desc2 = meta2.get("description", "").lower()
            
            for relation, keywords in common_relations.items():
                for keyword in keywords:
                    if keyword in desc1 or keyword in desc2:
                        return relation
        
        # Проверяем существующие связи
        if node1:
            for edge in self.knowledge_graph.get_edges(node1[0].id):
                if node2 and edge.target_id == node2[0].id:
                    return edge.relation_type
        
        # По умолчанию - общая связь
        return "related_to"
    
    def update_outdated_knowledge(self, domain: Optional[str] = None, 
                                max_age_days: int = 365) -> int:
        """
        Обновляет устаревшие знания.
        
        Args:
            domain: Область знаний (опционально)
            max_age_days: Максимальный возраст знаний в днях
            
        Returns:
            int: Количество обновленных знаний
        """
        try:
            logger.info(f"Обновление устаревших знаний (домен: {domain or 'все'}, возраст: {max_age_days} дней)")
            
            # Получаем устаревшие узлы
            outdated_nodes = self._get_outdated_nodes(domain, max_age_days)
            
            updated_count = 0
            for node in outdated_nodes:
                # Пытаемся обновить информацию
                updated = self._update_node_from_web(node)
                if updated:
                    updated_count += 1
            
            logger.info(f"Обновление устаревших знаний завершено. Обновлено {updated_count} новых знаний")
            return updated_count
            
        except Exception as e:
            logger.error(f"Ошибка обновления устаревших знаний: {e}")
            
            # Уведомляем SelfAnalyzer об ошибке
            if self.brain and hasattr(self.brain, 'self_analyzer'):
                self.brain.self_analyzer.add_learning_opportunity(
                    concept="knowledge_update",
                    opportunity_type="updating",
                    priority=0.9,
                    domain="system",
                    evidence=[f"Ошибка обновления устаревших знаний: {str(e)}"],
                    suggested_actions=[
                        "Проверить подключение к интернету",
                        "Увеличить интервал обновления"
                    ]
                )
            
            return 0
    
    def _get_outdated_nodes(self, domain: Optional[str] = None, 
                          max_age_days: int = 365) -> List[KnowledgeNode]:
        """Получает устаревшие узлы."""
        try:
            conn = sqlite3.connect(self.knowledge_graph.db_path)
            cursor = conn.cursor()
            
            # Формируем запрос
            sql = "SELECT * FROM nodes WHERE last_updated < ?"
            params = [time.time() - max_age_days * 86400]
            
            if domain:
                sql += " AND domain = ?"
                params.append(domain)
            
            # Выполняем запрос
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            nodes = []
            for row in rows:
                node = KnowledgeNode(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    node_type=row[3],
                    domain=row[4],
                    strength=row[5],
                    timestamp=row[6],
                    meta=json.loads(row[7]) if row[7] else {},
                    version=row[8] if len(row) > 8 else 1
                )
                nodes.append(node)
            
            return nodes
        
        except Exception as e:
            logger.error(f"Ошибка получения устаревших узлов: {e}")
            return []
        finally:
            try:
                if 'conn' in locals():
                    conn.close()
            except Exception:
                pass
    
    def _update_node_from_web(self, node: KnowledgeNode) -> bool:
        """Обновляет узел с помощью веб-поиска."""
        try:
            # Проверяем, доступен ли WebSearchEngine
            if not self.brain or not hasattr(self.brain, 'web_search_engine'):
                return False
            
            # Выполняем веб-поиск
            knowledge = self.brain.web_search_engine.web_search_and_learn(
                node.name, 
                num_results=1
            )
            
            if not knowledge:
                return False
            
            # Обновляем узел
            return self.knowledge_graph.update_concept(
                node.id,
                new_description=knowledge[0]["content"],
                source=knowledge[0]["source"]
            )
            
        except Exception as e:
            logger.error(f"Ошибка обновления узла {node.id} из веба: {e}")
            return False
    
    def generate_explanation(self, concept: str, level: float = 0.5) -> str:
        """
        Генерирует объяснение концепта с адаптированной детализацией.
        
        Args:
            concept: Концепт для объяснения
            level: Уровень детализации (0.0-1.0)
            
        Returns:
            str: Сгенерированное объяснение
        """
        try:
            logger.info(f"Генерация объяснения для концепта '{concept}' (уровень детализации: {level})")
            
            # Получаем детали концепта
            nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if not nodes:
                # Если концепт не найден, пытаемся расширить знания
                expanded = self.expand_knowledge(concept, depth=1)
                if not expanded:
                    return f"К сожалению, у меня нет информации о концепте '{concept}'."
                nodes = self.knowledge_graph.search_nodes(concept, limit=1)
                if not nodes:
                    return f"К сожалению, у меня нет информации о концепте '{concept}'."
            
            node = nodes[0]
            concept_details = self.knowledge_graph.get_node_details(node.id)
            
            # Формируем базовый ответ
            node_desc = node.meta.get('description', '') if hasattr(node, 'meta') else ''
            response = f"{concept}: {node_desc or 'Описание отсутствует'}\n\n"
            
            # Добавляем информацию в зависимости от уровня детализации
            if level < 0.3:  # Низкая детализация
                # Упрощаем ответ - оставляем только основные предложения
                sentences = re.split(r'[.!?]+', response)
                if len(sentences) > 2:
                    # Берем первое и последнее предложение
                    response = sentences[0].strip() + ". " + sentences[-1].strip() + "."
                else:
                    response = sentences[0].strip() + "."
            
            elif level > 0.7:  # Высокая детализация
                # Добавляем больше деталей
                if concept_details["connections"]:
                    response += "Этот концепт связан с:\n"
                    for i, conn in enumerate(concept_details["connections"][:5], 1):
                        response += f"{i}. {conn['target_content']} - {conn['relation']}\n"
                
                # Добавляем примеры, если доступно
                if hasattr(self.brain, 'response_generator'):
                    examples = self.brain.response_generator.generate_examples(
                        concept, 
                        num_examples=2
                    )
                    if examples:
                        response += "\nПримеры:\n"
                        for i, example in enumerate(examples, 1):
                            response += f"{i}. {example}\n"
            
            else:  # Средняя детализация
                # Добавляем основные связи
                if concept_details["connections"]:
                    response += "Основные связи:\n"
                    for i, conn in enumerate(concept_details["connections"][:3], 1):
                        response += f"- {conn['relation']}: {conn['target_content']}\n"
            
            logger.info(f"Объяснение для концепта '{concept}' сгенерировано")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации объяснения для концепта '{concept}': {e}")
            return f"Произошла ошибка при генерации объяснения для концепта '{concept}'."
    
    def generate_comparison(self, concept1: str, concept2: str) -> str:
        """
        Генерирует сравнение двух концептов.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            
        Returns:
            str: Сгенерированное сравнение
        """
        try:
            logger.info(f"Генерация сравнения между '{concept1}' и '{concept2}'")
            
            # Получаем детали концептов
            nodes1 = self.knowledge_graph.search_nodes(concept1, limit=1)
            nodes2 = self.knowledge_graph.search_nodes(concept2, limit=1)
            
            if not nodes1 or not nodes2:
                missing = []
                if not nodes1: missing.append(concept1)
                if not nodes2: missing.append(concept2)
                return f"Не могу сравнить, так как не найдена информация о {', '.join(missing)}."
            
            details1 = self.knowledge_graph.get_node_details(nodes1[0].id)
            details2 = self.knowledge_graph.get_node_details(nodes2[0].id)
            
            # Анализируем сходства и различия
            similarities = []
            differences = []
            
            # Сравниваем домены
            if details1["node"]["domain"] == details2["node"]["domain"]:
                similarities.append(f"Оба концепта относятся к домену '{details1['node']['domain']}'")
            else:
                differences.append(f"Концепты относятся к разным доменам: '{details1['node']['domain']}' и '{details2['node']['domain']}'")
            
            # Сравниваем связи
            relations1 = {conn["relation"] for conn in details1["connections"]}
            relations2 = {conn["relation"] for conn in details2["connections"]}
            
            common_relations = relations1 & relations2
            unique_relations1 = relations1 - relations2
            unique_relations2 = relations2 - relations1
            
            if common_relations:
                similarities.append(f"Общие типы связей: {', '.join(list(common_relations)[:3])}")
            
            if unique_relations1:
                differences.append(f"'{concept1}' имеет уникальные связи: {', '.join(list(unique_relations1)[:2])}")
            
            if unique_relations2:
                differences.append(f"'{concept2}' имеет уникальные связи: {', '.join(list(unique_relations2)[:2])}")
            
            # Формируем ответ
            response = f"Сравнение '{concept1}' и '{concept2}':\n\n"
            
            if similarities:
                response += "Сходства:\n"
                for i, sim in enumerate(similarities, 1):
                    response += f"{i}. {sim}\n"
                response += "\n"
            
            if differences:
                response += "Различия:\n"
                for i, diff in enumerate(differences, 1):
                    response += f"{i}. {diff}\n"
                response += "\n"
            
            # Добавляем обобщение
            if len(similarities) > len(differences):
                response += f"В целом, '{concept1}' и '{concept2}' имеют больше общего, чем различий."
            elif len(differences) > len(similarities):
                response += f"В целом, '{concept1}' и '{concept2}' значительно различаются."
            else:
                response += f"В целом, '{concept1}' и '{concept2}' имеют сбалансированное соотношение сходств и различий."
            
            logger.info(f"Сравнение между '{concept1}' и '{concept2}' сгенерировано")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации сравнения между '{concept1}' и '{concept2}': {e}")
            return f"Произошла ошибка при сравнении концептов '{concept1}' и '{concept2}'."
    
    def verify_knowledge(self, concept: str) -> Dict[str, Any]:
        """
        Проверяет надежность и актуальность знаний по концепту.
        
        Args:
            concept: Концепт для проверки
            
        Returns:
            Dict[str, Any]: Результаты проверки
        """
        try:
            logger.info(f"Проверка знаний по концепту '{concept}'")
            
            # Получаем узел
            nodes = self.knowledge_graph.search_nodes(concept, limit=1)
            if not nodes:
                return {
                    "concept": concept,
                    "verified": False,
                    "reliability_score": 0.0,
                    "freshness_score": 0.0,
                    "issues": [f"Концепт '{concept}' не найден в базе знаний"],
                    "timestamp": time.time()
                }
            
            node = nodes[0]
            
            # Анализируем актуальность
            current_time = time.time()
            age_days = (current_time - node.last_updated) / 86400
            freshness_score = max(0.0, 1.0 - (age_days / 365))
            
            # Анализируем авторитетность домена
            domain_authority = self.get_domain_authority(node.domain)
            
            # Общий показатель надежности
            reliability_score = (
                freshness_score * 0.4 +
                domain_authority * 0.3
            )
            
            # Формируем отчет
            report = {
                "concept": concept,
                "reliability_score": reliability_score,
                "freshness_score": freshness_score,
                "domain_authority": domain_authority,
                "age_days": age_days,
                "issues": [],
                "timestamp": time.time()
            }
            
            # Добавляем проблемы
            if reliability_score < 0.4:
                report["issues"].append("Низкая надежность информации")
            if freshness_score < 0.3:
                report["issues"].append("Устаревшая информация (более года)")
            
            # Если есть проблемы, предлагаем действия
            if report["issues"]:
                report["suggested_actions"] = [
                    "Обновить информацию из авторитетных источников",
                    "Усилить связи с другими концептами"
                ]
            
            logger.info(f"Проверка знаний по концепту '{concept}' завершена")
            return report
            
        except Exception as e:
            logger.error(f"Ошибка проверки знаний по концепту '{concept}': {e}")
            return {
                "concept": concept,
                "verified": False,
                "error": str(e),
                "timestamp": time.time()
            }