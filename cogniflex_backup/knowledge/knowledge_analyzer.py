"""Модуль анализа графа знаний для CogniFlex - полная реализация"""
import logging
import time
import re
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import networkx.algorithms.community as community_louvain
COMMUNITY_DETECTION_AVAILABLE = True


logger = logging.getLogger("cogniflex.knowledge.analyzer")

class KnowledgeAnalyzer:
    """Анализирует граф знаний для выявления пробелов, кластеров и паттернов."""
    
    def __init__(self, knowledge_graph, brain=None):
        """
        Инициализирует анализатор графа знаний.
        
        Args:
            knowledge_graph: Ссылка на граф знаний
            brain: Ссылка на ядро системы (опционально)
        """
        self.knowledge_graph = knowledge_graph
        self.brain = brain
        logger.info("KnowledgeAnalyzer инициализирован")
    
    def analyze_knowledge_gaps(self, domain: Optional[str] = None, 
                             num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Анализирует пробелы в знаниях в указанной области.
        
        Args:
            domain: Область знаний (опционально)
            num_samples: Количество примеров для анализа
            
        Returns:
            List[Dict[str, Any]]: Выявленные пробелы в знаниях
        """
        try:
            logger.info(f"Анализ пробелов в знаниях (домен: {domain or 'все'}, количество: {num_samples})")
            
            # Получаем узлы из области знаний
            nodes = self.knowledge_graph.get_all_nodes()
            if domain:
                nodes = [node for node in nodes if node.domain == domain]
            
            gaps = []
            for node in nodes[:num_samples]:
                # Анализируем связи узла
                edges = self.knowledge_graph.get_edges(node.id)
                
                # Проверяем, достаточно ли связей
                if len(edges) < 3:  # Пороговое значение для "пробела"
                    gap = {
                        "node_id": node.id,
                        "concept": node.name,
                        "domain": node.domain,
                        "current_connections": len(edges),
                        "suggested_connections": 5,  # Целевое количество связей
                        "priority": min(0.9, 0.3 + (5 - len(edges)) * 0.2),
                        "evidence": [
                            f"Концепт '{node.name}' имеет только {len(edges)} связей, что ниже порога в 3 связи"
                        ]
                    }
                    gaps.append(gap)
                
                # Проверяем актуальность
                time_diff = time.time() - node.last_updated
                if time_diff > 365 * 86400:  # Больше года
                    gap = {
                        "node_id": node.id,
                        "concept": node.name,
                        "domain": node.domain,
                        "gap_type": "outdated",
                        "severity": 0.6,
                        "evidence": [
                            f"Информация о концепте '{node.name}' устарела"
                        ],
                        "suggested_actions": [
                            f"Обновить информацию о концепте '{node.name}'",
                            "Проверить актуальные источники"
                        ]
                    }
                    gaps.append(gap)
            
            logger.info(f"Обнаружено {len(gaps)} пробелов в знаниях")
            return gaps
            
        except Exception as e:
            logger.error(f"Ошибка анализа пробелов в знаниях: {e}", exc_info=True)
            return []
    
    def analyze_knowledge_coverage(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализирует покрытие знаний в указанной области.
        
        Args:
            domain: Область знаний для анализа (опционально)
            
        Returns:
            Dict[str, Any]: Результаты анализа покрытия
        """
        try:
            logger.info(f"Анализ покрытия знаний (домен: {domain or 'все'})")
            
            # Получаем статистику по доменам
            stats = self.knowledge_graph.get_statistics()
            domain_stats = stats["by_domain"]
            
            # Определяем целевые домены
            target_domains = [domain] if domain else list(domain_stats.keys())
            
            coverage_report = {
                "domains": {},
                "overall_coverage": 0.0,
                "coverage_trend": "stable",
                "recommendations": [],
                "timestamp": time.time()
            }
            
            total_nodes = stats["total_nodes"]
            if total_nodes == 0:
                return coverage_report
            
            # Анализируем каждый домен
            for dom in target_domains:
                if dom not in domain_stats:
                    continue
                    
                # Получаем информацию о домене
                domain_info = {
                    "node_count": domain_stats[dom],
                    "coverage_score": min(1.0, domain_stats[dom] / (total_nodes * 0.2)),  # Нормализация
                    "trend": "stable"
                }
                
                # Сохраняем в отчет
                coverage_report["domains"][dom] = domain_info
            
            # Вычисляем общий показатель
            if coverage_report["domains"]:
                coverage_report["overall_coverage"] = sum(
                    info["coverage_score"] for info in coverage_report["domains"].values()
                ) / len(coverage_report["domains"])
            
            # Определяем тренд
            if coverage_report["overall_coverage"] < 0.3:
                coverage_report["coverage_trend"] = "critical"
                coverage_report["recommendations"].append(
                    "Критически низкое покрытие знаний. Рекомендуется расширить базу знаний."
                )
            elif coverage_report["overall_coverage"] < 0.6:
                coverage_report["coverage_trend"] = "warning"
                coverage_report["recommendations"].append(
                    "Низкое покрытие знаний. Рассмотрите добавление новых источников информации."
                )
            
            logger.info(f"Анализ покрытия знаний завершен. Общий показатель: {coverage_report['overall_coverage']:.2f}")
            return coverage_report
            
        except Exception as e:
            logger.error(f"Ошибка анализа покрытия знаний: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def analyze_knowledge_clusters(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализирует кластеры знаний в графе.
        
        Args:
            domain: Область знаний для анализа (опционально)
            
        Returns:
            Dict[str, Any]: Результаты анализа кластеров
        """
        try:
            logger.info(f"Анализ кластеров знаний (домен: {domain or 'все'})")
            
            # Получаем узлы
            nodes = self.knowledge_graph.get_all_nodes()
            if domain:
                nodes = [node for node in nodes if node.domain == domain]
            
            # Если недостаточно узлов для анализа
            if len(nodes) < 2:
                logger.warning("Недостаточно узлов для анализа кластеров")
                return {
                    "status": "insufficient_data",
                    "message": "Недостаточно узлов для анализа",
                    "patterns": [],
                    "timestamp": time.time()
                }
            
            # Пытаемся создать граф с помощью NetworkX
            try:
                G = self._build_networkx_graph(domain=domain)
            except Exception as e:
                logger.error(f"Ошибка построения графа NetworkX: {e}")
                return {
                    "status": "graph_error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
            # Анализируем кластеры
            clusters = self._identify_knowledge_clusters(G)
            
            # Анализируем каждый кластер
            analysis = {
                "total_clusters": len(clusters),
                "clusters": [],
                "patterns": [],
                "recommendations": [],
                "timestamp": time.time()
            }
            
            for cluster in clusters:
                # Определяем тему кластера
                topic = self._determine_cluster_topic(cluster)
                
                # Оцениваем качество кластера
                quality = self._evaluate_cluster_quality(cluster)
                
                # Формируем анализ
                cluster_analysis = {
                    "id": cluster["id"],
                    "size": len(cluster["nodes"]),
                    "topic": topic,
                    "quality_score": quality["score"],
                    "quality_metrics": quality["metrics"],
                    "key_concepts": quality["key_concepts"][:5],
                    "potential_gaps": quality["potential_gaps"]
                }
                
                analysis["clusters"].append(cluster_analysis)
                
                # Формируем рекомендации
                if quality["score"] < 0.6:
                    analysis["recommendations"].append({
                        "cluster_id": cluster["id"],
                        "topic": topic,
                        "issue": "Низкое качество кластера",
                        "suggestion": "Рассмотрите добавление связей между концептами или включение дополнительных концептов"
                    })
            
            # Поиск паттернов
            patterns = self._identify_knowledge_patterns(clusters)
            analysis["patterns"] = patterns
            
            logger.info(f"Анализ кластеров завершен: {len(clusters)} кластеров обнаружено")
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа кластеров знаний: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _build_networkx_graph(self, domain: Optional[str] = None) -> nx.Graph:
        """
        Строит граф NetworkX из графа знаний.
        
        Args:
            domain: Область знаний для фильтрации (опционально)
            
        Returns:
            nx.Graph: Граф для анализа
        """
        G = nx.Graph()
        
        # Добавляем узлы
        for node in self.knowledge_graph.get_all_nodes():
            if domain and node.domain != domain:
                continue
            G.add_node(node.id, name=node.name, domain=node.domain)
        
        # Добавляем связи
        for edge in self.knowledge_graph.get_all_edges():
            if edge.source_id in G.nodes and edge.target_id in G.nodes:
                G.add_edge(
                    edge.source_id, 
                    edge.target_id, 
                    relation=edge.relation_type,
                    strength=edge.strength
                )
        
        return G
    
    def _identify_knowledge_clusters(self, G: nx.Graph) -> List[Dict[str, Any]]:
        """
        Идентифицирует кластеры знаний.
        
        Args:
            G: Граф NetworkX для анализа
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных кластеров
        """
        clusters = []
        
        try:
            # Проверяем, доступен ли community_louvain
            if not COMMUNITY_DETECTION_AVAILABLE:
                logger.warning("community_louvain не установлен. Используем упрощенный алгоритм кластеризации.")
                return self._simple_cluster_identification(G)
            
            # Проверяем размер графа
            if len(G) <= 1:
                logger.warning("Недостаточно узлов для обнаружения сообществ (требуется >1 узел)")
                return []
            
            # Находим сообщества с помощью алгоритма Лувена
            partition = community_louvain.best_partition(G)
            
            # Группируем узлы по кластерам
            cluster_dict = {}
            for node, cluster_id in partition.items():
                if cluster_id not in cluster_dict:
                    cluster_dict[cluster_id] = []
                cluster_dict[cluster_id].append(node)
            
            # Формируем список кластеров
            for cluster_id, nodes in cluster_dict.items():
                if len(nodes) > 1:  # Игнорируем кластеры с одним узлом
                    clusters.append({
                        "id": f"cluster_{cluster_id}",
                        "nodes": nodes,
                        "size": len(nodes)
                    })
            
            logger.debug(f"Обнаружено {len(clusters)} кластеров знаний")
            return clusters
            
        except Exception as e:
            logger.error(f"Ошибка идентификации кластеров: {e}", exc_info=True)
            # В случае ошибки используем упрощенный алгоритм
            return self._simple_cluster_identification(G)
    
    def _simple_cluster_identification(self, G: nx.Graph) -> List[Dict[str, Any]]:
        """
        Упрощенная идентификация кластеров (используется, если community_louvain недоступен).
        
        Args:
            G: Граф NetworkX для анализа
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных кластеров
        """
        clusters = []
        visited = set()
        
        # Проходим по всем узлам и формируем кластеры на основе связности
        for node in G.nodes():
            if node in visited:
                continue
                
            # Начинаем обход в ширину для поиска связного компонента
            cluster_nodes = []
            queue = [node]
            visited.add(node)
            
            while queue:
                current = queue.pop(0)
                cluster_nodes.append(current)
                
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Добавляем кластер, если он содержит более одного узла
            if len(cluster_nodes) > 1:
                clusters.append({
                    "id": f"cluster_{len(clusters)}",
                    "nodes": cluster_nodes,
                    "size": len(cluster_nodes)
                })
        
        logger.debug(f"Упрощенный алгоритм обнаружил {len(clusters)} кластеров знаний")
        return clusters
    
    def _determine_cluster_topic(self, cluster: Dict[str, Any]) -> str:
        """
        Определяет тему кластера на основе его узлов.
        
        Args:
            cluster: Кластер для анализа
            
        Returns:
            str: Определенная тема кластера
        """
        try:
            # Если кластер по домену, используем название домена
            if cluster["id"].startswith("domain_"):
                return cluster["id"][7:]
            
            # Анализируем содержимое узлов
            contents = [self.knowledge_graph.get_node(node_id).name for node_id in cluster["nodes"]]
            
            # Извлекаем ключевые слова
            all_text = " ".join(contents)
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            # Удаляем стоп-слова
            stop_words = {'и', 'в', 'на', 'с', 'к', 'от', 'по', 'для', 'о', 'об', 'про',
                          'и', 'в', 'во', 'не', 'что', 'он', 'на', 'с', 'со', 'как', 'а', 'то', 'все',
                          'этот', 'который', 'его', 'так', 'его', 'ее', 'их', 'быть', 'был', 'была', 
                          'были', 'будет', 'может', 'также', 'которые', 'который', 'которая'}
            words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Подсчитываем частоту слов
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Получаем самые частые слова
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            common_words = [word for word, count in sorted_words[:3]]
            
            return " ".join(common_words) if common_words else "неизвестная тема"
            
        except Exception as e:
            logger.error(f"Ошибка определения темы кластера: {e}", exc_info=True)
            return "неизвестная тема"
    
    def _evaluate_cluster_quality(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оценивает качество кластера.
        
        Args:
            cluster: Кластер для оценки
            
        Returns:
            Dict[str, Any]: Оценка качества кластера
        """
        try:
            # Проверяем плотность связей внутри кластера
            internal_edges = 0
            total_possible_edges = len(cluster["nodes"]) * (len(cluster["nodes"]) - 1) / 2
            
            for node_id in cluster["nodes"]:
                node = self.knowledge_graph.get_node(node_id)
                if not node:
                    continue
                    
                edges = self.knowledge_graph.get_edges(node_id)
                for edge in edges:
                    # Проверяем, что целевой узел тоже в кластере
                    if edge.target_id in cluster["nodes"]:
                        internal_edges += 1
            
            density = internal_edges / max(1, total_possible_edges)
            
            # Проверяем среднюю силу связей
            total_strength = 0
            strength_count = 0
            for node_id in cluster["nodes"]:
                node = self.knowledge_graph.get_node(node_id)
                if not node:
                    continue
                
                edges = self.knowledge_graph.get_edges(node_id)
                for edge in edges:
                    if edge.target_id in cluster["nodes"]:
                        total_strength += edge.strength
                        strength_count += 1
            
            avg_strength = total_strength / strength_count if strength_count > 0 else 0
            
            # Проверяем разнообразие доменов
            domains = set()
            for node_id in cluster["nodes"]:
                node = self.knowledge_graph.get_node(node_id)
                if node:
                    domains.add(node.domain)
            
            domain_diversity = len(domains) / max(1, len(cluster["nodes"]))
            
            # Рассчитываем общий показатель качества
            # Высокая плотность, низкая диверсивность доменов и высокая сила связей = хорошее качество
            quality = (density * 0.4) + ((1 - domain_diversity) * 0.3) + (avg_strength * 0.3)
            
            # Определяем ключевые концепты
            key_concepts = []
            for node_id in cluster["nodes"]:
                node = self.knowledge_graph.get_node(node_id)
                if node:
                    key_concepts.append(node.name)
            
            # Определяем потенциальные пробелы
            potential_gaps = []
            if density < 0.3:
                potential_gaps.append("Низкая плотность связей внутри кластера")
            if domain_diversity > 0.7:
                potential_gaps.append("Высокое разнообразие доменов в кластере")
            if avg_strength < 0.4:
                potential_gaps.append("Низкая сила связей внутри кластера")
            
            return {
                "score": min(1.0, max(0.0, quality)),
                "metrics": {
                    "density": density,
                    "domain_diversity": domain_diversity,
                    "avg_strength": avg_strength
                },
                "key_concepts": key_concepts,
                "potential_gaps": potential_gaps
            }
            
        except Exception as e:
            logger.error(f"Ошибка оценки качества кластера: {e}", exc_info=True)
            return {
                "score": 0.3,
                "metrics": {
                    "density": 0.0,
                    "domain_diversity": 1.0,
                    "avg_strength": 0.0
                },
                "key_concepts": [],
                "potential_gaps": ["Ошибка анализа кластера"]
            }
    
    def _identify_knowledge_patterns(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Идентифицирует паттерны в знаниях.
        
        Args:
            clusters: Список кластеров для анализа
            
        Returns:
            List[Dict[str, Any]]: Выявленные паттерны
        """
        patterns = []
        
        try:
            # Анализ распределения размеров кластеров
            sizes = [cluster["size"] for cluster in clusters]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                large_clusters = [c for c in clusters if c["size"] > avg_size * 1.5]
                
                if large_clusters:
                    patterns.append({
                        "type": "large_clusters",
                        "description": f"Обнаружено {len(large_clusters)} крупных кластеров (размер > {avg_size * 1.5:.1f})",
                        "clusters": [c["id"] for c in large_clusters],
                        "priority": 0.7
                    })
            
            # Анализ маленьких кластеров (возможные пробелы)
            small_clusters = [c for c in clusters if c["size"] < 3]
            if len(small_clusters) > 5:
                patterns.append({
                    "type": "small_clusters",
                    "description": f"Обнаружено {len(small_clusters)} маленьких кластеров (размер < 3)",
                    "clusters": [c["id"] for c in small_clusters],
                    "priority": 0.8,
                    "suggestions": [
                        "Рассмотрите объединение маленьких кластеров",
                        "Исследуйте возможность добавления связей между кластерами"
                    ]
                })
            
            logger.debug(f"Идентифицировано {len(patterns)} паттернов знаний")
            return patterns
            
        except Exception as e:
            logger.error(f"Ошибка идентификации паттернов знаний: {e}", exc_info=True)
            return []
    
    def analyze_knowledge_evolution(self, time_window: str = "month") -> Dict[str, Any]:
        """
        Анализирует эволюцию знаний за указанный период.
        
        Args:
            time_window: Период анализа (day, week, month, year)
            
        Returns:
            Dict[str, Any]: Отчет об эволюции знаний
        """
        try:
            logger.info(f"Анализ эволюции знаний (период: {time_window})")
            
            # Определяем временные рамки
            now = time.time()
            if time_window == "day":
                start_time = now - 86400  # 24 часа
            elif time_window == "week":
                start_time = now - 604800  # 7 дней
            elif time_window == "year":
                start_time = now - 31536000  # 1 год
            else:  # month по умолчанию
                start_time = now - 2592000  # 30 дней
            
            # Получаем узлы, добавленные за период
            new_nodes = []
            for node in self.knowledge_graph.get_all_nodes():
                if hasattr(node, 'timestamp') and node.timestamp >= start_time:
                    new_nodes.append(node)
            
            # Получаем узлы, обновленные за период
            updated_nodes = []
            for node in self.knowledge_graph.get_all_nodes():
                if hasattr(node, 'last_updated') and node.last_updated >= start_time:
                    updated_nodes.append(node)
            
            # Анализируем рост
            growth_rate = len(new_nodes) / max(1, len(self.knowledge_graph.get_all_nodes()) - len(new_nodes))
            
            # Анализируем темпы роста
            trends = {
                "growth": {
                    "new_nodes": len(new_nodes),
                    "updated_nodes": len(updated_nodes),
                    "total_nodes": len(self.knowledge_graph.get_all_nodes()),
                    "growth_rate": growth_rate
                }
            }
            
            # Определяем, ускоряется ли рост
            if growth_rate > 0.1:
                trends["growth"]["trend"] = "increasing"
                trends["growth"]["analysis"] = "Знания активно пополняются"
            elif growth_rate > 0.01:
                trends["growth"]["trend"] = "stable"
                trends["growth"]["analysis"] = "Стабильный рост знаний"
            else:
                trends["growth"]["trend"] = "decreasing"
                trends["growth"]["analysis"] = "Замедление роста знаний"
            
            # Формируем отчет
            report = {
                "time_window": time_window,
                "start_time": start_time,
                "end_time": now,
                "trends": trends,
                "new_nodes": len(new_nodes),
                "updated_nodes": len(updated_nodes),
                "total_nodes": len(self.knowledge_graph.get_all_nodes()),
                "timestamp": time.time()
            }
            
            # Добавляем возможность улучшения, если рост замедляется
            if report["trends"].get("growth", {}).get("trend") == "decreasing":
                if self.brain and hasattr(self.brain, 'self_analyzer'):
                    self.brain.self_analyzer.add_learning_opportunity(
                        concept="knowledge_growth",
                        opportunity_type="expansion",
                        priority=0.8,
                        domain="system",
                        evidence=["Замедление роста знаний"],
                        suggested_actions=[
                            "Увеличить источники знаний",
                            "Оптимизировать процесс интеграции новых знаний"
                        ]
                    )
            
            logger.info("Анализ эволюции знаний завершен")
            return report
            
        except Exception as e:
            logger.error(f"Ошибка анализа эволюции знаний: {e}", exc_info=True)
            return {}
    
    def analyze_knowledge_patterns(self) -> Dict[str, Any]:
        """
        Анализирует паттерны знаний для выявления возможностей.
        
        Returns:
            Dict[str, Any]: Результат анализа
        """
        try:
            logger.info("Анализ паттернов знаний...")
            
            # Получаем все узлы
            nodes = self.knowledge_graph.get_all_nodes(limit=5000)
            
            # Если узлов мало, пропускаем анализ
            if len(nodes) < 10:
                logger.info("Недостаточно узлов для анализа паттернов знаний")
                return {
                    "status": "info",
                    "message": "Недостаточно узлов для анализа",
                    "patterns": [],
                    "timestamp": time.time()
                }
            
            patterns = []
            
            # Анализируем типы узлов
            node_types = {}
            for node in nodes:
                node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
            
            # Проверяем доминирующие типы
            if "fact" in node_types and node_types["fact"] / len(nodes) > 0.7:
                patterns.append({
                    "type": "fact_overload",
                    "description": "Преобладание фактов в знаниях",
                    "priority": 0.6,
                    "suggestions": [
                        "Добавить больше концептов и отношений",
                        "Улучшить структурирование знаний"
                    ]
                })
            
            # Анализируем домены
            domains = {}
            for node in nodes:
                domains[node.domain] = domains.get(node.domain, 0) + 1
            
            # Проверяем дисбаланс доменов
            if domains:
                max_domain = max(domains, key=domains.get)
                if domains[max_domain] / len(nodes) > 0.6:
                    patterns.append({
                        "type": "domain_imbalance",
                        "description": f"Преобладание домена '{max_domain}'",
                        "priority": 0.7,
                        "suggestions": [
                            f"Расширить знания в других доменах",
                            "Интегрировать данные из домена '{max_domain}' в другие области"
                        ]
                    })
            
            logger.info(f"Анализ паттернов знаний завершен. Обнаружено {len(patterns)} паттернов")
            return {
                "status": "success",
                "patterns": patterns,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа паттернов знаний: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def analyze_knowledge_relevance(self, time_window: str = "month") -> Dict[str, Any]:
        """
        Анализирует актуальность знаний.
        
        Args:
            time_window: Период для анализа устаревших знаний
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        try:
            logger.info(f"Анализ актуальности знаний (период: {time_window})")
            
            # Определяем временные рамки для устаревших знаний
            now = time.time()
            if time_window == "week":
                outdated_threshold = now - 7 * 86400  # 7 дней
            elif time_window == "year":
                outdated_threshold = now - 365 * 86400  # 1 год
            else:  # month по умолчанию
                outdated_threshold = now - 30 * 86400  # 30 дней
            
            # Получаем все узлы
            nodes = self.knowledge_graph.get_all_nodes()
            
            # Анализируем устаревшие знания
            outdated_nodes = [node for node in nodes 
                             if hasattr(node, 'last_updated') and node.last_updated < outdated_threshold]
            
            # Анализируем неактуальные источники
            outdated_sources = []
            for node in outdated_nodes:
                if hasattr(node, 'meta') and 'sources' in node.meta:
                    for source in node.meta['sources']:
                        if source.get('timestamp', 0) < outdated_threshold:
                            outdated_sources.append({
                                "concept": node.name,
                                "source": source.get('source', 'unknown'),
                                "last_update": source.get('timestamp', 0)
                            })
            
            # Формируем отчет
            report = {
                "total_nodes": len(nodes),
                "outdated_nodes": len(outdated_nodes),
                "outdated_percentage": len(outdated_nodes) / max(1, len(nodes)),
                "outdated_sources": outdated_sources,
                "time_window": time_window,
                "timestamp": time.time()
            }
            
            # Добавляем возможность улучшения, если много устаревших знаний
            if report["outdated_percentage"] > 0.2:  # Более 20% устаревших
                if self.brain and hasattr(self.brain, 'self_analyzer'):
                    self.brain.self_analyzer.add_learning_opportunity(
                        concept="knowledge_relevance",
                        opportunity_type="updating",
                        priority=min(1.0, report["outdated_percentage"] * 1.5),
                        domain="system",
                        evidence=[f"{len(outdated_nodes)} устаревших узлов"],
                        suggested_actions=[
                            "Обновить устаревшие знания",
                            "Проверить источники информации"
                        ]
                    )
            
            logger.info(f"Анализ актуальности знаний завершен. {len(outdated_nodes)} устаревших узлов")
            return report
            
        except Exception as e:
            logger.error(f"Ошибка анализа актуальности знаний: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия в знаниях.
        
        Returns:
            List[Dict[str, Any]]: Список обнаруженных противоречий
        """
        try:
            logger.info("Анализ противоречий в знаниях...")
            
            contradictions = []
            nodes = self.knowledge_graph.get_all_nodes()
            
            # Сравниваем узлы для обнаружения противоречий
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    # Проверяем противоположные отношения
                    edges1 = self.knowledge_graph.get_edges(node1.id, direction="source")
                    edges2 = self.knowledge_graph.get_edges(node2.id, direction="source")
                    
                    for edge1 in edges1:
                        for edge2 in edges2:
                            # Проверяем, являются ли отношения противоположными
                            if (edge1["relation_type"] == "opposite" and edge2["relation_type"] == "opposite" and
                                edge1["target_id"] == node2.id and edge2["target_id"] == node1.id):
                                contradictions.append({
                                    "concept1": node1.name,
                                    "concept2": node2.name,
                                    "type": "opposite_relations",
                                    "strength": min(edge1["strength"], edge2["strength"]),
                                    "evidence": [
                                        f"'{node1.name}' является противоположностью '{edge1['target_id']}'",
                                        f"'{node2.name}' является противоположностью '{edge2['target_id']}'"
                                    ]
                                })
                    
                    # Проверяем противоречивые утверждения
                    if node1.name == node2.name and node1.domain != node2.domain:
                        # Возможно, это противоречивые определения одной концепции
                        contradictions.append({
                            "concept": node1.name,
                            "domains": [node1.domain, node2.domain],
                            "strength": min(node1.strength, node2.strength),
                            "evidence": [
                                f"Концепт '{node1.name}' имеет разные определения в доменах '{node1.domain}' и '{node2.domain}'"
                            ]
                        })
                    
                    # Проверяем числовые противоречия
                    if node1.node_type == "fact" and node2.node_type == "fact":
                        try:
                            val1 = float(node1.description)
                            val2 = float(node2.description)
                            if abs(val1 - val2) > 0.1 * max(abs(val1), abs(val2)):
                                contradictions.append({
                                    "concept": node1.name,
                                    "values": [val1, val2],
                                    "divergence": abs(val1 - val2) / max(abs(val1), abs(val2)),
                                    "evidence": [
                                        f"Значение в домене '{node1.domain}': {val1}",
                                        f"Значение в домене '{node2.domain}': {val2}"
                                    ]
                                })
                        except (ValueError, TypeError):
                            pass
            
            # Добавляем обнаруженные противоречия как возможности для обучения
            for contradiction in contradictions:
                if self.brain and hasattr(self.brain, 'self_analyzer'):
                    self.brain.self_analyzer.add_learning_opportunity(
                        concept=contradiction.get("concept", contradiction.get("concept1", "unknown")),
                        opportunity_type="contradiction_resolution",
                        priority=min(1.0, contradiction["strength"] * 1.2),
                        domain="knowledge_consistency",
                        evidence=contradiction.get("evidence", []),
                        suggested_actions=[
                            "Исследовать концепт глубже",
                            "Проверить источники информации",
                            "Интегрировать противоречивые знания"
                        ]
                    )
            
            logger.info(f"Анализ противоречий завершен. Обнаружено {len(contradictions)} противоречий")
            return contradictions
            
        except Exception as e:
            logger.error(f"Ошибка анализа противоречий: {e}", exc_info=True)
            return []
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по знаниям.
        
        Returns:
            Dict[str, Any]: Статистика по знаниям
        """
        try:
            # Получаем узлы
            nodes = self.knowledge_graph.get_all_nodes()
            
            # Статистика по доменам
            domains = {}
            for node in nodes:
                if hasattr(node, 'domain'):
                    domain = node.domain
                    domains[domain] = domains.get(domain, 0) + 1
            
            # Статистика по типам узлов
            node_types = {}
            for node in nodes:
                if hasattr(node, 'node_type'):
                    node_type = node.node_type
                    node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Статистика по связям
            total_edges = 0
            for node in nodes:
                edges = self.knowledge_graph.get_edges(node.id)
                total_edges += len(edges)
            
            # Среднее количество связей на узел
            avg_connections = total_edges / max(1, len(nodes))
            
            return {
                "total_nodes": len(nodes),
                "total_edges": total_edges,
                "domains": domains,
                "node_types": node_types,
                "avg_connections_per_node": avg_connections,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики знаний: {e}", exc_info=True)
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "domains": {},
                "node_types": {},
                "avg_connections_per_node": 0,
                "timestamp": time.time()
            }