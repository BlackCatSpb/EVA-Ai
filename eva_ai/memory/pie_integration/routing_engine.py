"""
L2: Routing Engine - Маршрутизация и адаптация параметров

Хранит правила генерации для доменов:
- temperature, repeat_penalty, max_tokens
- fallback_chain
- priority, success_rate

Узел типа: routing_rule
Связи: applies_to_domain → concept
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import numpy as np

from .fractal_graph_l1_l2 import FractalGraphL1L2, RoutingRuleData

logger = logging.getLogger("eumi.routing")


@dataclass
class RoutingParams:
    """Параметры маршрутизации."""
    temperature: float = 0.3
    repeat_penalty: float = 1.8
    max_tokens: int = 1024
    quant_profile: str = "Q4_K_M"
    fallback_chain: List[str] = None
    priority: float = 1.0
    rule_id: Optional[str] = None
    
    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = ["L3_memory", "keyword_response"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return {
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
            "quant_profile": self.quant_profile,
            "fallback_chain": self.fallback_chain,
            "priority": self.priority,
            "rule_id": self.rule_id
        }


class RoutingEngine:
    """
    L2: Routing & Behavior - адаптивная маршрутизация.
    
    Выбирает оптимальные параметры генерации на основе домена запроса.
    """
    
    def __init__(
        self,
        graph: FractalGraphL1L2,
        profiler: Optional[Any] = None
    ):
        """
        Args:
            graph: FractalGraphL1L2 instance
            profiler: ActivationProfiler для поиска домена по embedding (optional)
        """
        self.graph = graph
        self.profiler = profiler
        
        # Загружаем доменные настройки из brain_config.json
        self.domain_configs = self._load_routing_config()
        
        # Устанавливаем DEFAULT_PARAMS на основе конфигурации
        self.DEFAULT_PARAMS = self._get_default_params()
    
    def _load_routing_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию маршрутизации из brain_config.json."""
        default_domains = {
            "general": {"max_tokens": 1024, "temperature": 0.3, "repeat_penalty": 1.8},
            "coding": {"max_tokens": 2048, "temperature": 0.2, "repeat_penalty": 1.5},
            "creative": {"max_tokens": 1536, "temperature": 0.8, "repeat_penalty": 1.2},
            "science": {"max_tokens": 1536, "temperature": 0.3, "repeat_penalty": 1.8},
            "chat": {"max_tokens": 512, "temperature": 0.7, "repeat_penalty": 1.3}
        }
        
        try:
            import os
            import json
            # Находим путь к brain_config.json (три уровня вверх от eva_ai/memory/pie_integration/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
            config_path = os.path.join(project_root, "brain_config.json")
            
            if not os.path.exists(config_path):
                return default_domains
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            routing_config = config.get("routing", {})
            domains = routing_config.get("domains", default_domains)
            return domains if domains else default_domains
            
        except Exception:
            return default_domains
    
    def _get_default_params(self) -> RoutingParams:
        """Возвращает параметры по умолчанию на основе домена 'general'."""
        general_config = self.domain_configs.get("general", {})
        return RoutingParams(
            temperature=general_config.get("temperature", 0.3),
            repeat_penalty=general_config.get("repeat_penalty", 1.8),
            max_tokens=general_config.get("max_tokens", 1024),
            quant_profile="Q4_K_M",
            fallback_chain=["L3_memory", "keyword_response"],
            priority=1.0,
            rule_id=None
        )
        
    def create_rule(
        self,
        domain: str,
        params: Optional[RoutingParams] = None,
        link_to_concept: bool = True
    ) -> str:
        """
        Создать правило маршрутизации для домена.
        
        Args:
            domain: Домен (например, "astrophysics")
            params: Параметры или DEFAULT_PARAMS
            link_to_concept: Связать с concept узлом домена
            
        Returns:
            rule_id: ID созданного узла
        """
        if params is None:
            params = self.DEFAULT_PARAMS
        
        rule_id = self.graph.create_routing_rule(
            domain=domain,
            temperature=params.temperature,
            repeat_penalty=params.repeat_penalty,
            max_tokens=params.max_tokens,
            quant_profile=params.quant_profile,
            fallback_chain=params.fallback_chain,
            priority=params.priority
        )
        
        # Связываем с concept узлом если нужно
        if link_to_concept:
            # Ищем или создаём concept узел
            concept_id = self._ensure_concept_node(domain)
            if concept_id:
                self.graph.link_rule_to_domain(rule_id, concept_id)
        
        logger.info(f"Created routing rule: {rule_id} for domain={domain}")
        return rule_id
    
    def get_rule(
        self,
        domain_hint: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        query_text: Optional[str] = None
    ) -> RoutingParams:
        """
        Получить правило маршрутизации.
        
        Порядок поиска:
        1. Если domain_hint → поиск по домену
        2. Иначе если query_text и profiler → определить домен через profiler
        3. Иначе если query_embedding → поиск похожих профилей
        4. Если не найдено → DEFAULT_PARAMS
        
        Args:
            domain_hint: Подсказка домена (опционально)
            query_embedding: Вектор запроса для ANN-поиска (опционально)
            query_text: Текст запроса для определения домена (опционально)
            
        Returns:
            Параметры маршрутизации
        """
        # 1. Если domain_hint указан явно
        if domain_hint:
            rule_data = self.graph.get_routing_rule(domain_hint)
            if rule_data:
                return self._data_to_params(rule_data)
        
        # 2. Определяем домен через profiler по тексту
        if query_text and self.profiler:
            domains = self.profiler.get_domain_from_query(query_text, top_k=1)
            if domains:
                domain, confidence = domains[0]
                if confidence > 0.5:  # Порог уверенности
                    rule_data = self.graph.get_routing_rule(domain)
                    if rule_data:
                        logger.debug(f"Resolved domain '{domain}' from query (confidence={confidence:.2f})")
                        return self._data_to_params(rule_data)
        
        # 3. Поиск по embedding через profiler
        if query_embedding is not None and self.profiler:
            # Находим похожие профили
            similar = self.profiler.graph.find_similar_profiles(query_embedding, top_k=1)
            if similar:
                profile_id, similarity = similar[0]
                if similarity > 0.7:  # Порог сходства
                    # Получаем профиль и его домен
                    all_profiles = self.profiler.graph.list_activation_profiles()
                    for profile in all_profiles:
                        if profile.node_id == profile_id:
                            rule_data = self.graph.get_routing_rule(profile.domain)
                            if rule_data:
                                logger.debug(f"Found rule via embedding similarity={similarity:.2f}")
                                return self._data_to_params(rule_data)
        
        # 4. Возвращаем параметры из brain_config.json или дефолтные
        if domain_hint and domain_hint in self.domain_configs:
            config = self.domain_configs[domain_hint]
            logger.debug(f"Using routing params from brain_config.json for domain '{domain_hint}'")
            return RoutingParams(
                temperature=config.get("temperature", self.DEFAULT_PARAMS.temperature),
                repeat_penalty=config.get("repeat_penalty", self.DEFAULT_PARAMS.repeat_penalty),
                max_tokens=config.get("max_tokens", self.DEFAULT_PARAMS.max_tokens),
                quant_profile=self.DEFAULT_PARAMS.quant_profile,
                fallback_chain=self.DEFAULT_PARAMS.fallback_chain,
                priority=config.get("priority", self.DEFAULT_PARAMS.priority)
            )
        
        logger.debug("Using default routing params")
        return self.DEFAULT_PARAMS
    
    def update_rule_stats(
        self,
        rule_id: str,
        success: bool
    ) -> bool:
        """
        Обновить статистику использования правила.
        
        Args:
            rule_id: ID правила
            success: Был ли ответ успешным
            
        Returns:
            True если успешно
        """
        return self.graph.update_routing_rule_stats(rule_id, success)
    
    def record_feedback(
        self,
        rule_id: Optional[str],
        positive: bool
    ) -> bool:
        """
        Записать обратную связь о правиле.
        
        Args:
            rule_id: ID правила (None если использовались DEFAULT_PARAMS)
            positive: Положительная ли обратная связь
            
        Returns:
            True если успешно
        """
        if rule_id is None:
            logger.debug("No rule_id for feedback (default params used)")
            return False
        
        return self.update_rule_stats(rule_id, positive)
    
    def get_default_rule(self) -> RoutingParams:
        """Вернуть правила по умолчанию."""
        return self.DEFAULT_PARAMS
    
    def list_rules(
        self,
        domain_filter: Optional[str] = None
    ) -> List[RoutingParams]:
        """
        Список всех правил.
        
        Args:
            domain_filter: Фильтр по домену (опционально)
            
        Returns:
            Список правил
        """
        rules_data = self.graph.list_routing_rules()
        
        if domain_filter:
            rules_data = [r for r in rules_data if domain_filter in r.domain]
        
        return [self._data_to_params(r) for r in rules_data]
    
    def get_rule_for_generation(
        self,
        query: str,
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Получить полный набор параметров для генерации.
        
        Args:
            query: Текст запроса
            mode: Режим генерации (auto, condensed, extended)
            
        Returns:
            Словарь с параметрами для backend.generate()
        """
        # Определяем параметры маршрутизации
        params = self.get_rule(query_text=query)
        
        # Адаптируем под режим
        if mode == "extended":
            # Для extended увеличиваем max_tokens
            max_tokens = min(params.max_tokens * 2, 2048)
        elif mode == "condensed":
            # Для condensed уменьшаем
            max_tokens = min(params.max_tokens, 512)
        else:
            max_tokens = params.max_tokens
        
        return {
            "temperature": params.temperature,
            "repeat_penalty": params.repeat_penalty,
            "max_tokens": max_tokens,
            "quant_profile": params.quant_profile,
            "fallback_chain": params.fallback_chain,
            "rule_id": params.rule_id
        }
    
    def create_default_rules(self) -> List[str]:
        """
        Создать дефолтные правила для распространённых доменов.
        
        Returns:
            Список созданных rule_id
        """
        default_configs = [
            ("general", RoutingParams(temperature=0.3, repeat_penalty=1.8, max_tokens=1024)),
            ("coding", RoutingParams(temperature=0.2, repeat_penalty=1.5, max_tokens=4096)),
            ("creative", RoutingParams(temperature=0.8, repeat_penalty=1.2, max_tokens=1536)),
            ("science", RoutingParams(temperature=0.3, repeat_penalty=1.8, max_tokens=1536)),
            ("chat", RoutingParams(temperature=0.7, repeat_penalty=1.3, max_tokens=512)),
        ]
        
        created_ids = []
        for domain, params in default_configs:
            # Проверяем, существует ли уже
            existing = self.graph.get_routing_rule(domain)
            if existing is None:
                rule_id = self.create_rule(domain, params)
                created_ids.append(rule_id)
            else:
                created_ids.append(existing.node_id)
        
        logger.info(f"Created/verified {len(created_ids)} default routing rules")
        return created_ids
    
    def _data_to_params(self, data: RoutingRuleData) -> RoutingParams:
        """Конвертировать RoutingRuleData в RoutingParams."""
        return RoutingParams(
            temperature=data.temperature,
            repeat_penalty=data.repeat_penalty,
            max_tokens=data.max_tokens,
            quant_profile=data.quant_profile,
            fallback_chain=data.fallback_chain,
            priority=data.priority,
            rule_id=data.node_id
        )
    
    def _ensure_concept_node(self, domain: str) -> Optional[str]:
        """
        Убедиться, что concept узел для домена существует.
        
        Args:
            domain: Имя домена
            
        Returns:
            concept_id или None
        """
        # В реальной реализации здесь был бы поиск/создание concept узла
        # Для простоты возвращаем None - связь будет создана позже
        # или можно реализовать через отдельный KnowledgeGraph adapter
        return None
    
    def get_rule_stats(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить статистику правила.
        
        Args:
            rule_id: ID правила
            
        Returns:
            Статистика или None
        """
        rules = self.graph.list_routing_rules()
        
        for rule in rules:
            if rule.node_id == rule_id:
                return {
                    "rule_id": rule.node_id,
                    "domain": rule.domain,
                    "access_count": rule.access_count,
                    "success_count": rule.success_count,
                    "success_rate": rule.success_count / rule.access_count if rule.access_count > 0 else 0.0,
                    "priority": rule.priority,
                    "created_at": rule.created_at,
                    "last_used": rule.last_used
                }
        
        return None
    
    def optimize_params(
        self,
        domain: str,
        metric: str = "success_rate",
        min_samples: int = 10
    ) -> Optional[RoutingParams]:
        """
        Оптимизировать параметры на основе статистики.
        
        Args:
            domain: Домен
            metric: Метрика для оптимизации
            min_samples: Минимальное количество образцов
            
        Returns:
            Оптимизированные параметры или None
        """
        rule_data = self.graph.get_routing_rule(domain)
        
        if rule_data is None:
            return None
        
        if rule_data.access_count < min_samples:
            logger.debug(f"Not enough samples for optimization: {rule_data.access_count} < {min_samples}")
            return None
        
        success_rate = rule_data.success_count / rule_data.access_count
        
        # Простая эвристика: если success_rate низкий, пробуем изменить температуру
        params = self._data_to_params(rule_data)
        
        if success_rate < 0.5:
            # Понижаем температуру для более детерминированных ответов
            params.temperature = max(0.1, params.temperature - 0.1)
            logger.info(f"Optimized params for {domain}: temperature -> {params.temperature}")
        elif success_rate > 0.9:
            # Можно немного повысить для креативности
            params.temperature = min(0.9, params.temperature + 0.05)
        
        return params


def create_default_engine(
    graph: FractalGraphL1L2,
    profiler: Optional[Any] = None
) -> RoutingEngine:
    """Фабричный метод создания routing engine."""
    return RoutingEngine(graph, profiler)
