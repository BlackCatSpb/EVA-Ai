"""
Pie Adapter for DualGenerator

Интеграция Pie Architecture с существующим DualGenerator.
Позволяет использовать L1/L2 слои без изменения основного кода.

Usage:
    from eva_ai.memory.pie_integration import PieIntegration
    
    # В DualGenerator.__init__
    self.pie = PieIntegration(fractal_graph, config)
    
    # В DualGenerator.generate
    if self.use_pie:
        params = self.pie.get_generation_params(query)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .fractal_graph_l1_l2 import FractalGraphL1L2, create_l1l2_graph
from .activation_profiler import ActivationProfiler, create_default_profiler
from .routing_engine import RoutingEngine, create_default_engine, RoutingParams

logger = logging.getLogger("eva_ai.memory.pie_integration.adapter")


@dataclass
class GenerationMetadata:
    """Метаданные генерации для L1."""
    domain: str
    model_id: str
    entropy: float
    latency_ms: float
    quality: float
    query_type: Optional[str] = None
    context_nodes: Optional[List[str]] = None


class PieIntegration:
    """
    Главный адаптер интеграции Pie Architecture.
    
    Объединяет:
    - L1: ActivationProfiler (профилирование)
    - L2: RoutingEngine (маршрутизация)
    - L3: FractalGraphV2 (знания)
    
    Предоставляет простой API для интеграции с существующими генераторами.
    """
    
    def __init__(
        self,
        fractal_graph,
        config: Optional[Dict] = None,
        enable_l1: bool = True,
        enable_l2: bool = True,
        db_path: Optional[str] = None
    ):
        """
        Инициализация Pie Integration.
        
        Args:
            fractal_graph: FractalGraphV2 instance (L3)
            config: Конфигурация
            enable_l1: Включить L1 (ActivationProfiler)
            enable_l2: Включить L2 (RoutingEngine)
            db_path: Путь к SQLite (опционально)
        """
        self.fractal_graph = fractal_graph
        self.config = config or {}
        
        # Определяем путь к БД
        if db_path is None:
            db_path = self._get_db_path_from_graph()
        
        # Инициализация L1/L2
        self._init_l1_l2(db_path, enable_l1, enable_l2)
        
        logger.info(
            f"PieIntegration initialized: L1={enable_l1}, L2={enable_l2}"
        )
    
    def _get_db_path_from_graph(self) -> str:
        """Получить путь к БД из fractal_graph."""
        # Пробуем разные атрибуты
        if hasattr(self.fractal_graph, 'db_path'):
            return str(self.fractal_graph.db_path)
        
        if hasattr(self.fractal_graph, 'path'):
            path = str(self.fractal_graph.path)
            if path.endswith('.sqlite'):
                return path
            return path.replace('.sqlite', '_pie_l1_l2.sqlite')
        
        # Fallback - создаём в памяти
        return ':memory:'
    
    def _init_l1_l2(self, db_path: str, enable_l1: bool, enable_l2: bool):
        """Инициализация L1 и L2 слоёв."""
        # Создаём граф L1/L2
        self.l1l2_graph = create_l1l2_graph(db_path)
        
        # L1: Activation Profiler
        if enable_l1:
            self.profiler = create_default_profiler(self.l1l2_graph)
            logger.debug("L1 ActivationProfiler initialized")
        else:
            self.profiler = None
        
        # L2: Routing Engine
        if enable_l2:
            self.router = create_default_engine(
                self.l1l2_graph,
                self.profiler
            )
            # Создаём дефолтные правила
            self.router.create_default_rules()
            logger.debug("L2 RoutingEngine initialized with default rules")
        else:
            self.router = None
    
    # ==================== L2: Routing ====================
    
    def get_generation_params(
        self,
        query: str,
        mode: str = "auto",
        domain_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получить параметры генерации от L2.
        
        Args:
            query: Текст запроса
            mode: Режим ('auto', 'condensed', 'extended')
            domain_hint: Подсказка домена (опционально)
            
        Returns:
            Dict с параметрами:
            - temperature: float
            - repeat_penalty: float
            - max_tokens: int
            - quant_profile: str
            - fallback_chain: List[str]
            - rule_id: str
            - domain: str
        """
        if self.router is None:
            return self._get_default_params()
        
        # Получаем параметры от RoutingEngine
        params = self.router.get_rule_for_generation(query, mode)
        
        # Добавляем домен если есть hint
        if domain_hint:
            params['domain'] = domain_hint
        
        return params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Дефолтные параметры если L2 отключен."""
        return {
            'temperature': 0.7,
            'repeat_penalty': 1.1,
            'max_tokens': 512,
            'quant_profile': 'Q4_K_M',
            'fallback_chain': ['L3_memory', 'keyword_response'],
            'rule_id': None,
            'domain': 'general'
        }
    
    def suggest_domain(self, query: str) -> Tuple[str, float]:
        """
        Предложить домен для запроса через L1.
        
        Returns:
            (domain, confidence)
        """
        if self.profiler is None:
            return ('general', 0.0)
        
        domains = self.profiler.get_domain_from_query(query, top_k=1)
        
        if domains:
            return domains[0]
        
        return ('general', 0.0)
    
    def record_feedback(
        self,
        positive: bool,
        rule_id: Optional[str] = None
    ) -> bool:
        """
        Записать обратную связь для L2.
        
        Args:
            positive: Положительная ли обратная связь
            rule_id: ID правила (если None - используется последнее)
            
        Returns:
            True если успешно
        """
        if self.router is None:
            return False
        
        return self.router.record_feedback(rule_id, positive)
    
    # ==================== L1: Profiling ====================
    
    def record_generation(
        self,
        query: str,
        response: str,
        metadata: GenerationMetadata
    ) -> bool:
        """
        Записать генерацию в профиль L1.
        
        Args:
            query: Текст запроса
            response: Текст ответа
            metadata: Метаданные генерации
            
        Returns:
            True если успешно
        """
        if self.profiler is None:
            return False
        
        return self.profiler.record_generation(
            domain=metadata.domain,
            model_id=metadata.model_id,
            query=query,
            response=response,
            entropy=metadata.entropy,
            latency_ms=metadata.latency_ms,
            quality=metadata.quality
        )
    
    def get_profile_stats(self, domain: str) -> Optional[Dict]:
        """Получить статистику профиля для домена."""
        if self.profiler is None:
            return None
        
        profile = self.profiler.get_profile(domain, 'model_a')
        
        if profile is None:
            return None
        
        return {
            'domain': profile.domain,
            'sample_count': profile.sample_count,
            'avg_entropy': profile.avg_entropy,
            'avg_latency_ms': profile.avg_latency_ms,
            'avg_quality': profile.avg_quality
        }
    
    # ==================== Combined Operations ====================
    
    def before_generation(
        self,
        query: str,
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Подготовка перед генерацией (L2 + L1).
        
        Returns:
            Dict с параметрами и метаданными
        """
        # L2: Получаем параметры
        params = self.get_generation_params(query, mode)
        
        # L1: Определяем домен если не указан
        if 'domain' not in params or params['domain'] == 'general':
            suggested_domain, confidence = self.suggest_domain(query)
            if confidence > 0.5:
                params['domain'] = suggested_domain
        
        return params
    
    def after_generation(
        self,
        query: str,
        response: str,
        params: Dict[str, Any],
        latency_ms: float
    ):
        """
        Сохранение результатов после генерации (L1).
        """
        if self.profiler is None:
            return
        
        # Оцениваем качество (упрощённо)
        quality = self._estimate_quality(response)
        
        # Estimate entropy from response characteristics
        words = response.split()
        if len(words) > 1:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            entropy = 1.0 - repetition_ratio if repetition_ratio < 1.0 else 0.0
        else:
            entropy = 0.5

        # Создаём метаданные
        metadata = GenerationMetadata(
            domain=params.get('domain', 'general'),
            model_id=params.get('model_id', 'model_a'),
            entropy=entropy,
            latency_ms=latency_ms,
            quality=quality
        )
        
        # Записываем в профиль
        self.record_generation(query, response, metadata)
    
    def _estimate_quality(self, response: str) -> float:
        """Оценить качество ответа (0-1)."""
        # Простая эвристика
        if not response or len(response) < 10:
            return 0.3
        
        if len(response) < 50:
            return 0.6
        
        # Проверяем на повторы
        words = response.split()
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.5:
            return 0.5  # Много повторов
        
        return 0.8
    
    # ==================== Stats ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику интеграции."""
        stats = {
            'l1_enabled': self.profiler is not None,
            'l2_enabled': self.router is not None,
        }
        
        if self.router:
            rules = self.router.list_rules()
            stats['routing_rules_count'] = len(rules)
            stats['routing_domains'] = list(set(r.fallback_chain[0] for r in rules if r.fallback_chain))
        
        if self.profiler:
            profiles = self.profiler.list_all_profiles()
            stats['profiles_count'] = len(profiles)
            stats['total_samples'] = sum(p.sample_count for p in profiles)
        
        return stats


# Convenience function для быстрой инициализации
def create_pie_integration(
    fractal_graph,
    config: Optional[Dict] = None,
    enable_l1: bool = True,
    enable_l2: bool = True
) -> PieIntegration:
    """
    Быстрое создание PieIntegration.
    
    Usage:
        from eva_ai.memory.pie_integration import create_pie_integration
        
        pie = create_pie_integration(fractal_graph)
        params = pie.get_generation_params("Hello!")
    """
    return PieIntegration(
        fractal_graph=fractal_graph,
        config=config,
        enable_l1=enable_l1,
        enable_l2=enable_l2
    )
