"""
UnifiedContradictionManager - единая точка входа для работы с противоречиями

Интегрирует три системы:
- ContradictionGenerator: быстрая генерация шаблонных противоречий
- ContradictionMiner: глубокий анализ графа для реальных противоречий
- Legacy Detector: проверка согласованности фактов

По спецификации C3: Объединение трёх систем детекции противоречий
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("eva_ai.contradiction.unified")

class ContradictionSource(Enum):
    """Источник противоречия"""
    GENERATOR = "generator"    # Шаблонная генерация
    MINER = "miner"           # Анализ графа
    LEGACY = "legacy"         # Legacy детектор
    AUTO = "auto"             # Автоматический выбор


@dataclass
class UnifiedContradiction:
    """
    Унифицированный формат противоречия для всех систем.

    Используется как единый формат хранения и передачи данных
    между компонентами системы противоречий.
    """
    id: str
    source_layer: str  # 'generator' | 'miner' | 'legacy'

    # Основное содержимое
    concept: str
    conflicting_statements: List[str]  # [statement_a, statement_b]

    # Метрики
    divergence_level: float = 0.0
    confidence: float = 0.0

    # Статус и разрешение
    status: str = "detected"  # 'detected' | 'analyzing' | 'resolved' | 'archived'
    resolution: Optional[str] = None
    resolution_node_id: Optional[str] = None

    # Оригинальные данные от системы-источника
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    # Временные метки
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    # Для генерации самодиалога
    resolution_question: str = ""
    title: str = ""


class UnifiedContradictionManager:
    """
    Единая точка входа для работы с противоречиями.

    Интегрирует три системы:
    1. Generator (Fast Layer) - шаблонная генерация для концептов
    2. Miner (Deep Layer) - анализ графа для реальных противоречий
    3. Legacy (Fact Layer) - проверка согласованности фактов

    Основной API:
    - unified_detect(): детекция противоречий
    - unified_resolve(): разрешение противоречия
    - get_context_for_prompt(): контекст для генерации
    - get_unified_stats(): статистика по всем системам
    """

    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация UnifiedContradictionManager.

        Args:
            brain: CoreBrain или None
            config: Конфигурация {
                'enable_generator': bool,
                'enable_miner': bool,
                'enable_legacy': bool,
                'auto_select': bool
            }
        """
        self.brain = brain
        self.config = config or {}

        # Включение/выключение систем
        self.enable_generator = self.config.get('enable_generator', True)
        self.enable_miner = self.config.get('enable_miner', True)
        self.enable_legacy = self.config.get('enable_legacy', True)
        self.auto_select = self.config.get('auto_select', True)

        # Реестр противоречий (унифицированный формат)
        self._contradictions: List[UnifiedContradiction] = []

        # Ссылка на компоненты (lazy loading)
        self._generator = None
        self._miner = None
        self._legacy_detector = None
        self._cache = None

        logger.info("UnifiedContradictionManager инициализирован")

    @property
    def generator(self):
        """Lazy loading ContradictionGenerator"""
        if self._generator is None and self.enable_generator:
            try:
                from eva_ai.contradiction.contradiction_generator import ContradictionGenerator
                self._generator = ContradictionGenerator(brain=self.brain)
                logger.debug("ContradictionGenerator подключён")
            except Exception as e:
                logger.warning(f"Не удалось подключить ContradictionGenerator: {e}")
        return self._generator

    @property
    def miner(self):
        """Lazy loading ContradictionMiner"""
        if self._miner is None and self.enable_miner:
            try:
                from eva_ai.contradiction.contradiction_miner import ContradictionMiner
                self._miner = ContradictionMiner(brain=self.brain)
                logger.debug("ContradictionMiner подключён")
            except Exception as e:
                logger.warning(f"Не удалось подключить ContradictionMiner: {e}")
        return self._miner

    @property
    def legacy_detector(self):
        """Lazy loading Legacy Detector"""
        if self._legacy_detector is None and self.enable_legacy:
            try:
                from eva_ai.contradiction.contradiction_manager import ContradictionManager
                self._legacy_detector = ContradictionManager(brain=self.brain)
                logger.debug("Legacy Detector подключён")
            except Exception as e:
                logger.warning(f"Не удалось подключить Legacy Detector: {e}")
        return self._legacy_detector

    def _generate_id(self, source: str, concept: str) -> str:
        """Генерация уникального ID"""
        import hashlib
        timestamp = str(time.time())
        data = f"{source}:{concept}:{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def unified_detect(
        self,
        source: Optional[str] = None,
        target: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> List[UnifiedContradiction]:
        """
        Единый метод детекции противоречий.

        Args:
            source: 'generator' | 'miner' | 'legacy' | 'auto' (по умолчанию)
            target: концепт или текст для проверки
            options: дополнительные параметры {
                'domain': str,  # domain для generator
                'priority': float,
                'max_results': int
            }

        Returns:
            List[UnifiedContradiction] - список обнаруженных противоречий
        """
        options = options or {}
        results = []

        # Определение источников для проверки
        if source == 'auto' or source is None:
            sources_to_check = []
            if self.enable_generator:
                sources_to_check.append('generator')
            if self.enable_miner:
                sources_to_check.append('miner')
            if self.enable_legacy:
                sources_to_check.append('legacy')
        else:
            sources_to_check = [source] if source in ['generator', 'miner', 'legacy'] else []

        logger.debug(f"Unified detect: sources={sources_to_check}, target={target}")

        # Проверка каждого источника
        for src in sources_to_check:
            try:
                if src == 'generator':
                    results.extend(self._detect_generator(target, options))
                elif src == 'miner':
                    results.extend(self._detect_miner(target, options))
                elif src == 'legacy':
                    results.extend(self._detect_legacy(target, options))
            except Exception as e:
                logger.warning(f"Error detecting from {src}: {e}")

        # Добавляем в реестр
        for contr in results:
            self._add_contradiction(contr)

        return results

    def _detect_generator(self, target: str, options: Dict) -> List[UnifiedContradiction]:
        """Детекция через ContradictionGenerator (шаблоны)"""
        if not self.generator:
            return []

        results = []
        concept = target if isinstance(target, str) else str(target)
        domain = options.get('domain', 'general')

        try:
            # Generator создаёт противоречие из шаблонов
            gen_contr = self.generator.generate_contradiction(concept, domain)

            if gen_contr:
                unified = UnifiedContradiction(
                    id=self._generate_id('generator', concept),
                    source_layer='generator',
                    concept=concept,
                    conflicting_statements=[
                        gen_contr.viewpoint_a,
                        gen_contr.viewpoint_b
                    ],
                    divergence_level=gen_contr.divergence_level,
                    confidence=0.7,  # Шаблонная генерация - средняя уверенность
                    status='detected',
                    source_metadata={
                        'reasoning_a': gen_contr.reasoning_a,
                        'reasoning_b': gen_contr.reasoning_b,
                        'domain': domain
                    },
                    title=f"Противоречие: {concept}"
                )
                results.append(unified)

        except Exception as e:
            logger.debug(f"Generator detection error: {e}")

        return results

    def _detect_miner(self, target: Optional[str], options: Dict) -> List[UnifiedContradiction]:
        """Детекция через ContradictionMiner (анализ графа)"""
        if not self.miner:
            return []

        results = []

        try:
            # Miner работает в фоне, получаем текущие кандидаты
            if hasattr(self.miner, 'get_active_contradictions'):
                candidates = self.miner.get_active_contradictions()

                for cand in candidates[:options.get('max_results', 3)]:
                    unified = UnifiedContradiction(
                        id=cand.id if hasattr(cand, 'id') else self._generate_id('miner', cand.get('concept', 'unknown')),
                        source_layer='miner',
                        concept=cand.get('concept', target or 'unknown'),
                        conflicting_statements=cand.get('statements', []),
                        divergence_level=cand.get('priority', 0.5),
                        confidence=cand.get('max_contra_score', 0.65),
                        status='detected',
                        resolution_question=cand.get('resolution_question', ''),
                        title=cand.get('title', ''),
                        source_metadata=cand
                    )
                    results.append(unified)

        except Exception as e:
            logger.debug(f"Miner detection error: {e}")

        return results

    def _detect_legacy(self, target: Optional[Union[str, Dict]], options: Dict) -> List[UnifiedContradiction]:
        """Детекция через Legacy Detector (факты)"""
        if not self.legacy_detector:
            return []

        results = []

        try:
            # Legacy детектор проверяет концепт или факты
            if isinstance(target, str):
                legacy_results = self.legacy_detector.detect_contradictions(concept=target)
            elif isinstance(target, dict):
                legacy_results = self.legacy_detector.detect_contradictions_in_new_fact(target)
            else:
                legacy_results = self.legacy_detector.detect_contradictions() or []

            for lr in legacy_results:
                if isinstance(lr, dict):
                    unified = UnifiedContradiction(
                        id=self._generate_id('legacy', lr.get('concept', 'unknown')),
                        source_layer='legacy',
                        concept=lr.get('concept', 'unknown'),
                        conflicting_statements=self._extract_statements_from_legacy(lr),
                        divergence_level=lr.get('divergence_level', 0.5),
                        confidence=0.8,
                        status='detected',
                        source_metadata=lr
                    )
                    results.append(unified)

        except Exception as e:
            logger.debug(f"Legacy detection error: {e}")

        return results

    def _extract_statements_from_legacy(self, lr: Dict) -> List[str]:
        """Извлечение противоречивых утверждений из legacy формата"""
        statements = []

        if 'conflicting_facts' in lr:
            for fact in lr['conflicting_facts']:
                if isinstance(fact, dict):
                    statements.append(str(fact.get('value', str(fact))))
                else:
                    statements.append(str(fact))
        elif 'contradiction' in lr:
            statements.append(str(lr['contradiction']))

        return statements if statements else ["Факты противоречат друг другу"]

    def _add_contradiction(self, contradiction: UnifiedContradiction) -> None:
        """Добавление противоречия в реестр (без дубликатов)"""
        # Проверка на дубликат
        for existing in self._contradictions:
            if (existing.concept == contradiction.concept and
                existing.source_layer == contradiction.source_layer):
                return  # Уже есть

        self._contradictions.append(contradiction)

        # Публикация события для SelfDialogLearning
        if self.brain and hasattr(self.brain, 'event_bus'):
            try:
                self.brain.event_bus.publish(
                    'contradiction.detected',
                    contradiction,
                    priority=2  # HIGH
                )
            except Exception:
                pass

    def unified_resolve(
        self,
        contradiction_id: str,
        resolution_data: Dict[str, Any]
    ) -> bool:
        """
        Единое разрешение противоречия.

        Args:
            contradiction_id: ID противоречия
            resolution_data: {
                'resolution': str,  # Текст разрешения
                'strategy': str,    # Стратегия разрешения
                'winning_statement': str
            }

        Returns:
            bool: успех операции
        """
        # Находим противоречие
        contr = self._find_contradiction(contradiction_id)
        if not contr:
            logger.warning(f"Contradiction {contradiction_id} not found")
            return False

        try:
            contr.resolution = resolution_data.get('resolution', '')
            contr.resolution_node_id = resolution_data.get('resolution_node_id')
            contr.status = 'resolved'
            contr.resolved_at = time.time()

            # Делегируем в соответствующую систему
            if contr.source_layer == 'generator' and self.generator:
                self.generator.save_resolution(contr.id, resolution_data)
            elif contr.source_layer == 'miner' and self.miner:
                if hasattr(self.miner, 'resolve_contradiction'):
                    self.miner.resolve_contradiction(contr.id, resolution_data)
            elif contr.source_layer == 'legacy' and self.legacy_detector:
                if hasattr(self.legacy_detector, 'resolve_contradiction'):
                    self.legacy_detector.resolve_contradiction(contr.id, resolution_data)

            logger.info(f"Contradiction {contradiction_id} resolved via {contr.source_layer}")
            return True

        except Exception as e:
            logger.error(f"Error resolving {contradiction_id}: {e}")
            return False

    def _find_contradiction(self, contr_id: str) -> Optional[UnifiedContradiction]:
        """Поиск противоречия по ID"""
        for contr in self._contradictions:
            if contr.id == contr_id:
                return contr
        return None

    def get_context_for_prompt(
        self,
        concept_name: str,
        max_count: int = 3
    ) -> str:
        """
        Получение контекста противоречий для генерации промпта.

        Args:
            concept_name: Название концепта
            max_count: Максимальное количество противоречий

        Returns:
            str: форматированный контекст для промпта
        """
        context_parts = []

        # Находим противоречия для данного концепта
        relevant = [
            c for c in self._contradictions
            if c.concept == concept_name and c.status != 'archived'
        ]

        for contr in relevant[:max_count]:
            if contr.source_layer == 'generator':
                context_parts.append(
                    f"Противоречие: {contr.conflicting_statements[0]} ↔ {contr.conflicting_statements[1]}"
                )
            elif contr.source_layer == 'miner':
                context_parts.append(
                    f"Конфликт узлов графа: {contr.title or contr.concept}"
                )
            elif contr.source_layer == 'legacy':
                context_parts.append(
                    f"Конфликт фактов: {', '.join(contr.conflicting_statements[:2])}"
                )

        if context_parts:
            return "\nКонтекст противоречий:\n- " + "\n- ".join(context_parts)
        return ""

    def get_unified_stats(self) -> Dict[str, Any]:
        """Получение статистики по всем системам"""
        stats = {
            'total': len(self._contradictions),
            'by_source': {
                'generator': len([c for c in self._contradictions if c.source_layer == 'generator']),
                'miner': len([c for c in self._contradictions if c.source_layer == 'miner']),
                'legacy': len([c for c in self._contradictions if c.source_layer == 'legacy'])
            },
            'by_status': {
                'detected': len([c for c in self._contradictions if c.status == 'detected']),
                'analyzing': len([c for c in self._contradictions if c.status == 'analyzing']),
                'resolved': len([c for c in self._contradictions if c.status == 'resolved'])
            },
            'components_available': {
                'generator': self.generator is not None,
                'miner': self.miner is not None,
                'legacy': self.legacy_detector is not None
            }
        }

        return stats

    def get_all_contradictions(self) -> List[UnifiedContradiction]:
        """Получение всех противоречий"""
        return self._contradictions.copy()

    def get_contradiction_by_id(self, contr_id: str) -> Optional[UnifiedContradiction]:
        """Получение противоречия по ID"""
        return self._find_contradiction(contr_id)