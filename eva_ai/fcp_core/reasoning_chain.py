"""
ReasoningChain - Накопление цепочки рассуждений (EVA.txt)

Модуль для отслеживания и накопления многошаговых рассуждений.
Интегрирован с: MemorySnapshot, ScenarioTCM, KCA, SRG, ThinkingController.

Функции:
- Накопление промежуточных выводов
- Форматирование для включения в промпт
- Сохранение/восстановление состояния
- Анализ согласованности рассуждений
"""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("FCP.ReasoningChain")


@dataclass
class ReasoningStep:
    """Один шаг рассуждения"""
    step_id: int
    timestamp: float
    prompt: str
    reasoning: str           # Содержание рассуждения (<think> content)
    conclusion: str          # Вывод из этого шага
    intermediate_claims: List[str] = field(default_factory=list)  # Промежуточные утверждения
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningChain:
    """
    Цепочка рассуждений - рабочая память для многошаговых задач.
    
    Интегрирован с:
    - MemorySnapshot (6.1): снимки состояний для восстановления
    - ScenarioTCM (6.3): сохранение сценариев рассуждений
    - KCA (3.1-3.3): проверка согласованности с графом знаний
    - SRG (5): семантическая маршрутизация
    - ThinkingController: управление режимом рассуждений
    """
    
    def __init__(self, 
                 max_steps: int = 20,
                 similarity_threshold: float = 0.7,
                 fractal_graph: Any = None,
                 memory_snapshot: Any = None,
                 scenario_tcm: Any = None):
        """
        Args:
            max_steps: максимальное количество шагов в цепочке
            similarity_threshold: порог схожести для определения того же рассуждения
            fractal_graph: ссылка на FractalGraphV2 для проверки согласованности
            memory_snapshot: ссылка на MemorySnapshot для снимков состояний
            scenario_tcm: ссылка на ScenarioTCM для сохранения сценариев
        """
        self.max_steps = max_steps
        self.similarity_threshold = similarity_threshold
        self.fractal_graph = fractal_graph
        self.memory_snapshot = memory_snapshot
        self.scenario_tcm = scenario_tcm
        
        # Текущая цепочка рассуждений
        self.steps: List[ReasoningStep] = []
        
        # Текущая сессия рассуждений
        self.current_session_id: Optional[str] = None
        self.session_start_time: float = 0
        self.is_active: bool = False
        
        # Индекс для быстрого поиска
        self._step_index: Dict[str, ReasoningStep] = {}
        
        # Статистика
        self.total_sessions: int = 0
        self.total_steps: int = 0
        
        logger.info(f"ReasoningChain initialized: max_steps={max_steps}")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Начать новую сессию рассуждений"""
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        
        self.current_session_id = session_id
        self.session_start_time = time.time()
        self.is_active = True
        self.steps = []
        self._step_index = {}
        
        logger.info(f"ReasoningChain: Started session {session_id}")
        return session_id
    
    def end_session(self, save_to_tcm: bool = True) -> Dict[str, Any]:
        """Завершить текущую сессию"""
        if not self.is_active:
            return {"status": "no_active_session"}
        
        session_summary = {
            "session_id": self.current_session_id,
            "duration": time.time() - self.session_start_time,
            "steps_count": len(self.steps),
            "steps": [
                {
                    "id": s.step_id,
                    "conclusion": s.conclusion,
                    "confidence": s.confidence
                }
                for s in self.steps
            ]
        }
        
        # Сохраняем в ScenarioTCM если доступно
        if save_to_tcm and self.scenario_tcm and self.steps:
            try:
                scenario_text = self._format_for_scenario()
                if hasattr(self.scenario_tcm, 'add_dialog_turn'):
                    self.scenario_tcm.add_dialog_turn(
                        role="system",
                        text=f"Рассуждение: {scenario_text}"
                    )
                logger.info(f"ReasoningChain: Saved to ScenarioTCM")
            except Exception as e:
                logger.debug(f"Failed to save to TCM: {e}")
        
        # Делаем снимок состояния если доступно
        if self.memory_snapshot and self.steps:
            try:
                snapshot_data = self.get_state()
                if hasattr(self.memory_snapshot, 'save_snapshot'):
                    self.memory_snapshot.save_snapshot(
                        state_type="reasoning_chain",
                        data=snapshot_data
                    )
            except Exception as e:
                logger.debug(f"Failed to save snapshot: {e}")
        
        self.is_active = False
        self.total_sessions += 1
        
        logger.info(f"ReasoningChain: Ended session {self.current_session_id}, steps={len(self.steps)}")
        return session_summary
    
    def add_step(self, 
                 prompt: str,
                 reasoning: str,
                 conclusion: str,
                 intermediate_claims: Optional[List[str]] = None,
                 confidence: float = 0.5,
                 metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Добавить шаг рассуждения в цепочку.
        
        Returns:
            step_id: id добавленного шага
        """
        if not self.is_active:
            self.start_session()
        
        step_id = len(self.steps)
        step = ReasoningStep(
            step_id=step_id,
            timestamp=time.time(),
            prompt=prompt[:500],  # Ограничиваем длину
            reasoning=reasoning[:2000],
            conclusion=conclusion[:500],
            intermediate_claims=intermediate_claims or [],
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.steps.append(step)
        self._step_index[f"step_{step_id}"] = step
        self.total_steps += 1
        
        # Проверяем согласованность с графом знаний если доступно
        if self.fractal_graph and hasattr(self.fractal_graph, 'check_consistency'):
            self._verify_consistency(step)
        
        # Обрезаем если превысили лимит
        if len(self.steps) > self.max_steps:
            self.steps = self.steps[-self.max_steps:]
            self._rebuild_index()
        
        logger.debug(f"ReasoningChain: Added step {step_id}, conclusion: {conclusion[:50]}...")
        return step_id
    
    def _verify_consistency(self, step: ReasoningStep):
        """Проверить согласованность шага с графом знаний (KCA интеграция)"""
        try:
            # Проверяем ключевые утверждения на согласованность
            claims_to_check = step.intermediate_claims[:3]  # Проверяем первые 3
            
            for claim in claims_to_check:
                if hasattr(self.fractal_graph, 'verify_fact'):
                    is_verified = self.fractal_graph.verify_fact(claim)
                    step.metadata[f"claim_{step.step_id}_verified"] = is_verified
                    
                    if not is_verified:
                        logger.debug(f"ReasoningChain: Claim not verified: {claim[:50]}...")
                        
        except Exception as e:
            logger.debug(f"Consistency check failed: {e}")
    
    def _rebuild_index(self):
        """Перестроить индекс шагов"""
        self._step_index = {f"step_{s.step_id}": s for s in self.steps}
    
    def get_context(self, 
                   include_reasoning: bool = True,
                   max_steps: int = 10) -> str:
        """
        Получить форматированный контекст рассуждений для включения в промпт.
        
        Args:
            include_reasoning: включать ли полный текст рассуждений
            max_steps: максимальное количество последних шагов для включения
        """
        if not self.steps:
            return ""
        
        # Берём последние max_steps шагов
        recent_steps = self.steps[-max_steps:]
        
        context_parts = ["[Цепочка предыдущих рассуждений]"]
        
        for step in recent_steps:
            context_parts.append(f"\n--- Шаг {step.step_id + 1} ---")
            context_parts.append(f"Вопрос: {step.prompt}")
            
            if include_reasoning and step.reasoning:
                # Сокращаем reasoning если слишком длинный
                reasoning_preview = step.reasoning[:300] + "..." if len(step.reasoning) > 300 else step.reasoning
                context_parts.append(f"Рассуждение: {reasoning_preview}")
            
            context_parts.append(f"Вывод: {step.conclusion}")
            
            if step.intermediate_claims:
                claims_str = "; ".join(step.intermediate_claims[:3])
                context_parts.append(f"Утверждения: {claims_str}")
        
        context_parts.append("\n[Продолжай рассуждение с учётом выводов выше]")
        
        return "\n".join(context_parts)
    
    def get_conclusions_summary(self) -> str:
        """Получить краткую сводку всех выводов"""
        if not self.steps:
            return ""
        
        lines = ["[Итоги предыдущих шагов]"]
        for step in self.steps:
            lines.append(f"{step.step_id + 1}. {step.conclusion}")
        
        return "\n".join(lines)
    
    def get_last_conclusion(self) -> Optional[str]:
        """Получить последний вывод"""
        if self.steps:
            return self.steps[-1].conclusion
        return None
    
    def get_step(self, step_id: int) -> Optional[ReasoningStep]:
        """Получить конкретный шаг по ID"""
        return self._step_index.get(f"step_{step_id}")
    
    def find_step_by_conclusion(self, keyword: str) -> Optional[ReasoningStep]:
        """Найти шаг по ключевому слову в выводе"""
        keyword_lower = keyword.lower()
        for step in reversed(self.steps):
            if keyword_lower in step.conclusion.lower():
                return step
        return None
    
    def analyze_consistency(self) -> Dict[str, Any]:
        """
        Проанализировать согласованность всей цепочки.
        
        Returns:
            Dict с результатами анализа
        """
        if len(self.steps) < 2:
            return {"status": "insufficient_steps", "consistent": True}
        
        issues = []
        
        # Проверяем логическую связь между шагами
        for i in range(len(self.steps) - 1):
            current = self.steps[i]
            next_step = self.steps[i + 1]
            
            # Проверяем, есть ли связь между выводом и следующим вопросом
            if current.conclusion.lower() not in next_step.prompt.lower():
                # Не критично, но отмечаем
                pass
        
        # Проверяем противоречия в выводах
        all_conclusions = [s.conclusion.lower() for s in self.steps]
        
        # Ищем возможные противоречия (упрощённо)
        contradiction_keywords = [
            ("не", "да"), ("нет", "да"), ("неверно", "верно"),
            ("假的", "真的"), ("false", "true")
        ]
        
        for kw1, kw2 in contradiction_keywords:
            if any(kw1 in c for c in all_conclusions) and any(kw2 in c for c in all_conclusions):
                issues.append(f"Possible contradiction: {kw1} vs {kw2}")
        
        return {
            "status": "analyzed",
            "steps_count": len(self.steps),
            "issues": issues,
            "consistent": len(issues) == 0,
            "confidence_avg": sum(s.confidence for s in self.steps) / len(self.steps)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Получить текущее состояние для снимка (MemorySnapshot интеграция)"""
        return {
            "session_id": self.current_session_id,
            "is_active": self.is_active,
            "steps_count": len(self.steps),
            "total_sessions": self.total_sessions,
            "total_steps": self.total_steps,
            "steps": [
                {
                    "step_id": s.step_id,
                    "prompt": s.prompt,
                    "reasoning": s.reasoning,
                    "conclusion": s.conclusion,
                    "intermediate_claims": s.intermediate_claims,
                    "confidence": s.confidence,
                    "timestamp": s.timestamp
                }
                for s in self.steps
            ]
        }
    
    def restore_state(self, state: Dict[str, Any]):
        """Восстановить состояние из снимка"""
        self.current_session_id = state.get("session_id")
        self.is_active = state.get("is_active", False)
        self.total_sessions = state.get("total_sessions", 0)
        self.total_steps = state.get("total_steps", 0)
        
        self.steps = []
        for step_data in state.get("steps", []):
            step = ReasoningStep(**step_data)
            self.steps.append(step)
        
        self._rebuild_index()
        logger.info(f"ReasoningChain: Restored {len(self.steps)} steps from snapshot")
    
    def _format_for_scenario(self) -> str:
        """Форматировать цепочку для сохранения в ScenarioTCM"""
        if not self.steps:
            return ""
        
        lines = [f"Рассуждение из {len(self.steps)} шагов:"]
        for step in self.steps:
            lines.append(f"Шаг {step.step_id + 1}: {step.conclusion}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Очистить текущую цепочку"""
        self.steps = []
        self._step_index = {}
        self.is_active = False
        self.current_session_id = None
        logger.info("ReasoningChain: Cleared")


class ReasoningChainManager:
    """
    Менеджер для управления несколькими цепочками рассуждений.
    Полезно для параллельных задач.
    """
    
    def __init__(self, default_config: Optional[Dict] = None):
        self.default_config = default_config or {}
        self.chains: Dict[str, ReasoningChain] = {}
        self.active_chain_id: Optional[str] = None
    
    def create_chain(self, 
                     chain_id: str,
                     fractal_graph: Any = None,
                     memory_snapshot: Any = None,
                     scenario_tcm: Any = None) -> ReasoningChain:
        """Создать новую цепочку"""
        chain = ReasoningChain(
            max_steps=self.default_config.get("max_steps", 20),
            similarity_threshold=self.default_config.get("similarity_threshold", 0.7),
            fractal_graph=fractal_graph,
            memory_snapshot=memory_snapshot,
            scenario_tcm=scenario_tcm
        )
        self.chains[chain_id] = chain
        return chain
    
    def get_active_chain(self) -> Optional[ReasoningChain]:
        """Получить активную цепочку"""
        if self.active_chain_id and self.active_chain_id in self.chains:
            return self.chains[self.active_chain_id]
        return None
    
    def set_active_chain(self, chain_id: str):
        """Установить активную цепочку"""
        if chain_id in self.chains:
            self.active_chain_id = chain_id
        else:
            logger.warning(f"Chain {chain_id} not found")
    
    def remove_chain(self, chain_id: str):
        """Удалить цепочку"""
        if chain_id in self.chains:
            del self.chains[chain_id]
            if self.active_chain_id == chain_id:
                self.active_chain_id = None