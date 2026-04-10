"""
Протокол семантического арбитража противоречий.
При contradiction_flag=True: Model A формирует тезис/антитезис, 
Model B оценивает доказательства из памяти/веба, CoreBrain фиксирует резолюцию.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.arbitration")

@dataclass
class ArbitrationResult:
    """Результат арбитража противоречия."""
    resolved: bool
    thesis: str = ""
    antithesis: str = ""
    evidence_pro: List[str] = None
    evidence_con: List[str] = None
    resolution: str = ""
    winning_side: str = ""  # "thesis" or "antithesis"
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.evidence_pro is None:
            self.evidence_pro = []
        if self.evidence_con is None:
            self.evidence_con = []


class ContradictionArbitrator:
    """
    Протокол семантического арбитража для разрешения противоречий в графе.
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
    def initiate_arbitration(self, node_a_id: str, node_b_id: str, 
                             context: Dict[str, Any]) -> ArbitrationResult:
        """
        Инициировать арбитраж между двумя противоречащими узлами.
        
        Args:
            node_a_id: ID первого узла
            node_b_id: ID второго узла  
            context: Дополнительный контекст
            
        Returns:
            ArbitrationResult с резолюцией
        """
        logger.info(f"Starting arbitration: {node_a_id} vs {node_b_id}")
        
        # 1. Получаем содержимое узлов
        node_a_content = self._get_node_content(node_a_id)
        node_b_content = self._get_node_content(node_b_id)
        
        if not node_a_content or not node_b_content:
            return ArbitrationResult(
                resolved=False,
                resolution="Cannot retrieve node content"
            )
        
        # 2. Model A формирует тезис/антитезис (упрощённая версия)
        thesis, antithesis = self._formulate_positions(node_a_content, node_b_content)
        
        # 3. Model B оценивает доказательства (упрощённая версия)
        evidence_pro, evidence_con = self._evaluate_evidence(thesis, antithesis)
        
        # 4. Определяем победителя
        winning_side, confidence = self._resolve_conflict(
            thesis, antithesis, evidence_pro, evidence_con
        )
        
        result = ArbitrationResult(
            resolved=True,
            thesis=thesis,
            antithesis=antithesis,
            evidence_pro=evidence_pro,
            evidence_con=evidence_con,
            resolution=self._create_resolution(winning_side, confidence),
            winning_side=winning_side,
            confidence=confidence
        )
        
        # 5. Применяем резолюцию к графу
        self._apply_resolution(node_a_id, node_b_id, result)
        
        logger.info(f"Arbitration complete: winner={winning_side}, confidence={confidence:.2f}")
        
        return result
    
    def _get_node_content(self, node_id: str) -> Optional[str]:
        """Получить содержимое узла из графа."""
        if not self.brain:
            return None
            
        fractal_graph = getattr(self.brain, 'fractal_graph_v2', None)
        if fractal_graph and hasattr(fractal_graph, 'get_node'):
            node = fractal_graph.get_node(node_id)
            if node:
                return getattr(node, 'content', None) or getattr(node, 'description', '')
                
        knowledge_graph = getattr(self.brain, 'knowledge_graph', None)
        if knowledge_graph and hasattr(knowledge_graph, 'get_node'):
            node = knowledge_graph.get_node(node_id)
            if node:
                return getattr(node, 'content', '') or getattr(node, 'description', '')
                
        return None
    
    def _formulate_positions(self, content_a: str, content_b: str) -> Tuple[str, str]:
        """Сформулировать тезис и антитезис на основе содержимого."""
        # Упрощённая версия - в реальной системе здесь был бы вызов Model A
        thesis = f"Позиция A: {content_a[:200]}"
        antithesis = f"Позиция B: {content_b[:200]}"
        return thesis, antithesis
    
    def _evaluate_evidence(self, thesis: str, antithesis: str) -> Tuple[List[str], List[str]]:
        """Оценить доказательства для каждой позиции."""
        # Упрощённая версия - в реальной системе здесь был бы вызов Model B + поиск
        evidence_pro = ["Совпадение с более новыми источниками", "Подтверждено несколькими узлами"]
        evidence_con = ["Противоречит более надёжным источникам", "Имеет более низкую confidence"]
        return evidence_pro, evidence_con
    
    def _resolve_conflict(self, thesis: str, antithesis: str,
                          evidence_pro: List[str], 
                          evidence_con: List[str]) -> Tuple[str, float]:
        """Разрешить конфликт между позициями."""
        # Простая эвристика - побеждает позиция с большим количеством доказательств
        if len(evidence_pro) > len(evidence_con):
            return "thesis", min(0.9, 0.5 + len(evidence_pro) * 0.1)
        elif len(evidence_con) > len(evidence_pro):
            return "antithesis", min(0.9, 0.5 + len(evidence_con) * 0.1)
        else:
            return "thesis", 0.5  # Неопределённость
    
    def _create_resolution(self, winning_side: str, confidence: float) -> str:
        """Создать текст резолюции."""
        side_text = "тезис" if winning_side == "thesis" else "антитезис"
        return f"Принят {side_text} с уверенностью {confidence:.2f}. Противоречие разрешено."
    
    def _apply_resolution(self, node_a_id: str, node_b_id: str, result: ArbitrationResult):
        """Применить резолюцию к графу - пометить проигравшего как устаревший."""
        if not self.brain:
            return
            
        fractal_graph = getattr(self.brain, 'fractal_graph_v2', None)
        if fractal_graph and hasattr(fractal_graph, 'update_node'):
            # Помечаем проигравший узел как устаревший
            loser_id = node_b_id if result.winning_side == "thesis" else node_a_id
            try:
                fractal_graph.update_node(loser_id, {
                    "is_contradiction": True,
                    "metadata.resolution": result.resolution,
                    "metadata.arbitration_confidence": result.confidence
                })
                logger.info(f"Marked node {loser_id} as resolved contradiction")
            except Exception as e:
                logger.warning(f"Failed to apply resolution: {e}")


def create_arbitrator(brain=None) -> ContradictionArbitrator:
    """Создать инстанс арбитра."""
    return ContradictionArbitrator(brain)