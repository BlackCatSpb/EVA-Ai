"""
ConceptMiner adapter - импортирует из knowledge_old для обратной совместимости
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
old_dir = os.path.join(os.path.dirname(current_dir), 'knowledge_old')

if old_dir not in sys.path:
    sys.path.insert(0, old_dir)

from concept_miner import ConceptMiner, PhantomCandidate, create_concept_miner

__all__ = ['ConceptMiner', 'PhantomCandidate', 'create_concept_miner']
