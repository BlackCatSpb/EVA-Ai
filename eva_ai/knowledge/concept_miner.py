"""
ConceptMiner adapter - импортирует из deprecated_modules/knowledge_old для обратной совместимости
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
old_dir = os.path.join(project_root, 'deprecated_modules', 'knowledge_old')

if old_dir not in sys.path:
    sys.path.insert(0, old_dir)

from concept_miner import ConceptMiner, PhantomCandidate, create_concept_miner

__all__ = ['ConceptMiner', 'PhantomCandidate', 'create_concept_miner']
