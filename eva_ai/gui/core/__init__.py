"""
Core GUI components for ЕВА
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Import the main GUI class from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core_gui import ЕВАGUI, create_gui

logger = logging.getLogger("eva_ai.gui.core")

# Re-export the main GUI class and function
__all__ = ['ЕВАGUI', 'create_gui']
