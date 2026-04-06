"""Text display utilities for chat messages."""
import logging
from typing import Any

logger = logging.getLogger("eva.gui.chat")


def _to_display_str(val: Any) -> str:
    """Безопасно преобразует значение к строке для отображения в UTF-8."""
    try:
        if isinstance(val, bytes):
            s = val.decode("utf-8", errors="replace")
        else:
            s = str(val)
        return s.replace("\r\n", "\n").replace("\r", "\n")
    except (UnicodeError, TypeError, ValueError, AttributeError) as e:
        logger.debug(f"Error converting value to display string: {e}")
        return str(val)


def _looks_mojibake(s: str) -> bool:
    """Грубая эвристика для детекции mojibake."""
    if not s:
        return False
    bad_chars = set("ÐÑÂÃĤĭĮıİıĝĞġĠ")
    return any(ch in bad_chars for ch in s)


def _fix_mojibake(s: str) -> str:
    """Пытается исправить типичный mojibake (UTF-8, показанный как Latin-1)."""
    try:
        s0 = _to_display_str(s)
        if _looks_mojibake(s0):
            try:
                repaired = s0.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                if repaired and repaired != s0:
                    return repaired
            except (UnicodeError, TypeError, ValueError, AttributeError):
                pass
        return s0
    except (UnicodeError, TypeError, ValueError, AttributeError):
        return _to_display_str(s)
