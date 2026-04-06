"""
Document Text Reader Module for ЕВА
Чтение текстовых файлов для отображения в чате
"""
import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("eva.document_reader")

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.log', '.json', '.xml', '.csv', '.yaml', '.yml'}


@dataclass
class DocumentContent:
    """Содержимое документа"""
    filename: str
    filepath: str
    content: str
    lines: List[str]
    metadata: Dict[str, Any]


class DocumentTextReader:
    """
    Читает текстовые файлы и возвращает содержимое для отображения в чате.
    Поддерживает: .txt, .md, .log, .json, .xml, .csv, .yaml, .yml
    """
    
    def __init__(self, max_chars: int = 100000):
        self.max_chars = max_chars
    
    def read(self, filepath: str) -> Optional[DocumentContent]:
        """
        Читает файл и возвращает содержимое.
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            DocumentContent или None при ошибке
        """
        if not os.path.exists(filepath):
            logger.error(f"Файл не найден: {filepath}")
            return None
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Неподдерживаемый формат: {ext}")
            return None
        
        try:
            return self._read_file(filepath, ext)
        except Exception as e:
            logger.error(f"Ошибка чтения файла {filepath}: {e}")
            return None
    
    def _read_file(self, filepath: str, ext: str) -> DocumentContent:
        """Внутренний метод чтения файла."""
        filename = os.path.basename(filepath)
        
        # Выбор encoding
        encoding = self._detect_encoding(filepath)
        
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # Ограничение по размеру
        if len(content) > self.max_chars:
            content = content[:self.max_chars] + f"\n\n[Файл обрезан. Показано первых {self.max_chars} символов из {len(content)}]"
        
        lines = content.split('\n')
        
        metadata = {
            'size': os.path.getsize(filepath),
            'extension': ext,
            'encoding': encoding,
            'lines': len(lines),
            'chars': len(content)
        }
        
        return DocumentContent(
            filename=filename,
            filepath=filepath,
            content=content,
            lines=lines,
            metadata=metadata
        )
    
    def _detect_encoding(self, filepath: str) -> str:
        """Определяет кодировку файла."""
        # Пробуем разные кодировки
        for encoding in ['utf-8', 'utf-8-sig', 'cp1251', 'koi8-r', 'iso-8859-5']:
            try:
                with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                    f.read(1024)
                return encoding
            except Exception:
                continue
        return 'utf-8'
    
    def read_as_messages(self, filepath: str, max_lines: int = 100) -> List[Dict[str, str]]:
        """
        Читает файл и возвращает список сообщений для чата.
        
        Args:
            filepath: Путь к файлу
            max_lines: Максимальное количество строк для отображения
            
        Returns:
            Список сообщений для добавления в чат
        """
        doc = self.read(filepath)
        
        if not doc:
            return [{"sender": "ЕВА", "text": "Не удалось прочитать файл", "type": "error"}]
        
        messages = []
        
        # Заголовок
        messages.append({
            "sender": "ЕВА",
            "text": f"[FILE] {doc.filename}",
            "type": "system"
        })
        
        # Метаинформация
        meta = getattr(doc, 'metadata', {}) or {}
        meta_text = f"Строк: {meta.get('lines', 'N/A')}, Символов: {meta.get('chars', 'N/A')}, Кодировка: {meta.get('encoding', 'N/A')}"
        messages.append({
            "sender": "ЕВА", 
            "text": meta_text,
            "type": "system"
        })
        
        # Содержимое (ограниченное)
        display_lines = doc.lines[:max_lines]
        content_text = '\n'.join(display_lines)
        
        if len(doc.lines) > max_lines:
            content_text += f"\n\n[...] Показано {max_lines} из {len(doc.lines)} строк"
        
        messages.append({
            "sender": "ЕВА",
            "text": f"```\n{content_text}\n```",
            "type": "assistant"
        })
        
        return messages


def read_text_file_simple(filepath: str) -> Optional[str]:
    """
    Простая функция для чтения текстового файла.
    
    Args:
        filepath: Путь к файлу
        
    Returns:
        Содержимое файла или None
    """
    if not os.path.exists(filepath):
        return None
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ошибка чтения {filepath}: {e}")
        return None
