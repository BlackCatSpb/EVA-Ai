"""
Import Pipeline for TXT, PDF, and EPUB with optional OCR fallback.
Provides normalized, chunked iterable for self-dialog learning.
"""
from __future__ import annotations

import os
import io
import re
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional
import logging

logger = logging.getLogger("eva.import.pipeline")


def _safe_import(module: str):
    try:
        return __import__(module)
    except Exception:
        return None


ebooklib = _safe_import("ebooklib")
pdfminer = _safe_import("pdfminer")
pypdf = _safe_import("pypdf") or _safe_import("PyPDF2")
pytesseract = _safe_import("pytesseract")
PIL = _safe_import("PIL")

# Configure Tesseract path for OCR
if pytesseract:
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        logger.info("Tesseract path configured in import_pipeline")
    except Exception as e:
        logger.warning(f"Failed to configure Tesseract path: {e}")


@dataclass
class ImportedDocument:
    id: str
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    _segments: Optional[List[str]] = None

    def iter_segments(self) -> Iterable[str]:
        if self._segments is None:
            return []
        return iter(self._segments)


class ImportPipeline:
    """
    High-level pipeline:
    - Load file
    - Normalize with UnifiedTextProcessor if available
    - Chunk with overlap for stable context windows
    """

    def __init__(
        self,
        brain: Any,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        max_doc_chars: int = 2_000_000,
    ) -> None:
        self.brain = brain
        self.chunk_tokens = max(64, chunk_tokens)
        self.overlap_tokens = max(0, overlap_tokens)
        self.max_doc_chars = max_doc_chars

        # Optional: use brain text processor for normalization and tokenization length
        ml_unit = getattr(brain, "ml_unit", None)
        self.text_processor = getattr(ml_unit, "text_processor", None) if ml_unit else None
        # Optional tokenizer length estimator
        self.token_streamer = getattr(ml_unit, "token_streamer", None) if ml_unit else None

    # ----------------------------
    # Public API
    # ----------------------------
    def import_path(self, path: str, doc_id: Optional[str] = None) -> ImportedDocument:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ext = os.path.splitext(path)[1].lower()
        text = ""
        if ext in (".txt", ".md", ".log"):
            text = self._read_txt(path)
        elif ext in (".pdf",):
            text = self._read_pdf(path)
        elif ext in (".epub",):
            text = self._read_epub(path)
        else:
            # Fallback: try reading as text
            try:
                text = self._read_txt(path)
            except Exception:
                text = ""

        text = self._normalize_text(text)[: self.max_doc_chars]
        segments = self._chunk_text(text)

        return ImportedDocument(
            id=doc_id or self._derive_id(path),
            source_path=path,
            metadata={"ext": ext, "size": os.path.getsize(path)},
            _segments=segments,
        )

    # ----------------------------
    # Readers
    # ----------------------------
    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        # Prefer pdfminer if available for better layout handling
        try:
            if pdfminer:
                from pdfminer.high_level import extract_text
                return extract_text(path) or ""
        except Exception as e:
            logger.debug(f"pdfminer failed: {e}")
        # Fallback to pypdf
        try:
            if pypdf:
                reader = pypdf.PdfReader(path)
                pages = []
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                return "\n".join(pages)
        except Exception as e:
            logger.debug(f"pypdf failed: {e}")
        # Optional OCR fallback for scanned PDFs
        try:
            if pytesseract and PIL:
                from pdf2image import convert_from_path  # may not be installed
                images = convert_from_path(path)
                texts = [pytesseract.image_to_string(img) for img in images]
                return "\n".join(texts)
        except Exception as e:
            logger.debug(f"OCR fallback failed: {e}")
        return ""

    def _read_epub(self, path: str) -> str:
        if not ebooklib:
            return ""
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup  # optional but common
        except Exception:
            return ""
        try:
            book = epub.read_epub(path)
            texts: List[str] = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    try:
                        soup = BeautifulSoup(item.get_content(), "html.parser")
                        texts.append(soup.get_text(" "))
                    except Exception:
                        pass
            return "\n".join(texts)
        except Exception as e:
            logger.debug(f"EPUB read failed: {e}")
            return ""

    # ----------------------------
    # Normalization & Chunking
    # ----------------------------
    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        try:
            if self.text_processor and hasattr(self.text_processor, "normalize_text"):
                return self.text_processor.normalize_text(text)
        except Exception:
            pass
        # Simple default normalization
        text = re.sub(r"\r\n?|\u00A0", "\n", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        # If text_processor provides advanced segmenter
        try:
            if self.text_processor and hasattr(self.text_processor, "segment_text"):
                segs = self.text_processor.segment_text(text, target_tokens=self.chunk_tokens, overlap=self.overlap_tokens)
                if isinstance(segs, list) and segs:
                    return [s for s in segs if isinstance(s, str) and s.strip()]
        except Exception as e:
            logger.debug(f"segment_text failed: {e}")
        # Simple fallback: sentence-ish splitting and greedy packing by approximate token length
        sentences = re.split(r"(?<=[.!?])\s+", text)
        segments: List[str] = []
        buf: List[str] = []
        cur_len = 0
        for sent in sentences:
            tok_len = self._approx_tokens(sent)
            if cur_len + tok_len + (self.overlap_tokens if buf else 0) > self.chunk_tokens and buf:
                seg = " ".join(buf)
                segments.append(seg)
                # overlap by last ~overlap fraction
                if self.overlap_tokens > 0:
                    tail = seg.split()[-self.overlap_tokens :]
                    buf = [" ".join(tail)]
                    cur_len = len(tail)
                else:
                    buf, cur_len = [], 0
            buf.append(sent)
            cur_len += tok_len
        if buf:
            segments.append(" ".join(buf))
        return segments

    def _approx_tokens(self, s: str) -> int:
        # If we have token_streamer with an estimator use it
        try:
            if self.token_streamer and hasattr(self.token_streamer, "estimate_token_count"):
                return int(self.token_streamer.estimate_token_count(s))
        except Exception:
            pass
        # rough heuristic: 1 token ~ 0.75 words for typical BPE
        return max(1, int(len(s.split()) / 0.75))

    # ----------------------------
    # Utils
    # ----------------------------
    def _derive_id(self, path: str) -> str:
        base = os.path.basename(path)
        return f"doc::{base}::{hashlib.md5(path.encode('utf-8')).hexdigest()[:8]}"
