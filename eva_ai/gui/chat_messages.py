"""Message display, rendering, formatting, and timestamps for the chat module."""
import tkinter as tk
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from eva_ai.gui.chat_text_utils import _to_display_str, _fix_mojibake

logger = logging.getLogger("eva_ai.gui.chat")

CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\r?\n(.*?)```', re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')
MATH_INLINE_PATTERN = re.compile(r'\$([^\$]+)\$')
MATH_BLOCK_PATTERN = re.compile(r'\$\$([^\$]+)\$\$', re.DOTALL)

PYTHON_KEYWORDS = {'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 
                   'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'raise',
                   'pass', 'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'True',
                   'False', 'None', 'lambda', 'assert', 'del', 'global', 'nonlocal',
                   'async', 'await', 'print', 'range', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple'}

STRING_PATTERN = re.compile(r'("[^"]*"|\'[^\']*\')')
COMMENT_PATTERN = re.compile(r'#.*$', re.MULTILINE)
NUMBER_PATTERN = re.compile(r'\b\d+\.?\d*\b')


class ChatMessagesMixin:
    """Mixin providing message display, rendering, formatting, and timestamp functionality."""

    def _add_message(self, sender: str, message: str, msg_type: str = "user",
                     timestamp: Optional[float] = None,
                     process_formatting: bool = True,
                     extras: Optional[Dict[str, Any]] = None):
        """Добавляет сообщение в чат."""
        import json
        if timestamp is None:
            timestamp = time.time()

        # Сохранение в историю
        if not getattr(self, '_suppress_history_append', False):
            entry = {
                "sender": sender,
                "message": message,
                "type": msg_type,
                "timestamp": timestamp
            }

            if extras and isinstance(extras, dict):
                try:
                    safe_extras = json.loads(json.dumps(extras, ensure_ascii=False, default=str))
                except Exception as e:
                    logger.debug(f"Error serializing extras: {e}")
                    safe_extras = {}
                for k, v in extras.items():
                    try:
                        safe_extras[k] = json.loads(json.dumps(v, ensure_ascii=False, default=str))
                    except Exception as e:
                        logger.debug(f"Error serializing extra key {k}: {e}")
                        safe_extras[k] = str(v)
                entry.update({"extras": safe_extras})

            with self._history_lock:
                self.message_history.append(entry)
                self._save_history_incremental()

                # Ограничение размера истории
                if len(self.message_history) > 500:
                    entry = self.message_history[0]
                    if isinstance(entry.get("extras"), dict):
                        large_keys = [k for k, v in entry["extras"].items()
                                      if isinstance(v, (list, dict)) and len(str(v)) > 1000]
                        for k in large_keys:
                            entry["extras"][k] = f"<truncated {len(entry['extras'][k])} items>"
                    self.message_history = self.message_history[-500:]

        # Отображение
        try:
            if not self.chat_display.winfo_exists():
                return

            self.chat_display.config(state=tk.NORMAL)

            # Новая строка
            try:
                last_char = self.chat_display.get("end-2c", "end-1c")
                if last_char not in ("\n", ""):
                    self.chat_display.insert(tk.END, "\n")
            except Exception as e:
                logger.debug(f"Error adding new line: {e}")

            # Временная метка
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            self.chat_display.insert(tk.END, f"[{time_str}] ", "timestamp")

            # Отправитель
            tag = "user" if msg_type == "user" else msg_type
            self.chat_display.insert(tk.END, f"{sender}: ", tag)

            # Сообщение
            if process_formatting:
                self._process_and_insert_formatted_message(_to_display_str(message))
            else:
                self.chat_display.insert(tk.END, _to_display_str(message))

            self.chat_display.insert(tk.END, "\n")
            self.chat_display.config(state=tk.DISABLED)

            # Force widget to update/refresh to prevent display artifacts
            self.chat_display.update_idletasks()
            self.chat_display.see(tk.END)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка добавления сообщения в чат: {e}", exc_info=True)

    def _process_and_insert_formatted_message(self, message: str):
        """Обрабатывает и вставляет форматированное сообщение."""
        try:
            if not self.chat_display.winfo_exists():
                return

            # Предварительная обработка Markdown элементов
            message = self._preprocess_markdown(message)

            start_index = self.chat_display.index(tk.END)
            self.chat_display.insert(tk.END, message)

            # Форматирование: bold, italic, code
            current_pos = 0
            
            # Сначала обрабатываем inline code отдельно чтобы не мешалось с bold/italic
            inline_code_pat = re.compile(r'`([^`]+)`')
            code_matches = list(inline_code_pat.finditer(message))
            
            for match in code_matches:
                start_idx = match.start()
                end_idx = match.end()
                matched_text = match.group(0)
                code_content = match.group(1)
                
                if not code_content:
                    continue
                
                start_tag_pos = f"{start_index}+{start_idx + current_pos}c"
                end_tag_pos = f"{start_index}+{start_idx + current_pos + len(matched_text)}c"
                
                # Замена
                self.chat_display.delete(start_tag_pos, end_tag_pos)
                self.chat_display.insert(start_tag_pos, code_content)
                self.chat_display.tag_add("code", start_tag_pos, f"{start_tag_pos}+{len(code_content)}c")
                
                current_pos -= len(matched_text) - len(code_content)
            
            # Пересчитываем позиции для bold/italic
            current_pos = 0
            for match in self.formatting_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()
                matched_text = match.group(0)

                if matched_text.startswith("**") or matched_text.startswith("__"):
                    format_type = "bold"
                    content = match.group(1) or match.group(2)
                elif matched_text.startswith("_") and not matched_text.startswith("__"):
                    format_type = "italic"
                    content = match.group(3)
                else:
                    continue

                if not content:
                    continue

                # Позиции
                start_tag_pos = f"{start_index}+{start_idx + current_pos}c"
                end_tag_pos = f"{start_index}+{start_idx + current_pos + len(matched_text)}c"
                content_start = f"{start_index}+{start_idx + current_pos + 2}c"
                content_end = f"{start_index}+{start_idx + current_pos + len(content) + 2}c"

                # Замена
                self.chat_display.delete(start_tag_pos, end_tag_pos)
                self.chat_display.insert(start_tag_pos, content)
                self.chat_display.tag_add(format_type, content_start, content_end)

                current_pos -= len(matched_text) - len(content)

            # Гиперссылки
            self._process_markdown_links(message, start_index)
            self._process_urls(message, start_index)
            self._process_emojis(message, start_index)
            self._process_images(message, start_index)
            
            # Комментарии в коде §...§
            self._process_comments(message, start_index)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки форматирования сообщения: {e}", exc_info=True)

    def _preprocess_markdown(self, message: str) -> str:
        """
        Полная предобработка Markdown элементов: заголовки, списки, разделители, код, математика.
        
        Converts:
        - ### Header → **Header** (bold заголовок)
        - - item → • item
        - 🔹 item → • item  
        - --- → ─── (разделитель)
        - $math$ → Unicode math
        - ```code``` → подсветка синтаксиса
        """
        # Сохраняем code blocks и math чтобы не обрабатывать их как обычный текст
        code_blocks: List[Tuple[str, str]] = []
        math_blocks: List[Tuple[str, str]] = []
        
        def replace_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            placeholder = f"\n████CODE_BLOCK_{len(code_blocks)}████\n"
            code_blocks.append((lang, code))
            return placeholder
        
        def replace_math_block(match):
            math_expr = match.group(1)
            math_blocks.append(math_expr)
            return f"\n████MATH_BLOCK_{len(math_blocks) - 1}████\n"
        
        # Заменяем code blocks и math blocks на плейсхолдеры
        message = CODE_BLOCK_PATTERN.sub(replace_code_block, message)
        message = MATH_BLOCK_PATTERN.sub(replace_math_block, message)
        message = MATH_INLINE_PATTERN.sub(replace_math_block, message)
        
        lines = message.split('\n')
        processed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Заголовки ### Header → **Header**
            if stripped.startswith('### '):
                header_text = stripped[4:]
                processed_lines.append(f"\n**{header_text}**")
                i += 1
            
            # Заголовки ## Header → **Header**
            elif stripped.startswith('## '):
                header_text = stripped[3:]
                processed_lines.append(f"\n**{header_text}**")
                i += 1
            
            # Заголовки # Header → **Header**
            elif stripped.startswith('# '):
                header_text = stripped[2:]
                processed_lines.append(f"\n**{header_text}**")
                i += 1
            
            # Блок-цитата > text
            elif stripped.startswith('>'):
                quote_text = stripped[1:].strip()
                processed_lines.append(f"│ {quote_text}")
                i += 1
            
            # Списки - item → • item
            elif stripped.startswith('- ') or stripped.startswith('* '):
                item_text = stripped[2:]
                processed_lines.append(f"  • {item_text}")
                i += 1
            
            # Чекбоксы - [ ] или [x]
            elif re.match(r'^\- \[ \]', stripped):
                processed_lines.append(f"  ○ {stripped[4:]}")
                i += 1
            elif re.match(r'^\- \[x\]', stripped, re.IGNORECASE):
                processed_lines.append(f"  ◉ {stripped[4:]}")
                i += 1
            
            # Нумерованные списки 1. item → • item
            elif re.match(r'^\d+\.\s+', stripped):
                match = re.match(r'^(\d+)\.\s+(.*)$', stripped)
                if match:
                    processed_lines.append(f"  • {match.group(2)}")
                i += 1
            
            # Эмодзи буллиты 🔹, 🔸 и т.д.
            elif re.match(r'^[\U0001F537-\U0001F93A]\s+', stripped):
                item_text = re.sub(r'^[\U0001F537-\U0001F93A]\s+', '', stripped)
                processed_lines.append(f"  • {item_text}")
                i += 1
            
            # Разделители --- или *** → ───
            elif re.match(r'^[\-\*]{3,}$', stripped):
                processed_lines.append('─' * 30)
                i += 1
            
            # Таблицы - простая обработка
            elif stripped.startswith('|') and stripped.endswith('|'):
                # Пропускаем разделители таблиц
                if re.match(r'^\|[\s\-\|:]+\|$', stripped):
                    i += 1
                    continue
                # Обрабатываем строку таблицы
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                table_row = "│ " + " │ ".join(cells) + " │"
                processed_lines.append(table_row)
                i += 1
            
            else:
                processed_lines.append(line)
                i += 1
        
        result = '\n'.join(processed_lines)
        
        # Восстанавливаем math blocks с Unicode math
        for idx, math_expr in enumerate(math_blocks):
            unicode_math = self._convert_math_to_unicode(math_expr)
            result = result.replace(f"████MATH_BLOCK_{idx}████", unicode_math)
        
        # Восстанавливаем code blocks с подсветкой синтаксиса
        for idx, (lang, code) in enumerate(code_blocks):
            highlighted = self._highlight_syntax(code, lang)
            result = result.replace(f"████CODE_BLOCK_{idx}████", f"\n{highlighted}\n")
        
        return result
    
    def _convert_math_to_unicode(self, expr: str) -> str:
        """Конвертирует математическое выражение в Unicode символы."""
        math_map = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
            'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
            'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π', 'rho': 'ρ',
            'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ', 'chi': 'χ',
            'psi': 'ψ', 'omega': 'ω',
            'Alpha': 'Α', 'Beta': 'Β', 'Gamma': 'Γ', 'Delta': 'Δ', 'Epsilon': 'Ε',
            'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ',
            'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
            'infty': '∞', 'infinity': '∞',
            'pm': '±', 'mp': '∓', 'times': '×', 'div': '÷', 'cdot': '·',
            'leq': '≤', 'geq': '≥', 'neq': '≠', 'approx': '≈', 'equiv': '≡',
            'cong': '≅', 'sim': '∼', 'propto': '∝', 'll': '≪', 'gg': '≫',
            'subset': '⊂', 'supset': '⊃', 'subseteq': '⊆', 'supseteq': '⊇',
            'cup': '∪', 'cap': '∩', 'emptyset': '∅', 'in': '∈', 'notin': '∉',
            'forall': '∀', 'exists': '∃', 'partial': '∂', 'nabla': '∇',
            'sum': '∑', 'prod': '∏', 'integral': '∫',
            'rightarrow': '→', 'leftarrow': '←', 'uparrow': '↑', 'downarrow': '↓',
            'Rightarrow': '⇒', 'Leftarrow': '⇐', 'leftrightarrow': '↔',
            'sqrt': '√', 'cbrt': '∛',
        }
        
        result = expr
        for tex, uni in math_map.items():
            result = re.sub(rf'\\{tex}(?![a-zA-Z])', uni, result)
        
        # Степени и индексы
        result = re.sub(r'\^2', '²', result)
        result = re.sub(r'\^3', '³', result)
        result = re.sub(r'\^(\d+)', r'^\1', result)
        result = re.sub(r'_(\d+)', r'_\1', result)
        result = re.sub(r'\{([^}]+)\}', r'\1', result)
        
        return f"[{result}]"
    
    def _highlight_syntax(self, code: str, language: str = "") -> str:
        """
        Подсветка синтаксиса кода.
        
        Returns formatted string with ASCII art highlighting for terminal display.
        """
        if not code.strip():
            return code
        
        # Определяем язык если не указан
        lang_lower = language.lower()
        
        # Для Python
        if 'python' in lang_lower or not lang_lower:
            highlighted = self._highlight_python(code)
        # Для JavaScript/TypeScript
        elif 'javascript' in lang_lower or 'js' in lang_lower or 'typescript' in lang_lower or 'ts' in lang_lower:
            highlighted = self._highlight_js(code)
        # Для C/C++
        elif 'c' in lang_lower or 'cpp' in lang_lower or 'c++' in lang_lower:
            highlighted = self._highlight_c(code)
        # Для HTML
        elif 'html' in lang_lower:
            highlighted = self._highlight_html(code)
        # Для SQL
        elif 'sql' in lang_lower:
            highlighted = self._highlight_sql(code)
        # Для JSON
        elif 'json' in lang_lower:
            highlighted = self._highlight_json(code)
        else:
            # Generic formatting
            highlighted = self._generic_highlight(code)
        
        return highlighted
    
    def _highlight_python(self, code: str) -> str:
        """Подсветка Python кода."""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Комментарии
            if '#' in line:
                parts = line.split('#', 1)
                code_part = parts[0]
                comment_part = '#' + parts[1]
                result_lines.append(code_part + f" §{comment_part}§")
            else:
                result_lines.append(line)
        
        # Формируем блок с рамкой
        result = '\n'.join(result_lines)
        return f"┌─Python────────\n{result}\n└────────────────"
    
    def _highlight_js(self, code: str) -> str:
        """Подсветка JavaScript/TypeScript кода."""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Комментарии //
            if '//' in line:
                parts = line.split('//', 1)
                code_part = parts[0]
                comment_part = '//' + parts[1]
                result_lines.append(code_part + f" §{comment_part}§")
            else:
                result_lines.append(line)
        
        result = '\n'.join(result_lines)
        return f"┌─JavaScript────\n{result}\n└────────────────"
    
    def _highlight_c(self, code: str) -> str:
        """Подсветка C/C++ кода."""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Комментарии //
            if '//' in line:
                parts = line.split('//', 1)
                code_part = parts[0]
                comment_part = '//' + parts[1]
                result_lines.append(code_part + f" §{comment_part}§")
            else:
                result_lines.append(line)
        
        result = '\n'.join(result_lines)
        return f"┌─C/C++─────────\n{result}\n└────────────────"
    
    def _highlight_html(self, code: str) -> str:
        """Подсветка HTML кода."""
        result = re.sub(r'<(\w+)"', r'<\1', code)
        return f"┌─HTML──────────\n{result}\n└────────────────"
    
    def _highlight_sql(self, code: str) -> str:
        """Подсветка SQL кода."""
        keywords = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
                   'ON', 'AND', 'OR', 'NOT', 'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET',
                   'DELETE', 'CREATE', 'TABLE', 'DROP', 'ALTER', 'INDEX', 'ORDER', 'BY',
                   'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'AS', 'DISTINCT', 'COUNT', 'SUM',
                   'AVG', 'MAX', 'MIN', 'NULL', 'IS', 'LIKE', 'IN', 'BETWEEN', 'CASE', 'WHEN',
                   'THEN', 'END', 'UNION', 'ALL', 'EXISTS', 'PRIMARY', 'KEY', 'FOREIGN'}
        
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Комментарии --
            if '--' in line:
                parts = line.split('--', 1)
                result_lines.append(parts[0] + f" §--{parts[1]}§")
            else:
                result_lines.append(line)
        
        result = '\n'.join(result_lines)
        return f"┌─SQL───────────\n{result}\n└────────────────"
    
    def _highlight_json(self, code: str) -> str:
        """Подсветка JSON кода."""
        return f"┌─JSON──────────\n{code}\n└────────────────"
    
    def _generic_highlight(self, code: str) -> str:
        """通用ная подсветка для других языков."""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Удаляем комментарии # и //
            if '#' in line:
                parts = line.split('#', 1)
                result_lines.append(parts[0] + f" §#{parts[1]}§")
            elif '//' in line:
                parts = line.split('//', 1)
                result_lines.append(parts[0] + f" §//{parts[1]}§")
            else:
                result_lines.append(line)
        
        result = '\n'.join(result_lines)
        return f"┌─Code──────────\n{result}\n└────────────────"

    def _process_urls(self, message: str, start_index: str):
        """Обрабатывает URL в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return

            for match in self.url_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()

                url_start = f"{start_index}+{start_idx}c"
                url_end = f"{start_index}+{end_idx}c"

                self.chat_display.tag_add("url", url_start, url_end)
                self.chat_display.tag_add(f"url_{match.group(0)}", url_start, url_end)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки URL: {e}", exc_info=True)

    def _process_markdown_links(self, message: str, start_index: str):
        """Обрабатывает markdown-ссылки [текст](url)."""
        try:
            if not self.chat_display.winfo_exists():
                return

            offset_correction = 0
            for match in self.markdown_link_pattern.finditer(message):
                link_text = match.group(1)
                url = match.group(2)
                md_start = match.start()
                md_end = match.end()

                start_pos = f"{start_index}+{md_start + offset_correction}c"
                end_pos = f"{start_index}+{md_end + offset_correction}c"

                self.chat_display.delete(start_pos, end_pos)
                self.chat_display.insert(start_pos, link_text)

                link_end_pos = f"{start_pos}+{len(link_text)}c"
                self.chat_display.tag_add("url", start_pos, link_end_pos)
                self.chat_display.tag_add(f"url_{url}", start_pos, link_end_pos)

                offset_correction -= (md_end - md_start) - len(link_text)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки markdown-ссылок: {e}", exc_info=True)

    def _process_emojis(self, message: str, start_index: str):
        """Обрабатывает эмодзи в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return

            for match in self.emoji_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()

                emoji_start = f"{start_index}+{start_idx}c"
                emoji_end = f"{start_index}+{end_idx}c"

                self.chat_display.tag_add("emoji", emoji_start, emoji_end)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки эмодзи: {e}", exc_info=True)

    def _process_images(self, message: str, start_index: str):
        """Обрабатывает изображения в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return

            for match in self.image_pattern.finditer(message):
                url = match.group(1)
                start_idx = match.start(1)
                end_idx = match.end(1)

                url_start = f"{start_index}+{start_idx}c"
                url_end = f"{start_index}+{end_idx}c"

                self.chat_display.delete(url_start, url_end)
                self.chat_display.insert(url_start, "[Изображение]")

                self.chat_display.tag_add("url", url_start, f"{url_start}+10c")
                self.chat_display.tag_bind("url", "<Button-1>",
                    lambda e, u=url: self._open_image(u))

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки изображений: {e}", exc_info=True)

    def _process_comments(self, message: str, start_index: str):
        """Обрабатывает комментарии в коде §...§."""
        try:
            if not self.chat_display.winfo_exists():
                return

            comment_pattern = re.compile(r'§([^§]+)§')
            offset_correction = 0
            
            for match in comment_pattern.finditer(message):
                comment_text = match.group(1)
                md_start = match.start()
                md_end = match.end()

                start_pos = f"{start_index}+{md_start + offset_correction}c"
                end_pos = f"{start_index}+{md_end + offset_correction}c"

                self.chat_display.delete(start_pos, end_pos)
                self.chat_display.insert(start_pos, comment_text)

                comment_end_pos = f"{start_pos}+{len(comment_text)}c"
                self.chat_display.tag_add("comment", start_pos, comment_end_pos)

                offset_correction -= (md_end - md_start) - len(comment_text)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки комментариев: {e}", exc_info=True)

    def _configure_chat_tags(self):
        """Настраивает стили тегов для сообщений."""
        self.chat_display.tag_configure("user",
            foreground=self.gui.colors['primary'],
            font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("system",
            foreground=self.gui.colors['text'],
            font=('Segoe UI', 10))
        self.chat_display.tag_configure("reasoning",
            foreground=self.gui.colors['text-muted'],
            font=('Segoe UI', 10, 'italic'))
        self.chat_display.tag_configure("timestamp",
            foreground=self.gui.colors['text-muted'],
            font=('Segoe UI', 8))
        self.chat_display.tag_configure("url",
            foreground=self.gui.colors['primary'],
            underline=True)
        self.chat_display.tag_configure("bold", font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("italic", font=('Segoe UI', 10, 'italic'))
        self.chat_display.tag_configure("code",
            background=self.gui.colors['bg'],
            font=('Consolas', 9))
        self.chat_display.tag_configure("emoji", font=('Segoe UI Emoji', 10))
        self.chat_display.tag_configure("heading1",
            font=('Segoe UI', 14, 'bold'),
            foreground=self.gui.colors['primary'])
        self.chat_display.tag_configure("heading2",
            font=('Segoe UI', 12, 'bold'),
            foreground=self.gui.colors['primary'])
        self.chat_display.tag_configure("heading3",
            font=('Segoe UI', 11, 'bold'),
            foreground=self.gui.colors['text'])
        self.chat_display.tag_configure("list_item",
            font=('Segoe UI', 10))
        self.chat_display.tag_configure("separator",
            foreground=self.gui.colors['text-muted'])
        self.chat_display.tag_configure("comment",
            foreground='#6B7280',
            font=('Segoe UI', 9, 'italic'))
        self.chat_display.tag_configure("code_frame",
            foreground=self.gui.colors['text-muted'],
            font=('Segoe UI', 8))

    def _remove_last_message(self):
        """Удаляет последнее сообщение из чата."""
        try:
            if not self.chat_display.winfo_exists():
                return

            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end")
            self.chat_display.config(state=tk.DISABLED)

            with self._history_lock:
                if self.message_history:
                    self.message_history.pop()

        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка удаления последнего сообщения: {e}", exc_info=True)

    def _show_welcome_message(self):
        """Показывает приветственное сообщение."""
        welcome_msg = (
            "Добро пожаловать в ЕВА!\n"
            "Я - когнитивная система с поддержкой:\n"
            "• Этического анализа\n"
            "• Адаптации под пользователя\n"
            "• Распределенных вычислений\n"
            "• Управления знаниями\n"
            "Задайте ваш первый вопрос или нажмите F1 для просмотра справки."
        )
        self._add_message("ЕВА", welcome_msg, "system")

    def _redraw_chat(self):
        """Перерисовывает чат с новой темой."""
        if not self.message_history:
            return

        try:
            if not self.chat_display.winfo_exists():
                return

            current_pos = self.chat_display.yview()

            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)

            for msg in self.message_history:
                self._add_message(
                    msg["sender"],
                    msg["message"],
                    msg["type"],
                    timestamp=msg["timestamp"],
                    process_formatting=False
                )

            self.chat_display.yview_moveto(current_pos[0])
            self.chat_display.config(state=tk.DISABLED)

        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка перерисовки чата: {e}", exc_info=True)
