
import os
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

class KGAdapter:
    """
    Минимальный адаптер графа знаний.
    Ожидает структуру каталога:
      base_dir/
        nodes.jsonl  # по строке на узел: {"id": str, "title": str, "text": str, "tags": [str]}
        edges.jsonl  # по строке на ребро: {"src": str, "dst": str, "type": str}
        index.csv    # упрощённый индекс: id,title,tags
    """

    def __init__(self, base_dir: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.base_dir = base_dir
        self.config = config or {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.tags_index: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        nodes_path = os.path.join(self.base_dir, "nodes.jsonl")
        edges_path = os.path.join(self.base_dir, "edges.jsonl")
        # edges пока не используем, но читаем для консистентности
        if os.path.exists(nodes_path):
            with open(nodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    node_id = str(obj.get("id"))
                    if not node_id:
                        continue
                    self.nodes[node_id] = obj
                    for t in obj.get("tags", []) or []:
                        self.tags_index.setdefault(str(t).lower(), []).append(node_id)
        if os.path.exists(edges_path):
            # можно загрузить для будущего использования
            pass

    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q = (query or "").lower()
        filters = filters or {}
        tag = str(filters.get("tag", "")).lower()
        candidates: List[Tuple[int, str]] = []
        iterable_ids = list(self.nodes.keys())
        if tag and tag in self.tags_index:
            iterable_ids = self.tags_index[tag]
        for nid in iterable_ids:
            n = self.nodes.get(nid) or {}
            text = f"{n.get('title','')}\n{n.get('text','')}".lower()
            score = 0
            if q:
                if q in text:
                    score += 10
                # простейшая эвристика по словам запроса
                for token in q.split():
                    if token and token in text:
                        score += 1
            candidates.append((score, nid))
        candidates.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for _, nid in candidates[: max(1, k)]:
            out.append(self.nodes[nid])
        return out

    def expand_context(self, query: str, current_context: str, task_type: str = "general", k: int = 5, max_chars: int = 2000) -> str:
        chunks = self.retrieve(query, k=k)
        parts: List[str] = []
        for ch in chunks:
            title = ch.get("title", "")
            text = ch.get("text", "")
            parts.append(f"[KG:{ch.get('id')}] {title}\n{text}")
        kg_block = "\n\n".join(parts)
        # Ограничим размер
        if len(kg_block) > max_chars:
            kg_block = kg_block[:max_chars]
        # Соберём итоговый контекст
        ctx = current_context or ""
        if kg_block:
            if ctx:
                ctx = f"{ctx}\n\n# Knowledge Graph Context\n{kg_block}"
            else:
                ctx = f"# Knowledge Graph Context\n{kg_block}"
        return ctx

    def predict(self, query: str) -> Optional[str]:
        # Опциональная генерация из KG (например, шаблоны ответов). Пока None
        return None

    def add_node(self, node: Dict[str, Any]) -> bool:
        """
        Добавляет/обновляет узел в графе знаний и сохраняет его в nodes.jsonl (append-only простая семантика).
        Ожидаемый формат node: {"id": str, "title": str, "text": str, "tags": [str]}
        Возвращает True при успешной записи.
        """
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            return False
        # Нормализуем поля
        obj = {
            "id": node_id,
            "title": str(node.get("title", "")),
            "text": str(node.get("text", "")),
            "tags": [str(t).lower() for t in (node.get("tags") or [])],
        }
        nodes_path = os.path.join(self.base_dir, "nodes.jsonl")
        os.makedirs(self.base_dir, exist_ok=True)
        try:
            with self._lock:
                # Обновляем в памяти
                self.nodes[node_id] = obj
                # Индекс по тегам
                for t in obj.get("tags", []) or []:
                    lst = self.tags_index.setdefault(str(t).lower(), [])
                    if node_id not in lst:
                        lst.append(node_id)
                # Аппенд в файл (простая стратегия без дедупликации на диске)
                with open(nodes_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False

import os
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

class KGAdapter:
    """
    Минимальный адаптер графа знаний.
    Ожидает структуру каталога:
      base_dir/
        nodes.jsonl  # по строке на узел: {"id": str, "title": str, "text": str, "tags": [str]}
        edges.jsonl  # по строке на ребро: {"src": str, "dst": str, "type": str}
        index.csv    # упрощённый индекс: id,title,tags
    """

    def __init__(self, base_dir: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.base_dir = base_dir
        self.config = config or {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.tags_index: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        nodes_path = os.path.join(self.base_dir, "nodes.jsonl")
        edges_path = os.path.join(self.base_dir, "edges.jsonl")
        # edges пока не используем, но читаем для консистентности
        if os.path.exists(nodes_path):
            with open(nodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    node_id = str(obj.get("id"))
                    if not node_id:
                        continue
                    self.nodes[node_id] = obj
                    for t in obj.get("tags", []) or []:
                        self.tags_index.setdefault(str(t).lower(), []).append(node_id)
        if os.path.exists(edges_path):
            # Загружаем рёбра для будущего использования
            with open(edges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        edge = json.loads(line)
                        # Валидация формата ребра
                        if all(key in edge for key in ["src", "dst", "type"]):
                            self.edges.append(edge)
                    except Exception as e:
                        # Логируем ошибку, но продолжаем загрузку
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Ошибка при загрузке ребра: {e}")
                        continue

    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q = (query or "").lower()
        filters = filters or {}
        tag = str(filters.get("tag", "")).lower()
        candidates: List[Tuple[int, str]] = []
        iterable_ids = list(self.nodes.keys())
        if tag and tag in self.tags_index:
            iterable_ids = self.tags_index[tag]
        for nid in iterable_ids:
            n = self.nodes.get(nid) or {}
            text = f"{n.get('title','')}\n{n.get('text','')}".lower()
            score = 0
            if q:
                if q in text:
                    score += 10
                # простейшая эвристика по словам запроса
                for token in q.split():
                    if token and token in text:
                        score += 1
            candidates.append((score, nid))
        candidates.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for _, nid in candidates[: max(1, k)]:
            out.append(self.nodes[nid])
        return out

    def expand_context(self, query: str, current_context: str, task_type: str = "general", k: int = 5, max_chars: int = 2000) -> str:
        chunks = self.retrieve(query, k=k)
        parts: List[str] = []
        for ch in chunks:
            title = ch.get("title", "")
            text = ch.get("text", "")
            parts.append(f"[KG:{ch.get('id')}] {title}\n{text}")
        kg_block = "\n\n".join(parts)
        # Ограничим размер
        if len(kg_block) > max_chars:
            kg_block = kg_block[:max_chars]
        # Соберём итоговый контекст
        ctx = current_context or ""
        if kg_block:
            if ctx:
                ctx = f"{ctx}\n\n# Knowledge Graph Context\n{kg_block}"
            else:
                ctx = f"# Knowledge Graph Context\n{kg_block}"
        return ctx

    def predict(self, query: str) -> Optional[str]:
        """
        Генерирует ответ на основе графа знаний.
        Ищет релевантные узлы и формирует ответ из их содержимого.
        """
        try:
            # Получаем релевантные узлы
            relevant_nodes = self.retrieve(query, k=3)
            
            if not relevant_nodes:
                return None
            
            # Формируем ответ на основе найденных узлов
            response_parts = []
            
            for node in relevant_nodes:
                title = node.get("title", "")
                text = node.get("text", "")
                
                if title and text:
                    response_parts.append(f"{title}: {text}")
                elif text:
                    response_parts.append(text)
            
            if response_parts:
                # Объединяем найденную информацию в связный ответ
                response = "\n\n".join(response_parts)
                # Ограничиваем размер ответа
                if len(response) > 1000:
                    response = response[:1000] + "..."
                return response
            
            return None
            
        except Exception as e:
            # Логируем ошибку, но не прерываем работу
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка в predict() для запроса '{query}': {e}")
            return None

    def add_node(self, node: Dict[str, Any]) -> bool:
        """
        Добавляет/обновляет узел в графе знаний и сохраняет его в nodes.jsonl (append-only простая семантика).
        Ожидаемый формат node: {"id": str, "title": str, "text": str, "tags": [str]}
        Возвращает True при успешной записи.
        """
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            return False
        # Нормализуем поля
        obj = {
            "id": node_id,
            "title": str(node.get("title", "")),
            "text": str(node.get("text", "")),
            "tags": [str(t).lower() for t in (node.get("tags") or [])],
        }
        nodes_path = os.path.join(self.base_dir, "nodes.jsonl")
        os.makedirs(self.base_dir, exist_ok=True)
        try:
            with self._lock:
                # Обновляем в памяти
                self.nodes[node_id] = obj
                # Индекс по тегам
                for t in obj.get("tags", []) or []:
                    lst = self.tags_index.setdefault(str(t).lower(), [])
                    if node_id not in lst:
                        lst.append(node_id)
                # Аппенд в файл (простая стратегия без дедупликации на диске)
                with open(nodes_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False
