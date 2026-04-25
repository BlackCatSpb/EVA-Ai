"""
FMF Full Implementation - Все компоненты согласно FMF.txt

Реализованные компоненты:
1. HybridTokenizer - Aho-Corasick + виртуальные токены
2. DocumentKnowledgeLoader - временные сессии
3. BatchPipeline - чанкинг + батчи + агрегация
4. FractalGraphV2 с временными сессиями
5. LoRAManager
"""

import time
import json
import sqlite3
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("FMF_Full")


# === 1. Aho-Corasick Automaton ===
class AhoCorasickAutomaton:
    """Aho-Corasick для поиска подстрок"""
    
    def __init__(self):
        self.next = [{}]
        self.fail = [0]
        self.output = [[]]
    
    def add_word(self, word: str, token_id: int):
        """Добавить слово"""
        node = 0
        for char in word:
            if char not in self.next[node]:
                self.next[node][char] = len(self.next)
                self.next.append({})
                self.fail.append(0)
                self.output.append([])
            node = self.next[node][char]
        self.output[node].append(token_id)
    
    def make_automaton(self):
        """Построить автомат"""
        from collections import deque
        queue = deque([0])
        
        while queue:
            v = queue.popleft()
            for char, u in self.next[v].items():
                queue.append(u)
                if v != 0:
                    self.fail[u] = self.next[self.fail[v]].get(char, 0)
                    self.output[u].extend(self.output[self.fail[u]])
    
    def search(self, text: str) -> List[tuple]:
        """Поиск всех вхождений"""
        results = []
        node = 0
        
        for i, char in enumerate(text):
            while char not in self.next[node] and node != 0:
                node = self.fail[node]
            node = self.next[node].get(char, 0)
            
            for token_id in self.output[node]:
                results.append((i, token_id))
        
        return results


# === 2. HybridTokenizer ===
class HybridTokenizer:
    """
    Гибридный токенизатор
    
    Методы:
    - encode(text) -> токены + эмбеддинги
    - add_temporary_nodes(session_id, graph)
    - remove_temporary_nodes(session_id)
    - decode(token_ids) -> текст
    """
    VIRTUAL_TOKEN_OFFSET = 151936
    
    def __init__(self, base_tokenizer, graph=None):
        self.base_tokenizer = base_tokenizer
        self.graph = graph
        self.automaton = AhoCorasickAutomaton()
        self.virtual_tokens = {}  # token_id -> (node_id, layer, content)
        self.temp_nodes = {}  # session_id -> [node_ids]
        
        # Загрузить концепты из графа
        self._load_concepts()
    
    def _load_concepts(self):
        """Загрузить концепты в автомат"""
        if not self.graph:
            return
        
        try:
            conn = sqlite3.connect(self.graph.db_path)
            cur = conn.cursor()
            cur.execute("SELECT id, content, level FROM nodes WHERE node_type='concept' AND level IN (2,3)")
            
            for i, row in enumerate(cur.fetchall()):
                node_id, content, level = row
                token_id = self.VIRTUAL_TOKEN_OFFSET + i
                self.virtual_tokens[token_id] = (node_id, level, content)
                self.automaton.add_word(content.lower(), token_id)
            
            self.automaton.make_automaton()
            logger.info(f"HybridTokenizer: {len(self.virtual_tokens)} concepts loaded")
            conn.close()
        except Exception as e:
            logger.warning(f"Load concepts: {e}")
    
    def encode(self, text: str, return_embeddings: bool = False) -> tuple:
        """
        Токенизация текста с заменой концептов на виртуальные токены
        """
        # BPE токены
        bpe_tokens = self.base_tokenizer.encode(text)
        
        # Поиск концептов
        found_concepts = []
        matches = self.automaton.search(text.lower())
        
        for end_pos, token_id in matches:
            if token_id in self.virtual_tokens:
                node_id, layer, content = self.virtual_tokens[token_id]
                found_concepts.append({
                    "node_id": node_id,
                    "token_id": token_id,
                    "content": content,
                    "layer": layer
                })
        
        return bpe_tokens, found_concepts
    
    def add_temporary_nodes(self, session_id: str, nodes: List[dict]):
        """Добавить временные узлы (document loader)"""
        if session_id not in self.temp_nodes:
            self.temp_nodes[session_id] = []
        
        for node in nodes:
            content = node.get("content", "")
            token_id = self.VIRTUAL_TOKEN_OFFSET + len(self.virtual_tokens)
            self.virtual_tokens[token_id] = (node["id"], 3, content)
            self.automaton.add_word(content.lower(), token_id)
            self.temp_nodes[session_id].append(token_id)
        
        self.automaton.make_automaton()
        logger.info(f"Added {len(nodes)} temp nodes for session: {session_id}")
    
    def remove_temporary_nodes(self, session_id: str):
        """Удалить временные узлы"""
        if session_id in self.temp_nodes:
            for token_id in self.temp_nodes[session_id]:
                if token_id in self.virtual_tokens:
                    del self.virtual_tokens[token_id]
            del self.temp_nodes[session_id]
            self.automaton = AhoCorasickAutomaton()
            self._load_concepts()
            logger.info(f"Removed temp nodes for session: {session_id}")


# === 3. FractalGraphV2 Extensions ===
class FractalGraphV2Full:
    """
    FractalGraphV2 с поддержкой временных сессий
    
    Методы:
    - begin_temp_session()
    - add_temp_node()
    - rollback_temp_session()
    - commit_temp_session()
    - create_activation_profile()
    - update_activation_profile()
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.temp_sessions = {}
        logger.info(f"FractalGraphV2: {Path(db_path).name}")
    
    def begin_temp_session(self, session_id: str):
        """Начать временную сессию"""
        self.temp_sessions[session_id] = []
        logger.info(f"Session started: {session_id}")
    
    def add_temp_node(self, session_id: str, content: str, node_type: str = "temp_paragraph", metadata: dict = None):
        """Добавить временный узел"""
        import uuid
        node_id = f"temp_{uuid.uuid4().hex[:8]}"
        
        self.cursor.execute('''
            INSERT INTO nodes (id, content, node_type, level, metadata, created_at, is_static)
            VALUES (?, ?, ?, 3, ?, ?, 0)
        ''', (node_id, content, node_type, json.dumps(metadata or {}), time.time()))
        
        if session_id not in self.temp_sessions:
            self.temp_sessions[session_id] = []
        self.temp_sessions[session_id].append(node_id)
        
        self.conn.commit()
        return node_id
    
    def rollback_temp_session(self, session_id: str):
        """Откатить сессию"""
        if session_id in self.temp_sessions:
            for node_id in self.temp_sessions[session_id]:
                self.cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            del self.temp_sessions[session_id]
            self.conn.commit()
            logger.info(f"Session rolled back: {session_id}")
    
    def commit_temp_session(self, session_id: str):
        """Зафиксировать сессию"""
        logger.info(f"Session committed: {session_id}")
    
    def create_activation_profile(self, domain: str, model_id: str) -> str:
        """Создать профиль активации (L1)"""
        import uuid
        node_id = f"profile_{uuid.uuid4().hex[:8]}"
        metadata = json.dumps({
            "domain": domain,
            "model_id": model_id,
            "avg_latency": 0,
            "sample_count": 0
        })
        
        self.cursor.execute('''
            INSERT INTO nodes (id, content, node_type, level, metadata, created_at)
            VALUES (?, ?, 'activation_profile', 1, ?, ?)
        ''', (node_id, f"profile_{domain}_{model_id}", metadata, time.time()))
        self.conn.commit()
        
        return node_id
    
    def update_activation_profile(self, node_id: str, latency_ms: float, fingerprint: Any = None):
        """Обновить профиль активации (L1)"""
        # Получить текущий профиль
        self.cursor.execute("SELECT metadata FROM nodes WHERE id = ?", (node_id,))
        row = self.cursor.fetchone()
        if not row:
            return
        
        meta = json.loads(row[0])
        count = meta.get("sample_count", 0)
        avg_latency = meta.get("avg_latency", 0)
        
        # Обновить
        new_avg = (avg_latency * count + latency_ms) / (count + 1)
        meta["avg_latency"] = new_avg
        meta["sample_count"] = count + 1
        
        self.cursor.execute("UPDATE nodes SET metadata = ? WHERE id = ?", (json.dumps(meta), node_id))
        self.conn.commit()
    
    def get_concepts(self):
        """Получить все концепты"""
        self.cursor.execute("SELECT id, content FROM nodes WHERE node_type='concept'")
        return [{"id": r[0], "content": r[1]} for r in self.cursor.fetchall()]
    
    def get_routing_rules(self):
        """Получить правила маршрутизации"""
        self.cursor.execute("SELECT id, content, metadata FROM nodes WHERE node_type='routing_rule'")
        return [{"id": r[0], "content": r[1], "params": json.loads(r[2])} for r in self.cursor.fetchall()]
    
    def close(self):
        self.conn.close()


# === 4. DocumentKnowledgeLoader ===
class DocumentKnowledgeLoader:
    """
    Загрузчик документа во временный подграф
    
    Методы:
    - load_document(doc_text, doc_id) -> session_id
    - unload_document(session_id)
    """
    
    def __init__(self, graph: FractalGraphV2Full, tokenizer: HybridTokenizer):
        self.graph = graph
        self.tokenizer = tokenizer
        logger.info("DocumentKnowledgeLoader initialized")
    
    def load_document(self, doc_text: str, doc_id: str) -> str:
        """Загрузить документ как временный подграф"""
        import uuid
        session_id = f"doc_{doc_id}_{int(time.time())}"
        
        # Начать сессию
        self.graph.begin_temp_session(session_id)
        
        # Разбить на параграфы
        paragraphs = [p.strip() for p in doc_text.split('\n\n') if p.strip()]
        
        nodes = []
        for i, para in enumerate(paragraphs):
            node_id = self.graph.add_temp_node(
                session_id, para, "temp_paragraph",
                {"index": i, "length": len(para)}
            )
            nodes.append({"id": node_id, "content": para})
        
        # Добавить в токенизатор
        self.tokenizer.add_temporary_nodes(session_id, nodes)
        
        logger.info(f"Document loaded: {doc_id}, session: {session_id}, {len(paragraphs)} paragraphs")
        return session_id
    
    def unload_document(self, session_id: str):
        """Выгрузить документ"""
        self.tokenizer.remove_temporary_nodes(session_id)
        self.graph.rollback_temp_session(session_id)
        logger.info(f"Document unloaded: {session_id}")


# === 5. BatchPipeline ===
class BatchPipeline:
    """
    Батчевая обработка больших документов
    
    Компоненты:
    - FractalChunker - разбиение на чанки
    - BatchScheduler - параллельная обработка
    - StructuralAggregator - объединение результатов
    """
    
    def __init__(self, generator, max_chunk_tokens: int = 2048, batch_size: int = 4):
        self.generator = generator
        self.max_chunk_tokens = max_chunk_tokens
        self.batch_size = batch_size
        logger.info(f"BatchPipeline: chunks={max_chunk_tokens}, batch={batch_size}")
    
    def chunk_document(self, text: str, doc_id: str) -> List[dict]:
        """Разбить документ на чанки"""
        paragraphs = text.split('\n\n')
        chunks = []
        current = []
        current_tokens = 0
        
        for para in paragraphs:
            tokens = len(self.generator.tokenizer.encode(para))
            
            if current_tokens + tokens > self.max_chunk_tokens and current:
                content = '\n\n'.join(current)
                chunks.append({
                    "id": f"{doc_id}_{len(chunks)}",
                    "content": content,
                    "prompt": f"<|doc_start|>\n<|seq {len(chunks)}|>\n{content}\n<|doc_end|>"
                })
                current = [para]
                current_tokens = tokens
            else:
                current.append(para)
                current_tokens += tokens
        
        # Последний чанк
        if current:
            content = '\n\n'.join(current)
            chunks.append({
                "id": f"{doc_id}_{len(chunks)}",
                "content": content,
                "prompt": f"<|doc_start|>\n<|seq {len(chunks)}|>\n{content}\n<|doc_end|>"
            })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_chunks(self, chunks: List[dict], task: str = "summarize") -> List[dict]:
        """Обработать чанки"""
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = []
                for chunk in batch:
                    future = executor.submit(
                        self.generator.pipe.generate,
                        chunk["prompt"],
                        512, 0.3, True
                    )
                    futures.append((chunk["id"], future))
                
                for chunk_id, future in futures:
                    try:
                        output = future.result(timeout=60)
                        results.append({"chunk_id": chunk_id, "output": output})
                    except Exception as e:
                        results.append({"chunk_id": chunk_id, "error": str(e)})
        
        return results
    
    def aggregate(self, results: List[dict], method: str = "map_reduce") -> str:
        """Объединить результаты"""
        if method == "map_reduce":
            outputs = [r.get("output", "") for r in results]
            return "\n\n---\n\n".join(outputs)
        return ""


# === 6. LoRAManager ===
class LoRAManager:
    """
    Управление LoRA адаптерами
    
    Методы:
    - load_adapter(domain)
    - apply_adapter(alias, alpha)
    - unload_adapter(alias)
    """
    
    def __init__(self, pipeline, adapters_dir: str = "./lora_adapters"):
        self.pipeline = pipeline
        self.adapters_dir = Path(adapters_dir)
        self.loaded_adapters = {}
        self.active_adapter = None
        logger.info(f"LoRAManager: {adapters_dir}")
    
    def load_adapter(self, domain: str) -> Optional[str]:
        """Загрузить адаптер"""
        adapter_path = self.adapters_dir / f"{domain}_lora"
        
        if not adapter_path.exists():
            logger.warning(f"Adapter not found: {domain}")
            return None
        
        if domain not in self.loaded_adapters:
            logger.info(f"Loading adapter: {domain}")
            self.loaded_adapters[domain] = str(adapter_path)
        
        return domain
    
    def apply_adapter(self, alias: str, alpha: float = 1.0):
        """Применить адаптер"""
        if alias not in self.loaded_adapters:
            logger.warning(f"Adapter not loaded: {alias}")
            return
        
        logger.info(f"Applying adapter: {alias} (alpha={alpha})")
        self.active_adapter = alias
        # Note: OpenVINO GenAPI не имеет полной поддержки LoRA в текущей версии
    
    def unload_adapter(self, alias: str):
        """Выгрузить адаптер"""
        if alias in self.loaded_adapters:
            del self.loaded_adapters[alias]
            if self.active_adapter == alias:
                self.active_adapter = None
            logger.info(f"Unloaded adapter: {alias}")


# === Main Factory ===
class FMFGeneratorFull:
    """
    Полный FMF генератор
    
    Методы:
    - generate(prompt, mode, use_lora, enable_thinking, context_nodes)
    - load_document(doc_text, doc_id)
    - unload_document(session_id)
    - process_large_document(doc_text, doc_id, task, batch_size)
    """
    
    def __init__(self, model_path: str, graph_path: str = None, device: str = "CPU"):
        logger.info("=== FMF Full: Initialization ===")
        
        # L0: OpenVINO Pipeline
        import openvino_genai as ov_genai
        self.pipe = ov_genai.LLMPipeline(model_path, device)
        logger.info("L0: Pipeline OK")
        
        # Токенизатор
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # L3: Graph
        self.graph = FractalGraphV2Full(graph_path) if graph_path else None
        
        # HybridTokenizer
        self.hybrid_tokenizer = HybridTokenizer(self.tokenizer, self.graph)
        
        # L2: Routing cache
        self._routing_cache = {}
        self._load_routing()
        
        # LoRA Manager
        self.lora_manager = LoRAManager(self.pipe)
        
        logger.info("=== FMF Full: Ready ===")
    
    def _load_routing(self):
        """Загрузить routing правила"""
        if not self.graph:
            return
        
        rules = self.graph.get_routing_rules()
        for rule in rules:
            domain = rule["content"].replace("rule_", "")
            self._routing_cache[domain] = rule["params"]
        
        logger.info(f"L2: {len(rules)} rules")
    
    def get_routing_config(self, domain: str = "general") -> dict:
        """Получить конфигурацию (L2)"""
        return self._routing_cache.get(domain, self._routing_cache.get("general", {
            "temperature": 0.7, "max_tokens": 512
        }))
    
    def generate(
        self,
        prompt: str,
        mode: str = "condensed",
        use_lora: bool = False,
        enable_thinking: bool = False,
        context_nodes: List[str] = None
    ) -> dict:
        """
        Основной метод генерации
        """
        start = time.time()
        
        # Определить домен и параметры из L2
        domain = self._detect_domain(prompt)
        config = self.get_routing_config(domain)
        
        # Построить промпт
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template с thinking
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # L0: Генерация
        response = self.pipe.generate(
            full_prompt,
            max_new_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            do_sample=config.get("temperature", 0.7) > 0
        )
        
        latency = (time.time() - start) * 1000
        
        return {
            "response": response,
            "domain": domain,
            "latency_ms": latency,
            "config": config
        }
    
    def _detect_domain(self, prompt: str) -> str:
        """Определить домен по ключевым словам"""
        prompt_lower = prompt.lower()
        
        keywords = {
            "technology": ["искусственный", "машинное", "нейронные", "компьютер", "программ"],
            "science": ["физика", "химия", "биология", "наука"],
            "history": ["история", "война", "год"],
        }
        
        for domain, words in keywords.items():
            for word in words:
                if word in prompt_lower:
                    return domain
        
        return "general"
    
    def load_document(self, doc_text: str, doc_id: str) -> str:
        """Загрузить документ (контекстный режим)"""
        loader = DocumentKnowledgeLoader(self.graph, self.hybrid_tokenizer)
        return loader.load_document(doc_text, doc_id)
    
    def process_large_document(self, doc_text: str, doc_id: str, task: str = "summarize") -> dict:
        """Обработать большой документ (пакетный режим)"""
        batch = BatchPipeline(self)
        
        # Чанкинг
        chunks = batch.chunk_document(doc_text, doc_id)
        
        # Обработка
        results = batch.process_chunks(chunks, task)
        
        # Агрегация
        summary = batch.aggregate(results)
        
        return {
            "summary": summary,
            "num_chunks": len(chunks),
            "results": results
        }


# === Test ===
if __name__ == "__main__":
    import os
    NUM_THREADS = 8
    
    try:
        import torch
        torch.set_num_threads(NUM_THREADS)
    except:
        pass
    
    print(f"=== FMF Full Test (8 cores) ===")
    print(f"CPU: {os.cpu_count()}")
    
    paths = {
        "model": "C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\models\\ruadapt_qwen3_4b_openvino",
        "graph": "C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\eva_ai\\memory\\fractal_graph_v2\\fractal_graph_v2_data\\fractal_graph.db"
    }
    
    print("\n=== FMF Full: Testing ===")
    fmf = FMFGeneratorFull(paths["model"], paths["graph"], "CPU")
    
    # Test 1: Basic generation
    print("\n--- Test 1: Generation ---")
    r = fmf.generate("Привет! Как дела?", enable_thinking=True)
    
    # Save to file to preserve Unicode
    with open("C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\test_output.txt", "w", encoding="utf-8") as f:
        f.write("=== TEST 1: GENERATION ===\n")
        f.write(r['response'])
        f.write(f"\n\n=== METADATA ===\n")
        f.write(f"Domain: {r['domain']}\n")
        f.write(f"Latency: {r['latency_ms']:.0f}ms\n")
    
    print(f"Response saved to test_output.txt")
    print(f"Length: {len(r['response'])} chars")
    print(f"Domain: {r['domain']}, Latency: {r['latency_ms']:.0f}ms")
    
    # Test 2: Generation with context
    print("\n--- Test 2: Generation with context (technology) ---")
    r2 = fmf.generate("Расскажи про искусственный интеллект", enable_thinking=True)
    print(f"\n=== FULL RESPONSE ===")
    print(r2['response'])
    print(f"\n=== END RESPONSE ===")
    print(f"Domain: {r2['domain']}, Latency: {r2['latency_ms']:.0f}ms")
    
    # Test 3: Concepts in Graph
    print("\n--- Test 3: Concepts in Graph ---")
    concepts = fmf.graph.get_concepts()
    print(f"Total concepts: {len(concepts)}")
    for i, c in enumerate(concepts):
        print(f"  [{i+1}] {c['content']}")
    
    print("\n--- Test 4: Graph Operations ---")
    import uuid
    node_id = fmf.graph.add_temp_node("test_session", "Нейронная сеть - это система связанных узлов", "temp_paragraph", {"test": True})
    print(f"Added temp node: {node_id}")
    
    print("\n=== FMF Full: Complete ===")