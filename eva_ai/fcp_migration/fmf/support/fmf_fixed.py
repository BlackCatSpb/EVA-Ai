"""
FMF Full - Fixed Thread Safety
Согласно FMF optimize 3.txt
"""

import time
import json
import sqlite3
import logging
import threading
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("FMF_Fixed")

NUM_THREADS = 8


# === LRU Cache ===
class LRUCache:
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.cache = OrderedDict()
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


# === Aho-Corasick ===
class AhoCorasickAutomaton:
    def __init__(self):
        self.next = [{}]
        self.fail = [0]
        self.output = [[]]
        self.output_text = {}
    
    def add_word(self, word: str, token_id: int):
        node = 0
        for char in word:
            if char not in self.next[node]:
                self.next[node][char] = len(self.next)
                self.next.append({})
                self.fail.append(0)
                self.output.append([])
            node = self.next[node][char]
        
        if token_id not in self.output[node]:
            self.output[node].append(token_id)
        self.output_text[token_id] = word
    
    def make_automaton(self):
        from collections import deque
        queue = deque([0])
        while queue:
            v = queue.popleft()
            for char, u in self.next[v].items():
                queue.append(u)
                if v != 0:
                    self.fail[u] = self.next[self.fail[v]].get(char, 0)
                    self.output[u].extend(self.output[self.fail[u]])
    
    def search(self, text: str):
        results = []
        node = 0
        text_lower = text.lower()
        
        for i, char in enumerate(text_lower):
            while char not in self.next[node] and node != 0:
                node = self.fail[node]
            node = self.next[node].get(char, 0)
            
            for token_id in self.output[node]:
                matched_text = self.output_text.get(token_id, "")
                results.append({
                    "token_id": token_id,
                    "text": matched_text,
                    "position": i - len(matched_text) + 1
                })
        
        return results


# === FractalGraphV2 - Thread Safe ===
class FractalGraphV2ThreadSafe:
    """
    Thread-safe SQLite с использованием threading.local()
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        logger.info(f"FractalGraphV2: {Path(db_path).name}")
        self._fix_duplicates()
    
    def _get_connection(self):
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _fix_duplicates(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, content FROM nodes WHERE node_type='concept' ORDER BY created_at")
        seen = {}
        duplicates = []
        for row in cur.fetchall():
            content = row[1]
            if content in seen:
                duplicates.append(row[0])
            else:
                seen[content] = row[0]
        
        for dup_id in duplicates:
            cur.execute("DELETE FROM nodes WHERE id = ?", (dup_id,))
        
        if duplicates:
            conn.commit()
            logger.info(f"Deduplication: removed {len(duplicates)} duplicates")
        conn.close()
    
    def get_concepts(self):
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, content FROM nodes WHERE node_type='concept'")
        return [{"id": r[0], "content": r[1]} for r in cur.fetchall()]
    
    def get_routing_rules(self):
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, content, metadata FROM nodes WHERE node_type='routing_rule'")
        return [{"id": r[0], "content": r[1], "params": json.loads(r[2])} for r in cur.fetchall()]
    
    def create_activation_profile(self, domain: str, model_id: str) -> str:
        import uuid
        node_id = f"profile_{uuid.uuid4().hex[:8]}"
        metadata = json.dumps({"domain": domain, "model_id": model_id, "avg_latency": 0, "sample_count": 0})
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO nodes (id, content, node_type, level, metadata, created_at)
            VALUES (?, ?, 'activation_profile', 1, ?, ?)
        ''', (node_id, f"profile_{domain}_{model_id}", metadata, time.time()))
        conn.commit()
        return node_id
    
    def add_temp_node(self, session_id: str, content: str, node_type: str = "temp_paragraph"):
        import uuid
        node_id = f"temp_{uuid.uuid4().hex[:8]}"
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO nodes (id, content, node_type, level, created_at, is_static)
            VALUES (?, ?, ?, 3, ?, 0)
        ''', (node_id, content, node_type, time.time()))
        conn.commit()
        return node_id
    
    def begin_temp_session(self, session_id: str):
        pass
    
    def rollback_temp_session(self, session_id: str):
        pass


# === HybridTokenizer - Fixed ===
class HybridTokenizerFixed:
    VIRTUAL_TOKEN_OFFSET = 151936
    
    def __init__(self, base_tokenizer, graph=None):
        self.base_tokenizer = base_tokenizer
        self.graph = graph
        self.automaton = AhoCorasickAutomaton()
        self.virtual_tokens = {}
        self._load_concepts()
    
    def _load_concepts(self):
        if not self.graph:
            return
        try:
            concepts = self.graph.get_concepts()
            logger.info(f"Loading {len(concepts)} concepts from graph")
            
            for i, concept in enumerate(concepts):
                content = concept["content"]
                if content:
                    token_id = self.VIRTUAL_TOKEN_OFFSET + i
                    self.virtual_tokens[token_id] = content
                    self.automaton.add_word(content.lower(), token_id)
                    logger.info(f"  Added: '{content}' -> token {token_id}")
            
            self.automaton.make_automaton()
            logger.info(f"HybridTokenizer: {len(self.virtual_tokens)} concepts in automaton")
            
        except Exception as e:
            logger.warning(f"Load concepts: {e}")
    
    def encode(self, text: str):
        bpe_tokens = self.base_tokenizer.encode(text)
        
        # Search with automaton
        matches = self.automaton.search(text)
        
        found = []
        for match in matches:
            token_id = match["token_id"]
            content = self.virtual_tokens.get(token_id, "")
            if content:
                found.append({
                    "token_id": token_id,
                    "text": content,
                    "position": match["position"]
                })
        
        return bpe_tokens, found
    
    def count_virtual_tokens(self, tokens: List[int]) -> int:
        count = 0
        for t in tokens:
            if t >= self.VIRTUAL_TOKEN_OFFSET:
                count += 1
        return count


# === LoRAManager ===
class LoRAManagerOptimized:
    def __init__(self, pipeline, adapters_dir: str = "./lora_adapters"):
        self.pipeline = pipeline
        self.adapters_dir = Path(adapters_dir)
        self._cache = {}
        self._missing = set()
        self._active = None
        logger.info(f"LoRAManager: {adapters_dir}")
    
    def load_adapter(self, domain: str) -> Optional[str]:
        if domain in self._missing:
            return None
        if domain in self._cache:
            return domain
        adapter_path = self.adapters_dir / f"{domain}_lora"
        if not adapter_path.exists():
            self._missing.add(domain)
            return None
        logger.info(f"Loading adapter: {domain}")
        self._cache[domain] = domain
        return domain
    
    def apply_adapter(self, alias: str, alpha: float = 1.0):
        if alias in self._cache:
            self._active = alias
            logger.info(f"Applied: {alias} (alpha={alpha})")


# === FMF Generator - Thread Safe Fixed ===
class FMFGeneratorFixed:
    def __init__(self, model_path: str, graph_path: str = None, device: str = "CPU"):
        logger.info("=== FMF Fixed: Init ===")
        
        import openvino_genai as ov_genai
        self.pipe = ov_genai.LLMPipeline(model_path, device)
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        logger.info("Tokenizer: OK")
        
        # L3: Thread-safe graph
        self.graph = FractalGraphV2ThreadSafe(graph_path) if graph_path else None
        
        # HybridTokenizer
        self.hybrid_tokenizer = HybridTokenizerFixed(self.tokenizer, self.graph)
        
        # L2: Routing cache
        self._routing_cache = {}
        self._load_routing()
        
        # Template cache
        self._template_cache = LRUCache(maxsize=50)
        
        # LoRA
        self.lora_manager = LoRAManagerOptimized(self.pipe)
        
        logger.info("=== FMF Fixed: Ready ===")
    
    def _load_routing(self):
        if not self.graph:
            return
        rules = self.graph.get_routing_rules()
        for rule in rules:
            domain = rule["content"].replace("rule_", "")
            self._routing_cache[domain] = rule["params"]
        logger.info(f"L2: {len(rules)} rules cached")
    
    def _detect_domain(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        keywords = {
            "technology": ["искусственный", "машинное", "нейронные", "компьютер", "программ", "ии"],
            "science": ["физика", "химия", "биология", "наука"],
        }
        for domain, words in keywords.items():
            for word in words:
                if word in prompt_lower:
                    return domain
        return "general"
    
    def _apply_cached_template(self, content: str, enable_thinking: bool = True) -> str:
        cache_key = (content[:30], enable_thinking)
        cached = self._template_cache.get(cache_key)
        if cached is not None:
            return cached
        
        messages = [{"role": "user", "content": content}]
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        self._template_cache.put(cache_key, result)
        return result
    
    def _schedule_async_update(self, domain: str):
        def update():
            try:
                if self.graph:
                    self.graph.create_activation_profile(domain, "qwen3-4b")
                    logger.info(f"Async L1: profile updated for {domain}")
            except Exception as e:
                logger.warning(f"Async: {e}")
        
        thread = threading.Thread(target=update, daemon=True)
        thread.start()
    
    def generate(
        self,
        prompt: str,
        enable_thinking: bool = True
    ) -> dict:
        start = time.time()
        
        # Tokenize & find concepts
        tokens, found_concepts = self.hybrid_tokenizer.encode(prompt)
        virtual_count = len(found_concepts)
        
        logger.info(f"Found {virtual_count} virtual tokens: {[f['text'] for f in found_concepts]}")
        
        # Domain detection
        domain = self._detect_domain(prompt)
        
        # Routing config
        config = self._routing_cache.get(domain, self._routing_cache.get("general", {
            "temperature": 0.7, "max_tokens": 2048  # Increased for thinking
        }))
        
        # Template
        full_prompt = self._apply_cached_template(prompt, enable_thinking)
        
        # Generation
        response = self.pipe.generate(
            full_prompt,
            max_new_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            do_sample=config.get("temperature", 0.7) > 0
        )
        
        latency = (time.time() - start) * 1000
        
        # Async update
        self._schedule_async_update(domain)
        
        return {
            "response": response,
            "domain": domain,
            "latency_ms": latency,
            "virtual_tokens": virtual_count,
            "found_concepts": found_concepts
        }


# === Test ===
if __name__ == "__main__":
    import os
    try:
        import torch
        torch.set_num_threads(NUM_THREADS)
    except:
        pass
    
    print(f"=== FMF Fixed Test (8 cores) ===")
    print(f"CPU: {os.cpu_count()}")
    
    paths = {
        "model": "C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\models\\ruadapt_qwen3_4b_openvino",
        "graph": "C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\eva_ai\\memory\\fractal_graph_v2\\fractal_graph_v2_data\\fractal_graph.db"
    }
    
    print("\n=== Init ===")
    fmf = FMFGeneratorFixed(paths["model"], paths["graph"], "CPU")
    
    # Concepts check
    concepts = fmf.graph.get_concepts()
    print(f"\n--- Concepts in graph: {len(concepts)} ---")
    for c in concepts:
        print(f"  - {c['content']}")
    
    # Test 1
    print("\n--- Test 1: Basic ---")
    r1 = fmf.generate("Привет! Как дела?", enable_thinking=True)
    
    with open("C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\test_fixed.txt", "w", encoding="utf-8") as f:
        f.write(f"=== TEST 1: Привет! Как дела? ===\n\n")
        f.write(r1['response'])
        f.write(f"\n\n=== METADATA ===\n")
        f.write(f"Domain: {r1['domain']}\n")
        f.write(f"Latency: {r1['latency_ms']:.0f}ms\n")
        f.write(f"Virtual tokens: {r1['virtual_tokens']}\n")
        f.write(f"Found concepts: {r1['found_concepts']}\n")
    
    print(f"Domain: {r1['domain']}, Latency: {r1['latency_ms']:.0f}ms")
    print(f"Found: {r1['virtual_tokens']} virtual tokens")
    
    # Test 2
    print("\n--- Test 2: Technology ---")
    r2 = fmf.generate("Расскажи про искусственный интеллект", enable_thinking=True)
    
    with open("C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\test_fixed.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n=== TEST 2: ИИ ===\n\n")
        f.write(r2['response'])
        f.write(f"\n\n=== METADATA ===\n")
        f.write(f"Domain: {r2['domain']}\n")
        f.write(f"Latency: {r2['latency_ms']:.0f}ms\n")
        f.write(f"Found concepts: {r2['found_concepts']}\n")
    
    print(f"Domain: {r2['domain']}, Latency: {r2['latency_ms']:.0f}ms")
    print(f"Found: {r2['found_concepts']}")
    
    print("\n=== Complete ===")
    print("Saved to test_fixed.txt")