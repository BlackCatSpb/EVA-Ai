"""
FMF Interactive - Управляющие токены для продолжения
Согласно FMF optimize 5.txt
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
logger = logging.getLogger("FMF_Interactive")

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


# === HybridTokenizer с Control Tokens ===
class HybridTokenizerInteractive:
    VIRTUAL_TOKEN_OFFSET = 151936
    
    # Управляющие токены (FMF optimize 5.txt)
    CONTROL_TOKEN_START = 250000
    CONTROL_TOKENS = {
        250000: {"type": "continue_request", "text": "<|continue_request|>"},
        250001: {"type": "continue_yes", "text": "<|continue_yes|>"},
        250002: {"type": "continue_no", "text": "<|continue_no|>"},
        250003: {"type": "response_end", "text": "<|response_end|>"},
    }
    
    def __init__(self, base_tokenizer, graph=None):
        self.base_tokenizer = base_tokenizer
        self.graph = graph
        self.automaton = AhoCorasickAutomaton()
        self.virtual_tokens = {}
        self.control_tokens = {}
        
        self._register_control_tokens()
        self._load_concepts()
    
    def _register_control_tokens(self):
        """Регистрируем управляющие токены"""
        for tid, info in self.CONTROL_TOKENS.items():
            self.virtual_tokens[tid] = {
                "type": "control",
                "text": info["text"],
                "control_type": info["type"]
            }
            self.control_tokens[info["type"]] = tid
            # Добавляем в автомат для распознавания
            self.automaton.add_word(info["text"].lower(), tid)
        
        self.automaton.make_automaton()
        logger.info(f"Control tokens: {len(self.CONTROL_TOKENS)} registered")
    
    def _load_concepts(self):
        if not self.graph:
            return
        try:
            concepts = self.graph.get_concepts()
            for i, concept in enumerate(concepts):
                content = concept["content"]
                if content:
                    token_id = self.VIRTUAL_TOKEN_OFFSET + i
                    self.virtual_tokens[token_id] = content
                    self.automaton.add_word(content.lower(), token_id)
            self.automaton.make_automaton()
            logger.info(f"Loaded {len(concepts)} concepts")
        except Exception as e:
            logger.warning(f"Load concepts: {e}")
    
    def encode(self, text: str):
        bpe_tokens = self.base_tokenizer.encode(text)
        matches = self.automaton.search(text)
        
        found = []
        for match in matches:
            token_id = match["token_id"]
            if token_id in self.virtual_tokens:
                info = self.virtual_tokens[token_id]
                found.append({
                    "token_id": token_id,
                    "text": info if isinstance(info, str) else info.get("text", ""),
                    "type": "concept" if isinstance(info, str) else info.get("type", "concept")
                })
        
        return bpe_tokens, found
    
    def check_continuation_token(self, response: str) -> bool:
        """Проверяет, есть ли в ответе токен continue_request"""
        return "<|continue_request|>" in response
    
    def clean_response(self, response: str) -> str:
        """Очищает ответ от управляющих токенов"""
        for tid, info in self.CONTROL_TOKENS.items():
            response = response.replace(info["text"], "")
        return response.strip()


# === Embeddings Manager для FMF ===
class FMFEmbeddingsManager:
    """Упрощённый менеджер эмбеддингов для FMF"""
    
    _instance = None
    
    def __init__(self, device: str = "CPU"):
        self.device = device
        self._model = None
        self._embedding_dim = 768
        
        # Локальный кэш EVA проекта
        self._cache_dir = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\hf_cache\models--intfloat--multilingual-e5-base\snapshots"
    
    @classmethod
    def get_instance(cls, device: str = "CPU"):
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance
    
    def _ensure_model(self):
        if self._model is None:
            try:
                import os
                from sentence_transformers import SentenceTransformer
                
                # Устанавливаем HF_HOME перед загрузкой
                os.environ['HF_HOME'] = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\hf_cache"
                os.environ['TRANSFORMERS_CACHE'] = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\hf_cache"
                
                # Проверяем локальный путь
                local_path = self._cache_dir
                if os.path.exists(local_path):
                    # Найти первую папку с моделью
                    for item in os.listdir(local_path):
                        sub_path = os.path.join(local_path, item)
                        if os.path.isdir(sub_path):
                            model_file = os.path.join(sub_path, "model.safetensors")
                            if os.path.exists(model_file):
                                local_path = sub_path
                                break
                    
                    # Загружаем из локального пути
                    self._model = SentenceTransformer(local_path, device=self.device)
                    logger.info(f"FMF Embeddings loaded from: {local_path}")
                else:
                    # Fallback - загружаем normally ( будет использовать кэш)
                    self._model = SentenceTransformer("intfloat/multilingual-e5-base", device=self.device)
                    logger.info("FMF Embeddings: loaded (fallback)")
                
self._embedding_dim = 768
            except Exception as e:
                logger.warning(f"Embeddings load failed: {e}")
    
    def encode(self, texts: list) -> list:
        """Получить эмбеддинги для текстов"""
        self._ensure_model()
        if self._model is None:
            return None
        try:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def get_embedding(self, text: str) -> list:
        """Получить эмбеддинг для одного текста"""
        result = self.encode([text])
        return result[0] if result else None


# === FractalGraphV2 Thread Safe ===
class FractalGraphV2ThreadSafe:
    def __init__(self, db_path: str, enable_embeddings: bool = True):
        self.db_path = db_path
        self._local = threading.local()
        self.temp_sessions = {}  # In-memory session storage
        self.enable_embeddings = enable_embeddings
        self.embeddings = None
        
        if enable_embeddings:
            self.embeddings = FMFEmbeddingsManager.get_instance("CPU")
        
        logger.info(f"Graph: {Path(db_path).name} (embeddings={enable_embeddings})")
        self._fix_duplicates()
    
    def _get_connection(self):
        if not hasattr(self._local, 'conn'):
            import sqlite3
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
            logger.info(f"Deduplication: {len(duplicates)} removed")
        conn.close()
    
    def get_all_concepts(self):
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT content FROM nodes WHERE node_type='concept'")
        return [r[0] for r in cur.fetchall()]
    
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
        metadata = json.dumps({"domain": domain, "model_id": model_id})
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO nodes (id, content, node_type, level, metadata, created_at)
            VALUES (?, ?, 'activation_profile', 1, ?, ?)
        ''', (node_id, f"profile_{domain}_{model_id}", metadata, time.time()))
        conn.commit()
        return node_id
    
    def concept_exists(self, content: str) -> bool:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM nodes WHERE node_type='concept' AND content = ?", (content,))
        return cur.fetchone() is not None
    
    def add_unique_concept(self, content: str, embedding = None, metadata = None):
        if self.concept_exists(content):
            logger.info(f"Concept exists: {content}")
            return None
        
        import uuid
        node_id = f"concept_{uuid.uuid4().hex[:8]}"
        
        # Вычисляем embedding если включен и не передан
        if embedding is None and self.embeddings:
            embedding = self.embeddings.get_embedding(content)
        
        conn = self._get_connection()
        cur = conn.cursor()
        
        # Convert embedding to bytes if exists
        emb_bytes = None
        if embedding and isinstance(embedding, list):
            import numpy as np
            emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        cur.execute('''
            INSERT INTO nodes (id, content, node_type, level, metadata, created_at, is_static, embedding)
            VALUES (?, ?, 'concept', 3, ?, ?, 1, ?)
        ''', (node_id, content, json.dumps(metadata or {}), time.time(), emb_bytes))
        
        conn.commit()
        logger.info(f"Added concept: {content} (embedding={embedding is not None})")
        return node_id
    
    def fact_exists(self, subject: str, predicate: str, obj: str) -> bool:
        conn = self._get_connection()
        cur = conn.cursor()
        content = f"{subject} {predicate} {obj}"
        cur.execute("SELECT id FROM nodes WHERE node_type='fact' AND content = ?", (content,))
        return cur.fetchone() is not None
    
    def add_unique_fact(self, subject: str, predicate: str, obj: str, metadata = None):
        if self.fact_exists(subject, predicate, obj):
            logger.info(f"Fact exists: {subject} {predicate} {obj}")
            return None
        
        import uuid
        node_id = f"fact_{uuid.uuid4().hex[:8]}"
        content = f"{subject} {predicate} {obj}"
        
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO nodes (id, content, node_type, level, metadata, created_at, is_static)
            VALUES (?, ?, 'fact', 3, ?, ?, 1)
        ''', (node_id, content, json.dumps(metadata or {"subject": subject, "predicate": predicate, "object": obj}), time.time()))
        
        conn.commit()
        logger.info(f"Added fact: {content}")
        return node_id
    
    # === Временные сессии ===
    def begin_temp_session(self, session_id: str):
        self.temp_sessions[session_id] = []
        logger.info(f"Session started: {session_id}")
    
    def add_temp_node(self, session_id: str, content: str, node_type: str = "temp_message"):
        import uuid
        node_id = f"temp_{uuid.uuid4().hex[:8]}"
        
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO nodes (id, content, node_type, level, created_at, is_static)
            VALUES (?, ?, ?, 3, ?, 0)
        ''', (node_id, content, node_type, time.time()))
        conn.commit()
        
        if session_id in self.temp_sessions:
            self.temp_sessions[session_id].append(node_id)
        
        return node_id
    
    def get_session_messages(self, session_id: str) -> List[dict]:
        if session_id not in self.temp_sessions:
            return []
        
        messages = []
        conn = self._get_connection()
        cur = conn.cursor()
        
        for node_id in self.temp_sessions[session_id]:
            cur.execute("SELECT content, metadata FROM nodes WHERE id = ?", (node_id,))
            row = cur.fetchone()
            if row:
                messages.append({
                    "id": node_id,
                    "content": row[0],
                    "metadata": json.loads(row[1]) if row[1] else {}
                })
        
        return messages
    
    def rollback_temp_session(self, session_id: str):
        if session_id not in self.temp_sessions:
            return
        
        conn = self._get_connection()
        cur = conn.cursor()
        
        for node_id in self.temp_sessions[session_id]:
            cur.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        
        conn.commit()
        del self.temp_sessions[session_id]
        logger.info(f"Session rolled back: {session_id}")


# === LoRAManager ===
class LoRAManagerOptimized:
    def __init__(self, pipeline, adapters_dir: str = "./lora_adapters"):
        self.pipeline = pipeline
        self.adapters_dir = Path(adapters_dir)
        self._cache = {}
        self._missing = set()
    
    def load_adapter(self, domain: str) -> Optional[str]:
        if domain in self._missing:
            return None
        if domain in self._cache:
            return domain
        adapter_path = self.adapters_dir / f"{domain}_lora"
        if not adapter_path.exists():
            self._missing.add(domain)
            return None
        self._cache[domain] = domain
        return domain
    
    def apply_adapter(self, alias: str, alpha: float = 1.0):
        if alias in self._cache:
            logger.info(f"Applied: {alias} (alpha={alpha})")


# === Главный класс с Interactive продолжением ===
class FMFGeneratorInteractive:
    def __init__(self, model_path: str, graph_path: str = None, device: str = "CPU", enable_embeddings: bool = True):
        logger.info("=== FMF Interactive: Init (Optimized) ===")
        
        import openvino_genai as ov_genai
        
        # === GenerationConfig ===
        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 2048
        generation_config.temperature = 0.2  # Точные ответы
        generation_config.top_p = 0.9       # Nucleus sampling
        generation_config.top_k = 40
        generation_config.repetition_penalty = 1.1
        generation_config.do_sample = True
        
        # === SchedulerConfig (continuous batching) ===
        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.max_num_batched_tokens = 2048
        scheduler_config.max_num_seqs = 8
        scheduler_config.enable_prefix_caching = True
        
        # Create pipeline with config
        config = {
            "generation_config": generation_config,
            "scheduler_config": scheduler_config
        }
        
        self.pipe = ov_genai.LLMPipeline(model_path, device, config=config)
        
        from openvino_genai import Tokenizer
        self.tokenizer = Tokenizer(model_path)
        
        self.graph = FractalGraphV2ThreadSafe(graph_path, enable_embeddings=enable_embeddings) if graph_path else None
        
        self.hybrid_tokenizer = HybridTokenizerInteractive(self.tokenizer, self.graph)
        
        self._routing_cache = {}
        self._load_routing()
        
        self._template_cache = LRUCache(50)
        
        self.lora_manager = LoRAManagerOptimized(self.pipe)
        
        # Состояние для интерактивного продолжения
        self._continuation_state = None
        
        logger.info("=== Ready ===")
    
    def _load_routing(self):
        if not self.graph:
            return
        rules = self.graph.get_routing_rules()
        for rule in rules:
            domain = rule["content"].replace("rule_", "")
            self._routing_cache[domain] = rule["params"]
        logger.info(f"L2: {len(rules)} rules")
    
    def _detect_domain(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        keywords = {
            "technology": ["искусственный", "машинное", "нейронные", "компьютер", "программ", "ии"],
            "science": ["физика", "химия", "��иология"],
        }
        for domain, words in keywords.items():
            for word in words:
                if word in prompt_lower:
                    return domain
        return "general"
    
    def _add_continuation_instruction(self, prompt: str) -> str:
        """Добавляет инструкцию для использования continue_request"""
        max_chars = 10000
        instruction = (
            f"Ты — ассистент. Отвечай кратко и по существу. "
            f"Максимум {max_chars} символов (включая рассуждения, если есть).\n\n"
            "Если ответ получается очень длинный — вставь <|continue_request|> и спроси 'Продолжить?'\n"
            "Если ответ короткий — используй <|response_end|>\n\n"
            "Вопрос: "
        )
        return instruction + prompt
    
    def _build_prompt(self, content: str, enable_thinking: bool = True) -> str:
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
    
    def _schedule_async(self, domain: str):
        def update():
            try:
                if self.graph:
                    self.graph.create_activation_profile(domain, "qwen3-4b")
            except:
                pass
        threading.Thread(target=update, daemon=True).start()
    
    def generate(
        self,
        prompt: str,
        enable_thinking: bool = True,
        max_tokens: int = 2048
    ) -> dict:
        start = time.time()
        
        tokens, found = self.hybrid_tokenizer.encode(prompt)
        
        domain = self._detect_domain(prompt)
        config = self._routing_cache.get(domain, self._routing_cache.get("general", {
            "temperature": 0.7, "max_tokens": max_tokens
        }))
        
        full_prompt = self._build_prompt(prompt, enable_thinking)
        
        response = self.pipe.generate(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=config.get("temperature", 0.7),
            do_sample=config.get("temperature", 0.7) > 0,
            eos_token_id=-1
        )
        
        latency = (time.time() - start) * 1000
        
        self._schedule_async(domain)
        
        return {
            "response": response,
            "domain": domain,
            "latency_ms": latency,
            "found": found
        }
    
    # === Interactive методы (FMF optimize 5.txt) ===
    def generate_interactive(
        self,
        prompt: str,
        max_tokens_per_chunk: int = 512,
        enable_thinking: bool = True
    ) -> dict:
        """
        Генерирует ответ. Если встречается <|continue_request|>,
        возвращает need_continuation=True и сохраняет состояние.
        """
        # Добавляем инструкцию
        enhanced_prompt = self._add_continuation_instruction(prompt)
        
        full_prompt = self._build_prompt(enhanced_prompt, enable_thinking)
        
        # Генерируем первый фрагмент
        response = self.pipe.generate(
            full_prompt,
            max_new_tokens=max_tokens_per_chunk,
            temperature=0.7,
            do_sample=True
        )
        
        # Проверяем наличие continue_request
        need_continuation = self.hybrid_tokenizer.check_continuation_token(response)
        
        # Очищаем ответ для показа
        clean_response = self.hybrid_tokenizer.clean_response(response)
        
        if need_continuation:
            # Сохраняем состояние для продолжения
            self._continuation_state = {
                "original_prompt": prompt,
                "generated_so_far": clean_response,
                "full_raw": response,
                "enable_thinking": enable_thinking,
                "max_tokens": max_tokens_per_chunk
            }
        
        return {
            "response": clean_response,
            "need_continuation": need_continuation,
            "full_raw": response
        }
    
    def continue_generation(self, user_agreed: bool = True) -> Optional[dict]:
        """
        Продолжает генерацию после согласия пользователя.
        """
        if not self._continuation_state:
            logger.warning("No continuation state")
            return None
        
        state = self._continuation_state
        self._continuation_state = None
        
        if not user_agreed:
            # Вставляем <|continue_no|> и генерируем завершение
            continuation_marker = "<|continue_no|>"
        else:
            # Вставляем <|continue_yes|> и продолжаем
            continuation_marker = "<|continue_yes|>"
        
        # Формируем новый промпт
        continuation_prompt = (
            state["original_prompt"] + "\n\n" + 
            state["generated_so_far"] + "\n" + continuation_marker
        )
        
        full_prompt = self._build_prompt(continuation_prompt, state["enable_thinking"])
        
        # Продолжаем генерацию
        next_part = self.pipe.generate(
            full_prompt,
            max_new_tokens=state["max_tokens"],
            temperature=0.7,
            do_sample=True
        )
        
        clean_next = self.hybrid_tokenizer.clean_response(next_part)
        
        return {
            "response": state["generated_so_far"] + "\n\n" + clean_next,
            "continued": user_agreed,
            "full_raw": next_part
        }


# === Test ===
if __name__ == "__main__":
    import os
    try:
        import torch
        torch.set_num_threads(NUM_THREADS)
    except:
        pass
    
    print(f"=== FMF Interactive Test (8 cores) ===")
    print(f"CPU: {os.cpu_count()}")
    
    paths = {
        "model": "C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\models\\ruadapt_qwen3_4b_openvino",
        "graph": "C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\eva_ai\\memory\\fractal_graph_v2\\fractal_graph_v2_data\\fractal_graph.db"
    }
    
    print("\n=== Init ===")
    fmf = FMFGeneratorInteractive(paths["model"], paths["graph"], "CPU")
    
    concepts = fmf.graph.get_concepts()
    print(f"Concepts: {len(concepts)}")
    
    print("\n--- Test 1: Interactive Generation ---")
    r1 = fmf.generate_interactive("Расскажи подробно про квантовую механику", max_tokens_per_chunk=512)
    
    with open("C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\test_interactive.txt", "w", encoding="utf-8") as f:
        f.write("=== TEST: Interactive ===\n\n")
        f.write(f"Response:\n{r1['response']}\n\n")
        f.write(f"Need continuation: {r1['need_continuation']}\n")
        f.write(f"Raw contains token: {'<|continue_request|>' in r1['full_raw']}\n")
    
    print(f"Need continuation: {r1['need_continuation']}")
    
    # Если нужно продолжение - симулируем согласие
    if r1['need_continuation']:
        print("\n--- Continue: User agreed ---")
        cont = fmf.continue_generation(user_agreed=True)
        if cont:
            with open("C:\\Users\\black\\OneDrive\\Desktop\\FMF_EVA\\test_interactive.txt", "a", encoding="utf-8") as f:
                f.write(f"\n=== CONTINUATION ===\n\n")
                f.write(cont['response'])
            
            print(f"Continuation received!")
    
    print("\n=== Complete ===")
    print("Saved to test_interactive.txt")
