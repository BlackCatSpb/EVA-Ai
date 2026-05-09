"""
Scenario TCM - Эпизодическая память

STCM: Сохраняет цепочки диалогов как сценарии.

Features:
- Сценарное тегирование
- Семантический поиск похожих сценариев
- Интеграция с TCM для cross-referencing
- Кластеризация сценариев по доменам
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from uuid import uuid4
import time
import json
import os


class ScenarioTCM:
    """
    STCM: Эпизодическая память для сценариев.
    
    Сохраняет цепочки диалогов и поддерживает:
    - Семантический поиск
    - Тегирование по доменам
    - Интеграция с основной TCM
    """

    def __init__(self, graph=None, storage_dir: str = None):
        self.graph = graph
        self.current_chain: List[Dict] = []
        self.storage_dir = storage_dir or self._get_default_storage_dir()
        self._scenario_cache: Dict[str, Dict] = {}
        
        self._load_scenarios()
    
    def _get_default_storage_dir(self) -> str:
        """Получить директорию хранения."""
        base = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(base, "scenario_tcm_data")
    
    def _load_scenarios(self):
        """Загрузить сохранённые сценарии."""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            cache_file = os.path.join(self.storage_dir, "scenarios.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self._scenario_cache = json.load(f)
        except Exception:
            self._scenario_cache = {}
    
    def _save_scenarios(self):
        """Сохранить сценарии."""
        try:
            cache_file = os.path.join(self.storage_dir, "scenarios.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._scenario_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def add_turn(
        self,
        role: str,
        text: str,
        embedding: np.ndarray,
        metadata: Dict = None
    ):
        """
        STCM: Добавить ход диалога с метаданными.
        """
        turn = {
            "role": role,
            "text": text,
            "emb": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        self.current_chain.append(turn)

        if self._is_end(text):
            self._save_chain()

    def _is_end(self, text: str) -> bool:
        """Определить конец сценария."""
        end_keywords = ["спасибо", "пока", "новый вопрос", "до свидания", "конец"]
        text_lower = text.lower()
        return any(kw in text_lower for kw in end_keywords)
    
    def add_turn_batch(self, turns: List[Dict]):
        """STCM: Добавить несколько ходов сразу."""
        for turn in turns:
            role = turn.get('role', 'unknown')
            text = turn.get('text', '')
            emb = turn.get('embedding', np.zeros(2560))
            metadata = turn.get('metadata', {})
            
            self.add_turn(role, text, emb, metadata)
    
    def tag_scenario(self, scenario_id: str, tags: List[str]):
        """STCM: Тегировать сценарий."""
        if scenario_id in self._scenario_cache:
            self._scenario_cache[scenario_id]['tags'] = tags
            self._save_scenarios()
    
    def set_scenario_domain(self, scenario_id: str, domain: str):
        """STCM: Установить домен сценария."""
        if scenario_id in self._scenario_cache:
            self._scenario_cache[scenario_id]['domain'] = domain
            self._save_scenarios()
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[Dict]:
        """Получить сценарий по ID."""
        return self._scenario_cache.get(scenario_id)
    
    def get_scenarios_by_domain(self, domain: str) -> List[Dict]:
        """STCM: Получить все сценарии домена."""
        return [
            s for s in self._scenario_cache.values()
            if s.get('domain') == domain
        ]
    
    def find_similar_scenarios(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        domain: str = None
    ) -> List[Tuple[Dict, float]]:
        """
        STCM: Найти похожие сценарии по эмбеддингу.
        
        Returns:
            [(scenario, similarity), ...]
        """
        candidates = self._scenario_cache.values()
        if domain:
            candidates = [s for s in candidates if s.get('domain') == domain]
        
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for scenario in candidates:
            chain = scenario.get('chain', [])
            if not chain or len(chain) < 2:
                continue
            
            emb = np.array(chain[-1].get('emb', [0] * 2560))
            sim = float(np.dot(emb, query_embedding) / (
                np.linalg.norm(emb) * query_norm + 1e-8
            ))
            similarities.append((scenario, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_recent_scenarios(self, k: int = 5) -> List[Dict]:
        """Получить недавние сценарии."""
        sorted_scenarios = sorted(
            self._scenario_cache.items(),
            key=lambda x: x[1].get('timestamp', 0),
            reverse=True
        )
        return [s[1] for s in sorted_scenarios[:k]]
    
    def cluster_scenarios_by_theme(self) -> Dict[str, List[str]]:
        """
        STCM: Кластеризовать сценарии по темам.
        
        Returns:
            {"theme_name": [scenario_ids, ...]}
        """
        theme_clusters = {}
        
        for scenario_id, scenario in self._scenario_cache.items():
            tags = scenario.get('tags', [])
            domain = scenario.get('domain', 'general')
            
            cluster_key = f"{domain}:{','.join(tags[:2])}"
            if cluster_key not in theme_clusters:
                theme_clusters[cluster_key] = []
            theme_clusters[cluster_key].append(scenario_id)
        
        return theme_clusters
    
    def _save_chain(self):
        """Сохранить цепочку в граф."""
        if not self.current_chain:
            return
        
        scenario_id = str(uuid4())
        
        scenario_data = {
            "id": scenario_id,
            "chain": self.current_chain.copy(),
            "timestamp": time.time(),
            "turn_count": len(self.current_chain),
            "domain": self._infer_domain(self.current_chain),
            "tags": self._extract_tags(self.current_chain)
        }
        
        self._scenario_cache[scenario_id] = scenario_data
        self._save_scenarios()
        
        prev_id = None

        for turn in self.current_chain:
            emb_array = np.array(turn["emb"]) if isinstance(turn["emb"], list) else turn["emb"]
            
            if self.graph:
                node_id = self.graph.add_node(
                    content=turn["text"],
                    node_type="scenario_turn",
                    embedding=emb_array.tobytes() if hasattr(emb_array, 'tobytes') else emb_array
                )
                if prev_id:
                    self.graph.add_edge(prev_id, node_id, "next_turn")
                prev_id = node_id

        self.current_chain.clear()
    
    def _infer_domain(self, chain: List[Dict]) -> str:
        """STCM: Определить домен сценария."""
        text_sample = " ".join([t.get('text', '')[:100] for t in chain[:3]])
        
        domain_keywords = {
            "technical": ["код", "программ", "ошибк", "debug", "api"],
            "creative": ["стих", "история", "написать", "творч"],
            "reasoning": ["почему", "объясни", "как работает", "логик"],
            "factual": ["факт", "данные", "число", "статистик"]
        }
        
        text_lower = text_sample.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        
        return "general"
    
    def _extract_tags(self, chain: List[Dict]) -> List[str]:
        """STCM: Извлечь теги из сценария."""
        full_text = " ".join([t.get('text', '') for t in chain])
        
        tag_patterns = [
            (r'python', 'python'),
            (r'javascript|js', 'javascript'),
            (r'машинн\w* обуч', 'ml'),
            (r'нейронн\w* сеть', 'neural_networks'),
            (r'веб\s*-?\s*дизайн', 'web_design'),
            (r'алгоритм', 'algorithms')
        ]
        
        tags = []
        for pattern, tag in tag_patterns:
            import re
            if re.search(pattern, full_text, re.IGNORECASE):
                tags.append(tag)
        
        return tags[:5]

    def get_current_chain(self) -> List[Dict]:
        """Получить текущую цепочку."""
        return self.current_chain.copy()

    def clear(self):
        """Очистить текущую цепочку."""
        self.current_chain.clear()
    
    def get_statistics(self) -> Dict:
        """STCM: Получить статистику сценариев."""
        total = len(self._scenario_cache)
        by_domain = {}
        
        for s in self._scenario_cache.values():
            domain = s.get('domain', 'unknown')
            by_domain[domain] = by_domain.get(domain, 0) + 1
        
        return {
            "total_scenarios": total,
            "by_domain": by_domain,
            "avg_turns": sum(s.get('turn_count', 0) for s in self._scenario_cache.values()) / max(total, 1),
            "storage_size": len(json.dumps(self._scenario_cache))
        }
    
    def integrate_with_tcm(self, tcm_instance) -> bool:
        """
        STCM: Интегрировать с основной TCM.
        
        Позволяет использовать cross-referencing между сценариями и памятью.
        """
        if tcm_instance is None:
            return False
        
        for scenario_id, scenario in self._scenario_cache.items():
            chain = scenario.get('chain', [])
            if not chain:
                continue
            
            for turn in chain:
                emb = np.array(turn.get('emb', [0] * 2560))
                role = turn.get('role', 'unknown')
                text = turn.get('text', '')
                
                if hasattr(tcm_instance, 'write'):
                    tcm_instance.write(
                        text=text,
                        segment_type=role,
                        embedding=emb
                    )
        
        return True


class ScenarioMemory:
    """Память сценариев с поиском."""

    def __init__(self):
        self.scenarios: List[Dict] = []

    def add_scenario(self, chain: List[Dict]):
        """Добавить сценарий."""
        self.scenarios.append({
            "id": str(uuid4()),
            "chain": chain,
            "timestamp": time.time()
        })

    def search(self, query_emb: np.ndarray, k: int = 3) -> List[Dict]:
        """Поиск похожих сценариев."""
        results = []

        for scenario in self.scenarios:
            embs = [t["emb"] for t in scenario["chain"] if "emb" in t]
            if not embs:
                continue

            mean_emb = np.mean(embs, axis=0)

            sim = np.dot(query_emb, mean_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(mean_emb) + 1e-8)

            results.append((sim, scenario))

        results.sort(key=lambda x: x[0], reverse=True)

        return [r[1] for r in results[:k]]
