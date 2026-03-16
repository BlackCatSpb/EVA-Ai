"""
SQLite-backed index for addressable routing of batches/segments/token-nodes.
Location: hybrid_cache/disk_storage/cache_index.db under brain.cache_dir.
"""
from __future__ import annotations
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
logger = logging.getLogger(__name__)

class CacheIndex:
    """Maintains an addressable index for the hybrid token cache.

    Schema:
      - batches(batch_id TEXT PRIMARY KEY, source TEXT, created_ts REAL, total_tokens INTEGER, priority REAL, status TEXT)
      - segments(segment_id TEXT PRIMARY KEY, batch_id TEXT, offset INTEGER, length INTEGER,
                 token_count INTEGER, disk_path TEXT, checksum TEXT,
                 FOREIGN KEY(batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE)
      - token_nodes(node_id TEXT PRIMARY KEY, segment_id TEXT, token_start INTEGER, token_end INTEGER,
                    node_hash TEXT, kg_node_id TEXT NULL,
                    FOREIGN KEY(segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE)
      - weights(item_id TEXT, item_type TEXT, weight_type TEXT, value REAL, context_id TEXT,
                PRIMARY KEY(item_id, item_type, weight_type, COALESCE(context_id, '')))
    Indexes on (batch_id), (segment_id), (node_hash), (weight_type, value DESC)
    """

    def __init__(self, brain):
        self.brain = brain
        base_dir = os.path.join(brain.cache_dir, "hybrid_cache", "disk_storage")
        os.makedirs(base_dir, exist_ok=True)
        self.db_path = os.path.join(base_dir, "cache_index.db")
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA busy_timeout=5000;")
            finally:
                cur.close()
        self._create_schema()

    def _exec(self, sql: str, params: Tuple = (), commit: bool = False):
        attempts = 0
        last: Optional[Exception] = None
        while attempts < 5:
            with self._lock:
                try:
                    cur = self.conn.cursor()
                    try:
                        cur.execute(sql, params)
                        if commit:
                            self.conn.commit()
                        return cur
                    finally:
                        cur.close()
                except sqlite3.OperationalError as e:
                    last = e
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        time.sleep(0.05 * (attempts + 1))
                        attempts += 1
                        continue
                    raise
        if last:
            raise last

    def _create_schema(self):
        with self._lock:
            c = self.conn.cursor()
            try:
                c.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS batches (
                        batch_id TEXT PRIMARY KEY,
                        source TEXT,
                        created_ts REAL,
                        total_tokens INTEGER,
                        priority REAL,
                        status TEXT
                    );
                    CREATE TABLE IF NOT EXISTS segments (
                        segment_id TEXT PRIMARY KEY,
                        batch_id TEXT,
                        offset INTEGER,
                        length INTEGER,
                        token_count INTEGER,
                        disk_path TEXT,
                        checksum TEXT,
                        FOREIGN KEY(batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS token_nodes (
                        node_id TEXT PRIMARY KEY,
                        segment_id TEXT,
                        token_start INTEGER,
                        token_end INTEGER,
                        node_hash TEXT,
                        kg_node_id TEXT NULL,
                        FOREIGN KEY(segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS weights (
                        item_id TEXT,
                        item_type TEXT,
                        weight_type TEXT,
                        value REAL,
                        context_id TEXT NOT NULL DEFAULT '',
                        PRIMARY KEY(item_id, item_type, weight_type, context_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_segments_batch ON segments(batch_id);
                    CREATE INDEX IF NOT EXISTS idx_nodes_segment ON token_nodes(segment_id);
                    CREATE INDEX IF NOT EXISTS idx_nodes_hash ON token_nodes(node_hash);
                    CREATE INDEX IF NOT EXISTS idx_weights_type_val ON weights(weight_type, value DESC);
                    """
                )
                self.conn.commit()
                # ---- Migration: remove expressions from PRIMARY KEY on weights ----
                try:
                    cur = self.conn.cursor()
                    try:
                        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='weights';")
                        row = cur.fetchone()
                        sql_def = row[0] if row else ''
                        if 'COALESCE(context_id' in (sql_def or ''):
                            logger.info("CacheIndex: migrating 'weights' table schema to remove COALESCE from PRIMARY KEY")
                            self.conn.execute("BEGIN IMMEDIATE;")
                            self.conn.execute(
                                """
                                CREATE TABLE IF NOT EXISTS weights_new (
                                    item_id TEXT,
                                    item_type TEXT,
                                    weight_type TEXT,
                                    value REAL,
                                    context_id TEXT NOT NULL DEFAULT '',
                                    PRIMARY KEY(item_id, item_type, weight_type, context_id)
                                );
                                """
                            )
                            self.conn.execute(
                                "INSERT OR REPLACE INTO weights_new(item_id, item_type, weight_type, value, context_id)\n"
                                "SELECT item_id, item_type, weight_type, value, COALESCE(context_id, '') FROM weights;"
                            )
                            self.conn.execute("DROP TABLE weights;")
                            self.conn.execute("ALTER TABLE weights_new RENAME TO weights;")
                            self.conn.execute(
                                "CREATE INDEX IF NOT EXISTS idx_weights_type_val ON weights(weight_type, value DESC);"
                            )
                            self.conn.commit()
                    finally:
                        try:
                            cur.close()
                        except Exception:
                            pass
                except Exception as mig_err:
                    logger.warning(f"CacheIndex: migration check failed or skipped: {mig_err}")
            finally:
                c.close()

    # ---- Public API ----
    def upsert_batch(self, batch_id: str, source: str, total_tokens: int, priority: float = 0.0, status: str = "queued") -> None:
        self._exec(
            """INSERT INTO batches(batch_id, source, created_ts, total_tokens, priority, status)
                 VALUES(?, ?, ?, ?, ?, ?)
                 ON CONFLICT(batch_id) DO UPDATE SET source=excluded.source, total_tokens=excluded.total_tokens,
                   priority=excluded.priority, status=excluded.status""",
            (batch_id, source, time.time(), int(total_tokens), float(priority), str(status)),
            commit=True,
        )

    def upsert_segment(self, segment_id: str, batch_id: str, offset: int, length: int, token_count: int,
                       disk_path: str, checksum: str = "") -> None:
        self._exec(
            """INSERT INTO segments(segment_id, batch_id, offset, length, token_count, disk_path, checksum)
                 VALUES(?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(segment_id) DO UPDATE SET batch_id=excluded.batch_id, offset=excluded.offset,
                   length=excluded.length, token_count=excluded.token_count, disk_path=excluded.disk_path,
                   checksum=excluded.checksum""",
            (segment_id, batch_id, int(offset), int(length), int(token_count), str(disk_path), str(checksum)),
            commit=True,
        )

    def add_token_nodes(self, nodes: Sequence[Tuple[str, str, int, int, str]]):
        # (node_id, segment_id, token_start, token_end, node_hash)
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.executemany(
                    """INSERT OR IGNORE INTO token_nodes(node_id, segment_id, token_start, token_end, node_hash)
                        VALUES(?, ?, ?, ?, ?)""",
                    [(nid, sid, int(ts), int(te), h) for (nid, sid, ts, te, h) in nodes],
                )
                self.conn.commit()
            finally:
                cur.close()

    def link_nodes_to_kg(self, pairs: Sequence[Tuple[str, str]]):
        # (node_id, kg_node_id)
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.executemany(
                    "UPDATE token_nodes SET kg_node_id = ? WHERE node_id = ?",
                    [(kg, nid) for (nid, kg) in pairs],
                )
                self.conn.commit()
            finally:
                cur.close()

    def set_weight(self, item_id: str, item_type: str, weight_type: str, value: float, context_id: Optional[str] = None):
        ctx = context_id or ''
        self._exec(
            """INSERT INTO weights(item_id, item_type, weight_type, value, context_id)
                 VALUES(?, ?, ?, ?, ?)
                 ON CONFLICT(item_id, item_type, weight_type, context_id)
                 DO UPDATE SET value=excluded.value""",
            (item_id, item_type, weight_type, float(value), ctx), commit=True,
        )

    def rank_segments(self, top_k: int = 10, context_id: Optional[str] = None, weight_type: str = "task_relevance") -> List[str]:
        sql = (
            "SELECT s.segment_id FROM segments s "
            "LEFT JOIN weights w ON w.item_id = s.segment_id AND w.item_type = 'segment' AND w.weight_type = ? "
        )
        params: List[Any] = [weight_type]
        if context_id:
            sql += "AND (w.context_id = ? OR w.context_id = '') "
            params.append(context_id)
        sql += "ORDER BY COALESCE(w.value, 0.0) DESC LIMIT ?"
        params.append(int(top_k))
        cur = self._exec(sql, tuple(params))
        try:
            rows = cur.fetchall() if cur else []
            return [r[0] for r in rows]
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def address_of(self, segment_id: str, token_start: Optional[int] = None, token_end: Optional[int] = None) -> str:
        base = f"cache://batch/seg/{segment_id}"
        if token_start is not None and token_end is not None:
            base += f"#tokens={token_start}:{token_end}"
        return base

    def get_segment_path(self, segment_id: str) -> Optional[str]:
        cur = self._exec("SELECT disk_path FROM segments WHERE segment_id = ?", (segment_id,))
        try:
            row = cur.fetchone() if cur else None
            return row[0] if row else None
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def close(self):
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass
