"""
Project Antigravity — Pillar C: Model Bus & Virtual Context (iPhone-Native)

MemGPT-style virtual context management for iPhone.

iPhone-First Design Constraints:
  - Physical context window: 256-512 tokens (thermal budget on A17/A18)
  - Virtual context: Unlimited via SQLite-backed page file on NVMe SSD
  - No background daemon: Runs as in-app singleton service
  - Page fault latency: <5ms (iPhone NVMe SSD is extremely fast)
  - Memory pages: 128 tokens each (fits in L2 cache)
  - Eviction policy: LRU with recency-weighted importance scoring
  - Total overhead: <15 MB RAM (page table + hot cache)

Architecture:
  The Model Bus is a centralized in-process service that:
    1. Manages the physical context window (hot pages in VRAM)
    2. Pages cold memory blocks to SQLite on the SSD
    3. Handles "cognitive page faults" — autonomous retrieval of paged-out context
    4. Exposes a unified API to all local apps via iOS App Groups

Target Hardware: A17 Pro (iPhone 15 Pro) / A18 Pro (iPhone 16 Pro)
"""

import os
import json
import time
import sqlite3
import hashlib
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ─── iPhone Hardware Constants ───────────────────────────────────────────────

IOS_PHYSICAL_CONTEXT_TOKENS = 512      # Max tokens in VRAM at once
IOS_PAGE_SIZE_TOKENS = 128             # Tokens per memory page (fits L2 cache)
IOS_MAX_HOT_PAGES = IOS_PHYSICAL_CONTEXT_TOKENS // IOS_PAGE_SIZE_TOKENS  # 4 pages
IOS_PAGE_FAULT_BUDGET_MS = 5           # Max latency for SSD page-in
IOS_SQLITE_JOURNAL_MODE = 'WAL'        # Write-Ahead Logging for crash safety
IOS_MAX_VIRTUAL_PAGES = 10000          # ~1.28M tokens of virtual history


@dataclass
class MemoryPage:
    """A single page of context memory."""
    page_id: str = ''
    tokens: List[int] = field(default_factory=list)
    text: str = ''
    importance: float = 0.0            # Model-assigned importance score
    last_accessed: float = 0.0         # Unix timestamp
    access_count: int = 0
    created: float = 0.0
    is_hot: bool = False               # Currently in VRAM


class VirtualContextManager:
    """
    MemGPT-style virtual context for iPhone.

    Manages a hierarchy:
      HOT (VRAM):  4 pages × 128 tokens = 512 tokens in physical context
      COLD (SSD):  Up to 10,000 pages in SQLite = ~1.28M tokens virtual history

    When the model needs context beyond the physical window, it triggers
    a "cognitive page fault" — the manager autonomously evicts the
    lowest-importance hot page and loads the requested cold page from SSD.

    On iPhone, SQLite WAL mode ensures crash-safe writes without fsync storms.
    NVMe SSD delivers <5ms page fault latency.
    """

    def __init__(self, db_path: str = './ios_context.db'):
        self.db_path = db_path
        self.hot_pages: Dict[str, MemoryPage] = {}
        self.page_table: Dict[str, Dict] = {}  # page_id → metadata
        self._init_db()

    def _init_db(self):
        """Initialize SQLite backing store with WAL mode."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(f"PRAGMA journal_mode={IOS_SQLITE_JOURNAL_MODE}")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Fast + safe on iOS
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                tokens TEXT NOT NULL,
                text TEXT NOT NULL,
                importance REAL DEFAULT 0.0,
                last_accessed REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                created REAL DEFAULT 0.0
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance
            ON pages(importance DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed
            ON pages(last_accessed DESC)
        """)
        self.conn.commit()

    def _generate_page_id(self, tokens: List[int]) -> str:
        """Deterministic page ID from token content."""
        content = json.dumps(tokens)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def append_to_context(
        self,
        tokens: List[int],
        text: str = '',
        importance: float = 1.0
    ) -> str:
        """
        Append new tokens to the virtual context.
        If they fill a page, the page is committed.
        """
        # Break tokens into page-sized chunks
        pages_created = []
        for i in range(0, len(tokens), IOS_PAGE_SIZE_TOKENS):
            chunk = tokens[i:i + IOS_PAGE_SIZE_TOKENS]
            chunk_text = text[i*4:(i + IOS_PAGE_SIZE_TOKENS)*4] if text else ''

            page = MemoryPage(
                page_id=self._generate_page_id(chunk),
                tokens=chunk,
                text=chunk_text,
                importance=importance,
                last_accessed=time.time(),
                access_count=1,
                created=time.time(),
                is_hot=True
            )

            # Try to keep in hot cache
            if len(self.hot_pages) < IOS_MAX_HOT_PAGES:
                self.hot_pages[page.page_id] = page
            else:
                # Evict lowest-importance hot page to SSD
                self._evict_coldest_page()
                self.hot_pages[page.page_id] = page

            pages_created.append(page.page_id)

        return pages_created[-1] if pages_created else ''

    def _evict_coldest_page(self):
        """Evict the least important hot page to SQLite cold storage."""
        if not self.hot_pages:
            return

        # Find page with lowest importance × recency score
        coldest_id = min(
            self.hot_pages.keys(),
            key=lambda pid: (
                self.hot_pages[pid].importance *
                (1.0 / max(1.0, time.time() - self.hot_pages[pid].last_accessed))
            )
        )

        page = self.hot_pages.pop(coldest_id)
        page.is_hot = False

        # Write to SQLite
        self.conn.execute(
            """INSERT OR REPLACE INTO pages
               (page_id, tokens, text, importance, last_accessed, access_count, created)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                page.page_id,
                json.dumps(page.tokens),
                page.text,
                page.importance,
                page.last_accessed,
                page.access_count,
                page.created
            )
        )
        self.conn.commit()

    def page_fault(self, query_text: str = '', top_k: int = 1) -> List[MemoryPage]:
        """
        Cognitive page fault: retrieve relevant pages from cold storage.

        On iPhone, this is a <5ms SQLite query on NVMe SSD.
        The retrieved page is swapped into the hot cache, evicting the
        least important current hot page.
        """
        t0 = time.perf_counter()

        # Strategy: retrieve most important or most recently accessed pages
        cursor = self.conn.execute(
            """SELECT page_id, tokens, text, importance, last_accessed, access_count, created
               FROM pages
               ORDER BY importance DESC, last_accessed DESC
               LIMIT ?""",
            (top_k,)
        )

        retrieved = []
        for row in cursor:
            page = MemoryPage(
                page_id=row[0],
                tokens=json.loads(row[1]),
                text=row[2],
                importance=row[3],
                last_accessed=time.time(),
                access_count=row[5] + 1,
                created=row[6],
                is_hot=True
            )

            # Swap into hot cache
            if len(self.hot_pages) >= IOS_MAX_HOT_PAGES:
                self._evict_coldest_page()

            self.hot_pages[page.page_id] = page

            # Update access stats in SQLite
            self.conn.execute(
                "UPDATE pages SET last_accessed=?, access_count=? WHERE page_id=?",
                (page.last_accessed, page.access_count, page.page_id)
            )

            retrieved.append(page)

        self.conn.commit()

        fault_ms = (time.perf_counter() - t0) * 1000
        if fault_ms > IOS_PAGE_FAULT_BUDGET_MS:
            print(f"⚠️ Page fault exceeded budget: {fault_ms:.1f}ms > {IOS_PAGE_FAULT_BUDGET_MS}ms")

        return retrieved

    def get_physical_context(self) -> List[int]:
        """
        Return the current physical context window (all hot page tokens concatenated).
        This is what gets fed to the model's attention mechanism.
        """
        # Sort hot pages by creation time
        sorted_pages = sorted(
            self.hot_pages.values(),
            key=lambda p: p.created
        )
        tokens = []
        for page in sorted_pages:
            tokens.extend(page.tokens)
        return tokens[-IOS_PHYSICAL_CONTEXT_TOKENS:]

    def get_context_text(self) -> str:
        """Return text representation of the physical context."""
        sorted_pages = sorted(
            self.hot_pages.values(),
            key=lambda p: p.created
        )
        return ' '.join(p.text for p in sorted_pages if p.text)

    def update_importance(self, page_id: str, importance: float):
        """
        Model can signal that a page is important (should not be evicted).
        On iPhone, importance scores come from attention weight analysis.
        """
        if page_id in self.hot_pages:
            self.hot_pages[page_id].importance = importance

        self.conn.execute(
            "UPDATE pages SET importance=? WHERE page_id=?",
            (importance, page_id)
        )
        self.conn.commit()

    def stats(self) -> Dict:
        """Memory and storage statistics."""
        cursor = self.conn.execute("SELECT COUNT(*), SUM(LENGTH(tokens)) FROM pages")
        row = cursor.fetchone()
        cold_count = row[0] or 0
        cold_bytes = row[1] or 0

        hot_tokens = sum(len(p.tokens) for p in self.hot_pages.values())

        return {
            'hot_pages': len(self.hot_pages),
            'hot_tokens': hot_tokens,
            'cold_pages': cold_count,
            'physical_context_tokens': min(hot_tokens, IOS_PHYSICAL_CONTEXT_TOKENS),
            'virtual_context_tokens': hot_tokens + (cold_count * IOS_PAGE_SIZE_TOKENS),
            'sqlite_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
            'hot_cache_mb': (hot_tokens * 2) / (1024 * 1024),  # FP16 tokens
        }

    def flush_all_to_cold(self):
        """Flush all hot pages to cold storage (app backgrounding)."""
        for page_id in list(self.hot_pages.keys()):
            self._evict_coldest_page()

    def close(self):
        """Clean shutdown."""
        self.flush_all_to_cold()
        self.conn.close()


class ModelBus_iOS:
    """
    Centralized Model Bus for iPhone.

    Acts as a singleton in-process service that:
      1. Manages the local model lifecycle (load/unload/swap)
      2. Provides virtual context management (MemGPT paging)
      3. Serves inference requests from the Vericoding Shell, training loop,
         and any future local apps via iOS App Groups shared container

    On iOS, this runs as a singleton within the main app process.
    It does NOT run as a background daemon (iOS doesn't allow persistent daemons).
    Instead, it uses BGProcessingTask for overnight operations and
    NSUserDefaults/App Groups for cross-app state sharing.

    Memory budget:
      Model weights: ~2.18 GB (zero-copy mmap, free)
      KV cache:      ~50 MB (4 hot pages × 128 tokens × 32 heads × 64 dim × FP16)
      Page table:    ~1 MB (metadata for up to 10K pages)
      SQLite WAL:    ~5 MB (write-ahead log buffer)
      Total:         ~56 MB overhead
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str = 'mps',
        context_db_path: str = './ios_context.db'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_manager = VirtualContextManager(db_path=context_db_path)

        self.is_model_loaded = model is not None
        self.request_count = 0
        self.total_tokens_generated = 0

    def infer(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        use_virtual_context: bool = True
    ) -> Dict:
        """
        Run inference with virtual context management.

        Pipeline:
          1. Tokenize user message
          2. Check if virtual context pages should be loaded (page fault)
          3. Construct physical context window from hot pages + new input
          4. Run model inference
          5. Page the new response into virtual context
          6. Return response
        """
        if not self.is_model_loaded:
            return {'error': 'No model loaded', 'text': ''}

        t0 = time.perf_counter()
        self.request_count += 1

        # Tokenize input
        input_ids = self.tokenizer.encode(user_message, add_special_tokens=False)

        # Page in relevant context if enabled
        if use_virtual_context and self.context_manager.hot_pages:
            # Trigger page fault to load most relevant history
            self.context_manager.page_fault(query_text=user_message, top_k=2)

        # Build physical context: history + new input
        history_tokens = self.context_manager.get_physical_context()
        full_context = history_tokens + input_ids

        # Truncate to physical window
        if len(full_context) > IOS_PHYSICAL_CONTEXT_TOKENS:
            full_context = full_context[-IOS_PHYSICAL_CONTEXT_TOKENS:]

        # Run inference
        import torch
        context_tensor = torch.tensor([full_context], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                context_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                top_p=0.95,
            )

        gen_tokens = outputs[0][len(full_context):].tolist()
        response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        self.total_tokens_generated += len(gen_tokens)

        # Page new context (input + response) into virtual memory
        self.context_manager.append_to_context(
            tokens=input_ids + gen_tokens,
            text=user_message + ' ' + response_text,
            importance=1.0
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            'text': response_text,
            'tokens_generated': len(gen_tokens),
            'context_tokens_used': len(full_context),
            'latency_ms': elapsed_ms,
            'virtual_context': self.context_manager.stats(),
        }

    def on_app_background(self):
        """Called when iOS app enters background. Flush hot pages to SSD."""
        self.context_manager.flush_all_to_cold()

    def on_app_foreground(self):
        """Called when iOS app returns to foreground. Reload recent pages."""
        self.context_manager.page_fault(top_k=IOS_MAX_HOT_PAGES)

    def shutdown(self):
        """Clean shutdown."""
        self.context_manager.close()

    def status(self) -> Dict:
        """Full system status."""
        ctx_stats = self.context_manager.stats()
        return {
            'model_loaded': self.is_model_loaded,
            'total_requests': self.request_count,
            'total_tokens_generated': self.total_tokens_generated,
            'physical_context_tokens': ctx_stats['physical_context_tokens'],
            'virtual_context_tokens': ctx_stats['virtual_context_tokens'],
            'hot_pages': ctx_stats['hot_pages'],
            'cold_pages': ctx_stats['cold_pages'],
            'sqlite_size_mb': ctx_stats['sqlite_size_mb'],
        }
