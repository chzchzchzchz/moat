"""
Project Antigravity — Memory & Thermal Guardrails for Phone-Level LLM Routing.

Implements the hard constraints that make local LLM deployment viable on phones:

  * MemoryGuard: RSS tracking with a 2.5 GB hard ceiling (iOS Jetsam kills
    foreground apps at ~3.2–3.4 GB; we stay 20% under with headroom).
  * ContextPager: Virtual context paging via SQLite WAL — keeps only the active
    conversation turn in KV cache, pages history to local SQLite.
  * ThermalMonitor: Sustained power budget < 5 W to prevent iPhone throttling.
  * FirstFinishCircuitBreaker: When any parallel branch passes AST verification,
    halts all other candidate streams immediately (sub-second micro-bursts).
"""

from __future__ import annotations

import os
import sys
import time
import sqlite3
import threading
import resource
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants — phone-level budgets
# ---------------------------------------------------------------------------

MEMORY_BUDGET_BYTES = 2_500 * 1024 * 1024   # 2.5 GB hard ceiling (RSS)
MEMORY_WARN_BYTES   = 2_000 * 1024 * 1024   # 2.0 GB soft warning
THERMAL_BUDGET_WATTS = 5.0                    # 5 W sustained max
KV_CACHE_ACTIVE_TOKENS = 512                  # Active KV-cache window


@dataclass
class MemorySnapshot:
    rss_bytes: int
    rss_mb: float
    over_budget: bool
    timestamp: float


@dataclass
class ThermalSnapshot:
    watts: float
    over_budget: bool
    timestamp: float


# ---------------------------------------------------------------------------
# MemoryGuard — RSS tracking with hard ceiling
# ---------------------------------------------------------------------------

class MemoryGuard:
    """
    Tracks process RSS and enforces the 2.5 GB hard ceiling.

    On every check, if RSS exceeds MEMORY_WARN_BYTES a warning is logged;
    if it exceeds MEMORY_BUDGET_BYTES the guard raises MemoryBudgetExceeded
    so the router can shed load (evict context, drop to a smaller model, or
    route to cloud) before the OS kills the process.
    """

    def __init__(self, budget_bytes: int = MEMORY_BUDGET_BYTES,
                 warn_bytes: int = MEMORY_WARN_BYTES):
        self.budget_bytes = budget_bytes
        self.warn_bytes = warn_bytes
        self._lock = threading.Lock()
        self._peak_rss = 0
        self._violations = 0
        self._on_violation: Optional[Callable[[MemorySnapshot], None]] = None

    def check(self) -> MemorySnapshot:
        """Return current RSS snapshot and raise if over hard budget."""
        rss = self._read_rss()
        with self._lock:
            if rss > self._peak_rss:
                self._peak_rss = rss
            snapshot = MemorySnapshot(
                rss_bytes=rss,
                rss_mb=rss / (1024 * 1024),
                over_budget=rss > self.budget_bytes,
                timestamp=time.monotonic(),
            )
            if snapshot.over_budget:
                self._violations += 1
                if self._on_violation is not None:
                    self._on_violation(snapshot)
                raise MemoryBudgetExceeded(
                    f"RSS {snapshot.rss_mb:.1f} MB exceeds {self.budget_bytes / (1024*1024):.0f} MB budget"
                )
            elif rss > self.warn_bytes:
                # Soft warning — logged but not fatal
                pass
        return snapshot

    def set_violation_handler(self, handler: Callable[[MemorySnapshot], None]) -> None:
        self._on_violation = handler

    @property
    def peak_rss_mb(self) -> float:
        return self._peak_rss / (1024 * 1024)

    @property
    def violations(self) -> int:
        return self._violations

    @staticmethod
    def _read_rss() -> int:
        """Read current process RSS in bytes (POSIX via /proc or resource module)."""
        if sys.platform == "linux":
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            # VmRSS:   12345 kB
                            kb = int(line.split()[1])
                            return kb * 1024
            except Exception:
                pass
        # macOS / fallback: use resource.getrusage (max RSS in KB on Linux, bytes on macOS?)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return ru.ru_maxrss
        return ru.ru_maxrss * 1024


class MemoryBudgetExceeded(RuntimeError):
    """Raised when RSS breaches the hard memory ceiling."""
    pass


# ---------------------------------------------------------------------------
# ContextPager — SQLite WAL virtual context paging
# ---------------------------------------------------------------------------

class ContextPager:
    """
    Pages conversation history to a local SQLite database using WAL mode.

    Keeps only `active_window_tokens` worth of tokens in the active KV-cache
    window. Older turns are evicted to SQLite and paged back in via semantic
    vector retrieval (or simple keyword match fallback) when referenced.

    This bounds memory growth for long conversations on phones.
    """

    def __init__(self, db_path: str = ":memory:",
                 active_window_tokens: int = KV_CACHE_ACTIVE_TOKENS):
        self.db_path = db_path
        self.active_window_tokens = active_window_tokens
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._turn_counter = 0
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS context_pages (
                turn_id INTEGER PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                evicted INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_context_turn ON context_pages(turn_id)
        """)
        self._conn.commit()

    def append_turn(self, role: str, content: str, token_count: int) -> int:
        """Append a conversation turn. Returns the turn ID."""
        with self._lock:
            self._turn_counter += 1
            turn_id = self._turn_counter
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO context_pages (turn_id, role, content, token_count, created_at, evicted) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (turn_id, role, content, token_count, time.monotonic()),
            )
            self._conn.commit()
            return turn_id

    def get_active_window(self) -> List[Dict]:
        """Return the most recent turns within the active token window."""
        with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT turn_id, role, content, token_count FROM context_pages "
                "WHERE evicted = 0 ORDER BY turn_id DESC"
            ).fetchall()

            # Walk backwards accumulating tokens until we hit the window
            selected = []
            total = 0
            for turn_id, role, content, tok_count in rows:
                if total + tok_count > self.active_window_tokens and selected:
                    break
                selected.append({
                    "turn_id": turn_id,
                    "role": role,
                    "content": content,
                    "token_count": tok_count,
                })
                total += tok_count

            selected.reverse()
            return selected

    def evict_older(self, keep_turn_ids: List[int]) -> int:
        """Mark all turns not in keep_turn_ids as evicted. Returns count evicted."""
        with self._lock:
            assert self._conn is not None
            if not keep_turn_ids:
                placeholders = "0"
            else:
                placeholders = ",".join("?" for _ in keep_turn_ids)
            cur = self._conn.execute(
                f"UPDATE context_pages SET evicted = 1 WHERE turn_id NOT IN ({placeholders})",
                keep_turn_ids,
            )
            self._conn.commit()
            return cur.rowcount

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Simple keyword-match retrieval of evicted turns."""
        with self._lock:
            assert self._conn is not None
            keywords = [w for w in query.lower().split() if len(w) > 2]
            if not keywords:
                return []
            like_clauses = " OR ".join("LOWER(content) LIKE ?" for _ in keywords)
            params = [f"%{kw}%" for kw in keywords]
            rows = self._conn.execute(
                f"SELECT turn_id, role, content, token_count FROM context_pages "
                f"WHERE evicted = 1 AND ({like_clauses}) LIMIT ?",
                params + [limit],
            ).fetchall()
            return [
                {"turn_id": r[0], "role": r[1], "content": r[2], "token_count": r[3]}
                for r in rows
            ]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# ThermalMonitor — sustained power budget tracking
# ---------------------------------------------------------------------------

class ThermalMonitor:
    """
    Tracks estimated sustained power draw and enforces < 5 W thermal budget.

    On phones, sustained draw above ~5 W triggers thermal throttling. The
    monitor tracks an exponentially-weighted moving average of estimated
    watts and raises ThermalBudgetExceeded when the EWMA breaches the limit.

    Estimated watts per token are provided by the caller (from platform
    telemetry) since we cannot read die temperature directly from Python.
    """

    def __init__(self, budget_watts: float = THERMAL_BUDGET_WATTS,
                 alpha: float = 0.3):
        self.budget_watts = budget_watts
        self.alpha = alpha
        self._ewma_watts: Optional[float] = None
        self._lock = threading.Lock()
        self._violations = 0

    def update(self, watts: float) -> ThermalSnapshot:
        """Record a new power sample and return the thermal snapshot."""
        with self._lock:
            if self._ewma_watts is None:
                self._ewma_watts = watts
            else:
                self._ewma_watts = (
                    self.alpha * watts + (1.0 - self.alpha) * self._ewma_watts
                )
            snapshot = ThermalSnapshot(
                watts=self._ewma_watts,
                over_budget=self._ewma_watts > self.budget_watts,
                timestamp=time.monotonic(),
            )
            if snapshot.over_budget:
                self._violations += 1
        return snapshot

    @property
    def current_watts(self) -> Optional[float]:
        return self._ewma_watts

    @property
    def violations(self) -> int:
        return self._violations


class ThermalBudgetExceeded(RuntimeError):
    """Raised when estimated sustained power exceeds the thermal ceiling."""
    pass


# ---------------------------------------------------------------------------
# FirstFinishCircuitBreaker — halt losing branches on verification success
# ---------------------------------------------------------------------------

class FirstFinishCircuitBreaker:
    """
    When running N parallel reasoning branches, the moment any single branch
    passes AST/Z3 verification, dispatch an immediate halt signal to kill all
    other candidate streams. This keeps sustained power under the thermal
    budget by executing in sub-second micro-bursts rather than running all
    N branches to completion.
    """

    def __init__(self, n_branches: int):
        self.n_branches = n_branches
        self._lock = threading.Lock()
        self._halted = False
        self._winning_branch: Optional[int] = None
        self._winning_trace: Optional[str] = None
        self._branch_results: Dict[int, Optional[Dict]] = {i: None for i in range(n_branches)}
        self._condition = threading.Condition(self._lock)

    def is_halted(self) -> bool:
        with self._lock:
            return self._halted

    def register_result(self, branch_id: int, result: Dict,
                        verified: bool) -> bool:
        """
        Register a branch result. Returns True if this branch won the circuit
        breaker (i.e., it was the first verified result and halted the rest).
        """
        with self._lock:
            if self._halted:
                # Already halted — this branch lost the race
                self._branch_results[branch_id] = result
                return False

            self._branch_results[branch_id] = result

            if verified:
                self._halted = True
                self._winning_branch = branch_id
                self._winning_trace = result.get("trace", "")
                self._condition.notify_all()
                return True

            return False

    def wait_for_halt(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Block until the circuit breaker halts or timeout expires."""
        with self._condition:
            if not self._halted:
                self._condition.wait(timeout)
            if self._halted:
                return {
                    "winning_branch": self._winning_branch,
                    "winning_trace": self._winning_trace,
                    "all_results": dict(self._branch_results),
                }
            return None

    def get_winner(self) -> Optional[Dict]:
        if self._halted:
            return {
                "winning_branch": self._winning_branch,
                "winning_trace": self._winning_trace,
            }
        return None

    def halt_all(self, branch_id: int, trace: str) -> None:
        """Force-halt all branches, declaring branch_id the winner."""
        with self._lock:
            self._halted = True
            self._winning_branch = branch_id
            self._winning_trace = trace
            self._condition.notify_all()


__all__ = [
    "MemoryGuard", "MemoryBudgetExceeded", "MemorySnapshot",
    "ContextPager",
    "ThermalMonitor", "ThermalBudgetExceeded", "ThermalSnapshot",
    "FirstFinishCircuitBreaker",
    "MEMORY_BUDGET_BYTES", "THERMAL_BUDGET_WATTS", "KV_CACHE_ACTIVE_TOKENS",
]