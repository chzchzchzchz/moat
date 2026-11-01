"""
Project Antigravity — Tests for memory_guard.py

Tests the MemoryGuard RSS ceiling, ContextPager virtual context paging,
ThermalMonitor power budget, and FirstFinishCircuitBreaker.
"""

import sys
import os
import time
import tempfile
import threading
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from memory_guard import (
    MemoryGuard, MemoryBudgetExceeded, MemorySnapshot,
    ContextPager, ThermalMonitor, ThermalBudgetExceeded, ThermalSnapshot,
    FirstFinishCircuitBreaker,
    MEMORY_BUDGET_BYTES,
)


# ---------------------------------------------------------------------------
# MemoryGuard
# ---------------------------------------------------------------------------

def test_memory_guard_under_budget():
    """Normal RSS readings pass without raising."""
    guard = MemoryGuard(budget_bytes=MEMORY_BUDGET_BYTES)
    snap = guard.check()
    assert isinstance(snap, MemorySnapshot)
    assert snap.rss_bytes > 0
    assert snap.rss_mb > 0
    assert snap.over_budget is False


def test_memory_guard_over_budget_raises():
    """RSS above hard budget raises MemoryBudgetExceeded."""
    guard = MemoryGuard(budget_bytes=100)  # impossibly small
    try:
        guard.check()
        assert False, "Expected MemoryBudgetExceeded"
    except MemoryBudgetExceeded:
        pass


def test_memory_guard_peak_and_violations():
    """Peak RSS and violation counters track correctly."""
    guard = MemoryGuard(budget_bytes=MEMORY_BUDGET_BYTES)
    guard.check()  # one valid check
    assert guard.peak_rss_mb > 0
    assert guard.violations == 0


def test_memory_guard_violation_handler():
    """Violation handler is called on budget breach."""
    called = []

    def handler(snap: MemorySnapshot):
        called.append(snap)

    guard = MemoryGuard(budget_bytes=100)
    guard.set_violation_handler(handler)
    try:
        guard.check()
    except MemoryBudgetExceeded:
        pass
    assert len(called) == 1


# ---------------------------------------------------------------------------
# ContextPager
# ---------------------------------------------------------------------------

def test_context_pager_append_and_window():
    """Turns are appended and the active window respects the token budget."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        pager = ContextPager(db_path=db_path, active_window_tokens=100)
        pager.append_turn("user", "Hello", 10)
        pager.append_turn("assistant", "Hi there", 5)
        pager.append_turn("user", "A" * 200, 200)  # exceeds window alone

        window = pager.get_active_window()
        # The 200-token turn alone exceeds the 100-token window, so it should
        # still be included (a single turn always fits).
        assert len(window) >= 1
        total_tokens = sum(t["token_count"] for t in window)
        assert total_tokens <= 100 + 200  # window + oversize turn
    finally:
        os.unlink(db_path)


def test_context_pager_evict_older():
    """Eviction marks turns outside keep set as evicted."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        pager = ContextPager(db_path=db_path, active_window_tokens=10000)
        t1 = pager.append_turn("user", "first", 10)
        t2 = pager.append_turn("user", "second", 10)
        t3 = pager.append_turn("user", "third", 10)

        evicted = pager.evict_older([t1, t3])
        assert evicted == 1  # t2 evicted

        window = pager.get_active_window()
        ids = {t["turn_id"] for t in window}
        assert t2 not in ids
        assert t1 in ids
        assert t3 in ids
    finally:
        os.unlink(db_path)


def test_context_pager_search():
    """Keyword search finds evicted turns."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        pager = ContextPager(db_path=db_path, active_window_tokens=10)
        pager.append_turn("user", "The quick brown fox jumps", 5)
        pager.append_turn("user", "Lazy dogs sleep all day", 5)
        pager.evict_older([])  # evict all

        results = pager.search("brown fox", limit=5)
        assert len(results) == 1
        assert "brown fox" in results[0]["content"]

        results = pager.search("dogs", limit=5)
        assert len(results) == 1
        assert "dogs" in results[0]["content"]
    finally:
        os.unlink(db_path)