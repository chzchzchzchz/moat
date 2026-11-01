"""
Project Antigravity — DORA Orchestrator.

Coordinates parallel reasoning-channel rollouts (Best-of-N) across multiple
LLM backends, applies DORA uniqueness re-weighting to de-duplicate similar
candidates, and selects the best trace via list-wise verification.

This is the middleware layer: it does NOT run model inference itself. It
orchestrates calls to existing backends (llama.cpp, MLX, Strata, Ollama,
cloud APIs) and applies Antigravity's test-time-scaling logic on top.

Architecture:
    N parallel channels  →  DORA uniqueness re-weighting  →  List-wise
    verifier scoring  →  AST/Z3 verification  →  best trace

Designed for phone-level use: bounded memory, sub-second micro-bursts,
first-finish circuit breaker.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .llm_backend import GenerationResult, LLMBackend
from .memory_guard import FirstFinishCircuitBreaker
from .dora_clustering import DORAClusterer
from .verifier import ListWiseVerifier, AdaptiveReflectionManager
from .ast_verifier import ASTVerifier, Z3Verifier, SelfHealingLoop, VerificationResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DORAOrchestratorConfig:
    """Configuration for DORA orchestration."""

    n_channels: int = 8              # parallel reasoning channels
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Verification
    enable_verification: bool = True
    enable_self_healing: bool = True
    max_healing_retries: int = 2
    reflection_threshold: float = 0.75

    # Memory / thermal
    memory_budget_bytes: int = 2_500 * 1024 * 1024   # 2.5 GB
    thermal_budget_watts: float = 5.0

    # DORA parameters
    dora_tau: float = 0.5              # temperature for uniqueness softmax
    redundancy_threshold: float = 0.85  # similarity above which = redundant


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """Result from a single reasoning channel."""
    index: int
    text: str
    tokens: List[int]
    logprob: float
    latency_ms: float
    backend: str
    model: str
    error: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Final orchestrated result."""
    best_trace: str
    best_index: int
    best_score: float
    all_traces: List[str]
    all_scores: List[float]
    latency_ms: float
    tokens_used: int
    estimated_watts: float
    verified: bool
    verification_passed: bool
    candidates_evaluated: int
    reflection_triggered: bool
    dora_info: Dict[str, Any]
    healing_attempts: int


# ---------------------------------------------------------------------------
# DORA Orchestrator
# ---------------------------------------------------------------------------

class DORAOrchestrator:
    """
    Orchestrates N parallel reasoning-channel rollouts across LLM backends,
    applies DORA uniqueness re-weighting, and selects the best trace via
    list-wise verification with optional self-healing.

    Usage:
        orch = DORAOrchestrator(backend=mlx_backend, n_channels=8)
        result = await orch.orchestrate("What is 17 * 23?")
        print(result.best_trace)
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: Optional[DORAOrchestratorConfig] = None,
    ):
        self.backend = backend
        self.config = config or DORAOrchestratorConfig()

        # Verification components
        self.verifier = ListWiseVerifier()
        self.ast_verifier = ASTVerifier()
        self.z3_verifier = Z3Verifier()
        self.self_healing = SelfHealingLoop(max_retries=self.config.max_healing_retries)
        self.dora_clusterer = DORAClusterer()

        # Stats
        self._stats = {
            "total_requests": 0,
            "verified_count": 0,
            "reflection_triggered": 0,
            "circuit_breaker_hits": 0,
            "avg_latency_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def orchestrate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Run N parallel rollouts, apply DORA + verification, return best trace.

        Steps:
          1. Launch N parallel rollouts through the backend
          2. First-finish circuit breaker: halt losers once one verifies
          3. DORA uniqueness re-weighting (de-duplicate similar traces)
          4. List-wise verifier scoring
          5. AST/Z3 verification + self-healing loop on failure
          6. Return best verified trace
        """
        self._stats["total_requests"] += 1
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        t0 = time.monotonic()

        # Step 1: Parallel rollouts with circuit breaker
        breaker = FirstFinishCircuitBreaker(n_branches=self.config.n_channels)
        rollouts = await self._parallel_rollouts(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_channels=self.config.n_channels,
            breaker=breaker,
        )

        if not rollouts:
            raise RuntimeError("All DORA rollout channels failed")

        # Step 2: Collect traces and scores
        traces = [r["text"] for r in rollouts]
        logprobs = np.array([r.get("logprob", 0.0) for r in rollouts],
                            dtype=np.float32)

        # Step 3: DORA uniqueness re-weighting
        dora_result = self.dora_clusterer.cluster_candidates(traces)
        uniqueness = dora_result["uniqueness_weights"]
        n = len(traces)
        uniqueness = uniqueness[:n] if len(uniqueness) >= n else np.pad(
            uniqueness, (0, n - len(uniqueness)), constant_values=1.0
        )

        # Step 4: List-wise verifier scoring
        verification = self.verifier.score_candidates_listwise(traces, logprobs)
        verifier_scores = verification["scores"]  # softmax-normalized [0,1]

        # Combined: verifier score * uniqueness
        combined = verifier_scores * uniqueness
        best_idx = int(np.argmax(combined))
        best_trace = traces[best_idx]

        # Step 5: Verification + self-healing
        verified = False
        verification_result = None
        reflection_triggered = False
        healing_attempts = 0

        if self.config.enable_verification:
            # Build verification context from the best trace
            stated_answer = self._extract_stated_answer(best_trace)
            expression = self._extract_math_expression(best_trace)

            def dispatch(p: str) -> str:
                """Re-dispatch to backend for self-healing retry."""
                result = self.backend.generate(
                    p, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p
                )
                return result.text

            healed_trace, attempts = self.self_healing.run(
                dispatch=dispatch,
                verifiers=[self.ast_verifier, self.z3_verifier],
                initial_prompt=prompt,
                context={"expression": expression, "stated_answer": stated_answer},
            )

            healing_attempts = len(attempts)
            if attempts:
                last = attempts[-1]
                verification_result = last.result
                verified = last.result.passed
                if len(attempts) > 1:
                    reflection_triggered = True

            if verified:
                best_trace = healed_trace
                self._stats["verified_count"] += 1
                self._stats["circuit_breaker_hits"] += 1

            if reflection_triggered:
                self._stats["reflection_triggered"] += 1

        # Step 6: Compute metrics
        total_tokens = sum(len(t.split()) for t in traces)
        latency_ms = (time.monotonic() - t0) * 1000.0
        est_watts = total_tokens * 0.05  # rough estimate

        # Update rolling average latency
        self._stats["avg_latency_ms"] = (
            0.9 * self._stats["avg_latency_ms"] + 0.1 * latency_ms
        )

        return OrchestrationResult(
            best_trace=best_trace,
            best_index=best_idx,
            best_score=float(combined[best_idx]),
            all_traces=traces,
            all_scores=combined.tolist(),
            latency_ms=latency_ms,
            tokens_used=total_tokens,
            estimated_watts=est_watts,
            verified=verified,
            verification_passed=verified,
            candidates_evaluated=len(traces),
            reflection_triggered=reflection_triggered,
            dora_info={
                "uniqueness_weights": uniqueness.tolist(),
                "similarity_matrix": dora_result["similarity_matrix"].tolist(),
                "redundancy_pairs": dora_result["redundancy_pairs"],
                "verifier_scores": verifier_scores.tolist(),
                "combined_scores": combined.tolist(),
            },
            healing_attempts=healing_attempts,
        )

    # ------------------------------------------------------------------
    # Parallel rollout execution
    # ------------------------------------------------------------------
    async def _parallel_rollouts(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        n_channels: int,
        breaker: FirstFinishCircuitBreaker,
    ) -> List[Dict[str, Any]]:
        """
        Launch N concurrent rollout tasks. Results arrive as they complete;
        if the circuit breaker halts (one branch verified), stop early.
        """

        async def single(idx: int) -> Dict[str, Any]:
            if breaker.is_halted():
                return None
            t0 = time.monotonic()
            try:
                result = self.backend.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                return {
                    "index": idx,
                    "text": result.text,
                    "tokens_used": result.tokens_used,
                    "latency_ms": latency_ms,
                    "backend": result.backend,
                    "model": result.model,
                    # Placeholder logprob; overwritten by verifier later
                    "logprob": -result.latency_ms / 1000.0,
                    "score": 0.5,
                }
            except Exception as exc:
                return {
                    "index": idx,
                    "text": "",
                    "tokens_used": 0,
                    "latency_ms": 0.0,
                    "backend": getattr(self.backend, "backend_name", lambda: "unknown")(),
                    "model": getattr(self.backend, "model", "unknown"),
                    "logprob": -999.0,
                    "score": 0.0,
                    "error": str(exc),
                }

        tasks = [asyncio.create_task(single(i)) for i in range(n_channels)]
        results: List[Dict[str, Any]] = []

        for coro in asyncio.as_completed(tasks):
            r = await coro
            if r and r.get("text"):
                results.append(r)
                # Early termination: if this looks like a complete answer,
                # halt the circuit breaker so remaining branches stop.
                if self._looks_complete(r["text"]):
                    breaker.halt_all(branch_id=r["index"], trace=r["text"])
                    break

        # Cancel stragglers
        for t in tasks:
            if not t.done():
                t.cancel()

        return [r for r in results if r and r.get("text")]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _looks_complete(self, text: str) -> bool:
        """Heuristic: does the text look like a complete answer?"""
        if not text or len(text.strip()) < 5:
            return False
        stripped = text.strip()
        if stripped[-1] in ".!?":
            return True
        if "####" in stripped or "answer:" in stripped.lower():
            return True
        return False

    def _extract_stated_answer(self, trace: str) -> Optional[str]:
        """Extract GSM8K-style or boxed answer from trace."""
        import re
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", trace)
        if m:
            return m.group(1)
        m = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", trace)
        if m:
            return m.group(1)
        return None

    def _extract_math_expression(self, trace: str) -> Optional[str]:
        """Extract a math expression from trace for Z3 verification."""
        import re
        # Look for simple arithmetic expressions
        m = re.search(r"([\d\s\+\-\*\/\(\)\.]+)", trace)
        if m:
            return m.group(1).strip()
        return None

    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def orchestrate_async(
    backend: LLMBackend,
    prompt: str,
    config: Optional[DORAOrchestratorConfig] = None,
    **kwargs,
) -> OrchestrationResult:
    """Run DORA orchestration asynchronously."""
    orch = DORAOrchestrator(backend=backend, config=config)
    return await orch.orchestrate(prompt, **kwargs)


def orchestrate_sync(
    backend: LLMBackend,
    prompt: str,
    config: Optional[DORAOrchestratorConfig] = None,
    **kwargs,
) -> OrchestrationResult:
    """Run DORA orchestration synchronously."""
    orch = DORAOrchestrator(backend=backend, config=config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(orch.orchestrate(prompt, **kwargs))
    finally:
        loop.close()


__all__ = [
    "DORAOrchestratorConfig",
    "RolloutResult",
    "OrchestrationResult",
    "DORAOrchestrator",
    "orchestrate_sync",
    "orchestrate_async",
]
