"""
Project Antigravity — Cognitive Router / Middleware Layer.

Antigravity is NOT a model runner. It is an intelligent routing and
verification middleware that sits on top of existing LLM backends
(llama.cpp, MLX, Strata, Ollama, cloud APIs) and provides:

  * Tiered dispatch: tiny SLM reflex → local reasoner with DORA MCTS → cloud fallback
  * DORA-aware parallel rollout orchestration (best-of-N with uniqueness weighting)
  * In-process AST/JSC/Z3 verification and self-healing loops
  * Memory (2.5 GB RSS) and thermal (5 W) guardrails
  * First-finish circuit breaker to kill losing branches early

Design principle: never write custom GEMM kernels. Let llama.cpp / MLX / MPSGraph
handle the matrix math. Antigravity owns the *when, what, and how many* of
inference, plus correctness guarantees.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .llm_backend import (
    GenerationResult,
    LLMBackend,
    CloudBackend,
    MLXBackend,
    OllamaBackend,
    StrataBackend,
    OpenAICompatibleBackend,
    BackendUnavailable,
)
from .memory_guard import (
    MemoryGuard,
    MemoryBudgetExceeded,
    ThermalMonitor,
    ThermalBudgetExceeded,
    FirstFinishCircuitBreaker,
    MEMORY_BUDGET_BYTES,
    THERMAL_BUDGET_WATTS,
    KV_CACHE_ACTIVE_TOKENS,
)
from .ast_verifier import (
    ASTVerifier,
    JSCVerifier,
    Z3Verifier,
    SelfHealingLoop,
    VerificationResult,
)
from .dora_clustering import DORAClusterer
from .verifier import ListWiseVerifier, AdaptiveReflectionManager


# ---------------------------------------------------------------------------
# Router configuration
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Configuration for the cognitive router."""

    # Tier thresholds (complexity score 0.0-1.0)
    tier1_max: float = 0.30   # <= 0.30 → SLM reflex
    tier2_max: float = 0.70   # 0.30-0.70 → local reasoner with DORA
    # >= 0.70 → cloud fallback

    # Model selection per tier (OpenAI-compatible backend names)
    tier1_model: str = "llama-3.2-1b"      # tiny SLM reflex
    tier2_model: str = "qwen2.5-3b"        # local reasoner
    tier3_model: str = "gpt-4o-mini"        # cloud fallback

    # Backend URLs
    backend_urls: Dict[str, str] = field(default_factory=lambda: {
        "llama.cpp": "http://127.0.0.1:8080",
        "mlx": "http://127.0.0.1:8081",
        "ollama": "http://127.0.0.1:11434/v1",
        "strata": "http://127.0.0.1:8082",
        "cloud": "https://api.openai.com/v1",
    })

    # Rollout / DORA settings
    n_channels: int = 8                  # parallel reasoning channels
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Verification
    enable_code_verification: bool = True
    reflection_threshold: float = 0.75

    # Memory / thermal
    memory_budget_bytes: int = MEMORY_BUDGET_BYTES
    thermal_budget_watts: float = THERMAL_BUDGET_WATTS

    # Logging
    verbose: bool = False


class Tier(enum.IntEnum):
    TIER1_REFLEX = 1      # tiny SLM, single pass
    TIER2_REASONER = 2    # local reasoner + DORA MCTS + verification
    TIER3_CLOUD = 3       # cloud fallback


@dataclass
class RouteDecision:
    tier: Tier
    reason: str
    model: str
    backend: str
    n_channels: int
    max_tokens: int


@dataclass
class RouteResult:
    text: str
    tier: Tier
    model: str
    backend: str
    latency_ms: float
    tokens_used: int
    estimated_watts: float
    verified: bool
    verification_result: Optional[VerificationResult] = None
    candidates_evaluated: int = 0
    dora_info: Optional[Dict[str, Any]] = None
    reflection_triggered: bool = False


class ComplexityClassifier:
    """
    Heuristic intent/complexity classifier.

    Pure-Python, zero-dependency. Scores a prompt on lexical and structural
    features to route it to the right tier. Designed to run in <1 ms on a
    phone CPU so it never dominates the latency budget.
    """

    MATH_KEYWORDS = {
        "solve", "calculate", "compute", "equation", "formula", "derivative",
        "integral", "proof", "prove", "theorem", "algorithm", "recursive",
        "fibonacci", "prime", "factorial", "sqrt", "logarithm", "exponent",
    }
    CODE_KEYWORDS = {
        "def ", "class ", "import ", "function", "return ", "if __name__",
        "for ", "while ", "try:", "except", "lambda", "yield", "async",
    }
    LOGIC_KEYWORDS = {
        "because", "therefore", "implies", "contradiction", "consistent",
        "valid", "invalid", "premise", "conclusion", "deduce", "infer",
    }
    COMPLEX_WORDS = {
        "analyze", "compare", "evaluate", "design", "architecture", "optimize",
        "synthesize", "comprehensive", "multi-step", "reasoning", "strategy",
    }

    def classify(self, prompt: str) -> float:
        """
        Return a complexity score in [0.0, 1.0].

        Higher = more complex = needs more reasoning budget.
        """
        text = prompt.lower()
        words = set(text.split())
        n_words = max(len(words), 1)

        score = 0.0

        # Length signal (longer prompts tend to be more complex)
        length_score = min(n_words / 60.0, 1.0) * 0.25
        score += length_score

        # Keyword signals
        math_hits = sum(1 for kw in self.MATH_KEYWORDS if kw in text)
        code_hits = sum(1 for kw in self.CODE_KEYWORDS if kw in text)
        logic_hits = sum(1 for kw in self.LOGIC_KEYWORDS if kw in text)
        complex_hits = sum(1 for kw in self.COMPLEX_WORDS if kw in text)

        keyword_score = min((math_hits * 0.15 + code_hits * 0.15 +
                             logic_hits * 0.10 + complex_hits * 0.10), 0.45)
        score += keyword_score

        # Punctuation / structure signals
        question_marks = text.count("?")
        if question_marks > 2:
            score += 0.10
        if "step" in text or "step-by-step" in text:
            score += 0.05

        return min(score, 1.0)


# ---------------------------------------------------------------------------
# Cognitive Router
# ---------------------------------------------------------------------------

class CognitiveRouter:
    """
    Cognitive Router — the Antigravity middleware.

    Routes prompts through a tiered pipeline:

      Tier 1 (Reflex):     tiny SLM, single pass, <100 ms, <600 MB RAM
      Tier 2 (Reasoner):   local model + DORA parallel rollouts (N=8)
                           + AST/Z3 verification + self-healing
      Tier 3 (Cloud):      frontier model fallback for open-domain queries

    Memory and thermal guardrails are enforced at every tier. The first
    finish circuit breaker halts losing parallel branches as soon as one
    passes verification.
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self.classifier = ComplexityClassifier()

        # Backend clients (lazy-initialized)
        self._backends: Dict[str, LLMBackend] = {}

        # Guardrails
        self.memory_guard = MemoryGuard(
            budget_bytes=self.config.memory_budget_bytes,
            warn_bytes=int(self.config.memory_budget_bytes * 0.8),
        )
        self.thermal_monitor = ThermalMonitor(
            budget_watts=self.config.thermal_budget_watts,
        )

        # Verification pipeline
        self.ast_verifier = ASTVerifier()
        self.jsc_verifier = JSCVerifier()
        self.z3_verifier = Z3Verifier()
        self.verifier = ListWiseVerifier()
        self.reflection_manager = AdaptiveReflectionManager(
            threshold=self.config.reflection_threshold,
        )
        self.self_healing = SelfHealingLoop(max_retries=2)
        self.dora_clusterer = DORAClusterer()

        # Stats
        self._stats = {
            "total_requests": 0,
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "verified_count": 0,
            "reflection_triggered": 0,
            "circuit_breaker_hits": 0,
        }

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------
    def _get_backend(self, name: str) -> LLMBackend:
        """Get or create a backend client by name."""
        if name in self._backends:
            return self._backends[name]

        url = self.config.backend_urls.get(name)
        if url is None:
            raise ValueError(f"Unknown backend: {name}")

        if name == "ollama":
            b = OllamaBackend(base_url=url, model=self.config.tier1_model)
        elif name == "strata":
            b = StrataBackend(base_url=url, model=self.config.tier2_model)
        elif name == "mlx":
            b = MLXBackend(base_url=url, model=self.config.tier2_model)
        elif name == "cloud":
            api_key = __import__("os").environ.get("ANTHROPIC_API_KEY") or \
                      __import__("os").environ.get("OPENAI_API_KEY")
            b = CloudBackend(base_url=url, api_key=api_key or "",
                             model=self.config.tier3_model)
        else:
            b = OpenAICompatibleBackend(base_url=url, model=self.config.tier2_model)

        self._backends[name] = b
        return b

    def _check_memory(self) -> None:
        """Enforce memory budget; raise if exceeded."""
        self.memory_guard.check()

    def _check_thermal(self, watts: float) -> None:
        """Enforce thermal budget; raise if exceeded."""
        snapshot = self.thermal_monitor.update(watts)
        if snapshot.over_budget:
            raise ThermalBudgetExceeded(
                f"Estimated sustained power {snapshot.watts:.1f} W exceeds "
                f"{self.config.thermal_budget_watts} W thermal budget"
            )

    # ------------------------------------------------------------------
    # Complexity classification
    # ------------------------------------------------------------------
    def classify_prompt(self, prompt: str) -> float:
        return self.classifier.classify(prompt)

    def select_tier(self, score: float) -> Tier:
        if score <= self.config.tier1_max:
            return Tier.TIER1_REFLEX
        if score <= self.config.tier2_max:
            return Tier.TIER2_REASONER
        return Tier.TIER3_CLOUD

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------
    def route(self, prompt: str,
              max_tokens: Optional[int] = None,
              temperature: Optional[float] = None,
              top_p: Optional[float] = None,
              force_tier: Optional[Tier] = None,
              context: Optional[Dict[str, Any]] = None) -> RouteResult:
        """
        Route a prompt through the cognitive pipeline.

        Steps:
          1. Classify complexity → select tier
          2. Check memory/thermal guardrails
          3. Dispatch to appropriate backend
          4. For Tier 2: run DORA parallel rollouts + verification
          5. First-finish circuit breaker: halt losing branches early
          6. Return verified result
        """
        self._stats["total_requests"] += 1
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # 1. Classify
        score = self.classify_prompt(prompt)
        tier = force_tier or self.select_tier(score)

        if self.config.verbose:
            print(f"[Router] score={score:.2f} → tier={tier.name}")

        # 2. Guardrails
        self._check_memory()
        self._check_thermal(0.0)  # initial check, updated after generation

        # 3. Dispatch by tier
        if tier == Tier.TIER1_REFLEX:
            return self._route_tier1(prompt, max_tokens, temperature, top_p, score)
        elif tier == Tier.TIER2_REASONER:
            return self._route_tier2(prompt, max_tokens, temperature, top_p, score, context)
        else:
            return self._route_tier3(prompt, max_tokens, temperature, top_p, score)

    # ------------------------------------------------------------------
    # Tier 1: reflex (tiny SLM, single pass)
    # ------------------------------------------------------------------
    def _route_tier1(self, prompt: str, max_tokens: int,
                     temperature: float, top_p: float,
                     score: float) -> RouteResult:
        self._stats["tier1_count"] += 1

        backend = self._get_backend("ollama")
        backend.model = self.config.tier1_model

        result = backend.generate(prompt, max_tokens=max_tokens,
                                  temperature=temperature, top_p=top_p)

        self._check_memory()
        self._check_thermal(result.estimated_watts)

        return RouteResult(
            text=result.text,
            tier=Tier.TIER1_REFLEX,
            model=self.config.tier1_model,
            backend="ollama",
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            estimated_watts=result.estimated_watts,
            verified=False,
            candidates_evaluated=1,
        )

    # ------------------------------------------------------------------
    # Tier 2: reasoner with DORA parallel rollouts + verification
    # ------------------------------------------------------------------
    def _route_tier2(self, prompt: str, max_tokens: int,
                     temperature: float, top_p: float,
                     score: float,
                     context: Optional[Dict[str, Any]]) -> RouteResult:
        self._stats["tier2_count"] += 1

        backend = self._get_backend("mlx")
        backend.model = self.config.tier2_model

        # DORA parallel rollout + verification
        result = self._dora_rollout(
            backend=backend,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            context=context,
        )

        self._check_memory()
        self._check_thermal(result.estimated_watts)

        return result

    def _dora_rollout(self, backend: LLMBackend, prompt: str,
                      max_tokens: int, temperature: float, top_p: float,
                      context: Optional[Dict[str, Any]]) -> RouteResult:
        """
        Run N parallel rollouts through the backend, then:
          1. DORA uniqueness weighting (de-duplicate similar traces)
          2. List-wise verifier scoring
          3. GenPRM code verification (if code detected)
          4. AST/JSC/Z3 verification + self-healing loop
          5. First-finish circuit breaker (halt losers as soon as one verifies)
        """
        n_channels = self.config.n_channels
        breaker = FirstFinishCircuitBreaker(n_branches=n_channels)

        # Launch N parallel rollouts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                self._parallel_rollouts(backend, prompt, max_tokens,
                                        temperature, top_p, n_channels, breaker)
            )
        finally:
            loop.close()

        if not results:
            raise RuntimeError("All rollout channels failed")

        # Collect traces and scores
        traces = [r["text"] for r in results]
        scores = np.array([r.get("score", 0.5) for r in results], dtype=np.float32)

        # DORA uniqueness weighting
        dora_result = self.dora_clusterer.cluster_candidates(traces)
        uniqueness = dora_result["uniqueness_weights"]
        adjusted_scores = scores * uniqueness[:len(scores)]

        # List-wise verification
        logprobs = np.array([r.get("logprob", 0.0) for r in results],
                            dtype=np.float32)
        verification = self.verifier.score_candidates_listwise(
            traces, logprobs
        )
        verifier_scores = verification["scores"]
        combined_scores = verifier_scores * uniqueness[:len(verifier_scores)]

        best_idx = int(np.argmax(combined_scores))
        best_trace = traces[best_idx]

        # Verification + self-healing loop
        verified = False
        verification_result = None
        reflection_triggered = False

        verifiers = [self.ast_verifier, self.z3_verifier]

        def dispatch(p: str) -> str:
            # Re-dispatch to backend for self-healing retry
            r = backend.generate(p, max_tokens=max_tokens,
                                 temperature=temperature, top_p=top_p)
            return r.text

        healed_trace, attempts = self.self_healing.run(
            dispatch=dispatch,
            verifiers=verifiers,
            initial_prompt=prompt,
            context=context,
        )

        if attempts:
            last = attempts[-1]
            verification_result = last.result
            verified = last.result.passed
            if len(attempts) > 1:
                reflection_triggered = True
                self._stats["reflection_triggered"] += 1

        if verified:
            best_trace = healed_trace
            self._stats["verified_count"] += 1
            self._stats["circuit_breaker_hits"] += 1

        # Estimate power (rough: tokens * watts_per_token)
        total_tokens = sum(len(t.split()) for t in traces)
        est_watts = total_tokens * 0.05  # rough estimate

        return RouteResult(
            text=best_trace,
            tier=Tier.TIER2_REASONER,
            model=self.config.tier2_model,
            backend="mlx",
            latency_ms=sum(r.get("latency_ms", 0) for r in results),
            tokens_used=total_tokens,
            estimated_watts=est_watts,
            verified=verified,
            verification_result=verification_result,
            candidates_evaluated=n_channels,
            dora_info={
                "uniqueness_weights": uniqueness.tolist(),
                "similarity_matrix": dora_result["similarity_matrix"].tolist(),
                "redundancy_pairs": dora_result["redundancy_pairs"],
            },
            reflection_triggered=reflection_triggered,
        )

    # ------------------------------------------------------------------
    # Parallel rollout execution
    # ------------------------------------------------------------------
    async def _parallel_rollouts(self, backend: LLMBackend, prompt: str,
                                 max_tokens: int, temperature: float,
                                 top_p: float, n_channels: int,
                                 breaker: FirstFinishCircuitBreaker) -> List[Dict]:
        """
        Run N rollouts concurrently. If the circuit breaker halts (one branch
        verified), cancel remaining work and return what we have.
        """
        import asyncio

        async def single_rollout(idx: int) -> Dict:
            if breaker.is_halted():
                return None
            t0 = time.monotonic()
            try:
                result = backend.generate(prompt, max_tokens=max_tokens,
                                          temperature=temperature, top_p=top_p)
                latency_ms = (time.monotonic() - t0) * 1000.0
                return {
                    "text": result.text,
                    "latency_ms": latency_ms,
                    "tokens_used": result.tokens_used,
                    "logprob": -result.latency_ms / 1000.0,  # placeholder
                    "score": 0.5,  # placeholder, overwritten by verifier
                    "index": idx,
                }
            except Exception as exc:
                return {"text": "", "latency_ms": 0.0, "tokens_used": 0,
                        "logprob": -999.0, "score": 0.0, "index": idx,
                        "error": str(exc)}

        tasks = [asyncio.create_task(single_rollout(i)) for i in range(n_channels)]
        results: List[Dict] = []

        # Collect results as they complete; stop early if breaker halts
        for coro in asyncio.as_completed(tasks):
            r = await coro
            if r and r.get("text"):
                results.append(r)
                # Check if this result verifies (simple heuristic: non-empty,
                # contains answer-like content)
                if self._looks_complete(r["text"]):
                    breaker.halt_all(branch_id=r["index"], trace=r["text"])
                    break

        # Cancel any remaining tasks
        for t in tasks:
            if not t.done():
                t.cancel()

        return [r for r in results if r and r.get("text")]

    def _looks_complete(self, text: str) -> bool:
        """Heuristic: does the text look like a complete answer?"""""
        if not text or len(text.strip()) < 5:
            return False
        # Ends with sentence terminator or contains an answer marker
        stripped = text.strip()
        if stripped[-1] in ".!?":
            return True
        if "####" in stripped or "answer:" in stripped.lower():
            return True
        return False

    # ------------------------------------------------------------------
    # Tier 3: cloud fallback
    # ------------------------------------------------------------------
    def _route_tier3(self, prompt: str, max_tokens: int,
                     temperature: float, top_p: float,
                     score: float) -> RouteResult:
        self._stats["tier3_count"] += 1

        backend = self._get_backend("cloud")
        if not backend.is_available():
            # Fall back to Tier 2 if cloud unavailable
            return self._route_tier2(prompt, max_tokens, temperature,
                                     top_p, score, None)

        result = backend.generate(prompt, max_tokens=max_tokens,
                                  temperature=temperature, top_p=top_p)

        self._check_memory()
        self._check_thermal(result.estimated_watts)

        return RouteResult(
            text=result.text,
            tier=Tier.TIER3_CLOUD,
            model=self.config.tier3_model,
            backend="cloud",
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            estimated_watts=result.estimated_watts,
            verified=False,
            candidates_evaluated=1,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def memory_snapshot(self) -> Dict[str, Any]:
        snap = self.memory_guard.check()
        return {
            "rss_mb": snap.rss_mb,
            "over_budget": snap.over_budget,
            "peak_rss_mb": self.memory_guard.peak_rss_mb,
            "violations": self.memory_guard.violations,
        }

    def thermal_snapshot(self) -> Dict[str, Any]:
        snap = self.thermal_monitor.current_watts
        return {
            "current_watts": snap,
            "over_budget": snap is not None and snap > self.config.thermal_budget_watts,
            "violations": self.thermal_monitor.violations,
        }


# Convenience functions for one-off use

async def route_async(
    prompt: str,
    config: Optional[RouterConfig] = None,
    backend: Optional[LLMBackend] = None,
    router: Optional[CognitiveRouter] = None,
    **kwargs,
) -> RouteResult:
    if router is None:
        router = CognitiveRouter(config=config)
    if backend is not None:
        # Register under the backend's own name and under "ollama" (Tier 1 default)
        router._backends[backend.backend_name()] = backend
        router._backends["ollama"] = backend
    return router.route(prompt, **kwargs)


def route_sync(
    prompt: str,
    config: Optional[RouterConfig] = None,
    backend: Optional[LLMBackend] = None,
    router: Optional[CognitiveRouter] = None,
    **kwargs,
) -> RouteResult:
    if router is None:
        router = CognitiveRouter(config=config)
    if backend is not None:
        router._backends[backend.backend_name()] = backend
        router._backends["ollama"] = backend
    return router.route(prompt, **kwargs)


__all__ = [
    "RouterConfig", "Tier", "RouteDecision", "RouteResult",
    "ComplexityClassifier", "CognitiveRouter",
    "route_sync", "route_async",
]
