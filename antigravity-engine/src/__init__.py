"""
Project Antigravity — Cognitive Router & Middleware Package.

Antigravity is NOT a model runner. It is an intelligent routing and
verification middleware that sits on top of existing LLM backends
(llama.cpp, MLX, Strata, Ollama, cloud APIs) and provides:

  * Cognitive routing: intent classification → Tier 1 reflex / Tier 2
    reasoner with DORA MCTS / Tier 3 cloud fallback
  * DORA-aware parallel rollout orchestration (best-of-N with uniqueness
    weighting)
  * In-process AST/JSC/Z3 verification and self-healing loops
  * Memory (2.5 GB RSS) and thermal (5 W) guardrails
  * First-finish circuit breaker to kill losing branches early

Submodules:
  llm_backend      — Backend abstraction (llama.cpp, MLX, Strata, Ollama,
                     cloud) over OpenAI-compatible HTTP
  memory_guard     — RSS ceiling, context paging, thermal monitor,
                     first-finish circuit breaker
  ast_verifier     — AST/JSC/Z3 verification + self-healing loop
  dora_clustering  — Lexical diversity clustering + uniqueness weights
  verifier         — List-wise verifier, adaptive reflection, model swapper
  verification_strategy — BestOfN / REBASE selection strategies
  quality_scoring  — GSM8K answer extraction, Wilson intervals, McNemar
  cognitive_router — The Cognitive Router (main entry point)
  dora_orchestrator — Parallel rollout orchestration with DORA re-weighting
"""

from .llm_backend import (
    LLMBackend,
    GenerationResult,
    OpenAICompatibleBackend,
    OllamaBackend,
    StrataBackend,
    MLXBackend,
    CloudBackend,
    BackendUnavailable,
)

from .memory_guard import (
    MemoryGuard,
    MemoryBudgetExceeded,
    MemorySnapshot,
    ContextPager,
    ThermalMonitor,
    ThermalBudgetExceeded,
    ThermalSnapshot,
    FirstFinishCircuitBreaker,
    MEMORY_BUDGET_BYTES,
    THERMAL_BUDGET_WATTS,
    KV_CACHE_ACTIVE_TOKENS,
)

from .ast_verifier import (
    VerificationResult,
    HealingAttempt,
    ASTVerifier,
    JSCVerifier,
    Z3Verifier,
    SelfHealingLoop,
)

from .dora_clustering import DORAClusterer

from .verifier import (
    ListWiseVerifier,
    AdaptiveReflectionManager,
    SequentialModelSwapper,
    DEFAULT_REFLECTION_THRESHOLD,
)

from .verification_strategy import (
    BestOfNStrategy,
    RebaseStrategy,
)

from .cognitive_router import (
    RouterConfig,
    Tier,
    RouteDecision,
    RouteResult,
    ComplexityClassifier,
    CognitiveRouter,
    route_sync,
    route_async,
)

from .dora_orchestrator import (
    DORAOrchestratorConfig,
    RolloutResult,
    OrchestrationResult,
    DORAOrchestrator,
    orchestrate_sync,
    orchestrate_async,
)

__all__ = [
    # llm_backend
    "LLMBackend",
    "GenerationResult",
    "OpenAICompatibleBackend",
    "OllamaBackend",
    "StrataBackend",
    "MLXBackend",
    "CloudBackend",
    "BackendUnavailable",
    # memory_guard
    "MemoryGuard",
    "MemoryBudgetExceeded",
    "MemorySnapshot",
    "ContextPager",
    "ThermalMonitor",
    "ThermalBudgetExceeded",
    "ThermalSnapshot",
    "FirstFinishCircuitBreaker",
    "MEMORY_BUDGET_BYTES",
    "THERMAL_BUDGET_WATTS",
    "KV_CACHE_ACTIVE_TOKENS",
    # ast_verifier
    "VerificationResult",
    "HealingAttempt",
    "ASTVerifier",
    "JSCVerifier",
    "Z3Verifier",
    "SelfHealingLoop",
    # dora_clustering
    "DORAClusterer",
    # verifier
    "ListWiseVerifier",
    "AdaptiveReflectionManager",
    "SequentialModelSwapper",
    "DEFAULT_REFLECTION_THRESHOLD",
    # verification_strategy
    "BestOfNStrategy",
    "RebaseStrategy",
    # cognitive_router
    "RouterConfig",
    "Tier",
    "RouteDecision",
    "RouteResult",
    "ComplexityClassifier",
    "CognitiveRouter",
    "route_sync",
    "route_async",
    # dora_orchestrator
    "DORAOrchestratorConfig",
    "RolloutResult",
    "OrchestrationResult",
    "DORAOrchestrator",
    "orchestrate_sync",
    "orchestrate_async",
]
