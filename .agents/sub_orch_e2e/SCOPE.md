# Scope: E2E Testing Track — Project Antigravity

## Architecture & Test Philosophy
- Opaque-box, requirement-driven test suite derived strictly from `antigravity_prd.md` and `ORIGINAL_REQUEST.md`.
- Derived without depending on implementation details or internal private methods.
- Comprehensive 4-Tier Breakdown covering:
  - Tier 1: Feature Coverage (>=5 tests per feature across 7 core features = 35+ tests)
  - Tier 2: Boundary & Corner Cases (>=5 tests per feature across 7 core features = 35+ tests)
  - Tier 3: Cross-Feature Pairwise Interactions (8+ tests)
  - Tier 4: Real-World Application & Workload Scenarios (5+ tests)
- Minimum total test count: 83+ test cases (targeting 85+).

## Core Features Inventory (N=7)
1. F1: INT4 Group Quantization & Superblock Weight Repacking (`dequant`)
2. F2: Optimized Softmax LUT & Attention Computation (`attention`)
3. F3: Batched Parallel Decode Engine GEMV->GEMM (`batch_generator`)
4. F4: List-Wise Verifier & Best-of-N Candidate Selection (`verifier`)
5. F5: Threshold-Driven Adaptive Reflection Engine (`reflection`)
6. F6: Local OpenAI-Compatible API Endpoint (`run_server`)
7. F7: Performance, Accuracy & Token Efficiency Benchmark Harness (`benchmark`)

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Test Exploration & Matrix Design | Analyze specs, formulate test case inventory across Tiers 1-4 | none | DONE |
| 2 | Infrastructure & Test Implementation | Create TEST_INFRA.md, write tests/e2e/ test files, create test runner script | M1 | IN_PROGRESS |
| 3 | Review & Stress Challenge | Review test code, run test runner, verify tier matrix execution | M2 | PLANNED |
| 4 | Forensic Audit | Verify integrity, zero hardcoded values, zero fake tests | M3 | PLANNED |
| 5 | TEST_READY Publication & Handoff | Write TEST_READY.md and handoff.md, notify parent | M4 | PLANNED |

## Interface Contracts & Test Runner Specification
- Test runner location: `/Users/MohssineChazi2/moat/tests/e2e/run_e2e_tests.py` (or `pytest` runner entrypoint `/Users/MohssineChazi2/moat/tests/run_e2e_tests.py`)
- Invocation command: `python3 tests/e2e/run_e2e_tests.py`
- Pass/Fail criterion: Exit code 0 on 100% pass, non-zero on failure.
- Output format: Clear console test summary with breakdown per tier and feature.
