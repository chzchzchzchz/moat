"""
Project Antigravity — End-to-End Orchestrator & Server Test Suite

Tests the complete AntigravityEngine pipeline and OpenAI-compatible HTTP server.

Test hierarchy:
  1. AntigravityEngine (end-to-end Best-of-N query execution, trace output, latency)
  2. OpenAI HTTP server API endpoints (/health, /v1/models, /v1/chat/completions)
"""

import sys
import os
import unittest
import numpy as np
import json
import time
import urllib.request
import urllib.error
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orchestrator import AntigravityEngine
from run_server import HTTPServer, OpenAIRequestHandler, engine as server_engine_global


class TestAntigravityEngine(unittest.TestCase):
    """End-to-end unit tests for AntigravityEngine orchestrator."""

    def setUp(self):
        self.engine = AntigravityEngine(n_channels=8, vocab_size=1000, hidden_dim=256)

    def test_run_best_of_n_query_structure(self):
        """Engine query must return valid output trace, score, and token metrics."""
        result = self.engine.run_best_of_n_query(
            prompt="Prove that 2^n > n^2 for n >= 5",
            max_tokens=30,
            temperature=0.7
        )

        self.assertIn('best_trace', result)
        self.assertIn('best_score', result)
        self.assertEqual(result['candidates_evaluated'], 8)
        self.assertGreater(result['best_score'], 0.0)
        self.assertLessEqual(result['best_score'], 1.0)
        self.assertGreater(result['latency_ms'], 0.0)

    def test_repeatable_queries_no_nan(self):
        """Running 5 queries sequentially must produce zero NaN or Inf."""
        for q in range(5):
            res = self.engine.run_best_of_n_query(f"Query {q}", max_tokens=20)
            self.assertFalse(np.isnan(res['best_score']))
            self.assertFalse(np.isinf(res['best_score']))


class TestOpenAIServerEndpoint(unittest.TestCase):
    """Integration tests for OpenAI-compatible REST server."""

    @classmethod
    def setUpClass(cls):
        """Start local HTTP server on port 8999 in background thread."""
        cls.port = 9123
        cls.server = HTTPServer(("127.0.0.1", cls.port), OpenAIRequestHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        time.sleep(0.5)  # Wait for server startup

    @classmethod
    def tearDownClass(cls):
        """Shutdown local HTTP server."""
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_endpoint(self):
        """GET /health must return status healthy."""
        url = f"http://127.0.0.1:{self.port}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data['status'], 'healthy')

    def test_models_endpoint(self):
        """GET /v1/models must return model list."""
        url = f"http://127.0.0.1:{self.port}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data['object'], 'list')
            self.assertGreater(len(data['data']), 0)

    def test_chat_completions_endpoint(self):
        """POST /v1/chat/completions must return valid OpenAI format JSON."""
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        payload = {
            "model": "antigravity-1.5b-tts",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 20,
            "temperature": 0.7
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))

            self.assertEqual(data['object'], 'chat.completion')
            self.assertIn('choices', data)
            self.assertEqual(len(data['choices']), 1)
            self.assertIn('message', data['choices'][0])
            self.assertIn('content', data['choices'][0]['message'])
            self.assertIn('antigravity_metadata', data)
            self.assertEqual(data['antigravity_metadata']['candidates_evaluated'], 8)


if __name__ == '__main__':
    print("=" * 70)
    print("Project Antigravity — Orchestrator & Server Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)
