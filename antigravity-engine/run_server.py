"""
Project Antigravity — OpenAI-Compatible Local Server Endpoint

Exposes standard OpenAI /v1/chat/completions API format fully offline.
Powered by the Antigravity parallel Best-of-N engine.

Usage:
    python3 antigravity-engine/run_server.py --port 8080
"""

import sys
import os
import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from orchestrator import AntigravityEngine


# Global engine instance
engine = None


class OpenAIRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler implementing OpenAI REST API specification."""

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/health" or self.path == "/v1/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "healthy", "engine": "Project Antigravity 1.5B"}).encode('utf-8'))
        elif self.path == "/v1/models":
            self._set_headers(200)
            models_response = {
                "object": "list",
                "data": [
                    {
                        "id": "antigravity-1.5b-tts",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "antigravity"
                    }
                ]
            }
            self.wfile.write(json.dumps(models_response).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            try:
                request_json = json.loads(body_bytes.decode('utf-8'))
            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {str(e)}"}).encode('utf-8'))
                return

            messages = request_json.get("messages", [])
            if not messages:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Field 'messages' is required"}).encode('utf-8'))
                return

            # Extract user prompt from last message
            user_prompt = messages[-1].get("content", "")
            temperature = float(request_json.get("temperature", 0.7))
            max_tokens  = int(request_json.get("max_tokens", 50))

            # Run parallel Best-of-N query via engine orchestrator
            global engine
            if engine is None:
                engine = AntigravityEngine(n_channels=8)

            result = engine.run_best_of_n_query(
                prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Format response into standard OpenAI chat completion JSON
            response_json = {
                "id": f"chatcmpl-antigravity-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_json.get("model", "antigravity-1.5b-tts"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result['best_trace']
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_prompt.split()),
                    "completion_tokens": result['tokens_generated_total'],
                    "total_tokens": len(user_prompt.split()) + result['tokens_generated_total']
                },
                "antigravity_metadata": {
                    "candidates_evaluated": result['candidates_evaluated'],
                    "best_score": result['best_score'],
                    "reflection_triggered": result['reflection_triggered'],
                    "token_savings_pct": result['token_savings_pct'],
                    "latency_ms": result['latency_ms']
                }
            }

            self._set_headers(200)
            self.wfile.write(json.dumps(response_json, indent=2).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description="Project Antigravity Local API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    args = parser.parse_args()

    global engine
    print("=" * 70)
    print("Project Antigravity — OpenAI-Compatible Local Server")
    print(f"Initializing engine orchestrator (N=8 parallel channels)...")
    engine = AntigravityEngine(n_channels=8)
    print(f"Server listening on http://{args.host}:{args.port}")
    print(f"Endpoint: POST http://{args.host}:{args.port}/v1/chat/completions")
    print("=" * 70)

    server = HTTPServer((args.host, args.port), OpenAIRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == '__main__':
    main()
