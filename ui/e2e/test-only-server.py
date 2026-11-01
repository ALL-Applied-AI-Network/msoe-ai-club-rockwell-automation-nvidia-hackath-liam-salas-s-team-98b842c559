"""Explicit TEST-ONLY Playwright server; never used by production commands."""
import os
import sys

if os.environ.get("NODE_ENV") == "production":
    raise SystemExit("Refusing to launch TEST-ONLY runner with NODE_ENV=production")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from test_support.test_only_runner_contract import start_test_only_server

server = start_test_only_server(host="127.0.0.1", port=5173)
print("TEST-ONLY runner contract simulator listening on http://127.0.0.1:5173", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    server.shutdown()
