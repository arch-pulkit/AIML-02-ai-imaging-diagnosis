"""
Stand-in for the Go API (Step 12): accepts POST /api/results and saves each payload.

For local testing only.

    python -m src.serve.stub_api
    STUB_FAIL_FIRST=1 python -m src.serve.stub_api   # reject the first POST (retry test)
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.data.nih import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data" / "interim" / "stub_api"
PORT = int(os.environ.get("STUB_PORT", "8090"))
HOST = os.environ.get("STUB_HOST", "127.0.0.1")  # 0.0.0.0 inside Docker
FAIL_FIRST = int(os.environ.get("STUB_FAIL_FIRST", "0"))


class Handler(BaseHTTPRequestHandler):
    posts = 0

    def do_POST(self) -> None:
        if self.path != "/api/results":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Handler.posts += 1
        if Handler.posts <= FAIL_FIRST:
            self.send_error(503, "stub: simulated failure")
            return
        payload = json.loads(body)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # keyed by instance id: a re-sent result overwrites, never duplicates
        (OUT_DIR / f"{payload['orthanc_instance_id']}.json").write_bytes(body)
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        reply = {"stored": payload["orthanc_instance_id"], "post": Handler.posts}
        self.wfile.write(json.dumps(reply).encode())


def main() -> None:
    print(
        f"stub API on http://localhost:{PORT}/api/results -> {OUT_DIR} (fail first {FAIL_FIRST})"
    )
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
