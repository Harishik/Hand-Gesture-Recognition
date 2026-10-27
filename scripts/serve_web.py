"""Serve the browser app locally with caching disabled (so edits to JS modules show up on reload).

python scripts/serve_web.py [--port 8765]   then open http://localhost:8765

Camera access needs a secure context, which browsers grant to localhost.
"""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


class NoCacheHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = functools.partial(NoCacheHandler, directory=str(WEB))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Gesture Studio at http://localhost:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
