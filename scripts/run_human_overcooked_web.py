from __future__ import annotations

import sys

import uvicorn

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Starting Overcooked web server on port {port}")
    print(f"Open http://localhost:{port} to play")
    uvicorn.run(
        "multitask_personalization.web.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
