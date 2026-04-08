"""
server/app.py — Entry point for multi-mode deployment (required by openenv validate).
This module re-exports the FastAPI app from the root and provides a main() entry
point so that the 'serve' script in pyproject.toml can start the server.
"""
import sys
import os

# Ensure root is on path so 'from models import ...' etc. resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the app created in the root app.py
from app import app  # noqa: F401 – imported for use by uvicorn / gunicorn


def main():
    """Console-script entry point: `serve` → starts uvicorn on port 7860."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    main()
