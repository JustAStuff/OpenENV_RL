# Copyright (c) 2026 CtrlAltWin Team
"""
FastAPI application for the Tiffin Packing Environment.

Creates an HTTP + WebSocket server exposing the TiffinPackingEnvironment
via the OpenEnv interface.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server import create_app

from tiffin_packer.models import TiffinAction, TiffinObservation
from server.tiffin_environment import TiffinPackingEnvironment

# Create the FastAPI app
# Pass the class (factory) for WebSocket session support
app = create_app(
    TiffinPackingEnvironment,
    TiffinAction,
    TiffinObservation,
    env_name="tiffin_packer",
)


def main():
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
