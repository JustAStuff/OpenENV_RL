# Copyright (c) 2026 CtrlAltWin Team
"""
Tiffin Packer Environment Client.

Provides the client for connecting to a running TiffinPackingEnvironment server.
"""

try:
    from openenv.core.env_client import EnvClient
except ImportError:
    # Fallback if openenv not installed
    EnvClient = object

from .models import TiffinAction, TiffinObservation


class TiffinEnv(EnvClient):
    """
    Client for the Tiffin Packing Environment.

    Example:
        >>> with TiffinEnv(base_url="http://localhost:7860").sync() as env:
        ...     obs = env.reset(task_id="easy")
        ...     obs = env.step(TiffinAction(command="observe"))
        ...     print(obs.scene_description)
    """

    pass  # EnvClient provides all needed functionality
