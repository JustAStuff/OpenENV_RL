# Copyright (c) 2026 CtrlAltWin Team
# Smart Tiffin Packing Environment for OpenEnv

"""
Tiffin Packer — A multimodal RL environment for semantic-aware
constrained packing tasks inspired by real-world Indian meal organization.

An LLM agent controls a robotic arm to identify food items (via VLM),
reason about container compatibility, and pack a complete Indian meal
into tiffin containers.
"""

from .models import TiffinAction, TiffinObservation, TiffinState

try:
    from .client import TiffinEnv
except ImportError:
    TiffinEnv = None  # Client requires openenv-core

__all__ = ["TiffinAction", "TiffinObservation", "TiffinState", "TiffinEnv"]
