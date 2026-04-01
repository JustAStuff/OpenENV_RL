# Copyright (c) 2026 CtrlAltWin Team
# Smart Tiffin Packing Environment — Pydantic Models

"""
Typed data models for the Tiffin Packing OpenEnv environment.
Follows the OpenEnv specification with Action, Observation, and State base classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    try:
        from openenv.core.env_server.types import Action, Observation, State
    except ImportError:
        # Fallback: define compatible base classes when openenv is not installed
        from pydantic import BaseModel, ConfigDict

        class Action(BaseModel):
            model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class Observation(BaseModel):
            model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
            done: bool = Field(default=False)
            reward: Optional[float] = Field(default=None)
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class State(BaseModel):
            model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)
            episode_id: Optional[str] = Field(default=None)
            step_count: int = Field(default=0)


class TiffinAction(Action):
    """
    High-level command the LLM agent issues to the robotic arm.

    Available commands:
        - "observe"  : Get a full scene description (no target_id needed)
        - "identify" : Use VLM to classify a food item (target_id = food item ID)
        - "pick"     : Pick up a food item with the robotic arm (target_id = food item ID)
        - "place"    : Place the currently held item into a container (target_id = container ID)
        - "pour"     : Pour liquid from held bowl into a container (target_id = container ID)

    Attributes:
        command: The action command string.
        target_id: The ID of the food item or container to act on.
    """

    command: str = Field(
        description="One of: 'observe', 'identify', 'pick', 'place', 'pour'"
    )
    target_id: Optional[int] = Field(
        default=None,
        description="ID of food item (for identify/pick) or container (for place/pour)",
    )


class TiffinObservation(Observation):
    """
    Observation returned after each action.

    Contains a natural-language scene description, structured data about
    food items and containers, and feedback on the last action.

    Attributes:
        scene_description: Human-readable text describing the current scene.
        food_items: List of food item dicts with id, name, status, etc.
        containers: List of container dicts with id, type, capacity, contents.
        held_item: The food item currently held by the robotic arm, if any.
        vlm_result: VLM classification result after an 'identify' command.
        available_commands: Commands the agent can issue right now.
        step_feedback: Text feedback on the outcome of the last action.
    """

    scene_description: str = Field(
        default="", description="Natural language description of current scene state"
    )
    food_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of food items: [{id, name, status, position}]",
    )
    containers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of containers: [{id, type, capacity_ml, filled_ml, contents}]",
    )
    held_item: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Currently held food item, or None if gripper is empty",
    )
    vlm_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="VLM classification result after 'identify' command",
    )
    available_commands: List[str] = Field(
        default_factory=list,
        description="Valid commands the agent can issue right now",
    )
    step_feedback: str = Field(
        default="", description="Feedback on the last action (success/failure reason)"
    )


class TiffinState(State):
    """
    Internal episode state for tracking progress.

    Attributes:
        task_id: Which task is active (easy/medium/hard).
        items_packed: Number of items successfully packed.
        total_items: Total items that need to be packed.
        items_identified: Number of items that have been VLM-classified.
        packing_log: Record of each placement decision.
        constraints_violated: List of constraint violations.
    """

    task_id: str = Field(default="easy", description="Active task ID")
    items_packed: int = Field(default=0, description="Items successfully packed")
    total_items: int = Field(default=0, description="Total items to pack")
    items_identified: int = Field(default=0, description="Items VLM-classified")
    packing_log: List[Dict[str, Any]] = Field(
        default_factory=list, description="Record of placement decisions"
    )
    constraints_violated: List[str] = Field(
        default_factory=list, description="Constraint violations"
    )
