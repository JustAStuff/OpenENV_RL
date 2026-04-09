# Copyright (c) 2026 CtrlAltWin Team
"""
Tiffin Packing Environment — OpenEnv Server Implementation.

Wraps the packing simulation into the OpenEnv Environment base class,
exposing step(), reset(), and state() for LLM agent interaction.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server import Environment
except ImportError:
    # Fallback for local testing without openenv installed
    class Environment:
        def __init__(self, **kwargs): pass
        def reset(self, **kwargs): raise NotImplementedError
        def step(self, action, **kwargs): raise NotImplementedError
        @property
        def state(self): raise NotImplementedError

from tiffin_packer.models import TiffinAction, TiffinObservation, TiffinState
from tiffin_packer.simulation.engine import PackingSimulation
from tiffin_packer.vlm.classifier import FoodClassifier
from tiffin_packer.tasks import get_task_config, list_tasks
from tiffin_packer.grader import grade, grade_detailed


class TiffinPackingEnvironment(Environment):
    """
    OpenEnv-compliant tiffin packing environment.

    An LLM agent controls a robotic arm to identify food items using VLM
    and pack them into the correct tiffin containers under real-world
    constraints (type compatibility, volume, temperature, fragility).

    Supports 3 tasks: easy, medium, hard.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        super().__init__()
        self.sim = PackingSimulation()
        self.vlm = FoodClassifier()
        self._state = TiffinState()
        self._identified_items: set = set()
        self._task_config = None
        self._initialized = True

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TiffinObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional custom episode ID.
            **kwargs: Must include 'task_id' (easy/medium/hard).

        Returns:
            Initial TiffinObservation with scene description.
        """
        task_id = kwargs.get("task_id", "easy")

        # Load task configuration
        self._task_config = get_task_config(task_id, seed=seed)

        # Reset simulation
        self.sim.reset(
            food_items=self._task_config.food_items,
            containers=self._task_config.containers,
            seed=seed,
        )

        # Reset state
        self._state = TiffinState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            items_packed=0,
            total_items=len(self._task_config.food_items),
            items_identified=0,
            packing_log=[],
            constraints_violated=[],
        )
        self._identified_items = set()

        # Build initial observation
        return self._build_observation(
            reward=0.0,
            done=False,
            feedback=(
                f"Episode started! Task: {task_id.upper()}\n\n"
                f"{self._task_config.description}\n\n"
                f"You have {self._task_config.max_steps} steps to pack "
                f"{len(self._task_config.food_items)} food items into "
                f"{len(self._task_config.containers)} containers.\n\n"
                f"Start by using 'observe' to see the scene, then 'identify' "
                f"each food item before packing."
            ),
        )

    def step(
        self,
        action: TiffinAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TiffinObservation:
        """
        Execute one step in the environment.

        Args:
            action: TiffinAction with command and optional target_id.
            timeout_s: Optional timeout (unused).

        Returns:
            TiffinObservation with updated scene state.
        """
        self._state.step_count += 1
        reward = 0.0
        done = False
        vlm_result = None
        feedback = ""

        command = action.command.lower().strip()
        target_id = action.target_id

        # --- Dispatch command ---
        if command == "observe":
            _, feedback, reward = self.sim.observe()

        elif command == "identify":
            if target_id is None:
                feedback = "Error: 'identify' requires a target_id (food item ID)."
                reward = -0.1
            else:
                success, feedback, reward, vlm_result = self.sim.identify(target_id)
                if success and vlm_result and vlm_result.get("name"):
                    self._identified_items.add(target_id)
                    self._state.items_identified = len(self._identified_items)

        elif command == "pick":
            if target_id is None:
                feedback = "Error: 'pick' requires a target_id (food item ID)."
                reward = -0.1
            else:
                success, feedback, reward = self.sim.pick(target_id)

        elif command == "place":
            if target_id is None:
                feedback = "Error: 'place' requires a target_id (container ID)."
                reward = -0.1
            else:
                success, feedback, reward = self.sim.place(target_id)
                if success:
                    self._state.items_packed = sum(
                        1
                        for i in self.sim.food_items
                        if i.status == "packed"
                    )
                    self._state.packing_log = list(self.sim.packing_log)

        elif command == "pour":
            if target_id is None:
                feedback = "Error: 'pour' requires a target_id (container ID)."
                reward = -0.1
            else:
                success, feedback, reward = self.sim.pour(target_id)
                if success:
                    self._state.items_packed = sum(
                        1
                        for i in self.sim.food_items
                        if i.status == "packed"
                    )
                    self._state.packing_log = list(self.sim.packing_log)

        else:
            feedback = (
                f"Unknown command: '{command}'. "
                f"Available commands: {self.sim.get_available_commands()}"
            )
            reward = -0.1

        # --- Time penalty ---
        reward -= 0.02

        # --- Check termination ---
        done = (
            self.sim.all_packed
            or self._state.step_count >= self._task_config.max_steps
        )

        # --- Final grading ---
        final_score = None
        grade_breakdown = None
        if done:
            grade_breakdown = grade_detailed(
                self._state.packing_log, self._task_config
            )
            final_score = grade_breakdown["final_score"]
            reward += final_score  # bonus = final grade

            if self.sim.all_packed:
                feedback += f"\n\n🎉 All items packed! Final score: {final_score:.4f}"
            else:
                feedback += (
                    f"\n\n⏰ Time's up! {self.sim.unpacked_count} items remaining. "
                    f"Final score: {final_score:.4f}"
                )

        return self._build_observation(
            reward=reward,
            done=done,
            feedback=feedback,
            vlm_result=vlm_result,
            final_score=final_score,
            grade_breakdown=grade_breakdown,
        )

    @property
    def state(self) -> TiffinState:
        """Return the current episode state."""
        return self._state

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float = 0.0,
        done: bool = False,
        feedback: str = "",
        vlm_result: dict = None,
        final_score: float = None,
        grade_breakdown: dict = None,
    ) -> TiffinObservation:
        """Build a TiffinObservation from current state."""
        metadata = {}
        if final_score is not None:
            metadata["final_score"] = final_score
        if grade_breakdown is not None:
            metadata["grade_breakdown"] = grade_breakdown

        return TiffinObservation(
            done=done,
            reward=round(reward, 4),
            metadata=metadata,
            scene_description=self.sim.get_scene_description(),
            food_items=[
                item.to_dict(hide_unidentified=True)
                for item in self.sim.food_items
            ],
            containers=[c.to_dict() for c in self.sim.containers],
            held_item=(
                self.sim.held_item.to_dict(hide_unidentified=False)
                if self.sim.held_item
                else None
            ),
            vlm_result=vlm_result,
            available_commands=self.sim.get_available_commands(),
            step_feedback=feedback,
        )
