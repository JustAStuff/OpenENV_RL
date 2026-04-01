# Copyright (c) 2026 CtrlAltWin Team
"""
Tiffin Packing Simulation Engine — Pure Logic + PyBullet Physics.

This module implements the core packing simulation. It operates in two modes:

1. **Logic mode** (default): Pure Python state tracking — fast, lightweight,
   guaranteed to run on 2 vCPU / 8 GB RAM. Used for all OpenEnv interactions.

2. **Physics mode** (optional): PyBullet simulation with real URDF models
   (Kuka arm, table, containers, food cubes/spheres). Used for rendering
   and visual validation.

The LLM agent issues high-level commands (pick, place, pour, identify),
and this engine validates and executes them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoodItem:
    """Represents a food item on the table."""

    id: int
    name: str
    food_type: str  # "solid" | "liquid" | "semi-solid"
    volume_ml: float
    temperature: str  # "hot" | "cold" | "room"
    fragility: float  # 0.0 (sturdy) to 1.0 (very fragile)
    preferred_container: str  # "sealed" | "flat" | "deep"
    color: str = "unknown"
    special_notes: str = ""
    status: str = "on_table"  # "on_table" | "held" | "packed" | "dropped"
    identified: bool = False
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self, hide_unidentified: bool = True) -> Dict[str, Any]:
        """Convert to observation dict. Hides properties if not yet identified."""
        base = {"id": self.id, "status": self.status}
        if self.identified or not hide_unidentified:
            base.update(
                {
                    "name": self.name,
                    "food_type": self.food_type,
                    "volume_ml": self.volume_ml,
                    "temperature": self.temperature,
                    "fragility": self.fragility,
                    "preferred_container": self.preferred_container,
                    "color": self.color,
                }
            )
        else:
            base["name"] = f"Unknown food item #{self.id}"
            base["food_type"] = "unknown"
            base["hint"] = "Use 'identify' command to classify this item"
        return base


@dataclass
class Container:
    """Represents a tiffin container."""

    id: int
    name: str
    container_type: str  # "sealed_round" | "flat_open" | "deep_box" | "small_sealed"
    capacity_ml: float
    filled_ml: float = 0.0
    contents: List[str] = field(default_factory=list)
    content_types: List[str] = field(default_factory=list)  # food types inside
    content_temperatures: List[str] = field(default_factory=list)
    content_fragilites: List[float] = field(default_factory=list)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def remaining_ml(self) -> float:
        return max(0, self.capacity_ml - self.filled_ml)

    @property
    def fill_percentage(self) -> float:
        return (self.filled_ml / self.capacity_ml) * 100 if self.capacity_ml > 0 else 0

    @property
    def accepts_liquid(self) -> bool:
        """Sealed containers can hold liquids."""
        return "sealed" in self.container_type

    @property
    def is_flat(self) -> bool:
        return "flat" in self.container_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.container_type,
            "capacity_ml": self.capacity_ml,
            "filled_ml": round(self.filled_ml, 1),
            "remaining_ml": round(self.remaining_ml, 1),
            "fill_percentage": round(self.fill_percentage, 1),
            "contents": self.contents,
            "accepts_liquid": self.accepts_liquid,
        }


# ---------------------------------------------------------------------------
# Compatibility rules
# ---------------------------------------------------------------------------

CONTAINER_TYPE_COMPATIBILITY = {
    # food_type -> set of compatible container_types
    "liquid": {"sealed_round", "small_sealed"},
    "semi-solid": {"sealed_round", "small_sealed", "deep_box"},
    "solid": {"sealed_round", "flat_open", "deep_box", "small_sealed"},
}


def is_type_compatible(food_type: str, container_type: str) -> bool:
    """Check if a food type is compatible with a container type."""
    compatible = CONTAINER_TYPE_COMPATIBILITY.get(food_type, set())
    return container_type in compatible


# ---------------------------------------------------------------------------
# Main simulation engine
# ---------------------------------------------------------------------------

class PackingSimulation:
    """
    Pure-logic tiffin packing simulation.

    Models a robotic arm, food items, and containers as data structures.
    Validates all actions against physical constraints (volume, type
    compatibility, temperature zones, fragility ordering).
    """

    def __init__(self):
        self.arm_state: str = "idle"  # "idle" | "holding"
        self.held_item: Optional[FoodItem] = None
        self.food_items: List[FoodItem] = []
        self.containers: List[Container] = []
        self.packing_log: List[Dict[str, Any]] = []
        self._step_count: int = 0

    def reset(
        self,
        food_items: List[FoodItem],
        containers: List[Container],
        seed: Optional[int] = None,
    ):
        """Initialize simulation with food items and containers."""
        if seed is not None:
            random.seed(seed)

        self.arm_state = "idle"
        self.held_item = None
        self.food_items = food_items
        self.containers = containers
        self.packing_log = []
        self._step_count = 0

        # Randomize positions on table
        for i, item in enumerate(self.food_items):
            angle = (2 * math.pi * i) / max(len(self.food_items), 1)
            item.position = (
                0.3 * math.cos(angle),
                0.3 * math.sin(angle),
                0.65,  # table height
            )

        for i, container in enumerate(self.containers):
            container.position = (
                -0.4 + 0.25 * i,
                0.5,
                0.65,
            )

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def observe(self) -> Tuple[bool, str, float]:
        """Get detailed scene description. Returns (success, feedback, reward)."""
        desc = self.get_scene_description()
        return True, desc, 0.05  # small reward for observing

    def identify(self, item_id: int) -> Tuple[bool, str, float, Optional[Dict]]:
        """
        Classify a food item using VLM.
        Returns (success, feedback, reward, vlm_result_or_None).
        """
        item = self._find_item(item_id)
        if item is None:
            return False, f"No food item with ID {item_id} found.", -0.1, None

        if item.status == "packed":
            return (
                False,
                f"Item #{item_id} ({item.name}) is already packed.",
                -0.05,
                None,
            )

        if item.identified:
            # Re-identifying is allowed but gives no reward
            vlm_result = {
                "name": item.name,
                "type": item.food_type,
                "fragility": item.fragility,
                "preferred_container": item.preferred_container,
                "volume_ml": item.volume_ml,
                "temperature": item.temperature,
                "color": item.color,
                "special_notes": item.special_notes,
            }
            return (
                True,
                f"Item #{item_id} already identified as '{item.name}'. {item.special_notes}",
                0.0,
                vlm_result,
            )

        # First-time identification
        item.identified = True
        vlm_result = {
            "name": item.name,
            "type": item.food_type,
            "fragility": item.fragility,
            "preferred_container": item.preferred_container,
            "volume_ml": item.volume_ml,
            "temperature": item.temperature,
            "color": item.color,
            "special_notes": item.special_notes,
        }
        return (
            True,
            f"VLM identified item #{item_id}: '{item.name}' — "
            f"type={item.food_type}, volume={item.volume_ml}ml, "
            f"temperature={item.temperature}, fragility={item.fragility:.1f}, "
            f"preferred container={item.preferred_container}. "
            f"Note: {item.special_notes}",
            0.1,  # reward for gathering information
            vlm_result,
        )

    def pick(self, item_id: int) -> Tuple[bool, str, float]:
        """Pick up a food item. Returns (success, feedback, reward)."""
        if self.arm_state == "holding":
            return (
                False,
                f"Arm is already holding '{self.held_item.name}'. "
                f"Place or pour it first before picking another item.",
                -0.1,
            )

        item = self._find_item(item_id)
        if item is None:
            return False, f"No food item with ID {item_id} found.", -0.1

        if item.status != "on_table":
            return (
                False,
                f"Item #{item_id} ({item.name}) cannot be picked — status is '{item.status}'.",
                -0.1,
            )

        # Success — pick up the item
        item.status = "held"
        self.held_item = item
        self.arm_state = "holding"
        return (
            True,
            f"Successfully picked up item #{item_id} "
            f"({'identified as ' + item.name if item.identified else 'unidentified'}).",
            0.3,
        )

    def place(self, container_id: int) -> Tuple[bool, str, float]:
        """Place held item into container. Returns (success, feedback, reward)."""
        if self.arm_state != "holding" or self.held_item is None:
            return (
                False,
                "Arm is not holding any item. Use 'pick' first.",
                -0.1,
            )

        container = self._find_container(container_id)
        if container is None:
            return False, f"No container with ID {container_id} found.", -0.1

        item = self.held_item
        reward = 0.0
        feedback_parts = []

        # --- Check type compatibility ---
        type_ok = is_type_compatible(item.food_type, container.container_type)
        if not type_ok:
            reward -= 1.5
            feedback_parts.append(
                f"WARNING: {item.food_type} food in {container.container_type} "
                f"container is incompatible! (e.g. liquid will spill from open container)"
            )

        # --- Check volume overflow ---
        if item.volume_ml > container.remaining_ml:
            overflow = item.volume_ml - container.remaining_ml
            reward -= 1.0
            feedback_parts.append(
                f"WARNING: Overflow! Item needs {item.volume_ml}ml but container "
                f"only has {container.remaining_ml:.0f}ml remaining. "
                f"Overflow of {overflow:.0f}ml!"
            )

        # --- Check temperature mixing ---
        if container.content_temperatures:
            existing_temps = set(container.content_temperatures)
            if item.temperature == "hot" and "cold" in existing_temps:
                reward -= 0.5
                feedback_parts.append(
                    "WARNING: Placing hot food with cold items! "
                    "Temperature contamination will occur."
                )
            elif item.temperature == "cold" and "hot" in existing_temps:
                reward -= 0.5
                feedback_parts.append(
                    "WARNING: Placing cold food with hot items! "
                    "Temperature contamination will occur."
                )

        # --- Check fragility ---
        if container.content_fragilites and item.fragility < 0.5:
            # Placing heavy/sturdy item — check if fragile items are under
            max_existing_fragility = max(container.content_fragilites)
            if max_existing_fragility > 0.6:
                reward -= 0.3
                feedback_parts.append(
                    f"WARNING: Placing sturdy item on top of fragile item "
                    f"(fragility {max_existing_fragility:.1f}) — may crush it!"
                )

        # --- Positive rewards ---
        if type_ok:
            reward += 1.5  # correct container type
            if container.container_type == item.preferred_container or (
                item.preferred_container in container.container_type
            ):
                reward += 0.5  # preferred container bonus
                feedback_parts.append("Great choice — matches preferred container type!")

        if item.volume_ml <= container.remaining_ml:
            # Good volume fit
            utilization = item.volume_ml / container.capacity_ml
            reward += 0.3 * utilization  # reward proportional to space usage

        # --- Execute placement ---
        container.filled_ml += item.volume_ml
        container.contents.append(item.name)
        container.content_types.append(item.food_type)
        container.content_temperatures.append(item.temperature)
        container.content_fragilites.append(item.fragility)
        item.status = "packed"
        self.held_item = None
        self.arm_state = "idle"

        # Log the placement
        self.packing_log.append(
            {
                "food_name": item.name,
                "food_id": item.id,
                "food_type": item.food_type,
                "food_volume": item.volume_ml,
                "food_temperature": item.temperature,
                "food_fragility": item.fragility,
                "food_preferred_container": item.preferred_container,
                "container_id": container.id,
                "container_type": container.container_type,
                "container_name": container.name,
                "type_compatible": type_ok,
                "overflow": item.volume_ml > container.remaining_ml + item.volume_ml,
            }
        )

        if feedback_parts:
            feedback = f"Placed '{item.name}' in '{container.name}'. " + " ".join(
                feedback_parts
            )
        else:
            feedback = (
                f"Placed '{item.name}' in '{container.name}'. "
                f"Container now {container.fill_percentage:.0f}% full."
            )

        return True, feedback, round(reward, 2)

    def pour(self, container_id: int) -> Tuple[bool, str, float]:
        """Pour liquid from held item into container. Returns (success, feedback, reward)."""
        if self.arm_state != "holding" or self.held_item is None:
            return False, "Arm is not holding any item. Use 'pick' first.", -0.1

        item = self.held_item

        # Only liquids or semi-solids can be poured
        if item.food_type not in ("liquid", "semi-solid"):
            return (
                False,
                f"Cannot pour '{item.name}' — it is '{item.food_type}', not a pourable item. "
                f"Use 'place' instead.",
                -0.1,
            )

        # Pour is functionally same as place but gives extra reward for liquids
        success, feedback, reward = self.place(container_id)
        if success:
            reward += 0.2  # bonus for correctly using pour for liquids
            feedback = feedback.replace("Placed", "Poured")
        return success, feedback, reward

    # -------------------------------------------------------------------
    # Scene description
    # -------------------------------------------------------------------

    def get_scene_description(self) -> str:
        """Generate natural language description of the current scene."""
        lines = []
        lines.append("=" * 60)
        lines.append("TIFFIN PACKING SCENE")
        lines.append("=" * 60)

        # Arm state
        lines.append("")
        lines.append("🤖 ROBOTIC ARM STATUS:")
        if self.arm_state == "holding" and self.held_item:
            item = self.held_item
            if item.identified:
                lines.append(
                    f"   Currently holding: {item.name} "
                    f"(type={item.food_type}, volume={item.volume_ml}ml)"
                )
            else:
                lines.append(f"   Currently holding: Unknown food item #{item.id}")
        else:
            lines.append("   Arm is idle — ready to pick up an item")

        # Food items on table
        on_table = [i for i in self.food_items if i.status == "on_table"]
        packed = [i for i in self.food_items if i.status == "packed"]
        lines.append("")
        lines.append(f"🍛 FOOD ITEMS ON TABLE ({len(on_table)} remaining, {len(packed)} packed):")
        for item in self.food_items:
            status_icon = {"on_table": "⬜", "held": "🤏", "packed": "✅", "dropped": "❌"}.get(
                item.status, "?"
            )
            if item.identified:
                lines.append(
                    f"   {status_icon} [{item.id}] {item.name} — "
                    f"type={item.food_type}, volume={item.volume_ml}ml, "
                    f"temp={item.temperature}, fragility={item.fragility:.1f}, "
                    f"preferred={item.preferred_container}"
                )
            else:
                lines.append(
                    f"   {status_icon} [{item.id}] Unknown food item "
                    f"(use 'identify' to classify)"
                )

        # Containers
        lines.append("")
        lines.append("🍱 TIFFIN CONTAINERS:")
        for c in self.containers:
            bar_len = 20
            filled_bars = int((c.fill_percentage / 100) * bar_len)
            bar = "█" * filled_bars + "░" * (bar_len - filled_bars)
            lines.append(
                f"   [{c.id}] {c.name} ({c.container_type}) — "
                f"[{bar}] {c.fill_percentage:.0f}% "
                f"({c.filled_ml:.0f}/{c.capacity_ml:.0f}ml)"
            )
            if c.contents:
                lines.append(f"       Contains: {', '.join(c.contents)}")
            liquid_note = "✅ Can hold liquids" if c.accepts_liquid else "⚠️ Open — no liquids"
            lines.append(f"       {liquid_note}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def get_available_commands(self) -> List[str]:
        """Return list of valid commands given current state."""
        commands = ["observe"]

        unpacked = [i for i in self.food_items if i.status == "on_table"]
        unidentified = [i for i in self.food_items if not i.identified and i.status != "packed"]

        if unidentified:
            commands.append("identify")

        if self.arm_state == "idle" and unpacked:
            commands.append("pick")

        if self.arm_state == "holding" and self.held_item:
            commands.append("place")
            if self.held_item.food_type in ("liquid", "semi-solid"):
                commands.append("pour")

        return commands

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _find_item(self, item_id: int) -> Optional[FoodItem]:
        for item in self.food_items:
            if item.id == item_id:
                return item
        return None

    def _find_container(self, container_id: int) -> Optional[Container]:
        for c in self.containers:
            if c.id == container_id:
                return c
        return None

    @property
    def all_packed(self) -> bool:
        return all(i.status == "packed" for i in self.food_items)

    @property
    def unpacked_count(self) -> int:
        return sum(1 for i in self.food_items if i.status != "packed")
