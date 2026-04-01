# Copyright (c) 2026 CtrlAltWin Team
"""
Task Definitions — Easy, Medium, Hard difficulty levels.

Each task defines what food items are on the table, what containers are
available, what constraints are active, and how many steps the agent gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .simulation.engine import Container, FoodItem
from .vlm.classifier import FoodClassifier


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    task_id: str
    description: str
    food_items: List[FoodItem]
    containers: List[Container]
    constraints: List[str]
    max_steps: int
    seed: Optional[int] = None


_vlm = FoodClassifier()


def _make_food(id: int, name: str) -> FoodItem:
    """Create a FoodItem from the VLM database."""
    attrs = _vlm.classify(name)
    return FoodItem(
        id=id,
        name=name,
        food_type=attrs["type"],
        volume_ml=attrs["volume_ml"],
        temperature=attrs["temperature"],
        fragility=attrs["fragility"],
        preferred_container=attrs["preferred_container"],
        color=attrs.get("color", "unknown"),
        special_notes=attrs.get("special_notes", ""),
    )


def get_task_config(task_id: str, seed: Optional[int] = None) -> TaskConfig:
    """Get task configuration by ID."""
    tasks = {
        "easy": _task_easy,
        "medium": _task_medium,
        "hard": _task_hard,
    }
    if task_id not in tasks:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {list(tasks.keys())}"
        )
    config = tasks[task_id](seed)
    return config


def _task_easy(seed: Optional[int] = None) -> TaskConfig:
    """
    Task 1 — Basic Packing (Easy)
    
    2 food items, 2 containers. Just match food type to container type.
    Rice (solid) → open/deep container, Sambar (liquid) → sealed container.
    """
    return TaskConfig(
        task_id="easy",
        description=(
            "Basic Packing: You have 2 food items (rice and sambar) and "
            "2 containers (one sealed, one open). Place each food item in "
            "a compatible container. Liquids must go in sealed containers."
        ),
        food_items=[
            _make_food(1, "rice"),
            _make_food(2, "sambar"),
        ],
        containers=[
            Container(
                id=1,
                name="Sealed Round Container",
                container_type="sealed_round",
                capacity_ml=300,
            ),
            Container(
                id=2,
                name="Flat Open Container",
                container_type="flat_open",
                capacity_ml=400,
            ),
        ],
        constraints=["type_match"],
        max_steps=12,
        seed=seed,
    )


def _task_medium(seed: Optional[int] = None) -> TaskConfig:
    """
    Task 2 — Efficient Packing (Medium)
    
    4 food items, 3 containers. Must match types AND avoid overflow.
    Hot/cold separation matters.
    """
    return TaskConfig(
        task_id="medium",
        description=(
            "Efficient Packing: You have 4 food items (rice, sambar, chapati, "
            "pickle) and 3 containers. Place each item correctly:\n"
            "- Match food type to container type (liquids → sealed)\n"
            "- Don't overflow containers (check volumes!)\n"
            "- Keep hot and cold items separate"
        ),
        food_items=[
            _make_food(1, "rice"),
            _make_food(2, "sambar"),
            _make_food(3, "chapati"),
            _make_food(4, "pickle"),
        ],
        containers=[
            Container(
                id=1,
                name="Sealed Round Container",
                container_type="sealed_round",
                capacity_ml=200,
            ),
            Container(
                id=2,
                name="Flat Open Container",
                container_type="flat_open",
                capacity_ml=300,
            ),
            Container(
                id=3,
                name="Deep Box Container",
                container_type="deep_box",
                capacity_ml=350,
            ),
        ],
        constraints=["type_match", "no_overflow", "temperature_separation"],
        max_steps=20,
        seed=seed,
    )


def _task_hard(seed: Optional[int] = None) -> TaskConfig:
    """
    Task 3 — Smart Packing (Hard)
    
    6 food items, 4 containers. Full constraint set:
    type match, overflow, temperature, fragility, flavor mixing.
    
    Key challenges:
    - Curd (cold) ≠ hot items in same container
    - Papad (fragility=0.9) must not be crushed
    - Curry + sambar both liquid+hot → total 300ml but sealed_round only 250ml!
    - Must split liquids across containers
    """
    return TaskConfig(
        task_id="hard",
        description=(
            "Smart Packing: You have 6 food items and 4 containers. This is a "
            "complex meal with many constraints:\n"
            "- Match food type to container type\n"
            "- Don't overflow (watch the math!)\n"
            "- Separate hot and cold items\n"
            "- Don't crush fragile items (papad, chapati)\n"
            "- Consider flavor isolation (pickle, chutney)\n"
            "\nItems: rice, sambar, curd, chapati, papad, curry\n"
            "Containers: sealed_round (250ml), flat_open (200ml), "
            "deep_box (400ml), small_sealed (100ml)"
        ),
        food_items=[
            _make_food(1, "rice"),
            _make_food(2, "sambar"),
            _make_food(3, "curd"),
            _make_food(4, "chapati"),
            _make_food(5, "papad"),
            _make_food(6, "curry"),
        ],
        containers=[
            Container(
                id=1,
                name="Sealed Round Container",
                container_type="sealed_round",
                capacity_ml=250,
            ),
            Container(
                id=2,
                name="Flat Open Container",
                container_type="flat_open",
                capacity_ml=200,
            ),
            Container(
                id=3,
                name="Deep Box Container",
                container_type="deep_box",
                capacity_ml=400,
            ),
            Container(
                id=4,
                name="Small Sealed Container",
                container_type="small_sealed",
                capacity_ml=100,
            ),
        ],
        constraints=[
            "type_match",
            "no_overflow",
            "temperature_separation",
            "fragility_ordering",
            "flavor_isolation",
        ],
        max_steps=30,
        seed=seed,
    )


def list_tasks() -> List[str]:
    """Return list of available task IDs."""
    return ["easy", "medium", "hard"]
