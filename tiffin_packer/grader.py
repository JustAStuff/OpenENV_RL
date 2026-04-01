# Copyright (c) 2026 CtrlAltWin Team
"""
Deterministic Grader — Scores packing quality from 0.0 to 1.0.

Scoring formula:
    score = 0.4 * validity + 0.3 * efficiency + 0.2 * constraints + 0.1 * neatness

Each component:
    validity  — food placed in type-compatible container?
    efficiency — space utilization vs total capacity used
    constraints — temperature separation, fragility, flavor isolation
    neatness  — all items packed? nothing dropped?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .tasks import TaskConfig
from .simulation.engine import is_type_compatible


def grade(
    packing_log: List[Dict[str, Any]],
    task_config: TaskConfig,
) -> float:
    """
    Grade a packing episode. Returns score between 0.0 and 1.0.

    Args:
        packing_log: List of placement records from the simulation.
        task_config: The task configuration used for this episode.

    Returns:
        Final score (0.0 to 1.0), rounded to 4 decimal places.
    """
    total_items = len(task_config.food_items)

    if total_items == 0:
        return 0.0

    # ---- Validity (40%) ----
    validity = _score_validity(packing_log, total_items)

    # ---- Efficiency (30%) ----
    efficiency = _score_efficiency(packing_log, task_config)

    # ---- Constraint Satisfaction (20%) ----
    constraints = _score_constraints(packing_log, task_config)

    # ---- Neatness (10%) ----
    neatness = _score_neatness(packing_log, total_items)

    # ---- Final score ----
    score = 0.4 * validity + 0.3 * efficiency + 0.2 * constraints + 0.1 * neatness
    return round(max(0.0, min(1.0, score)), 4)


def grade_detailed(
    packing_log: List[Dict[str, Any]],
    task_config: TaskConfig,
) -> Dict[str, Any]:
    """Grade with full breakdown for debugging."""
    total_items = len(task_config.food_items)

    validity = _score_validity(packing_log, total_items)
    efficiency = _score_efficiency(packing_log, task_config)
    constraints = _score_constraints(packing_log, task_config)
    neatness = _score_neatness(packing_log, total_items)

    score = 0.4 * validity + 0.3 * efficiency + 0.2 * constraints + 0.1 * neatness
    score = round(max(0.0, min(1.0, score)), 4)

    return {
        "final_score": score,
        "validity": round(validity, 4),
        "efficiency": round(efficiency, 4),
        "constraints": round(constraints, 4),
        "neatness": round(neatness, 4),
        "items_packed": len(packing_log),
        "total_items": total_items,
        "weights": {
            "validity": 0.4,
            "efficiency": 0.3,
            "constraints": 0.2,
            "neatness": 0.1,
        },
    }


# -----------------------------------------------------------------------
# Component scorers
# -----------------------------------------------------------------------


def _score_validity(packing_log: List[Dict], total_items: int) -> float:
    """Score: food placed in type-compatible container? (0-1)"""
    if not packing_log:
        return 0.0

    correct = sum(1 for entry in packing_log if entry.get("type_compatible", False))
    return correct / max(total_items, 1)


def _score_efficiency(packing_log: List[Dict], task_config: TaskConfig) -> float:
    """Score: how well is container space utilized? (0-1)"""
    if not packing_log:
        return 0.0

    total_food_vol = sum(entry.get("food_volume", 0) for entry in packing_log)

    # Find which containers were used
    used_container_ids = set(entry.get("container_id") for entry in packing_log)
    total_capacity = sum(
        c.capacity_ml
        for c in task_config.containers
        if c.id in used_container_ids
    )

    if total_capacity == 0:
        return 0.0

    utilization = total_food_vol / total_capacity

    # Penalize overflow
    overflow_count = sum(1 for entry in packing_log if entry.get("overflow", False))
    if overflow_count > 0:
        utilization *= max(0.3, 1.0 - 0.2 * overflow_count)

    return min(1.0, utilization)


def _score_constraints(packing_log: List[Dict], task_config: TaskConfig) -> float:
    """Score: task-specific constraints satisfied? (0-1)"""
    if not packing_log:
        return 0.0

    scores = []
    active = set(task_config.constraints)

    if "temperature_separation" in active:
        scores.append(_check_temperature(packing_log))

    if "fragility_ordering" in active:
        scores.append(_check_fragility(packing_log))

    if "flavor_isolation" in active:
        scores.append(_check_flavor_isolation(packing_log))

    if "no_overflow" in active:
        overflow_count = sum(1 for e in packing_log if e.get("overflow", False))
        scores.append(1.0 if overflow_count == 0 else max(0.0, 1.0 - 0.3 * overflow_count))

    if "type_match" in active:
        correct = sum(1 for e in packing_log if e.get("type_compatible", False))
        scores.append(correct / max(len(packing_log), 1))

    if not scores:
        return 1.0  # no constraints to violate

    return sum(scores) / len(scores)


def _check_temperature(packing_log: List[Dict]) -> float:
    """Check if hot and cold items are kept separate."""
    # Group items by container
    container_temps: Dict[int, List[str]] = {}
    for entry in packing_log:
        cid = entry.get("container_id")
        temp = entry.get("food_temperature", "room")
        container_temps.setdefault(cid, []).append(temp)

    violations = 0
    total_containers = len(container_temps)
    for temps in container_temps.values():
        if "hot" in temps and "cold" in temps:
            violations += 1

    if total_containers == 0:
        return 1.0
    return max(0.0, 1.0 - violations / total_containers)


def _check_fragility(packing_log: List[Dict]) -> float:
    """Check if fragile items are not crushed by heavy items placed after them."""
    # Group by container, check placement order
    container_order: Dict[int, List[float]] = {}
    for entry in packing_log:
        cid = entry.get("container_id")
        frag = entry.get("food_fragility", 0.5)
        container_order.setdefault(cid, []).append(frag)

    violations = 0
    checks = 0
    for fragilites in container_order.values():
        for i in range(1, len(fragilites)):
            checks += 1
            # If a less fragile (heavy) item is placed AFTER a more fragile item
            if fragilites[i] < 0.4 and fragilites[i - 1] > 0.6:
                violations += 1

    if checks == 0:
        return 1.0
    return max(0.0, 1.0 - violations / max(checks, 1))


def _check_flavor_isolation(packing_log: List[Dict]) -> float:
    """Check that strong-flavor items (pickle, chutney) are isolated."""
    strong_flavors = {"pickle", "chutney"}
    # Group by container
    container_contents: Dict[int, List[str]] = {}
    for entry in packing_log:
        cid = entry.get("container_id")
        name = entry.get("food_name", "")
        container_contents.setdefault(cid, []).append(name)

    violations = 0
    total = 0
    for contents in container_contents.values():
        has_strong = any(c in strong_flavors for c in contents)
        has_others = any(c not in strong_flavors for c in contents)
        if has_strong and has_others and len(contents) > 1:
            violations += 1
            total += 1
        elif has_strong:
            total += 1

    if total == 0:
        return 1.0
    return max(0.0, 1.0 - violations / max(total, 1))


def _score_neatness(packing_log: List[Dict], total_items: int) -> float:
    """Score: fraction of items successfully packed. (0-1)"""
    if total_items == 0:
        return 0.0
    return len(packing_log) / total_items
