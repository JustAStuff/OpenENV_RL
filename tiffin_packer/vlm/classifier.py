# Copyright (c) 2026 CtrlAltWin Team
"""
VLM Food Classifier — Simulates Vision-Language Model food classification.

In production, this would call LLaVA / GPT-4V on a rendered PyBullet frame.
For the hackathon, uses pre-computed attributes from food_db.json.

The agent MUST call 'identify' before it knows a food item's properties.
Without identification, items appear as generic "Unknown food item".
"""

import json
import os
from typing import Any, Dict, Optional


class FoodClassifier:
    """Cached VLM food classifier.

    Loads pre-computed food attributes from food_db.json.
    In a production system, replace ``classify()`` with a real
    VLM API call (e.g. LLaVA, GPT-4V) on a rendered scene frame.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "food_db.json")
        with open(db_path, "r") as f:
            self.food_db: Dict[str, Dict[str, Any]] = json.load(f)

    def classify(self, food_name: str) -> Dict[str, Any]:
        """Classify a food item and return its attributes.

        Args:
            food_name: Name of the food item (e.g. "sambar", "rice").

        Returns:
            Dict with keys: type, fragility, preferred_container,
            volume_ml, temperature, color, special_notes.
        """
        key = food_name.lower().strip()
        if key in self.food_db:
            return {**self.food_db[key], "name": key, "classified": True}
        return self._unknown_default(food_name)

    def _unknown_default(self, food_name: str) -> Dict[str, Any]:
        """Fallback for foods not in the database."""
        return {
            "name": food_name,
            "type": "solid",
            "fragility": 0.5,
            "preferred_container": "deep",
            "volume_ml": 100,
            "temperature": "room",
            "color": "unknown",
            "special_notes": "Unknown food item — classification uncertain",
            "classified": False,
        }

    def get_all_foods(self) -> list:
        """Return list of all known food names."""
        return list(self.food_db.keys())
