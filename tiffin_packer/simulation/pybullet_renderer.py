# Copyright (c) 2026 CtrlAltWin Team
"""
PyBullet Rendering Module — Real physics visualization using URDF models.

Provides an optional physics-backed renderer that loads real URDF models
(Kuka robot arm, table, containers, food items) and renders frames.

This module is used for:
1. Generating visual frames for the frontend viewer
2. Physics validation of placements
3. Demo/presentation screenshots

The simulation engine (engine.py) handles all logic — this module only
provides visualization and optional physics validation.
"""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# PyBullet may not be available in all environments
try:
    import pybullet as p
    import pybullet_data

    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


# Color presets for food items
FOOD_COLORS = {
    "rice": [1.0, 1.0, 0.9, 1.0],  # white
    "sambar": [0.9, 0.5, 0.1, 1.0],  # orange
    "curd": [1.0, 1.0, 0.95, 1.0],  # off-white
    "chapati": [0.8, 0.6, 0.3, 1.0],  # brown
    "pickle": [0.8, 0.1, 0.1, 1.0],  # red
    "dal": [0.9, 0.8, 0.2, 1.0],  # yellow
    "rasam": [0.6, 0.1, 0.05, 1.0],  # dark red
    "poriyal": [0.2, 0.7, 0.2, 1.0],  # green
    "papad": [0.9, 0.8, 0.4, 1.0],  # golden
    "raita": [0.8, 0.9, 0.8, 1.0],  # pale green
    "idli": [1.0, 1.0, 0.95, 1.0],  # white
    "chutney": [0.1, 0.6, 0.1, 1.0],  # green
    "biryani": [0.9, 0.7, 0.2, 1.0],  # saffron
    "curry": [0.6, 0.3, 0.1, 1.0],  # brown
    "salad": [0.3, 0.8, 0.3, 1.0],  # mixed green
}

# Container colors
CONTAINER_COLORS = {
    "sealed_round": [0.7, 0.7, 0.8, 0.7],  # steel blue
    "flat_open": [0.8, 0.6, 0.3, 0.8],  # bronze
    "deep_box": [0.6, 0.6, 0.7, 0.7],  # grey steel
    "small_sealed": [0.9, 0.9, 0.95, 0.7],  # silver
}


class PyBulletRenderer:
    """
    Optional PyBullet-based renderer for the tiffin packing scene.

    Creates a physics simulation with:
    - Kuka IIWA robot arm (from pybullet_data)
    - Table (box primitive)
    - Food items (colored cubes/spheres on table)
    - Tiffin containers (open-top box composites)
    """

    def __init__(self, gui: bool = False):
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "pybullet is not installed. Install with: pip install pybullet"
            )

        self._gui = gui
        self._physics_client = None
        self._robot_id = None
        self._table_id = None
        self._food_ids: Dict[int, int] = {}  # food_item_id -> bullet_body_id
        self._container_ids: Dict[int, int] = {}  # container_id -> bullet_body_id
        self._initialized = False

    def initialize(self):
        """Start the PyBullet physics server."""
        if self._initialized:
            return

        if self._gui:
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground plane
        p.loadURDF("plane.urdf")

        self._initialized = True

    def setup_scene(
        self,
        food_items: list,
        containers: list,
    ):
        """
        Set up the full PyBullet scene with robot, table, food, containers.

        Args:
            food_items: List of FoodItem dataclasses
            containers: List of Container dataclasses
        """
        self.initialize()

        # Clear previous objects
        self._clear_objects()

        # --- Table ---
        table_half_extents = [0.4, 0.6, 0.02]
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        table_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=table_half_extents,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
        )
        self._table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0, 0, 0.6],
        )

        # Table legs
        for lx, ly in [(-0.35, -0.55), (-0.35, 0.55), (0.35, -0.55), (0.35, 0.55)]:
            leg_col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.3]
            )
            leg_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.02, 0.02, 0.3],
                rgbaColor=[0.5, 0.3, 0.15, 1.0],
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=leg_col,
                baseVisualShapeIndex=leg_vis,
                basePosition=[lx, ly, 0.3],
            )

        # --- Robot arm (Kuka IIWA) ---
        self._robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[-0.5, 0, 0.62],
            useFixedBase=True,
        )

        # --- Food items ---
        for item in food_items:
            color = FOOD_COLORS.get(item.name, [0.5, 0.5, 0.5, 1.0])

            if item.food_type == "liquid":
                # Sphere for liquids
                shape_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
                shape_vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.03, rgbaColor=color
                )
            elif item.fragility > 0.6:
                # Flat disc for fragile items (papad, chapati)
                shape_col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=0.04, height=0.01
                )
                shape_vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=0.04,
                    length=0.01,
                    rgbaColor=color,
                )
            else:
                # Cube for solid foods
                sz = 0.025
                shape_col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[sz, sz, sz]
                )
                shape_vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[sz, sz, sz], rgbaColor=color
                )

            body_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=shape_col,
                baseVisualShapeIndex=shape_vis,
                basePosition=[
                    item.position[0],
                    item.position[1],
                    item.position[2] + 0.03,
                ],
            )
            self._food_ids[item.id] = body_id

        # --- Containers (open-top boxes) ---
        for container in containers:
            color = CONTAINER_COLORS.get(
                container.container_type, [0.5, 0.5, 0.5, 0.7]
            )
            # Scale container size based on capacity
            scale = (container.capacity_ml / 300) ** 0.33
            w, d, h = 0.05 * scale, 0.05 * scale, 0.06 * scale

            # Bottom
            bottom_col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[w, d, 0.002]
            )
            bottom_vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[w, d, 0.002], rgbaColor=color
            )
            cx, cy, cz = container.position
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=bottom_col,
                baseVisualShapeIndex=bottom_vis,
                basePosition=[cx, cy, cz],
            )
            self._container_ids[container.id] = body_id

            # Walls (4 sides)
            wall_thickness = 0.003
            walls = [
                ([w, wall_thickness, h / 2], [cx, cy + d, cz + h / 2]),
                ([w, wall_thickness, h / 2], [cx, cy - d, cz + h / 2]),
                ([wall_thickness, d, h / 2], [cx + w, cy, cz + h / 2]),
                ([wall_thickness, d, h / 2], [cx - w, cy, cz + h / 2]),
            ]
            for wall_ext, wall_pos in walls:
                wall_col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=wall_ext
                )
                wall_vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=wall_ext, rgbaColor=color
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=wall_col,
                    baseVisualShapeIndex=wall_vis,
                    basePosition=wall_pos,
                )

        # Set up camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.6],
        )

    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera_distance: float = 1.2,
        camera_yaw: float = 45,
        camera_pitch: float = -30,
    ) -> np.ndarray:
        """
        Render the current scene as an RGB image.

        Returns:
            numpy array of shape (height, width, 3) with RGB values.
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call setup_scene() first.")

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.6],
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=3.0,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb

    def render_base64(self, **kwargs) -> str:
        """Render scene and return as base64-encoded PNG string."""
        rgb = self.render(**kwargs)

        from PIL import Image

        img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def move_food_to_container(self, food_item_id: int, container_id: int):
        """Visually move a food item into a container (for animation)."""
        if food_item_id not in self._food_ids or container_id not in self._container_ids:
            return

        food_body = self._food_ids[food_item_id]
        container_body = self._container_ids[container_id]

        # Get container position
        pos, _ = p.getBasePositionAndOrientation(container_body)
        # Place food slightly above container center
        new_pos = [pos[0], pos[1], pos[2] + 0.05]
        p.resetBasePositionAndOrientation(
            food_body, new_pos, [0, 0, 0, 1]
        )

    def close(self):
        """Disconnect from PyBullet."""
        if self._initialized:
            p.disconnect(self._physics_client)
            self._initialized = False

    def _clear_objects(self):
        """Remove all food and container objects."""
        for body_id in self._food_ids.values():
            try:
                p.removeBody(body_id)
            except Exception:
                pass
        for body_id in self._container_ids.values():
            try:
                p.removeBody(body_id)
            except Exception:
                pass
        self._food_ids.clear()
        self._container_ids.clear()

    def __del__(self):
        self.close()
