#!/usr/bin/env python3
# Copyright (c) 2026 CtrlAltWin Team
"""
Tiffin Packer — OpenEnv Inference Script.

Runs an LLM agent against the tiffin packing environment using the
OpenAI Client API with environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier for inference
    HF_TOKEN      — Hugging Face / API key

Usage:
    API_BASE_URL=https://api.openai.com/v1 \
    MODEL_NAME=gpt-4o \
    HF_TOKEN=your-key \
    python inference.py
"""

import json
import os
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. LLM calls will fail.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a tiffin packing assistant that controls a robotic arm.
Your goal: pack Indian meal items into the correct tiffin containers.

COMMANDS — respond with ONLY a JSON object, no other text:
  {"command": "observe"}                    — See the full scene
  {"command": "identify", "target_id": N}   — Classify food item N using VLM
  {"command": "pick", "target_id": N}       — Pick up food item N
  {"command": "place", "target_id": N}      — Place held item into container N
  {"command": "pour", "target_id": N}       — Pour held liquid into container N

PACKING RULES:
1. ALWAYS identify items before packing (you cannot see food properties otherwise)
2. Liquids (sambar, dal, rasam, curry) → sealed containers only
3. Solids (rice, chapati, idli) → any container type
4. Semi-solids (curd, pickle, chutney) → sealed containers preferred
5. FRAGILE items (papad=0.9, chapati=0.7) → don't crush under heavy items
6. HOT and COLD food must NOT share a container
7. Don't overflow containers — check volume math!
8. Strong-flavor items (pickle, chutney) should be isolated

STRATEGY:
1. First: observe the scene
2. Then: identify ALL food items (one by one)
3. Then: plan which food goes where based on constraints
4. Finally: pick and place/pour each item

Respond with ONLY valid JSON. No explanation, no markdown, no extra text."""


def parse_action(text: str) -> dict:
    """Parse LLM output into an action dict."""
    text = text.strip()

    # Try to extract JSON from the text
    if text.startswith("```"):
        # Handle markdown code blocks
        lines = text.split("\n")
        json_lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(json_lines).strip()

    # Try direct JSON parse
    try:
        action = json.loads(text)
        if "command" in action:
            return action
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the text
    for i in range(len(text)):
        if text[i] == "{":
            for j in range(len(text) - 1, i, -1):
                if text[j] == "}":
                    try:
                        action = json.loads(text[i : j + 1])
                        if "command" in action:
                            return action
                    except json.JSONDecodeError:
                        continue

    # Fallback
    print(f"  [WARN] Could not parse action: {text[:100]}")
    return {"command": "observe"}


def run_episode(task_id: str) -> dict:
    """Run one episode of the tiffin packing task."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset the environment
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        obs = result.get("observation", result)
    except Exception as e:
        print(f"  ERROR: Failed to reset environment: {e}")
        return {"task_id": task_id, "reward": 0.0, "score": 0.0, "error": str(e)}

    # Initialize conversation
    init_scene = obs.get("scene_description", "")
    init_feedback = obs.get("step_feedback", "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_id}\n\n"
                f"{init_feedback}\n\n"
                f"Scene:\n{init_scene}\n\n"
                f"Available commands: {obs.get('available_commands', [])}\n\n"
                f"What is your first action? Respond with JSON only."
            ),
        },
    ]

    total_reward = 0.0
    step = 0
    max_steps = 35  # safety limit

    while not obs.get("done", False) and step < max_steps:
        step += 1

        # Get LLM decision
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            action_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Step {step}] LLM error: {e}")
            action_text = '{"command": "observe"}'

        action = parse_action(action_text)
        print(f"  [Step {step}] Action: {json.dumps(action)}")

        # Execute step
        try:
            resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            obs = result.get("observation", result)
            reward = result.get("reward", obs.get("reward", 0.0))
            total_reward += reward or 0
        except Exception as e:
            print(f"  [Step {step}] Step error: {e}")
            break

        # Print feedback
        feedback = obs.get("step_feedback", "")[:200]
        print(f"           Reward: {reward:+.2f} | Feedback: {feedback}")

        # Update conversation with assistant response and new observation
        messages.append({"role": "assistant", "content": action_text})

        # Build concise next observation for LLM
        held = obs.get("held_item")
        held_str = (
            f"Holding: {held.get('name', 'unknown')}" if held else "Arm: idle"
        )
        items_status = [
            f"[{i['id']}] {i.get('name', '?')} ({i['status']})"
            for i in obs.get("food_items", [])
        ]
        containers_status = [
            f"[{c['id']}] {c['name']} {c.get('fill_percentage',0):.0f}% full"
            for c in obs.get("containers", [])
        ]

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Step {step} result (reward={reward:+.2f}):\n"
                    f"Feedback: {obs.get('step_feedback', '')}\n\n"
                    f"{held_str}\n"
                    f"Items: {', '.join(items_status)}\n"
                    f"Containers: {', '.join(containers_status)}\n"
                    f"Available: {obs.get('available_commands', [])}\n\n"
                    f"{'VLM Result: ' + json.dumps(obs.get('vlm_result')) if obs.get('vlm_result') else ''}\n\n"
                    f"Next action? JSON only."
                ),
            },
        )

    # Extract final score
    final_score = obs.get("metadata", {}).get("final_score", 0.0)
    grade_breakdown = obs.get("metadata", {}).get("grade_breakdown", {})

    print(f"\n  {'─'*40}")
    print(f"  Steps taken:  {step}")
    print(f"  Total reward: {total_reward:+.2f}")
    print(f"  Final score:  {final_score:.4f}")
    if grade_breakdown:
        print(f"  Breakdown:")
        print(f"    Validity:    {grade_breakdown.get('validity', 0):.4f} (x0.4)")
        print(f"    Efficiency:  {grade_breakdown.get('efficiency', 0):.4f} (x0.3)")
        print(f"    Constraints: {grade_breakdown.get('constraints', 0):.4f} (x0.2)")
        print(f"    Neatness:    {grade_breakdown.get('neatness', 0):.4f} (x0.1)")

    return {
        "task_id": task_id,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "score": final_score,
        "grade_breakdown": grade_breakdown,
    }


def main():
    """Run all 3 tasks and report results."""
    print("=" * 60)
    print("  TIFFIN PACKER — INFERENCE SCRIPT")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print(f"  Env:   {ENV_URL}")
    print("=" * 60)

    start_time = time.time()
    results = {}

    for task_id in ["easy", "medium", "hard"]:
        result = run_episode(task_id)
        results[task_id] = result

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for task_id, r in results.items():
        print(f"  {task_id:8s}: score={r['score']:.4f}  reward={r['total_reward']:+.2f}  steps={r.get('steps', '?')}")

    avg_score = sum(r["score"] for r in results.values()) / max(len(results), 1)
    print(f"\n  Average score: {avg_score:.4f}")
    print(f"  Total time:    {elapsed:.1f}s")

    # Save results
    os.makedirs("outputs/evals", exist_ok=True)
    with open("outputs/evals/results.json", "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "results": results,
                "average_score": avg_score,
                "elapsed_seconds": round(elapsed, 1),
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to outputs/evals/results.json")


if __name__ == "__main__":
    main()
