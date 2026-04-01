# Smart Tiffin Packing Environment 🍱🤖

> **Semantic-aware constrained packing under real-world constraints**
>
> An OpenEnv-compliant RL environment where an LLM agent controls a robotic arm
> to pack an Indian tiffin meal. The agent uses VLM-derived food classification
> to reason about container compatibility, volume constraints, temperature zones,
> and fragility — then physically executes packing decisions.

## 🎯 What is this?

This environment simulates the real-world task of **packing an Indian meal into tiffin containers**. An AI agent must:

1. **Identify** food items using a Vision-Language Model (VLM)
2. **Reason** about which container each item should go into
3. **Execute** packing commands via a robotic arm
4. **Satisfy** multiple constraints simultaneously

### Why Tiffin Packing?

Every day, millions of people in India pack tiffin boxes for lunch. It's a genuine spatial-reasoning task with real constraints:
- Liquids (sambar, dal) must go in sealed containers
- Fragile items (papad, chapati) shouldn't be crushed
- Hot and cold foods should be separated
- Volume limits mean you can't just stuff everything in one box

## 🏗️ Architecture

```
LLM Agent (via OpenAI API)
    │
    ├── observe → See scene description
    ├── identify → VLM classifies food item
    ├── pick → Robotic arm picks up food
    ├── place → Place item in container
    └── pour → Pour liquid into container
    │
    ▼
OpenEnv Server (FastAPI)
    │
    ├── Simulation Engine (logic + PyBullet physics)
    ├── VLM Classifier (cached food_db.json)
    ├── Task Manager (easy/medium/hard)
    └── Deterministic Grader (0.0-1.0)
```

## 🎮 Tasks

| Task | Items | Containers | Constraints | Difficulty |
|------|-------|-----------|-------------|------------|
| 🟢 Easy | rice, sambar (2) | sealed, flat (2) | Type matching | Straightforward |
| 🟡 Medium | rice, sambar, chapati, pickle (4) | sealed, flat, deep (3) | Types + overflow + temperature | Requires reasoning |
| 🔴 Hard | rice, sambar, curd, chapati, papad, curry (6) | sealed, flat, deep, small_sealed (4) | All constraints active | Genuinely challenging |

## 📊 Scoring (0.0 – 1.0)

| Component | Weight | Description |
|-----------|--------|-------------|
| Validity | 40% | Food placed in type-compatible container? |
| Efficiency | 30% | Space utilization vs capacity used |
| Constraints | 20% | Temperature, fragility, flavor isolation |
| Neatness | 10% | All items packed? Nothing dropped? |

## 🚀 Quick Start

### Run locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=your-api-key
export ENV_URL=http://localhost:7860
python inference.py
```

### Docker
```bash
docker build -t tiffin-packer .
docker run -p 7860:7860 tiffin-packer
```

## 🔧 Action Space

```json
{
  "command": "identify | pick | place | pour | observe",
  "target_id": 1
}
```

## 👁️ Observation Space

```json
{
  "scene_description": "Natural language scene state",
  "food_items": [{"id": 1, "name": "rice", "status": "on_table", ...}],
  "containers": [{"id": 1, "type": "sealed_round", "capacity_ml": 300, ...}],
  "held_item": null,
  "vlm_result": {"type": "solid", "fragility": 0.1, ...},
  "available_commands": ["observe", "identify", "pick"],
  "step_feedback": "Successfully picked up rice"
}
```

## 📁 Project Structure

```
tiffen-packer/
├── openenv.yaml          # OpenEnv manifest
├── inference.py           # LLM inference script (OpenAI Client)
├── Dockerfile             # HF Spaces deployment
├── tiffin_packer/         # Core package
│   ├── models.py          # Pydantic Action/Observation/State
│   ├── simulation/
│   │   ├── engine.py      # Logic simulation engine
│   │   └── pybullet_renderer.py  # Physics visualization
│   ├── vlm/
│   │   ├── classifier.py  # VLM food classifier
│   │   └── food_db.json   # 15 Indian food items
│   ├── tasks.py           # Easy/Medium/Hard task configs
│   └── grader.py          # Deterministic scoring
└── server/
    ├── tiffin_environment.py  # OpenEnv Environment
    └── app.py                 # FastAPI server
```

## 🏆 OpenEnv Compliance

- ✅ Typed Pydantic models (Action, Observation, State)
- ✅ `step()` / `reset()` / `state()` API
- ✅ `openenv.yaml` manifest
- ✅ 3 tasks with deterministic graders (0.0–1.0)
- ✅ Dense reward function with partial progress signals
- ✅ Baseline inference script using OpenAI Client
- ✅ Docker deployment for HF Spaces

## 👥 Team

**CtrlAltWin** — Meta PyTorch OpenEnv Hackathon 2026

## 📄 License

MIT
