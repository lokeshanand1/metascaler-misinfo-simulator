"""
FastAPI application for the Misinformation Crisis Simulator — Advanced Edition.

OpenEnv-compatible HTTP + WebSocket endpoints:
- POST /reset — start a new episode
- POST /step — take an action
- GET /state — get full internal state
- GET /health — health check
- GET /tasks — list available tasks
- WS /ws — WebSocket for persistent sessions
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import (
    MisinfoCrisisAction,
    MisinfoCrisisObservation,
    MisinfoCrisisState,
    ResetResult,
    StepResult,
)
from server.misinfo_environment import MisinfoCrisisEnvironment

logger = logging.getLogger(__name__)

# ── Session Management ─────────────────────────────────────────────────────────

_sessions: Dict[str, MisinfoCrisisEnvironment] = {}


def get_or_create_env(session_id: Optional[str] = None) -> MisinfoCrisisEnvironment:
    """Get an environment for the given session, or create a new one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    env = MisinfoCrisisEnvironment()
    if session_id:
        _sessions[session_id] = env
    return env


# ── Request / Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_obvious_misinfo"
    seed: int = 42
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action: MisinfoCrisisAction
    session_id: Optional[str] = None


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Misinformation Crisis Simulator — Advanced Edition started.")
    yield
    _sessions.clear()
    logger.info("Server shutting down, sessions cleared.")


app = FastAPI(
    title="Autonomous Misinformation Crisis Simulator — Advanced Edition",
    description=(
        "An OpenEnv-compatible environment simulating social media misinformation dynamics. "
        "Features coordinated adversarial campaigns, multi-modal posts, policy constraints, "
        "public reaction simulation, explainability requirements, delayed ground truth, "
        "and resource budget management."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "misinfo-crisis-simulator", "version": "2.0.0"}


# ── Reset ──────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResult)
async def reset(request: ResetRequest):
    """
    Reset the environment for a new episode.

    Available task_ids:
    - easy_obvious_misinfo: Obvious misinformation, clear signals, generous budget
    - medium_subtle_misinfo: Subtle misinfo, multi-modal confusion, small campaign
    - hard_cascade_crisis: Coordinated campaigns, cascading, tight budget, delayed truth
    """
    try:
        session_id = request.session_id or "default"
        env = get_or_create_env(session_id)
        result = env.reset(task_id=request.task_id, seed=request.seed)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Reset error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Step ───────────────────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResult)
async def step(request: StepRequest):
    """
    Execute one step. The agent submits a moderation action with justification.

    The environment:
    1. Validates the action (budget, post existence)
    2. Applies it and evaluates justification quality
    3. Advances simulation (propagation, campaigns, delayed truth)
    4. Returns observation, reward, done, and info
    """
    try:
        session_id = request.session_id or "default"
        env = get_or_create_env(session_id)
        result = env.step(request.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── State ──────────────────────────────────────────────────────────────────────

@app.get("/state")
async def get_state(session_id: Optional[str] = None):
    """Return the full internal state including ground truth, campaigns, trust."""
    try:
        sid = session_id or "default"
        env = get_or_create_env(sid)
        state = env.state()
        return state.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"State error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Tasks ──────────────────────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks():
    """List available task configurations."""
    from tasks.task_definitions import TASK_REGISTRY, get_task

    tasks = []
    for task_id in TASK_REGISTRY:
        config = get_task(task_id)
        tasks.append({
            "task_id": config.task_id,
            "difficulty": config.difficulty,
            "description": config.description,
            "max_steps": config.max_steps,
            "action_budget_per_step": config.action_budget_per_step,
            "num_campaigns": len(config.campaigns),
            "has_delayed_truth": bool(config.truth_reveal_schedule),
        })
    return {"tasks": tasks}


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for persistent sessions.

    Protocol:
    - Client sends JSON with "type": "reset" | "step" | "state"
    - Server responds with the corresponding result
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = get_or_create_env(session_id)
    logger.info(f"WebSocket session {session_id} connected")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")

            try:
                if msg_type == "reset":
                    task_id = message.get("task_id", "easy_obvious_misinfo")
                    seed = message.get("seed", 42)
                    result = env.reset(task_id=task_id, seed=seed)
                    await websocket.send_json({
                        "type": "reset_result",
                        "session_id": session_id,
                        "data": result.model_dump(),
                    })
                elif msg_type == "step":
                    action = MisinfoCrisisAction(**message.get("action", {}))
                    result = env.step(action)
                    await websocket.send_json({
                        "type": "step_result",
                        "session_id": session_id,
                        "data": result.model_dump(),
                    })
                elif msg_type == "state":
                    state = env.state()
                    await websocket.send_json({
                        "type": "state_result",
                        "session_id": session_id,
                        "data": state.model_dump(),
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket session {session_id} disconnected")
        _sessions.pop(session_id, None)
    except Exception:
        logger.error(f"WebSocket error: {traceback.format_exc()}")
        _sessions.pop(session_id, None)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
