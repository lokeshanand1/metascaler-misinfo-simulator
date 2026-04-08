"""
Client for the Misinformation Crisis Simulator — Advanced Edition.

Python interface to the OpenEnv HTTP server.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from models import (
    MisinfoCrisisAction,
    ResetResult,
    StepResult,
)


class MisinfoCrisisEnv:
    """
    Client for the Misinformation Crisis Simulator.

    Usage:
        env = MisinfoCrisisEnv(base_url="http://localhost:8000")
        result = env.reset(task_id="easy_obvious_misinfo", seed=42)

        result = env.step(MisinfoCrisisAction(
            action_type="label", post_id="post_e1",
            parameters={"label_type": "false"},
            justification="Post contains debunked health misinformation."
        ))
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        session_id: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    @classmethod
    def from_docker_image(
        cls,
        image: str = "misinfo-crisis-sim:latest",
        port: int = 8000,
        timeout: int = 30,
    ) -> "MisinfoCrisisEnv":
        """Start a Docker container and connect."""
        import subprocess

        container_id = (
            subprocess.check_output([
                "docker", "run", "-d", "--rm", "-p", f"{port}:{port}", image,
            ])
            .decode()
            .strip()
        )
        env = cls(base_url=f"http://localhost:{port}")
        env._container_id = container_id

        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{env.base_url}/health", timeout=2)
                if resp.status_code == 200:
                    return env
            except requests.ConnectionError:
                time.sleep(0.5)

        raise TimeoutError(f"Server did not start within {timeout}s")

    def reset(self, task_id: str = "easy_obvious_misinfo", seed: int = 42) -> ResetResult:
        payload = {"task_id": task_id, "seed": seed}
        if self.session_id:
            payload["session_id"] = self.session_id
        resp = requests.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return ResetResult(**resp.json())

    def step(self, action: MisinfoCrisisAction) -> StepResult:
        payload = {"action": action.model_dump()}
        if self.session_id:
            payload["session_id"] = self.session_id
        resp = requests.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> Dict[str, Any]:
        params = {}
        if self.session_id:
            params["session_id"] = self.session_id
        resp = requests.get(f"{self.base_url}/state", params=params)
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> List[Dict]:
        resp = requests.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()["tasks"]

    def health(self) -> Dict:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        cid = getattr(self, "_container_id", None)
        if cid:
            import subprocess
            subprocess.run(["docker", "stop", cid], capture_output=True)
            self._container_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
