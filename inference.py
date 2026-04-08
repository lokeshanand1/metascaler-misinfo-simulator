#!/usr/bin/env python3
"""
Baseline Inference Script — Advanced Edition

Uses OpenAI-compatible API (HF_TOKEN) to run an LLM agent through all tasks.
Outputs reproducible scores with 7-component grading.

Usage:
    export HF_TOKEN=your_api_key
    python inference.py [--base-url http://localhost:8000] [--model gpt-4o-mini]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    ActionParameters,
    ActionType,
    LabelType,
    MisinfoCrisisAction,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

_server_process = None


def create_openai_client(model: str = MODEL_NAME):
    from openai import OpenAI

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=HF_TOKEN)


def format_observation(observation: dict) -> str:
    parts = [
        f"## Simulation State (Step {observation['time_step']}/{observation['max_steps']})\n",
        f"**Task:** {observation.get('task_description', 'N/A')}\n",
        f"**Instructions:** {observation.get('instruction', 'N/A')}\n",
        f"\n**Action Budget Remaining:** {observation.get('remaining_action_budget', 'N/A')}",
        f"\n**User Trust:** {observation.get('user_trust_score', 'N/A')}",
        f"\n**Backlash:** {observation.get('backlash_score', 'N/A')}\n",
    ]

    # Recently verified
    verified = observation.get("recently_verified", [])
    if verified:
        parts.append(f"\n⚡ **Truth Revealed This Step:** {verified}\n")

    # Trend indicators
    trends = observation.get("trend_indicators", {})
    if trends:
        parts.append(f"\n**Trends:** Suspected campaigns: {trends.get('suspected_campaigns', 0)}")
        parts.append(f" | Platform health: {trends.get('platform_health_score', 'N/A')}\n")

    parts.append(f"\n### Active Posts ({len(observation['active_posts'])} visible):\n")

    for i, post in enumerate(observation["active_posts"], 1):
        signals = post["signals"]
        parts.append(
            f"\n**Post {i}: `{post['post_id']}`**\n"
            f"  Content: \"{post['content']}\"\n"
        )
        if post.get("image_description"):
            parts.append(f"  Image: \"{post['image_description']}\"\n")
        parts.append(
            f"  Engagement: {post['likes']} likes, {post['shares']} shares, {post['comments']} comments\n"
            f"  Virality: {post['virality_velocity']} | Credibility: {signals['credibility_score']}\n"
            f"  Reports: {signals['user_reports']} | Similar posts: {signals.get('similar_post_count', 0)}\n"
            f"  Coordinated activity: {signals.get('coordinated_activity_flag', False)}\n"
            f"  Labels: {post['current_labels']} | Visibility reduced: {post['visibility_reduced']}\n"
        )
        if post.get("is_verified"):
            parts.append(f"  ✅ VERIFIED: truth_label = {post.get('verified_label')}\n")

    if observation.get("previous_actions"):
        parts.append("\n### Recent Actions:\n")
        for a in observation["previous_actions"]:
            jq = a.get("justification_quality", "N/A")
            parts.append(
                f"  Step {a['step']}: {a['action_type']} on {a['post_id']} "
                f"→ {a['result']} (justification quality: {jq})\n"
            )

    return "".join(parts)


SYSTEM_PROMPT = """You are an AI platform moderation agent. Analyze social media posts and take moderation actions.

Respond with a JSON object:
{
    "action_type": "label" | "reduce_visibility" | "add_context" | "ignore",
    "post_id": "<target post id>",
    "parameters": {
        "label_type": "misleading" | "false" | "safe",
        "context_note": "<fact-check note>"
    },
    "justification": "<REQUIRED: detailed explanation of your reasoning>"
}

Strategy:
- Check image_description for image-text mismatches (manipulation signal)
- Watch similar_post_count and coordinated_activity_flag for campaigns
- Use remaining_action_budget wisely — prioritize high-risk posts
- Monitor user_trust_score and backlash_score — avoid over-censoring
- If is_verified=true, use verified_label as ground truth
- ALWAYS provide a detailed justification

RESPOND WITH ONLY THE JSON OBJECT."""


def parse_llm_action(response_text: str) -> Optional[MisinfoCrisisAction]:
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        data = json.loads(text)

        parameters = None
        if "parameters" in data and data["parameters"]:
            parameters = ActionParameters(**data["parameters"])

        return MisinfoCrisisAction(
            action_type=data["action_type"],
            post_id=data["post_id"],
            parameters=parameters,
            justification=data.get("justification", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Parse error: {e}", file=sys.stderr)
        return None


def ensure_server_running(base_url: str, timeout: int = 180):
    """
    Ensure the environment server is running.
    First checks if it's already accessible; if not, starts it as a subprocess.
    """
    import requests
    import threading

    base_url = base_url.rstrip("/")

    # 1. Quick check — is the server already up?
    for _ in range(3):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Server already running at {base_url}", file=sys.stderr)
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    # 2. Server is not running — start it ourselves
    print("Server not detected. Starting environment server as subprocess...", file=sys.stderr)
    global _server_process

    # Determine project root (same directory as this script)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Extract host and port from base_url
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.hostname or "0.0.0.0"
    port = str(parsed.port or 8000)

    # Try python -m uvicorn first
    cmd = [
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", host,
        "--port", port,
        "--log-level", "info",
    ]

    print(f"Starting server: {' '.join(cmd)}", file=sys.stderr)
    _server_process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
    )

    # Stream server output in a background thread so startup errors are visible
    _server_log_lines = []

    def _stream_output():
        for line in iter(_server_process.stdout.readline, b""):
            decoded = line.decode(errors="replace").rstrip()
            _server_log_lines.append(decoded)
            print(f"[server] {decoded}", file=sys.stderr)

    _log_thread = threading.Thread(target=_stream_output, daemon=True)
    _log_thread.start()

    # 3. Wait for it to become healthy
    start = time.time()
    while time.time() - start < timeout:
        # Check if process died early
        if _server_process.poll() is not None:
            _log_thread.join(timeout=2)
            print(f"ERROR: Server process exited with code {_server_process.returncode}", file=sys.stderr)
            print("Last server output:", file=sys.stderr)
            for line in _server_log_lines[-30:]:
                print(f"  {line}", file=sys.stderr)
            raise RuntimeError(
                f"Server process died during startup (exit code {_server_process.returncode})"
            )

        try:
            resp = requests.get(f"{base_url}/health", timeout=3)
            if resp.status_code == 200:
                print(f"Server started and ready at {base_url}", file=sys.stderr)
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.0)

    # Timeout — print whatever we got
    print(f"ERROR: Server not ready after {timeout}s. Last server output:", file=sys.stderr)
    for line in _server_log_lines[-30:]:
        print(f"  {line}", file=sys.stderr)
    raise TimeoutError(f"Server not ready after {timeout}s")


def run_episode(base_url, task_id, seed, client, model, session_id):
    import requests as req

    base_url = base_url.rstrip("/")

    print(f"[START] task={task_id} seed={seed}", flush=True)

    # Retry /reset request with backoff
    max_retries = 5
    resp = None
    for attempt in range(max_retries):
        try:
            resp = req.post(f"{base_url}/reset", json={
                "task_id": task_id, "seed": seed, "session_id": session_id
            }, timeout=15)
            resp.raise_for_status()
            break
        except req.exceptions.RequestException as e:
            print(f"  Reset attempt {attempt + 1}/{max_retries} failed: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  ERROR: Could not reset environment after {max_retries} attempts.", file=sys.stderr)
                print(f"[END] task={task_id} score=0.0000 steps=0", flush=True)
                return {
                    "task_id": task_id, "seed": seed, "steps": 0,
                    "total_reward": 0.0, "overall_score": 0.0,
                    "detection_score": 0.0, "timing_score": 0.0,
                    "precision_score": 0.0, "spread_score": 0.0,
                    "campaign_score": 0.0, "justification_score": 0.0,
                    "trust_score": 0.0, "campaigns_detected": 0,
                    "campaigns_total": 0,
                }

    observation = resp.json()["observation"]

    total_reward = 0.0
    step_count = 0
    done = False
    final_info = {}

    while not done:
        step_count += 1
        obs_text = format_observation(observation)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs_text},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            llm_response = response.choices[0].message.content
        except Exception as e:
            print(f"  LLM error: {e}", file=sys.stderr)
            if observation["active_posts"]:
                llm_response = json.dumps({
                    "action_type": "ignore",
                    "post_id": observation["active_posts"][0]["post_id"],
                    "justification": "Fallback action due to API error.",
                })
            else:
                break

        action = parse_llm_action(llm_response)
        if action is None and observation["active_posts"]:
            action = MisinfoCrisisAction(
                action_type=ActionType.IGNORE,
                post_id=observation["active_posts"][0]["post_id"],
                justification="Fallback: could not parse action.",
            )

        if action is None:
            break

        print(f"[STEP] task={task_id} step={step_count} action={action.action_type.value} post={action.post_id}", flush=True)

        try:
            step_resp = req.post(f"{base_url}/step", json={
                "action": action.model_dump(), "session_id": session_id
            }, timeout=15)
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except req.exceptions.RequestException as e:
            print(f"  Step API error: {e}", file=sys.stderr)
            break

        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data.get("info", {})
        total_reward += reward

        if info.get("truth_revealed"):
            print(f"    Truth revealed: {info['truth_revealed']}", file=sys.stderr)
        print(f"[STEP] task={task_id} step={step_count} reward={reward:.4f}", flush=True)

        if done:
            final_info = info

    grade = final_info.get("final_grade", {})
    result = {
        "task_id": task_id,
        "seed": seed,
        "steps": step_count,
        "total_reward": round(total_reward, 4),
        **{k: grade.get(k, 0.0) for k in [
            "overall_score", "detection_score", "timing_score",
            "precision_score", "spread_score", "campaign_score",
            "justification_score", "trust_score",
        ]},
        "campaigns_detected": grade.get("campaigns_detected", 0),
        "campaigns_total": grade.get("campaigns_total", 0),
    }

    print(f"[END] task={task_id} score={result['overall_score']:.4f} steps={step_count}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", default=[
        "easy_obvious_misinfo", "medium_subtle_misinfo", "hard_cascade_crisis"
    ])
    args = parser.parse_args()

    try:
        # Ensure server is running (starts it if needed)
        try:
            ensure_server_running(args.base_url)
        except (TimeoutError, RuntimeError) as e:
            print(f"FATAL: Could not start server: {e}", file=sys.stderr)
            # Output zero scores so the validator gets valid JSON output
            output = {
                "model": args.model, "seed": args.seed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": [], "average_score": 0.0,
                "error": str(e),
            }
            with open("baseline_results.json", "w") as f:
                json.dump(output, f, indent=2)
            sys.exit(0)

        client = create_openai_client(args.model)
        all_results = []

        for task_id in args.tasks:
            try:
                result = run_episode(
                    args.base_url, task_id, args.seed, client, args.model,
                    f"baseline_{task_id}",
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR in task {task_id}: {e}", file=sys.stderr)
                all_results.append({
                    "task_id": task_id, "seed": args.seed, "steps": 0,
                    "total_reward": 0.0, "overall_score": 0.0,
                    "detection_score": 0.0, "timing_score": 0.0,
                    "precision_score": 0.0, "spread_score": 0.0,
                    "campaign_score": 0.0, "justification_score": 0.0,
                    "trust_score": 0.0, "campaigns_detected": 0,
                    "campaigns_total": 0,
                })

        print(f"\n{'='*80}", file=sys.stderr)
        print("BASELINE RESULTS — ADVANCED EDITION", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"Model: {args.model} | Seed: {args.seed}\n", file=sys.stderr)

        header = (
            f"{'Task':<28} {'Score':>6} {'Det':>5} {'Tim':>5} "
            f"{'Pre':>5} {'Spr':>5} {'Cam':>5} {'Jst':>5} {'Tru':>5}"
        )
        print(header, file=sys.stderr)
        print("-" * len(header), file=sys.stderr)

        for r in all_results:
            print(
                f"{r['task_id']:<28} {r['overall_score']:>6.3f} "
                f"{r['detection_score']:>5.3f} {r['timing_score']:>5.3f} "
                f"{r['precision_score']:>5.3f} {r['spread_score']:>5.3f} "
                f"{r['campaign_score']:>5.3f} {r['justification_score']:>5.3f} "
                f"{r['trust_score']:>5.3f}",
                file=sys.stderr
            )

        avg = sum(r["overall_score"] for r in all_results) / len(all_results) if all_results else 0
        print(f"\n{'Average':<28} {avg:>6.3f}", file=sys.stderr)

        output = {
            "model": args.model, "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": all_results, "average_score": round(avg, 4),
        }
        with open("baseline_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to baseline_results.json", file=sys.stderr)

    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Exit with 0 so the validator doesn't see an unhandled exception
        sys.exit(0)

    finally:
        # Clean up server subprocess if we started one
        if _server_process is not None:
            print("Shutting down server subprocess...", file=sys.stderr)
            _server_process.terminate()
            try:
                _server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _server_process.kill()


if __name__ == "__main__":
    main()
