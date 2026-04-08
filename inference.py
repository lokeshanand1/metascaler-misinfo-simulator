#!/usr/bin/env python3
"""
Baseline Inference Script — Advanced Edition

Uses OpenAI-compatible API (HF_TOKEN) to run an LLM agent through all tasks.
Outputs reproducible scores with 7-component grading.

Usage:
    export HF_TOKEN=your_api_key
    python baseline_inference.py [--base-url http://localhost:8000] [--model gpt-4o-mini]
"""

from __future__ import annotations

import argparse
import json
import os
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

def create_openai_client(model: str = MODEL_NAME):
    from openai import OpenAI

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.")
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
        print(f"  ⚠ Parse error: {e}")
        return None


def run_episode(base_url, task_id, seed, client, model, session_id):
    import requests as req

    print(f"\n{'='*60}")
    print(f"START Task: {task_id} (seed={seed})")
    print(f"{'='*60}")

    # Retry /reset request with backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = req.post(f"{base_url.rstrip('/')}/reset", json={
                "task_id": task_id, "seed": seed, "session_id": session_id
            }, timeout=10)
            resp.raise_for_status()
            break
        except req.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: Server not ready, waiting...")
                time.sleep(1)
            else:
                raise
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
            print(f"  ⚠ LLM error: {e}")
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

        print(f"STEP {step_count}: {action.action_type.value} on {action.post_id}")

        try:
            step_resp = req.post(f"{base_url.rstrip('/')}/step", json={
                "action": action.model_dump(), "session_id": session_id
            }, timeout=10)
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except req.exceptions.RequestException as e:
            print(f"  ⚠ Step API error: {e}")
            break

        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data.get("info", {})
        total_reward += reward

        if info.get("truth_revealed"):
            print(f"    ⚡ Truth revealed: {info['truth_revealed']}")
        print(f"    → reward={reward:.4f}")

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

    print(f"\nEND Final Score: {result['overall_score']:.4f}")
    return result


def wait_for_server(base_url: str, timeout: int = 60):
    """Wait for server to be ready."""
    import requests
    import time
    
    base_url = base_url.rstrip("/")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"Server ready at {base_url}")
                return
            print(f"Waiting for server... Status: {resp.status_code}")
        except requests.exceptions.RequestException as e:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server not ready after {timeout}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", default=[
        "easy_obvious_misinfo", "medium_subtle_misinfo", "hard_cascade_crisis"
    ])
    args = parser.parse_args()

    # Wait for server to be ready
    wait_for_server(args.base_url)

    client = create_openai_client(args.model)
    all_results = []

    for task_id in args.tasks:
        result = run_episode(
            args.base_url, task_id, args.seed, client, args.model,
            f"baseline_{task_id}",
        )
        all_results.append(result)

    print(f"\n{'='*80}")
    print("BASELINE RESULTS — ADVANCED EDITION")
    print(f"{'='*80}")
    print(f"Model: {args.model} | Seed: {args.seed}\n")

    header = (
        f"{'Task':<28} {'Score':>6} {'Det':>5} {'Tim':>5} "
        f"{'Pre':>5} {'Spr':>5} {'Cam':>5} {'Jst':>5} {'Tru':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['task_id']:<28} {r['overall_score']:>6.3f} "
            f"{r['detection_score']:>5.3f} {r['timing_score']:>5.3f} "
            f"{r['precision_score']:>5.3f} {r['spread_score']:>5.3f} "
            f"{r['campaign_score']:>5.3f} {r['justification_score']:>5.3f} "
            f"{r['trust_score']:>5.3f}"
        )

    avg = sum(r["overall_score"] for r in all_results) / len(all_results) if all_results else 0
    print(f"\n{'Average':<28} {avg:>6.3f}")

    output = {
        "model": args.model, "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": all_results, "average_score": round(avg, 4),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to baseline_results.json")


if __name__ == "__main__":
    main()
