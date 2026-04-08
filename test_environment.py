#!/usr/bin/env python3
"""
End-to-end test suite for the Misinformation Crisis Simulator — Advanced Edition.

Tests all 7 advanced features through the HTTP API:
1. Multi-modal posts (image descriptions)
2. Coordinated adversarial campaigns
3. Policy constraints (action budget)
4. Public reaction (trust / backlash)
5. Explainability (justification grading)
6. Delayed ground truth
7. Resource budget enforcement

Run: python test_environment.py
(Server must be running on localhost:8000)
"""

import json
import sys
import requests

BASE_URL = "http://localhost:8000"
PASS = "✅"
FAIL = "❌"
results = []


def check(name, condition, detail=""):
    results.append((name, condition))
    msg = f"  {PASS if condition else FAIL} {name}"
    if detail and not condition:
        msg += f" — {detail}"
    print(msg)
    return condition


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_health():
    print("\n🏥 Health Check")
    r = requests.get(f"{BASE_URL}/health").json()
    check("Status healthy", r["status"] == "healthy")
    check("Version 2.0.0", r.get("version") == "2.0.0")


def test_tasks():
    print("\n📋 Tasks Endpoint")
    r = requests.get(f"{BASE_URL}/tasks").json()
    tasks = r["tasks"]
    check("Has 3 tasks", len(tasks) == 3)
    hard = [t for t in tasks if t["difficulty"] == "hard"][0]
    check("Hard has campaigns", hard.get("num_campaigns", 0) == 2)
    check("Hard has delayed truth", hard.get("has_delayed_truth") is True)
    check("Hard budget=2", hard.get("action_budget_per_step") == 2)


def test_multimodal_posts():
    print("\n🖼️  Multi-Modal Posts")
    r = requests.post(f"{BASE_URL}/reset", json={
        "task_id": "easy_obvious_misinfo", "seed": 42, "session_id": "test_mm"
    }).json()
    posts = r["observation"]["active_posts"]
    has_images = [p for p in posts if p.get("image_description")]
    check("Posts have image_description", len(has_images) > 0)
    check("Image description is descriptive", len(has_images[0]["image_description"]) > 20)


def test_action_budget():
    print("\n💰 Action Budget Enforcement")
    s = "test_budget"
    r = requests.post(f"{BASE_URL}/reset", json={
        "task_id": "hard_cascade_crisis", "seed": 42, "session_id": s
    }).json()
    budget = r["observation"]["remaining_action_budget"]
    check("Initial budget=2 for hard", budget == 2)

    # Action 1: budget 2→1, time should NOT advance
    r1 = requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "label", "post_id": "post_h1",
            "parameters": {"label_type": "false"},
            "justification": "This post contains misinformation."
        },
        "session_id": s
    }).json()
    check("Action 1: budget drops to 1", r1["info"].get("budget_remaining") == 1)
    check("Action 1: time did NOT advance", r1["info"].get("time_advanced") is False)

    # Action 2: budget 1→0, time SHOULD advance (budget exhausted)
    r2 = requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "label", "post_id": "post_h3",
            "parameters": {"label_type": "false"},
            "justification": "This post contains misinformation."
        },
        "session_id": s
    }).json()
    check("Action 2: budget exhausted (0)", r2["info"].get("budget_remaining") == 0)
    check("Action 2: time advanced (budget exhausted)", r2["info"].get("time_advanced") is True)

    # After time advance, budget resets for the new step
    check("Budget refreshes after advance", r2["observation"]["remaining_action_budget"] == 2)


def test_justification_grading():
    print("\n📝 Justification Quality Grading")
    s = "test_just"
    requests.post(f"{BASE_URL}/reset", json={
        "task_id": "easy_obvious_misinfo", "seed": 42, "session_id": s
    })

    # Good justification
    r1 = requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "label", "post_id": "post_e1",
            "parameters": {"label_type": "false"},
            "justification": (
                "This post makes the false and dangerous claim that drinking bleach "
                "cures diseases. The low credibility score and multiple user reports "
                "confirm misinformation. The image shows a misleading medical symbol "
                "on a bleach bottle which is manipulated."
            )
        },
        "session_id": s
    }).json()
    good_quality = r1["info"].get("justification_quality", 0)
    check("Good justification scores high", good_quality >= 0.5,
          f"quality={good_quality}")

    # Empty justification
    r2 = requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "label", "post_id": "post_e3",
            "parameters": {"label_type": "false"},
            "justification": ""
        },
        "session_id": s
    }).json()
    bad_quality = r2["info"].get("justification_quality", 1)
    check("Empty justification scores 0", bad_quality == 0.0,
          f"quality={bad_quality}")

    check("Good > Bad justification", good_quality > bad_quality)


def test_trust_backlash():
    print("\n💔 Public Trust & Backlash")
    s = "test_trust"
    r = requests.post(f"{BASE_URL}/reset", json={
        "task_id": "easy_obvious_misinfo", "seed": 42, "session_id": s
    }).json()
    initial_trust = r["observation"]["user_trust_score"]
    initial_backlash = r["observation"]["backlash_score"]
    check("Initial trust is 1.0", initial_trust == 1.0)
    check("Initial backlash is 0.0", initial_backlash == 0.0)

    # Over-censor: label true content as false (should increase backlash)
    requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "label", "post_id": "post_e2",
            "parameters": {"label_type": "false"},
            "justification": "I think this is false."
        },
        "session_id": s
    })
    # Take another step to let trust/backlash update
    r2 = requests.post(f"{BASE_URL}/step", json={
        "action": {"action_type": "ignore", "post_id": "post_e1",
                   "justification": "Waiting."},
        "session_id": s
    }).json()

    trust_after = r2["observation"]["user_trust_score"]
    backlash_after = r2["observation"]["backlash_score"]
    check("Trust drops after over-censorship", trust_after < initial_trust,
          f"was={initial_trust}, now={trust_after}")
    check("Backlash rises after over-censorship", backlash_after > initial_backlash,
          f"was={initial_backlash}, now={backlash_after}")


def test_delayed_truth():
    print("\n⏰ Delayed Ground Truth Revelation")
    s = "test_truth"
    requests.post(f"{BASE_URL}/reset", json={
        "task_id": "easy_obvious_misinfo", "seed": 42, "session_id": s
    })

    # Initial: post_e1 should NOT be verified
    r = requests.post(f"{BASE_URL}/step", json={
        "action": {"action_type": "ignore", "post_id": "post_e1",
                   "justification": "Waiting for verification."},
        "session_id": s
    }).json()

    post_e1 = [p for p in r["observation"]["active_posts"] if p["post_id"] == "post_e1"]
    if post_e1:
        check("post_e1 NOT verified initially", post_e1[0]["is_verified"] is False)

    # Advance to step 4 where truth is revealed
    truth_revealed = False
    for i in range(5):
        r = requests.post(f"{BASE_URL}/step", json={
            "action": {"action_type": "ignore", "post_id": "post_e1",
                       "justification": "Observing."},
            "session_id": s
        }).json()
        if r.get("info", {}).get("truth_revealed"):
            truth_revealed = True
            check("Truth revealed event fired", True)
            break

    if not truth_revealed:
        check("Truth revealed event fired", False, "Never fired")

    # Check if post is now verified in observation
    post_e1 = [p for p in r["observation"]["active_posts"] if p["post_id"] == "post_e1"]
    if post_e1:
        check("post_e1 verified after reveal", post_e1[0]["is_verified"] is True)
        check("Verified label is 'false'", post_e1[0].get("verified_label") == "false")


def test_campaigns():
    print("\n🎯 Coordinated Campaign Detection")
    s = "test_campaign"
    r = requests.post(f"{BASE_URL}/reset", json={
        "task_id": "hard_cascade_crisis", "seed": 42, "session_id": s
    }).json()

    # Check campaign signals on post_h1 (bioweapon campaign)
    post_h1 = [p for p in r["observation"]["active_posts"] if p["post_id"] == "post_h1"]
    if post_h1:
        signals = post_h1[0]["signals"]
        check("Campaign post has similar_post_count > 0",
              signals.get("similar_post_count", 0) > 0)

    # Run episode and check campaign detection in grade
    done = False
    step_data = r
    while not done:
        posts = step_data["observation"]["active_posts"]
        if not posts:
            break
        target = posts[0]
        step_data = requests.post(f"{BASE_URL}/step", json={
            "action": {
                "action_type": "label", "post_id": target["post_id"],
                "parameters": {"label_type": "false"},
                "justification": f"Flagging {target['post_id']} as part of coordinated campaign."
            },
            "session_id": s
        }).json()
        done = step_data["done"]

    if step_data.get("done"):
        grade = step_data["info"].get("final_grade", {})
        check("Campaign score in grade", "campaign_score" in grade)
        check("Campaigns total = 2", grade.get("campaigns_total") == 2)
        print(f"  📊 Campaign score: {grade.get('campaign_score', 0):.4f}")
        print(f"     Campaigns detected: {grade.get('campaigns_detected', 0)}/{grade.get('campaigns_total', 0)}")


def test_state_has_campaigns():
    print("\n🔍 State Endpoint (Campaigns + Ground Truth)")
    s = "test_state_adv"
    requests.post(f"{BASE_URL}/reset", json={
        "task_id": "hard_cascade_crisis", "seed": 42, "session_id": s
    })
    state = requests.get(f"{BASE_URL}/state", params={"session_id": s}).json()
    check("State has campaigns", "campaigns" in state and len(state["campaigns"]) > 0)

    bioweapon = state["campaigns"].get("campaign_bioweapon", {})
    check("Bioweapon campaign has narrative", len(bioweapon.get("narrative", "")) > 10)
    check("Bioweapon campaign has post_ids", len(bioweapon.get("post_ids", [])) > 0)
    check("State has truth_reveal_schedule", len(state.get("truth_reveal_schedule", {})) > 0)
    check("State has user_trust", "user_trust" in state)
    check("State has backlash", "backlash" in state)


def test_trend_indicators():
    print("\n📈 Trend Indicators")
    s = "test_trends"
    r = requests.post(f"{BASE_URL}/reset", json={
        "task_id": "hard_cascade_crisis", "seed": 42, "session_id": s
    }).json()
    trends = r["observation"].get("trend_indicators", {})
    check("Has trend_indicators", bool(trends))
    check("Has platform_health_score", "platform_health_score" in trends)
    check("Has suspected_campaigns", "suspected_campaigns" in trends)
    check("Platform health in [0,1]", 0 <= trends.get("platform_health_score", -1) <= 1)


def test_overcensorship():
    print("\n⚖️  Over-Censorship Penalty")
    s = "test_censor"
    requests.post(f"{BASE_URL}/reset", json={
        "task_id": "easy_obvious_misinfo", "seed": 42, "session_id": s
    })
    r = requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "label", "post_id": "post_e2",
            "parameters": {"label_type": "false"},
            "justification": "I think this is false."
        },
        "session_id": s
    }).json()
    check("Labeling true as false → negative", r["reward"] < 0, f"reward={r['reward']}")


def test_determinism():
    print("\n🎲 Determinism (same seed = same results)")
    rewards = [[], []]
    for run in range(2):
        sid = f"det_{run}"
        requests.post(f"{BASE_URL}/reset", json={
            "task_id": "easy_obvious_misinfo", "seed": 42, "session_id": sid
        })
        for pid in ["post_e1", "post_e3"]:
            r = requests.post(f"{BASE_URL}/step", json={
                "action": {
                    "action_type": "label", "post_id": pid,
                    "parameters": {"label_type": "false"},
                    "justification": "Misinformation detected based on credibility signals."
                },
                "session_id": sid
            }).json()
            rewards[run].append(r["reward"])

    check("Deterministic rewards", rewards[0] == rewards[1],
          f"run1={rewards[0]}, run2={rewards[1]}")


def test_full_hard_episode():
    print("\n🔴 Full Hard Episode (Integration)")
    s = "test_full_hard"
    r = requests.post(f"{BASE_URL}/reset", json={
        "task_id": "hard_cascade_crisis", "seed": 42, "session_id": s
    }).json()
    obs = r["observation"]
    check("Hard: trust=1.0 initially", obs["user_trust_score"] == 1.0)
    check("Hard: budget=2", obs["remaining_action_budget"] == 2)

    done = False
    step_data = None
    while not done:
        posts = obs["active_posts"] if step_data is None else step_data["observation"]["active_posts"]
        if not posts:
            break

        target = min(posts, key=lambda p: p["signals"]["credibility_score"])
        if target["signals"]["credibility_score"] < 0.5:
            action = {
                "action_type": "label", "post_id": target["post_id"],
                "parameters": {"label_type": "false"},
                "justification": f"Low credibility ({target['signals']['credibility_score']}), likely misinformation."
            }
        else:
            action = {
                "action_type": "ignore", "post_id": target["post_id"],
                "justification": f"High credibility ({target['signals']['credibility_score']}), appears legitimate."
            }

        step_data = requests.post(f"{BASE_URL}/step", json={
            "action": action, "session_id": s
        }).json()
        done = step_data["done"]

    check("Episode completes", step_data["done"] is True)
    grade = step_data["info"].get("final_grade", {})
    check("Has all 7 scoring components", all(
        k in grade for k in [
            "detection_score", "timing_score", "precision_score",
            "spread_score", "campaign_score", "justification_score", "trust_score"
        ]
    ))
    check("Overall score in [0,1]", 0 <= grade.get("overall_score", -1) <= 1)

    print(f"\n  📊 Final Grade: {grade.get('overall_score', 0):.4f}")
    for k in ["detection_score", "timing_score", "precision_score",
              "spread_score", "campaign_score", "justification_score", "trust_score"]:
        print(f"     {k}: {grade.get(k, 0):.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("🛡️  Misinformation Crisis Simulator — Advanced Test Suite")
    print("=" * 65)

    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.ConnectionError:
        print(f"\n{FAIL} Server not running at {BASE_URL}")
        sys.exit(1)

    test_health()
    test_tasks()
    test_multimodal_posts()
    test_action_budget()
    test_justification_grading()
    test_trust_backlash()
    test_delayed_truth()
    test_campaigns()
    test_state_has_campaigns()
    test_trend_indicators()
    test_overcensorship()
    test_determinism()
    test_full_hard_episode()

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'=' * 65}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        for name, ok in results:
            if not ok:
                print(f"  {FAIL} {name}")
    print(f"{'=' * 65}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
