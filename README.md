---
title: Autonomous Misinfo Crisis Simulator
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# 🛡️ Autonomous Misinformation Crisis Simulator — Advanced Edition

> A research-grade OpenEnv benchmark for evaluating AI moderation systems under real-world constraints.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-orange.svg)]()

---

## 🎯 Motivation

Misinformation on social media is one of the most pressing challenges of the digital age. False health claims, conspiracy theories, and **coordinated disinformation campaigns** spread exponentially — from vaccine hesitancy to public panic to real-world violence.

This environment simulates the **full complexity** of real-world platform moderation:

| Real-World Challenge | How We Simulate It |
|---|---|
| Coordinated campaigns | Multiple posts pushing the same narrative with paraphrasing and synchronized bursts |
| Manipulated media | Multi-modal posts with text-image mismatches (true image + false caption) |
| Platform policy constraints | Limited action budget per step — can't moderate everything |
| Public backlash | Over-censorship causes trust drops and backlash increases |
| Uncertainty | Ground truth is hidden; some posts get verified later (delayed truth) |
| Explainability requirements | Every action must include a justification (graded for quality) |
| Cascading effects | Misinformation posts spawn children, amplifying through networks |

**This is a benchmark Meta could realistically use to evaluate AI moderation systems.**

---

## 🚨 Advanced Features (7 Differentiators)

### 1. 🕸️ Coordinated Adversarial Campaigns
- Multiple posts pushing the same false narrative
- Slight paraphrasing across posts
- Time-synchronized bursts that amplify engagement
- Hidden `campaign_id` links posts; agent sees only noisy signals (`similar_post_count`, `coordinated_activity_flag`)

### 2. 🖼️ Multi-Modal Posts (Text + Image)
- Every post can include an `image_description`
- Cases: true image + false caption, manipulated screenshots, recontextualized photos
- Agent must detect text-image mismatches

### 3. 💰 Policy Constraints (Action Budget)
- Limited non-ignore actions per step:
  - Easy: 5/step | Medium: 3/step | Hard: 2/step
- Exceeding budget → penalty reward
- Forces strategic prioritization

### 4. 💔 Public Reaction Simulation
- `user_trust_score` (0–1): drops from unchecked misinfo OR over-censorship
- `backlash_score` (0–1): rises when true content is censored
- Agent must balance harm reduction vs free expression

### 5. 📝 Explainability Requirement
- Every action **MUST** include a `justification` string
- Graded deterministically for: length, content reference, action consistency, signal awareness
- Missing/poor justification → reward penalty + lower final grade

### 6. ⏰ Delayed Ground Truth
- Truth labels are hidden initially
- Some posts get verified at specific steps (`recently_verified` field)
- Agent must act under uncertainty, then adapt when truth is revealed

### 7. 🎒 Resource / Attention Budget
- Limited posts visible per step (top 10 by engagement)
- Budget constraints force resource allocation decisions

---

## 📦 Project Structure

```
misinfo-crisis-simulator/
├── __init__.py                   # Package exports
├── models.py                     # Pydantic Action/Observation/State models
├── client.py                     # OpenEnv HTTP client  
├── inference.py                  # LLM baseline agent (HF_TOKEN)
├── test_environment.py           # Comprehensive test suite
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml                # Project metadata
├── requirements.txt              # Dependencies
├── Dockerfile                    # Container deployment
├── README.md                     # This file
│
├── server/
│   ├── __init__.py
│   ├── app.py                    # FastAPI HTTP + WebSocket server
│   └── misinfo_environment.py    # Core environment (reset/step/state)
│
├── simulator/
│   ├── __init__.py
│   └── propagation_engine.py     # Propagation, campaigns, trust dynamics
│
├── tasks/
│   ├── __init__.py
│   └── task_definitions.py       # Easy / Medium / Hard with campaigns
│
└── graders/
    ├── __init__.py
    └── misinfo_grader.py         # 7-component deterministic scoring
```

---

## 🎮 Action Space

```json
{
    "action_type": "label | reduce_visibility | add_context | ignore",
    "post_id": "post_id_string",
    "parameters": {
        "label_type": "misleading | false | safe",
        "context_note": "Fact-check note text"
    },
    "justification": "REQUIRED: Detailed reasoning for this action"
}
```

| Action | Effect | Best For |
|--------|--------|----------|
| `label` | Applies label, cuts spread 70-80% | Clear misinformation |
| `reduce_visibility` | Cuts visibility 60%, halves spread | High-virality harmful content |
| `add_context` | Adds fact-check note, reduces spread 30% | Partially true posts |
| `ignore` | No intervention | Verified true content |

---

## 👁️ Observation Space

| Field | Description |
|-------|-------------|
| `active_posts[]` | Posts with content, `image_description`, engagement, and noisy signals |
| `active_posts[].signals.credibility_score` | Noisy credibility estimate (NOT ground truth) |
| `active_posts[].signals.similar_post_count` | Campaign detection signal |
| `active_posts[].signals.coordinated_activity_flag` | Coordinated behavior flag |
| `active_posts[].is_verified` | Whether ground truth has been revealed |
| `active_posts[].verified_label` | Revealed truth (only if verified) |
| `remaining_action_budget` | Non-ignore actions available this step |
| `user_trust_score` | Platform trust level (0–1) |
| `backlash_score` | Public backlash level (0–1) |
| `trend_indicators` | Suspected campaigns, platform health, trending topics |
| `recently_verified` | Post IDs verified this step |

---

## 📋 Tasks (3 Difficulty Levels)

### 🟢 Easy: Obvious Misinformation
- **5 posts** / **10 steps** / Budget: **5**/step
- Clear false claims (bleach, microchips, 5G)
- Multi-modal: images with obvious manipulation
- 1 delayed truth reveal
- **No campaigns**

### 🟡 Medium: Subtle Misinformation
- **8 posts** / **15 steps** / Budget: **3**/step
- Partial truths, misleading framing
- 1 coordinated anti-pharma campaign (2 posts)
- Multi-modal confusion (real stats + fake annotations)
- 2 delayed truth reveals

### 🔴 Hard: Cascade Crisis
- **13+ posts** / **20 steps** / Budget: **2**/step
- **2 coordinated campaigns**: bioweapon conspiracy + water panic
- Campaigns use synchronized bursts + cascading
- Image manipulation (old photos, fabricated documents)
- 4 delayed truth reveals
- Tight budget forces strategic prioritization

---

## 💰 Reward Design

### Dense Stepwise Rewards

| Outcome | Reward |
|---------|--------|
| Early correct detection of misinformation | **+0.3 to +0.5** |
| Campaign post detected | **+0.15** bonus |
| Good justification (quality ≥ 0.6) | **+0.1** |
| Correctly ignoring true content | **+0.1** |
| Over-censorship (labeling true as false) | **-0.4** |
| Missing/poor justification | **-0.1** |
| Budget exceeded | **-0.15** |
| High backlash penalty | **-0.05** |

### Final Grading (7 Components)

| Component | Weight | Measures |
|-----------|--------|----------|
| Detection | 20% | Fraction of misinfo posts identified |
| Timing | 15% | How early interventions occurred |
| Precision | 15% | Avoiding false positives |
| Spread | 15% | Total misinformation minimized |
| Campaign | 15% | Coordinated campaign detection |
| Justification | 10% | Quality of action explanations |
| Trust | 10% | Platform trust – backlash balance |

---

## 🤖 Step-by-Step Procedure to Run the Environment

Follow these steps exactly to run the simulation, connect an agent, and utilize all advanced features.

### Step 1: Install Dependencies
Ensure you have Python 3.10+ installed. Install the requirements:
```bash
cd misinfo-crisis-simulator
pip install -r requirements.txt
```

### Step 2: Start the Environment Server
The simulation runs as a standalone, stateful server. Open a terminal and start it:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
*Wait for the `Application startup complete.` message.*

### Step 3: Run the Health Check & Ensure Smoke Tests Pass
In a new terminal window, verify the server is running and all advanced mechanics (budgeting, truth revelation) are working properly:
```bash
# Basic HTTP health test
curl http://localhost:8000/health

# Run the full 45-point test suite covering all 7 advanced features
python test_environment.py
```
*(You should see `🎉 ALL TESTS PASSED!`)*

### Step 4: Initialize the Simulation Session
Start your agent script and initialize a new episode by sending a `POST /reset` request.

```python
import requests

session_id = "my_agent_run"
response = requests.post("http://localhost:8000/reset", json={
    "task_id": "hard_cascade_crisis", # Use 'easy_obvious_misinfo', 'medium_subtle_misinfo', or 'hard_cascade_crisis'
    "seed": 42,
    "session_id": session_id
})
observation = response.json()["observation"]
```

### Step 5: Read the Advanced Observation (Agent Input)
Your agent must parse the `observation` payload, paying close attention to these critical features:
- **`active_posts[].image_description`**: *[Feature 2]* Text describing the image. Compare this to the post text to detect multi-modal manipulation!
- **Signals**: Monitor `similar_post_count` and `coordinated_activity_flag` to detect hidden networks. *[Feature 1]*
- **Budget**: Read `remaining_action_budget`. If you have `2` remaining, choose the two most dangerous posts before the turn forcibly advances. *[Feature 3]*
- **Public Reaction**: Look at `user_trust_score` and `backlash_score`. *[Feature 4]*
- **Delayed Truth**: Check `recently_verified`. If a post just had its truth revealed, prioritize correcting your stance on it. *[Feature 6]*

### Step 6: Submit a Multi-Action Turn
You submit one `.step()` action at a time. Send actions until your budget is empty OR you send an `ignore`.

```python
# The agent must construct an explainable action
action_payload = {
    "action": {
        "action_type": "label", # label, reduce_visibility, add_context, or ignore
        "post_id": "post_h1",
        "parameters": {"label_type": "false"},
        
        # [Feature 5] MANDATORY: You must explain WHY using the signals!
        "justification": "This post uses a fabricated image_description and coordinated amplification (similar_post_count > 0). It is a dangerous cascade." 
    },
    "session_id": session_id
}
step_response = requests.post("http://localhost:8000/step", json=action_payload)
result = step_response.json()
```
*Note: Time in the simulation ONLY advances once your budget is exhausted (e.g., after 2 acts on Hard) or you intentionally send `action_type: "ignore"`. Observe the `info.time_advanced` flag.*

### Step 7: Analyze the Reward & Info
After each step, examine the results provided in `result['info']`:
- Did you get penalized for exceeding budget?
- What was your `justification_quality`? (Values < 0.5 mean the LLM needs better reasoning).
- Were there any new `truth_revealed` events?

### Step 8: Loop Until Done & Review 7-Component Grader
Repeat steps 5-7. The system tracks cascading posts, delayed reveals, and network drops automatically. When `result["done"] == True`, the episode ends. 

Check `result["info"]["final_grade"]` to view the 7-component research benchmark:
- **`detection_score`**: Did you find the misinfo?
- **`timing_score`**: Were you fast enough?
- **`precision_score`**: Did you avoid censoring truth?
- **`spread_score`**: Did you stop the virality?
- **`campaign_score`**: Did you untangle the network?
- **`justification_score`**: Was your reasoning sound?
- **`trust_score`**: Is the platform still reputable?

---

## 📊 Running the Example LLM Baseline

If you have an OpenAI-compatible endpoint, you can run the baseline reference agent which already implements the steps above:

```bash
export HF_TOKEN=your_openai_api_key
python inference.py --model gpt-4o-mini --seed 42
```

Expected output:
```
Task                         Score   Det   Tim   Pre   Spr   Cam   Jst   Tru
--------------------------------------------------------------------------------
easy_obvious_misinfo         0.750  0.800 0.900 0.850 0.600 1.000 0.500 0.900
medium_subtle_misinfo        0.580  0.600 0.700 0.650 0.400 0.500 0.450 0.700
hard_cascade_crisis          0.420  0.500 0.450 0.550 0.300 0.350 0.400 0.500

Average                      0.583
```

---

## 🏗️ Design Inspiration

| Reference | Feature Inspired |
|-----------|------------------|
| **Calendar Env** | Structured tool actions with clear schema |
| **Reasoning Gym** | Clean scoring with rubric-based evaluation |
| **TB2 Env** | Stepwise interaction loop with session management |
| **REPL Env** | Persistent evolving state across turns |
| **CARLA Env** | Long-horizon consequence tracking, irreversible actions |

---

## 🌐 Hugging Face Deployment

```bash
huggingface-cli upload-folder \
  --repo-id your-username/misinfo-crisis-simulator \
  --repo-type space \
  --folder .
```

Exposed endpoints:
- `/docs` — Interactive Swagger UI
- `/health` — Container monitoring
- `/ws` — WebSocket for persistent sessions

---

## 📜 License

MIT License

## DEVELOPER
Developed by Lokesh and Tejas Vilas Kondhalkar.
