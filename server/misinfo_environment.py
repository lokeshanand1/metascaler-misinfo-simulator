"""
Misinformation Crisis Environment — Advanced Edition

Core environment implementing the OpenEnv reset() / step() / state() contract.

Advanced features:
- Coordinated adversarial campaign tracking
- Multi-modal post handling (text + image)
- Policy constraints (action budget per step)
- Public reaction simulation (trust / backlash)
- Explainability requirement (justification grading)
- Delayed ground truth revelation
- Resource / attention budget enforcement
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is on the path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from graders.misinfo_grader import MisinfoGrader
from models import (
    ActionParameters,
    ActionSummary,
    ActionType,
    Campaign,
    FullPost,
    InterventionRecord,
    LabelType,
    MisinfoCrisisAction,
    MisinfoCrisisObservation,
    MisinfoCrisisState,
    ObservablePost,
    PostSignals,
    ResetResult,
    StepResult,
    TrendIndicator,
    TruthLabel,
)
from simulator.propagation_engine import PropagationEngine
from tasks.task_definitions import TaskConfig, get_task


class MisinfoCrisisEnvironment:
    """
    Core environment implementing the OpenEnv interface.

    Lifecycle:
        1. reset(task_id, seed) → initial observation
        2. step(action) → (observation, reward, done, info)  [repeat]
        3. state() → full internal state (ground truth)
    """

    def __init__(self):
        self._state: Optional[MisinfoCrisisState] = None
        self._engine: Optional[PropagationEngine] = None
        self._grader: MisinfoGrader = MisinfoGrader()
        self._task_config: Optional[TaskConfig] = None
        self._just_revealed: List[str] = []  # Posts revealed this step

    def reset(
        self,
        task_id: str = "easy_obvious_misinfo",
        seed: int = 42,
    ) -> ResetResult:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Which task to load.
            seed: Random seed for deterministic simulation.

        Returns:
            ResetResult with the initial observation.
        """
        self._task_config = get_task(task_id, seed)
        self._engine = PropagationEngine(seed=seed)
        self._just_revealed = []

        # Initialize state
        self._state = MisinfoCrisisState(
            task_id=task_id,
            difficulty=self._task_config.difficulty,
            time_step=0,
            max_steps=self._task_config.max_steps,
            done=False,
            seed=seed,
            action_budget_per_step=self._task_config.action_budget_per_step,
        )

        # Load initial posts
        for post in self._task_config.initial_posts:
            self._state.posts[post.post_id] = copy.deepcopy(post)

        # Load scheduled posts (stored but not active yet)
        for step, posts in self._task_config.scheduled_posts.items():
            self._state.posts_schedule[step] = []
            for post in posts:
                post_copy = copy.deepcopy(post)
                post_copy.is_active = False
                self._state.posts[post_copy.post_id] = post_copy
                self._state.posts_schedule[step].append(post_copy.post_id)

        # Load campaigns
        for campaign in self._task_config.campaigns:
            self._state.campaigns[campaign.campaign_id] = copy.deepcopy(campaign)

        # Load truth reveal schedule
        self._state.truth_reveal_schedule = copy.deepcopy(
            self._task_config.truth_reveal_schedule
        )

        observation = self._build_observation()
        return ResetResult(observation=observation)

    def step(self, action: MisinfoCrisisAction) -> StepResult:
        """
        Execute one action in the environment.

        The simulation advances to the next time step when either:
        - The agent exhausts their action budget for the current step
        - The agent uses 'ignore' (signals end of turn)

        Order of operations:
        1. Validate action (budget, post existence)
        2. Apply action to post
        3. Evaluate justification quality
        4. Compute stepwise reward
        5. Advance simulation IF budget exhausted or action is 'ignore'
        6. Check termination
        7. Return observation + reward + done + info
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() for a new one.")

        info: Dict[str, Any] = {}

        # ── 1. Validate action ─────────────────────────────────────────────
        post = self._state.posts.get(action.post_id)

        # Budget check for non-ignore actions
        budget_exceeded = False
        if action.action_type != ActionType.IGNORE:
            if self._state.actions_this_step >= self._state.action_budget_per_step:
                budget_exceeded = True

        if post is None:
            reward = -0.1
            info = {"error": f"Post '{action.post_id}' not found", "valid_action": False}
            self._record_intervention(action, reward, was_correct=False, just_quality=0.0)

        elif not post.is_active:
            reward = -0.05
            info = {"error": f"Post '{action.post_id}' is not active", "valid_action": False}
            self._record_intervention(action, reward, was_correct=False, just_quality=0.0)

        elif budget_exceeded:
            reward = -0.15
            info = {
                "error": "Action budget exceeded for this step. Use 'ignore' to advance.",
                "valid_action": False,
                "budget_remaining": 0,
            }
            self._record_intervention(action, reward, was_correct=False, just_quality=0.0)

        else:
            # ── 2. Apply action ────────────────────────────────────────────
            was_correct = self._apply_action(action, post)

            # Count non-ignore action
            if action.action_type != ActionType.IGNORE:
                self._state.actions_this_step += 1

            # ── 3. Evaluate justification ──────────────────────────────────
            just_quality = self._grader.evaluate_justification(
                action.justification, action.action_type.value, post
            )

            # ── 4. Compute reward ──────────────────────────────────────────
            reward = self._grader.compute_step_reward(
                state=self._state,
                action_type=action.action_type.value,
                post_id=action.post_id,
                was_correct=was_correct,
                post=post,
                justification=action.justification,
            )

            info = {
                "valid_action": True,
                "was_correct": was_correct,
                "action_applied": action.action_type.value,
                "justification_quality": round(just_quality, 4),
                "budget_remaining": max(
                    0,
                    self._state.action_budget_per_step - self._state.actions_this_step,
                ),
            }

            self._record_intervention(action, reward, was_correct, just_quality)

        # ── 5. Advance simulation IF budget fully used OR ignore ────────────
        should_advance = (
            action.action_type == ActionType.IGNORE
            or self._state.actions_this_step >= self._state.action_budget_per_step
            or budget_exceeded
            or post is None
            or (post is not None and not post.is_active)
        )

        if should_advance:
            self._just_revealed = self._engine.advance_step(self._state)
            # Reset action budget for the new step
            self._state.actions_this_step = 0
        else:
            self._just_revealed = []

        self._state.cumulative_reward += reward

        if self._just_revealed:
            info["truth_revealed"] = self._just_revealed

        info["time_advanced"] = should_advance

        # ── 6. Check termination ───────────────────────────────────────────
        if self._state.time_step >= self._state.max_steps:
            self._state.done = True
            final_grade = self._grader.grade(self._state)
            info["final_grade"] = final_grade
            info["episode_complete"] = True

        # ── 7. Build observation ───────────────────────────────────────────
        observation = self._build_observation()

        return StepResult(
            observation=observation,
            reward=round(reward, 4),
            done=self._state.done,
            info=info,
        )

    def state(self) -> MisinfoCrisisState:
        """Return the full internal state (including ground truth)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    # ── Private Methods ────────────────────────────────────────────────────

    def _record_intervention(
        self,
        action: MisinfoCrisisAction,
        reward: float,
        was_correct: bool,
        just_quality: float,
    ) -> None:
        """Record an intervention in the history."""
        self._state.intervention_history.append(
            InterventionRecord(
                step=self._state.time_step,
                action_type=action.action_type.value,
                post_id=action.post_id,
                parameters=(
                    action.parameters.model_dump() if action.parameters else None
                ),
                justification=action.justification,
                justification_quality=just_quality,
                reward=reward,
                was_correct=was_correct,
            )
        )

    def _apply_action(self, action: MisinfoCrisisAction, post: FullPost) -> bool:
        """
        Apply the action to the post and return whether it was correct.

        "Correct" definition per action:
        - label(false/misleading) on FALSE/PARTIALLY_TRUE → correct
        - label(safe) on TRUE → correct
        - reduce_visibility on FALSE/PARTIALLY_TRUE → correct
        - add_context on PARTIALLY_TRUE/FALSE → correct
        - ignore on TRUE → correct
        """
        was_correct = False

        if action.action_type == ActionType.LABEL:
            label_type = (
                action.parameters.label_type
                if action.parameters and action.parameters.label_type
                else LabelType.MISLEADING
            )
            post.labels_applied.append(label_type.value)

            if post.truth_label == TruthLabel.FALSE:
                was_correct = label_type in (LabelType.FALSE, LabelType.MISLEADING)
                if was_correct:
                    post.visibility_multiplier *= 0.3
                    post.current_spread_rate *= 0.2
            elif post.truth_label == TruthLabel.PARTIALLY_TRUE:
                was_correct = label_type == LabelType.MISLEADING
                if was_correct:
                    post.visibility_multiplier *= 0.5
                    post.current_spread_rate *= 0.4
            else:  # TRUE
                was_correct = label_type == LabelType.SAFE
                if not was_correct:
                    self._state.total_true_content_censored += 1

            if post.first_detected_step is None and was_correct:
                post.first_detected_step = self._state.time_step

        elif action.action_type == ActionType.REDUCE_VISIBILITY:
            post.visibility_reduced = True
            post.visibility_multiplier *= 0.4
            post.current_spread_rate *= 0.5
            was_correct = post.truth_label in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE)
            if not was_correct:
                self._state.total_true_content_censored += 1

        elif action.action_type == ActionType.ADD_CONTEXT:
            note = (
                action.parameters.context_note
                if action.parameters and action.parameters.context_note
                else "Fact-check context added."
            )
            post.context_notes.append(note)
            post.visibility_multiplier *= 0.7
            post.current_spread_rate *= 0.6
            was_correct = post.truth_label in (TruthLabel.PARTIALLY_TRUE, TruthLabel.FALSE)

        elif action.action_type == ActionType.IGNORE:
            was_correct = post.truth_label == TruthLabel.TRUE

        post.intervention_steps.append(self._state.time_step)
        return was_correct

    def _build_observation(self) -> MisinfoCrisisObservation:
        """Build the agent-visible observation from current state."""
        observable_data = self._engine.get_observable_posts(self._state, max_visible=10)

        active_posts = []
        for pd in observable_data:
            signals = pd["signals"]
            if isinstance(signals, dict):
                signals = PostSignals(**signals)

            active_posts.append(
                ObservablePost(
                    post_id=pd["post_id"],
                    content=pd["content"],
                    image_description=pd.get("image_description"),
                    timestamp=pd["timestamp"],
                    likes=pd["likes"],
                    shares=pd["shares"],
                    comments=pd["comments"],
                    virality_velocity=pd["virality_velocity"],
                    signals=signals,
                    current_labels=pd["current_labels"],
                    has_context_note=pd["has_context_note"],
                    visibility_reduced=pd["visibility_reduced"],
                    is_verified=pd.get("is_verified", False),
                    verified_label=pd.get("verified_label"),
                )
            )

        # Build action summaries (last 5)
        previous_actions = []
        for record in self._state.intervention_history[-5:]:
            result_str = "correct" if record.was_correct else "incorrect"
            previous_actions.append(
                ActionSummary(
                    step=record.step,
                    action_type=record.action_type,
                    post_id=record.post_id,
                    result=result_str,
                    justification_quality=round(record.justification_quality, 2),
                )
            )

        # Platform statistics
        active_count = len([p for p in self._state.posts.values() if p.is_active])
        total_engagement = sum(
            p.likes + p.shares + p.comments
            for p in self._state.posts.values()
            if p.is_active
        )

        platform_stats = {
            "total_active_posts": active_count,
            "total_engagement": total_engagement,
            "total_interventions": len(self._state.intervention_history),
            "cumulative_reward": round(self._state.cumulative_reward, 4),
        }

        # Trend indicators
        trend_data = self._engine.get_trend_indicators(self._state)
        trend_indicators = TrendIndicator(**trend_data)

        # Budget remaining
        budget_remaining = max(
            0,
            self._state.action_budget_per_step - self._state.actions_this_step,
        )

        return MisinfoCrisisObservation(
            active_posts=active_posts,
            time_step=self._state.time_step,
            max_steps=self._state.max_steps,
            previous_actions=previous_actions,
            platform_stats=platform_stats,
            task_description=self._task_config.description if self._task_config else "",
            instruction=self._task_config.instruction if self._task_config else "",
            remaining_action_budget=budget_remaining,
            user_trust_score=round(self._state.user_trust, 4),
            backlash_score=round(self._state.backlash, 4),
            trend_indicators=trend_indicators,
            recently_verified=self._just_revealed,
        )


# ── Direct testing ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = MisinfoCrisisEnvironment()

    print("=" * 70)
    print("MISINFORMATION CRISIS SIMULATOR — ADVANCED EDITION SMOKE TEST")
    print("=" * 70)

    for task_id in ["easy_obvious_misinfo", "medium_subtle_misinfo", "hard_cascade_crisis"]:
        print(f"\n{'─' * 60}")
        print(f"Task: {task_id}")
        print(f"{'─' * 60}")

        result = env.reset(task_id=task_id, seed=42)
        obs = result.observation
        print(f"  Posts: {len(obs.active_posts)}")
        print(f"  Max steps: {obs.max_steps}")
        print(f"  Action budget/step: {obs.remaining_action_budget}")
        print(f"  Trust: {obs.user_trust_score}")
        print(f"  Backlash: {obs.backlash_score}")

        for p in obs.active_posts:
            print(f"\n  [{p.post_id}]")
            print(f"    Content: {p.content[:70]}...")
            if p.image_description:
                print(f"    Image: {p.image_description[:60]}...")
            print(f"    Credibility: {p.signals.credibility_score:.3f}")
            print(f"    Campaign signal: similar={p.signals.similar_post_count} coord={p.signals.coordinated_activity_flag}")

        # Run quick episode
        done = False
        step_count = 0
        while not done:
            step_count += 1
            posts = obs.active_posts if step_count == 1 else step_result.observation.active_posts
            if not posts:
                break

            target = min(posts, key=lambda p: p.signals.credibility_score)
            if target.signals.credibility_score < 0.5:
                action = MisinfoCrisisAction(
                    action_type=ActionType.LABEL,
                    post_id=target.post_id,
                    parameters=ActionParameters(label_type=LabelType.FALSE),
                    justification=f"Post has low credibility score ({target.signals.credibility_score}) and appears to contain misinformation.",
                )
            else:
                action = MisinfoCrisisAction(
                    action_type=ActionType.IGNORE,
                    post_id=target.post_id,
                    justification=f"Post has high credibility ({target.signals.credibility_score}) and appears legitimate.",
                )

            step_result = env.step(action)
            done = step_result.done

            if step_result.info.get("truth_revealed"):
                print(f"  Step {step_count}: Truth revealed for {step_result.info['truth_revealed']}")

        if step_result.done:
            grade = step_result.info.get("final_grade", {})
            print(f"\n  📊 FINAL GRADE: {grade.get('overall_score', 0):.4f}")
            for k in ["detection_score", "timing_score", "precision_score", "spread_score",
                       "campaign_score", "justification_score", "trust_score"]:
                print(f"     {k}: {grade.get(k, 0):.4f}")
            print(f"     Campaigns detected: {grade.get('campaigns_detected', 0)}/{grade.get('campaigns_total', 0)}")
            print(f"     Trust: {grade.get('final_trust', 0):.4f} | Backlash: {grade.get('final_backlash', 0):.4f}")

    print(f"\n{'=' * 70}")
    print("✅ All tasks completed successfully!")
    print(f"{'=' * 70}")
