"""
Propagation Engine — Advanced Edition

Simulates post spread, engagement dynamics, cascading effects,
coordinated campaign amplification, public trust dynamics,
and delayed ground truth revelation.

Deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from models import (
    Campaign,
    FullPost,
    MisinfoCrisisState,
    PostSignals,
    TruthLabel,
)


class PropagationEngine:
    """
    Simulates social media post propagation with advanced dynamics.

    Each step:
    1. Existing posts grow according to spread rate & visibility
    2. New posts appear from schedule
    3. Campaign bursts amplify coordinated posts
    4. Cascading effects spawn child posts
    5. Delayed ground truths may be revealed
    6. Public trust / backlash metrics update
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def advance_step(self, state: MisinfoCrisisState) -> List[str]:
        """
        Advance the simulation by one time step.

        Modifies state in-place. Returns list of post_ids
        whose truth was revealed this step.

        Args:
            state: The full simulation state.

        Returns:
            List of post_ids with newly revealed truth labels.
        """
        state.time_step += 1
        # Note: actions_this_step is NOT reset here.
        # The environment resets it based on the action budget logic.

        # 1. Introduce scheduled posts
        self._introduce_scheduled_posts(state)

        # 2. Campaign burst amplification
        self._process_campaign_bursts(state)

        # 3. Update engagement for all active posts
        for post in state.posts.values():
            if post.is_active:
                self._update_engagement(post, state.time_step)

        # 4. Cascade: high-share posts spawn children
        self._process_cascades(state)

        # 5. Reveal delayed ground truths
        revealed = self._reveal_delayed_truths(state)

        # 6. Track misinformation spread
        self._update_misinformation_metrics(state)

        # 7. Update public trust dynamics
        self._update_trust_dynamics(state)

        return revealed

    # ── Post Scheduling ────────────────────────────────────────────────────

    def _introduce_scheduled_posts(self, state: MisinfoCrisisState) -> None:
        """Activate posts scheduled for the current time step."""
        step_key = state.time_step
        if step_key in state.posts_schedule:
            for post_id in state.posts_schedule[step_key]:
                if post_id in state.posts:
                    state.posts[post_id].is_active = True
                    state.posts[post_id].timestamp = state.time_step

    # ── Campaign Bursts ────────────────────────────────────────────────────

    def _process_campaign_bursts(self, state: MisinfoCrisisState) -> None:
        """
        Amplify coordinated campaign posts during burst steps.
        During a burst, all campaign posts get a temporary engagement boost.
        """
        for campaign in state.campaigns.values():
            if state.time_step in campaign.burst_steps:
                for post_id in campaign.post_ids:
                    post = state.posts.get(post_id)
                    if post and post.is_active:
                        # Burst amplification: temporary spread rate boost
                        boost = 1.5 + campaign.coordination_strength * 1.0
                        post.current_spread_rate *= boost
                        # Artificial engagement injection
                        post.likes += int(20 * campaign.coordination_strength * self.rng.uniform(0.8, 1.2))
                        post.shares += int(10 * campaign.coordination_strength * self.rng.uniform(0.8, 1.2))

    # ── Engagement Update ──────────────────────────────────────────────────

    def _update_engagement(self, post: FullPost, current_step: int) -> None:
        """
        Update likes, shares, comments based on spread rate and visibility.
        Uses a logistic growth model modulated by virality_score.
        """
        age = current_step - post.timestamp
        if age <= 0:
            return

        growth_factor = self._logistic_growth(age, post.virality_score)
        effective_rate = (
            post.current_spread_rate * post.visibility_multiplier * growth_factor
        )

        new_likes = max(0, int(effective_rate * self.rng.gauss(50, 15)))
        new_shares = max(0, int(effective_rate * self.rng.gauss(15, 5)))
        new_comments = max(0, int(effective_rate * self.rng.gauss(20, 8)))

        post.likes += new_likes
        post.shares += new_shares
        post.comments += new_comments
        post.total_reach += new_shares * 10 + new_likes

    def _logistic_growth(self, age: int, virality: float) -> float:
        """Logistic curve: peaks then decays."""
        peak_step = 3 + virality * 7
        k = 0.5 + virality * 0.5
        if age <= peak_step:
            return 1.0 / (1.0 + math.exp(-k * (age - peak_step / 2)))
        decay = math.exp(-0.1 * (age - peak_step))
        return max(0.05, decay)

    # ── Cascading ──────────────────────────────────────────────────────────

    def _process_cascades(self, state: MisinfoCrisisState) -> None:
        """High-share posts may spawn child posts in the propagation graph."""
        new_children: List[Tuple[str, FullPost]] = []

        for post_id, post in state.posts.items():
            if not post.is_active:
                continue

            cascade_threshold = int(50 / (post.virality_score + 0.1))
            if (
                post.shares > cascade_threshold
                and self.rng.random() < 0.15 * post.virality_score
            ):
                existing = state.propagation_graph.get(post_id, [])
                if len(existing) >= 3:
                    continue

                child_id = f"{post_id}_cascade_{state.time_step}"
                if child_id in state.posts:
                    continue

                child = FullPost(
                    post_id=child_id,
                    content=f"[Reshared/Modified] {post.content[:80]}...",
                    image_description=post.image_description,
                    truth_label=post.truth_label,
                    virality_score=max(
                        0.1, post.virality_score * self.rng.uniform(0.5, 0.9)
                    ),
                    sensitivity_level=post.sensitivity_level,
                    timestamp=state.time_step,
                    base_spread_rate=post.base_spread_rate * 0.7,
                    current_spread_rate=post.current_spread_rate * 0.7,
                    visibility_multiplier=1.0 if not post.visibility_reduced else 0.6,
                    campaign_id=post.campaign_id,
                    image_text_mismatch=post.image_text_mismatch,
                )
                new_children.append((child_id, child))

                if post_id not in state.propagation_graph:
                    state.propagation_graph[post_id] = []
                state.propagation_graph[post_id].append(child_id)

                # Also register with campaign if applicable
                if post.campaign_id and post.campaign_id in state.campaigns:
                    state.campaigns[post.campaign_id].post_ids.append(child_id)

        for child_id, child in new_children:
            state.posts[child_id] = child

    # ── Delayed Truth ──────────────────────────────────────────────────────

    def _reveal_delayed_truths(self, state: MisinfoCrisisState) -> List[str]:
        """Reveal ground truth for posts scheduled at this step."""
        revealed = []
        step_key = state.time_step

        if step_key in state.truth_reveal_schedule:
            for post_id in state.truth_reveal_schedule[step_key]:
                post = state.posts.get(post_id)
                if post and not post.truth_revealed:
                    post.truth_revealed = True
                    state.revealed_truths.append(post_id)
                    revealed.append(post_id)

        return revealed

    # ── Misinformation Metrics ─────────────────────────────────────────────

    def _update_misinformation_metrics(self, state: MisinfoCrisisState) -> None:
        """Track total misinformation spread across the platform."""
        step_misinfo = 0.0
        for post in state.posts.values():
            if not post.is_active:
                continue
            if post.truth_label in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE):
                spread = (
                    post.total_reach
                    * post.sensitivity_level
                    * post.visibility_multiplier
                )
                if post.truth_label == TruthLabel.PARTIALLY_TRUE:
                    spread *= 0.5
                step_misinfo += spread

        state.total_misinformation_spread += step_misinfo * 0.001

    # ── Public Trust / Backlash ────────────────────────────────────────────

    def _update_trust_dynamics(self, state: MisinfoCrisisState) -> None:
        """
        Update public trust and backlash based on recent interventions.

        Trust drops when:
        - Misinfo is left unchecked (people see harmful content)
        - The platform over-censors (people lose trust in fairness)

        Backlash rises when:
        - True content is censored
        - Many interventions happen in a short time
        """
        recent_interventions = [
            i
            for i in state.intervention_history
            if i.step == state.time_step - 1
        ]

        if not recent_interventions:
            # Passive trust decay if misinformation is spreading
            active_misinfo = sum(
                1
                for p in state.posts.values()
                if p.is_active
                and p.truth_label in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE)
                and not p.labels_applied
                and not p.visibility_reduced
            )
            if active_misinfo > 0:
                state.user_trust = max(
                    0.0, state.user_trust - 0.01 * active_misinfo
                )
            # Backlash decays naturally
            state.backlash = max(0.0, state.backlash - 0.02)
            return

        correct_actions = sum(1 for i in recent_interventions if i.was_correct)
        incorrect_actions = sum(1 for i in recent_interventions if not i.was_correct)
        total_actions = len(recent_interventions)

        # Trust increases with correct moderation
        if correct_actions > 0:
            state.user_trust = min(
                1.0, state.user_trust + 0.02 * correct_actions
            )

        # Trust drops and backlash rises with incorrect moderation
        if incorrect_actions > 0:
            state.user_trust = max(
                0.0, state.user_trust - 0.05 * incorrect_actions
            )
            state.backlash = min(
                1.0, state.backlash + 0.08 * incorrect_actions
            )

        # Excessive intervention causes backlash even if correct
        if total_actions >= 3:
            state.backlash = min(1.0, state.backlash + 0.03)

        # Natural backlash decay
        state.backlash = max(0.0, state.backlash - 0.01)

    # ── Observable Posts ───────────────────────────────────────────────────

    def get_observable_posts(
        self, state: MisinfoCrisisState, max_visible: int = 10
    ) -> List[Dict]:
        """
        Generate agent-visible posts with noisy signals.
        Does NOT include ground truth labels (unless truth revealed).
        """
        active_posts = [p for p in state.posts.values() if p.is_active]
        active_posts.sort(
            key=lambda p: p.likes + p.shares + p.comments, reverse=True
        )
        visible = active_posts[:max_visible]

        # Pre-compute campaign similarity counts
        campaign_counts: Dict[Optional[str], int] = {}
        for p in state.posts.values():
            if p.is_active and p.campaign_id:
                campaign_counts[p.campaign_id] = (
                    campaign_counts.get(p.campaign_id, 0) + 1
                )

        result = []
        for post in visible:
            # Generate noisy credibility signal
            if post.truth_label == TruthLabel.TRUE:
                base_cred = self.rng.uniform(0.6, 0.95)
            elif post.truth_label == TruthLabel.PARTIALLY_TRUE:
                base_cred = self.rng.uniform(0.3, 0.7)
            else:
                base_cred = self.rng.uniform(0.05, 0.45)

            noisy_cred = max(0.0, min(1.0, base_cred + self.rng.gauss(0, 0.1)))

            age = max(1, state.time_step - post.timestamp)
            virality_velocity = post.shares / age

            # Campaign detection signals (noisy)
            similar_count = 0
            coord_flag = False
            if post.campaign_id:
                real_count = campaign_counts.get(post.campaign_id, 0)
                # Noisy signal of campaign size
                similar_count = max(
                    0, real_count + int(self.rng.gauss(0, 1))
                )
                campaign = state.campaigns.get(post.campaign_id)
                coord_strength = campaign.coordination_strength if campaign else 0.0
                coord_flag = self.rng.random() < 0.4 + 0.3 * coord_strength

            signals = PostSignals(
                credibility_score=round(noisy_cred, 3),
                source_reputation=round(self.rng.uniform(0.2, 0.9), 3),
                fact_check_available=(
                    self.rng.random() < 0.3
                    if post.truth_label != TruthLabel.TRUE
                    else self.rng.random() < 0.1
                ),
                user_reports=max(
                    0,
                    int(
                        self.rng.gauss(
                            5 if post.truth_label == TruthLabel.FALSE else 1, 2
                        )
                    ),
                ),
                similar_post_count=similar_count,
                coordinated_activity_flag=coord_flag,
            )

            # Delayed truth: expose label if revealed
            verified_label = None
            is_verified = False
            if post.truth_revealed:
                is_verified = True
                verified_label = post.truth_label.value

            result.append(
                {
                    "post_id": post.post_id,
                    "content": post.content,
                    "image_description": post.image_description,
                    "timestamp": post.timestamp,
                    "likes": post.likes,
                    "shares": post.shares,
                    "comments": post.comments,
                    "virality_velocity": round(virality_velocity, 2),
                    "signals": signals,
                    "current_labels": post.labels_applied,
                    "has_context_note": len(post.context_notes) > 0,
                    "visibility_reduced": post.visibility_reduced,
                    "is_verified": is_verified,
                    "verified_label": verified_label,
                }
            )

        return result

    def get_trend_indicators(self, state: MisinfoCrisisState) -> Dict:
        """Compute aggregate trend indicators for the observation."""
        active = [p for p in state.posts.values() if p.is_active]
        new_this_step = sum(
            1 for p in active if p.timestamp == state.time_step
        )
        avg_virality = (
            sum(p.virality_score for p in active) / len(active)
            if active
            else 0.0
        )

        # Suspected campaigns: noisy estimate
        real_campaigns = len(
            [c for c in state.campaigns.values() if len(c.post_ids) > 1]
        )
        suspected = max(0, real_campaigns + int(self.rng.gauss(0, 1)))

        # Platform health = trust - backlash - misinformation ratio
        misinfo_ratio = (
            sum(
                1
                for p in active
                if p.truth_label
                in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE)
                and not p.labels_applied
            )
            / max(1, len(active))
        )
        platform_health = max(
            0.0,
            min(
                1.0,
                state.user_trust - state.backlash * 0.5 - misinfo_ratio * 0.3,
            ),
        )

        # Trending topics (from active posts)
        topics = []
        for p in active[:5]:
            words = p.content.split()[:3]
            topics.append(" ".join(words))

        return {
            "trending_topics": topics[:3],
            "new_posts_this_step": new_this_step,
            "average_virality": round(avg_virality, 3),
            "suspected_campaigns": suspected,
            "platform_health_score": round(platform_health, 3),
        }
