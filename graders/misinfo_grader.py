"""
Misinformation Grader — Advanced Edition

Deterministic scoring with expanded criteria:
1. Detection accuracy (correct identification of misinfo)
2. Timing of intervention (early detection bonuses)
3. Precision (avoiding false positives / over-censorship)
4. Spread minimization (total misinformation propagation)
5. Campaign detection (identifying coordinated campaigns)
6. Justification quality (explainability of decisions)
7. Trust balance (maintaining platform trust vs backlash)

Output: score between 0.0 and 1.0
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from models import (
    ActionType,
    Campaign,
    FullPost,
    InterventionRecord,
    LabelType,
    MisinfoCrisisState,
    TruthLabel,
)


class MisinfoGrader:
    """
    Deterministic grader for episode performance.

    Seven scoring components (weights sum to 1.0):
    - detection_score   (0.20): correctly identified misinfo posts
    - timing_score      (0.15): how early interventions occurred
    - precision_score   (0.15): avoiding false positives
    - spread_score      (0.15): misinformation propagation minimized
    - campaign_score    (0.15): detecting coordinated campaigns
    - justification_score (0.10): quality of action justifications
    - trust_score       (0.10): maintaining platform trust balance
    """

    WEIGHT_DETECTION = 0.20
    WEIGHT_TIMING = 0.15
    WEIGHT_PRECISION = 0.15
    WEIGHT_SPREAD = 0.15
    WEIGHT_CAMPAIGN = 0.15
    WEIGHT_JUSTIFICATION = 0.10
    WEIGHT_TRUST = 0.10

    def grade(self, state: MisinfoCrisisState) -> Dict:
        """
        Grade the agent's overall performance for the episode.

        Args:
            state: Final state after episode completion.

        Returns:
            Dict with overall score (0.0-1.0) and component breakdowns.
        """
        detection = self._compute_detection_score(state)
        timing = self._compute_timing_score(state)
        precision = self._compute_precision_score(state)
        spread = self._compute_spread_score(state)
        campaign = self._compute_campaign_score(state)
        justification = self._compute_justification_score(state)
        trust = self._compute_trust_score(state)

        overall = (
            self.WEIGHT_DETECTION * detection
            + self.WEIGHT_TIMING * timing
            + self.WEIGHT_PRECISION * precision
            + self.WEIGHT_SPREAD * spread
            + self.WEIGHT_CAMPAIGN * campaign
            + self.WEIGHT_JUSTIFICATION * justification
            + self.WEIGHT_TRUST * trust
        )

        return {
            "overall_score": round(max(0.0, min(1.0, overall)), 4),
            "detection_score": round(detection, 4),
            "timing_score": round(timing, 4),
            "precision_score": round(precision, 4),
            "spread_score": round(spread, 4),
            "campaign_score": round(campaign, 4),
            "justification_score": round(justification, 4),
            "trust_score": round(trust, 4),
            "total_misinformation_spread": round(state.total_misinformation_spread, 4),
            "final_trust": round(state.user_trust, 4),
            "final_backlash": round(state.backlash, 4),
            "total_interventions": len(state.intervention_history),
            "correct_interventions": sum(
                1 for i in state.intervention_history if i.was_correct
            ),
            "false_positives": sum(
                1
                for i in state.intervention_history
                if not i.was_correct
                and i.action_type in (ActionType.LABEL.value, ActionType.REDUCE_VISIBILITY.value)
            ),
            "campaigns_detected": sum(
                1 for c in state.campaigns.values() if c.detected
            ),
            "campaigns_total": len(state.campaigns),
        }

    # ── Step Reward ────────────────────────────────────────────────────────

    def compute_step_reward(
        self,
        state: MisinfoCrisisState,
        action_type: str,
        post_id: str,
        was_correct: bool,
        post: FullPost,
        justification: str = "",
    ) -> float:
        """
        Compute dense reward for a single step.

        Positive rewards:
        - Early correct detection: +0.3 to +0.5
        - Effective visibility reduction on misinfo: +0.2
        - Correct context on partial truths: +0.25
        - Campaign post detection bonus: +0.15
        - Good justification: +0.1

        Negative rewards:
        - Over-censorship (labeling true as false): -0.4
        - Ignoring high-risk misinfo: -0.3
        - Missing or poor justification: -0.1
        - Causing backlash / trust drop: -0.1
        """
        reward = 0.0
        step = state.time_step
        post_age = step - post.timestamp

        if action_type == ActionType.LABEL.value:
            reward += self._reward_label(post, was_correct, post_age, state)
        elif action_type == ActionType.REDUCE_VISIBILITY.value:
            reward += self._reward_reduce_visibility(post, was_correct, post_age)
        elif action_type == ActionType.ADD_CONTEXT.value:
            reward += self._reward_add_context(post, was_correct)
        elif action_type == ActionType.IGNORE.value:
            reward += self._reward_ignore(post, post_age)

        # Campaign bonus: extra reward for acting on campaign posts
        if was_correct and post.campaign_id:
            reward += 0.15

        # Justification quality bonus/penalty
        just_score = self.evaluate_justification(justification, action_type, post)
        if just_score >= 0.6:
            reward += 0.1 * just_score
        elif just_score < 0.2:
            reward -= 0.1

        # Trust/backlash penalty
        if state.backlash > 0.5:
            reward -= 0.05
        if state.user_trust < 0.5:
            reward -= 0.05

        return round(reward, 4)

    def _reward_label(
        self, post: FullPost, was_correct: bool, post_age: int, state: MisinfoCrisisState
    ) -> float:
        if was_correct:
            early_bonus = max(0.0, 0.2 * (1.0 - post_age / state.max_steps))
            base_reward = 0.3 + early_bonus
            if post.sensitivity_level > 0.7:
                base_reward += 0.1
            return base_reward
        else:
            if post.truth_label == TruthLabel.TRUE:
                return -0.4
            return -0.15

    def _reward_reduce_visibility(
        self, post: FullPost, was_correct: bool, post_age: int
    ) -> float:
        if post.truth_label in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE):
            base = 0.2
            if post.virality_score > 0.7:
                base += 0.1
            late_penalty = max(0.0, 0.05 * (post_age - 3))
            return max(0.05, base - late_penalty)
        else:
            return -0.25

    def _reward_add_context(self, post: FullPost, was_correct: bool) -> float:
        if post.truth_label == TruthLabel.PARTIALLY_TRUE:
            return 0.25
        elif post.truth_label == TruthLabel.FALSE:
            return 0.1
        else:
            return -0.05

    def _reward_ignore(self, post: FullPost, post_age: int) -> float:
        if post.truth_label == TruthLabel.TRUE:
            return 0.1
        elif post.truth_label == TruthLabel.FALSE:
            penalty = -0.1 * post.sensitivity_level * (1 + 0.1 * post_age)
            return max(-0.5, penalty)
        else:
            return -0.05

    # ── Justification Evaluator ────────────────────────────────────────────

    def evaluate_justification(
        self, justification: str, action_type: str, post: FullPost
    ) -> float:
        """
        Evaluate justification quality deterministically.

        Criteria:
        - Non-empty and minimum length (>20 chars)
        - References the post content
        - Consistent with action type
        - Mentions relevant signals

        Returns: 0.0 to 1.0
        """
        if not justification or len(justification.strip()) < 5:
            return 0.0

        score = 0.0
        text = justification.lower().strip()

        # 1. Length adequacy (0 to 0.2)
        if len(text) >= 20:
            score += 0.1
        if len(text) >= 50:
            score += 0.1

        # 2. References post content (0 to 0.2)
        post_words = set(post.content.lower().split())
        common_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "it", "this", "that"}
        meaningful_words = post_words - common_words
        matches = sum(1 for w in meaningful_words if w in text)
        if matches >= 2:
            score += 0.2
        elif matches >= 1:
            score += 0.1

        # 3. Action consistency (0 to 0.3)
        if action_type == ActionType.LABEL.value:
            if any(w in text for w in ["false", "misleading", "misinformation", "incorrect", "debunked", "fabricated"]):
                score += 0.2
            if any(w in text for w in ["safe", "true", "verified", "accurate", "legitimate", "credible"]):
                score += 0.1
        elif action_type == ActionType.REDUCE_VISIBILITY.value:
            if any(w in text for w in ["viral", "spreading", "harmful", "dangerous", "visibility", "reach"]):
                score += 0.3
        elif action_type == ActionType.ADD_CONTEXT.value:
            if any(w in text for w in ["context", "partial", "misleading", "framing", "nuance", "clarif"]):
                score += 0.3
        elif action_type == ActionType.IGNORE.value:
            if any(w in text for w in ["true", "legitimate", "credible", "verified", "safe", "accurate"]):
                score += 0.3

        # 4. Mentions signals (0 to 0.2)
        signal_words = ["credibility", "report", "reputation", "fact-check", "similar", "coordinated", "campaign", "image", "source"]
        signal_matches = sum(1 for w in signal_words if w in text)
        score += min(0.2, signal_matches * 0.05)

        # 5. Coherence — not just random words (0 to 0.1)
        words = text.split()
        if len(words) >= 5 and len(set(words)) / len(words) > 0.5:
            score += 0.1

        return min(1.0, score)

    # ── Component Scores ───────────────────────────────────────────────────

    def _compute_detection_score(self, state: MisinfoCrisisState) -> float:
        """Fraction of misinfo posts correctly detected."""
        misinfo_posts = {
            pid
            for pid, p in state.posts.items()
            if p.truth_label in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE)
        }
        if not misinfo_posts:
            return 1.0

        detected = set()
        for intervention in state.intervention_history:
            if (
                intervention.post_id in misinfo_posts
                and intervention.action_type != ActionType.IGNORE.value
            ):
                detected.add(intervention.post_id)

        return len(detected) / len(misinfo_posts)

    def _compute_timing_score(self, state: MisinfoCrisisState) -> float:
        """How early interventions occurred relative to post appearance."""
        misinfo_posts = {
            pid: p
            for pid, p in state.posts.items()
            if p.truth_label in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE)
        }
        if not misinfo_posts:
            return 1.0

        timing_scores = []
        for pid, post in misinfo_posts.items():
            first_action_step = None
            for intervention in state.intervention_history:
                if (
                    intervention.post_id == pid
                    and intervention.action_type != ActionType.IGNORE.value
                ):
                    first_action_step = intervention.step
                    break

            if first_action_step is None:
                timing_scores.append(0.0)
            else:
                delay = first_action_step - post.timestamp
                score = max(0.0, 1.0 - delay * 0.15)
                timing_scores.append(score)

        return sum(timing_scores) / len(timing_scores) if timing_scores else 1.0

    def _compute_precision_score(self, state: MisinfoCrisisState) -> float:
        """Avoid false positives."""
        total = len([
            i for i in state.intervention_history
            if i.action_type != ActionType.IGNORE.value
        ])
        if total == 0:
            return 0.5

        false_positives = sum(
            1
            for i in state.intervention_history
            if not i.was_correct
            and i.action_type in (ActionType.LABEL.value, ActionType.REDUCE_VISIBILITY.value)
        )
        return max(0.0, 1.0 - false_positives / total)

    def _compute_spread_score(self, state: MisinfoCrisisState) -> float:
        """Lower misinformation spread = higher score."""
        max_possible = 0.0
        for post in state.posts.values():
            if post.truth_label in (TruthLabel.FALSE, TruthLabel.PARTIALLY_TRUE):
                max_possible += (
                    post.virality_score * post.sensitivity_level * state.max_steps * 10.0
                )
        if max_possible == 0:
            return 1.0

        ratio = min(1.0, state.total_misinformation_spread / max(1.0, max_possible))
        return max(0.0, 1.0 - ratio)

    def _compute_campaign_score(self, state: MisinfoCrisisState) -> float:
        """
        Score for detecting coordinated campaigns.

        A campaign is "detected" if the agent acted on ≥50% of its posts
        with non-ignore actions.
        """
        if not state.campaigns:
            return 1.0

        campaign_scores = []
        for campaign in state.campaigns.values():
            if not campaign.post_ids:
                campaign_scores.append(1.0)
                continue

            acted_on = 0
            for post_id in campaign.post_ids:
                for intervention in state.intervention_history:
                    if (
                        intervention.post_id == post_id
                        and intervention.action_type != ActionType.IGNORE.value
                    ):
                        acted_on += 1
                        break

            detection_ratio = acted_on / len(campaign.post_ids)
            campaign_scores.append(detection_ratio)

            # Mark campaign as detected if ratio >= 0.5
            if detection_ratio >= 0.5:
                campaign.detected = True

        return sum(campaign_scores) / len(campaign_scores) if campaign_scores else 1.0

    def _compute_justification_score(self, state: MisinfoCrisisState) -> float:
        """Average justification quality across all interventions."""
        if not state.intervention_history:
            return 0.5

        qualities = [i.justification_quality for i in state.intervention_history]
        return sum(qualities) / len(qualities) if qualities else 0.5

    def _compute_trust_score(self, state: MisinfoCrisisState) -> float:
        """
        Score based on final trust and backlash levels.

        High trust + low backlash = good score.
        """
        trust_component = state.user_trust  # 0 to 1
        backlash_penalty = state.backlash * 0.5  # 0 to 0.5
        return max(0.0, min(1.0, trust_component - backlash_penalty))
