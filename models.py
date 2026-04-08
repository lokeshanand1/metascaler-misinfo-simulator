"""
Pydantic models for the Autonomous Misinformation Crisis Simulator — Advanced Edition.

Defines Action, Observation, State schemas with support for:
- Coordinated adversarial campaigns
- Multi-modal posts (text + image description)
- Policy constraints (action budget)
- Public reaction simulation (trust / backlash)
- Explainability requirement (justification grading)
- Delayed ground truth revelation
- Resource / attention budget
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """Available moderation actions the agent can perform."""
    LABEL = "label"
    REDUCE_VISIBILITY = "reduce_visibility"
    ADD_CONTEXT = "add_context"
    IGNORE = "ignore"


class LabelType(str, Enum):
    """Label categories for the label action."""
    MISLEADING = "misleading"
    FALSE = "false"
    SAFE = "safe"


class TruthLabel(str, Enum):
    """Ground truth classification of a post."""
    TRUE = "true"
    FALSE = "false"
    PARTIALLY_TRUE = "partially_true"


# ── Action Models ──────────────────────────────────────────────────────────────

class ActionParameters(BaseModel):
    """Parameters for an action. Fields are used depending on action_type."""
    label_type: Optional[LabelType] = Field(
        None, description="Required for 'label' action: misleading, false, or safe"
    )
    context_note: Optional[str] = Field(
        None, description="Required for 'add_context' action: contextual note to attach"
    )


class MisinfoCrisisAction(BaseModel):
    """
    Structured action the agent submits each step.

    The justification field is REQUIRED — actions without justification
    receive a grading penalty. This tests the agent's explainability.

    Examples:
        Label a post as false:
            {
                "action_type": "label",
                "post_id": "post_3",
                "parameters": {"label_type": "false"},
                "justification": "Post claims bleach cures diseases, contradicting WHO guidelines."
            }

        Add context:
            {
                "action_type": "add_context",
                "post_id": "post_1",
                "parameters": {"context_note": "This claim cherry-picks data from a retracted study."},
                "justification": "The post uses real statistics out of context to mislead."
            }
    """
    action_type: ActionType = Field(..., description="Type of moderation action")
    post_id: str = Field(..., description="ID of the target post")
    parameters: Optional[ActionParameters] = Field(
        None, description="Action-specific parameters"
    )
    justification: str = Field(
        "",
        description="REQUIRED: Structured justification for this action. Graded for quality."
    )


# ── Observable Post ────────────────────────────────────────────────────────────

class PostSignals(BaseModel):
    """Partial signals available to the agent (NOT ground truth)."""
    credibility_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Estimated credibility (noisy signal, not ground truth)"
    )
    source_reputation: float = Field(
        ..., ge=0.0, le=1.0,
        description="Reputation of the posting source"
    )
    fact_check_available: bool = Field(
        False, description="Whether a fact-check exists for this topic"
    )
    user_reports: int = Field(
        0, ge=0, description="Number of user reports on this post"
    )
    similar_post_count: int = Field(
        0, ge=0,
        description="Number of similar posts detected (campaign signal)"
    )
    coordinated_activity_flag: bool = Field(
        False,
        description="Whether coordinated inauthentic behavior is suspected"
    )


class ObservablePost(BaseModel):
    """A post as visible to the agent (subset of full information)."""
    post_id: str
    content: str
    image_description: Optional[str] = Field(
        None,
        description="Description of attached image/media (multi-modal content)"
    )
    timestamp: int = Field(..., description="Step at which the post appeared")
    likes: int = Field(0, ge=0)
    shares: int = Field(0, ge=0)
    comments: int = Field(0, ge=0)
    virality_velocity: float = Field(
        0.0, ge=0.0,
        description="Rate of engagement growth (shares per step)"
    )
    signals: PostSignals
    current_labels: List[str] = Field(
        default_factory=list,
        description="Labels already applied to this post"
    )
    has_context_note: bool = Field(
        False, description="Whether a context note has been added"
    )
    visibility_reduced: bool = Field(
        False, description="Whether visibility has been reduced"
    )
    is_verified: bool = Field(
        False,
        description="Whether ground truth has been revealed for this post (delayed truth)"
    )
    verified_label: Optional[str] = Field(
        None,
        description="Revealed truth label (only populated after verification)"
    )


class TrendIndicator(BaseModel):
    """Aggregate trend information visible to the agent."""
    trending_topics: List[str] = Field(default_factory=list)
    new_posts_this_step: int = 0
    average_virality: float = 0.0
    suspected_campaigns: int = 0
    platform_health_score: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Overall platform health indicator"
    )


class ActionSummary(BaseModel):
    """Summary of a previous action taken by the agent."""
    step: int
    action_type: str
    post_id: str
    result: str
    justification_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Quality score of the justification (graded)"
    )


# ── Observation ────────────────────────────────────────────────────────────────

class MisinfoCrisisObservation(BaseModel):
    """
    Observation returned to the agent after each step or reset.

    Contains visible posts, engagement metrics, action history,
    public sentiment, and resource constraints.
    Ground truth is NEVER exposed here (except for verified posts).
    """
    active_posts: List[ObservablePost] = Field(
        default_factory=list,
        description="Currently active posts visible to the agent"
    )
    time_step: int = Field(0, description="Current simulation time step")
    max_steps: int = Field(20, description="Maximum steps in this episode")
    previous_actions: List[ActionSummary] = Field(
        default_factory=list,
        description="Summary of actions taken in previous steps"
    )
    platform_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate platform statistics"
    )
    task_description: str = Field("", description="Description of the current task")
    instruction: str = Field("", description="Instructions for the agent")

    # ── Advanced features ──
    remaining_action_budget: int = Field(
        3,
        description="Number of non-ignore actions the agent can take this step"
    )
    user_trust_score: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Current public trust in the platform (affected by agent actions)"
    )
    backlash_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Current public backlash level (increases with over-censorship)"
    )
    trend_indicators: TrendIndicator = Field(
        default_factory=TrendIndicator,
        description="Aggregate trend and platform health indicators"
    )
    recently_verified: List[str] = Field(
        default_factory=list,
        description="Post IDs whose ground truth was just revealed this step"
    )


# ── Campaign (hidden) ─────────────────────────────────────────────────────────

class Campaign(BaseModel):
    """
    Coordinated adversarial campaign (hidden from agent).

    A group of posts pushing the same narrative with slight paraphrasing,
    time-synchronized bursts, and hidden network relationships.
    """
    campaign_id: str
    narrative: str = Field(..., description="The core false narrative being pushed")
    post_ids: List[str] = Field(default_factory=list)
    start_step: int = 0
    burst_steps: List[int] = Field(
        default_factory=list,
        description="Steps when synchronized bursts occur"
    )
    coordination_strength: float = Field(
        0.5, ge=0.0, le=1.0,
        description="How tightly coordinated the campaign posts are"
    )
    detected: bool = False
    detection_step: Optional[int] = None


# ── Full Post (hidden state) ──────────────────────────────────────────────────

class FullPost(BaseModel):
    """Complete post data including hidden ground truth (server-side only)."""
    post_id: str
    content: str
    image_description: Optional[str] = None
    truth_label: TruthLabel
    virality_score: float = Field(..., ge=0.0, le=1.0)
    sensitivity_level: float = Field(..., ge=0.0, le=1.0)
    timestamp: int
    likes: int = 0
    shares: int = 0
    comments: int = 0
    base_spread_rate: float = 0.1
    current_spread_rate: float = 0.1
    total_reach: int = 0
    is_active: bool = True

    # Moderation state
    labels_applied: List[str] = Field(default_factory=list)
    context_notes: List[str] = Field(default_factory=list)
    visibility_reduced: bool = False
    visibility_multiplier: float = 1.0
    first_detected_step: Optional[int] = None
    intervention_steps: List[int] = Field(default_factory=list)

    # Campaign link (hidden)
    campaign_id: Optional[str] = Field(
        None, description="ID of coordinated campaign this post belongs to"
    )

    # Delayed truth
    truth_revealed: bool = Field(
        False, description="Whether ground truth has been revealed to the agent"
    )
    truth_reveal_step: Optional[int] = Field(
        None, description="Step at which truth will be revealed"
    )

    # Multi-modal flags
    image_text_mismatch: bool = Field(
        False,
        description="True if the image contradicts or misleads relative to the text"
    )


class InterventionRecord(BaseModel):
    """Record of an intervention action."""
    step: int
    action_type: str
    post_id: str
    parameters: Optional[Dict[str, Any]] = None
    justification: str = ""
    justification_quality: float = 0.0
    reward: float = 0.0
    was_correct: bool = False


# ── State ──────────────────────────────────────────────────────────────────────

class MisinfoCrisisState(BaseModel):
    """
    Full internal state of the simulation (server-side).

    Includes ground truth labels, campaigns, propagation data, trust dynamics.
    """
    task_id: str = ""
    difficulty: str = "easy"
    time_step: int = 0
    max_steps: int = 20
    done: bool = False

    # Posts
    posts: Dict[str, FullPost] = Field(default_factory=dict)
    posts_schedule: Dict[int, List[str]] = Field(
        default_factory=dict,
        description="Mapping of step → list of post_ids to introduce"
    )

    # Campaigns
    campaigns: Dict[str, Campaign] = Field(
        default_factory=dict,
        description="Adversarial campaigns (hidden from agent)"
    )

    # Intervention tracking
    intervention_history: List[InterventionRecord] = Field(default_factory=list)
    propagation_graph: Dict[str, List[str]] = Field(default_factory=dict)

    # Metrics
    total_misinformation_spread: float = 0.0
    total_true_content_censored: float = 0.0
    cumulative_reward: float = 0.0

    # Public reaction
    user_trust: float = Field(1.0, ge=0.0, le=1.0)
    backlash: float = Field(0.0, ge=0.0, le=1.0)

    # Resource budget
    action_budget_per_step: int = Field(
        3, description="Max non-ignore actions per step"
    )
    actions_this_step: int = 0

    # Delayed truth
    truth_reveal_schedule: Dict[int, List[str]] = Field(
        default_factory=dict,
        description="step → post_ids whose truth gets revealed"
    )
    revealed_truths: List[str] = Field(
        default_factory=list,
        description="Post IDs whose truth has been revealed so far"
    )

    # Seed
    seed: int = 42


# ── Step / Reset Results ───────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Result returned from a step() call."""
    observation: MisinfoCrisisObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result returned from a reset() call."""
    observation: MisinfoCrisisObservation
