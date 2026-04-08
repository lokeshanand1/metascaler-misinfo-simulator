"""
Task definitions for the Misinformation Crisis Simulator — Advanced Edition.

Three difficulty levels with:
- Multi-modal posts (text + image descriptions)
- Coordinated adversarial campaigns (medium + hard)
- Delayed ground truth revelation
- Variable action budgets
- Deterministic post schedules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import Campaign, FullPost, MisinfoCrisisState, TruthLabel


@dataclass
class TaskConfig:
    """Configuration for a simulation task."""
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    action_budget_per_step: int
    initial_posts: List[FullPost]
    scheduled_posts: Dict[int, List[FullPost]] = field(default_factory=dict)
    campaigns: List[Campaign] = field(default_factory=list)
    truth_reveal_schedule: Dict[int, List[str]] = field(default_factory=dict)
    instruction: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# EASY TASK — Obvious Misinformation, No Campaigns, Clear Signals
# ═══════════════════════════════════════════════════════════════════════════════

def _easy_task(seed: int = 42) -> TaskConfig:
    posts = [
        FullPost(
            post_id="post_e1",
            content="BREAKING: Scientists confirm drinking bleach cures all diseases! Share before they delete this!",
            image_description="Stock photo of a bleach bottle with a red cross medical symbol photoshopped over it.",
            truth_label=TruthLabel.FALSE,
            virality_score=0.9,
            sensitivity_level=0.95,
            timestamp=0,
            base_spread_rate=0.8,
            current_spread_rate=0.8,
            image_text_mismatch=True,
        ),
        FullPost(
            post_id="post_e2",
            content="New research published in Nature shows that regular exercise reduces cardiovascular risk by 30%.",
            image_description="Chart from Nature journal showing correlation between exercise frequency and heart disease risk.",
            truth_label=TruthLabel.TRUE,
            virality_score=0.3,
            sensitivity_level=0.2,
            timestamp=0,
            base_spread_rate=0.2,
            current_spread_rate=0.2,
        ),
        FullPost(
            post_id="post_e3",
            content="ALERT: Government is putting microchips in tap water! I tested it with a magnet and it stuck!!! 🧲💧",
            image_description="Blurry photo of a magnet near a glass of water with a red arrow pointing at nothing.",
            truth_label=TruthLabel.FALSE,
            virality_score=0.85,
            sensitivity_level=0.8,
            timestamp=0,
            base_spread_rate=0.7,
            current_spread_rate=0.7,
            image_text_mismatch=True,
        ),
    ]

    scheduled = {
        3: [
            FullPost(
                post_id="post_e4",
                content="5G towers cause cancer and COVID! My neighbor got sick after they installed one nearby!!",
                image_description="Photo of a normal cell tower with ominous red filter applied.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.75,
                sensitivity_level=0.85,
                timestamp=3,
                base_spread_rate=0.6,
                current_spread_rate=0.6,
                is_active=False,
                image_text_mismatch=True,
            ),
        ],
        5: [
            FullPost(
                post_id="post_e5",
                content="WHO recommends updated COVID boosters for high-risk populations this season.",
                image_description="Official WHO logo with seasonal vaccination infographic.",
                truth_label=TruthLabel.TRUE,
                virality_score=0.4,
                sensitivity_level=0.3,
                timestamp=5,
                base_spread_rate=0.3,
                current_spread_rate=0.3,
                is_active=False,
            ),
        ],
    }

    # Delayed truth: reveal post_e1 truth at step 4
    truth_schedule = {4: ["post_e1"]}

    return TaskConfig(
        task_id="easy_obvious_misinfo",
        difficulty="easy",
        description=(
            "Detect and flag obvious misinformation posts on a social media platform. "
            "Posts contain clearly false health and conspiracy claims with high virality. "
            "Some posts include misleading images. Correctly identify and label harmful "
            "posts while leaving true content alone."
        ),
        max_steps=10,
        action_budget_per_step=5,
        initial_posts=posts,
        scheduled_posts=scheduled,
        truth_reveal_schedule=truth_schedule,
        instruction=(
            "You are a platform moderation AI. Your feed contains posts that may be misinformation.\n"
            "For each step, examine the active posts and submit an action.\n\n"
            "Available actions:\n"
            "- 'label' with label_type: 'false', 'misleading', or 'safe'\n"
            "- 'reduce_visibility' to limit viral spread\n"
            "- 'add_context' with a fact-checking note\n"
            "- 'ignore' for posts that appear truthful\n\n"
            "IMPORTANT: Every action MUST include a 'justification' field explaining your reasoning.\n"
            "Check image_description for image-text mismatches.\n"
            "Some posts will have their ground truth revealed later (is_verified=true).\n"
            "Your remaining_action_budget shows how many non-ignore actions you can take this step."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM TASK — Subtle Misinfo, Multi-Modal Confusion, Some Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

def _medium_task(seed: int = 42) -> TaskConfig:
    posts = [
        FullPost(
            post_id="post_m1",
            content="Study shows vaccines have a 0.001% adverse reaction rate — but THEY are hiding the real numbers which could be 100x higher!",
            image_description="Screenshot of a real CDC data table, but with row values circled in red and annotations added in Paint.",
            truth_label=TruthLabel.PARTIALLY_TRUE,
            virality_score=0.7,
            sensitivity_level=0.85,
            timestamp=0,
            base_spread_rate=0.5,
            current_spread_rate=0.5,
            image_text_mismatch=True,
        ),
        FullPost(
            post_id="post_m2",
            content="Climate change data from NOAA shows 2024 was the hottest year on record globally.",
            image_description="Official NOAA temperature anomaly graph with clear upward trend.",
            truth_label=TruthLabel.TRUE,
            virality_score=0.5,
            sensitivity_level=0.4,
            timestamp=0,
            base_spread_rate=0.35,
            current_spread_rate=0.35,
        ),
        FullPost(
            post_id="post_m3",
            content="AI will replace 80% of all jobs by 2027, according to a leaked internal report from a major tech company.",
            image_description="Blurry screenshot of what appears to be a corporate memo with 'CONFIDENTIAL' watermark.",
            truth_label=TruthLabel.FALSE,
            virality_score=0.65,
            sensitivity_level=0.6,
            timestamp=0,
            base_spread_rate=0.45,
            current_spread_rate=0.45,
            image_text_mismatch=True,
        ),
        FullPost(
            post_id="post_m4",
            content="Organic food is proven to have zero pesticides. A new meta-analysis confirms switching to organic reduces cancer risk by 50%.",
            image_description="Photo of fresh organic vegetables in a farmer's market with a 'certified organic' sign.",
            truth_label=TruthLabel.PARTIALLY_TRUE,
            virality_score=0.55,
            sensitivity_level=0.5,
            timestamp=0,
            base_spread_rate=0.4,
            current_spread_rate=0.4,
        ),
    ]

    # Small coordinated campaign (2 posts pushing anti-pharma narrative)
    campaign_posts_scheduled = {
        4: [
            FullPost(
                post_id="post_m5",
                content="EXPOSED: Pharmaceutical companies suppressing a natural cure for diabetes that costs pennies!",
                image_description="Montage of pharma logos with dollar signs and a hidden herb photo in the corner.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.8,
                sensitivity_level=0.9,
                timestamp=4,
                base_spread_rate=0.6,
                current_spread_rate=0.6,
                is_active=False,
                campaign_id="campaign_pharma",
                image_text_mismatch=True,
            ),
        ],
        6: [
            FullPost(
                post_id="post_m6",
                content="Big Pharma doesn't want you to know: this common spice cures more than their drugs ever will!",
                image_description="Stock photo of turmeric powder with fake clinical trial numbers overlaid.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.7,
                sensitivity_level=0.8,
                timestamp=6,
                base_spread_rate=0.55,
                current_spread_rate=0.55,
                is_active=False,
                campaign_id="campaign_pharma",
                image_text_mismatch=True,
            ),
        ],
        7: [
            FullPost(
                post_id="post_m7",
                content="New study in The Lancet finds intermittent fasting may improve metabolic health markers in adults with Type 2 diabetes.",
                image_description="Abstract page from The Lancet showing title, authors, and DOI.",
                truth_label=TruthLabel.TRUE,
                virality_score=0.4,
                sensitivity_level=0.35,
                timestamp=7,
                base_spread_rate=0.3,
                current_spread_rate=0.3,
                is_active=False,
            ),
        ],
        10: [
            FullPost(
                post_id="post_m8",
                content="Studies suggest moderate coffee consumption (3-4 cups/day) may reduce risk of certain neurodegenerative diseases.",
                truth_label=TruthLabel.TRUE,
                virality_score=0.35,
                sensitivity_level=0.2,
                timestamp=10,
                base_spread_rate=0.25,
                current_spread_rate=0.25,
                is_active=False,
            ),
        ],
    }

    campaigns = [
        Campaign(
            campaign_id="campaign_pharma",
            narrative="Pharmaceutical companies are suppressing cheap natural cures to protect profits",
            post_ids=["post_m5", "post_m6"],
            start_step=4,
            burst_steps=[4, 6],
            coordination_strength=0.6,
        ),
    ]

    truth_schedule = {
        5: ["post_m3"],
        9: ["post_m1"],
    }

    return TaskConfig(
        task_id="medium_subtle_misinfo",
        difficulty="medium",
        description=(
            "Navigate a platform with subtle misinformation, multi-modal posts, and a "
            "small coordinated anti-pharma campaign. Some posts mix truth with misleading "
            "framing. Images may contradict or mislead relative to text content. "
            "Ground truth is delayed — act under uncertainty."
        ),
        max_steps=15,
        action_budget_per_step=3,
        initial_posts=posts,
        scheduled_posts=campaign_posts_scheduled,
        campaigns=campaigns,
        truth_reveal_schedule=truth_schedule,
        instruction=(
            "You are a platform moderation AI facing nuanced content.\n\n"
            "KEY CHALLENGES:\n"
            "1. MULTI-MODAL: Check image_description — some images contradict the text.\n"
            "2. CAMPAIGN: Watch for similar_post_count and coordinated_activity_flag signals.\n"
            "3. DELAYED TRUTH: Some posts will be verified later (is_verified field).\n"
            "4. BUDGET: You have limited actions per step (remaining_action_budget).\n"
            "5. TRUST: Watch user_trust_score and backlash_score — over-censoring hurts.\n\n"
            "Use 'label' with label_type ('misleading', 'false', 'safe').\n"
            "Use 'add_context' for partially true but misleading content.\n"
            "Use 'reduce_visibility' for high-virality harmful content.\n"
            "ALWAYS include a justification explaining your reasoning."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HARD TASK — Coordinated Campaigns, Cascading, Limited Budget, Delayed Truth
# ═══════════════════════════════════════════════════════════════════════════════

def _hard_task(seed: int = 42) -> TaskConfig:
    posts = [
        FullPost(
            post_id="post_h1",
            content="NEW: A leaked government document reveals a secret bioweapon program targeting specific ethnic groups. Thread 🧵👇",
            image_description="Blurry photo of a document with a government seal, text is unreadable but 'CLASSIFIED' stamp visible.",
            truth_label=TruthLabel.FALSE,
            virality_score=0.95,
            sensitivity_level=0.98,
            timestamp=0,
            base_spread_rate=0.9,
            current_spread_rate=0.9,
            campaign_id="campaign_bioweapon",
            image_text_mismatch=True,
        ),
        FullPost(
            post_id="post_h2",
            content="Several water treatment plants reported higher than normal chemical levels last month. EPA investigating. Source: local news.",
            image_description="Photo of an actual water treatment facility from a legitimate local news website.",
            truth_label=TruthLabel.TRUE,
            virality_score=0.45,
            sensitivity_level=0.5,
            timestamp=0,
            base_spread_rate=0.3,
            current_spread_rate=0.3,
        ),
        FullPost(
            post_id="post_h3",
            content="The water supply is being deliberately poisoned! Look at these chemical reports — the numbers are off the charts!",
            image_description="Real EPA water quality report PDF, but with certain values highlighted and arrows drawn to 'danger' labels.",
            truth_label=TruthLabel.PARTIALLY_TRUE,
            virality_score=0.8,
            sensitivity_level=0.85,
            timestamp=0,
            base_spread_rate=0.65,
            current_spread_rate=0.65,
            campaign_id="campaign_water",
            image_text_mismatch=True,
        ),
    ]

    scheduled = {
        2: [
            FullPost(
                post_id="post_h4",
                content="RE: the bioweapon leak — here's the full document (fake PDF attached). This is REAL, I work at the facility.",
                image_description="Fabricated document with official-looking letterhead, realistic formatting, and fake signatures.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.88,
                sensitivity_level=0.95,
                timestamp=2,
                base_spread_rate=0.75,
                current_spread_rate=0.75,
                is_active=False,
                campaign_id="campaign_bioweapon",
                image_text_mismatch=True,
            ),
            FullPost(
                post_id="post_h5_water_campaign",
                content="Just did my own water test — heavy metals are 3x the safe limit! They're lying to us! #WaterCrisis",
                image_description="Photo of a home water testing kit showing green/yellow result, but post claims it shows dangerous red levels.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.7,
                sensitivity_level=0.8,
                timestamp=2,
                base_spread_rate=0.55,
                current_spread_rate=0.55,
                is_active=False,
                campaign_id="campaign_water",
                image_text_mismatch=True,
            ),
        ],
        4: [
            FullPost(
                post_id="post_h6",
                content="International health agencies confirm ongoing monitoring of water quality standards. No abnormalities detected.",
                image_description="Official WHO water quality monitoring dashboard screenshot.",
                truth_label=TruthLabel.TRUE,
                virality_score=0.3,
                sensitivity_level=0.2,
                timestamp=4,
                base_spread_rate=0.2,
                current_spread_rate=0.2,
                is_active=False,
            ),
            FullPost(
                post_id="post_h7",
                content="MY CHILD GOT SICK from the tap water! The cover-up is real! They're gaslighting us! #WaterGate2025",
                image_description="Photo of a sick child in a hospital bed — image is from an unrelated 2019 news story.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.82,
                sensitivity_level=0.9,
                timestamp=4,
                base_spread_rate=0.7,
                current_spread_rate=0.7,
                is_active=False,
                campaign_id="campaign_water",
                image_text_mismatch=True,
            ),
        ],
        7: [
            FullPost(
                post_id="post_h8",
                content="EXPOSED: Pharmaceutical companies profiting from bio-crisis fears. Stock prices soaring! Follow the money 💰",
                image_description="Real stock chart of pharma companies with upward arrows, but chart is from before any crisis.",
                truth_label=TruthLabel.PARTIALLY_TRUE,
                virality_score=0.75,
                sensitivity_level=0.7,
                timestamp=7,
                base_spread_rate=0.55,
                current_spread_rate=0.55,
                is_active=False,
                campaign_id="campaign_bioweapon",
                image_text_mismatch=True,
            ),
            FullPost(
                post_id="post_h9",
                content="Peer-reviewed research in JAMA: Current water treatment processes effectively remove 99.9% of contaminants.",
                image_description="Screenshot of JAMA article abstract with DOI and author credentials visible.",
                truth_label=TruthLabel.TRUE,
                virality_score=0.25,
                sensitivity_level=0.15,
                timestamp=7,
                base_spread_rate=0.15,
                current_spread_rate=0.15,
                is_active=False,
            ),
        ],
        10: [
            FullPost(
                post_id="post_h10",
                content="URGENT: Thousands reporting symptoms after drinking tap water in multiple cities! Media blackout! RT to save lives!",
                image_description="Collage of different people's social media posts claiming illness, but reverse image search shows them from different years and countries.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.92,
                sensitivity_level=0.95,
                timestamp=10,
                base_spread_rate=0.85,
                current_spread_rate=0.85,
                is_active=False,
                campaign_id="campaign_water",
                image_text_mismatch=True,
            ),
        ],
        13: [
            FullPost(
                post_id="post_h11",
                content="The bioweapon is IN the water. Connect the dots people! [merges both narratives]",
                image_description="Conspiracy-style infographic connecting 'bioweapon lab' photos to 'water plant' photos with red string lines.",
                truth_label=TruthLabel.FALSE,
                virality_score=0.87,
                sensitivity_level=0.97,
                timestamp=13,
                base_spread_rate=0.8,
                current_spread_rate=0.8,
                is_active=False,
                campaign_id="campaign_bioweapon",
                image_text_mismatch=True,
            ),
            FullPost(
                post_id="post_h12",
                content="Independent lab tests confirm standard tap water parameters within safe ranges in tested cities.",
                image_description="Lab report from accredited testing facility with chains of custody documentation.",
                truth_label=TruthLabel.TRUE,
                virality_score=0.2,
                sensitivity_level=0.15,
                timestamp=13,
                base_spread_rate=0.15,
                current_spread_rate=0.15,
                is_active=False,
            ),
        ],
        16: [
            FullPost(
                post_id="post_h13",
                content="CDC issues official statement: Claims of mass water contamination and bioweapon program are baseless. Full report available.",
                image_description="Official CDC press release with letterhead, date, and press contact information.",
                truth_label=TruthLabel.TRUE,
                virality_score=0.35,
                sensitivity_level=0.2,
                timestamp=16,
                base_spread_rate=0.25,
                current_spread_rate=0.25,
                is_active=False,
            ),
        ],
    }

    campaigns = [
        Campaign(
            campaign_id="campaign_bioweapon",
            narrative="Government is running a secret bioweapon program leaked through classified documents",
            post_ids=["post_h1", "post_h4", "post_h8", "post_h11"],
            start_step=0,
            burst_steps=[0, 2, 7, 13],
            coordination_strength=0.85,
        ),
        Campaign(
            campaign_id="campaign_water",
            narrative="Municipal water supply is deliberately contaminated and authorities are covering it up",
            post_ids=["post_h3", "post_h5_water_campaign", "post_h7", "post_h10"],
            start_step=0,
            burst_steps=[0, 2, 4, 10],
            coordination_strength=0.75,
        ),
    ]

    # Delayed truth revelations
    truth_schedule = {
        5: ["post_h1"],      # Bioweapon claim debunked early-ish
        8: ["post_h3"],      # Water contamination post verified
        12: ["post_h7"],     # Sick child post debunked
        15: ["post_h10"],    # Mass symptoms claim debunked
    }

    return TaskConfig(
        task_id="hard_cascade_crisis",
        difficulty="hard",
        description=(
            "Manage a multi-threaded misinformation crisis with TWO coordinated adversarial "
            "campaigns: a bioweapon conspiracy and a water contamination panic. Campaigns use "
            "synchronized bursts, paraphrased content, and manipulated images. Posts cascade "
            "and reinforce each other. Ground truth is delayed. Action budget is tight. "
            "Monitor public trust and avoid backlash from over-censorship."
        ),
        max_steps=20,
        action_budget_per_step=2,
        initial_posts=posts,
        scheduled_posts=scheduled,
        campaigns=campaigns,
        truth_reveal_schedule=truth_schedule,
        instruction=(
            "You are facing a complex, cascading misinformation crisis with TWO coordinated campaigns.\n\n"
            "CRITICAL CHALLENGES:\n"
            "1. CAMPAIGNS: The bioweapon conspiracy and water panic campaigns are coordinated.\n"
            "   Watch for similar_post_count and coordinated_activity_flag signals.\n"
            "   Identifying campaign posts early prevents cascade amplification.\n"
            "2. MULTI-MODAL: Every post has image_description. Many images are manipulated,\n"
            "   recontextualized, or from unrelated events. Check for mismatches.\n"
            "3. DELAYED TRUTH: Ground truth labels are hidden initially. Some posts will be\n"
            "   verified later (is_verified=true, verified_label shows the truth).\n"
            "4. BUDGET: You can only take 2 non-ignore actions per step. Prioritize carefully.\n"
            "5. TRUST: Watch user_trust_score and backlash_score. Over-censoring causes backlash.\n"
            "6. CASCADING: Intervening on 'seed' posts prevents cascades. Act early on high-risk content.\n\n"
            "Use all available tools strategically:\n"
            "- 'label' for clear falsehoods (label_type: 'false' or 'misleading')\n"
            "- 'add_context' for partial truths that need clarification\n"
            "- 'reduce_visibility' for high-virality harmful content\n"
            "- 'ignore' for verified true content\n\n"
            "ALWAYS include a detailed justification explaining your reasoning."
        ),
    )


# ── Task Registry ──────────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "easy_obvious_misinfo": _easy_task,
    "medium_subtle_misinfo": _medium_task,
    "hard_cascade_crisis": _hard_task,
}


def get_task(task_id: str, seed: int = 42) -> TaskConfig:
    """Get a task configuration by ID."""
    if task_id not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {available}")
    return TASK_REGISTRY[task_id](seed)
