"""
session_profiles.py — Named session-effect profiles for Overcooked.

Each profile defines per-subtask sensitivity weights that control how strongly
non-neutral sessions (fatigue / energy) affect each subtask's claiming probability.

These are *environment-level* parameters (how does fatigue physically manifest),
not human preferences (what does the human like).  A human who prefers fetch_ingredient
(theta > 0) still gets fatigued on it if the profile says so — the session effect
is additive to the preference.

Usage in experiments:
    The session profile name can be set in the Hydra config to test different
    fatigue/energy patterns and demonstrate that vector psi captures them better
    than scalar psi.

Weight semantics:
    weight > 1.0 → this subtask is MORE affected by fatigue/energy than baseline
    weight = 1.0 → baseline session effect (same as uniform/scalar psi)
    weight < 1.0 → this subtask is LESS affected
    weight = 0.0 → this subtask is immune to session effects

    psi_true[d] ~ N(weight[d] * ±mean_abs, std²)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ..layouts import ALL_SUBTASKS


@dataclass(frozen=True)
class SessionProfile:
    """Named session-effect profile."""

    name: str
    description: str
    # Per-subtask weights, order matches ALL_SUBTASKS.
    weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)

    def get_weight(self, subtask: str) -> float:
        """Return the session weight for a given subtask."""
        try:
            idx = ALL_SUBTASKS.index(subtask)
            return self.weights[idx] if idx < len(self.weights) else 1.0
        except ValueError:
            # Subtask not in ALL_SUBTASKS (e.g., fetch_onion, fetch_tomato).
            # Map to fetch_ingredient's weight as a reasonable default.
            if subtask in ("fetch_onion", "fetch_tomato"):
                try:
                    idx = ALL_SUBTASKS.index("fetch_ingredient")
                    return self.weights[idx] if idx < len(self.weights) else 1.0
                except ValueError:
                    pass
            return 1.0
        except ValueError:
            return 1.0


# ---------------------------------------------------------------------------
# Pre-defined profiles
# ---------------------------------------------------------------------------

# Default: physically demanding tasks are more affected by fatigue.
PHYSICAL_FATIGUE = SessionProfile(
    name="PhysicalFatigue",
    description=(
        "Fatigue scales with physical effort. Heavy tasks (load_pot, "
        "fetch_ingredient) are most affected; light tasks (deliver) least."
    ),
    # Order: fetch_ingredient, load_pot, fetch_dish, pickup_soup, deliver,
    #        place_on_counter, pickup_from_counter
    weights=(
        1.3,   # fetch_ingredient — walk to dispenser + grab
        1.5,   # load_pot — carry ingredient + precision place
        0.8,   # fetch_dish — simple pickup
        0.6,   # pickup_soup — timing-dependent, stationary
        0.4,   # deliver — simple walk to counter
        0.7,   # place_on_counter — short walk to counter
        0.7,   # pickup_from_counter — short walk to counter
    ),
)

TIRED_OF_WALKING = SessionProfile(
    name="TiredOfWalking",
    description=(
        "Fatigue targets navigation. The human hangs near the pot area and "
        "prefers stationary work (load_pot, pickup_soup). Fetching ingredients "
        "and delivering are exhausting."
    ),
    weights=(
        1.8,   # fetch_ingredient — long walk to dispenser
        0.3,   # load_pot — stationary at pot, barely affected
        1.2,   # fetch_dish — moderate walk to dish dispenser
        0.3,   # pickup_soup — stationary at pot
        1.6,   # deliver — long walk to serving counter
        0.5,   # place_on_counter — short walk
        0.5,   # pickup_from_counter — short walk
    ),
)

UNIFORM = SessionProfile(
    name="Uniform",
    description=(
        "All subtasks equally affected by session state. "
        "Equivalent to scalar psi — used as baseline."
    ),
    weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)

MENTAL_FATIGUE = SessionProfile(
    name="MentalFatigue",
    description=(
        "Fatigue targets cognitively demanding tasks. Pickup_soup (timing) "
        "and load_pot (coordination) are most affected; simple fetch/deliver less."
    ),
    weights=(
        0.6,   # fetch_ingredient — simple, low cognitive load
        1.4,   # load_pot — requires counting ingredients
        0.5,   # fetch_dish — simple
        1.8,   # pickup_soup — timing-critical (must wait for cooking)
        0.4,   # deliver — simple navigation
        0.5,   # place_on_counter — low cognitive load
        0.5,   # pickup_from_counter — low cognitive load
    ),
)

STRONG_DIFFERENTIATION = SessionProfile(
    name="StrongDifferentiation",
    description=(
        "Extreme weight spread for ablation. Some subtasks are heavily "
        "affected (2.0), others barely (0.2). Maximises vector psi advantage."
    ),
    weights=(
        2.0,   # fetch_ingredient — heavily affected
        0.2,   # load_pot — barely affected
        1.5,   # fetch_dish — moderately affected
        0.2,   # pickup_soup — barely affected
        1.8,   # deliver — heavily affected
        1.0,   # place_on_counter
        1.0,   # pickup_from_counter
    ),
)

EXTREME_ASYMMETRIC = SessionProfile(
    name="ExtremeAsymmetric",
    description=(
        "Opposite-sign session effects per subtask, tuned so the scalar "
        "optimum is exactly zero (weights sum to 0). Uses larger magnitudes "
        "than AsymmetricFatigue so each subtask's effect saturates the "
        "sigmoid when paired with a larger session_nonneutral_mean_abs. "
        "Designed for the scalar-vs-vector-psi ablation: training should "
        "use neutral-dominant sessions to let phi learn cleanly, and eval "
        "should use force_neutral_mood=False with a low prob_neutral_session "
        "so ψ is exercised at test time."
    ),
    weights=(
        +2.0,   # fetch_ingredient — strong avoidance when fatigued
        -2.0,   # load_pot — strong eagerness (stationary)
        +2.0,   # fetch_dish — avoidance
        -2.0,   # pickup_soup — eagerness (stationary, timing)
        +2.0,   # deliver — avoidance (long walk)
        -1.0,   # place_on_counter — mild eagerness
        -1.0,   # pickup_from_counter — mild eagerness
    ),  # sum = 0.0 → scalar optimum is exactly 0
)


ASYMMETRIC_FATIGUE = SessionProfile(
    name="AsymmetricFatigue",
    description=(
        "Opposite-sign session effects per subtask. Fatigue makes the human "
        "avoid navigation-heavy subtasks but paradoxically makes them MORE "
        "eager to do stationary tasks (they want to rest in one spot). "
        "Negative weights flip the sign relative to the session type. This "
        "cannot be captured by scalar psi: a single shared offset cannot have "
        "opposite effects on different subtasks. Maximises vector psi advantage."
    ),
    weights=(
        1.8,   # fetch_ingredient — avoidance (fatigue -> don't want to walk)
        -1.5,  # load_pot — stationary, MORE eager when fatigued
        1.5,   # fetch_dish — avoidance
        -1.2,  # pickup_soup — stationary, MORE eager
        1.8,   # deliver — avoidance (long walk)
        -0.8,  # place_on_counter — short stationary, mildly MORE eager
        -0.8,  # pickup_from_counter — short stationary, mildly MORE eager
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, SessionProfile] = {
    p.name: p
    for p in [
        PHYSICAL_FATIGUE,
        TIRED_OF_WALKING,
        UNIFORM,
        MENTAL_FATIGUE,
        STRONG_DIFFERENTIATION,
        ASYMMETRIC_FATIGUE,
        EXTREME_ASYMMETRIC,
    ]
}


def get_session_profile(name: str) -> SessionProfile:
    """Return a SessionProfile by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown session profile '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_session_profiles() -> list[str]:
    """Return sorted list of registered profile names."""
    return sorted(_REGISTRY)
