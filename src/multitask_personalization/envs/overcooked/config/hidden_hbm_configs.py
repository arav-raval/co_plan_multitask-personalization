"""
hidden_hbm_configs.py — Ground-truth human preference configurations for Overcooked.

Each configuration defines the true theta_mean per subtask for a simulated human.
Positive values mean the human prefers to do that subtask themselves; negative
means they prefer the robot to handle it.

Mirrors spices/config/hidden_hbm_configs.py but with Overcooked subtask vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------

def _prefers_prep(subtasks: list[str]) -> dict[str, float]:
    """Human prefers prep tasks (fetch + load); robot handles serve loop."""
    prep = {"fetch_ingredient", "load_pot"}
    return {s: +1.5 if s in prep else -1.5 for s in subtasks}


def _prefers_delivery(subtasks: list[str]) -> dict[str, float]:
    """Human prefers the serve loop (dish + pickup + deliver); robot preps."""
    serve = {"fetch_dish", "pickup_soup", "deliver"}
    return {s: +1.5 if s in serve else -1.5 for s in subtasks}


def _prefers_all(subtasks: list[str]) -> dict[str, float]:
    """Human wants to do everything (+2.0 across all subtasks)."""
    return {s: +2.0 for s in subtasks}


def _prefers_none(subtasks: list[str]) -> dict[str, float]:
    """Human defers everything to the robot (-2.0 across all subtasks)."""
    return {s: -2.0 for s in subtasks}


def _alternating(subtasks: list[str]) -> dict[str, float]:
    """Alternating pattern: +1.5 for even-indexed, -1.5 for odd-indexed."""
    return {s: 1.5 if i % 2 == 0 else -1.5 for i, s in enumerate(subtasks)}


# Semantically grounded mapping for a realistic simulated human.
# This human enjoys active prep work but dislikes the navigational serve loop.
_REALISTIC_THETA: dict[str, float] = {
    "fetch_ingredient": +1.0,   # enjoys fetching ingredients
    "load_pot":         +1.5,   # strongly prefers loading the pot (active)
    "fetch_dish":       +0.5,   # mild preference for grabbing dishes
    "pickup_soup":      -0.5,   # mild aversion (timing-critical, easy to drop)
    "deliver":          -1.5,   # dislikes delivery (navigation, time pressure)
}

_ROBOT_CENTRIC_THETA: dict[str, float] = {
    "fetch_ingredient": -1.0,
    "load_pot":         -1.0,
    "fetch_dish":       -1.5,
    "pickup_soup":      +0.5,
    "deliver":          +1.5,   # human wants to do the final delivery
}


def _realistic_theta(subtasks: list[str]) -> dict[str, float]:
    return {s: _REALISTIC_THETA.get(s, 0.0) for s in subtasks}


def _robot_centric_theta(subtasks: list[str]) -> dict[str, float]:
    return {s: _ROBOT_CENTRIC_THETA.get(s, 0.0) for s in subtasks}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HiddenHBMConfig:
    """Defines the ground-truth preferences for one simulated human."""

    name: str
    theta_mean: dict[str, float] = field(default_factory=dict)
    sigma0: float = 1.0
    sigma_h: float = 0.5
    sigma_r: float = 0.5
    sigma_obs: float = 1.0
    theta_generator: Optional[Callable[[list[str]], dict[str, float]]] = None

    def get_theta_for_subtask(self, subtask: str) -> float:
        if self.theta_generator is not None:
            raise RuntimeError(
                "Use generate_theta(subtasks) when theta_generator is set."
            )
        return self.theta_mean.get(subtask, 0.0)

    def generate_theta(self, subtasks: list[str]) -> dict[str, float]:
        if self.theta_generator is not None:
            return self.theta_generator(subtasks)
        return {s: self.theta_mean.get(s, 0.0) for s in subtasks}


# ---------------------------------------------------------------------------
# Pre-defined configurations
# ---------------------------------------------------------------------------

# Prefers handling prep work; robot delivers.
PREFERS_PREP = HiddenHBMConfig(
    name="PrefsPrep",
    theta_generator=_prefers_prep,
    sigma_h=0.5,
    sigma_r=0.4,
)

# Prefers delivery and plating; robot does ingredient work.
PREFERS_DELIVERY = HiddenHBMConfig(
    name="PrefsDelivery",
    theta_generator=_prefers_delivery,
    sigma_h=0.5,
    sigma_r=0.4,
)

# Wants to do everything — robot should stay out of the way.
PREFERS_ALL = HiddenHBMConfig(
    name="PrefsAll",
    theta_generator=_prefers_all,
    sigma_h=0.3,
    sigma_r=0.3,
)

# Hands off everything to the robot.
PREFERS_NONE = HiddenHBMConfig(
    name="PrefsNone",
    theta_generator=_prefers_none,
    sigma_h=0.3,
    sigma_r=0.3,
)

# Alternating — useful for learning tests.
ALTERNATING = HiddenHBMConfig(
    name="Alternating",
    theta_generator=_alternating,
    sigma_h=0.5,
    sigma_r=0.5,
)

# Realistic home cook: likes prep + chopping, dislikes delivery.
REALISTIC_COOK = HiddenHBMConfig(
    name="RealisticCook",
    theta_generator=_realistic_theta,
    sigma_h=0.5,
    sigma_r=0.5,
)

# Robot-centric: lets robot do everything except final delivery.
ROBOT_CENTRIC = HiddenHBMConfig(
    name="RobotCentric",
    theta_generator=_robot_centric_theta,
    sigma_h=0.5,
    sigma_r=0.5,
)

# High variance — hard learning problem.
VARIABLE_HUMAN = HiddenHBMConfig(
    name="VariableHuman",
    theta_generator=_realistic_theta,
    sigma_h=1.5,
    sigma_r=2.0,
)

# Low variance — easy reference case.
CONSISTENT_HUMAN = HiddenHBMConfig(
    name="ConsistentHuman",
    theta_generator=_realistic_theta,
    sigma_h=0.1,
    sigma_r=0.1,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, HiddenHBMConfig] = {
    cfg.name: cfg
    for cfg in [
        PREFERS_PREP,
        PREFERS_DELIVERY,
        PREFERS_ALL,
        PREFERS_NONE,
        ALTERNATING,
        REALISTIC_COOK,
        ROBOT_CENTRIC,
        VARIABLE_HUMAN,
        CONSISTENT_HUMAN,
    ]
}


def get_hidden_hbm_config(name: str) -> HiddenHBMConfig:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown hidden HBM config '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_hidden_hbm_configs() -> list[str]:
    return sorted(_REGISTRY)
