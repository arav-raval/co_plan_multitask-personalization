"""
layouts.py — Kitchen layout definitions for the Overcooked environment.

Analogous to recipes.py in the spices environment.  Each layout wraps an
overcooked_ai layout name and defines the set of subtasks that the robot can
be assigned to perform on behalf of (or alongside) the human.

Overcooked workflow
-------------------
The game has one type of recipe: soup, made from 1–3 ingredients (onions
and/or tomatoes).  The full workflow per delivery is:

  1. fetch_ingredient  — pick up an ingredient (onion or tomato) from a
                         dispenser (terrain 'O' or 'T')
  2. load_pot          — walk to a pot ('P') and place the ingredient in it;
                         once enough ingredients are loaded the pot starts
                         cooking automatically
  3. fetch_dish        — pick up a plate from the dish dispenser ('D')
  4. pickup_soup       — once cooking is done, use the dish to scoop the soup
                         out of the pot
  5. deliver           — walk to the serving counter ('S') to score a point

These 5 steps are the preference dimensions fed to the HBM.  The HBM learns
one phi per (layout, subtask) pair, so the robot learns which agent the human
prefers to handle each step.

Note on tomatoes
----------------
Most canonical layouts (cramped_room, asymmetric_advantages, coordination_ring,
forced_coordination) contain only onion dispensers ('O') — no 'T' tiles.
Tomatoes appear only in dedicated variants (e.g. asymmetric_advantages_tomato,
simple_tomato).  Because both ingredients involve the same fetch_ingredient
and load_pot steps, the 5-subtask vocabulary above covers all layouts without
needing a separate fetch_tomato dimension.  If you add a mixed-ingredient
layout, add it to the registry with the same subtask list — the HBM
automatically learns the preference for that layout's ingredient mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Canonical subtask vocabulary
# ---------------------------------------------------------------------------

ALL_SUBTASKS: list[str] = [
    "fetch_ingredient",  # pick up onion/tomato from dispenser (O or T)
    "load_pot",          # place ingredient into pot (P); triggers cooking
    "fetch_dish",        # pick up plate from dish dispenser (D)
    "pickup_soup",       # scoop cooked soup from pot into held dish
    "deliver",           # carry plated soup to serving location (S)
]


# ---------------------------------------------------------------------------
# Layout specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutSpec:
    """Specification for one Overcooked kitchen layout.

    Parameters
    ----------
    name:
        Human-readable identifier used as the "recipe" key in the HBM.
    layout_name:
        Name passed to ``OvercookedGridworld.from_layout_name()``.
    subtasks:
        Ordered list of subtasks valid in this layout.  These are the
        preference dimensions the HBM will learn for this layout.
        Defaults to ALL_SUBTASKS (all 5 steps).
    episode_length:
        Default horizon (timesteps) for this layout.
    description:
        Free-text note for documentation.
    """

    name: str
    layout_name: str
    subtasks: list[str] = field(default_factory=lambda: list(ALL_SUBTASKS))
    episode_length: int = 400
    description: str = ""

    def __post_init__(self) -> None:
        for s in self.subtasks:
            if s not in ALL_SUBTASKS:
                raise ValueError(
                    f"Unknown subtask '{s}' in layout '{self.name}'. "
                    f"Valid subtasks: {ALL_SUBTASKS}"
                )


# ---------------------------------------------------------------------------
# Predefined layouts
# ---------------------------------------------------------------------------

# All 5 canonical layouts use only onion dispensers; no fetch_tomato subtask
# needed.  All share the same 5-step workflow and therefore the same subtask
# list.  The descriptions explain the spatial challenge each layout poses, which
# drives *which* subtask preferences the robot needs to learn fastest.

CRAMPED_ROOM = LayoutSpec(
    name="CrampedRoom",
    layout_name="cramped_room",
    episode_length=400,
    description=(
        "Tight 5×4 grid with one pot and two onion dispensers on opposite "
        "sides.  Counter space is scarce so agents collide frequently — the "
        "key preference is who handles prep (fetch+load) vs. serve (dish+deliver)."
    ),
)

ASYMMETRIC_ADVANTAGES = LayoutSpec(
    name="AsymmetricAdvantages",
    layout_name="asymmetric_advantages",
    episode_length=400,
    description=(
        "9×5 grid where each agent starts near different resources.  Spatial "
        "asymmetry makes stable role specialisation (one agent preps, one "
        "serves) the most efficient strategy."
    ),
)

COORDINATION_RING = LayoutSpec(
    name="CoordinationRing",
    layout_name="coordination_ring",
    episode_length=400,
    description=(
        "Ring-shaped layout; agents must move in the same rotational direction "
        "to avoid blocking each other.  Reveals preferences about who leads "
        "the fetch-load loop vs. who handles the serve loop."
    ),
)

FORCED_COORDINATION = LayoutSpec(
    name="ForcedCoordination",
    layout_name="forced_coordination",
    episode_length=400,
    description=(
        "Layout where one agent cannot reach the serving counter without "
        "passing through the other agent's space.  Forces handoffs and reveals "
        "preferences about who initiates vs. completes deliveries."
    ),
)

COUNTER_CIRCUIT = LayoutSpec(
    name="CounterCircuit",
    layout_name="counter_circuit",
    episode_length=400,
    description=(
        "Counter-heavy layout where items are passed via shared counters.  "
        "Tests preferences about who fetches ingredients and who picks items "
        "up from the counter to continue the workflow."
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LAYOUT_REGISTRY: dict[str, LayoutSpec] = {
    spec.name: spec
    for spec in [
        CRAMPED_ROOM,
        ASYMMETRIC_ADVANTAGES,
        COORDINATION_RING,
        FORCED_COORDINATION,
        COUNTER_CIRCUIT,
    ]
}


def get_layout(name: str) -> LayoutSpec:
    """Return a LayoutSpec by name."""
    if name not in _LAYOUT_REGISTRY:
        raise KeyError(
            f"Unknown layout '{name}'. Available: {list(_LAYOUT_REGISTRY)}"
        )
    return _LAYOUT_REGISTRY[name]


def list_layouts() -> list[str]:
    """Return sorted list of registered layout names."""
    return sorted(_LAYOUT_REGISTRY)
