"""
layouts.py — Kitchen layout definitions for the Overcooked environment.

Analogous to recipes.py in the spices environment.  Each layout wraps an
overcooked_ai layout name and defines the set of subtasks that the robot can
be assigned to perform on behalf of (or alongside) the human.

Subtasks are the "preference dimensions" fed to the HBM, corresponding to
high-level actions in Overcooked:
  - fetch_onion   : retrieve onion from dispenser
  - fetch_tomato  : retrieve tomato from dispenser
  - fetch_dish    : retrieve plate/dish from dispenser
  - chop          : place ingredient in pot (triggers cooking)
  - plate_soup    : transfer cooked soup onto dish
  - deliver       : carry plated soup to serving counter

Not every layout supports every subtask — e.g. some layouts have no tomato
dispenser.  Each LayoutSpec declares which subtasks are valid.

The HBM uses one phi per (layout, subtask) pair, mirroring the
(recipe, spice) hierarchy in the spices environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Canonical subtask vocabulary
# ---------------------------------------------------------------------------

ALL_SUBTASKS: list[str] = [
    "fetch_onion",
    "fetch_tomato",
    "fetch_dish",
    "chop",        # place ingredient in pot
    "plate_soup",  # pick up soup / place in dish
    "deliver",     # carry plated soup to serving location
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
    episode_length:
        Default horizon (timesteps) for this layout.
    description:
        Free-text note for documentation.
    """

    name: str
    layout_name: str
    subtasks: list[str] = field(default_factory=list)
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

# Cramped room: two agents sharing a tight counter-heavy space.
# Division of labour is the key challenge.  Supports all 4 core subtasks;
# no tomato dispenser in the default grid.
CRAMPED_ROOM = LayoutSpec(
    name="CrampedRoom",
    layout_name="cramped_room",
    subtasks=["fetch_onion", "fetch_dish", "chop", "plate_soup", "deliver"],
    episode_length=400,
    description=(
        "Two agents in a confined space; tests who handles prep vs. delivery."
    ),
)

# Asymmetric advantages: agents have unequal access to ingredients.
# Encourages stable long-term role specialisation.
ASYMMETRIC_ADVANTAGES = LayoutSpec(
    name="AsymmetricAdvantages",
    layout_name="asymmetric_advantages",
    subtasks=["fetch_onion", "fetch_tomato", "fetch_dish", "chop", "plate_soup", "deliver"],
    episode_length=400,
    description=(
        "Each agent has a spatial advantage for certain tasks; good for "
        "learning individual role preferences."
    ),
)

# Coordination ring: circular layout requiring constant movement coordination.
# Tests whether the human wants to lead or follow the flow.
COORDINATION_RING = LayoutSpec(
    name="CoordinationRing",
    layout_name="coordination_ring",
    subtasks=["fetch_onion", "fetch_dish", "chop", "plate_soup", "deliver"],
    episode_length=400,
    description=(
        "Ring-shaped kitchen; coordination bottleneck reveals movement "
        "and role preferences."
    ),
)

# Forced coordination: certain tasks physically require both agents.
# Useful for studying cooperative vs. independent task preferences.
FORCED_COORDINATION = LayoutSpec(
    name="ForcedCoordination",
    layout_name="forced_coordination",
    subtasks=["fetch_onion", "fetch_dish", "chop", "plate_soup", "deliver"],
    episode_length=400,
    description=(
        "Layout design forces handoff interactions; reveals preferences "
        "about who initiates vs. completes tasks."
    ),
)

# Counter circuit: counter-based layout for studying handoff patterns.
COUNTER_CIRCUIT = LayoutSpec(
    name="CounterCircuit",
    layout_name="counter_circuit",
    subtasks=["fetch_onion", "fetch_tomato", "fetch_dish", "chop", "plate_soup", "deliver"],
    episode_length=400,
    description=(
        "Counter-centric; both agents pass items via counters, testing "
        "preferences about fetching vs. cooking roles."
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
