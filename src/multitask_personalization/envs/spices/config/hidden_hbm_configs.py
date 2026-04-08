"""
hidden_hbm_configs.py
Central repository for hidden HBM configurations (true human preferences).

Each configuration defines:
- theta_mean: Fixed human-level preference for each spice (the "true" values to learn).
  Positive values indicate the human prefers to add that spice; negative means robot.
- Variance parameters: sigma0, sigma_h, sigma_r, sigma_obs.
- theta_generator: Optional callable (spices -> theta_mean dict) for pattern-based configs.
  When present, it takes priority over theta_mean.

Note: base_satisfaction_bias is an environment-level parameter and is NOT stored here.
It is read from DEFAULT_CONFIG.satisfaction.base_satisfaction_bias wherever needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict


# ---------------------------------------------------------------------------
# Pattern generators (module-level functions, no string-matching needed).
# ---------------------------------------------------------------------------

def _alternating_theta(spices: list[str]) -> dict[str, float]:
    """Alternating pattern: +1.5 for even-indexed spices, -1.5 for odd."""
    return {s: 1.5 if i % 2 == 0 else -1.5 for i, s in enumerate(spices)}


def _first_half_theta(spices: list[str]) -> dict[str, float]:
    """First half: strong human preference (+2.0); second half: robot (-2.0)."""
    mid = len(spices) // 2
    return {s: 2.0 if i < mid else -2.0 for i, s in enumerate(spices)}


def _second_half_theta(spices: list[str]) -> dict[str, float]:
    """First half: robot preference (-2.0); second half: human (+2.0)."""
    mid = len(spices) // 2
    return {s: -2.0 if i < mid else 2.0 for i, s in enumerate(spices)}


# Comprehensive spice preference mapping covering the full ChefComplex vocabulary
# and beyond (80+ spices).  Preferences are semantically grounded: the simulated human
# enjoys aromatic/sweet/citrusy flavours and dislikes sharp heat and bitter ingredients.
# Unlisted spices default to 0.0 (no preference) in _spice_specific_theta().
_SPICE_SPECIFIC_THETA: dict[str, float] = {
    # ── Core aromatics (loved) ────────────────────────────────────────────────
    "salt":            +2.0,
    "garlic":          +1.5,
    "onion":           +1.0,
    "ginger":          +1.5,
    "saffron":         +1.5,
    "avocado":         +1.5,
    "mint":            +1.2,
    "honey":           +1.0,
    "tomato":          +1.0,
    "cumin":           +1.0,
    "lemongrass":      +1.0,
    "almonds":         +1.0,
    "tahini":          +1.0,
    "coconut":         +1.0,
    "turmeric":        +0.5,
    "coriander":       +0.8,
    "paprika":         +0.8,
    "ghee":            +0.8,
    "lemon":           +0.8,
    "lime":            +0.8,
    "basil":           +0.8,
    "kaffir_lime":     +0.8,
    "coconut_milk":    +0.8,
    "niter_kibbeh":    +0.8,
    "apricot":         +0.8,
    "brown_sugar":     +0.8,
    "orange_zest":     +0.8,
    # ── Mild positives ────────────────────────────────────────────────────────
    "olive_oil":       +0.5,
    "coconut_oil":     +0.5,
    "sesame_seeds":    +0.5,
    "parsley":         +0.5,
    "lavender":        +0.5,
    "black_olives":    +0.5,
    "scallion":        +0.5,
    "shallot":         +0.5,
    "raisins":         +0.5,
    "mirin":           +0.5,
    "jaggery":         +0.5,
    "yogurt":          +0.5,
    "cocoa":           +0.5,
    "zaatar":          +0.5,
    "cumin_seed":      +0.5,
    "pear":            +0.5,
    "pistachio":       +0.8,   # nutty, rich — loved like almonds
    "butter":          +0.5,   # rich and creamy, mildly loved
    "cilantro":        +0.5,
    "thyme":           +0.5,
    "rosemary":        +0.3,   # 1 training recipe (MediterraneanComplex) — below signal threshold
    "marjoram":        +0.3,   # 1 training recipe (HungarianGulash) — below signal threshold
    "sesame_oil":      +0.3,   # 1 training recipe (AsianFusionBowl) — below signal threshold
    "soy_sauce":       +0.5,
    "preserved_lemon": +0.3,   # 1 training recipe (MoroccanTagine) — below signal threshold
    "palm_sugar":      +0.5,
    "sumac":           +0.5,
    "fennel_seed":     +0.5,
    "curry_leaves":    +0.5,
    "galangal":        +0.5,
    "cinnamon":        +0.5,
    "milk":            +0.3,   # 1 training recipe (KashmiriWazwan) — below signal threshold
    "advieh":          +0.8,   # 1 training recipe (PersianTahdig) — strengthened so 1 recipe is enough
    "sugar":           +0.5,
    "bay_leaf":        +0.5,
    # ── Neutral / slightly negative ───────────────────────────────────────────
    "rice_vinegar":    -0.3,   # 1 training recipe (AsianFusionBowl) — below signal threshold
    "oregano":         -0.5,
    "capers":          -0.3,   # 1 training recipe (MediterraneanComplex) — below signal threshold
    "barberries":      -0.8,   # 1 training recipe (PersianTahdig) — strengthened so 1 recipe is enough
    "allspice":        -0.5,
    "tamarind":        -0.5,
    "caraway_seed":    -0.3,   # 1 training recipe (HungarianGulash) — below signal threshold
    # ── Disliked (heat, bitter, pungent) ──────────────────────────────────────
    "cayenne":         -0.8,   # hot spice, disliked
    "harissa":         -1.0,   # fiery chili paste, disliked
    "garam_masala":    -0.8,
    "cardamom":        -0.8,
    "mustard_seed":    -0.8,
    "fish_sauce":      -0.8,
    "mustard_oil":     -0.8,
    "clove":           -1.0,
    "star_anise":      -1.0,
    "chipotle":        -1.0,
    "shrimp_paste":    -1.0,
    "fenugreek":       -1.2,
    "jalapeno":        -1.2,
    "chili":           -1.5,
    "asafoetida":      -1.5,
    "gochujang":       -1.5,
    "gochugaru":       -1.5,
    "black_pepper":    -1.5,
    "berbere":         -1.5,
    "green_chili":     -1.5,
    "pepper":          -2.0,
}


def _spice_specific_theta(spices: list[str]) -> dict[str, float]:
    """Use the explicit spice-specific mapping; default to 0.0 for unlisted spices."""
    return {s: _SPICE_SPECIFIC_THETA.get(s, 0.0) for s in spices}


def _scaled_theta_map(base: dict[str, float], scale: float) -> dict[str, float]:
    return {k: scale * v for k, v in base.items()}


def _adjust_theta_map(
    base: dict[str, float],
    deltas: dict[str, float],
    clamp_abs: float = 3.0,
) -> dict[str, float]:
    """
    Additive tweak around a base map, with clipping for stability.

    Useful for defining coherent related human profiles for multi-human tests.
    """
    out: dict[str, float] = {}
    keys = set(base) | set(deltas)
    for k in keys:
        val = base.get(k, 0.0) + deltas.get(k, 0.0)
        out[k] = float(max(-clamp_abs, min(clamp_abs, val)))
    return out


def _spice_specific_theta_strong(spices: list[str]) -> dict[str, float]:
    """
    Strong-magnitude version of SpiceSpecificHuman.

    Preserves sign structure from _SPICE_SPECIFIC_THETA while increasing absolute
    values to reduce near-neutral preferences and make actor flips less frequent.
    """
    strong = _scaled_theta_map(_SPICE_SPECIFIC_THETA, scale=2.0)
    return {s: strong.get(s, 0.0) for s in spices}


def _consistent_variable_theta(spices: list[str]) -> dict[str, float]:
    """Same spice-specific pattern as SpiceSpecificHuman (variance differs in sigma_r)."""
    return _spice_specific_theta(spices)


# ---------------------------------------------------------------------------
# Multi-human coherent variants (shared backbone + interpretable offsets).
# ---------------------------------------------------------------------------

_HEAT_SEEKING_DELTAS: dict[str, float] = {
    "pepper": +1.4,
    "chili": +1.6,
    "black_pepper": +1.3,
    "jalapeno": +1.1,
    "green_chili": +1.1,
    "gochujang": +1.0,
    "gochugaru": +1.0,
    "cayenne": +1.0,
    "harissa": +1.0,
    "berbere": +1.0,
}

_AROMATIC_GENTLE_DELTAS: dict[str, float] = {
    "garlic": +0.4,
    "ginger": +0.4,
    "mint": +0.5,
    "cumin": +0.4,
    "turmeric": +0.4,
    "basil": +0.3,
    "lemongrass": +0.4,
    "chili": -0.8,
    "pepper": -0.8,
    "black_pepper": -0.6,
    "jalapeno": -0.7,
    "green_chili": -0.7,
    "gochujang": -0.6,
    "gochugaru": -0.6,
}


def _spice_specific_theta_heat_seeking(spices: list[str]) -> dict[str, float]:
    variant = _adjust_theta_map(_SPICE_SPECIFIC_THETA, _HEAT_SEEKING_DELTAS)
    return {s: variant.get(s, 0.0) for s in spices}


def _spice_specific_theta_aromatic_gentle(spices: list[str]) -> dict[str, float]:
    variant = _adjust_theta_map(_SPICE_SPECIFIC_THETA, _AROMATIC_GENTLE_DELTAS)
    return {s: variant.get(s, 0.0) for s in spices}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HiddenHBMConfig:
    """
    Configuration for a hidden HBM representing a human's true preferences.

    For pattern-based configs, provide a `theta_generator` callable rather than
    populating `theta_mean` manually.  The generator takes the full spice list and
    returns a {spice: theta_mean} dict, so patterns stay correct regardless of which
    recipe is used.

    Note: base_satisfaction_bias is NOT stored here — it belongs to the environment
    config (DEFAULT_CONFIG.satisfaction.base_satisfaction_bias).
    """
    name: str
    theta_mean: dict[str, float]

    sigma0: float = 1.0   # Global variance (Level 1: μ)
    sigma_h: float = 1.0  # Human-level variance (Level 2: θ)
    sigma_r: float = 1.0  # Recipe-level variance (Level 3: φ) — controls sampling variation
    sigma_obs: float = 1.0  # Observation noise

    # When True, the hidden HBM samples phi ~ N(theta, sigma_r²) each episode and
    # uses its sign as the ground-truth preferred actor.  This makes borderline spices
    # (|theta| small relative to sigma_r) flip preference across episodes, which
    # exposes the limits of methods that cannot track per-recipe uncertainty (CBTL).
    stochastic_preferences: bool = False

    # Per-recipe theta overrides (Option A). Maps recipe_name -> {spice -> theta}.
    # Spices listed here override the global theta_mean/theta_generator for that recipe.
    # This lets you define true per-recipe conflicts: a spice that the human prefers
    # to handle in one recipe (theta>0) but defers to the robot in another (theta<0).
    # CBTL shares w[human][spice] across all recipes and will average contradictory
    # gradients to ~0, making consistently mediocre decisions on conflicted spices.
    # The HBM's per-recipe phi tracks each correctly.
    recipe_theta_overrides: dict[str, dict[str, float]] = field(
        default_factory=dict, hash=False, compare=False, repr=False
    )

    # When provided, this callable generates the full theta_mean dict from the recipe's
    # spice list.  It takes priority over the `theta_mean` field.
    theta_generator: Callable[[list[str]], dict[str, float]] | None = field(
        default=None, hash=False, compare=False, repr=False
    )

    def get_theta_for_spice(self, spice: str) -> float:
        """Return theta_mean for a specific spice, or 0.0 if not listed."""
        return self.theta_mean.get(spice, 0.0)

    def generate_theta(self, spices: list[str]) -> dict[str, float]:
        """
        Return a {spice: theta_mean} mapping for the given spice list.

        If a theta_generator is provided it is called; otherwise theta_mean is
        used directly, filling in 0.0 for any unlisted spice.
        """
        if self.theta_generator is not None:
            return self.theta_generator(spices)
        return {s: self.theta_mean.get(s, 0.0) for s in spices}

    def generate_theta_for_recipe(self, spices: list[str], recipe_name: str) -> dict[str, float]:
        """
        Return a {spice: theta_mean} mapping for the given spice list, applying
        any per-recipe overrides from recipe_theta_overrides.

        recipe_theta_overrides[recipe_name] entries replace individual spice thetas
        while the global theta (from theta_generator or theta_mean) fills the rest.
        This allows true per-recipe conflicts without changing the global human profile.
        """
        base = self.generate_theta(spices)
        overrides = self.recipe_theta_overrides.get(recipe_name, {})
        if overrides:
            base = {s: overrides.get(s, base[s]) for s in spices}
        return base


# ---------------------------------------------------------------------------
# Pre-defined human configurations
# ---------------------------------------------------------------------------

ALTERNATING_HUMAN = HiddenHBMConfig(
    name="AlternatingHuman",
    theta_mean={},
    sigma_r=0.3,  # Low noise so phi almost never flips sign relative to theta
    sigma_h=0.3,  # phi_std = sqrt(sigma_h² + sigma_r²) ≈ 0.42; P(flip | theta=±1.5) < 0.1%
    theta_generator=_alternating_theta,
)

HUMAN_PREFERS_FIRST_HALF = HiddenHBMConfig(
    name="HumanPrefersFirstHalf",
    theta_mean={},
    sigma_r=1.0,
    sigma_h=1.0,
    theta_generator=_first_half_theta,
)

HUMAN_PREFERS_SECOND_HALF = HiddenHBMConfig(
    name="HumanPrefersSecondHalf",
    theta_mean={},
    sigma_r=1.0,
    sigma_h=1.0,
    theta_generator=_second_half_theta,
)

SPICE_SPECIFIC_HUMAN = HiddenHBMConfig(
    name="SpiceSpecificHuman",
    theta_mean=_SPICE_SPECIFIC_THETA,
    sigma_r=0.5,
    sigma_h=0.5,
    theta_generator=_spice_specific_theta,
)

SPICE_SPECIFIC_HUMAN_STRONG = HiddenHBMConfig(
    name="SpiceSpecificHumanStrong",
    theta_mean={},
    sigma_r=0.5,
    sigma_h=0.5,
    theta_generator=_spice_specific_theta_strong,
)

# Stochastic variant: phi ~ N(theta, sigma_r²) resampled each episode.
# Borderline spices (|theta| ≈ sigma_r) flip preferred actor across episodes,
# which exposes CBTL's inability to track per-recipe preference uncertainty.
SPICE_SPECIFIC_HUMAN_STRONG_STOCHASTIC = HiddenHBMConfig(
    name="SpiceSpecificHumanStrongStochastic",
    theta_mean={},
    sigma_r=0.5,
    sigma_h=0.5,
    theta_generator=_spice_specific_theta_strong,
    stochastic_preferences=True,
)

# Mid-strength stochastic variant: uses base (1×) theta magnitudes so nuanced spices
# (|theta|=0.5–1.0) sit at or below sigma_r and genuinely flip preferred actor across
# recipes/episodes. sigma_r=0.8 means:
#   - Strong spices (|theta|≥1.5): P(flip) < 3% — stable ground truth for anchoring
#   - Mid spices (|theta|≈0.8–1.0): P(flip) ≈ 16–32% — occasionally flip, challenging
#   - Nuanced spices (|theta|≈0.5): P(flip) ≈ 53% — near-random, hardest to learn
# The HBM's per-recipe phi posterior should track these; CBTL's shared w averages away.
SPICE_SPECIFIC_HUMAN_MID_STOCHASTIC = HiddenHBMConfig(
    name="SpiceSpecificHumanMidStochastic",
    theta_mean={},
    sigma_r=0.8,
    sigma_h=0.5,
    theta_generator=_spice_specific_theta,
    stochastic_preferences=True,
)

# Recipe-conflicting variant: per-recipe phi overrides model a realistic persona
# whose spice handling preferences vary by cuisine familiarity.
#
# Persona narrative:
#   A home cook confident in Mediterranean and pan-fusion cooking, but less
#   experienced with Indian and Asian cuisines.  They want to personally handle
#   spices they know well in familiar cuisines (positive phi → "human"), but
#   defer to the robot (negative phi → "robot") in unfamiliar ones where the
#   robot's trained recipes are more reliable.
#
#   For example, they insist on personally seasoning with cumin in a
#   Mediterranean or fusion context (they know the right amount), but defer
#   to the robot in an Asian stir-fry (where the balance with soy sauce and
#   ginger is less intuitive to them).  Similarly, they're precise about
#   yogurt in a fusion dish but defer in Indian curry where tempering yogurt
#   requires technique they haven't mastered.
#
# Design principles:
#   - Conflicts span all preference bands (strong, mid, weak) for thorough
#     evaluation — not just strong flips, which would be unrealistically stark.
#   - Asymmetric magnitudes: the override is not always a full sign flip.
#     Deference in unfamiliar cuisines (-1.0 to -1.5) is weaker than
#     confidence in familiar ones (+1.5 to +2.0) — the cook has opinions
#     but is cautious rather than averse.
#   - Anchor spices (salt, garlic, pepper, ginger, chili) are kept consistent
#     across all recipes so both CBTL and HBM have a stable learning signal.
#   - The theta_generator uses 2× (strong) magnitudes.  This raises the
#     satisfaction ceiling (E[tanh(|phi|)] ≈ 0.90 vs 0.70 at 1×) while keeping
#     the same narrative.  Override magnitudes are scaled proportionally so the
#     relative conflict structure is preserved.
#
# Conflict table (overrides only — blanks inherit from global θ at 2× scale):
#
#  Band   | Spice      | Ultra  | Asian  | Indian | Mediter | Global θ | Rationale
#  -------|------------|--------|--------|--------|---------|----------|----------
#  strong | cumin      |  +2.0  |  -3.0  |  -2.0  |  +3.0   |  +2.0   | Confident in Med/fusion cumin, defers in Asian/Indian amounts
#  strong | onion      |  +2.0  |  -3.0  |  -3.0  |  +3.0   |  +2.0   | Knows Med onion prep; unfamiliar with Asian/Indian technique
#  strong | honey      |  +2.0  |  -3.0  |   —    |   —     |  +2.0   | Precise about fusion glaze, defers to robot for Asian balance
#  mid    | coriander  |  +1.6  |  -2.0  |  -1.6  |  +2.0   |  +1.6   | Fresh cilantro vs ground coriander — different per cuisine
#  mid    | paprika    |  +1.6  |  -1.6  |   —    |  +2.0   |  +1.6   | Smoked paprika in Med/fusion (personal), Asian chili blend (defers)
#  mid    | turmeric   |  +1.0  |  -2.0  |  -1.0  |   —     |  +1.0   | Mild preference globally; defers in Asian (staining concerns)
#  weak   | yogurt     |  +1.0  |   —    |  -1.6  |   —     |  +1.0   | Fusion raita (personal) vs Indian curry tempering (defers)
#  weak   | cinnamon   |  +1.0  |  +1.0  |  -1.0  |   —     |  +1.0   | Sweet/baking contexts (positive) vs Indian savory (defers)
#
# CBTL impact: shared w[spice] averages contradictory labels → w≈0 on all
#   8 conflict spices, producing ~50% accuracy on them regardless of training.
# HBM impact: per-recipe phi tracks each cuisine's direction independently,
#   learning that cumin is "human" in Mediterranean but "robot" in Asian.
SPICE_SPECIFIC_HUMAN_RECIPE_CONFLICT = HiddenHBMConfig(
    name="SpiceSpecificHumanRecipeConflict",
    theta_mean={},
    sigma_r=0.5,
    sigma_h=0.5,
    theta_generator=_spice_specific_theta_strong,
    recipe_theta_overrides={
        "AsianFusionBowl": {
            # Strong: familiar spices become unfamiliar in Asian context
            "cumin":     -3.0,
            "onion":     -3.0,
            "honey":     -3.0,
            # Mid: ground spice handling differs
            "coriander": -2.0,
            "paprika":   -1.6,
            "turmeric":  -2.0,
        },
        "IndianFeastComplex": {
            # Strong: Indian technique is different from familiar Western prep
            "cumin":     -2.0,
            "onion":     -3.0,
            # Mid: Indian-specific handling
            "coriander": -1.6,
            "turmeric":  -1.0,
            # Weak: technique-dependent preferences
            "yogurt":    -1.6,
            "cinnamon":  -1.0,
        },
        "MediterraneanComplex": {
            # Boost: extra confident in Med cuisine — push above global theta
            "cumin":     +3.0,
            "onion":     +3.0,
            "coriander": +2.0,
            "paprika":   +2.0,
        },
    },
)

SPICE_SPECIFIC_HUMAN_HEAT_SEEKING = HiddenHBMConfig(
    name="SpiceSpecificHumanHeatSeeking",
    theta_mean={},
    sigma_r=0.5,
    sigma_h=0.5,
    theta_generator=_spice_specific_theta_heat_seeking,
)

SPICE_SPECIFIC_HUMAN_AROMATIC_GENTLE = HiddenHBMConfig(
    name="SpiceSpecificHumanAromaticGentle",
    theta_mean={},
    sigma_r=0.5,
    sigma_h=0.5,
    theta_generator=_spice_specific_theta_aromatic_gentle,
)

CONSISTENT_HUMAN = HiddenHBMConfig(
    name="ConsistentHuman",
    theta_mean={},
    sigma_r=0.1,  # Low noise: P(episode flip | theta=±0.8) < 4% → clean learning signal
    sigma_h=0.3,  # Low human-level variance: theta is stable across recipes
    theta_generator=_consistent_variable_theta,
)

VARIABLE_HUMAN = HiddenHBMConfig(
    name="VariableHuman",
    theta_mean={},
    sigma_r=2.0,  # Larger variance → more variation across recipes
    sigma_h=1.5,
    theta_generator=_consistent_variable_theta,
)

STRONG_PREFERENCES_HUMAN = HiddenHBMConfig(
    name="StrongPreferencesHuman",
    theta_mean={
        "salt": 3.0,
        "garlic": 2.5,
        "onion": 2.0,
        "pepper": -3.0,
        "chili": -2.5,
    },
    sigma_r=1.0,
    sigma_h=1.0,
)

WEAK_PREFERENCES_HUMAN = HiddenHBMConfig(
    name="WeakPreferencesHuman",
    theta_mean={
        "salt": 0.5,
        "garlic": 0.3,
        "onion": 0.2,
        "pepper": -0.5,
        "chili": -0.3,
    },
    sigma_r=0.8,
    sigma_h=0.6,
)

# ---------------------------------------------------------------------------
# Registry and lookup helpers
# ---------------------------------------------------------------------------

ALL_HIDDEN_HBM_CONFIGS: dict[str, HiddenHBMConfig] = {
    "AlternatingHuman": ALTERNATING_HUMAN,
    "HumanPrefersFirstHalf": HUMAN_PREFERS_FIRST_HALF,
    "HumanPrefersSecondHalf": HUMAN_PREFERS_SECOND_HALF,
    "SpiceSpecificHuman": SPICE_SPECIFIC_HUMAN,
    "SpiceSpecificHumanStrong": SPICE_SPECIFIC_HUMAN_STRONG,
    "SpiceSpecificHumanStrongStochastic": SPICE_SPECIFIC_HUMAN_STRONG_STOCHASTIC,
    "SpiceSpecificHumanMidStochastic": SPICE_SPECIFIC_HUMAN_MID_STOCHASTIC,
    "SpiceSpecificHumanRecipeConflict": SPICE_SPECIFIC_HUMAN_RECIPE_CONFLICT,
    "SpiceSpecificHumanHeatSeeking": SPICE_SPECIFIC_HUMAN_HEAT_SEEKING,
    "SpiceSpecificHumanAromaticGentle": SPICE_SPECIFIC_HUMAN_AROMATIC_GENTLE,
    "ConsistentHuman": CONSISTENT_HUMAN,
    "VariableHuman": VARIABLE_HUMAN,
    "StrongPreferencesHuman": STRONG_PREFERENCES_HUMAN,
    "WeakPreferencesHuman": WEAK_PREFERENCES_HUMAN,
}


def get_hidden_hbm_config(name: str) -> HiddenHBMConfig:
    """Retrieve a hidden HBM configuration by name."""
    if name not in ALL_HIDDEN_HBM_CONFIGS:
        raise ValueError(
            f"Unknown hidden HBM config name: {name!r}. "
            f"Available: {list(ALL_HIDDEN_HBM_CONFIGS)}"
        )
    return ALL_HIDDEN_HBM_CONFIGS[name]


def list_hidden_hbm_configs() -> list[str]:
    """List all available hidden HBM configuration names."""
    return list(ALL_HIDDEN_HBM_CONFIGS)


def create_theta_params_from_config(
    config: HiddenHBMConfig, spices: list[str]
) -> dict[str, float]:
    """Return the theta_mean dict for *config* given the recipe's spice list."""
    return config.generate_theta(spices)


if __name__ == "__main__":
    print("Available hidden HBM configurations:")
    for _name in list_hidden_hbm_configs():
        _cfg = get_hidden_hbm_config(_name)
        print(f"  - {_name}  (sigma_r={_cfg.sigma_r}, sigma_h={_cfg.sigma_h})")
        if _cfg.theta_mean:
            print(f"    Sample spices: {list(_cfg.theta_mean)[:5]}...")
