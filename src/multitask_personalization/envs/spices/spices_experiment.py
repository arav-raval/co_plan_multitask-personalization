"""Helpers for Hydra / ``run_single_experiment`` integration (spices domain)."""

from __future__ import annotations

from multitask_personalization.envs.spices.config import (
    create_theta_params_from_config,
    get_hidden_hbm_config,
)
from multitask_personalization.envs.spices.recipes import get_recipe
from multitask_personalization.envs.spices.spices_env import SpiceSceneSpec
from multitask_personalization.envs.spices.spices_hbm import (
    DEFAULT_HUMAN,
    HierarchicalPreferenceModel,
)

# 4-recipe pool (all ~18 spices each). Chosen for:
#   - Strong shared core (salt, garlic, ginger, onion, pepper, chili) — anchors learning
#   - Diverse nuanced spices across dishes (basil, lemon, fennel_seed, oregano,
#     coriander, ghee, kaffir_lime, cardamom, fenugreek, mustard_seed, soy_sauce…)
#     so nuanced theta values (|theta|=0.5–0.8) are exercised across multiple recipes.
#   - Mediterranean adds aromatic-positive spices (basil +0.8, lemon +0.8, fennel_seed
#     +0.5) and mild-negatives (oregano -0.5, capers -0.3) that the 3-recipe pool lacked.
#   - Overlap ratio ≈0.51 (37 unique spices from 4×18): enough sharing for theta
#     generalization without making all dishes identical.
MULTI_RECIPE_EXPERIMENT_POOL: tuple[str, ...] = (
    "UltraComplexFeast",
    "AsianFusionBowl",
    "IndianFeastComplex",
    "MediterraneanComplex",
)


def _union_spices_sorted(recipe_names: list[str]) -> list[str]:
    seen: set[str] = set()
    for name in recipe_names:
        for s in get_recipe(name).spices:
            seen.add(s)
    return sorted(seen)


def build_spice_scene_spec_multi(train_recipe_names: list[str]) -> SpiceSceneSpec:
    """Scene spec for multi-dish training: ``recipe`` starts as the first name; pool is fixed."""
    names = [str(n) for n in train_recipe_names]
    if not names:
        raise ValueError("train_recipe_names must be non-empty")
    return SpiceSceneSpec(
        recipe=get_recipe(names[0]),
        train_recipe_names=tuple(names),
    )


def build_spice_scene_spec_default_multi() -> SpiceSceneSpec:
    """Default medium–long multi-recipe pool (see ``MULTI_RECIPE_EXPERIMENT_POOL``)."""
    return build_spice_scene_spec_multi(list(MULTI_RECIPE_EXPERIMENT_POOL))


def build_spice_experiment_hidden_hbm(
    recipe_names: list[str],
    hidden_hbm_config_name: str,
) -> HierarchicalPreferenceModel:
    """Ground-truth HBM over the union of spices for all listed recipes."""
    names = [str(n) for n in recipe_names]
    if not names:
        raise ValueError("recipe_names must be non-empty")
    spices = _union_spices_sorted(names)
    cfg = get_hidden_hbm_config(hidden_hbm_config_name)
    theta_mean = create_theta_params_from_config(cfg, spices)
    # Do NOT pass recipes= to the constructor: register_recipe initializes phi from
    # theta, so it must run *after* set_theta.  Passing recipes= here would register
    # them while theta is still all-zeros (the HBM default), leaving phi=0 for every
    # non-overridden spice and making sample_episode_preferences return the wrong actor.
    hbm = HierarchicalPreferenceModel(
        spices=spices,
        recipes=[],
        sigma_r=cfg.sigma_r,
        sigma_h=cfg.sigma_h,
        sigma0=cfg.sigma0,
        sigma_obs=cfg.sigma_obs,
    )
    hbm.set_theta(DEFAULT_HUMAN, theta_mean, sigma_h=cfg.sigma_h)
    # Register recipes now that theta is set: phi will be initialized from the correct
    # theta values, giving the hidden HBM accurate per-recipe ground-truth phi.
    for name in names:
        hbm.register_recipe(DEFAULT_HUMAN, name)

    # Option A: apply per-recipe phi overrides so sample_episode_preferences returns
    # the correct recipe-specific ground truth for conflict configs.
    if cfg.recipe_theta_overrides:
        for recipe_name, overrides in cfg.recipe_theta_overrides.items():
            if recipe_name in names:
                recipe_spices = list(get_recipe(recipe_name).spices)
                # Start from the full recipe theta, then apply overrides
                recipe_theta = cfg.generate_theta_for_recipe(recipe_spices, recipe_name)
                # Register the recipe in the hidden HBM so phi slots exist
                hbm.register_recipe(DEFAULT_HUMAN, recipe_name)
                # Only set the overridden spices — the rest use theta via register_recipe
                hbm.set_phi(DEFAULT_HUMAN, recipe_name, overrides)

    return hbm


def build_spice_experiment_hidden_hbm_default_multi(
    hidden_hbm_config_name: str,
) -> HierarchicalPreferenceModel:
    """Hidden HBM aligned with ``MULTI_RECIPE_EXPERIMENT_POOL`` (same vocabulary as default scene)."""
    return build_spice_experiment_hidden_hbm(
        list(MULTI_RECIPE_EXPERIMENT_POOL),
        hidden_hbm_config_name,
    )


# ---------------------------------------------------------------------------
# Claim 2: Cross-recipe generalization helpers
# ---------------------------------------------------------------------------

# Default split: train on 3, eval on 1 held-out recipe.
CROSS_RECIPE_TRAIN: tuple[str, ...] = (
    "UltraComplexFeast",
    "AsianFusionBowl",
    "IndianFeastComplex",
)
CROSS_RECIPE_EVAL: tuple[str, ...] = (
    "MediterraneanComplex",
)


def build_spice_scene_spec_cross_recipe_train() -> SpiceSceneSpec:
    """Scene spec for cross-recipe training: 3 recipes, no MediterraneanComplex."""
    return build_spice_scene_spec_multi(list(CROSS_RECIPE_TRAIN))


def build_spice_scene_spec_cross_recipe_eval() -> SpiceSceneSpec:
    """Scene spec for cross-recipe eval: held-out MediterraneanComplex only."""
    return build_spice_scene_spec_multi(list(CROSS_RECIPE_EVAL))


def build_spice_experiment_hidden_hbm_cross_recipe(
    hidden_hbm_config_name: str,
) -> HierarchicalPreferenceModel:
    """Hidden HBM spanning ALL recipes (train + eval) so truth is available everywhere."""
    all_recipes = list(CROSS_RECIPE_TRAIN) + list(CROSS_RECIPE_EVAL)
    return build_spice_experiment_hidden_hbm(all_recipes, hidden_hbm_config_name)


# ---------------------------------------------------------------------------
# Leave-one-out cross-recipe splits (4x more data than single held-out test)
# ---------------------------------------------------------------------------
# Each split holds out one of the 4 recipes from the main pool. Running all
# 4 splits × 10 seeds gives 40 measurements of cross-recipe transfer, instead
# of 10 measurements from a single held-out recipe.

_ALL_SPLITS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # (train, eval) — leave Ultra out
    (("AsianFusionBowl", "IndianFeastComplex", "MediterraneanComplex"),
     ("UltraComplexFeast",)),
    # leave Asian out
    (("UltraComplexFeast", "IndianFeastComplex", "MediterraneanComplex"),
     ("AsianFusionBowl",)),
    # leave Indian out
    (("UltraComplexFeast", "AsianFusionBowl", "MediterraneanComplex"),
     ("IndianFeastComplex",)),
    # leave Mediterranean out (the existing default)
    (("UltraComplexFeast", "AsianFusionBowl", "IndianFeastComplex"),
     ("MediterraneanComplex",)),
]

# Tight 4-recipe Middle Eastern cluster with ~77% mean pairwise spice overlap.
# Used for the "hierarchy transfers within a coherent cuisine" experiment —
# addresses the OOD issue in the global 4-recipe pool where each split's
# held-out recipe was effectively out-of-distribution for the training mean.
# These four share 14+ spices; DBTL's μ/θ prior should transfer cleanly.
ME_TIGHT_POOL: tuple[str, ...] = (
    "MiddleEasternFeast",
    "LebaneseKafta",
    "TurkishKebab",
    "TurkishDoner",
)

_ME_SPLITS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # leave MiddleEasternFeast out
    (("LebaneseKafta", "TurkishKebab", "TurkishDoner"),
     ("MiddleEasternFeast",)),
    # leave LebaneseKafta out
    (("MiddleEasternFeast", "TurkishKebab", "TurkishDoner"),
     ("LebaneseKafta",)),
    # leave TurkishKebab out
    (("MiddleEasternFeast", "LebaneseKafta", "TurkishDoner"),
     ("TurkishKebab",)),
    # leave TurkishDoner out
    (("MiddleEasternFeast", "LebaneseKafta", "TurkishKebab"),
     ("TurkishDoner",)),
]


def build_spice_scene_spec_loo_train_ultra() -> SpiceSceneSpec:
    """Leave-Ultra-out: train on Asian, Indian, Mediterranean."""
    return build_spice_scene_spec_multi(list(_ALL_SPLITS[0][0]))

def build_spice_scene_spec_loo_eval_ultra() -> SpiceSceneSpec:
    """Leave-Ultra-out eval: held-out Ultra."""
    return build_spice_scene_spec_multi(list(_ALL_SPLITS[0][1]))

def build_spice_scene_spec_loo_train_asian() -> SpiceSceneSpec:
    """Leave-Asian-out: train on Ultra, Indian, Mediterranean."""
    return build_spice_scene_spec_multi(list(_ALL_SPLITS[1][0]))

def build_spice_scene_spec_loo_eval_asian() -> SpiceSceneSpec:
    """Leave-Asian-out eval: held-out Asian."""
    return build_spice_scene_spec_multi(list(_ALL_SPLITS[1][1]))

def build_spice_scene_spec_loo_train_indian() -> SpiceSceneSpec:
    """Leave-Indian-out: train on Ultra, Asian, Mediterranean."""
    return build_spice_scene_spec_multi(list(_ALL_SPLITS[2][0]))

def build_spice_scene_spec_loo_eval_indian() -> SpiceSceneSpec:
    """Leave-Indian-out eval: held-out Indian."""
    return build_spice_scene_spec_multi(list(_ALL_SPLITS[2][1]))


def build_spice_experiment_hidden_hbm_loo(
    hidden_hbm_config_name: str,
) -> HierarchicalPreferenceModel:
    """Hidden HBM spanning ALL 4 recipes (train + held-out across splits)."""
    return build_spice_experiment_hidden_hbm(
        list(MULTI_RECIPE_EXPERIMENT_POOL), hidden_hbm_config_name,
    )


# --- Middle Eastern tight-cluster LOO builders ---------------------------
def build_spice_scene_spec_me_train_feast() -> SpiceSceneSpec:
    """Leave-MiddleEasternFeast-out: train on Lebanese, Turkish Kebab, Doner."""
    return build_spice_scene_spec_multi(list(_ME_SPLITS[0][0]))

def build_spice_scene_spec_me_eval_feast() -> SpiceSceneSpec:
    return build_spice_scene_spec_multi(list(_ME_SPLITS[0][1]))

def build_spice_scene_spec_me_train_lebanese() -> SpiceSceneSpec:
    """Leave-LebaneseKafta-out: train on Feast, Kebab, Doner."""
    return build_spice_scene_spec_multi(list(_ME_SPLITS[1][0]))

def build_spice_scene_spec_me_eval_lebanese() -> SpiceSceneSpec:
    return build_spice_scene_spec_multi(list(_ME_SPLITS[1][1]))

def build_spice_scene_spec_me_train_kebab() -> SpiceSceneSpec:
    """Leave-TurkishKebab-out: train on Feast, Lebanese, Doner."""
    return build_spice_scene_spec_multi(list(_ME_SPLITS[2][0]))

def build_spice_scene_spec_me_eval_kebab() -> SpiceSceneSpec:
    return build_spice_scene_spec_multi(list(_ME_SPLITS[2][1]))

def build_spice_scene_spec_me_train_doner() -> SpiceSceneSpec:
    """Leave-TurkishDoner-out: train on Feast, Lebanese, Kebab."""
    return build_spice_scene_spec_multi(list(_ME_SPLITS[3][0]))

def build_spice_scene_spec_me_eval_doner() -> SpiceSceneSpec:
    return build_spice_scene_spec_multi(list(_ME_SPLITS[3][1]))


def build_spice_experiment_hidden_hbm_me_loo(
    hidden_hbm_config_name: str,
) -> HierarchicalPreferenceModel:
    """Hidden HBM spanning ALL 4 Middle Eastern recipes."""
    return build_spice_experiment_hidden_hbm(
        list(ME_TIGHT_POOL), hidden_hbm_config_name,
    )


# ---------------------------------------------------------------------------
# Expanded 8-recipe pool (broader cross-recipe transfer)
# ---------------------------------------------------------------------------
# Extends MULTI_RECIPE_EXPERIMENT_POOL with 4 diverse global cuisines:
#   - MoroccanTagine:   74% overlap with original pool, adds saffron/raisins/almonds
#   - EthiopianWat:     78% overlap, adds berbere/niter_kibbeh/allspice/lime
#   - ThaiCurryComplex: mid overlap, adds lemongrass/galangal/kaffir_lime/fish_sauce
#   - SpanishPaella:    83% overlap, adds saffron/pimenton/white_wine
# Union vocabulary: 55 unique spices (vs 37 in original pool).
# The broader pool gives 8-choose-1 = 8 LOO splits (vs 4), each held-out
# recipe has higher overlap with the 7 training recipes, and the vocabulary
# diversity forces the HBM to generalize theta across more spice contexts.
EXPANDED_RECIPE_POOL: tuple[str, ...] = (
    "UltraComplexFeast",
    "AsianFusionBowl",
    "IndianFeastComplex",
    "MediterraneanComplex",
    "MoroccanTagine",
    "EthiopianWat",
    "ThaiCurryComplex",
    "SpanishPaella",
)

_EXPANDED_SPLITS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # (train_7, eval_1) for each recipe held out
    (tuple(r for r in EXPANDED_RECIPE_POOL if r != held_out), (held_out,))
    for held_out in EXPANDED_RECIPE_POOL
]


def build_spice_scene_spec_expanded_multi() -> SpiceSceneSpec:
    """Scene spec for all 8 expanded-pool recipes."""
    return build_spice_scene_spec_multi(list(EXPANDED_RECIPE_POOL))


def build_spice_experiment_hidden_hbm_expanded(
    hidden_hbm_config_name: str,
) -> HierarchicalPreferenceModel:
    """Hidden HBM spanning all 8 expanded-pool recipes."""
    return build_spice_experiment_hidden_hbm(
        list(EXPANDED_RECIPE_POOL), hidden_hbm_config_name,
    )


# LOO builders for expanded pool — train on 7, eval on 1.
def _expanded_loo_train(idx: int) -> SpiceSceneSpec:
    return build_spice_scene_spec_multi(list(_EXPANDED_SPLITS[idx][0]))

def _expanded_loo_eval(idx: int) -> SpiceSceneSpec:
    return build_spice_scene_spec_multi(list(_EXPANDED_SPLITS[idx][1]))


def build_spice_scene_spec_exp_loo_train_ultra() -> SpiceSceneSpec:
    return _expanded_loo_train(0)
def build_spice_scene_spec_exp_loo_eval_ultra() -> SpiceSceneSpec:
    return _expanded_loo_eval(0)

def build_spice_scene_spec_exp_loo_train_asian() -> SpiceSceneSpec:
    return _expanded_loo_train(1)
def build_spice_scene_spec_exp_loo_eval_asian() -> SpiceSceneSpec:
    return _expanded_loo_eval(1)

def build_spice_scene_spec_exp_loo_train_indian() -> SpiceSceneSpec:
    return _expanded_loo_train(2)
def build_spice_scene_spec_exp_loo_eval_indian() -> SpiceSceneSpec:
    return _expanded_loo_eval(2)

def build_spice_scene_spec_exp_loo_train_mediterranean() -> SpiceSceneSpec:
    return _expanded_loo_train(3)
def build_spice_scene_spec_exp_loo_eval_mediterranean() -> SpiceSceneSpec:
    return _expanded_loo_eval(3)

def build_spice_scene_spec_exp_loo_train_moroccan() -> SpiceSceneSpec:
    return _expanded_loo_train(4)
def build_spice_scene_spec_exp_loo_eval_moroccan() -> SpiceSceneSpec:
    return _expanded_loo_eval(4)

def build_spice_scene_spec_exp_loo_train_ethiopian() -> SpiceSceneSpec:
    return _expanded_loo_train(5)
def build_spice_scene_spec_exp_loo_eval_ethiopian() -> SpiceSceneSpec:
    return _expanded_loo_eval(5)

def build_spice_scene_spec_exp_loo_train_thai() -> SpiceSceneSpec:
    return _expanded_loo_train(6)
def build_spice_scene_spec_exp_loo_eval_thai() -> SpiceSceneSpec:
    return _expanded_loo_eval(6)

def build_spice_scene_spec_exp_loo_train_spanish() -> SpiceSceneSpec:
    return _expanded_loo_train(7)
def build_spice_scene_spec_exp_loo_eval_spanish() -> SpiceSceneSpec:
    return _expanded_loo_eval(7)


def build_spice_experiment_hidden_hbm_expanded_loo(
    hidden_hbm_config_name: str,
) -> HierarchicalPreferenceModel:
    """Hidden HBM spanning all 8 expanded-pool recipes (used for all LOO splits)."""
    return build_spice_experiment_hidden_hbm(
        list(EXPANDED_RECIPE_POOL), hidden_hbm_config_name,
    )


# ---------------------------------------------------------------------------
# Claim 3: Cross-human transfer helpers
# ---------------------------------------------------------------------------

# Population of related-but-different humans for multi-human training.
# Each config shares the same base preference structure but with personality
# offsets (heat-seeking, aromatic-gentle, etc.).
POPULATION_HUMAN_CONFIGS: tuple[tuple[str, str], ...] = (
    ("human_base", "SpiceSpecificHumanStrong"),
    ("human_heat", "SpiceSpecificHumanHeatSeeking"),
    ("human_gentle", "SpiceSpecificHumanAromaticGentle"),
)

# The new human introduced at eval time (unseen during training).
NEW_HUMAN_CONFIG: tuple[str, str] = ("human_new", "SpiceSpecificHumanRecipeConflict")


def build_multi_human_hidden_hbms(
    recipe_names: list[str],
) -> dict[str, HierarchicalPreferenceModel]:
    """Build separate hidden HBMs for each human in the population.

    Returns {human_label: hidden_hbm} where each HBM represents that
    human's true preferences (used by the environment to generate behavior).
    """
    result: dict[str, HierarchicalPreferenceModel] = {}
    for human_label, config_name in POPULATION_HUMAN_CONFIGS:
        result[human_label] = build_spice_experiment_hidden_hbm(
            recipe_names, config_name,
        )
    return result


def build_new_human_hidden_hbm(
    recipe_names: list[str],
) -> HierarchicalPreferenceModel:
    """Build hidden HBM for the new (unseen) human used at eval time."""
    label, config_name = NEW_HUMAN_CONFIG
    return build_spice_experiment_hidden_hbm(recipe_names, config_name)


# ---------------------------------------------------------------------------
# Non-stationarity experiment helpers
# ---------------------------------------------------------------------------
# Sign-flip shift configs: negate theta for a band of spices, preserving
# magnitude so the satisfaction ceiling is unchanged.  Four variants:
#   - SpiceShiftSoft:   flip |theta| <= 1.0  (16 spices)
#   - SpiceShiftMedium: flip 1.0 < |theta| <= 2.0  (14 spices)
#   - SpiceShiftStrong: flip |theta| > 2.0  (7 spices)
#   - SpiceShiftRandom: flip ~40% of spices randomly (seed-dependent)

def build_spice_shift_hidden_hbm(
    recipe_names: list[str],
    shift_config_name: str = "SpiceSpecificHumanShifted",
) -> HierarchicalPreferenceModel:
    """Build the POST-shift hidden HBM (used as shift_hidden_hbm in SpiceHiddenSpec)."""
    return build_spice_experiment_hidden_hbm(recipe_names, shift_config_name)


def build_spice_shift_hidden_hbm_default_multi(
    shift_config_name: str = "SpiceSpecificHumanShifted",
) -> HierarchicalPreferenceModel:
    """Post-shift HBM for the default 4-recipe pool."""
    return build_spice_shift_hidden_hbm(
        list(MULTI_RECIPE_EXPERIMENT_POOL), shift_config_name,
    )


def build_spice_shift_hidden_hbm_random(
    recipe_names: list[str],
    flip_seed: int = 0,
    flip_fraction: float = 0.4,
) -> HierarchicalPreferenceModel:
    """Build a POST-shift HBM with randomly selected sign-flips.

    Each ``flip_seed`` produces a different random subset of flipped spices,
    so each experimental seed gets a unique shift pattern.
    """
    from multitask_personalization.envs.spices.config.hidden_hbm_configs import (
        _sign_flip_theta,
        HiddenHBMConfig,
    )

    names = [str(n) for n in recipe_names]
    spices = _union_spices_sorted(names)

    # Build a one-off config with a seed-specific generator
    def _gen(sp: list[str]) -> dict[str, float]:
        return _sign_flip_theta(sp, flip_band=None, flip_fraction=flip_fraction, seed=flip_seed)

    cfg = HiddenHBMConfig(
        name=f"SpiceShiftRandom_seed{flip_seed}",
        theta_mean={},
        sigma_r=0.5,
        sigma_h=0.5,
        theta_generator=_gen,
    )

    theta_mean = cfg.generate_theta(spices)
    hbm = HierarchicalPreferenceModel(
        spices=spices,
        recipes=[],
        sigma_r=cfg.sigma_r,
        sigma_h=cfg.sigma_h,
        sigma0=cfg.sigma0,
        sigma_obs=cfg.sigma_obs,
    )
    hbm.set_theta(DEFAULT_HUMAN, theta_mean, sigma_h=cfg.sigma_h)
    for name in names:
        hbm.register_recipe(DEFAULT_HUMAN, name)
    return hbm


def build_spice_shift_random_default_multi(
    flip_seed: int = 0,
) -> HierarchicalPreferenceModel:
    """Random-flip post-shift HBM for the default 4-recipe pool."""
    return build_spice_shift_hidden_hbm_random(
        list(MULTI_RECIPE_EXPERIMENT_POOL), flip_seed=flip_seed,
    )
