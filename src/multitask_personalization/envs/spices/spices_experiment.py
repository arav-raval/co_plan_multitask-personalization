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
