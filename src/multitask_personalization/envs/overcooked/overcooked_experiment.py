"""Helpers for Hydra / ``run_single_experiment`` integration (Overcooked domain).

Mirrors ``spices_experiment.py`` but builds an OvercookedSceneSpec and a
ground-truth OvercookedPreferenceModel from a hidden HBM config name.
"""

from __future__ import annotations

from multitask_personalization.envs.overcooked.config.hidden_hbm_configs import (
    get_hidden_hbm_config,
)
from multitask_personalization.envs.overcooked.layouts import (
    ALL_SUBTASKS,
    get_layout,
)
from multitask_personalization.envs.overcooked.overcooked_env import (
    OvercookedHiddenSpec,
    OvercookedSceneSpec,
)
from multitask_personalization.envs.overcooked.overcooked_hbm import (
    DEFAULT_HUMAN,
    OvercookedPreferenceModel,
)


def build_overcooked_scene_spec(layout_name: str = "CrampedRoom") -> OvercookedSceneSpec:
    """Build a scene spec for Hydra instantiation."""
    layout_spec = get_layout(layout_name)
    return OvercookedSceneSpec(
        layout_spec=layout_spec,
        subtask_list=tuple(layout_spec.subtasks),
    )


def build_overcooked_hidden_spec(
    layout_name: str = "CrampedRoom",
    hidden_hbm_config_name: str = "RealisticCook",
    preferred_ingredient_count: dict | None = None,
) -> OvercookedHiddenSpec:
    """Build a hidden spec with a ground-truth HBM for Hydra instantiation.

    The hidden HBM's theta is set from the named config so that
    ``_human_claims()`` samples stochastically via ``sigmoid(phi + psi_true)``.
    """
    layout_spec = get_layout(layout_name)
    subtasks = list(layout_spec.subtasks)
    cfg = get_hidden_hbm_config(hidden_hbm_config_name)
    theta = cfg.generate_theta(subtasks)

    # Build a ground-truth HBM with the config's theta.
    # IMPORTANT: don't pass layouts= to constructor — register_layout initializes
    # phi from theta, so it must run AFTER set_theta (same pattern as spices).
    hbm = OvercookedPreferenceModel(
        subtasks=subtasks,
        layouts=[],
        sigma_r=cfg.sigma_r,
        sigma_h=cfg.sigma_h,
        sigma0=cfg.sigma0,
        sigma_obs=cfg.sigma_obs,
    )
    hbm.set_theta(DEFAULT_HUMAN, theta, sigma_h=cfg.sigma_h)
    hbm.register_layout(DEFAULT_HUMAN, layout_spec.name)

    # Derive preferred_actor from theta sign (for oracle / metric tracking).
    preferred_actor = {s: ("human" if theta[s] > 0 else "robot") for s in subtasks}

    pref_count = dict(preferred_ingredient_count) if preferred_ingredient_count else {}
    return OvercookedHiddenSpec(
        preferred_actor=preferred_actor,
        hidden_hbm=hbm,
        preferred_ingredient_count=pref_count,
    )
