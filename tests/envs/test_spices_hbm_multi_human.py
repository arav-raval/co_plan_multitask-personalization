import numpy as np

from multitask_personalization.envs.spices.spices_hbm import (
    MultiHumanHierarchicalPreferenceModel,
)


def test_multi_human_hbm_single_human_single_recipe_reduces_to_two_level():
    """With one human and one recipe, μ and θ should largely track the same direction as φ."""
    spices = ["salt", "pepper"]
    recipes = ["R1"]
    humans = ["H1"]

    hbm = MultiHumanHierarchicalPreferenceModel(spices=spices, recipes=recipes, human_ids=humans)

    # Provide strong positive evidence for salt, negative for pepper for H1 on R1.
    for _ in range(20):
        hbm.update_phi("H1", "R1", "salt", g=+3.0)
        hbm.update_phi("H1", "R1", "pepper", g=-3.0)

    hbm.update_theta_and_mu()

    mu_salt = hbm.get_mu("salt")
    mu_pepper = hbm.get_mu("pepper")
    theta_salt = hbm.get_theta("H1", "salt")
    theta_pepper = hbm.get_theta("H1", "pepper")

    # Directions should be consistent with evidence.
    assert theta_salt > 0.0
    assert theta_pepper < 0.0
    assert mu_salt > 0.0
    assert mu_pepper < 0.0

    # With only one human and one recipe, μ and θ should be similar up to shrinkage.
    assert np.sign(mu_salt) == np.sign(theta_salt)
    assert np.sign(mu_pepper) == np.sign(theta_pepper)


def test_multi_human_hbm_two_humans_one_recipe_pools_into_mu():
    """Two humans with opposite preferences should yield opposite θ but μ near zero."""
    spices = ["salt"]
    recipes = ["R1"]
    humans = ["H1", "H2"]

    hbm = MultiHumanHierarchicalPreferenceModel(spices=spices, recipes=recipes, human_ids=humans)

    # H1 strongly prefers human (+g), H2 strongly prefers robot (-g)
    for _ in range(20):
        hbm.update_phi("H1", "R1", "salt", g=+3.0)
        hbm.update_phi("H2", "R1", "salt", g=-3.0)

    hbm.update_theta_and_mu()

    theta_h1 = hbm.get_theta("H1", "salt")
    theta_h2 = hbm.get_theta("H2", "salt")
    mu_salt = hbm.get_mu("salt")

    # Human-level preferences should diverge in opposite directions.
    assert theta_h1 > 0.0
    assert theta_h2 < 0.0

    # Global μ should sit between them and be closer to zero than either θ magnitude.
    assert np.sign(theta_h1) == 1
    assert np.sign(theta_h2) == -1
    assert abs(mu_salt) < max(abs(theta_h1), abs(theta_h2))


def test_multi_human_hbm_multiple_recipes_per_human_pool_into_theta():
    """For a single human, θ should reflect an average over that human's recipe-specific φ."""
    spices = ["salt"]
    recipes = ["R1", "R2"]
    humans = ["H1"]

    hbm = MultiHumanHierarchicalPreferenceModel(spices=spices, recipes=recipes, human_ids=humans)

    # Human H1 likes salt in R1 (+3) but dislikes it in R2 (-1).
    for _ in range(10):
        hbm.update_phi("H1", "R1", "salt", g=+3.0)
        hbm.update_phi("H1", "R2", "salt", g=-1.0)

    hbm.update_theta_and_mu()

    phi_r1 = hbm.get_phi("H1", "R1", "salt")
    phi_r2 = hbm.get_phi("H1", "R2", "salt")
    theta = hbm.get_theta("H1", "salt")
    mu = hbm.get_mu("salt")

    # φ should reflect the sign of the corresponding evidence.
    assert phi_r1 > 0.0
    assert phi_r2 < 0.0

    # θ should be between φ_R1 and φ_R2 (rough average for that human).
    assert min(phi_r2, phi_r1) <= theta <= max(phi_r1, phi_r2)

    # μ should be a shrunk version of θ toward zero.
    assert np.sign(theta) == np.sign(mu) or abs(mu) < 1e-6
