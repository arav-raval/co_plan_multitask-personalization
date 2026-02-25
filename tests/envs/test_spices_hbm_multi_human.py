import numpy as np

from multitask_personalization.envs.spices.spices_hbm import HierarchicalPreferenceModel


def test_multi_human_hbm_single_human_single_recipe_reduces_to_two_level():
    """With one human and one recipe, μ and θ should largely track the same direction as φ."""
    spices = ["salt", "pepper"]
    hbm = HierarchicalPreferenceModel(spices=spices)

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
    hbm = HierarchicalPreferenceModel(spices=spices)

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
    hbm = HierarchicalPreferenceModel(spices=spices)

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


def test_new_human_warm_started_from_global_mu():
    """A new human's θ should be initialized from the current μ, not from 0."""
    spices = ["salt"]
    hbm = HierarchicalPreferenceModel(spices=spices)

    # Train H1 with strong positive evidence so μ shifts positive.
    for _ in range(20):
        hbm.update_phi("H1", "R1", "salt", g=+3.0)
    hbm.update_theta_and_mu()

    mu_before = hbm.get_mu("salt")
    assert mu_before > 0.0, "μ should shift positive after H1 training"

    # Register H2 (who has never been seen).
    hbm.register_human("H2")
    theta_h2 = hbm.get_theta("H2", "salt")

    # H2's θ should be initialized from current μ, not 0.
    assert abs(theta_h2 - mu_before) < 1e-9, (
        f"New human θ should equal current μ={mu_before:.4f}, got {theta_h2:.4f}"
    )


def test_new_recipe_warm_started_from_theta():
    """A new recipe's φ should be initialized from current θ, not from 0."""
    spices = ["salt"]
    hbm = HierarchicalPreferenceModel(spices=spices)

    # Build up θ for H1 via R1.
    for _ in range(20):
        hbm.update_phi("H1", "R1", "salt", g=+3.0)
    hbm.update_theta_and_mu()

    theta_before = hbm.get_theta("H1", "salt")
    assert theta_before > 0.0

    # Register a brand-new recipe R2 for H1.
    hbm.register_recipe("H1", "R2")
    phi_r2 = hbm.get_phi("H1", "R2", "salt")

    # φ for the new recipe should equal current θ (warm-start), not 0.
    assert abs(phi_r2 - theta_before) < 1e-9, (
        f"New recipe φ should equal current θ={theta_before:.4f}, got {phi_r2:.4f}"
    )
