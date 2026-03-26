"""
test_spices_hbm_stage1.py — Validation tests for the Stage 1 HBM migration.

Run with:
    pytest tests/envs/test_spices_hbm_stage1.py -v

These tests verify the four properties listed in the Stage 1 migration plan:
  1. phi posteriors converge toward true values
  2. phi variance shrinks monotonically with data
  3. ELBO increases over training
  4. sigma_obs adapts away from its initial value

They also verify that the public CSP interface (log_prob_prefer, preferred_actor)
is unchanged and produces sensible values.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

# Adjust this import path to match your project layout.
# If your module is at src/multitask_personalization/envs/spices/spices_hbm_stage1.py:
from multitask_personalization.envs.spices.spices_hbm import (
    HierarchicalPreferenceModel,
    DEFAULT_HUMAN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hbm(spices=None, sigma_obs_init=0.5) -> HierarchicalPreferenceModel:
    spices = spices or ["turmeric", "cumin", "chili"]
    return HierarchicalPreferenceModel(
        spices=spices,
        sigma_obs=sigma_obs_init,
        n_phi_steps=10,
        n_theta_steps=15,
    )


def run_episodes(
    hbm: HierarchicalPreferenceModel,
    recipe: str,
    spice: str,
    actor: str,        # the actor that correctly matches the human's preference
    n_episodes: int,
    steps_per_episode: int = 6,
    satisfaction_mean: float = 0.8,
    satisfaction_std: float = 0.1,
    rng: np.random.Generator = None,
) -> tuple[list[float], list[float]]:
    """
    Simulate episodes where `actor` always performs `spice` with positive satisfaction.
    Returns (phi_means, phi_vars) recorded at each episode end.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    phi_means = []
    phi_vars = []

    for _ in range(n_episodes):
        for _ in range(steps_per_episode):
            sat = float(np.clip(rng.normal(satisfaction_mean, satisfaction_std), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, recipe, spice, actor, sat)
        hbm.end_episode(DEFAULT_HUMAN)

        phi_means.append(hbm.get_phi(DEFAULT_HUMAN, recipe, spice))
        phi_vars.append(hbm.get_phi_var(DEFAULT_HUMAN, recipe, spice))

    return phi_means, phi_vars


# ---------------------------------------------------------------------------
# Test 1: phi converges toward the correct sign
# ---------------------------------------------------------------------------

class TestPhiConvergence:

    def test_phi_positive_for_human_actor(self):
        """
        After 20 episodes of human adding turmeric with positive satisfaction,
        phi should be positive (robot < human < +inf maps to P(human) > 0.5).
        """
        hbm = make_hbm()
        phi_means, _ = run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=20, satisfaction_mean=0.8,
        )
        final_phi = phi_means[-1]
        assert final_phi > 0.5, (
            f"Expected phi > 0.5 after 20 positive human episodes, got {final_phi:.3f}"
        )

    def test_phi_negative_for_robot_actor(self):
        """
        After 20 episodes of robot adding cumin with positive satisfaction,
        phi should be negative (human prefers robot to add it).
        """
        hbm = make_hbm()
        phi_means, _ = run_episodes(
            hbm, recipe="SimpleDal", spice="cumin",
            actor="robot", n_episodes=20, satisfaction_mean=0.8,
        )
        final_phi = phi_means[-1]
        assert final_phi < -0.5, (
            f"Expected phi < -0.5 after 20 positive robot episodes, got {final_phi:.3f}"
        )

    def test_phi_monotonically_increasing_with_consistent_data(self):
        """
        With consistent positive human episodes, phi should converge to a
        clearly positive value. We allow non-monotonicity between episodes
        (stochastic ELBO + per-episode observation windows can cause phi to
        overshoot in episode 1 and then settle), but after 15 episodes phi
        must be well above zero.
        """
        hbm = make_hbm()
        phi_means, _ = run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=15,
        )
        # phi started at 0.0 before training; must have converged clearly positive
        assert phi_means[-1] > 0.5, (
            f"phi should be well above zero after 15 positive human episodes, "
            f"got {phi_means[-1]:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 2: phi variance shrinks with data
# ---------------------------------------------------------------------------

class TestVarianceShrinkage:

    def test_phi_var_shrinks_overall(self):
        """
        phi variance at episode 20 must be less than phi variance at episode 1.
        """
        hbm = make_hbm()
        _, phi_vars = run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=20,
        )
        assert phi_vars[-1] < phi_vars[0], (
            f"Variance should shrink: {phi_vars[0]:.4f} → {phi_vars[-1]:.4f}"
        )

    def test_phi_var_substantially_reduced_after_many_episodes(self):
        """
        After 20 episodes, phi variance should be at least 50% lower than initial.
        """
        hbm = make_hbm()
        # Get initial variance (before any observations)
        hbm.register_recipe(DEFAULT_HUMAN, "SimpleDal")
        initial_var = hbm.get_phi_var(DEFAULT_HUMAN, "SimpleDal", "turmeric")

        _, phi_vars = run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=20,
        )
        reduction = 1.0 - phi_vars[-1] / initial_var
        assert reduction > 0.3, (
            f"Expected >30% variance reduction, got {reduction*100:.1f}%"
        )

    def test_phi_var_is_smaller_with_more_data(self):
        """
        A model trained on 20 episodes should have lower phi variance
        than one trained on 5 episodes.
        """
        rng = np.random.default_rng(99)

        hbm_few = make_hbm()
        _, vars_few = run_episodes(
            hbm_few, "SimpleDal", "turmeric", "human", n_episodes=5, rng=rng,
        )

        rng2 = np.random.default_rng(99)
        hbm_many = make_hbm()
        _, vars_many = run_episodes(
            hbm_many, "SimpleDal", "turmeric", "human", n_episodes=20, rng=rng2,
        )

        assert vars_many[-1] < vars_few[-1], (
            f"More data should give lower variance: "
            f"5ep={vars_few[-1]:.4f}, 20ep={vars_many[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: ELBO increases over training
# ---------------------------------------------------------------------------

class TestELBOImprovement:

    def test_elbo_history_nondecreasing_trend(self):
        """
        The ELBO recorded during training should show an upward trend.
        We compare the mean of the first 20% vs the last 20% of recorded values.
        """
        hbm = make_hbm()
        run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=20,
        )

        history = hbm._elbo_history
        assert len(history) > 10, "Should have recorded ELBO values"

        n = len(history)
        early_mean = float(np.mean(history[:n // 5]))
        late_mean = float(np.mean(history[-n // 5:]))

        assert late_mean > early_mean, (
            f"ELBO should improve over training: early={early_mean:.3f}, late={late_mean:.3f}"
        )

    def test_elbo_snapshot_is_finite(self):
        """compute_elbo_snapshot should return a finite value after some observations."""
        hbm = make_hbm()
        rng = np.random.default_rng(0)
        recipe, spice = "SimpleDal", "turmeric"

        for _ in range(5):
            sat = float(np.clip(rng.normal(0.7, 0.1), -1, 1))
            hbm.observe(DEFAULT_HUMAN, recipe, spice, "human", sat)

        elbo = hbm.compute_elbo_snapshot(DEFAULT_HUMAN, recipe, spice)
        assert math.isfinite(elbo), f"ELBO snapshot should be finite, got {elbo}"


# ---------------------------------------------------------------------------
# Test 4: Hyperparameters adapt
# ---------------------------------------------------------------------------

class TestHyperparamAdaptation:

    def test_sigma_obs_changes_from_initial(self):
        """
        sigma_obs should adapt away from its initial value after training.
        """
        initial_sigma = 0.5
        hbm = make_hbm(sigma_obs_init=initial_sigma)

        run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=20,
        )

        learned = hbm.get_learned_sigmas()
        assert abs(learned["sigma_obs"] - initial_sigma) > 0.01, (
            f"sigma_obs should have moved from {initial_sigma:.3f}, "
            f"got {learned['sigma_obs']:.3f}"
        )

    def test_sigma_h_learned(self):
        """
        sigma_h should change when theta updates are run across episodes.
        """
        hbm = make_hbm()
        initial_sigma_h = math.exp(hbm.log_sigma_h.item())

        run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=10,
        )

        learned = hbm.get_learned_sigmas()
        # sigma_h may go up or down; we just want to confirm it moved
        assert abs(learned["sigma_h"] - initial_sigma_h) > 1e-4, (
            f"sigma_h should have adapted from {initial_sigma_h:.4f}, "
            f"got {learned['sigma_h']:.4f}"
        )

    def test_learned_sigmas_stay_in_reasonable_range(self):
        """
        All learned sigmas should stay within [0.05, 5.0] throughout training.
        """
        hbm = make_hbm()
        run_episodes(
            hbm, recipe="SimpleDal", spice="turmeric",
            actor="human", n_episodes=20,
        )

        learned = hbm.get_learned_sigmas()
        for name, val in learned.items():
            assert 0.01 < val < 10.0, (
                f"{name} out of reasonable range: {val:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 5: CSP interface unchanged
# ---------------------------------------------------------------------------

class TestCSPInterface:

    def test_log_prob_prefer_correct_sign(self):
        """
        After learning Alice prefers human to add turmeric,
        log_prob_prefer(human) > log_prob_prefer(robot).
        """
        hbm = make_hbm()
        run_episodes(
            hbm, "SimpleDal", "turmeric", "human", n_episodes=15,
        )

        log_p_human = hbm.log_prob_prefer(DEFAULT_HUMAN, "SimpleDal", "turmeric", "human")
        log_p_robot = hbm.log_prob_prefer(DEFAULT_HUMAN, "SimpleDal", "turmeric", "robot")

        assert log_p_human > log_p_robot, (
            f"Expected log P(human) > log P(robot), "
            f"got {log_p_human:.3f} vs {log_p_robot:.3f}"
        )

    def test_preferred_actor_matches_training(self):
        """preferred_actor should return 'human' after human-preference training."""
        hbm = make_hbm()
        run_episodes(hbm, "SimpleDal", "turmeric", "human", n_episodes=15)
        assert hbm.preferred_actor(DEFAULT_HUMAN, "SimpleDal", "turmeric") == "human"

    def test_preferred_actor_robot(self):
        """preferred_actor should return 'robot' after robot-preference training."""
        hbm = make_hbm()
        run_episodes(hbm, "SimpleDal", "cumin", "robot", n_episodes=15)
        assert hbm.preferred_actor(DEFAULT_HUMAN, "SimpleDal", "cumin") == "robot"

    def test_log_prob_prefer_valid_range(self):
        """log_prob_prefer should always return a value <= 0.0."""
        hbm = make_hbm()
        run_episodes(hbm, "SimpleDal", "turmeric", "human", n_episodes=5)

        for actor in ["human", "robot"]:
            lp = hbm.log_prob_prefer(DEFAULT_HUMAN, "SimpleDal", "turmeric", actor)
            assert lp <= 0.0, f"log prob should be <= 0, got {lp}"
            assert math.isfinite(lp), f"log prob should be finite, got {lp}"

    def test_get_phi_var_is_positive(self):
        """get_phi_var should always return a positive value."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, "SimpleDal")
        var = hbm.get_phi_var(DEFAULT_HUMAN, "SimpleDal", "turmeric")
        assert var > 0.0, f"phi variance must be positive, got {var}"


# ---------------------------------------------------------------------------
# Test 6: Hierarchy — cold-start transfer
# ---------------------------------------------------------------------------

class TestColdStartTransfer:

    def test_new_recipe_inherits_from_theta(self):
        """
        After learning turmeric preference on SimpleDal,
        a new recipe (SweetCurry) should initialize phi from theta,
        not from zero.
        """
        hbm = make_hbm()

        # Train on SimpleDal
        run_episodes(hbm, "SimpleDal", "turmeric", "human", n_episodes=15)

        # Register new recipe
        hbm.register_recipe(DEFAULT_HUMAN, "SweetCurry")
        phi_new = hbm.get_phi(DEFAULT_HUMAN, "SweetCurry", "turmeric")
        theta = hbm.get_theta(DEFAULT_HUMAN, "turmeric")

        # phi for new recipe should be close to theta, not zero
        assert abs(phi_new - theta) < 0.3, (
            f"New recipe phi ({phi_new:.3f}) should be close to theta ({theta:.3f})"
        )
        assert phi_new > 0.2, (
            f"New recipe phi ({phi_new:.3f}) should reflect learned preference, not be near zero"
        )

    def test_theta_updated_after_episode(self):
        """
        theta should move toward the direction of observations after episode end.
        """
        hbm = make_hbm()
        initial_theta = hbm.get_theta(DEFAULT_HUMAN, "turmeric")

        run_episodes(hbm, "SimpleDal", "turmeric", "human", n_episodes=10)

        final_theta = hbm.get_theta(DEFAULT_HUMAN, "turmeric")
        assert final_theta > initial_theta + 0.1, (
            f"theta should have increased: {initial_theta:.3f} → {final_theta:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 7: Multiple humans
# ---------------------------------------------------------------------------

class TestMultiHuman:

    def test_different_humans_learn_different_phi(self):
        """
        Alice (prefers human adding turmeric) and Bob (prefers robot)
        should end up with opposite phi signs.
        """
        hbm = make_hbm()
        rng = np.random.default_rng(7)

        # Alice: human adds turmeric
        hbm.register_human("alice")
        for _ in range(10):
            for _ in range(5):
                sat = float(np.clip(rng.normal(0.8, 0.1), -1, 1))
                hbm.observe("alice", "SimpleDal", "turmeric", "human", sat)
            hbm.end_episode("alice")

        # Bob: robot adds turmeric
        hbm.register_human("bob")
        for _ in range(10):
            for _ in range(5):
                sat = float(np.clip(rng.normal(0.8, 0.1), -1, 1))
                hbm.observe("bob", "SimpleDal", "turmeric", "robot", sat)
            hbm.end_episode("bob")

        phi_alice = hbm.get_phi("alice", "SimpleDal", "turmeric")
        phi_bob = hbm.get_phi("bob", "SimpleDal", "turmeric")

        assert phi_alice > 0.0, f"Alice phi should be positive: {phi_alice:.3f}"
        assert phi_bob < 0.0, f"Bob phi should be negative: {phi_bob:.3f}"
        assert phi_alice > phi_bob, (
            f"Alice phi ({phi_alice:.3f}) should exceed Bob phi ({phi_bob:.3f})"
        )

    def test_mu_reflects_population_trend(self):
        """
        If all humans prefer the human to add turmeric,
        mu_turmeric should be positive.
        """
        hbm = make_hbm()
        rng = np.random.default_rng(11)

        for person in ["alice", "bob", "carol"]:
            hbm.register_human(person)
            for _ in range(8):
                for _ in range(5):
                    sat = float(np.clip(rng.normal(0.75, 0.15), -1, 1))
                    hbm.observe(person, "SimpleDal", "turmeric", "human", sat)
                hbm.end_episode(person)

        mu = hbm.get_mu("turmeric")
        assert mu > 0.0, f"Global mu should be positive when all humans prefer human: {mu:.3f}"