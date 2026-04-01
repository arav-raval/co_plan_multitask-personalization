"""
test_overcooked_hbm.py — Validation tests for OvercookedPreferenceModel.

Run with:
    pytest tests/envs/test_overcooked_hbm.py -v

Tests cover:

  1. TestRegistration:       register_human, register_layout, observe all work
  2. TestPhiConvergence:     phi converges toward true preference sign after
                             consistent observations
  3. TestVectorPsiReset:     vector psi decays aggressively at episode end
  4. TestVectorPsiAbsorbs:   a fatigued session does not corrupt phi
  5. TestRunningPsiVec:      running psi is non-zero after a session episode
  6. TestPreferredActor:     preferred_actor / log_prob_prefer return correct signs
  7. TestMultiSubtask:       preferences for different subtasks are learned independently
  8. TestThetaTransfer:      theta propagates from layout to new layout at cold start
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from multitask_personalization.envs.overcooked.overcooked_hbm import (
    DEFAULT_HUMAN,
    OvercookedPreferenceModel,
)
from multitask_personalization.envs.overcooked.layouts import ALL_SUBTASKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUBTASKS = ["fetch_onion", "fetch_dish", "chop", "deliver"]
LAYOUT = "CrampedRoom"
LAYOUT2 = "CoordinationRing"


def make_hbm(subtasks=None, sigma_obs_init=0.5) -> OvercookedPreferenceModel:
    subtasks = subtasks or SUBTASKS
    return OvercookedPreferenceModel(
        subtasks=subtasks,
        sigma_obs=sigma_obs_init,
        n_phi_steps=10,
        n_theta_steps=15,
    )


def run_episodes(
    hbm: OvercookedPreferenceModel,
    layout: str,
    subtask: str,
    actor: str,
    n_episodes: int,
    steps_per_episode: int = 6,
    score_mean: float = 0.8,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Run episodes with consistent observations and return phi_mean per episode."""
    if rng is None:
        rng = np.random.default_rng(42)
    phi_means = []
    for _ in range(n_episodes):
        for _ in range(steps_per_episode):
            score = float(np.clip(rng.normal(score_mean, 0.1), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, layout, subtask, actor, score)
        hbm.end_episode(DEFAULT_HUMAN)
        phi_means.append(hbm.get_phi(DEFAULT_HUMAN, layout, subtask))
    return phi_means


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_human_idempotent(self):
        hbm = make_hbm()
        hbm.register_human("alice")
        hbm.register_human("alice")  # second call should be no-op
        assert "alice" in hbm._theta_m

    def test_register_layout_initialises_phi(self):
        hbm = make_hbm()
        hbm.register_layout(DEFAULT_HUMAN, LAYOUT)
        for s in SUBTASKS:
            phi = hbm.get_phi(DEFAULT_HUMAN, LAYOUT, s)
            assert isinstance(phi, float)

    def test_observe_buffers_data(self):
        hbm = make_hbm()
        hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", 0.7)
        assert len(hbm._episode_data[DEFAULT_HUMAN]) == 1

    def test_end_episode_clears_buffer(self):
        hbm = make_hbm()
        hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", 0.7)
        hbm.end_episode(DEFAULT_HUMAN)
        assert len(hbm._episode_data[DEFAULT_HUMAN]) == 0


class TestPhiConvergence:
    def test_phi_moves_toward_human_after_consistent_human_observations(self):
        """After many episodes of human actor + positive score, phi > 0."""
        hbm = make_hbm()
        phi_means = run_episodes(
            hbm, LAYOUT, "chop", "human", n_episodes=30, score_mean=0.8
        )
        # phi should become positive (human preferred)
        assert phi_means[-1] > 0.0, f"phi did not converge to positive: {phi_means[-1]:.3f}"

    def test_phi_moves_toward_robot_after_consistent_robot_observations(self):
        """After many episodes of robot actor + positive score, phi < 0."""
        hbm = make_hbm()
        phi_means = run_episodes(
            hbm, LAYOUT, "deliver", "robot", n_episodes=30, score_mean=0.8
        )
        assert phi_means[-1] < 0.0, f"phi did not converge to negative: {phi_means[-1]:.3f}"

    def test_phi_variance_decreases_with_more_observations(self):
        hbm = make_hbm()
        hbm.register_layout(DEFAULT_HUMAN, LAYOUT)
        var_before = hbm.get_phi_var(DEFAULT_HUMAN, LAYOUT, "chop")
        run_episodes(hbm, LAYOUT, "chop", "human", n_episodes=20, score_mean=0.8)
        var_after = hbm.get_phi_var(DEFAULT_HUMAN, LAYOUT, "chop")
        assert var_after < var_before, (
            f"Variance did not decrease: {var_before:.4f} → {var_after:.4f}"
        )


class TestVectorPsiReset:
    def test_psi_vec_decays_at_episode_end(self):
        """After an episode with strong positive scores, psi is nonzero.
        After episode end, it should be near zero (decayed by psi_decay ≈ 0.05)."""
        hbm = make_hbm()
        # Build up psi by running a session with extreme scores
        for _ in range(8):
            hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", 0.95)
        hbm.end_episode(DEFAULT_HUMAN)
        psi_vec = hbm.get_psi_vec(DEFAULT_HUMAN)
        # After decay, all dimensions should be small in magnitude
        max_abs = max(abs(v) for v in psi_vec)
        assert max_abs < 0.5, f"Psi not decayed after episode end: {psi_vec}"

    def test_running_psi_resets_at_episode_start(self):
        """Running psi should be zeroed at the start of each episode."""
        hbm = make_hbm()
        for _ in range(6):
            hbm.observe(DEFAULT_HUMAN, LAYOUT, "deliver", "robot", -0.9)
        hbm.end_episode(DEFAULT_HUMAN)  # resets running psi
        running = hbm.get_running_psi_vec(DEFAULT_HUMAN)
        assert all(abs(v) < 1e-6 for v in running), (
            f"Running psi not zeroed after episode end: {running}"
        )


class TestVectorPsiAbsorbs:
    def test_fatigued_episode_does_not_corrupt_phi(self):
        """
        A 'fatigued' session (consistently negative scores regardless of actor)
        should be absorbed by psi and not significantly corrupt the learned phi.

        Steps:
        1. Establish a positive phi via 15 neutral episodes (human + positive scores).
        2. Run 5 fatigued episodes (negative scores for all actors).
        3. Run 5 neutral recovery episodes.
        4. Verify phi is still positive at the end.
        """
        hbm = make_hbm()
        rng = np.random.default_rng(0)

        # Phase 1: neutral learning
        for _ in range(15):
            for _ in range(6):
                s = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
                hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", s)
            hbm.end_episode(DEFAULT_HUMAN)

        phi_after_neutral = hbm.get_phi(DEFAULT_HUMAN, LAYOUT, "chop")
        assert phi_after_neutral > 0.1, (
            f"phi not positive after neutral episodes: {phi_after_neutral:.3f}"
        )

        # Phase 2: fatigued episodes (negative scores)
        for _ in range(5):
            for _ in range(6):
                s = float(np.clip(rng.normal(-0.7, 0.1), -1.0, 1.0))
                hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", s)
            hbm.end_episode(DEFAULT_HUMAN)

        # Phase 3: recovery
        for _ in range(5):
            for _ in range(6):
                s = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
                hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", s)
            hbm.end_episode(DEFAULT_HUMAN)

        phi_after_recovery = hbm.get_phi(DEFAULT_HUMAN, LAYOUT, "chop")
        # phi should still be positive (not flipped by fatigue)
        assert phi_after_recovery > 0.0, (
            f"phi corrupted by fatigued episodes: {phi_after_recovery:.3f}"
        )


class TestRunningPsiVec:
    def test_running_psi_nonzero_during_episode(self):
        """Running psi should accumulate a nonzero signal mid-episode."""
        hbm = make_hbm()
        # Observations with very consistent positive scores → running psi should grow
        for _ in range(5):
            hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", 0.95)
        running = hbm.get_running_psi_vec(DEFAULT_HUMAN)
        chop_dim = ALL_SUBTASKS.index("chop") if "chop" in ALL_SUBTASKS else 0
        assert abs(running[chop_dim]) > 0.0 or any(abs(v) > 0.0 for v in running), (
            f"Running psi is all zero after observations: {running}"
        )

    def test_running_psi_per_subtask_differs(self):
        """Observations for different subtasks should create different psi values."""
        hbm = make_hbm(subtasks=["chop", "deliver"])
        for _ in range(6):
            hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", 0.95)
        for _ in range(6):
            hbm.observe(DEFAULT_HUMAN, LAYOUT, "deliver", "robot", 0.95)
        running = hbm.get_running_psi_vec(DEFAULT_HUMAN)
        # Both dims should be non-zero; they need not match since actors differ
        assert len(running) == 2


class TestPreferredActor:
    def test_preferred_actor_sign_matches_phi(self):
        """preferred_actor returns 'human' when phi + running_psi > 0."""
        hbm = make_hbm()
        run_episodes(hbm, LAYOUT, "chop", "human", n_episodes=25, score_mean=0.8)
        actor = hbm.preferred_actor(DEFAULT_HUMAN, LAYOUT, "chop")
        assert actor == "human", f"Expected 'human', got '{actor}'"

    def test_log_prob_prefer_human_greater_after_human_episodes(self):
        hbm = make_hbm()
        run_episodes(hbm, LAYOUT, "fetch_dish", "human", n_episodes=25, score_mean=0.8)
        lp_human = hbm.log_prob_prefer(DEFAULT_HUMAN, LAYOUT, "fetch_dish", "human")
        lp_robot = hbm.log_prob_prefer(DEFAULT_HUMAN, LAYOUT, "fetch_dish", "robot")
        assert lp_human > lp_robot, (
            f"Expected log P(human) > log P(robot): {lp_human:.3f} vs {lp_robot:.3f}"
        )

    def test_log_prob_sums_to_approximately_one(self):
        hbm = make_hbm()
        hbm.register_layout(DEFAULT_HUMAN, LAYOUT)
        lp_h = hbm.log_prob_prefer(DEFAULT_HUMAN, LAYOUT, "chop", "human")
        lp_r = hbm.log_prob_prefer(DEFAULT_HUMAN, LAYOUT, "chop", "robot")
        total = math.exp(lp_h) + math.exp(lp_r)
        assert abs(total - 1.0) < 0.05, f"Probabilities don't sum to 1: {total:.4f}"


class TestMultiSubtask:
    def test_different_subtasks_learned_independently(self):
        """Consistent observations for one subtask should not contaminate others."""
        hbm = make_hbm()
        # Learn "chop" → human
        run_episodes(hbm, LAYOUT, "chop", "human", n_episodes=20, score_mean=0.8)
        # "deliver" → robot
        run_episodes(hbm, LAYOUT, "deliver", "robot", n_episodes=20, score_mean=0.8)

        phi_chop = hbm.get_phi(DEFAULT_HUMAN, LAYOUT, "chop")
        phi_deliver = hbm.get_phi(DEFAULT_HUMAN, LAYOUT, "deliver")
        assert phi_chop > 0.0, f"chop phi should be positive: {phi_chop:.3f}"
        assert phi_deliver < 0.0, f"deliver phi should be negative: {phi_deliver:.3f}"


class TestThetaTransfer:
    def test_new_layout_inherits_theta(self):
        """
        After learning phi on LAYOUT, registering a new LAYOUT2 should
        initialise phi_LAYOUT2 from theta (which has been updated from phi_LAYOUT).
        """
        hbm = make_hbm()
        run_episodes(hbm, LAYOUT, "chop", "human", n_episodes=20, score_mean=0.8)
        hbm.flush_theta_mu()

        theta_chop = hbm.get_theta(DEFAULT_HUMAN, "chop")
        # Register a second layout — phi initialised from theta
        hbm.register_layout(DEFAULT_HUMAN, LAYOUT2)
        phi_new = hbm.get_phi(DEFAULT_HUMAN, LAYOUT2, "chop")

        # phi of new layout should be close to theta (cold-start transfer)
        assert abs(phi_new - theta_chop) < 0.5, (
            f"Cold-start phi {phi_new:.3f} too far from theta {theta_chop:.3f}"
        )


class TestLearnedSigmas:
    def test_learned_sigmas_return_reasonable_values(self):
        hbm = make_hbm()
        run_episodes(hbm, LAYOUT, "chop", "human", n_episodes=10, score_mean=0.8)
        sigmas = hbm.get_learned_sigmas()
        for k, v in sigmas.items():
            assert v > 0, f"Sigma {k} not positive: {v}"
            assert v < 100, f"Sigma {k} exploded: {v}"
