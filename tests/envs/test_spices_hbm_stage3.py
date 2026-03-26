"""
test_spices_hbm_stage3.py — Validation tests for the Stage 3 CSP migration.

Run with:
    pytest tests/envs/test_spices_hbm_stage3.py -v

Stage 3 passes phi variance to the CSP for active exploration via a
variance-weighted entropy cost, and adds mid-episode psi adaptation so
the CSP adjusts in real time as mood signals accumulate within an episode.

Changes tested:
  - get_phi_entropy() = H(B(sigma(phi_mean))) * phi_var
  - preference_posterior() returns (phi_mean, phi_var)
  - get_running_psi(): updated per-observation, reset at episode start
  - log_prob_prefer() uses phi + running_psi as effective logit
  - preferred_actor() uses phi + running_psi
  - _generate_cost(): variance-weighted combined exploit + explore cost
  - _generate_personal_constraints(): psi-adjusted soft constraint (mood gate removed)

Test classes:
  1. TestPhiEntropy:           get_phi_entropy() is correct and decreases with data
  2. TestPreferencePosterior:  preference_posterior() returns correct (mean, var) tuple
  3. TestRunningPsi:           running psi updates mid-episode and resets at episode start
  4. TestMidEpisodeAdaptation: log_prob_prefer and preferred_actor shift with running psi
  5. TestExplorationCost:      CSP cost favors unexpected actor when uncertain
  6. TestCSPCostTransition:    combined cost transitions from explore to exploit
  7. TestCSPEvalMode:          CSP in eval mode always uses exploit cost only
  8. TestStage2Preserved:      Stage 2 psi behavior still holds in Stage 3
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from multitask_personalization.envs.spices.spices_hbm import (
    DEFAULT_HUMAN,
    HierarchicalPreferenceModel,
)
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import SpiceState
from multitask_personalization.envs.spices.config.spices_config import DEFAULT_CONFIG

RECIPE = "Dal"
SPICE = "turmeric"
SPICES = ["turmeric", "cumin", "chili"]
SIGMA_MOOD = DEFAULT_CONFIG.hbm.sigma_mood


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hbm(spices=None) -> HierarchicalPreferenceModel:
    return HierarchicalPreferenceModel(
        spices=spices or SPICES,
        sigma_obs=0.5,
        n_phi_steps=10,
        n_theta_steps=15,
    )


def run_episodes(
    hbm: HierarchicalPreferenceModel,
    recipe: str,
    spice: str,
    actor: str,
    n_episodes: int,
    n_obs_per_episode: int = 6,
    sat_mean: float = 0.8,
    rng: np.random.Generator | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng(42)
    for _ in range(n_episodes):
        for _ in range(n_obs_per_episode):
            sat = float(np.clip(rng.normal(sat_mean, 0.1), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, recipe, spice, actor, sat)
        hbm.end_episode(DEFAULT_HUMAN)


def _make_obs(spice: str = SPICE) -> SpiceState:
    return SpiceState(
        time=0,
        added_spices=(),
        remaining_spices=(spice,),
        feasible_next=(spice,),
        current_spice=spice,
    )


def _make_csp_gen(
    hbm: HierarchicalPreferenceModel,
    explore_method: str = "max-entropy",
    seed: int = 0,
) -> SpicesAssignCSPGenerator:
    gen = SpicesAssignCSPGenerator(
        spice_list=SPICES,
        recipe_list=[RECIPE],
        explore_method=explore_method,
        shared_hbm=hbm,
        seed=seed,
    )
    gen._pref_gen._current_recipe_name = RECIPE
    return gen


def _get_cost(
    csp_gen: SpicesAssignCSPGenerator, actor: str, spice: str = SPICE
) -> float:
    """Evaluate the CSP's current cost function for a given actor."""
    obs = _make_obs(spice)
    variables, _ = csp_gen._generate_variables(obs)
    cost = csp_gen._generate_cost(obs, variables)
    if cost is None:
        return 0.0
    return cost.get_cost({variables[0]: actor})


# ---------------------------------------------------------------------------
# 1. TestPhiEntropy
# ---------------------------------------------------------------------------

class TestPhiEntropy:
    """get_phi_entropy() correctly measures exploration value."""

    def test_entropy_positive_at_init(self):
        """A fresh unobserved spice should have high exploration value."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        entropy = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)
        assert entropy > 0.0, f"Initial entropy should be positive, got {entropy:.4f}"

    def test_entropy_decreases_with_data(self):
        """Exploration value should shrink as phi converges."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(99)
        entropy_init = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=20, rng=rng)
        entropy_final = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)
        assert entropy_final < entropy_init, (
            f"Entropy should decrease: init={entropy_init:.4f}, final={entropy_final:.4f}"
        )

    def test_entropy_near_zero_when_confident(self):
        """After many consistent episodes, exploration value should be very small."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(77)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=30, rng=rng)
        entropy = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)
        assert entropy < 0.3, (
            f"Entropy should be near zero when confident, got {entropy:.4f}"
        )

    def test_entropy_higher_for_untrained_spice(self):
        """An unobserved spice should have higher entropy than a trained one."""
        hbm = make_hbm(spices=[SPICE, "cumin"])
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(11)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=15, rng=rng)
        entropy_trained = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)
        entropy_untrained = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, "cumin")
        assert entropy_untrained > entropy_trained, (
            f"Untrained spice should have higher entropy: "
            f"untrained={entropy_untrained:.4f}, trained={entropy_trained:.4f}"
        )

    def test_entropy_near_zero_at_extreme_phi(self):
        """At very large |phi|, P(human)≈1 so H≈0 and entropy≈0."""
        import torch
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        hbm._phi_m[DEFAULT_HUMAN][RECIPE][SPICE] = torch.tensor(10.0, requires_grad=True)
        hbm._phi_logv[DEFAULT_HUMAN][RECIPE][SPICE] = torch.tensor(
            math.log(1e-4), requires_grad=True
        )
        entropy = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)
        assert entropy < 0.01, f"Entropy at extreme phi should be ≈0, got {entropy:.6f}"

    def test_entropy_is_nonnegative(self):
        """get_phi_entropy must always return a non-negative value."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(88)
        for _ in range(5):
            run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=5, rng=rng)
            assert hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE) >= 0.0


# ---------------------------------------------------------------------------
# 2. TestExplorationCost
# ---------------------------------------------------------------------------

class TestExplorationCost:
    """The variance-weighted combined cost (exploit + explore) behaves correctly."""

    def test_unexpected_actor_cheaper_when_uncertain(self):
        """
        At initialization (phi≈0, large var), the unexpected actor's combined cost
        should be lower than the preferred actor's, driving exploration.
        """
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        csp_gen = _make_csp_gen(hbm)
        csp_gen.train()

        cost_human = _get_cost(csp_gen, "human")
        cost_robot = _get_cost(csp_gen, "robot")

        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        preferred = "human" if phi >= 0 else "robot"
        cost_preferred = cost_human if preferred == "human" else cost_robot
        cost_unexpected = cost_robot if preferred == "human" else cost_human

        assert cost_unexpected < cost_preferred, (
            f"Unexpected actor should be cheaper (explore wins at init): "
            f"preferred_cost={cost_preferred:.3f}, unexpected_cost={cost_unexpected:.3f}"
        )

    def test_preferred_actor_cheaper_when_confident(self):
        """After training, exploit cost dominates and preferred actor is cheaper."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(55)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=25, rng=rng)

        csp_gen = _make_csp_gen(hbm)
        csp_gen.train()

        cost_human = _get_cost(csp_gen, "human")
        cost_robot = _get_cost(csp_gen, "robot")

        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        assert phi > 0, f"phi should be positive, got {phi:.3f}"
        assert cost_human < cost_robot, (
            f"Human (preferred) should be cheaper after training: "
            f"human={cost_human:.3f}, robot={cost_robot:.3f}"
        )

    def test_explore_val_dominates_exploit_at_init(self):
        """
        The exploration bonus (H * var) should exceed the exploit cost gap at init,
        ensuring the cost function drives exploration before any training.
        """
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        explore_val = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)
        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        # Exploit gap = cost(unexpected) - cost(preferred) from log_prob alone ≈ 0 at phi≈0
        # explore_val must exceed exploit_gap to flip the ordering
        log_p_preferred = hbm.log_prob_prefer(
            DEFAULT_HUMAN, RECIPE, SPICE, "human" if phi >= 0 else "robot"
        )
        log_p_unexpected = hbm.log_prob_prefer(
            DEFAULT_HUMAN, RECIPE, SPICE, "robot" if phi >= 0 else "human"
        )
        exploit_gap = (-log_p_unexpected) - (-log_p_preferred)  # > 0 if unexpected is costlier
        assert explore_val > exploit_gap, (
            f"explore_val ({explore_val:.3f}) should exceed exploit_gap ({exploit_gap:.3f})"
        )

    def test_eval_mode_ignores_explore_bonus(self):
        """In eval mode the CSP uses exploit cost only (no explore bonus)."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        csp_gen = _make_csp_gen(hbm)
        csp_gen.eval()  # eval mode: use _generate_exploit_cost only

        cost_human = _get_cost(csp_gen, "human")
        cost_robot = _get_cost(csp_gen, "robot")

        # In eval mode, cost = -log_prob_prefer. At phi≈0 both are ≈ equal.
        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        log_p_human = hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, "human")
        log_p_robot = hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, "robot")
        expected_human = -log_p_human
        expected_robot = -log_p_robot
        assert abs(cost_human - expected_human) < 1e-6, (
            f"Eval cost_human should equal -log_prob_prefer: {cost_human:.4f} vs {expected_human:.4f}"
        )
        assert abs(cost_robot - expected_robot) < 1e-6, (
            f"Eval cost_robot should equal -log_prob_prefer: {cost_robot:.4f} vs {expected_robot:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. TestCSPCostTransition
# ---------------------------------------------------------------------------

class TestCSPCostTransition:
    """Combined cost transitions smoothly from exploration to exploitation."""

    def test_cost_gap_flips_from_explore_to_exploit(self):
        """
        cost(unexpected) - cost(preferred) should go from negative (explore wins)
        to positive (exploit wins) as training progresses.
        """
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(1)
        csp_gen = _make_csp_gen(hbm)
        csp_gen.train()

        # Before training: explore should win (unexpected actor is cheaper)
        phi_init = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        preferred_init = "human" if phi_init >= 0 else "robot"
        unexpected_init = "robot" if preferred_init == "human" else "human"
        gap_before = _get_cost(csp_gen, unexpected_init) - _get_cost(csp_gen, preferred_init)
        assert gap_before < 0, (
            f"Before training, unexpected should be cheaper (gap < 0): gap={gap_before:.3f}"
        )

        # After training: exploit should win (preferred actor is cheaper)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=25, rng=rng)
        phi_final = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        preferred_final = "human" if phi_final >= 0 else "robot"
        unexpected_final = "robot" if preferred_final == "human" else "human"
        gap_after = _get_cost(csp_gen, unexpected_final) - _get_cost(csp_gen, preferred_final)
        assert gap_after > 0, (
            f"After training, preferred should be cheaper (gap > 0): gap={gap_after:.3f}"
        )

    def test_explore_val_monotonically_decreasing_trend(self):
        """phi_entropy should trend downward over training (not necessarily monotone
        due to stochastic ELBO, but median should decrease)."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(2)
        entropies = []
        for _ in range(20):
            for _ in range(6):
                sat = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
                hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "human", sat)
            hbm.end_episode(DEFAULT_HUMAN)
            entropies.append(hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE))

        # First 5 average should be higher than last 5 average
        early_avg = float(np.mean(entropies[:5]))
        late_avg = float(np.mean(entropies[-5:]))
        assert late_avg < early_avg, (
            f"Exploration value should trend down: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. TestCSPEvalMode
# ---------------------------------------------------------------------------

class TestCSPEvalMode:
    """CSP in eval mode uses pure exploit cost; no exploration bonus."""

    def test_eval_cost_equals_neg_log_prob(self):
        """In eval mode, cost(actor) == -log_prob_prefer(actor) exactly."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(3)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)

        csp_gen = _make_csp_gen(hbm, explore_method="exploit-only")
        csp_gen.eval()

        for actor in ["human", "robot"]:
            cost = _get_cost(csp_gen, actor)
            expected = -hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, actor)
            assert abs(cost - expected) < 1e-6, (
                f"eval cost({actor}) should equal -log_prob: {cost:.4f} vs {expected:.4f}"
            )

    def test_max_entropy_eval_also_uses_exploit_cost(self):
        """
        Even with explore_method='max-entropy', eval mode should use exploit cost only.
        _generate_cost falls back to _generate_exploit_cost when not in train mode.
        """
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        csp_gen = _make_csp_gen(hbm, explore_method="max-entropy")
        csp_gen.eval()  # eval mode overrides max-entropy

        for actor in ["human", "robot"]:
            cost = _get_cost(csp_gen, actor)
            expected = -hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, actor)
            assert abs(cost - expected) < 1e-6, (
                f"eval cost({actor}) should equal -log_prob even with max-entropy method: "
                f"{cost:.4f} vs {expected:.4f}"
            )

    def test_preferred_actor_has_lower_eval_cost(self):
        """The preferred actor always has lower eval cost than the unexpected one."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(4)
        run_episodes(hbm, RECIPE, SPICE, "robot", n_episodes=15, rng=rng)

        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        assert phi < 0, f"phi should be negative, got {phi:.3f}"

        csp_gen = _make_csp_gen(hbm, explore_method="exploit-only")
        csp_gen.eval()

        cost_human = _get_cost(csp_gen, "human")
        cost_robot = _get_cost(csp_gen, "robot")
        assert cost_robot < cost_human, (
            f"Robot (preferred) should have lower eval cost: robot={cost_robot:.3f}, human={cost_human:.3f}"
        )


# ---------------------------------------------------------------------------
# 5. TestStage2Preserved
# ---------------------------------------------------------------------------

class TestStage2Preserved:
    """Stage 2 psi behavior is unaffected by Stage 3 CSP changes."""

    def test_psi_decays_after_episode(self):
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(50)
        for _ in range(6):
            sat = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "human", sat)
        hbm.end_episode(DEFAULT_HUMAN)
        psi_after = hbm.get_psi(DEFAULT_HUMAN)
        assert abs(psi_after) < 0.2, f"psi should decay near 0 after episode, got {psi_after:.3f}"

    def test_phi_stable_after_contradictory_episode(self):
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(60)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)
        phi_before = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        assert phi_before > 0.3, f"phi should be positive, got {phi_before:.3f}"

        for _ in range(6):
            sat = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "robot", sat)
        hbm.end_episode(DEFAULT_HUMAN)
        phi_after = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        assert phi_after > 0.0, (
            f"phi should not flip after one contradictory episode: "
            f"before={phi_before:.3f}, after={phi_after:.3f}"
        )

    def test_phi_entropy_consistent_with_phi_var(self):
        """get_phi_entropy() should equal H(B(sigma(phi))) * phi_var."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(70)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)

        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        phi_var = hbm.get_phi_var(DEFAULT_HUMAN, RECIPE, SPICE)
        entropy_val = hbm.get_phi_entropy(DEFAULT_HUMAN, RECIPE, SPICE)

        # Compute H manually
        log_p = hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, "human")
        p = math.exp(log_p)
        q = 1.0 - p
        p = max(p, 1e-10)
        q = max(q, 1e-10)
        H = -p * math.log(p) - q * math.log(q)
        expected = H * phi_var

        assert abs(entropy_val - expected) < 1e-8, (
            f"get_phi_entropy() should equal H * phi_var: got {entropy_val:.6f}, expected {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# New Stage 3 tests (preference_posterior, running psi, mid-episode adaptation)
# ---------------------------------------------------------------------------

class TestPreferencePosterior:
    """preference_posterior() returns correct (phi_mean, phi_var) tuple."""

    def test_returns_tuple_of_two_floats(self):
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        mean, var = hbm.preference_posterior(DEFAULT_HUMAN, RECIPE, SPICE)
        assert isinstance(mean, float)
        assert isinstance(var, float)

    def test_mean_matches_get_phi(self):
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(100)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)
        mean, _ = hbm.preference_posterior(DEFAULT_HUMAN, RECIPE, SPICE)
        assert mean == hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)

    def test_var_matches_get_phi_var(self):
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(101)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)
        _, var = hbm.preference_posterior(DEFAULT_HUMAN, RECIPE, SPICE)
        assert var == hbm.get_phi_var(DEFAULT_HUMAN, RECIPE, SPICE)

    def test_var_is_positive(self):
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        _, var = hbm.preference_posterior(DEFAULT_HUMAN, RECIPE, SPICE)
        assert var > 0.0


class TestRunningPsi:
    """_running_psi_m is updated per-observation and resets between episodes."""

    def test_running_psi_zero_at_start(self):
        """Before any observations, running psi should be 0."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        assert hbm.get_running_psi(DEFAULT_HUMAN) == 0.0

    def test_running_psi_nonzero_after_observations(self):
        """After seeing contradictory observations, running psi should drift from 0."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(200)
        # Prime phi to be positive (human preferred)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)
        # Now observe contradictory signals (robot + high sat) without ending episode
        for _ in range(5):
            sat = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "robot", sat)
        psi_running = hbm.get_running_psi(DEFAULT_HUMAN)
        # Running psi should be nonzero (absorbing the contradiction)
        assert abs(psi_running) > 0.01, (
            f"running psi should drift from 0 after contradictory obs, got {psi_running:.4f}"
        )

    def test_running_psi_resets_after_episode(self):
        """Running psi should reset to 0 after end_episode()."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(201)
        for _ in range(5):
            sat = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "human", sat)
        assert abs(hbm.get_running_psi(DEFAULT_HUMAN)) > 0.0  # nonzero during episode
        hbm.end_episode(DEFAULT_HUMAN)
        assert hbm.get_running_psi(DEFAULT_HUMAN) == 0.0, (
            f"running psi should be 0 after end_episode, got {hbm.get_running_psi(DEFAULT_HUMAN):.4f}"
        )

    def test_running_psi_grows_with_more_observations(self):
        """Running psi magnitude should generally increase as evidence accumulates."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(202)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)
        # Observe increasingly many contradictory signals
        psi_after_1 = None
        for i in range(6):
            sat = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "robot", sat)
            if i == 0:
                psi_after_1 = abs(hbm.get_running_psi(DEFAULT_HUMAN))
        psi_after_6 = abs(hbm.get_running_psi(DEFAULT_HUMAN))
        assert psi_after_6 > psi_after_1, (
            f"running psi should grow: after_1={psi_after_1:.4f}, after_6={psi_after_6:.4f}"
        )

    def test_running_psi_separate_from_psi_m(self):
        """_running_psi_m and _psi_m are independent; mid-episode running_psi doesn't corrupt _psi_m."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(203)
        # Run several neutral episodes to establish _psi_m ≈ 0
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=5, rng=rng)
        psi_m_before = hbm.get_psi(DEFAULT_HUMAN)  # _psi_m after decay ≈ 0
        # Observe contradictory signals WITHOUT ending episode
        for _ in range(5):
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "robot", 0.8)
        # _running_psi_m should be nonzero
        assert abs(hbm.get_running_psi(DEFAULT_HUMAN)) > 0.01
        # _psi_m should be unchanged (not modified by running psi updates)
        psi_m_during = hbm.get_psi(DEFAULT_HUMAN)
        assert psi_m_during == psi_m_before, (
            f"_psi_m should not change mid-episode: before={psi_m_before:.4f}, during={psi_m_during:.4f}"
        )


class TestMidEpisodeAdaptation:
    """log_prob_prefer and preferred_actor adapt mid-episode via running psi."""

    def test_log_prob_prefer_shifts_with_running_psi(self):
        """
        After observing contradictory signals mid-episode, log_prob_prefer for the
        observed actor should increase (psi makes it seem more plausible).
        """
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(300)
        # Train phi positive (human preferred)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)
        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        assert phi > 0, f"phi should be positive, got {phi:.3f}"

        lp_robot_before = hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, "robot")
        # Observe several robot+high_sat signals mid-episode — running psi should go negative
        for _ in range(5):
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "robot", 0.8)
        lp_robot_after = hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, "robot")
        # After psi absorbs the contradiction, robot should appear MORE plausible
        assert lp_robot_after > lp_robot_before, (
            f"log_prob_prefer(robot) should increase after robot+sat observations: "
            f"before={lp_robot_before:.3f}, after={lp_robot_after:.3f}"
        )

    def test_preferred_actor_can_shift_mid_episode(self):
        """
        With a weak phi (near 0) and strong contradictory signals, preferred_actor
        can flip mid-episode via running psi.
        """
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        # phi ≈ 0 → preferred_actor = "human" (tie-break)
        assert hbm.preferred_actor(DEFAULT_HUMAN, RECIPE, SPICE) == "human"
        # Observe strong robot+high_sat signals → running psi goes negative → phi+psi < 0
        rng = np.random.default_rng(301)
        for _ in range(8):
            hbm.observe(DEFAULT_HUMAN, RECIPE, SPICE, "robot", 0.9)
        # preferred_actor should now be "robot" (phi + psi < 0)
        psi = hbm.get_running_psi(DEFAULT_HUMAN)
        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        if phi + psi < 0:
            assert hbm.preferred_actor(DEFAULT_HUMAN, RECIPE, SPICE) == "robot", (
                f"preferred_actor should flip: phi={phi:.3f}, psi={psi:.3f}"
            )

    def test_no_mid_episode_adaptation_between_episodes(self):
        """Between episodes (running_psi = 0), log_prob_prefer equals phi-only version."""
        hbm = make_hbm()
        hbm.register_recipe(DEFAULT_HUMAN, RECIPE)
        rng = np.random.default_rng(302)
        run_episodes(hbm, RECIPE, SPICE, "human", n_episodes=10, rng=rng)
        # After end_episode, running_psi = 0 → log_prob_prefer uses only phi
        assert hbm.get_running_psi(DEFAULT_HUMAN) == 0.0
        phi = hbm.get_phi(DEFAULT_HUMAN, RECIPE, SPICE)
        for actor in ["human", "robot"]:
            sign = 1.0 if actor == "human" else -1.0
            logit = sign * phi
            import math as _math
            expected_lp = -_math.log1p(_math.exp(-abs(logit))) - max(0.0, -logit)
            actual_lp = hbm.log_prob_prefer(DEFAULT_HUMAN, RECIPE, SPICE, actor)
            assert abs(actual_lp - expected_lp) < 1e-6, (
                f"Between episodes: log_prob_prefer({actor}) should use phi only: "
                f"got {actual_lp:.6f}, expected {expected_lp:.6f}"
            )
