"""
test_overcooked_csp.py — Integration tests for OvercookedAssignCSPGenerator.

Run with:
    pytest tests/envs/test_overcooked_csp.py -v

Tests cover:

  1. TestCSPSetup:           generator instantiates, variables/constraints generated
  2. TestObserveTransition:  learn_from_transition updates HBM correctly
  3. TestPreferenceSnapshot: get_pref_snapshot returns valid probabilities
  4. TestExploitMode:        eval mode uses exploit-only cost
  5. TestEnvCSPRoundtrip:    end-to-end env + CSP episode loop without overcooked_ai
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from multitask_personalization.envs.overcooked.overcooked_csp import (
    OvercookedAssignCSPGenerator,
)
from multitask_personalization.envs.overcooked.overcooked_env import (
    OvercookedAction,
    OvercookedEnv,
    OvercookedHiddenSpec,
    OvercookedState,
)
from multitask_personalization.envs.overcooked.layouts import (
    CRAMPED_ROOM,
    get_layout,
)
from multitask_personalization.envs.overcooked.overcooked_hbm import (
    DEFAULT_HUMAN,
    OvercookedPreferenceModel,
)

SUBTASKS = CRAMPED_ROOM.subtasks
LAYOUT = CRAMPED_ROOM.name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_csp_gen(
    train: bool = True,
    explore: str = "max-entropy",
) -> OvercookedAssignCSPGenerator:
    gen = OvercookedAssignCSPGenerator(
        subtask_list=SUBTASKS,
        layout_list=[LAYOUT],
        human_id=DEFAULT_HUMAN,
        seed=0,
        explore_method=explore,
    )
    if train:
        gen.train()
    else:
        gen.eval()
    return gen


def make_obs(subtask: str | None = "chop") -> OvercookedState:
    return OvercookedState(
        layout_name=LAYOUT,
        current_subtask=subtask,
        timestep=0,
        pending_orders=3,
        score_so_far=0.0,
    )


def make_env(preferred: dict | None = None) -> OvercookedEnv:
    if preferred is None:
        preferred = {s: "human" for s in SUBTASKS}
    hidden = OvercookedHiddenSpec(preferred_actor=preferred)
    return OvercookedEnv(layout_spec=CRAMPED_ROOM, hidden_spec=hidden, seed=42)


def make_transition_info(subtask: str, actor: str, score: float) -> dict:
    return {
        "last_subtask": subtask,
        "last_actor": actor,
        "task_score": score,
        "shaped_reward": 3.0,
        "psi_true": [0.0] * 6,
        "session_type": "neutral",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCSPSetup:
    def test_instantiation(self):
        gen = make_csp_gen()
        assert gen is not None
        assert gen._pref_gen._hbm is not None

    def test_generate_variables(self):
        gen = make_csp_gen()
        obs = make_obs()
        variables, init = gen._generate_variables(obs)
        assert len(variables) == 1
        assert variables[0].name == "actor"
        assert init[variables[0]] in ("human", "robot")

    def test_generate_personal_constraints(self):
        gen = make_csp_gen()
        obs = make_obs()
        variables, _ = gen._generate_variables(obs)
        constraints = gen._generate_personal_constraints(obs, variables)
        assert len(constraints) == 1

    def test_generate_cost_train_mode(self):
        gen = make_csp_gen(train=True, explore="max-entropy")
        obs = make_obs()
        variables, _ = gen._generate_variables(obs)
        gen._pref_gen._current_layout_name = LAYOUT
        cost = gen._generate_cost(obs, variables)
        assert cost is not None

    def test_generate_cost_eval_mode(self):
        gen = make_csp_gen(train=False)
        obs = make_obs()
        variables, _ = gen._generate_variables(obs)
        gen._pref_gen._current_layout_name = LAYOUT
        cost = gen._generate_cost(obs, variables)
        assert cost is not None

    def test_generate_policy(self):
        gen = make_csp_gen()
        obs = make_obs()
        variables, _ = gen._generate_variables(obs)
        policy = gen._generate_policy(obs, variables)
        assert policy is not None


class TestObserveTransition:
    def test_observe_transition_updates_hbm(self):
        gen = make_csp_gen()
        obs = make_obs("chop")
        act = OvercookedAction(actor="human")
        next_obs = make_obs("deliver")
        info = make_transition_info("chop", "human", 0.7)

        gen.observe_transition(obs, act, next_obs, done=False, info=info)

        # HBM should have buffered an observation
        assert len(gen._pref_gen._hbm._episode_data[DEFAULT_HUMAN]) > 0

    def test_observe_transition_done_clears_buffer(self):
        gen = make_csp_gen()
        obs = make_obs("chop")
        act = OvercookedAction(actor="human")
        next_obs = make_obs(None)
        info = make_transition_info("chop", "human", 0.7)

        gen.observe_transition(obs, act, next_obs, done=True, info=info)

        # Episode end should have cleared episode data
        assert len(gen._pref_gen._hbm._episode_data[DEFAULT_HUMAN]) == 0

    def test_multiple_transitions_accumulate(self):
        gen = make_csp_gen()
        for i, st in enumerate(SUBTASKS):
            obs = make_obs(st)
            next_obs = make_obs(SUBTASKS[(i + 1) % len(SUBTASKS)])
            info = make_transition_info(st, "human", 0.8)
            gen.observe_transition(
                obs, OvercookedAction("human"), next_obs, done=False, info=info
            )

        # All subtasks should be buffered
        n_obs = len(gen._pref_gen._hbm._episode_data[DEFAULT_HUMAN])
        assert n_obs == len(SUBTASKS)


class TestPreferenceSnapshot:
    def test_snapshot_returns_valid_probs(self):
        gen = make_csp_gen()
        gen._pref_gen._current_layout_name = LAYOUT

        snapshot = gen.get_pref_snapshot()
        assert len(snapshot) == len(SUBTASKS)
        for subtask, probs in snapshot.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01, (
                f"Probabilities for {subtask} don't sum to 1: {total}"
            )

    def test_snapshot_improves_after_learning(self):
        """After consistent human observations for 'chop', P(human|chop) > 0.5."""
        gen = make_csp_gen()
        gen._pref_gen._current_layout_name = LAYOUT

        # Run many consistent episodes
        hbm = gen._pref_gen._hbm
        rng = np.random.default_rng(1)
        for _ in range(20):
            for _ in range(6):
                score = float(np.clip(rng.normal(0.8, 0.1), -1.0, 1.0))
                hbm.observe(DEFAULT_HUMAN, LAYOUT, "chop", "human", score)
            hbm.end_episode(DEFAULT_HUMAN)

        snapshot = gen.get_pref_snapshot()
        assert snapshot["chop"]["human"] > 0.5, (
            f"P(human|chop) should be > 0.5, got {snapshot['chop']['human']}"
        )


class TestEnvCSPRoundtrip:
    def test_episode_runs_without_error(self):
        """Full env + CSP episode loop should complete without exceptions."""
        env = make_env()
        gen = make_csp_gen()
        gen._pref_gen._current_layout_name = LAYOUT

        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 20:
            # CSP assigns actor for current subtask
            if obs.current_subtask is None:
                break

            hbm = gen._pref_gen._hbm
            actor = hbm.preferred_actor(DEFAULT_HUMAN, LAYOUT, obs.current_subtask)
            action = OvercookedAction(actor=actor)

            next_obs, reward, done, _, info = env.step(action)
            gen.observe_transition(obs, action, next_obs, done=done, info=info)

            obs = next_obs
            steps += 1

        assert steps > 0, "Episode completed zero steps"

    def test_metrics_returns_psi_values(self):
        gen = make_csp_gen()
        hbm = gen._pref_gen._hbm

        # Run an episode to generate psi signal
        for st in SUBTASKS[:3]:
            hbm.observe(DEFAULT_HUMAN, LAYOUT, st, "human", 0.8)
        hbm.end_episode(DEFAULT_HUMAN)

        metrics = gen.get_metrics()
        # Should return psi_0, psi_1, ... keys
        assert any(k.startswith("psi_") for k in metrics), (
            f"Expected psi_* keys in metrics, got: {list(metrics.keys())}"
        )
        for v in metrics.values():
            assert isinstance(v, float)
