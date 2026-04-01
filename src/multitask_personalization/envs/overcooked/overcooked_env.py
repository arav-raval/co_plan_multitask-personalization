"""
overcooked_env.py — Gymnasium wrapper for the Overcooked AI environment.

This module wraps ``overcooked_ai_py``'s OvercookedEnv so that it integrates
with our HBM + CSP planning framework.

Key design decisions
--------------------
1.  **Robot agent is agent 0; human is agent 1.**
    The CSP assigns subtasks to the robot.  The human executes a fixed policy
    (random or hand-coded) that the robot does not control.

2.  **Subtask assignment as the action interface.**
    At each "decision step" the CSP assigns the *current subtask* to human or
    robot.  The lower-level overcooked_ai primitive actions (UP/DOWN/INTERACT
    etc.) needed to execute that subtask are produced by a simple scripted
    sub-policy that is NOT part of the preference learning problem.

3.  **Task score as feedback.**
    After each subtask completes, a task_score in [-1, +1] is computed from
    the shaped reward received during execution.  This is the signal passed to
    the HBM via ``observe()``.

4.  **Session (psi) signal.**
    At episode start the environment samples a session type (neutral /
    energised / fatigued) and a per-subtask psi_true vector.  This generates
    the within-episode variation in satisfaction that the HBM's vector psi is
    designed to absorb.

Observation structure (``OvercookedState``)
-------------------------------------------
The public observable state passed to the CSP contains:
  - current_subtask : str | None — next subtask to assign
  - layout_name     : str        — current kitchen layout
  - timestep        : int        — within-episode step count
  - pending_orders  : int        — number of remaining orders
  - score_so_far    : float      — cumulative shaped reward so far

Hidden spec contains:
  - preferred_actor  : dict[subtask, str] — ground-truth preferences
  - hidden_hbm       : OvercookedPreferenceModel | None
  - psi_true         : list[float] — true session offset vector

This module intentionally does NOT import overcooked_ai_py at module load
time; the import is deferred to __init__ so that the rest of the codebase can
be used without the package installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .config.overcooked_config import DEFAULT_CONFIG, OvercookedConfig
from .layouts import ALL_SUBTASKS, LayoutSpec, get_layout
from .overcooked_hbm import DEFAULT_HUMAN, OvercookedPreferenceModel

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# State / spec dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OvercookedState:
    """Public observable state passed to the CSP at each decision step."""

    layout_name: str
    current_subtask: Optional[str]   # subtask to assign; None when episode done
    timestep: int
    pending_orders: int
    score_so_far: float


@dataclass(frozen=True)
class OvercookedAction:
    """
    High-level assignment action from the CSP.

    actor: "human" or "robot" — who should execute the current subtask.
    """

    actor: str


@dataclass(frozen=True)
class OvercookedHiddenSpec:
    """Ground-truth preference information hidden from the learning agent."""

    preferred_actor: Dict[str, str]   # subtask → "human" | "robot"
    hidden_hbm: Optional[OvercookedPreferenceModel] = None
    psi_true: List[float] = field(default_factory=list)
    force_neutral_session: bool = False


# ---------------------------------------------------------------------------
# Sub-policy stubs
# ---------------------------------------------------------------------------

class _SubtaskExecutor:
    """
    Stub scripted sub-policy that converts a subtask assignment into a sequence
    of overcooked_ai primitive actions.

    In a full implementation this would plan a path to the relevant object and
    execute the required INTERACT action.  Here it returns STAY (action 5) as a
    placeholder so that the env can be instantiated and tested without a
    complete low-level planner.

    Replace ``execute()`` with a real planner when integrating overcooked_ai.
    """

    # Overcooked primitive action indices
    STAY = 5

    def execute(
        self,
        subtask: str,
        actor_idx: int,
        state: Any,
        mdp: Any,
    ) -> int:
        """Return a single primitive action for (subtask, actor)."""
        return self.STAY


# ---------------------------------------------------------------------------
# Main Gymnasium environment
# ---------------------------------------------------------------------------

class OvercookedEnv(gym.Env):
    """
    Gymnasium environment wrapping overcooked_ai for preference learning.

    The step interface is at the *subtask assignment* level:
      - ``reset()``  → OvercookedState
      - ``step(OvercookedAction)``  → (OvercookedState, reward, terminated, truncated, info)

    The reward returned is the *task_score* for the just-completed subtask,
    normalised to [-1, +1].  The info dict includes:
      ``last_subtask``   : str — subtask that was just executed
      ``last_actor``     : str — "human" or "robot"
      ``task_score``     : float — normalised feedback in [-1, +1]
      ``shaped_reward``  : float — raw shaped reward from overcooked_ai
      ``psi_true``       : list[float] — true session offsets (for oracle tests)
      ``session_type``   : str — "neutral" | "energised" | "fatigued"
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        layout_spec: LayoutSpec,
        hidden_spec: OvercookedHiddenSpec,
        config: Optional[OvercookedConfig] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.layout_spec = layout_spec
        self.hidden_spec = hidden_spec
        self.config = config if config is not None else DEFAULT_CONFIG
        self._rng = np.random.default_rng(seed)
        self._subtask_executor = _SubtaskExecutor()

        # Subtask list for this layout
        self._subtasks: List[str] = list(layout_spec.subtasks)
        self._subtask_index: Dict[str, int] = {
            s: i for i, s in enumerate(ALL_SUBTASKS)
        }

        # Gymnasium spaces (high-level)
        self.observation_space = gym.spaces.Dict({
            "layout_name": gym.spaces.Discrete(1),   # placeholder
            "current_subtask_idx": gym.spaces.Discrete(len(ALL_SUBTASKS) + 1),
            "timestep": gym.spaces.Discrete(layout_spec.episode_length + 1),
            "pending_orders": gym.spaces.Discrete(10),
            "score_so_far": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })
        self.action_space = gym.spaces.Discrete(2)  # 0=human, 1=robot

        # Try to import overcooked_ai; gracefully degrade if not installed.
        self._overcooked_env: Any = None
        self._mdp: Any = None
        self._overcooked_state: Any = None
        self._overcooked_available = self._try_init_overcooked()

        # Episode state
        self._timestep: int = 0
        self._pending_orders: int = 3      # default; reset properly in reset()
        self._score_so_far: float = 0.0
        self._current_subtask_idx: int = 0
        self._session_type: str = "neutral"
        self._psi_true: List[float] = [0.0] * len(ALL_SUBTASKS)

    def _try_init_overcooked(self) -> bool:
        """Attempt to import and initialise overcooked_ai. Returns True on success."""
        try:
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
            from overcooked_ai_py.env.overcooked_env import (
                OvercookedEnv as _OCEnv,
            )
            self._mdp = OvercookedGridworld.from_layout_name(
                self.layout_spec.layout_name
            )
            self._overcooked_env = _OCEnv(
                mdp=self._mdp,
                episode_length=self.layout_spec.episode_length,
                use_shaped_reward=True,
            )
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[OvercookedState, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample session type and psi_true
        self._session_type = self._sample_session_type()
        self._psi_true = self._sample_psi_true(self._session_type)

        # Reset overcooked_ai if available
        if self._overcooked_available and self._overcooked_env is not None:
            self._overcooked_env.reset()
            self._overcooked_state = self._overcooked_env.state
            self._pending_orders = len(self._mdp.order_list or []) or 3
        else:
            self._pending_orders = 3

        self._timestep = 0
        self._score_so_far = 0.0
        self._current_subtask_idx = 0

        obs = self._make_obs()
        info: Dict[str, Any] = {
            "session_type": self._session_type,
            "psi_true": self._psi_true,
        }
        return obs, info

    def step(
        self, action: OvercookedAction
    ) -> Tuple[OvercookedState, float, bool, bool, Dict[str, Any]]:
        """
        Execute a subtask assignment and return the resulting state + feedback.

        The action specifies which agent (human or robot) should execute the
        current subtask.  The sub-policy then runs low-level primitive actions
        until the subtask completes (stub: one STAY step).
        """
        current_subtask = self._subtasks[self._current_subtask_idx]
        actor = action.actor
        assert actor in ("human", "robot")

        # Run the subtask in overcooked_ai (or stub)
        shaped_reward = self._execute_subtask(current_subtask, actor)
        task_score = self._compute_task_score(
            current_subtask, actor, shaped_reward
        )

        self._score_so_far += shaped_reward
        self._timestep += 1

        # Advance subtask pointer
        self._current_subtask_idx = (
            self._current_subtask_idx + 1
        ) % len(self._subtasks)

        # Episode ends when we've cycled through all subtasks once or time runs out
        terminated = self._current_subtask_idx == 0
        truncated = self._timestep >= self.layout_spec.episode_length

        obs = self._make_obs()
        info: Dict[str, Any] = {
            "last_subtask": current_subtask,
            "last_actor": actor,
            "task_score": task_score,
            "shaped_reward": shaped_reward,
            "psi_true": self._psi_true,
            "session_type": self._session_type,
        }
        return obs, task_score, terminated or truncated, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> OvercookedState:
        if self._current_subtask_idx < len(self._subtasks):
            current_subtask: Optional[str] = self._subtasks[self._current_subtask_idx]
        else:
            current_subtask = None
        return OvercookedState(
            layout_name=self.layout_spec.name,
            current_subtask=current_subtask,
            timestep=self._timestep,
            pending_orders=self._pending_orders,
            score_so_far=self._score_so_far,
        )

    def _sample_session_type(self) -> str:
        """Sample a session type from the configured prior."""
        prior = self.config.get_session_type_prior()
        types = list(prior.keys())
        probs = list(prior.values())
        return str(self._rng.choice(types, p=probs))

    def _sample_psi_true(self, session_type: str) -> List[float]:
        """
        Sample the true per-subtask session offset vector.

        Neutral session: psi_true ~ N(0, neutral_std²) per dimension.
        Non-neutral: psi_true ~ N(±mean_abs, non_neutral_std²) per dimension.
        """
        sc = self.config.session
        n_dims = len(ALL_SUBTASKS)
        if session_type == "neutral":
            return list(
                float(x)
                for x in self._rng.normal(0.0, sc.session_neutral_std, size=n_dims)
            )
        elif session_type == "energised":
            return list(
                float(x)
                for x in self._rng.normal(
                    sc.session_nonneutral_mean_abs,
                    sc.session_nonneutral_std,
                    size=n_dims,
                )
            )
        else:  # fatigued
            return list(
                float(x)
                for x in self._rng.normal(
                    -sc.session_nonneutral_mean_abs,
                    sc.session_nonneutral_std,
                    size=n_dims,
                )
            )

    def _execute_subtask(self, subtask: str, actor: str) -> float:
        """
        Execute one subtask in the overcooked_ai environment (or stub).

        Returns the shaped reward collected during execution.
        """
        if not self._overcooked_available or self._overcooked_env is None:
            # Stub: return a small positive reward for "completing" a subtask
            return 3.0

        actor_idx = 0 if actor == "robot" else 1
        action = self._subtask_executor.execute(
            subtask, actor_idx, self._overcooked_state, self._mdp
        )
        # Build joint action (both agents act; non-executing agent STAYs)
        joint_action = (action, 5) if actor == "robot" else (5, action)
        _, reward, _, _, info = self._overcooked_env.step(joint_action)
        self._overcooked_state = self._overcooked_env.state
        # shaped_r may be a dict or float depending on overcooked_ai version
        shaped = info.get("shaped_r", 0.0)
        if isinstance(shaped, dict):
            shaped = sum(shaped.values())
        return float(reward) + float(shaped)

    def _compute_task_score(
        self, subtask: str, actor: str, shaped_reward: float
    ) -> float:
        """
        Convert shaped_reward into a normalised task_score in [-1, +1].

        The generative model: satisfaction depends on actor matching the true
        preferred actor, with a session offset psi_true[subtask_dim] and
        Beta noise (same structure as spices env).

        If a hidden HBM is available, use its theta as the preference strength.
        Otherwise use the hidden_spec.preferred_actor directly.
        """
        subtask_dim = self._subtask_index.get(subtask, 0)
        psi_true = self._psi_true[subtask_dim] if subtask_dim < len(self._psi_true) else 0.0

        preferred = self.hidden_spec.preferred_actor.get(subtask, "robot")

        # Preference magnitude
        if (
            self.hidden_spec.hidden_hbm is not None
            and subtask in self.hidden_spec.hidden_hbm._theta_m.get(DEFAULT_HUMAN, {})
        ):
            phi_mag = abs(
                self.hidden_spec.hidden_hbm._theta_m[DEFAULT_HUMAN][subtask].item()
            )
        else:
            phi_mag = self.config.task.base_task_bias

        phi_latent = phi_mag if preferred == "human" else -phi_mag
        sign_actor = 1.0 if actor == "human" else -1.0
        logit = sign_actor * (phi_latent + psi_true)
        temp = max(self.config.task.task_logit_temperature, 1e-6)
        expected = float(np.tanh(logit / temp))

        # Beta-distributed noise around expected score → [-1, +1]
        kappa = self.config.task.task_beta_kappa
        p = (expected + 1.0) / 2.0
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        alpha = p * kappa + 1.0
        beta_ = (1.0 - p) * kappa + 1.0
        sample = float(self._rng.beta(alpha, beta_))
        return 2.0 * sample - 1.0   # rescale [0,1] → [-1,+1]
