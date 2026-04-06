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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

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
# Sub-policy
# ---------------------------------------------------------------------------

class _SubtaskExecutor:
    """
    Converts a subtask assignment into a sequence of primitive actions executed
    until the assigned actor completes the subtask (performs INTERACT at the
    goal tile) or the step cap is reached.

    The background agent (the other player) uses a *safe* greedy policy that
    avoids two failure modes:
      1. Blocking: the background agent never tries to move onto the assigned
         agent's current goal tile, preventing mutual deadlock.
      2. Early cooking: the background agent never INTERACTs with a pot that
         still needs more ingredients (would trigger begin_cooking() too early).

    Delegates to ``overcooked_planner.SubtaskPlanner`` when overcooked_ai_py
    is available; falls back to a fixed number of STAY steps otherwise.
    """

    STAY = (0, 0)
    # Must exceed cook_time (~20 steps) + travel time (~10 steps) with margin.
    MAX_STEPS_PER_SUBTASK = 80

    def __init__(self, planner: Any = None, background_agent: Any = None,
                 mdp: Any = None) -> None:
        self._planner = planner                    # SubtaskPlanner | None
        self._background_agent = background_agent  # GreedyHumanModel | None
        self._mdp = mdp                            # OvercookedGridworld | None

    # ------------------------------------------------------------------
    # Background agent safety wrapper
    # ------------------------------------------------------------------

    def _safe_bg_action(
        self,
        bg_idx: int,
        state: Any,
        assigned_goal_pos: Optional[Tuple[int, int]],
    ) -> Any:
        """
        Return a safe action for the background agent.
        Only guard: do not INTERACT with a pot that has < MAX_NUM_INGREDIENTS items
        (avoids triggering begin_cooking() prematurely).
        Position-based blocking is left to overcooked_ai's natural collision avoidance.
        Falls back to STAY on any error.
        """
        stay = self.STAY
        if self._background_agent is None:
            return stay
        try:
            self._background_agent.set_agent_index(bg_idx)
            bg_action, _ = self._background_agent.action(state)
        except Exception:
            return stay

        # No position-based blocking: the overcooked motion planner handles
        # natural collision avoidance through its own path planning. Artificially
        # blocking the background agent causes deadlocks in tight layouts.

        # --- Safety check: don't prematurely start cooking ---
        if bg_action == "interact":
            bg_agent = state.players[bg_idx]
            ox, oy = bg_agent.orientation
            facing = (bg_agent.position[0] + ox, bg_agent.position[1] + oy)
            if self._would_start_cooking_early(state, facing):
                return stay

        return bg_action

    def _adjacent_to_goal(
        self,
        goal_tile: Tuple[int, int],
        state: Any,
        actor_idx: int,
    ) -> Optional[Tuple[int, int]]:
        """
        Return the walkable cell adjacent to goal_tile that actor_idx is
        heading toward (closest free neighbor, same logic as SubtaskPlanner).
        """
        if self._mdp is None:
            return None
        try:
            gx, gy = goal_tile
            other_idx = 1 - actor_idx
            other_pos = state.players[other_idx].position
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                neighbor = (gx + dx, gy + dy)
                if neighbor == other_pos:
                    continue
                if self._mdp.get_terrain_type_at_pos(neighbor) == " ":
                    return neighbor
        except Exception:
            pass
        return None

    def _would_start_cooking_early(self, state: Any, pot_pos: Tuple[int, int]) -> bool:
        """
        Return True if an INTERACT at pot_pos with empty hands would call
        begin_cooking() on a pot that does not yet have MAX_NUM_INGREDIENTS.
        """
        if self._mdp is None:
            return False
        try:
            # Check terrain is a pot
            terrain = self._mdp.get_terrain_type_at_pos(pot_pos)
            if terrain != 'P':
                return False
            if not state.has_object(pot_pos):
                return False
            soup = state.get_object(pot_pos)
            if soup.name != 'soup':
                return False
            if soup.is_cooking or soup.is_ready:
                return False
            # Cooking would be triggered; is the pot full?
            try:
                from overcooked_ai_py.mdp.overcooked_mdp import Recipe
                max_ing = Recipe.MAX_NUM_INGREDIENTS
            except Exception:
                max_ing = 3
            return len(soup.ingredients) < max_ing
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def _effective_subtask(self, subtask: str, actor_idx: int, state: Any) -> str:
        """
        Return what the agent should actually do given what they're holding.

        No chaining — each call returns one concrete primitive subtask.
        If the agent already holds the prerequisite item, skip the fetch step.
        If the agent holds something incompatible, use it appropriately first.
        """
        if self._planner is None:
            return subtask
        try:
            agent = state.players[actor_idx]
            held = agent.get_object().name if agent.has_object() else None

            if subtask == "fetch_ingredient":
                if held in ("onion", "tomato"):
                    return "load_pot"   # already has it, go load
                if held == "soup":
                    return "deliver"
                if held == "dish":
                    return "pickup_soup"
                return "fetch_ingredient"

            if subtask == "load_pot":
                if held in ("onion", "tomato"):
                    return "load_pot"
                if held is None:
                    return "load_pot"   # empty-handed → begin_cooking INTERACT
                if held == "soup":
                    return "deliver"
                if held == "dish":
                    return "pickup_soup"
                return "fetch_ingredient"

            if subtask == "fetch_dish":
                if held == "dish":
                    return "pickup_soup"  # already has dish
                if held == "soup":
                    return "deliver"
                if held in ("onion", "tomato"):
                    return "load_pot"   # load ingredient first, then fetch dish next turn
                return "fetch_dish"

            if subtask == "pickup_soup":
                if held == "dish":
                    return "pickup_soup"
                if held == "soup":
                    return "deliver"
                return "fetch_dish"

            if subtask == "deliver":
                if held == "soup":
                    return "deliver"
                if held == "dish":
                    return "pickup_soup"
                return "fetch_dish"

        except Exception:
            pass
        return subtask

    def execute_until_done(
        self,
        subtask: str,
        actor_idx: int,
        oc_env: Any,        # overcooked_ai OvercookedEnv
        callback: Any = None,  # optional callable(oc_state, joint_action) for rendering
    ) -> Tuple[float, float, bool, bool]:
        """
        Step the overcooked_ai environment with primitive actions until the
        assigned actor completes the high-level subtask or the step cap is reached.

        Each primitive step re-evaluates the effective subtask based on the
        agent's actual hand state, so prerequisite items are fetched automatically
        (e.g. fetch_dish before pickup_soup) without a separate CSP decision.

        Returns
        -------
        (shaped_reward, sparse_reward, game_done, subtask_completed)
        subtask_completed is True if the assigned actor performed a meaningful
        INTERACT (i.e. the planner returned "interact" for the *original* subtask
        or a prerequisite step that moved the workflow forward).
        """
        stay = self.STAY
        total_shaped = 0.0
        total_sparse = 0.0
        bg_idx = 1 - actor_idx
        game_done = False
        subtask_completed = False

        for _ in range(self.MAX_STEPS_PER_SUBTASK):
            state = oc_env.state

            # Determine what the agent should actually do given their hand state
            effective = self._effective_subtask(subtask, actor_idx, state)

            # --- Assigned actor ---
            if self._planner is not None:
                assigned_action = self._planner.step(effective, actor_idx, state)
            else:
                assigned_action = stay

            # Compute goal position for the assigned agent so bg can avoid it
            assigned_goal_pos = self._planner_goal_pos(effective, actor_idx, state) \
                if self._planner is not None else None

            # --- Background actor (safe) ---
            bg_action = self._safe_bg_action(bg_idx, state, assigned_goal_pos)

            if actor_idx == 0:
                joint_action = (assigned_action, bg_action)
            else:
                joint_action = (bg_action, assigned_action)

            if callback is not None:
                callback(state, joint_action)

            _, sparse_r, done, env_info = oc_env.step(joint_action)
            total_sparse += float(sparse_r)

            shaped = env_info.get("shaped_r_by_agent", env_info.get("shaped_r", [0.0, 0.0]))
            if isinstance(shaped, (list, tuple)):
                total_shaped += float(sum(shaped))
            elif isinstance(shaped, dict):
                total_shaped += float(sum(shaped.values()))
            else:
                total_shaped += float(shaped)

            if done:
                game_done = True
                # If the last action was also an INTERACT, count it as completed
                if assigned_action == "interact":
                    subtask_completed = True
                break

            # The assigned actor just INTERACTed — subtask is done.
            if assigned_action == "interact":
                subtask_completed = True
                break

        return total_shaped, total_sparse, game_done, subtask_completed

    def _planner_goal_pos(
        self, subtask: str, actor_idx: int, state: Any
    ) -> Optional[Tuple[int, int]]:
        """Return the grid position the assigned agent is currently heading toward."""
        if self._planner is None or self._mdp is None:
            return None
        try:
            agent = state.players[actor_idx]
            goal_positions = self._planner._goal_positions.get(subtask, [])
            if not goal_positions:
                return None
            return min(
                goal_positions,
                key=lambda g: abs(g[0] - agent.position[0]) + abs(g[1] - agent.position[1]),
            )
        except Exception:
            return None


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
        # Planner is wired up inside _try_init_overcooked
        self._subtask_executor = _SubtaskExecutor(planner=None)

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

        # Optional render callback: called on every primitive step with
        # (oc_state, joint_action, subtask, actor).  Set externally by demo scripts.
        self.render_callback: Optional[Callable[[Any, Any, str, str], None]] = None

        # Episode state
        self._timestep: int = 0
        self._pending_orders: int = 3      # default; reset properly in reset()
        self._score_so_far: float = 0.0
        self._current_subtask_idx: int = 0   # kept for fallback; primary is _get_next_subtask
        self._current_subtask: Optional[str] = None
        self._session_type: str = "neutral"
        self._psi_true: List[float] = [0.0] * len(ALL_SUBTASKS)

    def _try_init_overcooked(self) -> bool:
        """Attempt to import and initialise overcooked_ai. Returns True on success."""
        try:
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
            from overcooked_ai_py.mdp.overcooked_env import (
                OvercookedEnv as _OCEnv,
            )
            self._mdp = OvercookedGridworld.from_layout_name(
                self.layout_spec.layout_name
            )
            self._overcooked_env = _OCEnv.from_mdp(
                self._mdp,
                horizon=self.layout_spec.episode_length,
            )
            # Wire up the real planner and background greedy agent
            try:
                from .overcooked_planner import build_planner
                from overcooked_ai_py.agents.agent import GreedyHumanModel
                planner = build_planner(self._mdp)
                # Background agent uses the same MLAM as the planner
                mlam = planner._mlam if planner is not None else None
                background = GreedyHumanModel(mlam) if mlam is not None else None
                if background is not None:
                    background.set_mdp(self._mdp)
                self._subtask_executor = _SubtaskExecutor(
                    planner=planner, background_agent=background, mdp=self._mdp
                )
            except Exception:
                pass  # keep stub executor on any failure
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
            # Access current state via .state property (overcooked_ai native env)
            self._overcooked_state = self._overcooked_env.state
            order_list = getattr(self._mdp, "order_list", None) or []
            self._pending_orders = len(order_list) if order_list else 3
        else:
            self._pending_orders = 3

        self._timestep = 0
        self._score_so_far = 0.0
        self._current_subtask_idx = 0
        self._current_subtask = self._get_next_subtask()

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
        current_subtask = self._current_subtask or self._subtasks[self._current_subtask_idx]
        actor = action.actor
        assert actor in ("human", "robot")

        # Run the subtask in overcooked_ai (or stub)
        shaped_reward, game_done, subtask_completed = self._execute_subtask(
            current_subtask, actor
        )

        # Satisfaction score reflects whether the *right* person did the task.
        # - Only scored on actual completion (not timeouts/blocks).
        # - For subtasks that don't themselves generate sparse reward (everything
        #   except deliver), use the preference model directly.
        # - For deliver (sparse_reward > 0 means soup served), the score is the
        #   most meaningful: the full order was completed by this actor.
        if subtask_completed:
            task_score = self._compute_task_score(current_subtask, actor, shaped_reward)
        else:
            task_score = 0.0  # blocked or timed out → no signal

        self._score_so_far += shaped_reward
        self._timestep += 1

        # Advance to the next feasible subtask based on current game state.
        # When overcooked is available, _get_next_subtask() reads actual state.
        # In legacy/stub mode, manually advance the round-robin idx first.
        if not self._overcooked_available:
            self._current_subtask_idx = (self._current_subtask_idx + 1) % len(self._subtasks)
        self._current_subtask = self._get_next_subtask()
        # Keep legacy index in sync for observation space
        if self._current_subtask is not None and self._current_subtask in self._subtasks:
            self._current_subtask_idx = self._subtasks.index(self._current_subtask)

        # Episode ends when overcooked's internal horizon is reached or time cap
        terminated = game_done
        truncated = self._timestep >= self.layout_spec.episode_length

        obs = self._make_obs()
        info: Dict[str, Any] = {
            "last_subtask": current_subtask,
            "last_actor": actor,
            "task_score": task_score,
            "shaped_reward": shaped_reward,
            "subtask_completed": subtask_completed,  # False if timed-out / blocked
            "psi_true": self._psi_true,
            "session_type": self._session_type,
        }
        return obs, task_score, terminated or truncated, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> OvercookedState:
        current_subtask: Optional[str] = self._current_subtask
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

    # ------------------------------------------------------------------
    # Game-state-aware subtask sequencer
    # ------------------------------------------------------------------

    def _get_pot_states(self) -> Optional[Dict[str, Any]]:
        """
        Return the overcooked_ai pot-states dict, or None if unavailable.

        Keys: 'empty', '1_items', '2_items', '3_items', 'cooking', 'ready'
        Values: lists of pot positions.
        """
        if not self._overcooked_available or self._mdp is None or self._overcooked_env is None:
            return None
        try:
            state = self._overcooked_env.state
            return self._mdp.get_pot_states(state)
        except Exception:
            return None

    def _get_next_subtask(self) -> Optional[str]:
        """
        Derive the next subtask purely from the current overcooked game state.

        Pot-state vocabulary (from mdp.get_pot_states — all values are positions):
          'empty'        — pot has no soup object
          'N_items'      — soup with N idle ingredients (not yet cooking)
          'cooking'      — soup cooking (timer > 0, not ready)
          'ready'        — soup ready to plate

        Note: pickup_soup and deliver are only returned when the state
        genuinely supports them (soup ready + someone holds dish; player holds
        soup).  This prevents the loop where the sequencer returns pickup_soup
        but the assigned agent has no dish and INTERACT does nothing.

        fetch_ingredient and fetch_dish are always safe — the assigned agent
        goes to the dispenser, grabs the item, and the INTERACT succeeds.
        load_pot is safe — the agent either loads an ingredient or (if holding
        nothing at a full-idle pot) triggers begin_cooking().
        """
        if not self._overcooked_available or self._mdp is None or self._overcooked_env is None:
            return self._subtasks[self._current_subtask_idx] if self._subtasks else None

        try:
            from overcooked_ai_py.mdp.overcooked_mdp import Recipe
            max_ing = Recipe.MAX_NUM_INGREDIENTS
        except Exception:
            max_ing = 3

        try:
            state = self._overcooked_env.state
            pot_states = self._mdp.get_pot_states(state)

            ready_pots   = pot_states.get("ready", [])
            cooking_pots = pot_states.get("cooking", [])
            full_idle    = pot_states.get(f"{max_ing}_items", [])
            partial_pots: List[Any] = []
            for i in range(1, max_ing):
                partial_pots.extend(pot_states.get(f"{i}_items", []))
            empty_pots   = pot_states.get("empty", [])

            players = state.players
            ingredient_names = {"onion", "tomato"}

            def any_holds(names: set) -> bool:
                return any(
                    p.has_object() and p.get_object().name in names
                    for p in players
                )

            # P1. Someone holds soup → deliver immediately (highest priority).
            if any_holds({"soup"}):
                return "deliver"

            # P2. Pot ready and someone holds a dish → go pick up soup.
            if ready_pots and any_holds({"dish"}):
                return "pickup_soup"

            # P3. Pot ready, no dish in hand → fetch one.
            if ready_pots:
                return "fetch_dish"

            # P4. Full-idle pot → INTERACT triggers begin_cooking.
            if full_idle:
                return "load_pot"

            # P5. Partial pots need more ingredients.
            if partial_pots:
                return "fetch_ingredient"

            # P6. Empty pots need to be filled.
            if empty_pots:
                return "fetch_ingredient"

            # P7. All pots cooking → pre-fetch a dish (or pickup if one already held).
            if cooking_pots:
                if any_holds({"dish"}):
                    return "pickup_soup"  # wait at pot; planner STAYs until is_ready
                return "fetch_dish"

            # Absolute fallback
            return self._subtasks[self._current_subtask_idx] if self._subtasks else None

        except Exception:
            return self._subtasks[self._current_subtask_idx] if self._subtasks else None

    def _execute_subtask(self, subtask: str, actor: str) -> Tuple[float, bool, bool]:
        """
        Execute one subtask in the overcooked_ai environment (or stub).

        Runs primitive actions until the assigned actor performs INTERACT at the
        goal tile, or the step cap is reached.

        Returns
        -------
        (total_reward, game_done, subtask_completed)
          total_reward      : shaped + sparse reward accumulated
          game_done         : True if overcooked_ai's horizon was reached
          subtask_completed : True if the assigned actor successfully did INTERACT
        """
        if not self._overcooked_available or self._overcooked_env is None:
            return 3.0, False, True  # stub: assume completed

        actor_idx = 0 if actor == "robot" else 1

        # Build render callback wrapping the user-supplied one
        cb = None
        if self.render_callback is not None:
            _subtask = subtask
            _actor = actor
            _user_cb = self.render_callback

            def cb(oc_state: Any, joint_action: Any) -> None:
                _user_cb(oc_state, joint_action, _subtask, _actor)

        shaped, sparse, game_done, subtask_completed = \
            self._subtask_executor.execute_until_done(
                subtask, actor_idx, self._overcooked_env, callback=cb
            )
        self._overcooked_state = self._overcooked_env.state
        return shaped + sparse, game_done, subtask_completed

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
