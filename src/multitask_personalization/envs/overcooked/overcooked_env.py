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

from multitask_personalization.structs import PublicSceneSpec

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
    forced_actor: Optional[str] = None  # if set, item ownership forces this agent


@dataclass(frozen=True)
class OvercookedSceneSpec(PublicSceneSpec):
    """Public scene specification for the Overcooked environment.

    Passed to CSPApproach so the factory can identify the environment type
    and build the correct CSPGenerator.
    """

    layout_spec: LayoutSpec = None  # type: ignore[assignment]
    subtask_list: tuple[str, ...] = ()
    # Per-agent feasibility: which subtasks each agent can physically reach.
    feasibility: Optional[Dict[str, Dict[str, bool]]] = None
    # Multi-layout training: list of layout names to rotate through.
    train_layout_names: Optional[tuple[str, ...]] = None
    # --- FUTURE WORK: continuous preferences (phase 1) ---
    # Gated skeleton; default False. When True, the CSP adds a joint
    # ingredient_count decision variable on load_pot subtasks, and the env
    # evaluates continuous satisfaction from hidden_spec.preferred_ingredient_count.
    # Only enabled by the dedicated overcooked_continuous.yaml Hydra config.
    # See src/.../overcooked/continuous_prefs.py for status and known issues.
    continuous_prefs_enabled: bool = False
    ingredient_count_choices: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class OvercookedAction:
    """
    High-level assignment action from the CSP.

    Autonomous-human semantics:
      flag = 0: robot claims the current subtask (predicting human won't want it)
      flag = 1: robot passes (predicting human will claim this subtask)

    The human autonomously decides whether to claim the subtask based on:
        P(human claims) = sigma(phi[subtask] + psi_true[subtask_dim])

    Outcomes (simultaneous commitment, human wins conflicts):
      flag=0, human does NOT claim → robot executes, task_score = -1
      flag=0, human ALSO claims   → CONFLICT: human executes, task_score = +1
      flag=1, human claims        → human executes, task_score = +1
      flag=1, human does NOT claim→ robot must execute anyway, task_score = -1

    The task_score is a clean binary behavioral label: +1 if the human ended up
    performing the subtask, -1 if the robot did. Conflict information is captured
    separately in info["conflict"] and via the coordination penalty applied to
    the satisfaction signal.

    Continuous preferences (phase 1, opt-in):
      ingredient_count: optional int in {1, 2, 3} that specifies how many
      ingredients to place in a pot before triggering cooking. When set, the
      environment treats this as the chosen continuous parameter value and
      evaluates the human's preference satisfaction against it. When None
      (default), no continuous preference is active — the legacy binary-only
      behavior is preserved.
    """

    flag: int  # 0 = robot claims, 1 = robot passes
    ingredient_count: Optional[int] = None  # continuous pref: ingredients before cook trigger


@dataclass(frozen=False)
class OvercookedHiddenSpec:
    """Ground-truth preference information hidden from the learning agent."""

    preferred_actor: Dict[str, str]   # subtask → "human" | "robot"
    hidden_hbm: Optional[OvercookedPreferenceModel] = None
    psi_true: List[float] = field(default_factory=list)
    force_neutral_session: bool = False
    # Continuous preferences (phase 1): ground-truth preferred ingredient count.
    # Dict maps layout_name -> (preferred_mean, preferred_std). If empty, no
    # continuous preference is active and continuous satisfaction contributes 0.
    preferred_ingredient_count: Dict[str, tuple] = field(default_factory=dict)

    @property
    def force_neutral_mood(self) -> bool:
        """Alias for run_single_experiment compat (spices uses force_neutral_mood)."""
        return self.force_neutral_session

    @force_neutral_mood.setter
    def force_neutral_mood(self, value: bool) -> None:
        self.force_neutral_session = value


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
                 mdp: Any = None,
                 feasibility: Optional[Dict[str, Dict[str, bool]]] = None,
                 shared_counters: Optional[List[Any]] = None) -> None:
        self._planner = planner                    # SubtaskPlanner | None
        self._background_agent = background_agent  # GreedyHumanModel | None
        self._mdp = mdp                            # OvercookedGridworld | None
        self._feasibility_ref = feasibility or {}  # per-agent reachability
        self._shared_counters: List[Any] = list(shared_counters) if shared_counters else []

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
        Return an action for the background agent.

        The background agent moves around (avoiding blocking the assigned
        agent) but does NOT INTERACT with anything. This ensures:
        - Only the assigned agent picks up / places items
        - The visual matches the CSP assignment
        - The sequential subtask sequencer isn't confused by unexpected
          state changes from background activity

        Note: Phase 3 (parallel execution) will replace this with a proper
        dual-subtask model where both agents work simultaneously on
        independently-assigned subtasks.
        """
        stay = self.STAY
        if self._background_agent is None:
            return stay
        try:
            self._background_agent.set_agent_index(bg_idx)
            bg_action, _ = self._background_agent.action(state)
        except Exception:
            return stay

        # Block any INTERACT — only the assigned agent should work.
        if bg_action == "interact":
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

            if subtask in ("fetch_ingredient", "fetch_onion", "fetch_tomato"):
                if held in ("onion", "tomato"):
                    return subtask  # already has ingredient; will complete
                if held is None:
                    return subtask
                return subtask  # holding wrong item; will timeout

            if subtask == "load_pot":
                if held in ("onion", "tomato"):
                    # Check if any pot can accept the ingredient.
                    if self._mdp is not None:
                        try:
                            pot_states = self._mdp.get_pot_states(state)
                            can_accept = False
                            for key in ("empty", "1_items", "2_items"):
                                if pot_states.get(key, []):
                                    can_accept = True
                                    break
                            if not can_accept:
                                # No pot can accept — surplus onion. Return subtask
                                # to cause a timeout; sequencer will move on.
                                return subtask
                        except Exception:
                            pass
                    return "load_pot"
                if held is None:
                    # Empty-handed: only go to pot if it's FULL (to trigger cooking).
                    if self._mdp is not None:
                        try:
                            from overcooked_ai_py.mdp.overcooked_mdp import Recipe
                            max_ing = Recipe.MAX_NUM_INGREDIENTS
                            pot_states = self._mdp.get_pot_states(state)
                            full_idle = pot_states.get(f"{max_ing}_items", [])
                            if full_idle:
                                return "load_pot"
                            else:
                                return "fetch_ingredient"
                        except Exception:
                            pass
                    return "load_pot"
                # Holding wrong item — return subtask (will timeout).
                return subtask

            if subtask == "fetch_dish":
                if held == "dish":
                    return "fetch_dish"  # already has it; will complete
                if held is None:
                    return "fetch_dish"
                # Holding wrong item — return subtask (will timeout).
                return subtask

            if subtask == "pickup_soup":
                if held == "dish":
                    if self._mdp is not None:
                        try:
                            pot_states = self._mdp.get_pot_states(state)
                            if pot_states.get("ready", []):
                                return "pickup_soup"
                        except Exception:
                            pass
                    return subtask  # no ready soup; will timeout
                if held == "soup":
                    return "pickup_soup"  # already has it
                if held is None:
                    return "fetch_dish"
                return subtask  # wrong item

            if subtask == "deliver":
                if held == "soup":
                    return "deliver"
                if held is None:
                    return "fetch_dish"
                # Holding wrong item (dish without ready soup, onion, etc.)
                return subtask  # will timeout

            if subtask == "place_on_counter":
                # Agent must be holding something to place on counter.
                if held is not None:
                    return "place_on_counter"
                # Nothing to place — subtask is a no-op.
                return subtask

            if subtask == "pickup_from_counter":
                if held is None:
                    return "pickup_from_counter"
                # Holding something — can't pick up. Will timeout.
                return subtask

        except Exception:
            pass
        return subtask

    def _pick_next_for_agent(
        self,
        agent_idx: int,
        state: Any,
        current_subtasks: list,
    ) -> Optional[str]:
        """
        After an agent completes, pick a reasonable next subtask based on
        what they're holding and what the other agent is working on.

        Returns None if no obvious next task.
        """
        try:
            agent = state.players[agent_idx]
            held = agent.get_object().name if agent.has_object() else None
            other_subtask = current_subtasks[1 - agent_idx]

            # If holding an item, continue the chain.
            if held in ("onion", "tomato"):
                return "load_pot"
            if held == "dish":
                return "pickup_soup"
            if held == "soup":
                return "deliver"

            # Empty-handed — pick a task different from the other agent's.
            candidates = ["fetch_ingredient", "fetch_dish"]
            for c in candidates:
                if c != other_subtask:
                    return c
            return "fetch_ingredient"
        except Exception:
            return None

    def execute_until_done(
        self,
        subtask: str,
        actor_idx: int,
        oc_env: Any,
        callback: Any = None,
    ) -> Tuple[float, float, bool, bool]:
        """Single-agent execution (legacy). Delegates to execute_dual."""
        result = self.execute_dual(
            subtask_0=subtask if actor_idx == 0 else None,
            subtask_1=subtask if actor_idx == 1 else None,
            oc_env=oc_env,
            callback=callback,
        )
        return (result["shaped_reward"], result["sparse_reward"],
                result["game_done"], result[f"completed_{actor_idx}"])

    def execute_dual(
        self,
        subtask_0: Optional[str],
        subtask_1: Optional[str],
        oc_env: Any,
        callback: Any = None,
        both_use_planner: bool = False,
    ) -> Dict[str, Any]:
        """
        Run both agents simultaneously toward their respective subtasks.

        Each agent has its own subtask (or None = idle/STAY).  The loop
        runs primitive steps until EITHER agent completes their subtask
        via INTERACT, or the step cap is reached.

        Parameters
        ----------
        subtask_0 : subtask for agent 0 (robot), or None to idle
        subtask_1 : subtask for agent 1 (human), or None to idle
        oc_env    : overcooked_ai OvercookedEnv
        callback  : optional (oc_state, joint_action, subtask_0, subtask_1) callable

        Returns
        -------
        dict with keys:
            shaped_reward, sparse_reward, game_done,
            completed_0, completed_1  (bool: did each agent complete?)
        """
        if oc_env is None:
            # Stub mode: assume both completed instantly.
            return {
                "shaped_reward": 3.0, "sparse_reward": 0.0,
                "game_done": False,
                "completed_0": subtask_0 is not None,
                "completed_1": subtask_1 is not None,
            }

        stay = self.STAY
        total_shaped = 0.0
        total_sparse = 0.0
        deliveries = 0
        game_done = False
        completed = [False, False]
        subtasks = [subtask_0, subtask_1]
        # After the first agent completes, give the second agent a grace
        # period to finish before ending the round. This prevents the fast
        # agent from idling for 70+ steps while the slow agent works.
        first_complete_step: Optional[int] = None
        GRACE_STEPS = 20  # extra steps after first completion

        for step_i in range(self.MAX_STEPS_PER_SUBTASK):
            state = oc_env.state
            actions = [stay, stay]

            # Determine primary agent (the one assigned by CSP) —
            # they get planner priority. Secondary uses GreedyHumanModel.
            primary_idx = 0 if subtask_0 is not None else (1 if subtask_1 is not None else -1)
            if subtask_0 is not None and subtask_1 is not None:
                # Both have tasks — agent 0 (robot) gets planner priority
                # since the CSP chose its task.
                primary_idx = 0

            for idx in range(2):
                if subtasks[idx] is None or completed[idx]:
                    # Agent has no task or already finished.
                    # Don't auto-chain to next task — let the CSP reassign
                    # on the next step() to avoid race conditions where one
                    # agent steals items the other agent needs.
                    # Idle: use greedy movement, no INTERACT.
                    if self._background_agent is not None:
                        try:
                            self._background_agent.set_agent_index(idx)
                            bg_act, _ = self._background_agent.action(state)
                            actions[idx] = stay if bg_act == "interact" else bg_act
                        except Exception:
                            actions[idx] = stay
                elif idx == primary_idx:
                    # Primary agent uses the planner for precise navigation.
                    effective = self._effective_subtask(subtasks[idx], idx, state)
                    if self._planner is not None:
                        actions[idx] = self._planner.step(effective, idx, state)
                    else:
                        actions[idx] = stay
                else:
                    # Secondary agent: use planner in separated layouts (no
                    # collision risk), GreedyHumanModel in shared layouts.
                    if both_use_planner:
                        effective = self._effective_subtask(subtasks[idx], idx, state)
                        if self._planner is not None:
                            act = self._planner.step(effective, idx, state)
                            # Block secondary from INTERACTing with shared
                            # counters when the primary is doing pickup/place.
                            if act == "interact":
                                other_sub = subtasks[1 - idx]
                                if other_sub in ("pickup_from_counter", "place_on_counter"):
                                    agent = state.players[idx]
                                    ox, oy = agent.orientation
                                    facing = (agent.position[0]+ox, agent.position[1]+oy)
                                    if facing in self._shared_counters:
                                        act = stay
                            actions[idx] = act
                        else:
                            actions[idx] = stay
                    elif self._background_agent is not None:
                        try:
                            self._background_agent.set_agent_index(idx)
                            bg_act, _ = self._background_agent.action(state)
                            if bg_act == "interact":
                                effective = self._effective_subtask(subtasks[idx], idx, state)
                                if self._planner is not None:
                                    actions[idx] = self._planner.step(effective, idx, state)
                                else:
                                    actions[idx] = stay
                            else:
                                actions[idx] = bg_act
                        except Exception:
                            actions[idx] = stay
                    else:
                        effective = self._effective_subtask(subtasks[idx], idx, state)
                        if self._planner is not None:
                            actions[idx] = self._planner.step(effective, idx, state)
                        else:
                            actions[idx] = stay

            joint_action = tuple(actions)

            if callback is not None:
                callback(state, joint_action, subtask_0 or "", subtask_1 or "")

            _, sparse_r, done, env_info = oc_env.step(joint_action)
            total_sparse += float(sparse_r)
            if float(sparse_r) > 0:
                deliveries += 1

            shaped = env_info.get("shaped_r_by_agent", env_info.get("shaped_r", [0.0, 0.0]))
            if isinstance(shaped, (list, tuple)):
                total_shaped += float(sum(shaped))
            elif isinstance(shaped, dict):
                total_shaped += float(sum(shaped.values()))
            else:
                total_shaped += float(shaped)

            if done:
                game_done = True
                for idx in range(2):
                    if actions[idx] == "interact" and subtasks[idx] is not None:
                        completed[idx] = True
                break

            # Check if either agent just INTERACTed (completed their subtask).
            for idx in range(2):
                if actions[idx] == "interact" and subtasks[idx] is not None and not completed[idx]:
                    completed[idx] = True
                    if first_complete_step is None:
                        first_complete_step = step_i

            # End when BOTH agents have completed (or are idle).
            if all(completed[i] or subtasks[i] is None for i in range(2)):
                break
            # End after grace period expires (first agent done + N more steps).
            if first_complete_step is not None and (step_i - first_complete_step) >= GRACE_STEPS:
                break

        return {
            "shaped_reward": total_shaped,
            "sparse_reward": total_sparse,
            "deliveries": deliveries,
            "game_done": game_done,
            "completed_0": completed[0],
            "completed_1": completed[1],
        }

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
        eval_mode: bool = False,
    ) -> None:
        super().__init__()
        self.layout_spec = layout_spec
        self.hidden_spec = hidden_spec
        self._hidden_spec = hidden_spec  # alias for run_single_experiment compat
        # If the hidden spec has a populated preferred_ingredient_count, turn on
        # the continuous preferences branch in the CSP.
        cont_enabled = bool(
            hidden_spec and getattr(hidden_spec, "preferred_ingredient_count", {})
        )
        self.scene_spec = OvercookedSceneSpec(
            layout_spec=layout_spec,
            subtask_list=tuple(layout_spec.subtasks),
            continuous_prefs_enabled=cont_enabled,
        )
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
        self._mlam: Any = None
        self._overcooked_state: Any = None
        self._shared_counters: List[Any] = []
        self._feasibility: Dict[str, Dict[str, bool]] = {
            "robot": {s: True for s in ALL_SUBTASKS},
            "human": {s: True for s in ALL_SUBTASKS},
        }
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
        self._conflict_count: int = 0
        self._satisfaction_history: List[float] = []
        self._prediction_history: List[bool] = []
        self._forced_actor: Optional[str] = None
        self._deliveries: int = 0
        self._stall_count: int = 0
        self._last_primary: Optional[str] = None

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
            # Compute shared counters first (needed for proper MLAM params).
            self._overcooked_env.reset()
            try:
                from .layout_feasibility import (
                    compute_feasibility,
                    find_shared_counter_positions,
                    get_feasibility_summary,
                )
                self._shared_counters = find_shared_counter_positions(
                    self._mdp, self._overcooked_env.state
                )
            except Exception:
                self._shared_counters = []

            # Build planner with counter-aware MLAM (matches original benchmark).
            try:
                from .overcooked_planner import build_planner
                from overcooked_ai_py.agents.agent import GreedyHumanModel
                planner = build_planner(
                    self._mdp,
                    shared_counter_positions=self._shared_counters or None,
                )
                mlam = planner._mlam if planner is not None else None
                background = GreedyHumanModel(mlam) if mlam is not None else None
                if background is not None:
                    background.set_mdp(self._mdp)
                self._subtask_executor = _SubtaskExecutor(
                    planner=planner, background_agent=background, mdp=self._mdp
                )
                self._mlam = mlam
            except Exception:
                self._mlam = None

            # Compute per-agent feasibility using the counter-aware MLAM.
            try:
                self._feasibility = compute_feasibility(
                    self._mdp, self._overcooked_env.state, self._mlam
                )
                self._subtask_executor._feasibility_ref = self._feasibility
                self._subtask_executor._shared_counters = list(self._shared_counters)
                import logging
                logging.info(
                    f"Layout feasibility for {self.layout_spec.name}:\n"
                    + get_feasibility_summary(self._feasibility)
                )
            except Exception:
                self._feasibility = {
                    "robot": {s: True for s in ALL_SUBTASKS},
                    "human": {s: True for s in ALL_SUBTASKS},
                }
            # Update scene_spec with feasibility info (preserve
            # continuous_prefs_enabled from the initial construction).
            cont_enabled = bool(
                self.hidden_spec
                and getattr(self.hidden_spec, "preferred_ingredient_count", {})
            )
            self.scene_spec = OvercookedSceneSpec(
                layout_spec=self.layout_spec,
                subtask_list=tuple(self.layout_spec.subtasks),
                feasibility=self._feasibility,
                continuous_prefs_enabled=cont_enabled,
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
            # Access current state via .state property (overcooked_ai native env)
            self._overcooked_state = self._overcooked_env.state
            order_list = getattr(self._mdp, "order_list", None) or []
            self._pending_orders = len(order_list) if order_list else 3
        else:
            self._pending_orders = 3

        self._timestep = 0
        self._score_so_far = 0.0
        self._current_subtask_idx = 0
        self._conflict_count = 0
        self._satisfaction_history = []
        self._prediction_history = []
        self._forced_actor = None
        self._deliveries = 0
        self._stall_count = 0
        self._last_primary = None
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
        Parallel execution step: both agents work simultaneously.

        1. Get all available subtasks from game state.
        2. For the primary (highest-priority) subtask, the CSP decides via flag.
        3. The human independently picks a secondary task from remaining options.
        4. Both agents execute their subtasks simultaneously.
        5. Preference signals emitted for the primary subtask assignment.
           The human's independent choice is also recorded as a behavioral observation.
        """
        available = self._get_available_subtasks()
        if not available:
            available = [(self._fetch_ingredient_subtask(), None)]

        primary_subtask, primary_forced = available[0]
        current_subtask = primary_subtask

        # --- Deadlock detection ---
        # If the same subtask has been the primary 3+ times in a row without
        # progress, the game is deadlocked (e.g., both agents hold onions,
        # can't fetch dish). Force one agent to drop their item.
        if current_subtask == self._last_primary:
            self._stall_count += 1
        else:
            self._stall_count = 0
            self._last_primary = current_subtask

        if self._stall_count >= 3 and self._overcooked_available and self._overcooked_env is not None:
            # Deadlock: same subtask 3+ times with no progress.
            # Burn some primitive steps (both agents STAY) to let cooking
            # timers advance, then recalculate. This prevents the agent
            # from endlessly retrying a subtask they can't complete.
            try:
                for _ in range(20):
                    if self._overcooked_env.is_done():
                        break
                    self._overcooked_env.step(((0, 0), (0, 0)))
                self._stall_count = 0
            except Exception:
                pass

        # --- Primary subtask: CSP assignment or forced ---
        forced = primary_forced is not None

        if forced:
            robot_subtask: Optional[str] = primary_subtask if primary_forced == "robot" else None
            human_subtask: Optional[str] = primary_subtask if primary_forced == "human" else None
            task_score = 0.0
            conflict = False
            prediction_correct = True
            actor = primary_forced
        else:
            robot_claims = (action.flag == 0)
            human_claims = self._human_claims(current_subtask)

            if robot_claims and human_claims:
                # Conflict: human wins. task_score is +1 (clean behavioral label
                # — the human acted). The conflict diagnostic is preserved
                # separately in info["conflict"] and via the satisfaction
                # coordination penalty.
                self._conflict_count += 1
                actor = "human"
                task_score = 1.0
                conflict = True
                prediction_correct = False
            else:
                conflict = False
                if human_claims:
                    actor = "human"
                    task_score = 1.0
                    prediction_correct = True
                elif robot_claims:
                    actor = "robot"
                    task_score = -1.0
                    prediction_correct = True
                else:
                    actor = "robot"
                    task_score = -1.0
                    prediction_correct = False

            robot_subtask = current_subtask if actor == "robot" else None
            human_subtask = current_subtask if actor == "human" else None

        # --- Handoff detection: if assigned agent doesn't have the item ---
        # When the subtask requires an item (deliver needs soup, pickup_soup
        # needs dish, load_pot needs ingredient) and the assigned agent doesn't
        # have it but the other agent does, execute a handoff first:
        # other agent places on counter, assigned agent picks up, then continues.
        _ITEM_REQUIRED: Dict[str, set] = {
            "deliver": {"soup"},
            "pickup_soup": {"dish"},
            "load_pot": {"onion", "tomato"},
        }
        needs_handoff = False
        if not forced and not conflict and current_subtask in _ITEM_REQUIRED:
            required_items = _ITEM_REQUIRED[current_subtask]
            if self._overcooked_available and self._overcooked_env is not None:
                try:
                    state = self._overcooked_env.state
                    actor_idx = 0 if actor == "robot" else 1
                    other_idx = 1 - actor_idx
                    actor_held = state.players[actor_idx].get_object().name if state.players[actor_idx].has_object() else None
                    other_held = state.players[other_idx].get_object().name if state.players[other_idx].has_object() else None

                    if actor_held not in required_items and other_held in required_items:
                        # The other agent has what we need. In separated layouts,
                        # do a physical handoff via counter. In shared layouts,
                        # execute with the HOLDER but record the CSP's preference
                        # signal for the ASSIGNED agent — no physical handoff needed.
                        if has_separated_workspaces:
                            other_name = "human" if actor == "robot" else "robot"
                            try:
                                if actor_held is not None:
                                    self._subtask_executor.execute_dual(
                                        subtask_0="place_on_counter" if actor == "robot" else None,
                                        subtask_1="place_on_counter" if actor == "human" else None,
                                        oc_env=self._overcooked_env,
                                    )
                                    if self._overcooked_env.is_done():
                                        needs_handoff = True
                                        raise StopIteration
                                handoff_result = self._subtask_executor.execute_dual(
                                    subtask_0="place_on_counter" if other_name == "robot" else None,
                                    subtask_1="place_on_counter" if other_name == "human" else None,
                                    oc_env=self._overcooked_env,
                                )
                                if not handoff_result.get("game_done", False):
                                    self._subtask_executor.execute_dual(
                                        subtask_0="pickup_from_counter" if actor == "robot" else None,
                                        subtask_1="pickup_from_counter" if actor == "human" else None,
                                        oc_env=self._overcooked_env,
                                    )
                                needs_handoff = True
                            except (Exception, StopIteration):
                                needs_handoff = True
                        else:
                            # Shared layout: holder executes, CSP preference recorded.
                            # Swap the actor to the holder for physical execution.
                            other_name = "human" if actor == "robot" else "robot"
                            actor = other_name
                            robot_subtask = current_subtask if actor == "robot" else None
                            human_subtask = current_subtask if actor == "human" else None
                except Exception:
                    pass

        # --- Secondary subtask: the idle agent picks from remaining options ---
        # Only assign a secondary task if the layout has separated workspaces
        # (some subtasks are exclusive to one agent). In shared-space layouts
        # (CrampedRoom), parallel execution causes deadlocks.
        has_separated_workspaces = any(
            not ok for agent_feas in self._feasibility.values()
            for ok in agent_feas.values()
        )
        idle_agent = "human" if robot_subtask is not None and human_subtask is None else (
            "robot" if human_subtask is not None and robot_subtask is None else None
        )
        secondary_subtask: Optional[str] = None
        secondary_forced = False

        if has_separated_workspaces and idle_agent is not None and len(available) > 1:
            # Check if idle agent is holding an item — they need to deal with
            # it first (load/place/deliver) before taking a new task.
            if self._overcooked_available and self._overcooked_env is not None:
                try:
                    idle_idx = 0 if idle_agent == "robot" else 1
                    idle_player = self._overcooked_env.state.players[idle_idx]
                    idle_held = idle_player.get_object().name if idle_player.has_object() else None
                    if idle_held is not None:
                        # Agent holds something — determine what to do with it.
                        if idle_held in ("onion", "tomato"):
                            if self._feasibility.get(idle_agent, {}).get("load_pot", True):
                                secondary_subtask = "load_pot"
                            else:
                                secondary_subtask = "place_on_counter"
                        elif idle_held == "dish":
                            if self._feasibility.get(idle_agent, {}).get("pickup_soup", True):
                                secondary_subtask = "pickup_soup"
                            else:
                                secondary_subtask = "place_on_counter"
                        elif idle_held == "soup":
                            if self._feasibility.get(idle_agent, {}).get("deliver", True):
                                secondary_subtask = "deliver"
                            else:
                                secondary_subtask = "place_on_counter"
                        secondary_forced = True  # physical necessity
                except Exception:
                    pass

            # First pass: preference-based selection (only if no forced secondary).
            if secondary_subtask is None:
                for subtask_candidate, cand_forced in available[1:]:
                    if subtask_candidate == current_subtask:
                        continue
                    if not self._feasibility.get(idle_agent, {}).get(subtask_candidate, True):
                        continue
                    if cand_forced is not None and cand_forced != idle_agent:
                        continue
                    if idle_agent == "human" and cand_forced is None:
                        if not self._human_claims(subtask_candidate):
                            continue
                    secondary_subtask = subtask_candidate
                    secondary_forced = cand_forced is not None
                    break

            # Phase 4: Minimum-effort fallback — if no preferred task found,
            # pick the least-disliked feasible task above the threshold.
            # This models realistic behavior: a human who doesn't love any
            # available task will still help if the work isn't too aversive.
            if secondary_subtask is None and idle_agent == "human":
                threshold = self.config.task.min_effort_threshold
                best_candidate: Optional[str] = None
                best_logit = float("-inf")
                for subtask_candidate, cand_forced in available[1:]:
                    if subtask_candidate == current_subtask:
                        continue
                    if not self._feasibility.get("human", {}).get(subtask_candidate, True):
                        continue
                    if cand_forced is not None and cand_forced != "human":
                        continue
                    # Compute effective preference logit.
                    subtask_dim = self._subtask_index.get(subtask_candidate, 0)
                    psi_true = self._psi_true[subtask_dim] if subtask_dim < len(self._psi_true) else 0.0
                    phi = 0.0
                    if self.hidden_spec.hidden_hbm is not None:
                        try:
                            phi = float(self.hidden_spec.hidden_hbm.get_phi(
                                DEFAULT_HUMAN, self.layout_spec.name, subtask_candidate
                            ))
                        except Exception:
                            pass
                    logit = phi + psi_true
                    if logit > threshold and logit > best_logit:
                        best_logit = logit
                        best_candidate = subtask_candidate
                if best_candidate is not None:
                    secondary_subtask = best_candidate
                    secondary_forced = False  # still a preference signal (reluctant choice)

            if secondary_subtask is not None:
                if idle_agent == "robot":
                    robot_subtask = secondary_subtask
                else:
                    human_subtask = secondary_subtask

        # --- Check if game ended during handoff ---
        game_ended_in_handoff = False
        if self._overcooked_available and self._overcooked_env is not None:
            try:
                game_ended_in_handoff = self._overcooked_env.is_done()
            except Exception:
                pass

        if game_ended_in_handoff:
            # Game ended during handoff — return terminal state.
            result = {"shaped_reward": 0, "sparse_reward": 0, "game_done": True,
                      "completed_0": False, "completed_1": False, "deliveries": 0}
        else:
            # --- Execute both agents simultaneously ---
            result = self._subtask_executor.execute_dual(
                subtask_0=robot_subtask,
                subtask_1=human_subtask,
                oc_env=self._overcooked_env if self._overcooked_available else None,
                callback=self._make_render_callback(robot_subtask, human_subtask),
                both_use_planner=has_separated_workspaces,
            )

        shaped_reward = result["shaped_reward"] + result["sparse_reward"]
        game_done = result["game_done"]
        self._deliveries += result.get("deliveries", 0)
        primary_actor_idx = 0 if actor == "robot" else 1
        subtask_completed = result[f"completed_{primary_actor_idx}"]

        # If primary subtask wasn't completed (timeout), override task_score
        # to indicate no information. Conflicts still produce a clean +1 label
        # since the human ended up acting.
        if not subtask_completed:
            task_score = 0.0

        # --- Satisfaction for primary subtask ---
        # Emit satisfaction for any completed subtask, including conflicts.
        # On conflicts, prediction_correct=False so the coordination penalty is
        # applied — this captures coordination quality through the satisfaction
        # channel rather than discarding the observation. Only timeouts (no
        # completion) produce a zero satisfaction signal.
        if subtask_completed:
            satisfaction = self._compute_satisfaction(
                current_subtask, actor, prediction_correct
            )
        else:
            satisfaction = 0.0

        # --- FUTURE WORK: continuous preferences (phase 1: ingredient_count) ---
        # Gated: only fires when an ingredient_count is attached to the action
        # AND hidden_spec.preferred_ingredient_count is populated (empty dict by
        # default). For all existing configs this is a no-op.
        continuous_sat = self._compute_continuous_satisfaction(
            current_subtask, action.ingredient_count
        )
        if continuous_sat is not None and subtask_completed and not conflict:
            # Blend: 50% binary + 50% continuous
            satisfaction = 0.5 * satisfaction + 0.5 * continuous_sat

        self._satisfaction_history.append(satisfaction)
        self._prediction_history.append(prediction_correct)

        # --- Record human's secondary choice as behavioral observation ---
        # If the human independently chose a task, that's informative about preferences.
        human_secondary_sat = 0.0
        if (secondary_subtask is not None and idle_agent == "human"
                and not secondary_forced and result.get("completed_1", False)):
            human_secondary_sat = self._compute_satisfaction(
                secondary_subtask, "human", True
            )

        self._score_so_far += shaped_reward
        self._timestep += 1

        # Advance subtask state.
        if not self._overcooked_available:
            self._current_subtask_idx = (self._current_subtask_idx + 1) % len(self._subtasks)
        self._current_subtask = self._get_next_subtask()
        if self._current_subtask is not None and self._current_subtask in self._subtasks:
            self._current_subtask_idx = self._subtasks.index(self._current_subtask)

        terminated = game_done
        truncated = self._timestep >= self.layout_spec.episode_length

        pred_acc = (
            sum(self._prediction_history) / len(self._prediction_history)
            if self._prediction_history else 0.0
        )
        time_left = -1
        if self._overcooked_available and self._overcooked_env is not None:
            try:
                t = self._overcooked_env.state.timestep
                time_left = max(0, self._overcooked_env.horizon - t)
            except Exception:
                pass

        obs = self._make_obs()
        info: Dict[str, Any] = {
            "last_subtask": current_subtask,
            "last_actor": actor,
            "task_score": task_score,
            "satisfaction": satisfaction,
            "user_satisfaction": satisfaction,
            "conflict": conflict,
            "conflict_count": self._conflict_count,
            "shaped_reward": shaped_reward,
            "subtask_completed": subtask_completed,
            "prediction_correct": prediction_correct,
            "prediction_accuracy": pred_acc,
            "average_satisfaction": float(
                np.mean(self._satisfaction_history)
                if self._satisfaction_history else 0.0
            ),
            "forced": forced,
            "deliveries": self._deliveries,
            "robot_subtask": robot_subtask,
            "human_subtask": human_subtask,
            "secondary_subtask": secondary_subtask,
            "human_secondary_satisfaction": human_secondary_sat,
            "psi_true": self._psi_true,
            "session_type": self._session_type,
            "time_left": time_left,
            # Continuous preference info (phase 1: ingredient_count)
            "ingredient_count": action.ingredient_count,
            "continuous_satisfaction": continuous_sat,
        }
        return obs, 0.0, terminated or truncated, False, info

    def _make_render_callback(
        self, robot_subtask: Optional[str], human_subtask: Optional[str]
    ) -> Optional[Any]:
        """Build render callback for dual execution, or None if no callback set."""
        if self.render_callback is None:
            return None
        _r_sub = robot_subtask or ""
        _h_sub = human_subtask or ""
        _user_cb = self.render_callback

        def _cb(oc_state: Any, joint_action: Any, *args: Any) -> None:
            _user_cb(oc_state, joint_action, _r_sub, _h_sub)

        return _cb

    def _compute_satisfaction(
        self, subtask: str, actor: str, prediction_correct: bool
    ) -> float:
        """
        Compute continuous satisfaction for a completed subtask assignment.

        Returns a value in [-1, +1] encoding preference magnitude, session
        effects, and coordination quality. This is the signal the HBM learns
        from (Gaussian likelihood term). Mirrors spices _compute_satisfaction.

        The generative model:
            phi_latent = pref_sign * phi_mag
            phi_mag    = |hidden_theta(subtask)| when available, else base_task_bias
            logit      = actor_sign * (phi_latent + psi_true[subtask_dim])
            p          = (tanh(logit / T) + 1) / 2
            sat      ~ Beta(p * kappa, (1 - p) * kappa) rescaled to [-1, +1]

        Coordination cost: when robot mispredicts, coordination_cost is
        subtracted from satisfaction. This penalises poor prediction quality
        independently of preference strength.
        """
        subtask_dim = self._subtask_index.get(subtask, 0)
        psi_true = self._psi_true[subtask_dim] if subtask_dim < len(self._psi_true) else 0.0

        actor_sign = 1.0 if actor == "human" else -1.0

        # Determine true preference direction from hidden spec.
        preferred = self.hidden_spec.preferred_actor.get(subtask, "robot")
        pref_sign = 1.0 if preferred == "human" else -1.0

        # Use true phi magnitude if hidden HBM is available.
        phi_mag = self.config.task.base_task_bias
        if self.hidden_spec.hidden_hbm is not None:
            try:
                layout_name = self.layout_spec.name
                phi_val = self.hidden_spec.hidden_hbm.get_phi(
                    DEFAULT_HUMAN, layout_name, subtask
                )
                phi_mag = max(abs(float(phi_val)), 1e-6)
            except Exception:
                pass

        phi_latent = pref_sign * phi_mag
        logit = actor_sign * (phi_latent + psi_true)

        temp = max(self.config.task.task_logit_temperature, 1e-6)
        p = float(np.tanh(logit / temp))
        # Rescale tanh output from [-1, 1] to [0, 1] for Beta params.
        p = float(np.clip((p + 1.0) / 2.0, 1e-6, 1.0 - 1e-6))
        alpha = p * self.config.task.task_beta_kappa
        beta_param = (1.0 - p) * self.config.task.task_beta_kappa
        p_sampled = self._rng.beta(alpha, beta_param)
        raw_sat = float(np.clip(2.0 * p_sampled - 1.0, -1.0, 1.0))

        # Coordination cost: misprediction subtracts a flat penalty.
        if not prediction_correct:
            raw_sat -= self.config.task.coordination_cost
            raw_sat = max(raw_sat, -1.0)

        return raw_sat

    def _compute_continuous_satisfaction(
        self, subtask: str, chosen_count: Optional[int]
    ) -> Optional[float]:
        """
        Compute the continuous-preference satisfaction contribution for an
        ingredient-count decision. Returns None if no continuous preference
        is configured or the action doesn't carry an ingredient count.

        Generative model (Gaussian acceptance):
            preferred_mean, preferred_std = hidden_spec.preferred_ingredient_count[layout]
            s_raw = exp(-(chosen - preferred_mean)^2 / (2 * preferred_std^2))
            s = 2 * s_raw - 1  (rescale to [-1, +1])

        This is only applicable for the load_pot subtask since that's where
        the ingredient count decision is meaningful. For other subtasks, the
        function returns None (no contribution).
        """
        if chosen_count is None:
            return None
        if subtask != "load_pot":
            return None
        # Try both the pretty name and the canonical layout_name so configs
        # can key on either.
        pref = self.hidden_spec.preferred_ingredient_count.get(
            self.layout_spec.name
        ) or self.hidden_spec.preferred_ingredient_count.get(
            self.layout_spec.layout_name
        )
        if pref is None:
            return None
        try:
            mu, sigma = float(pref[0]), float(pref[1])
        except (TypeError, ValueError, IndexError):
            return None
        if sigma <= 0:
            sigma = 0.5  # fall back
        acceptance = float(np.exp(-((chosen_count - mu) ** 2) / (2.0 * sigma * sigma)))
        return 2.0 * acceptance - 1.0  # rescale to [-1, +1]

    def _human_claims(self, subtask: str) -> bool:
        """
        Sample whether the human autonomously claims this subtask.

        P(human claims) = sigma(phi[subtask] + psi_true[subtask_dim])

        If the human cannot physically reach the subtask's goal tiles
        (layout feasibility constraint), they never claim it regardless
        of preference.

        Uses the hidden HBM's theta if available; otherwise falls back to
        preferred_actor as a deterministic prior.
        """
        # Physical feasibility check: can the human reach this subtask?
        if not self._feasibility.get("human", {}).get(subtask, True):
            return False

        subtask_dim = self._subtask_index.get(subtask, 0)
        psi_true = self._psi_true[subtask_dim] if subtask_dim < len(self._psi_true) else 0.0

        if self.hidden_spec.hidden_hbm is not None:
            try:
                layout_name = self.layout_spec.name
                phi = float(self.hidden_spec.hidden_hbm.get_phi(DEFAULT_HUMAN, layout_name, subtask))
            except Exception:
                phi = 0.0
            logit = phi + psi_true
            p_claim = float(1.0 / (1.0 + np.exp(-logit)))
        else:
            preferred = self.hidden_spec.preferred_actor.get(subtask, "robot")
            p_claim = 1.0 if preferred == "human" else 0.0

        return bool(self._rng.random() < p_claim)

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
            forced_actor=self._forced_actor,
        )

    def _sample_session_type(self) -> str:
        """Sample a session type from the configured prior."""
        if self.hidden_spec.force_neutral_session:
            return "neutral"
        prior = self.config.get_session_type_prior()
        types = list(prior.keys())
        probs = list(prior.values())
        return str(self._rng.choice(types, p=probs))

    def _sample_psi_true(self, session_type: str) -> List[float]:
        """
        Sample the true per-subtask session offset vector.

        Neutral session: psi_true[d] ~ N(0, neutral_std²) per dimension.
        Non-neutral: psi_true[d] ~ N(weight[d] * ±mean_abs, std²) per dimension,
        where weight[d] comes from the named session profile in SessionConfig.

        The differentiated weights make certain subtasks more affected by
        fatigue/energy than others, giving vector psi a structural advantage
        over scalar psi.
        """
        from .config.session_profiles import get_session_profile

        sc = self.config.session
        n_dims = len(ALL_SUBTASKS)
        profile = get_session_profile(sc.session_profile_name)
        weights = profile.weights
        if len(weights) < n_dims:
            weights = weights + (1.0,) * (n_dims - len(weights))

        if session_type == "neutral":
            return [
                float(self._rng.normal(0.0, sc.session_neutral_std))
                for _ in range(n_dims)
            ]
        sign = 1.0 if session_type == "energised" else -1.0
        return [
            float(self._rng.normal(
                sign * weights[d] * sc.session_nonneutral_mean_abs,
                sc.session_nonneutral_std,
            ))
            for d in range(n_dims)
        ]

    # ------------------------------------------------------------------
    # Handoff helpers
    # ------------------------------------------------------------------

    def _needs_handoff(self, subtask: str, holder: Optional[str]) -> bool:
        """
        Return True if the subtask requires an item that is held by an agent
        who cannot perform the subtask (layout constraint), requiring a
        counter handoff.
        """
        if holder is None:
            return False
        can_holder_do_it = self._feasibility.get(holder, {}).get(subtask, True)
        return not can_holder_do_it

    def _counter_has_item(self) -> bool:
        """Return True if any shared counter tile has an object on it."""
        if not self._overcooked_available or self._overcooked_env is None:
            return False
        try:
            state = self._overcooked_env.state
            for pos in getattr(self, "_shared_counters", []):
                if state.has_object(pos):
                    return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Game-state-aware subtask sequencer
    # ------------------------------------------------------------------

    def _get_available_subtasks(self) -> List[Tuple[str, Optional[str]]]:
        """
        Return ALL currently feasible subtasks with their forced actor (if any).

        Returns a list of (subtask_name, forced_actor_or_None) tuples.
        The first entry is the highest-priority subtask (same as what
        _get_next_subtask would return). Additional entries are lower-priority
        subtasks that a second agent could work on simultaneously.

        Used by the parallel execution model: the CSP picks the robot's task,
        the human independently picks from the remaining available tasks.
        """
        if not self._overcooked_available or self._mdp is None or self._overcooked_env is None:
            st = self._subtasks[self._current_subtask_idx] if self._subtasks else None
            return [(st, None)] if st else []

        try:
            from overcooked_ai_py.mdp.overcooked_mdp import Recipe
            max_ing = Recipe.MAX_NUM_INGREDIENTS
        except Exception:
            max_ing = 3

        try:
            state = self._overcooked_env.state
            pot_states = self._mdp.get_pot_states(state)
            ready_pots = pot_states.get("ready", [])
            cooking_pots = pot_states.get("cooking", [])
            full_idle = pot_states.get(f"{max_ing}_items", [])
            partial_pots: List[Any] = []
            for i in range(1, max_ing):
                partial_pots.extend(pot_states.get(f"{i}_items", []))
            empty_pots = pot_states.get("empty", [])

            players = state.players
            ingredient_names = {"onion", "tomato"}

            available: List[Tuple[str, Optional[str]]] = []
            seen: set[str] = set()

            def _add(subtask: str, forced: Optional[str] = None) -> None:
                if subtask not in seen:
                    seen.add(subtask)
                    available.append((subtask, forced))

            def holder_of(names: set) -> Optional[str]:
                holders = []
                for idx, p in enumerate(players):
                    if p.has_object() and p.get_object().name in names:
                        holders.append("robot" if idx == 0 else "human")
                return holders[0] if len(holders) == 1 else None

            def any_holds(names: set) -> bool:
                return any(
                    p.has_object() and p.get_object().name in names
                    for p in players
                )

            # Check shared counter items
            counter_has_item = False
            for pos in getattr(self, "_shared_counters", []):
                try:
                    if state.has_object(pos):
                        counter_has_item = True
                        break
                except Exception:
                    pass

            if counter_has_item:
                robot_can = self._feasibility.get("robot", {}).get("pickup_from_counter", True)
                forced = "robot" if robot_can else "human"
                _add("pickup_from_counter", forced)

            # Check if this is a separated workspace (some subtasks exclusive).
            has_sep = any(
                not ok for agent_feas in self._feasibility.values()
                for ok in agent_feas.values()
            )

            # Soup in hand → deliver. CSP decides who; handoff if needed.
            if any_holds({"soup"}):
                h = holder_of({"soup"})
                if h and not self._feasibility.get(h, {}).get("deliver", True):
                    _add("place_on_counter", h)
                else:
                    _add("deliver", None)

            # Ready pot + dish → pickup soup. CSP decides who; handoff if needed.
            if ready_pots and any_holds({"dish"}):
                h = holder_of({"dish"})
                if h and not self._feasibility.get(h, {}).get("pickup_soup", True):
                    _add("place_on_counter", h)
                else:
                    _add("pickup_soup", None)

            # Ready pot, no dish → fetch_dish
            if ready_pots:
                _add("fetch_dish", None)

            # Full-idle pot → begin cooking (load_pot with empty hands)
            if full_idle:
                _add("load_pot", None)

            # Ingredient in hand AND pot can accept → load pot.
            # Don't offer load_pot if all pots are cooking/ready/full.
            if any_holds(ingredient_names) and (partial_pots or empty_pots):
                h = holder_of(ingredient_names)
                if h and not self._feasibility.get(h, {}).get("load_pot", True):
                    _add("place_on_counter", h)
                else:
                    _add("load_pot", None)

            # Pots need ingredients — only if pots can accept more AND
            # we don't already have enough ingredients in hand / on counter.
            if partial_pots or empty_pots:
                # Count how many more ingredients the pots need.
                needed = 0
                for pot_pos in empty_pots:
                    needed += max_ing
                for pot_pos in partial_pots:
                    try:
                        soup = state.get_object(pot_pos)
                        needed += max_ing - len(soup.ingredients)
                    except Exception:
                        needed += 1
                # Count ingredients already in hand or on counter.
                in_hand = sum(
                    1 for p in players
                    if p.has_object() and p.get_object().name in ingredient_names
                )
                on_counter = sum(
                    1 for pos in getattr(self, "_shared_counters", [])
                    if state.has_object(pos) and state.get_object(pos).name in ingredient_names
                )
                supply = in_hand + on_counter
                if supply < needed:
                    _add(self._fetch_ingredient_subtask(), None)

            # Pots cooking → pre-fetch dish for when they're ready.
            if cooking_pots:
                _add("fetch_dish", None)

            # Fallback: if nothing else is available, fetch_ingredient
            # only if pots actually need filling.
            if not available:
                if partial_pots or empty_pots:
                    available.append((self._fetch_ingredient_subtask(), None))
                elif cooking_pots:
                    available.append(("fetch_dish", None))
                else:
                    available.append((self._fetch_ingredient_subtask(), None))

            return available

        except Exception:
            st = self._subtasks[self._current_subtask_idx] if self._subtasks else None
            return [(st, None)] if st else []

    def _fetch_ingredient_subtask(self) -> str:
        """Return the appropriate fetch subtask for this layout.

        In tomato layouts (fetch_onion/fetch_tomato in subtask list),
        alternates between onion and tomato to give both dimensions
        equal observation time. In standard layouts, returns fetch_ingredient.
        """
        if "fetch_onion" in self._subtasks:
            # Alternate based on timestep
            if self._timestep % 2 == 0:
                return "fetch_onion"
            else:
                return "fetch_tomato"
        return "fetch_ingredient"

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

            def holder_of(names: set) -> Optional[str]:
                """Return 'robot'/'human' if exactly one agent holds an item in names."""
                holders = []
                for idx, p in enumerate(players):
                    if p.has_object() and p.get_object().name in names:
                        holders.append("robot" if idx == 0 else "human")
                return holders[0] if len(holders) == 1 else None

            # --- Check if a shared counter has an item waiting for pickup ---
            counter_item = None
            counter_pos = None
            for pos in getattr(self, "_shared_counters", []):
                try:
                    if state.has_object(pos):
                        counter_item = state.get_object(pos).name
                        counter_pos = pos
                        break
                except Exception:
                    pass

            # P0. Item on shared counter → pick it up (forced to the agent
            #     that can reach the counter AND the next goal).
            if counter_item is not None:
                # The agent on the receiving side should pick it up.
                robot_can = self._feasibility.get("robot", {}).get("pickup_from_counter", True)
                human_can = self._feasibility.get("human", {}).get("pickup_from_counter", True)
                # Prefer the agent that can use the item (e.g., robot for load_pot)
                if robot_can:
                    self._forced_actor = "robot"
                elif human_can:
                    self._forced_actor = "human"
                else:
                    self._forced_actor = None
                return "pickup_from_counter"

            def _force_holder_if_needed(
                holder: Optional[str], next_subtask: str
            ) -> Optional[str]:
                """
                Return a forced actor ONLY when layout constraints require it.

                If the holder can do the subtask → force them (physical necessity:
                only the person holding the item can use it, and they CAN reach
                the goal).

                If the holder CANNOT do the subtask → force a counter handoff
                so the other agent can take over.

                If there's no holder, or both agents can do the subtask freely,
                return None (let the CSP decide — this is a preference decision).
                """
                if holder is None:
                    return None
                # Can the holder do the next subtask?
                holder_can = self._feasibility.get(holder, {}).get(next_subtask, True)
                other = "human" if holder == "robot" else "robot"
                other_can = self._feasibility.get(other, {}).get(next_subtask, True)
                if not holder_can:
                    # Holder can't do it → must hand off via counter.
                    return holder  # force holder to place_on_counter
                if not other_can:
                    # Other agent can't do it → holder must continue.
                    return holder
                # Both can do it → don't force; let CSP decide.
                return None

            # P0b. Agent holds ingredient and a pot can accept it → load first.
            # This prevents deadlocks where both agents hold onions and can't
            # proceed to the serve chain (fetch_dish requires empty hands).
            if any_holds(ingredient_names) and (partial_pots or empty_pots):
                h = holder_of(ingredient_names)
                if h is None:
                    # Both agents hold ingredients — pick the one who CAN load.
                    for idx in range(2):
                        agent_name = "robot" if idx == 0 else "human"
                        if (players[idx].has_object()
                                and players[idx].get_object().name in ingredient_names):
                            if self._feasibility.get(agent_name, {}).get("load_pot", True):
                                h = agent_name
                                break
                            else:
                                # Can't load → place on counter.
                                self._forced_actor = agent_name
                                return "place_on_counter"
                if h is not None:
                    if not self._feasibility.get(h, {}).get("load_pot", True):
                        self._forced_actor = h
                        return "place_on_counter"
                    self._forced_actor = _force_holder_if_needed(h, "load_pot")
                    return "load_pot"

            # P1. Someone holds soup → deliver.
            # NOT forced — CSP decides who delivers. If the CSP assigns a
            # different agent than the holder, step() will insert a handoff.
            if any_holds({"soup"}):
                h = holder_of({"soup"})
                if h and not self._feasibility.get(h, {}).get("deliver", True):
                    self._forced_actor = h
                    return "place_on_counter"
                self._forced_actor = None  # let CSP decide
                return "deliver"

            # P2. Pot ready + someone holds dish → pickup soup.
            if ready_pots and any_holds({"dish"}):
                h = holder_of({"dish"})
                if h and not self._feasibility.get(h, {}).get("pickup_soup", True):
                    self._forced_actor = h
                    return "place_on_counter"
                self._forced_actor = None
                return "pickup_soup"

            # P3. Pot ready, no dish → fetch one.
            if ready_pots:
                self._forced_actor = None
                return "fetch_dish"

            # P4. Full-idle pot → begin cooking.
            if full_idle:
                self._forced_actor = None
                return "load_pot"

            # P5/P6. Pots need ingredients → fetch or load.
            if partial_pots or empty_pots:
                h = holder_of(ingredient_names)
                if h is not None:
                    if not self._feasibility.get(h, {}).get("load_pot", True):
                        self._forced_actor = h
                        return "place_on_counter"
                    self._forced_actor = _force_holder_if_needed(h, "load_pot")
                    return "load_pot"
                self._forced_actor = None
                return self._fetch_ingredient_subtask()

            # P5b. Agent holds surplus ingredient (no pot needs it).
            # In separated layouts: drop on counter for handoff.
            # In shared layouts: skip — let the agent hold it while P7 proceeds.
            # Dropping in shared layouts creates pickup/putdown loops.
            has_sep = any(
                not ok for agent_feas in self._feasibility.values()
                for ok in agent_feas.values()
            )
            if has_sep and any_holds(ingredient_names) and not partial_pots and not empty_pots:
                h = holder_of(ingredient_names)
                if h is None:
                    for idx in range(2):
                        agent_name = "robot" if idx == 0 else "human"
                        if (players[idx].has_object()
                                and players[idx].get_object().name in ingredient_names):
                            h = agent_name
                            break
                if h is not None:
                    self._forced_actor = h
                    return "place_on_counter"

            # P7. All pots cooking → pre-fetch dish.
            if cooking_pots:
                if any_holds({"dish"}):
                    h = holder_of({"dish"})
                    if h and not self._feasibility.get(h, {}).get("pickup_soup", True):
                        self._forced_actor = h
                        return "place_on_counter"
                    self._forced_actor = _force_holder_if_needed(h, "pickup_soup")
                    return "pickup_soup"
                self._forced_actor = None
                return "fetch_dish"

            # Absolute fallback
            self._forced_actor = None
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

    def save_state(self, filepath: Any) -> None:
        """Persist minimal state for crash debugging (matches other envs' interface)."""
        import pickle
        payload = {
            "timestep": self._timestep,
            "score_so_far": self._score_so_far,
            "session_type": self._session_type,
            "psi_true": self._psi_true,
            "conflict_count": self._conflict_count,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)
