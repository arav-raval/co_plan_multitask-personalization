"""
overcooked_planner.py — Low-level subtask executor for Overcooked AI.

This module is **fully detached** from the HBM, CSP, and satisfaction
subsystems.  It knows nothing about phi, psi, or preference learning.
Its only job is to convert a (subtask, actor_idx, game_state) tuple into
a sequence of overcooked_ai primitive actions.

Usage
-----
    from multitask_personalization.envs.overcooked.overcooked_planner import (
        SubtaskPlanner,
        build_planner,
    )

    planner = build_planner(mdp)
    # Execute one subtask step; returns the single primitive action to take.
    action = planner.step(subtask="load_pot", actor_idx=0, state=oc_state)

Design
------
The planner wraps overcooked_ai_py's MediumLevelActionManager (MLAM) and
MotionPlanner.  For each subtask it identifies the relevant goal tiles on the
gridworld and uses the motion planner to find the shortest path.

Actor indices follow overcooked_ai convention:
  0 = robot (agent 0 in overcooked_ai)
  1 = human (agent 1 in overcooked_ai)

Subtask → goal tile mapping
-----------------------------
  fetch_ingredient  → onion/tomato dispenser tile adjacent to a reachable cell
  load_pot          → pot tile (holding an ingredient → INTERACT at pot)
  fetch_dish        → dish dispenser tile
  pickup_soup       → pot tile (holding a dish → INTERACT at pot when done cooking)
  deliver           → serving counter tile

If overcooked_ai_py is not installed this module degrades gracefully:
``build_planner`` returns ``None`` and callers fall back to the STAY stub.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# overcooked_ai primitive actions (as direction tuples / strings)
# NORTH=(0,-1), SOUTH=(0,1), EAST=(1,0), WEST=(-1,0), STAY=(0,0), INTERACT='interact'
_STAY_ACTION = (0, 0)
_INTERACT_ACTION = "interact"

# Map from subtask name to the terrain type that is the goal.
# When the actor needs to *pick up* from a tile they must be adjacent; when
# they need to *interact* they stand adjacent and take INTERACT.
_SUBTASK_TERRAIN: Dict[str, str] = {
    "fetch_ingredient": "O",   # onion dispenser (most layouts); also 'T'
    "load_pot":         "P",   # pot
    "fetch_dish":       "D",   # dish dispenser
    "pickup_soup":      "P",   # pot again (this time to pick up cooked soup)
    "deliver":          "S",   # serving counter
}

# Secondary terrain fallback for fetch_ingredient when 'O' is absent
_INGREDIENT_FALLBACKS: List[str] = ["T"]


class SubtaskPlanner:
    """
    Low-level greedy planner that converts subtask assignments into
    overcooked_ai primitive actions.

    Parameters
    ----------
    mdp:
        An ``OvercookedGridworld`` instance.
    mlam:
        A ``MediumLevelActionManager`` built for *mdp*.  Pass ``None`` to
        force stub mode (always returns STAY).
    """

    def __init__(self, mdp: Any, mlam: Optional[Any] = None) -> None:
        self._mdp = mdp
        self._mlam = mlam
        # Cache of (subtask → goal positions) computed once from the gridworld.
        self._goal_positions: Dict[str, List[Tuple[int, int]]] = {}
        if mdp is not None:
            self._build_goal_cache()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        subtask: str,
        actor_idx: int,
        state: Any,
    ) -> Any:
        """
        Return the next primitive action for *actor_idx* to advance *subtask*.

        If the motion planner is unavailable (MLAM is None) or the goal
        cannot be reached, returns STAY.

        Parameters
        ----------
        subtask:
            One of the 5 canonical subtask names.
        actor_idx:
            0 for robot, 1 for human.
        state:
            Current ``OvercookedState`` from overcooked_ai.

        Returns
        -------
        The next primitive action (a tuple like (1, 0) or the string
        "interact"), suitable for passing directly to
        ``OvercookedEnv.step(joint_action)``.
        """
        if self._mlam is None or state is None:
            return _STAY_ACTION

        try:
            return self._greedy_action(subtask, actor_idx, state)
        except Exception:
            return _STAY_ACTION

    def plan_to_goal(
        self,
        actor_idx: int,
        state: Any,
        goal_pos_and_or: Tuple,
    ) -> List[Any]:
        """
        Return the full action sequence from the actor's current position to
        *goal_pos_and_or* using the motion planner.

        Returns an empty list if no path is found.
        """
        if self._mlam is None or state is None:
            return []
        try:
            agent = state.players[actor_idx]
            start = (agent.position, agent.orientation)
            plan, _path, _cost = self._mlam.motion_planner.get_plan(
                start, goal_pos_and_or
            )
            return list(plan)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_goal_cache(self) -> None:
        """Precompute goal positions for each subtask from the gridworld."""
        terrain_mtx = self._mdp.terrain_mtx
        for subtask, terrain_char in _SUBTASK_TERRAIN.items():
            positions = self._find_terrain(terrain_mtx, terrain_char)
            if not positions and subtask == "fetch_ingredient":
                # Fall back to tomato dispenser if no onion dispenser
                for alt in _INGREDIENT_FALLBACKS:
                    positions = self._find_terrain(terrain_mtx, alt)
                    if positions:
                        break
            self._goal_positions[subtask] = positions

    @staticmethod
    def _find_terrain(
        terrain_mtx: List[List[str]], char: str
    ) -> List[Tuple[int, int]]:
        """Return all (col, row) positions with the given terrain character."""
        positions = []
        for row_idx, row in enumerate(terrain_mtx):
            for col_idx, cell in enumerate(row):
                if cell == char:
                    positions.append((col_idx, row_idx))
        return positions

    def _greedy_action(self, subtask: str, actor_idx: int, state: Any) -> Any:
        """
        Plan one step toward the subtask goal using MLAM.

        Strategy:
        1.  Check if the actor is already adjacent to the goal terrain and
            can interact → return INTERACT.
        2.  Otherwise find the closest goal tile and use the motion planner
            to get the first action on the shortest path.
        """
        agent = state.players[actor_idx]
        agent_pos = agent.position
        agent_or = agent.orientation

        goal_positions = self._goal_positions.get(subtask, [])
        if not goal_positions:
            return _STAY_ACTION

        # Check if adjacent to AND facing a goal tile
        if self._can_interact_now(agent_pos, goal_positions, state, subtask, agent_or):
            return _INTERACT_ACTION

        # For pickup_soup: if already adjacent and facing the pot but soup isn't
        # ready yet, STAY and wait — do not let the motion planner re-issue INTERACT.
        if subtask == "pickup_soup":
            x, y = agent_pos
            for gx, gy in goal_positions:
                if abs(x - gx) + abs(y - gy) == 1:
                    ox, oy = agent_or
                    if (x + ox, y + oy) == (gx, gy):
                        # Adjacent and facing pot but soup not ready → wait
                        return _STAY_ACTION

        # Find closest goal and plan path
        closest_goal = min(
            goal_positions,
            key=lambda g: abs(g[0] - agent_pos[0]) + abs(g[1] - agent_pos[1]),
        )

        # Build goal pos_and_or: stand adjacent to the goal tile, facing it
        goal_pos_and_or = self._adjacent_goal(closest_goal, state, actor_idx)
        if goal_pos_and_or is None:
            return _STAY_ACTION

        try:
            plan, _path, _cost = self._mlam.motion_planner.get_plan(
                (agent_pos, agent_or), goal_pos_and_or
            )
            if plan:
                return plan[0]
        except Exception:
            pass

        return _STAY_ACTION

    def _can_interact_now(
        self,
        agent_pos: Tuple[int, int],
        goal_positions: List[Tuple[int, int]],
        state: Any,
        subtask: str,
        agent_or: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        True if the agent is adjacent to a goal tile AND facing it.

        Overcooked requires the agent to face the terrain tile to INTERACT.
        Adjacency alone is not sufficient — the agent must be oriented toward
        the tile (i.e. agent_pos + orientation == goal_pos).
        """
        x, y = agent_pos
        for gx, gy in goal_positions:
            if abs(x - gx) + abs(y - gy) == 1:
                # Check orientation: agent must be facing this goal tile
                if agent_or is not None:
                    ox, oy = agent_or
                    if (x + ox, y + oy) != (gx, gy):
                        continue  # adjacent but not facing — keep planning
                # Additional check: for pickup_soup the soup must be ready
                if subtask == "pickup_soup":
                    return self._soup_is_ready(state, (gx, gy))
                return True
        return False

    def _soup_is_ready(self, state: Any, pot_pos: Tuple[int, int]) -> bool:
        """Return True if the pot at *pot_pos* contains a cooked soup."""
        try:
            # Use mdp.get_pot_states — state has no pot_states attribute directly
            pot_states = self._mdp.get_pot_states(state)
            ready_positions = pot_states.get("ready", [])
            return pot_pos in ready_positions
        except Exception:
            pass
        # Fallback: check the object at that position directly
        try:
            obj = state.get_object(pot_pos)
            return obj is not None and obj.name == "soup" and obj.is_ready
        except Exception:
            pass
        return False

    def _adjacent_goal(
        self,
        goal_tile: Tuple[int, int],
        state: Any,
        actor_idx: int,
    ) -> Optional[Tuple]:
        """
        Return (position, orientation) for standing adjacent to *goal_tile*.

        Tries the four cardinal neighbors and returns the first reachable one
        (not a wall, not occupied by the other agent).
        """
        gx, gy = goal_tile
        other_idx = 1 - actor_idx
        other_pos = state.players[other_idx].position

        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W
        orientations = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # facing inward

        for (dx, dy), facing in zip(directions, orientations):
            neighbor = (gx + dx, gy + dy)
            if neighbor == other_pos:
                continue
            try:
                if self._mdp.get_terrain_type_at_pos(neighbor) == " ":
                    return (neighbor, facing)
            except Exception:
                continue
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_planner(mdp: Any) -> Optional[SubtaskPlanner]:
    """
    Construct a ``SubtaskPlanner`` with a full MLAM if overcooked_ai_py is
    available, otherwise return ``None``.

    Parameters
    ----------
    mdp:
        An ``OvercookedGridworld`` instance.
    """
    try:
        from overcooked_ai_py.planning.planners import (
            MediumLevelActionManager,
            NO_COUNTERS_PARAMS,
        )
        mlam = MediumLevelActionManager.from_pickle_or_compute(
            mdp, NO_COUNTERS_PARAMS, force_compute=False
        )
        return SubtaskPlanner(mdp, mlam)
    except ImportError:
        return None
    except Exception:
        # MLAM construction can fail for unusual layouts; fall back to stub
        return SubtaskPlanner(mdp, mlam=None)
