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
    "fetch_onion":      "O",   # onion dispenser only
    "fetch_tomato":     "T",   # tomato dispenser only
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

    def __init__(
        self,
        mdp: Any,
        mlam: Optional[Any] = None,
        shared_counter_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self._mdp = mdp
        self._mlam = mlam
        # Shared counter tiles for handoff subtasks (computed externally).
        self._shared_counters: List[Tuple[int, int]] = (
            list(shared_counter_positions) if shared_counter_positions else []
        )
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
        # Handoff subtasks target counter tiles.
        # For place_on_counter, use ALL counter locations (agents can drop items
        # on any reachable counter, not just shared ones).
        # For pickup_from_counter, use shared counters (cross-agent handoffs).
        all_counters = list(self._shared_counters)
        if self._mdp is not None:
            try:
                all_counters = list(set(all_counters + self._mdp.get_counter_locations()))
            except Exception:
                pass
        self._goal_positions["place_on_counter"] = all_counters
        self._goal_positions["pickup_from_counter"] = list(self._shared_counters) if self._shared_counters else all_counters

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

        goal_positions = list(self._goal_positions.get(subtask, []))
        if not goal_positions:
            return _STAY_ACTION

        # For counter subtasks, filter to actionable positions.
        if subtask == "pickup_from_counter":
            goal_positions = [g for g in goal_positions if state.has_object(g)]
            if not goal_positions:
                return _STAY_ACTION
        elif subtask == "place_on_counter":
            goal_positions = [g for g in goal_positions if not state.has_object(g)]
            if not goal_positions:
                return _STAY_ACTION

        # Check if adjacent to AND facing a goal tile
        if self._can_interact_now(agent_pos, goal_positions, state, subtask, agent_or):
            return _INTERACT_ACTION

        # Check if adjacent but facing wrong direction — turn to face the goal.
        x, y = agent_pos
        for gx, gy in goal_positions:
            if abs(x - gx) + abs(y - gy) == 1:
                needed_orient = (gx - x, gy - y)
                if agent_or != needed_orient:
                    # Return the direction as a movement action — in overcooked_ai,
                    # moving toward a wall/counter turns the agent to face it.
                    return needed_orient

        # For pickup_soup: if already adjacent and facing the pot but soup isn't
        # ready yet, STAY and wait — do not let the motion planner re-issue INTERACT.
        if subtask == "pickup_soup":
            for gx, gy in goal_positions:
                if abs(x - gx) + abs(y - gy) == 1:
                    ox, oy = agent_or
                    if (x + ox, y + oy) == (gx, gy):
                        return _STAY_ACTION

        # Find closest goal and plan path
        closest_goal = min(
            goal_positions,
            key=lambda g: abs(g[0] - agent_pos[0]) + abs(g[1] - agent_pos[1]),
        )

        # Build goal pos_and_or: stand adjacent to the goal tile, facing it
        goal_pos_and_or = self._adjacent_goal(closest_goal, state, actor_idx)
        if goal_pos_and_or is None:
            # Fallback for counter subtasks: use greedy navigation.
            # The motion planner may not support arbitrary counter positions.
            if subtask in ("place_on_counter", "pickup_from_counter"):
                return self._greedy_navigate_to_counter(
                    closest_goal, agent_pos, state, actor_idx
                )
            return _STAY_ACTION

        try:
            plan, _path, _cost = self._mlam.motion_planner.get_plan(
                (agent_pos, agent_or), goal_pos_and_or
            )
            if plan:
                return plan[0]
        except Exception:
            # Fallback for counter subtasks
            if subtask in ("place_on_counter", "pickup_from_counter"):
                return self._greedy_navigate_to_counter(
                    closest_goal, agent_pos, state, actor_idx
                )

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
                # Additional checks for specific subtasks
                if subtask == "pickup_soup":
                    return self._soup_is_ready(state, (gx, gy))
                if subtask == "pickup_from_counter":
                    # Counter must have an object to pick up
                    return state.has_object((gx, gy))
                if subtask == "place_on_counter":
                    # Counter must be empty to place on
                    return not state.has_object((gx, gy))
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

    def _greedy_navigate_to_counter(
        self,
        counter_pos: Tuple[int, int],
        agent_pos: Tuple[int, int],
        state: Any,
        actor_idx: int,
    ) -> Any:
        """
        Simple greedy navigation to a counter tile, bypassing the motion planner.

        The motion planner's precomputed graph may not include paths to counter
        tiles (it's optimized for dispensers/pots/serving). This method uses
        simple Manhattan-distance-based greedy movement.
        """
        cx, cy = counter_pos
        ax, ay = agent_pos
        other_idx = 1 - actor_idx
        other_pos = state.players[other_idx].position

        # Find the walkable cell adjacent to the counter that we should stand on.
        best_neighbor = None
        best_dist = float("inf")
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            neighbor = (cx + dx, cy + dy)
            if neighbor == other_pos:
                continue
            try:
                if self._mdp.get_terrain_type_at_pos(neighbor) != " ":
                    continue
            except Exception:
                continue
            dist = abs(neighbor[0] - ax) + abs(neighbor[1] - ay)
            if dist < best_dist:
                best_dist = dist
                best_neighbor = neighbor

        if best_neighbor is None:
            return _STAY_ACTION

        # If we're already at the target neighbor, face the counter.
        if agent_pos == best_neighbor:
            needed_orient = (cx - ax, cy - ay)
            return needed_orient  # turn to face counter

        # Move toward the target neighbor greedily.
        nx, ny = best_neighbor
        dx = nx - ax
        dy = ny - ay
        # Prefer the axis with larger distance.
        if abs(dx) >= abs(dy):
            move = (1 if dx > 0 else -1, 0)
        else:
            move = (0, 1 if dy > 0 else -1)

        # Check if move is valid (walkable, not occupied).
        target = (ax + move[0], ay + move[1])
        if target == other_pos:
            # Try the other axis.
            if abs(dx) >= abs(dy):
                move = (0, 1 if dy > 0 else (-1 if dy < 0 else 1))
            else:
                move = (1 if dx > 0 else (-1 if dx < 0 else 1), 0)
            target = (ax + move[0], ay + move[1])

        try:
            if self._mdp.get_terrain_type_at_pos(target) != " ":
                return _STAY_ACTION
        except Exception:
            return _STAY_ACTION

        return move

    def _adjacent_goal(
        self,
        goal_tile: Tuple[int, int],
        state: Any,
        actor_idx: int,
    ) -> Optional[Tuple]:
        """
        Return (position, orientation) for standing adjacent to *goal_tile*.

        Tries the four cardinal neighbors and returns one that is:
        - walkable (terrain ' ')
        - not occupied by the other agent
        - reachable by the assigned agent (verified via motion planner)

        The closest reachable neighbor is preferred.
        """
        gx, gy = goal_tile
        agent = state.players[actor_idx]
        other_idx = 1 - actor_idx
        other_pos = state.players[other_idx].position

        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W
        orientations = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # facing inward

        candidates: List[Tuple] = []
        for (dx, dy), facing in zip(directions, orientations):
            neighbor = (gx + dx, gy + dy)
            if neighbor == other_pos:
                continue
            try:
                if self._mdp.get_terrain_type_at_pos(neighbor) != " ":
                    continue
            except Exception:
                continue
            # Verify the agent can actually reach this neighbor via motion planner.
            if self._mlam is not None:
                try:
                    start = (agent.position, agent.orientation)
                    goal = (neighbor, facing)
                    if not self._mlam.motion_planner.is_valid_motion_start_goal_pair(
                        start, goal
                    ):
                        continue
                except Exception:
                    continue
            candidates.append((neighbor, facing))

        if not candidates:
            return None

        # Return closest candidate by Manhattan distance.
        return min(
            candidates,
            key=lambda c: abs(c[0][0] - agent.position[0]) + abs(c[0][1] - agent.position[1]),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_planner(
    mdp: Any,
    shared_counter_positions: Optional[List[Tuple[int, int]]] = None,
) -> Optional[SubtaskPlanner]:
    """
    Construct a ``SubtaskPlanner`` with a full MLAM if overcooked_ai_py is
    available, otherwise return ``None``.

    Parameters
    ----------
    mdp:
        An ``OvercookedGridworld`` instance.
    shared_counter_positions:
        Counter tile positions accessible from both sides of the kitchen.
        When provided, the MLAM is built with counter_drop / counter_pickup
        support so the motion planner natively plans paths to/from counters.
        This matches the original Overcooked AI benchmark's approach.
    """
    try:
        from overcooked_ai_py.planning.planners import (
            MediumLevelActionManager,
            NO_COUNTERS_PARAMS,
        )
        counters = list(shared_counter_positions) if shared_counter_positions else []
        if counters:
            # Build with counter support — matches the original benchmark.
            mlam_params = {
                "start_orientations": False,
                "wait_allowed": False,
                "counter_goals": counters,
                "counter_drop": counters,
                "counter_pickup": counters,
                "same_motion_goals": True,
            }
        else:
            mlam_params = NO_COUNTERS_PARAMS

        mlam = MediumLevelActionManager.from_pickle_or_compute(
            mdp, mlam_params, force_compute=bool(counters)
        )
        return SubtaskPlanner(mdp, mlam, shared_counter_positions=counters)
    except ImportError:
        return None
    except Exception:
        # MLAM construction can fail for unusual layouts; fall back to stub
        return SubtaskPlanner(mdp, mlam=None)
