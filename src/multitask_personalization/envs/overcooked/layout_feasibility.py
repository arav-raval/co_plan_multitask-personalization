"""
layout_feasibility.py — Per-agent subtask reachability for Overcooked layouts.

Computes which subtasks each agent can physically perform in a given layout,
based on whether the agent can reach the required terrain tile (dispenser,
pot, serving counter, etc.) via the motion planner.

This is a hard physical constraint — independent of preferences. The CSP uses
it to avoid assigning subtasks to agents that cannot reach the goal tile.

Usage:
    feasibility = compute_feasibility(mdp, state)
    # feasibility["robot"]["fetch_ingredient"] = True/False
    # feasibility["human"]["deliver"] = True/False
"""

from __future__ import annotations

from typing import Any, Dict

from .layouts import ALL_SUBTASKS

# Maps subtask name → terrain character that the agent must be adjacent to.
_SUBTASK_TERRAIN: Dict[str, str] = {
    "fetch_ingredient": "O",  # onion dispenser (or 'T' for tomato)
    "load_pot":         "P",  # pot
    "fetch_dish":       "D",  # dish dispenser
    "pickup_soup":      "P",  # pot (to scoop cooked soup)
    "deliver":          "S",  # serving counter
}

# Additional terrain chars to check (tomato dispenser as alternative for fetch_ingredient).
_TERRAIN_ALTERNATIVES: Dict[str, list[str]] = {
    "fetch_ingredient": ["O", "T"],
}


def compute_feasibility(
    mdp: Any,
    state: Any,
    mlam: Any = None,
) -> Dict[str, Dict[str, bool]]:
    """
    Compute per-agent subtask feasibility for the current layout.

    Parameters
    ----------
    mdp : OvercookedGridworld
    state : OvercookedState (with player positions)
    mlam : MediumLevelActionManager (optional; computed if not provided)

    Returns
    -------
    Dict with keys "robot", "human", each mapping subtask → bool.
    True means the agent can physically reach the required terrain tile.
    """
    if mlam is None:
        try:
            from overcooked_ai_py.planning.planners import (
                MediumLevelActionManager,
                NO_COUNTERS_PARAMS,
            )
            mlam = MediumLevelActionManager.from_pickle_or_compute(
                mdp, NO_COUNTERS_PARAMS, force_compute=False
            )
        except Exception:
            # If we can't compute, assume all feasible.
            return {
                "robot": {s: True for s in ALL_SUBTASKS},
                "human": {s: True for s in ALL_SUBTASKS},
            }

    mp = mlam.motion_planner
    terrain_mtx = mdp.terrain_mtx
    result: Dict[str, Dict[str, bool]] = {"robot": {}, "human": {}}

    # Find shared counter positions for handoff subtasks.
    shared_counters = find_shared_counter_positions(mdp, state)

    for subtask in ALL_SUBTASKS:
        if subtask in ("place_on_counter", "pickup_from_counter"):
            # Handoff subtasks target shared counter tiles.
            # Both agents can do these if shared counters exist.
            for agent_idx in range(2):
                agent_name = "robot" if agent_idx == 0 else "human"
                if not shared_counters:
                    result[agent_name][subtask] = False
                else:
                    player = state.players[agent_idx]
                    can_reach = _can_agent_reach_any(
                        mp, mdp, player.position, player.orientation,
                        shared_counters,
                    )
                    result[agent_name][subtask] = can_reach
            continue

        # Standard subtasks: check terrain reachability.
        terrain_chars = _TERRAIN_ALTERNATIVES.get(subtask, [_SUBTASK_TERRAIN[subtask]])
        goal_positions: list[tuple[int, int]] = []
        for r, row in enumerate(terrain_mtx):
            for c, cell in enumerate(row):
                if cell in terrain_chars:
                    goal_positions.append((c, r))

        for agent_idx in range(2):
            agent_name = "robot" if agent_idx == 0 else "human"
            player = state.players[agent_idx]
            can_reach = _can_agent_reach_any(
                mp, mdp, player.position, player.orientation, goal_positions
            )
            result[agent_name][subtask] = can_reach

    return result


def _can_agent_reach_any(
    mp: Any,
    mdp: Any,
    start_pos: tuple[int, int],
    start_orient: tuple[int, int],
    goal_positions: list[tuple[int, int]],
) -> bool:
    """Check if an agent can reach any walkable cell adjacent to any goal position."""
    orientations = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    start = (start_pos, start_orient)

    for goal_pos in goal_positions:
        gx, gy = goal_pos
        for dx, dy in orientations:
            neighbor = (gx + dx, gy + dy)
            try:
                if mdp.get_terrain_type_at_pos(neighbor) != " ":
                    continue
            except Exception:
                continue
            for orient in orientations:
                goal = (neighbor, orient)
                try:
                    if mp.is_valid_motion_start_goal_pair(start, goal):
                        return True
                except Exception:
                    pass
    return False


def find_shared_counter_positions(mdp: Any, state: Any) -> list[tuple[int, int]]:
    """
    Find counter (X) tiles that both agents can INTERACT with from adjacent
    walkable cells on opposite sides of the counter.

    In forced_coordination, the column-2 wall has tiles that both agents
    can reach from their respective sides. These are the handoff positions.
    """
    terrain = mdp.terrain_mtx
    # Find walkable cells reachable by each agent
    # We use a simple heuristic: partition walkable cells by connected component
    # relative to each agent's starting position.
    players = state.players

    def walkable_neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = pos
        result = []
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            try:
                if mdp.get_terrain_type_at_pos((nx, ny)) == " ":
                    result.append((nx, ny))
            except Exception:
                pass
        return result

    def flood_fill(start: tuple[int, int]) -> set[tuple[int, int]]:
        visited: set[tuple[int, int]] = set()
        queue = [start]
        while queue:
            pos = queue.pop()
            if pos in visited:
                continue
            visited.add(pos)
            queue.extend(walkable_neighbors(pos))
        return visited

    reachable_0 = flood_fill(players[0].position)
    reachable_1 = flood_fill(players[1].position)

    shared_counters: list[tuple[int, int]] = []
    for r, row in enumerate(terrain):
        for c, cell in enumerate(row):
            if cell != "X":
                continue
            pos = (c, r)
            # Check if this X tile has adjacent walkable cells reachable
            # by agent 0 AND by agent 1 (possibly on different sides).
            adj_by_0 = False
            adj_by_1 = False
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                neighbor = (c + dx, r + dy)
                if neighbor in reachable_0:
                    adj_by_0 = True
                if neighbor in reachable_1:
                    adj_by_1 = True
            if adj_by_0 and adj_by_1:
                shared_counters.append(pos)

    return shared_counters


def get_feasibility_summary(feasibility: Dict[str, Dict[str, bool]]) -> str:
    """Format feasibility as a human-readable string for logging."""
    lines = []
    for agent in ["robot", "human"]:
        reachable = [s for s, ok in feasibility[agent].items() if ok]
        unreachable = [s for s, ok in feasibility[agent].items() if not ok]
        if unreachable:
            lines.append(f"  {agent}: can reach {reachable}, CANNOT reach {unreachable}")
        else:
            lines.append(f"  {agent}: can reach all subtasks")
    return "\n".join(lines)
