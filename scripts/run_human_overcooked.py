"""
run_human_overcooked.py — Play Overcooked with a real human via keyboard.

The human controls agent 1 (green hat) with arrow keys + spacebar.
The robot (agent 0, blue hat) is controlled by the CSP + planner.
After each episode the HBM updates phi/theta/psi from observed behavior.

Multiple players (family members) are supported: switch between players
across episodes to learn per-person preferences and population-level mu.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_human_overcooked.py

Controls:
    Arrow keys  Move (up/down/left/right)
    Space       Interact (pick up / place / serve)
    S           Stay (do nothing this tick)
    P           Pause / unpause
    Esc / Q     Quit

Options:
    --layout     Layout name (default: CrampedRoom)
    --episodes   Number of episodes per player (default: 3)
    --fps        Game speed in ticks per second (default: 6)
    --log-dir    Directory for JSON preference logs (default: logs/human_play/)
    --seed       Random seed (default: 42)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Overcooked-AI primitive actions
# ---------------------------------------------------------------------------
ACTION_NORTH = (0, -1)
ACTION_SOUTH = (0, 1)
ACTION_EAST = (1, 0)
ACTION_WEST = (-1, 0)
ACTION_STAY = (0, 0)
ACTION_INTERACT = "interact"


# ---------------------------------------------------------------------------
# Subtask detection from low-level state transitions
# ---------------------------------------------------------------------------

class SubtaskDetector:
    """Infer which subtask the human just completed from state changes.

    Watches the overcooked_ai game state before/after each tick and detects
    when the human (agent 1) completes a meaningful action:
      - Picked up an ingredient from a dispenser  -> fetch_ingredient
      - Placed an ingredient into a pot            -> load_pot
      - Picked up a dish from the dish dispenser   -> fetch_dish
      - Scooped soup from a pot into a dish        -> pickup_soup
      - Delivered soup at the serving counter       -> deliver
    """

    def __init__(self, mdp: Any) -> None:
        self._mdp = mdp

    def detect(
        self,
        state_before: Any,
        state_after: Any,
        human_action: Any,
        human_idx: int = 1,
    ) -> Optional[str]:
        """Return the subtask name the human completed, or None."""
        try:
            player_before = state_before.players[human_idx]
            player_after = state_after.players[human_idx]
            held_before = (
                player_before.get_object().name
                if player_before.has_object() else None
            )
            held_after = (
                player_after.get_object().name
                if player_after.has_object() else None
            )

            # Only detect on INTERACT actions
            if human_action != ACTION_INTERACT:
                return None

            # Picked up ingredient (nothing -> onion/tomato)
            if held_before is None and held_after in ("onion", "tomato"):
                return "fetch_ingredient"

            # Placed ingredient into pot (onion/tomato -> nothing)
            if held_before in ("onion", "tomato") and held_after is None:
                return "load_pot"

            # Picked up dish (nothing -> dish)
            if held_before is None and held_after == "dish":
                return "fetch_dish"

            # Scooped soup from pot (dish -> soup)
            if held_before == "dish" and held_after == "soup":
                return "pickup_soup"

            # Delivered soup (soup -> nothing, at serving counter)
            if held_before == "soup" and held_after is None:
                return "deliver"

        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Recipe-aware robot controller
# ---------------------------------------------------------------------------

class RobotController:
    """Robot that understands the Overcooked recipe state machine and defers
    to the human.

    Design principles:
      1. **Recipe-aware**: reads pot state to know exactly what the kitchen
         needs next (ingredients? start cooking? fetch dish? deliver?).
      2. **Human-aware**: infers what the human is *currently doing* from
         their held item and position, then picks a *complementary* task.
      3. **Deferential**: yields when adjacent to the human, never blocks
         key tiles (pot, dispensers) when empty-handed.
      4. **HBM-guided**: uses phi/entropy to break ties — early episodes
         explore uncertain subtasks, later episodes exploit learned prefs.
    """

    STUCK_THRESHOLD = 3

    def __init__(
        self,
        mdp: Any,
        planner: Any,
        hbm: Any,
        subtasks: List[str],
        layout_name: str,
    ) -> None:
        self._mdp = mdp
        self._planner = planner
        self._hbm = hbm
        self._subtasks = subtasks
        self._layout = layout_name

        self._current_subtask: Optional[str] = None
        self._player_name: Optional[str] = None
        self._consecutive_stays = 0
        self._adjacent_ticks = 0  # how long we've been next to human
        self._human_idle_ticks = 0  # how long human has been idle
        self._last_human_pos: Optional[Tuple[int, int]] = None
        self._episode_num = 0

        # Cache pot positions (fixed per layout)
        self._pot_positions: List[Tuple[int, int]] = []
        try:
            for r, row in enumerate(mdp.terrain_mtx):
                for c, cell in enumerate(row):
                    if cell == "P":
                        self._pot_positions.append((c, r))
        except Exception:
            pass

        # Neutral position: a walkable cell that doesn't block any
        # dispenser, pot, or serving counter. Precomputed from layout.
        self._neutral_pos = self._find_neutral_pos(mdp)

    def _find_neutral_pos(self, mdp: Any) -> Tuple[int, int]:
        """Find the best neutral parking position — a walkable cell that
        is not adjacent to any functional terrain (pot, dispenser, serve)."""
        functional = set()
        walkable = []
        try:
            for r, row in enumerate(mdp.terrain_mtx):
                for c, cell in enumerate(row):
                    if cell in ("P", "O", "T", "D", "S"):
                        functional.add((c, r))
                    elif cell == " ":
                        walkable.append((c, r))
        except Exception:
            return (2, 2)  # fallback for CrampedRoom

        # Score each walkable cell: prefer cells not adjacent to functional
        best = None
        best_score = -1
        for (cx, cy) in walkable:
            adj_functional = 0
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                if (cx + dx, cy + dy) in functional:
                    adj_functional += 1
            # Lower is better (fewer blocked things)
            score = -adj_functional
            if score > best_score:
                best_score = score
                best = (cx, cy)

        return best or (2, 2)

    def set_player(self, name: str) -> None:
        self._player_name = name

    def new_episode(self, episode_num: int) -> None:
        self._episode_num = episode_num
        self._current_subtask = None
        self._consecutive_stays = 0
        self._adjacent_ticks = 0
        self._human_idle_ticks = 0
        self._last_human_pos = None

    @property
    def current_subtask(self) -> Optional[str]:
        return self._current_subtask

    # ------------------------------------------------------------------
    # Main step: recipe logic → planner → anti-block safety
    # ------------------------------------------------------------------

    def step(self, state: Any) -> Any:
        """Return the robot's primitive action for this tick."""
        if self._player_name is None:
            return ACTION_STAY

        robot = state.players[0]
        human = state.players[1]
        robot_pos = robot.position
        human_pos = human.position
        dist = abs(robot_pos[0] - human_pos[0]) + abs(robot_pos[1] - human_pos[1])

        # --- Track human idle time ---
        # Human is "idle" if they haven't moved and aren't holding anything.
        human_held = human.get_object().name if human.has_object() else None
        if (human_pos == self._last_human_pos
                and human_held is None):
            self._human_idle_ticks += 1
        else:
            self._human_idle_ticks = 0
        self._last_human_pos = human_pos

        # --- Safety: yield if blocking the human ---
        # Only yield if adjacent for several ticks (not just passing by)
        # and the robot is empty-handed (not actively working).
        if dist <= 1 and not robot.has_object():
            self._adjacent_ticks += 1
            if self._adjacent_ticks >= 3:
                action = self._move_away_from(human_pos, robot_pos, state)
                if action != ACTION_STAY:
                    self._consecutive_stays = 0
                    return action
        else:
            self._adjacent_ticks = 0

        # --- Pick subtask from recipe state + human awareness ---
        self._current_subtask = self._read_recipe_state(state)

        if self._current_subtask is None:
            # No task — go to neutral position instead of blocking things
            self._current_subtask = "idle"
            if robot_pos != self._neutral_pos:
                action = self._navigate_to(self._neutral_pos, robot_pos, state)
                return action
            self._consecutive_stays = 0
            return ACTION_STAY

        # --- Execute via planner ---
        try:
            action = self._planner.step(self._current_subtask, 0, state)
        except Exception:
            action = ACTION_STAY

        # --- Anti-stuck ---
        # The planner doesn't account for the other player, so it may
        # repeatedly try to move into the human's cell. Detect this and
        # try an alternate direction.
        if action == ACTION_STAY:
            self._consecutive_stays += 1
        elif action != ACTION_INTERACT:
            # Check if the planned move would collide with the human
            dx, dy = action if isinstance(action, tuple) else (0, 0)
            target = (robot_pos[0] + dx, robot_pos[1] + dy)
            if target == human_pos:
                # Planner wants to move into human's cell — go around
                action = self._unstick(robot_pos, state)
                self._consecutive_stays += 1
            else:
                self._consecutive_stays = 0
        else:
            self._consecutive_stays = 0

        if self._consecutive_stays >= self.STUCK_THRESHOLD:
            action = self._unstick(robot_pos, state)
            self._consecutive_stays = 0

        return action

    # ------------------------------------------------------------------
    # Recipe state machine: what does the kitchen need RIGHT NOW?
    # ------------------------------------------------------------------

    def _read_recipe_state(self, state: Any) -> Optional[str]:
        """Determine the single best subtask for the robot given the full
        kitchen state AND what the human is currently doing.

        Returns the *effective* subtask (what the planner should execute),
        not a high-level goal. Returns None if the robot should idle.

        Advanced recipe logic:
          - If human loaded some onions then grabbed a plate, robot finishes
            loading the remaining onions and starts cooking.
          - Robot only approaches the pot when it needs to INTERACT (load
            ingredient or start cooking), not to "stand near it".
          - When pot is cooking, robot pre-fetches a dish (productive wait).
        """
        robot = state.players[0]
        human = state.players[1]
        robot_held = robot.get_object().name if robot.has_object() else None
        human_held = human.get_object().name if human.has_object() else None

        # --- Read pot state first (needed by all branches) ---
        try:
            pot_states = self._mdp.get_pot_states(state)
        except Exception:
            pot_states = {}

        ready_pots = pot_states.get("ready", [])
        cooking_pots = pot_states.get("cooking", [])
        full_unstarted = pot_states.get("3_items", [])

        ingredients_in_pot = 0
        for key in ["1_items", "2_items", "3_items"]:
            if pot_states.get(key):
                try:
                    ingredients_in_pot = int(key.split("_")[0])
                except ValueError:
                    pass
        ingredients_needed = 3 - ingredients_in_pot
        pot_is_empty = bool(pot_states.get("empty"))
        if pot_is_empty:
            ingredients_needed = 3

        human_is_idle = self._human_idle_ticks >= 18

        # --- If holding something, finish the chain ---
        if robot_held in ("onion", "tomato"):
            return "load_pot"
        if robot_held == "dish":
            if ready_pots:
                return "pickup_soup"
            # Soup not ready. If pot still needs ingredients and human is
            # idle, put the dish down on a counter and go help load.
            if human_is_idle and ingredients_needed > 0:
                return "place_on_counter"
            # Otherwise wait at neutral (don't hover near pot)
            return None
        if robot_held == "soup":
            return "deliver"

        # --- Infer what the human is doing ---
        human_task = self._infer_human_task(human_held, human.position)

        # --- Priority 1: Full pot needs cooking activation ---
        # Critical: if pot has 3 items and isn't cooking, someone must
        # INTERACT to start it. If human is idle or not heading there,
        # robot does it immediately.
        if full_unstarted:
            if human_is_idle or human_task != "load_pot":
                return "load_pot"

        # --- Priority 2: Soup ready → fetch dish ---
        if ready_pots:
            if human_held != "dish" and human_task != "fetch_dish":
                return "fetch_dish"
            # Human is handling it — idle
            return None

        # --- Priority 3: Cooking → pre-fetch dish ---
        # Don't approach the pot — go to the dish dispenser instead.
        if cooking_pots:
            if human_task != "fetch_dish" and human_held != "dish":
                return "fetch_dish"
            return None

        # --- Priority 4: Pot needs ingredients ---
        if ingredients_needed > 0:
            # Human switched away (e.g., grabbed a plate) while pot is
            # partially loaded → robot finishes loading.
            human_switched_away = (
                human_held in ("dish", "soup")
                or human_task in ("fetch_dish", "pickup_soup", "deliver")
            )

            if human_switched_away:
                return "fetch_ingredient"

            # Human has been idle too long — robot takes initiative
            if human_is_idle:
                return "fetch_ingredient"

            # Human is actively loading → robot pre-fetches dish
            if human_task in ("fetch_ingredient", "load_pot"):
                return "fetch_dish"

            # Human is doing something else — robot fetches
            return "fetch_ingredient"

        # --- Default: nothing urgent, pre-fetch dish ---
        return "fetch_dish"

    def _infer_human_task(
        self, human_held: Optional[str], human_pos: Tuple[int, int]
    ) -> Optional[str]:
        """Guess what subtask the human is currently working on.

        Returns None if the human appears idle (standing still without
        an item for too long) — even if near a functional tile.
        """
        # If human has been idle too long, they're not doing anything
        # regardless of position.
        if self._human_idle_ticks >= 18:
            return None

        if human_held in ("onion", "tomato"):
            return "load_pot"
        if human_held == "dish":
            return "pickup_soup"
        if human_held == "soup":
            return "deliver"

        # Empty-handed and recently active: infer from position
        for px, py in self._pot_positions:
            if abs(human_pos[0] - px) + abs(human_pos[1] - py) <= 1:
                return "load_pot"

        return None

    # ------------------------------------------------------------------
    # HBM-guided tie-breaking (used by game loop for exploration)
    # ------------------------------------------------------------------

    def get_explore_weight(self) -> float:
        """Current exploration weight (decays over episodes)."""
        return max(0.1, 0.8 * (0.65 ** self._episode_num))

    # ------------------------------------------------------------------
    # Anti-blocking movement
    # ------------------------------------------------------------------

    def _move_away_from(
        self, away_from: Tuple[int, int], current: Tuple[int, int],
        state: Any,
    ) -> Any:
        """Move robot away from a position. Picks the walkable direction
        that increases distance most."""
        cx, cy = current
        ax, ay = away_from
        other_pos = state.players[1].position
        best_action = ACTION_STAY
        best_dist = 0

        for action, (dx, dy) in [
            (ACTION_NORTH, (0, -1)),
            (ACTION_SOUTH, (0, 1)),
            (ACTION_EAST, (1, 0)),
            (ACTION_WEST, (-1, 0)),
        ]:
            nx, ny = cx + dx, cy + dy
            try:
                if self._mdp.get_terrain_type_at_pos((nx, ny)) != " ":
                    continue
                if (nx, ny) == other_pos:
                    continue
            except Exception:
                continue
            dist = abs(nx - ax) + abs(ny - ay)
            if dist > best_dist:
                best_dist = dist
                best_action = action

        return best_action

    def _navigate_to(
        self, target: Tuple[int, int], current: Tuple[int, int],
        state: Any,
    ) -> Any:
        """Simple greedy navigation toward a target position."""
        if current == target:
            return ACTION_STAY
        cx, cy = current
        tx, ty = target
        other_pos = state.players[1].position

        # Try each direction, preferring the one that reduces distance most
        candidates = []
        for action, (dx, dy) in [
            (ACTION_NORTH, (0, -1)),
            (ACTION_SOUTH, (0, 1)),
            (ACTION_EAST, (1, 0)),
            (ACTION_WEST, (-1, 0)),
        ]:
            nx, ny = cx + dx, cy + dy
            try:
                if self._mdp.get_terrain_type_at_pos((nx, ny)) != " ":
                    continue
                if (nx, ny) == other_pos:
                    continue
            except Exception:
                continue
            dist = abs(nx - tx) + abs(ny - ty)
            candidates.append((dist, action))

        if candidates:
            candidates.sort()
            return candidates[0][1]
        return ACTION_STAY

    def _unstick(self, current: Tuple[int, int], state: Any) -> Any:
        """Try any walkable direction to get unstuck. Avoids the human."""
        cx, cy = current
        other_pos = state.players[1].position

        for action, (dx, dy) in [
            (ACTION_SOUTH, (0, 1)),
            (ACTION_EAST, (1, 0)),
            (ACTION_WEST, (-1, 0)),
            (ACTION_NORTH, (0, -1)),
        ]:
            nx, ny = cx + dx, cy + dy
            try:
                if self._mdp.get_terrain_type_at_pos((nx, ny)) != " ":
                    continue
                if (nx, ny) == other_pos:
                    continue
            except Exception:
                continue
            return action

        return ACTION_STAY

    # Keep _effective_subtask for the game loop's robot-execution path
    def _effective_subtask(self, subtask: str, state: Any) -> str:
        """Map high-level subtask to what the planner should do now."""
        try:
            held = (state.players[0].get_object().name
                    if state.players[0].has_object() else None)
            if held in ("onion", "tomato"):
                return "load_pot"
            if held == "dish":
                return "pickup_soup"
            if held == "soup":
                return "deliver"
            if subtask in ("fetch_ingredient", "load_pot"):
                return "fetch_ingredient"
            if subtask in ("fetch_dish", "pickup_soup", "deliver"):
                return "fetch_dish"
        except Exception:
            pass
        return subtask


# ---------------------------------------------------------------------------
# Player / family database (persisted as JSON)
# ---------------------------------------------------------------------------

class PlayerDatabase:
    """Simple JSON-backed store for player profiles and families."""

    def __init__(self, path: str = "logs/human_play/players.json") -> None:
        self._path = path
        self._data: Dict[str, Any] = {"players": {}, "families": {}}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def list_players(self) -> List[str]:
        return list(self._data["players"].keys())

    def list_families(self) -> List[str]:
        return list(self._data["families"].keys())

    def add_player(self, name: str, family: Optional[str] = None) -> None:
        if name not in self._data["players"]:
            self._data["players"][name] = {
                "family": family,
                "episodes_played": 0,
                "created": datetime.now().isoformat(),
            }
        if family:
            self._add_to_family(name, family)
        self._save()

    def _add_to_family(self, player: str, family: str) -> None:
        if family not in self._data["families"]:
            self._data["families"][family] = {"members": []}
        if player not in self._data["families"][family]["members"]:
            self._data["families"][family]["members"].append(player)
        self._data["players"][player]["family"] = family

    def get_family_members(self, family: str) -> List[str]:
        return self._data.get("families", {}).get(family, {}).get("members", [])

    def increment_episodes(self, player: str) -> None:
        if player in self._data["players"]:
            self._data["players"][player]["episodes_played"] += 1
            self._save()

    def get_player_info(self, player: str) -> Dict[str, Any]:
        return self._data.get("players", {}).get(player, {})

    def delete_player(self, name: str) -> bool:
        """Remove a player and clean up their family membership."""
        if name not in self._data["players"]:
            return False
        family = self._data["players"][name].get("family")
        del self._data["players"][name]
        # Remove from family member list
        if family and family in self._data["families"]:
            members = self._data["families"][family]["members"]
            if name in members:
                members.remove(name)
            # Delete empty families
            if not members:
                del self._data["families"][family]
        self._save()
        return True

    def delete_all(self) -> None:
        """Wipe all players and families."""
        self._data = {"players": {}, "families": {}}
        self._save()


# ---------------------------------------------------------------------------
# Preference logger (JSON lines)
# ---------------------------------------------------------------------------

class PreferenceLogger:
    """Append-only JSONL logger for preference observations and summaries."""

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._obs_path = os.path.join(log_dir, f"observations_{ts}.jsonl")
        self._summary_path = os.path.join(log_dir, f"summary_{ts}.json")
        self._observations: List[Dict[str, Any]] = []

    def log_observation(
        self,
        player: str,
        layout: str,
        episode: int,
        subtask: str,
        task_score: float,
        actor: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "player": player,
            "layout": layout,
            "episode": episode,
            "subtask": subtask,
            "task_score": task_score,
            "actor": actor,
        }
        self._observations.append(entry)
        with open(self._obs_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        summary["timestamp"] = datetime.now().isoformat()
        summary["total_observations"] = len(self._observations)
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {self._summary_path}")
        print(f"Observations saved to {self._obs_path}")


# ---------------------------------------------------------------------------
# Terminal-based player selection (works without pygame)
# ---------------------------------------------------------------------------

def select_player_terminal(db: PlayerDatabase) -> Tuple[str, Optional[str]]:
    """Interactive terminal prompt to select or create a player."""
    players = db.list_players()
    families = db.list_families()

    print("\n" + "=" * 50)
    print("  PLAYER SELECT")
    print("=" * 50)

    if players:
        print("\nExisting players:")
        for i, name in enumerate(players, 1):
            info = db.get_player_info(name)
            fam = info.get("family", "none")
            eps = info.get("episodes_played", 0)
            print(f"  [{i}] {name}  (family={fam}, episodes={eps})")

    print(f"\n  [N] New player")
    print(f"  [Q] Quit\n")

    while True:
        choice = input("Select: ").strip()
        if choice.upper() == "Q":
            sys.exit(0)
        if choice.upper() == "N":
            return _create_player_terminal(db)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(players):
                name = players[idx]
                family = db.get_player_info(name).get("family")
                return name, family
        except ValueError:
            pass
        print("Invalid choice, try again.")


def _create_player_terminal(db: PlayerDatabase) -> Tuple[str, Optional[str]]:
    """Create a new player (and optionally a new family)."""
    name = input("  Player name: ").strip()
    if not name:
        print("  Name cannot be empty.")
        return _create_player_terminal(db)

    families = db.list_families()
    family: Optional[str] = None

    if families:
        print(f"\n  Existing families: {', '.join(families)}")
    print("  Enter family name (or press Enter for none, or type a new name):")
    fam_input = input("  Family: ").strip()
    if fam_input:
        family = fam_input

    db.add_player(name, family)
    print(f"  Created player '{name}'" + (f" in family '{family}'" if family else ""))
    return name, family


# ---------------------------------------------------------------------------
# Pygame-based player select screen (overlay)
# ---------------------------------------------------------------------------

def _pygame_text_input(
    screen: Any, font: Any, small_font: Any,
    prompt: str, hint: str = "",
) -> Optional[str]:
    """Render an in-window text input field. Returns the string or None on Esc."""
    import pygame

    bg_color = (30, 30, 40)
    text_color = (220, 220, 220)
    dim_color = (140, 150, 170)
    cursor_color = (180, 220, 255)
    buffer = ""
    cursor_visible = True
    cursor_timer = 0

    while True:
        screen.fill(bg_color)
        w, h = screen.get_size()

        # Prompt
        prompt_surf = font.render(prompt, True, cursor_color)
        screen.blit(prompt_surf, (w // 2 - prompt_surf.get_width() // 2, h // 3 - 30))

        # Hint
        if hint:
            hint_surf = small_font.render(hint, True, dim_color)
            screen.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, h // 3))

        # Text field box
        field_w = min(400, w - 80)
        field_x = w // 2 - field_w // 2
        field_y = h // 3 + 30
        field_rect = pygame.Rect(field_x, field_y, field_w, 36)
        pygame.draw.rect(screen, (60, 60, 80), field_rect, border_radius=4)
        pygame.draw.rect(screen, (100, 120, 160), field_rect, width=2, border_radius=4)

        # Text + cursor
        display_text = buffer + ("|" if cursor_visible else " ")
        text_surf = font.render(display_text, True, text_color)
        screen.blit(text_surf, (field_x + 8, field_y + 7))

        # Instructions
        inst = small_font.render("Enter to confirm, Esc to cancel", True, dim_color)
        screen.blit(inst, (w // 2 - inst.get_width() // 2, field_y + 50))

        pygame.display.flip()

        # Blink cursor
        cursor_timer += 1
        if cursor_timer % 10 == 0:
            cursor_visible = not cursor_visible

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    stripped = buffer.strip()
                    if stripped:
                        return stripped
                    # Don't accept empty names
                elif event.key == pygame.K_BACKSPACE:
                    buffer = buffer[:-1]
                elif event.unicode and event.unicode.isprintable():
                    buffer += event.unicode

        pygame.time.wait(50)


def _create_player_pygame(
    db: PlayerDatabase, screen: Any, font: Any, small_font: Any
) -> Optional[Tuple[str, Optional[str]]]:
    """In-pygame flow to create a new player with name + optional family."""
    import pygame

    # Step 1: Get player name
    name = _pygame_text_input(screen, font, small_font, "Enter player name:")
    if name is None:
        return None

    # Step 2: Get family (optional)
    families = db.list_families()
    hint = f"Existing: {', '.join(families)}" if families else "Leave blank for none"
    family = _pygame_text_input(
        screen, font, small_font,
        f"Family for {name}?",
        hint=hint,
    )
    # family can be None (Esc) or empty string — both mean no family
    if family is not None:
        family = family.strip() or None

    db.add_player(name, family)
    print(f"  Created player '{name}'" + (f" in family '{family}'" if family else ""))
    return name, family


def _confirm_pygame(
    screen: Any, font: Any, small_font: Any, message: str
) -> bool:
    """Show a Y/N confirmation dialog. Returns True if user presses Y."""
    import pygame

    bg_color = (30, 30, 40)
    while True:
        screen.fill(bg_color)
        w, h = screen.get_size()
        msg_surf = font.render(message, True, (255, 180, 180))
        screen.blit(msg_surf, (w // 2 - msg_surf.get_width() // 2, h // 2 - 20))
        hint = small_font.render("Y = confirm, N / Esc = cancel", True, (140, 150, 170))
        screen.blit(hint, (w // 2 - hint.get_width() // 2, h // 2 + 20))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True
                if event.key in (pygame.K_n, pygame.K_ESCAPE):
                    return False
        pygame.time.wait(50)


def _menu_loop(
    screen: Any, font: Any, small_font: Any,
    title_text: str,
    options: List[str],
    labels: Optional[List[str]] = None,
    hint_lines: Optional[List[str]] = None,
    allow_delete: bool = False,
    deletable_count: int = 0,
) -> Optional[Tuple[int, str]]:
    """Generic menu loop. Returns (index, key) where key is 'enter' or 'delete',
    or None on quit/escape.

    ``deletable_count`` is the number of leading options that can be deleted
    (the D key only works on indices < deletable_count).
    """
    import pygame

    bg_color = (30, 30, 40)
    highlight = (70, 90, 130)
    delete_hl = (130, 60, 60)
    text_color = (220, 220, 220)
    dim_color = (140, 150, 170)
    selected = 0
    if labels is None:
        labels = [f"  {o}" for o in options]
    if hint_lines is None:
        hint_lines = ["Up/Down = select, Enter = confirm, Esc = back"]
        if allow_delete:
            hint_lines[0] += ", D = delete"

    while True:
        screen.fill(bg_color)
        w, h = screen.get_size()

        t_surf = font.render(title_text, True, (180, 220, 255))
        screen.blit(t_surf, (w // 2 - t_surf.get_width() // 2, 30))

        y = 80
        for i, label in enumerate(labels):
            is_danger = options[i] in ("Wipe All Profiles",)
            color = text_color if i == selected else dim_color
            if i == selected and is_danger:
                bg = delete_hl
            elif i == selected:
                bg = highlight
            else:
                bg = bg_color
            rect = pygame.Rect(40, y, w - 80, 30)
            pygame.draw.rect(screen, bg, rect, border_radius=4)
            text_surf = small_font.render(label, True, color)
            screen.blit(text_surf, (48, y + 6))
            y += 36

        for line in hint_lines:
            inst = small_font.render(line, True, dim_color)
            screen.blit(inst, (w // 2 - inst.get_width() // 2, y + 20))
            y += 18

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_d and allow_delete:
                    if selected < deletable_count:
                        return (selected, "delete")
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return (selected, "enter")
                elif event.key == pygame.K_ESCAPE:
                    return None

        pygame.time.wait(50)


def select_player_pygame(
    db: PlayerDatabase, screen: Any, font: Any, small_font: Any,
    on_delete: Optional[Any] = None,
    on_wipe: Optional[Any] = None,
) -> Optional[Tuple[str, Optional[str]]]:
    """Two-level player select: first pick scope (family / individual / new),
    then pick a player within that scope.

    Returns (player_name, family_name) or None to quit.
    """
    import pygame

    while True:
        # ----- Level 1: Choose scope -----
        families = db.list_families()
        solo_players = [
            p for p in db.list_players()
            if not db.get_player_info(p).get("family")
        ]

        options_l1: List[str] = []
        labels_l1: List[str] = []

        for fam in families:
            members = db.get_family_members(fam)
            options_l1.append(f"family:{fam}")
            labels_l1.append(f"  Family: {fam}  ({', '.join(members)})")

        for p in solo_players:
            eps = db.get_player_info(p).get("episodes_played", 0)
            options_l1.append(f"solo:{p}")
            labels_l1.append(f"  {p}  (individual, ep={eps})")

        options_l1 += ["+ New Player", "Wipe All Profiles", "Quit"]
        labels_l1 += ["  + New Player", "  Wipe All Profiles", "  Quit"]

        n_selectable = len(families) + len(solo_players)

        result = _menu_loop(
            screen, font, small_font,
            "Who's playing?",
            options_l1, labels_l1,
            hint_lines=[
                "Up/Down = select, Enter = confirm, D = delete, Esc = quit",
            ],
            allow_delete=True,
            deletable_count=n_selectable,
        )

        if result is None:
            return None

        idx, key = result
        chosen = options_l1[idx]

        # Handle delete at level 1
        if key == "delete":
            if chosen.startswith("family:"):
                fam = chosen.split(":", 1)[1]
                if _confirm_pygame(
                    screen, font, small_font,
                    f"Delete family '{fam}' and all its members?",
                ):
                    for m in list(db.get_family_members(fam)):
                        db.delete_player(m)
                        if on_delete:
                            on_delete(m)
                    print(f"  Deleted family '{fam}'")
            elif chosen.startswith("solo:"):
                name = chosen.split(":", 1)[1]
                if _confirm_pygame(
                    screen, font, small_font,
                    f"Delete '{name}' and their learned preferences?",
                ):
                    db.delete_player(name)
                    if on_delete:
                        on_delete(name)
                    print(f"  Deleted player '{name}'")
            continue  # back to level 1

        # Handle enter on action items
        if chosen == "Quit":
            return None

        if chosen == "Wipe All Profiles":
            if _confirm_pygame(
                screen, font, small_font,
                "Delete ALL players and learned preferences?",
            ):
                db.delete_all()
                if on_wipe:
                    on_wipe()
                print("  Wiped all profiles.")
            continue

        if chosen == "+ New Player":
            created = _create_player_pygame(db, screen, font, small_font)
            if created is not None:
                return created
            continue

        # Solo player selected directly
        if chosen.startswith("solo:"):
            name = chosen.split(":", 1)[1]
            return name, None

        # ----- Level 2: Pick member within a family -----
        if chosen.startswith("family:"):
            fam = chosen.split(":", 1)[1]

            while True:
                members = db.get_family_members(fam)
                if not members:
                    break  # family was emptied, go back

                options_l2: List[str] = []
                labels_l2: List[str] = []
                for m in members:
                    eps = db.get_player_info(m).get("episodes_played", 0)
                    options_l2.append(m)
                    labels_l2.append(f"  {m}  (ep={eps})")

                options_l2 += [f"+ New member of {fam}", "Back"]
                labels_l2 += [f"  + New member of {fam}", "  Back"]

                result2 = _menu_loop(
                    screen, font, small_font,
                    f"Family: {fam}",
                    options_l2, labels_l2,
                    hint_lines=[
                        "Up/Down = select, Enter = play, D = delete, Esc = back",
                    ],
                    allow_delete=True,
                    deletable_count=len(members),
                )

                if result2 is None:
                    break  # back to level 1

                idx2, key2 = result2

                if key2 == "delete" and idx2 < len(members):
                    name = members[idx2]
                    if _confirm_pygame(
                        screen, font, small_font,
                        f"Delete '{name}' from family '{fam}'?",
                    ):
                        db.delete_player(name)
                        if on_delete:
                            on_delete(name)
                        print(f"  Deleted player '{name}'")
                    continue  # refresh level 2

                if key2 == "enter":
                    opt2 = options_l2[idx2]
                    if opt2 == "Back":
                        break
                    if opt2.startswith("+ New member"):
                        # Create player pre-assigned to this family
                        pname = _pygame_text_input(
                            screen, font, small_font,
                            f"New member name for {fam}:",
                        )
                        if pname:
                            db.add_player(pname, fam)
                            print(f"  Added '{pname}' to family '{fam}'")
                            return pname, fam
                        continue
                    # Selected an existing member
                    return opt2, fam


# ---------------------------------------------------------------------------
# Build family preference summary
# ---------------------------------------------------------------------------

def build_preference_summary(
    hbm: Any,
    subtasks: List[str],
    layout: str,
    players: List[str],
    db: PlayerDatabase,
) -> Dict[str, Any]:
    """Build a structured summary of learned preferences."""
    summary: Dict[str, Any] = {
        "layout": layout,
        "population": {},
        "players": {},
        "families": {},
    }

    # Population-level mu
    for s in subtasks:
        mu = hbm.get_mu(s)
        summary["population"][s] = {
            "mu": round(float(mu), 3),
            "tendency": "human" if mu > 0 else "robot",
        }

    # Per-player theta and phi
    for player in players:
        player_data: Dict[str, Any] = {"theta": {}, "phi": {}}
        for s in subtasks:
            theta = hbm.get_theta(player, s)
            phi = hbm.get_phi(player, layout, s)
            player_data["theta"][s] = {
                "value": round(float(theta), 3),
                "prefers": "human" if theta > 0 else "robot",
            }
            player_data["phi"][s] = {
                "value": round(float(phi), 3),
                "prefers": "human" if phi > 0 else "robot",
            }
        info = db.get_player_info(player)
        player_data["family"] = info.get("family")
        player_data["episodes_played"] = info.get("episodes_played", 0)
        summary["players"][player] = player_data

    # Per-family aggregation
    for family in db.list_families():
        members = db.get_family_members(family)
        active = [m for m in members if m in players]
        if not active:
            continue
        family_data: Dict[str, Any] = {"members": active, "average_theta": {}}
        for s in subtasks:
            thetas = [float(hbm.get_theta(m, s)) for m in active]
            avg = sum(thetas) / len(thetas)
            family_data["average_theta"][s] = {
                "value": round(avg, 3),
                "prefers": "human" if avg > 0 else "robot",
            }
        summary["families"][family] = family_data

    # Learned hyperparameters
    summary["learned_sigmas"] = hbm.get_learned_sigmas()

    return summary


def print_preference_summary(summary: Dict[str, Any], subtasks: List[str]) -> None:
    """Pretty-print the preference summary to terminal."""
    print("\n" + "=" * 70)
    print("  FAMILY PREFERENCE SUMMARY")
    print("=" * 70)

    print(f"\n  Layout: {summary['layout']}")

    # Population
    print(f"\n  --- Population-level (mu) ---")
    for s in subtasks:
        data = summary["population"][s]
        bar = "+" * int(abs(data["mu"]) * 5)
        direction = "Human" if data["tendency"] == "human" else "Robot"
        print(f"    {s:20s}  mu={data['mu']:+.2f}  -> {direction} {bar}")

    # Per player
    for player, pdata in summary["players"].items():
        fam_str = f" ({pdata['family']})" if pdata.get("family") else ""
        print(f"\n  --- {player}{fam_str}  [episodes: {pdata['episodes_played']}] ---")
        for s in subtasks:
            theta = pdata["theta"][s]
            phi = pdata["phi"][s]
            print(f"    {s:20s}  theta={theta['value']:+.2f}  "
                  f"phi={phi['value']:+.2f}  -> {phi['prefers'].title()}")

    # Families
    for family, fdata in summary.get("families", {}).items():
        print(f"\n  --- Family '{family}' ({', '.join(fdata['members'])}) ---")
        for s in subtasks:
            avg = fdata["average_theta"][s]
            print(f"    {s:20s}  avg_theta={avg['value']:+.2f}  -> {avg['prefers'].title()}")

    sigmas = summary.get("learned_sigmas", {})
    if sigmas:
        print(f"\n  Learned hyperparameters:")
        for k, v in sigmas.items():
            print(f"    {k}: {v:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Human-playable Overcooked with preference learning")
    p.add_argument("--layout", default="CrampedRoom",
                   help="Single layout, or 'rotate' to cycle through all")
    p.add_argument("--episodes", type=int, default=3,
                   help="Episodes per layout per player session")
    p.add_argument("--fps", type=int, default=6,
                   help="Game ticks per second (lower = easier)")
    p.add_argument("--episode-length", type=int, default=400, dest="episode_length",
                   help="Episode length in game ticks (default 400, ~67s at 6fps)")
    p.add_argument("--log-dir", default="logs/human_play/", dest="log_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-robot-planner", action="store_true", dest="no_robot_planner",
                   help="Robot stays idle (for testing human controls)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Deferred imports
    # ------------------------------------------------------------------
    import pygame
    from multitask_personalization.envs.overcooked.layouts import (
        CORE_SUBTASKS, get_layout,
    )
    from multitask_personalization.envs.overcooked.overcooked_hbm import (
        OvercookedPreferenceModel,
    )
    from multitask_personalization.envs.overcooked.overcooked_planner import (
        build_planner,
    )
    from multitask_personalization.envs.overcooked.overcooked_visualizer import (
        OvercookedVisualizer,
    )

    # ------------------------------------------------------------------
    # Layout rotation setup
    # ------------------------------------------------------------------
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _OCEnv

    ALL_LAYOUT_NAMES = [
        "CrampedRoom", "AsymmetricAdvantages", "CoordinationRing",
        "ForcedCoordination", "CounterCircuit",
    ]

    if args.layout == "rotate":
        layout_rotation = list(ALL_LAYOUT_NAMES)
    else:
        layout_rotation = [args.layout]

    # Validate all layouts load
    for ln in layout_rotation:
        get_layout(ln)

    # Union of all subtasks across layouts (for the shared HBM)
    all_subtasks_set: List[str] = []
    all_layout_names: List[str] = []
    for ln in layout_rotation:
        spec = get_layout(ln)
        all_layout_names.append(spec.name)
        for s in spec.subtasks:
            if s not in all_subtasks_set:
                all_subtasks_set.append(s)

    # ------------------------------------------------------------------
    # HBM (shared across all players and layouts)
    # ------------------------------------------------------------------
    hbm_save_dir = Path(args.log_dir) / "hbm_state"
    hbm_save_dir.mkdir(parents=True, exist_ok=True)

    hbm = OvercookedPreferenceModel(
        subtasks=all_subtasks_set,
        layouts=all_layout_names,
    )

    if (hbm_save_dir / "overcooked_hbm.pkl").exists():
        hbm.load(hbm_save_dir)
        print("Loaded saved HBM preferences from previous sessions.")

    def _save_hbm() -> None:
        hbm.save(hbm_save_dir)

    def _reset_hbm() -> None:
        """Rebuild the HBM from scratch (used on wipe all)."""
        nonlocal hbm
        hbm = OvercookedPreferenceModel(
            subtasks=all_subtasks_set,
            layouts=all_layout_names,
        )
        _save_hbm()
        print("  HBM reset to fresh state.")

    def _on_delete_player(name: str) -> None:
        _save_hbm()
        print(f"  HBM saved. '{name}' preferences will be ignored on next load.")

    # Per-layout objects — rebuilt when switching layouts
    def _build_layout(layout_name: str) -> Dict[str, Any]:
        """Build all per-layout objects (mdp, env, planner, viz, robot)."""
        spec = get_layout(layout_name)
        m = OvercookedGridworld.from_layout_name(spec.layout_name)
        env = _OCEnv.from_mdp(m, horizon=args.episode_length)
        p = build_planner(m)
        if p is None:
            print(f"WARNING: Could not build planner for {layout_name}, skipping.")
            return {}
        d = SubtaskDetector(m)
        v = OvercookedVisualizer(tile_size=75, mdp=m)
        r = RobotController(
            mdp=m, planner=p, hbm=hbm,
            subtasks=list(spec.subtasks), layout_name=spec.name,
        )
        return {
            "spec": spec, "mdp": m, "oc_env": env, "planner": p,
            "detector": d, "viz": v, "robot": r,
            "subtasks": list(spec.subtasks),
        }

    # Build the first layout to verify everything works
    first_layout = _build_layout(layout_rotation[0])
    if not first_layout:
        print("ERROR: Could not build first layout.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Player database + logger
    # ------------------------------------------------------------------
    db = PlayerDatabase(path=os.path.join(args.log_dir, "players.json"))
    logger = PreferenceLogger(args.log_dir)

    # ------------------------------------------------------------------
    # Pygame init + visualizer
    # ------------------------------------------------------------------
    pygame.init()
    pygame.display.set_caption("Overcooked — Human Play")
    # Start with a small window; show_composite will resize.
    screen = pygame.display.set_mode((800, 500))
    viz = OvercookedVisualizer(tile_size=75, mdp=mdp)
    if not viz.is_available:
        print("ERROR: StateVisualizer not available.")
        sys.exit(1)

    font = pygame.font.SysFont("monospace", 16, bold=True)
    small_font = pygame.font.SysFont("monospace", 13)

    # Pygame key -> overcooked action mapping
    key_map = {
        pygame.K_UP: ACTION_NORTH,
        pygame.K_DOWN: ACTION_SOUTH,
        pygame.K_LEFT: ACTION_WEST,
        pygame.K_RIGHT: ACTION_EAST,
        pygame.K_SPACE: ACTION_INTERACT,
        pygame.K_s: ACTION_STAY,
    }

    # Short display names for subtasks
    subtask_short = {
        "fetch_ingredient": "Fetch Ingr.",
        "load_pot": "Load Pot",
        "fetch_dish": "Fetch Dish",
        "pickup_soup": "Pickup Soup",
        "deliver": "Deliver",
    }

    # Track all players who played this session (for final summary)
    session_players: List[str] = []
    all_observations: List[Dict[str, Any]] = []

    tick_ms = max(1, 1000 // args.fps)
    running = True

    # ------------------------------------------------------------------
    # Outer loop: player selection -> episodes -> repeat
    # ------------------------------------------------------------------
    while running:
        # --- Player select ---
        result = select_player_pygame(
            db, screen, font, small_font,
            on_delete=_on_delete_player, on_wipe=_reset_hbm,
        )
        if result is None:
            break
        player_name, family_name = result

        # Register player for all layouts in the HBM
        hbm.register_human(player_name)
        for ln in all_layout_names:
            hbm.register_layout(player_name, ln)
        if player_name not in session_players:
            session_players.append(player_name)

        n_layouts = len(layout_rotation)
        total_episodes = args.episodes * n_layouts
        print(f"\n>>> {player_name} is playing!")
        print(f"    {n_layouts} layout(s), {args.episodes} episodes each"
              f" = {total_episodes} total episodes")
        print("    Controls: Arrows=Move  Space=Interact  P=Pause  Q=Quit")

        global_ep = 0  # episode counter across all layouts

        # --- Layout rotation loop ---
        for layout_idx, layout_name in enumerate(layout_rotation):
            if not running:
                break

            # Build per-layout objects
            lo = _build_layout(layout_name)
            if not lo:
                print(f"  Skipping {layout_name} (could not build)")
                continue

            layout_spec = lo["spec"]
            oc_env = lo["oc_env"]
            planner_l = lo["planner"]
            detector = lo["detector"]
            viz = lo["viz"]
            robot = lo["robot"]
            subtasks = lo["subtasks"]

            if not viz.is_available:
                print(f"  Skipping {layout_name} (visualizer unavailable)")
                continue

            robot.set_player(player_name)

            print(f"\n  --- Layout {layout_idx + 1}/{n_layouts}:"
                  f" {layout_spec.name} ---")

            # --- Episode loop for this layout ---
            for ep in range(args.episodes):
                if not running:
                    break

                oc_env.reset()
                state = oc_env.state
                game_done = False
                paused = False

                ep_subtask_counts: Dict[str, int] = {s: 0 for s in subtasks}
                robot.new_episode(global_ep)
                global_ep += 1

                print(f"\n  Episode {ep + 1}/{args.episodes}"
                      f"  ({layout_spec.name})")

                while not game_done:
                    human_action = None

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            game_done = True
                            break
                        if event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                                running = False
                                game_done = True
                                break
                            if event.key == pygame.K_p:
                                paused = not paused
                                continue
                            if event.key in key_map:
                                human_action = key_map[event.key]

                    if not running:
                        break

                    if paused:
                        pause_surf = font.render(
                            "PAUSED (P to resume)", True, (255, 220, 100)
                        )
                        sw, sh = screen.get_size()
                        screen.blit(
                            pause_surf,
                            (sw // 2 - pause_surf.get_width() // 2,
                             sh // 2),
                        )
                        pygame.display.flip()
                        pygame.time.wait(100)
                        continue

                    if human_action is None:
                        human_action = ACTION_STAY

                    if args.no_robot_planner:
                        robot_action = ACTION_STAY
                    else:
                        robot_action = robot.step(state)

                    state_before = state
                    sparse_r = 0.0

                    try:
                        _, sparse_r, done, _ = oc_env.step(
                            (robot_action, human_action)
                        )
                    except Exception as e:
                        print(f"  [env error] {e}")
                        break

                    state = oc_env.state
                    game_done = done

                    human_completed = detector.detect(
                        state_before, state, human_action, human_idx=1
                    )
                    robot_completed = detector.detect(
                        state_before, state, robot_action, human_idx=0
                    )

                    if human_completed is not None:
                        ep_subtask_counts[human_completed] = (
                            ep_subtask_counts.get(human_completed, 0) + 1
                        )
                        hbm.observe(
                            player_name, layout_spec.name,
                            human_completed, "human", 1.0,
                        )
                        logger.log_observation(
                            player_name, layout_spec.name, ep + 1,
                            human_completed, 1.0, "human",
                        )
                        prep_tasks = {"fetch_ingredient", "load_pot"}
                        service_tasks = {"fetch_dish", "pickup_soup", "deliver"}
                        if human_completed in prep_tasks:
                            for neg_task in service_tasks:
                                hbm.observe(
                                    player_name, layout_spec.name,
                                    neg_task, "robot", -0.3,
                                )
                        elif human_completed in service_tasks:
                            for neg_task in prep_tasks:
                                hbm.observe(
                                    player_name, layout_spec.name,
                                    neg_task, "robot", -0.3,
                                )
                        short = subtask_short.get(
                            human_completed, human_completed
                        )
                        print(f"    Human: {short}")

                    if robot_completed is not None:
                        ep_subtask_counts[robot_completed] = (
                            ep_subtask_counts.get(robot_completed, 0) + 1
                        )
                        hbm.observe(
                            player_name, layout_spec.name,
                            robot_completed, "robot", -1.0,
                        )
                        logger.log_observation(
                            player_name, layout_spec.name, ep + 1,
                            robot_completed, -1.0, "robot",
                        )
                        short = subtask_short.get(
                            robot_completed, robot_completed
                        )
                        print(f"    Robot: {short}")

                    # --- Render ---
                    robot_status = subtask_short.get(
                        robot.current_subtask or "", "idle"
                    )
                    status_lines = [
                        f"Player: {player_name}    "
                        f"Ep {ep+1}/{args.episodes}    "
                        f"Layout {layout_idx+1}/{n_layouts}",
                        f"Layout: {layout_spec.name}    Robot: {robot_status}",
                        f"Arrows=Move  Space=Interact  P=Pause  Q=Quit",
                    ]
                    if sparse_r > 0:
                        status_lines.append(
                            f"*** SOUP DELIVERED! +{sparse_r:.0f} ***"
                        )

                    table_header = ["Subtask", "Phi", "Psi", "Unc.", "Pref"]
                    table_rows = []
                    for s in subtasks:
                        short = subtask_short.get(s, s[:11])
                        phi = hbm.get_phi(
                            player_name, layout_spec.name, s
                        )
                        psi = hbm.get_running_psi(player_name, s)
                        entropy = hbm.get_phi_entropy(
                            player_name, layout_spec.name, s
                        )
                        p_human = 1.0 / (1.0 + np.exp(-(phi + psi)))
                        pref_str = (
                            f"H {p_human:.0%}" if p_human >= 0.5
                            else f"R {1-p_human:.0%}"
                        )
                        table_rows.append([
                            short, f"{phi:+.2f}", f"{psi:+.2f}",
                            f"{entropy:.2f}", pref_str,
                        ])

                    table_footer = [
                        f"Family: {family_name or 'none'}",
                    ]

                    surface = viz.render_composite(
                        state, status_lines, table_rows,
                        table_header, table_footer=table_footer,
                    )
                    if surface is not None:
                        sw, sh = surface.get_size()
                        if screen.get_size() != (sw, sh):
                            screen = pygame.display.set_mode((sw, sh))
                        screen.blit(surface, (0, 0))
                        pygame.display.flip()

                    pygame.time.wait(tick_ms)

                # --- End of episode ---
                hbm.end_episode(player_name)
                _save_hbm()
                db.increment_episodes(player_name)

                total = sum(ep_subtask_counts.values())
                print(f"  Episode done.  Total subtasks: {total}")
                for s, count in ep_subtask_counts.items():
                    if count > 0:
                        short = subtask_short.get(s, s)
                        phi = hbm.get_phi(
                            player_name, layout_spec.name, s
                        )
                        print(f"    {short}: {count}x  (phi={phi:+.2f})")

        if not running:
            break

        print(f"\n>>> {player_name}'s session complete!")
        pygame.time.wait(500)

    # ------------------------------------------------------------------
    # Final summary (use first layout for display)
    # ------------------------------------------------------------------
    if session_players:
        try:
            hbm.flush_theta_mu()
            # Print summary per layout
            for ln in layout_rotation:
                spec = get_layout(ln)
                summary = build_preference_summary(
                    hbm, list(spec.subtasks), spec.name,
                    session_players, db,
                )
                print_preference_summary(summary, list(spec.subtasks))
            # Save summary for the last layout
            logger.write_summary(summary)
        except Exception as e:
            print(f"\nCould not generate summary: {e}")
            print("(This can happen if no episodes were completed.)")

    pygame.quit()
    print("Done!")


if __name__ == "__main__":
    main()
