"""
overcooked_visualizer.py — Rendering utilities for the Overcooked environment.

This module is **fully detached** from the HBM, CSP, and satisfaction
subsystems.  It knows nothing about phi, psi, or preference learning.
Its only job is to turn overcooked_ai game states into images or pygame
windows.

Usage
-----
Render a single frame to a file::

    from multitask_personalization.envs.overcooked.overcooked_visualizer import (
        OvercookedVisualizer,
    )
    viz = OvercookedVisualizer(tile_size=75)
    viz.save_frame(state, path="frame.png", hud={"subtask": "load_pot"})

Render a live episode in a pygame window::

    viz = OvercookedVisualizer(tile_size=75)
    for state, joint_action in episode:
        viz.show_frame(state, hud={"last_action": str(joint_action)})

Render a full trajectory as a sequence of PNG frames::

    viz.save_trajectory(states, actions, out_dir="frames/")

If overcooked_ai_py is not installed all methods degrade gracefully
(``is_available`` returns False; render calls are no-ops).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


class OvercookedVisualizer:
    """
    Thin wrapper around overcooked_ai_py's ``StateVisualizer``.

    Parameters
    ----------
    tile_size:
        Pixel size of each grid tile.  Default 75 matches the demo.
    """

    def __init__(self, tile_size: int = 75, mdp: Any = None) -> None:
        self._tile_size = tile_size
        self._grid = mdp.terrain_mtx if mdp is not None else None
        self._viz: Any = None
        self._window: Any = None   # pygame display surface, created lazily
        self._available = self._try_init()

    @property
    def is_available(self) -> bool:
        """True if overcooked_ai_py's StateVisualizer was successfully loaded."""
        return self._available

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def save_frame(
        self,
        state: Any,
        path: str,
        hud: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Render *state* to a PNG at *path*.

        Parameters
        ----------
        state:
            An ``OvercookedState`` from overcooked_ai.
        path:
            Output file path (must end in .png).
        hud:
            Optional dict of label→value to overlay as text on the image.

        Returns
        -------
        True if the image was saved successfully, False otherwise.
        """
        if not self._available or self._viz is None:
            return False
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._viz.display_rendered_state(
                state,
                img_path=path,
                ipython_display=False,
                window_display=False,
                hud_data=hud if hud else None,
            )
            return True
        except Exception:
            return False

    def show_frame(
        self,
        state: Any,
        hud: Optional[Dict[str, Any]] = None,
        pause_ms: int = 100,
    ) -> bool:
        """
        Display *state* in a live pygame window (blocking for *pause_ms* ms).

        Parameters
        ----------
        state:
            An ``OvercookedState`` from overcooked_ai.
        hud:
            Optional dict to overlay as text.
        pause_ms:
            How long to keep the frame visible (milliseconds).

        Returns
        -------
        True if the frame was displayed.
        """
        if not self._available or self._viz is None:
            return False
        try:
            import pygame
            # Render to a surface (no window_display — that call blocks)
            surface = self._viz.render_state(
                state, self._grid, hud if hud else None
            )
            # Create or resize the window to match the surface
            w, h = surface.get_size()
            if self._window is None or self._window.get_size() != (w, h):
                self._window = pygame.display.set_mode((w, h))
                pygame.display.set_caption("Overcooked")
            self._window.blit(surface, (0, 0))
            pygame.display.flip()
            # Pump events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            pygame.time.wait(pause_ms)
            return True
        except Exception:
            return False

    def render_to_array(
        self,
        state: Any,
        hud: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Render *state* to a numpy array (H, W, 3) suitable for video encoding.

        Returns None if rendering is unavailable.
        """
        if not self._available or self._viz is None:
            return None
        try:
            import pygame
            import numpy as np
            surface = self._viz.render_state(
                state, self._grid, hud if hud else None
            )
            arr = pygame.surfarray.array3d(surface)
            return np.transpose(arr, (1, 0, 2))
        except Exception:
            return None

    def render_composite(
        self,
        state: Any,
        status_lines: Optional[List[str]] = None,
        table_rows: Optional[List[List[str]]] = None,
        table_header: Optional[List[str]] = None,
        table_footer: Optional[List[str]] = None,
    ) -> Any:
        """
        Render game state + status bar (left) + preference table (right).

        Returns a pygame surface (or None). Layout:

            ┌──────────────────┬─────────────────────┐
            │  Status lines    │  Preference Table   │
            │  (left, top)     │  (right, top)       │
            │                  │                     │
            │  Game render     │  Header row         │
            │  (left, below    │  Data rows...       │
            │   status)        │                     │
            │                  │  Config footer      │
            └──────────────────┴─────────────────────┘

        Parameters
        ----------
        state: overcooked_ai state
        status_lines: list of strings for the left status area
        table_rows: list of [col1, col2, ...] for the preference table
        table_header: header row for the table
        table_footer: list of strings shown below the table (e.g. config info)
        """
        if not self._available or self._viz is None:
            return None
        try:
            import pygame

            # Render game without HUD
            game_surface = self._viz.render_state(state, self._grid, None)
            gw, gh = game_surface.get_size()

            # Font setup
            pygame.font.init()
            font = pygame.font.SysFont("monospace", 14)
            small_font = pygame.font.SysFont("monospace", 12)
            line_h = 18
            small_line_h = 16

            # Compute table panel width
            table_w = 340
            if table_rows and table_header:
                # Estimate width from content
                max_row_text = max(
                    (len("  ".join(row)) for row in [table_header] + table_rows),
                    default=30,
                )
                table_w = max(table_w, max_row_text * 8 + 20)

            # Compute status area height and width
            n_status = len(status_lines) if status_lines else 0
            status_h = max(n_status * line_h + 8, 0)

            # Ensure the game side is wide enough for status text
            status_w = 0
            if status_lines:
                status_w = max(len(line) * 8 + 16 for line in status_lines)

            # Total canvas size — expand game side if status text is wider
            left_w = max(gw, status_w)
            total_w = left_w + table_w
            total_h = max(gh + status_h, 300)

            canvas = pygame.Surface((total_w, total_h))
            canvas.fill((40, 40, 50))  # dark background

            # --- Draw status lines (top-left, above game) ---
            if status_lines:
                y = 4
                for line in status_lines:
                    text_surf = font.render(line, True, (220, 220, 220))
                    canvas.blit(text_surf, (8, y))
                    y += line_h

            # --- Draw game (left, below status) ---
            canvas.blit(game_surface, (0, status_h))

            # --- Draw preference table (right panel) ---
            if table_header and table_rows:
                panel_x = left_w + 4
                y = 4

                # Table title
                title_surf = font.render("Learned vs True", True, (180, 220, 255))
                canvas.blit(title_surf, (panel_x + 4, y))
                y += line_h + 4

                # Compute column widths from content
                all_rows = [table_header] + table_rows
                n_cols = len(table_header)
                col_widths = [0] * n_cols
                for row in all_rows:
                    for c in range(min(n_cols, len(row))):
                        col_widths[c] = max(col_widths[c], len(row[c]) * 8 + 8)

                # Header
                x = panel_x + 4
                for c, hdr in enumerate(table_header):
                    color = (160, 180, 200)
                    text_surf = small_font.render(hdr, True, color)
                    canvas.blit(text_surf, (x, y))
                    x += col_widths[c]
                y += small_line_h

                # Separator line
                pygame.draw.line(
                    canvas, (100, 100, 120),
                    (panel_x + 4, y), (panel_x + sum(col_widths) + 4, y)
                )
                y += 4

                # Data rows
                for row in table_rows:
                    x = panel_x + 4
                    for c, cell in enumerate(row):
                        if c == 0:
                            color = (220, 220, 220)
                        else:
                            color = (200, 200, 200)
                        text_surf = small_font.render(cell, True, color)
                        canvas.blit(text_surf, (x, y))
                        x += col_widths[c]
                    y += small_line_h

            # --- Config footer (below table, right panel) ---
            if table_footer:
                y_footer = max(y + 12, status_h + gh - len(table_footer) * small_line_h - 8)
                panel_x = left_w + 4
                pygame.draw.line(
                    canvas, (80, 80, 100),
                    (panel_x + 4, y_footer - 4),
                    (panel_x + table_w - 8, y_footer - 4),
                )
                for line in table_footer:
                    text_surf = small_font.render(line, True, (140, 150, 170))
                    canvas.blit(text_surf, (panel_x + 4, y_footer))
                    y_footer += small_line_h

            return canvas
        except Exception:
            return None

    def show_composite(
        self,
        state: Any,
        status_lines: Optional[List[str]] = None,
        table_rows: Optional[List[List[str]]] = None,
        table_header: Optional[List[str]] = None,
        table_footer: Optional[List[str]] = None,
        pause_ms: int = 100,
    ) -> bool:
        """Display composite frame in a live pygame window."""
        surface = self.render_composite(state, status_lines, table_rows, table_header, table_footer)
        if surface is None:
            return False
        try:
            import pygame
            w, h = surface.get_size()
            if self._window is None or self._window.get_size() != (w, h):
                self._window = pygame.display.set_mode((w, h))
                pygame.display.set_caption("Overcooked — Preference Learning")
            self._window.blit(surface, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            pygame.time.wait(pause_ms)
            return True
        except Exception:
            return False

    def composite_to_array(
        self,
        state: Any,
        status_lines: Optional[List[str]] = None,
        table_rows: Optional[List[List[str]]] = None,
        table_header: Optional[List[str]] = None,
        table_footer: Optional[List[str]] = None,
    ) -> Any:
        """Render composite to numpy array (H, W, 3) for video encoding."""
        surface = self.render_composite(state, status_lines, table_rows, table_header, table_footer)
        if surface is None:
            return None
        try:
            import pygame
            import numpy as np
            arr = pygame.surfarray.array3d(surface)
            return np.transpose(arr, (1, 0, 2))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Trajectory rendering
    # ------------------------------------------------------------------

    def save_trajectory(
        self,
        states: Sequence[Any],
        actions: Sequence[Any],
        out_dir: str,
        prefix: str = "frame_",
    ) -> List[str]:
        """
        Save each state in a trajectory as a numbered PNG frame.

        Parameters
        ----------
        states:
            Sequence of overcooked_ai states (one per timestep).
        actions:
            Sequence of joint actions, same length as *states*.
        out_dir:
            Directory to write frames into (created if needed).
        prefix:
            Filename prefix for each frame.

        Returns
        -------
        List of saved file paths (empty if unavailable).
        """
        if not self._available:
            return []

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i, (state, action) in enumerate(zip(states, actions)):
            hud = {"step": i, "action": str(action)}
            path = os.path.join(out_dir, f"{prefix}{i:04d}.png")
            if self.save_frame(state, path=path, hud=hud):
                paths.append(path)
        return paths

    def show_trajectory(
        self,
        states: Sequence[Any],
        actions: Sequence[Any],
        fps: int = 5,
    ) -> bool:
        """
        Play a trajectory in a live pygame window at *fps* frames per second.

        Parameters
        ----------
        states:
            Sequence of overcooked_ai states.
        actions:
            Sequence of joint actions.
        fps:
            Playback speed.

        Returns
        -------
        True if the trajectory was displayed without errors.
        """
        if not self._available:
            return False
        pause_ms = max(1, 1000 // fps)
        ok = True
        for state, action in zip(states, actions):
            hud = {"action": str(action)}
            if not self.show_frame(state, hud=hud, pause_ms=pause_ms):
                ok = False
        return ok

    def render_trajectory_to_file(
        self,
        states: Sequence[Any],
        actions: Sequence[Any],
        path: str,
    ) -> bool:
        """
        Use overcooked_ai's built-in ``display_rendered_trajectory`` to render
        a full trajectory to a single file (gif or video if pygame is available).

        Parameters
        ----------
        states, actions:
            Episode trajectory.
        path:
            Output file path.

        Returns
        -------
        True on success.
        """
        if not self._available or self._viz is None:
            return False
        try:
            trajectory = {
                "ep_states": [list(states)],
                "ep_actions": [list(actions)],
                "ep_rewards": [[0.0] * len(states)],
                "ep_dones": [[False] * (len(states) - 1) + [True]],
                "ep_infos": [[{}] * len(states)],
            }
            self._viz.display_rendered_trajectory(
                trajectory,
                img_directory_path=os.path.dirname(path) or ".",
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_init(self) -> bool:
        """Attempt to create a StateVisualizer. Returns True on success."""
        try:
            from overcooked_ai_py.visualization.state_visualizer import (
                StateVisualizer,
            )
            kwargs: Dict[str, Any] = {"tile_size": self._tile_size}
            if self._grid is not None:
                kwargs["grid"] = self._grid
            self._viz = StateVisualizer(**kwargs)
            return True
        except ImportError:
            return False
        except Exception:
            return False



# ---------------------------------------------------------------------------
# Convenience runner for quick demos
# ---------------------------------------------------------------------------

def run_demo(
    layout_name: str = "cramped_room",
    n_steps: int = 50,
    tile_size: int = 75,
    save_dir: Optional[str] = None,
) -> None:
    """
    Run a short demo episode with two greedy agents and optionally save frames.

    Both agents use overcooked_ai's ``GreedyHumanModel``.  This demo is
    intentionally minimal — it has no HBM, no CSP, and no preference learning.

    Parameters
    ----------
    layout_name:
        Any layout supported by overcooked_ai (e.g. "cramped_room").
    n_steps:
        Number of primitive-action steps to simulate.
    tile_size:
        Tile pixel size passed to the visualizer.
    save_dir:
        If provided, frames are saved to this directory as PNGs.
        If None, frames are shown in a live pygame window.
    """
    try:
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _OCEnv
        from overcooked_ai_py.agents.agent import (
            AgentPair,
            GreedyHumanModel,
        )
        from overcooked_ai_py.planning.planners import (
            MediumLevelActionManager,
            NO_COUNTERS_PARAMS,
        )
    except ImportError as exc:
        print(f"overcooked_ai_py not available: {exc}")
        return

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = _OCEnv.from_mdp(mdp, horizon=n_steps)
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, NO_COUNTERS_PARAMS, force_compute=False
    )
    agent0 = GreedyHumanModel(mlam)
    agent1 = GreedyHumanModel(mlam)
    agent_pair = AgentPair(agent0, agent1)
    agent_pair.set_mdp(mdp)

    viz = OvercookedVisualizer(tile_size=tile_size, mdp=mdp)
    if not viz.is_available:
        print("StateVisualizer not available — cannot render.")
        return

    obs = env.reset()
    state = env.state
    states: List[Any] = [state]
    joint_actions: List[Any] = []
    total_score = 0

    for step in range(n_steps):
        # joint_action() returns ((action0, info0), (action1, info1))
        (action0, _), (action1, _) = agent_pair.joint_action(state)
        joint_action = (action0, action1)
        obs, reward, done, info = env.step(joint_action)
        total_score += reward
        state = env.state
        states.append(state)
        joint_actions.append(joint_action)

        # Describe what each agent is currently holding
        p0_held = state.players[0].held_object.name if state.players[0].held_object else "empty"
        p1_held = state.players[1].held_object.name if state.players[1].held_object else "empty"
        shaped = info.get("shaped_r_by_agent", [0, 0])

        hud = {
            "step": f"{step+1}/{n_steps}",
            "score": int(total_score),
            "robot(blue)": p0_held,
            "human(green)": p1_held,
            "shaped_r": f"{shaped[0]:.0f}/{shaped[1]:.0f}",
        }

        if save_dir is not None:
            viz.save_frame(state, path=os.path.join(save_dir, f"frame_{step:04d}.png"), hud=hud)
        else:
            if not viz.show_frame(state, hud=hud, pause_ms=150):
                break  # window was closed

        if done:
            break

    if save_dir:
        print(f"Saved {len(states)-1} frames to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Overcooked demo visualizer")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--tile-size", type=int, default=75)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()
    run_demo(
        layout_name=args.layout,
        n_steps=args.steps,
        tile_size=args.tile_size,
        save_dir=args.save_dir,
    )
