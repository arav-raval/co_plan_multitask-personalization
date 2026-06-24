"""FastAPI server for the human-playable Overcooked web demo.

Architecture:
  - GET /              → family select page
  - POST /start        → create session, redirect to /play/<sid>
  - GET /play/<sid>    → game page (HTML/JS)
  - WS  /ws/<sid>      → real-time gameplay (input + frame stream)
  - GET /thanks        → end-of-session thank you page

Two HBMs are maintained:
  - "Raval-Trivedi" — your actual family
  - "General Population Sample" — everyone else
"""

from __future__ import annotations

# Headless pygame rendering — must be set BEFORE importing pygame
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import asyncio
import base64
import io
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame
pygame.init()

from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .game_logic import (
    ACTION_EAST,
    ACTION_INTERACT,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_STAY,
    ACTION_WEST,
    CUSTOM_LAYOUTS,
    PlayerDatabase,
    PreferenceLogger,
    RobotController,
    SubtaskDetector,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Two HBM groups
HBM_GROUPS = ["Raval-Trivedi", "General Population Sample"]

# Layout rotation: 1 episode per layout
LAYOUT_ROTATION = ["CrampedRoom"] + list(CUSTOM_LAYOUTS.keys())
EPISODE_LENGTH_TICKS = 270  # ~45 seconds at 6 fps
TICK_INTERVAL_SEC = 1.0 / 6.0  # 6 fps server-side game loop

# Storage
LOG_DIR = Path("logs/web")
LOG_DIR.mkdir(parents=True, exist_ok=True)
HBM_DIR = LOG_DIR / "hbm_state"
HBM_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR = LOG_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

PLAYERS_DB = PlayerDatabase(path=str(LOG_DIR / "players.json"))


# ---------------------------------------------------------------------------
# Per-family HBM management (lazy-loaded)
# ---------------------------------------------------------------------------

def _hbm_path(group: str) -> Path:
    safe = group.replace(" ", "_").replace("/", "_")
    return HBM_DIR / f"{safe}.pkl"


def _load_or_create_hbm(group: str):
    """Load HBM for a group, or create one if missing."""
    from multitask_personalization.envs.overcooked.layouts import CORE_SUBTASKS
    from multitask_personalization.envs.overcooked.overcooked_hbm import (
        OvercookedPreferenceModel,
    )

    hbm = OvercookedPreferenceModel(
        subtasks=list(CORE_SUBTASKS),
        layouts=LAYOUT_ROTATION,
        sigma_session=0.1,
    )
    pkl = _hbm_path(group)
    if pkl.exists():
        hbm.load(HBM_DIR / "tmp")  # ensure dir exists; load by parent
        # Actually we need a custom load — the pkl is at HBM_DIR / safe.pkl
        # not HBM_DIR / "overcooked_hbm.pkl". Override:
        import pickle
        try:
            with open(pkl, "rb") as f:
                state = pickle.load(f)
            # Re-apply state manually
            import torch
            for h, ldict in state.get("phi_m", {}).items():
                if h not in hbm._phi_m:
                    hbm.register_human(h)
                for L, sdict in ldict.items():
                    if L not in hbm._phi_m.get(h, {}):
                        hbm.register_layout(h, L)
                    for s, v in sdict.items():
                        if s in hbm._phi_m.get(h, {}).get(L, {}):
                            hbm._phi_m[h][L][s] = torch.tensor(
                                v, dtype=torch.float32, requires_grad=True
                            )
            for h, ldict in state.get("phi_logv", {}).items():
                for L, sdict in ldict.items():
                    for s, v in sdict.items():
                        if s in hbm._phi_logv.get(h, {}).get(L, {}):
                            hbm._phi_logv[h][L][s] = torch.tensor(
                                v, dtype=torch.float32, requires_grad=True
                            )
            for h, sdict in state.get("theta_m", {}).items():
                for s, v in sdict.items():
                    if s in hbm._theta_m.get(h, {}):
                        hbm._theta_m[h][s] = torch.tensor(
                            v, dtype=torch.float32, requires_grad=True
                        )
            for h, sdict in state.get("theta_logv", {}).items():
                for s, v in sdict.items():
                    if s in hbm._theta_logv.get(h, {}):
                        hbm._theta_logv[h][s] = torch.tensor(
                            v, dtype=torch.float32, requires_grad=True
                        )
            for s, v in state.get("mu_mean", {}).items():
                if s in hbm.mu_mean:
                    hbm.mu_mean[s] = float(v)
            for s, v in state.get("mu_var", {}).items():
                if s in hbm.mu_var:
                    hbm.mu_var[s] = float(v)
        except Exception as e:
            print(f"  [HBM load warning for {group}] {e}")
    return hbm


def _save_hbm(group: str, hbm) -> None:
    """Save HBM directly to a group-specific .pkl path."""
    import pickle
    state = {
        "phi_m": {
            h: {L: {s: p.detach().cpu().item() for s, p in sdict.items()}
                 for L, sdict in ldict.items()}
            for h, ldict in hbm._phi_m.items()
        },
        "phi_logv": {
            h: {L: {s: p.detach().cpu().item() for s, p in sdict.items()}
                 for L, sdict in ldict.items()}
            for h, ldict in hbm._phi_logv.items()
        },
        "theta_m": {
            h: {s: p.detach().cpu().item() for s, p in sdict.items()}
            for h, sdict in hbm._theta_m.items()
        },
        "theta_logv": {
            h: {s: p.detach().cpu().item() for s, p in sdict.items()}
            for h, sdict in hbm._theta_logv.items()
        },
        "mu_mean": {s: float(v) for s, v in hbm.mu_mean.items()},
        "mu_var": {s: float(v) for s, v in hbm.mu_var.items()},
        "log_sigma_obs": hbm.log_sigma_obs.detach().cpu().item(),
        "log_sigma_r": hbm.log_sigma_r.detach().cpu().item(),
    }
    pkl = _hbm_path(group)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump(state, f)

    # Also snapshot
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = group.replace(" ", "_")
    try:
        shutil.copy2(pkl, SNAPSHOT_DIR / f"{safe}_{ts}.pkl")
    except Exception as e:
        print(f"  [snapshot warning] {e}")


# In-memory cache of loaded HBMs (one per group)
_HBM_CACHE: Dict[str, Any] = {}


def get_hbm(group: str):
    if group not in _HBM_CACHE:
        _HBM_CACHE[group] = _load_or_create_hbm(group)
    return _HBM_CACHE[group]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class GameSession:
    """Server-side state for one player's complete play session.

    Holds the env, robot controller, current layout index, episode state,
    and the WebSocket for streaming frames + receiving keypresses.
    """

    def __init__(self, sid: str, group: str, player_name: str):
        self.sid = sid
        self.group = group
        self.player_name = player_name
        self.layout_idx = 0
        self.episode_count = 0
        self.deliveries_total = 0

        # Per-session JSONL observation logger
        self.pref_logger = PreferenceLogger(str(LOG_DIR))

        # Per-layout state (rebuilt each layout)
        self.mdp = None
        self.oc_env = None
        self.planner = None
        self.detector = None
        self.viz = None
        self.robot = None
        self.subtasks: List[str] = []
        self.layout_name = ""

        # Per-episode state
        self.tick = 0
        self.ep_deliveries = 0
        self.ep_subtask_counts: Dict[str, int] = {}
        self.ep_human_counts: Dict[str, int] = {}
        self.ep_robot_counts: Dict[str, int] = {}
        self.episode_done = False
        self.session_done = False

    def build_layout(self, name: str) -> bool:
        """Build all per-layout objects."""
        from multitask_personalization.envs.overcooked.layouts import (
            CORE_SUBTASKS, get_layout,
        )
        from multitask_personalization.envs.overcooked.overcooked_planner import (
            build_planner,
        )
        from multitask_personalization.envs.overcooked.overcooked_visualizer import (
            OvercookedVisualizer,
        )
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _OCEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

        try:
            if name in CUSTOM_LAYOUTS:
                self.mdp = OvercookedGridworld.from_grid(CUSTOM_LAYOUTS[name])
                self.subtasks = list(CORE_SUBTASKS)
            else:
                spec = get_layout(name)
                self.mdp = OvercookedGridworld.from_layout_name(spec.layout_name)
                self.subtasks = list(spec.subtasks)

            self.oc_env = _OCEnv.from_mdp(self.mdp, horizon=EPISODE_LENGTH_TICKS)
            self.planner = build_planner(self.mdp)
            if self.planner is None:
                return False
            self.detector = SubtaskDetector(self.mdp)
            self.viz = OvercookedVisualizer(tile_size=60, mdp=self.mdp)
            if not self.viz.is_available:
                return False

            hbm = get_hbm(self.group)
            hbm.register_human(self.player_name)
            for ln in LAYOUT_ROTATION:
                hbm.register_layout(self.player_name, ln)

            self.robot = RobotController(
                mdp=self.mdp, planner=self.planner, hbm=hbm,
                subtasks=self.subtasks, layout_name=name,
            )
            self.robot.set_player(self.player_name)
            self.robot.new_episode(self.episode_count)

            self.layout_name = name
            self.tick = 0
            self.ep_deliveries = 0
            self.ep_subtask_counts = {s: 0 for s in self.subtasks}
            self.ep_human_counts = {s: 0 for s in self.subtasks}
            self.ep_robot_counts = {s: 0 for s in self.subtasks}
            self.episode_done = False
            self.oc_env.reset()
            return True
        except Exception as e:
            print(f"[GameSession] build_layout({name}) failed: {e}")
            import traceback
            traceback.print_exc()
            return False


_SESSIONS: Dict[str, GameSession] = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Overcooked Preference Study")

_THIS_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_THIS_DIR / "templates"))
app.mount(
    "/static", StaticFiles(directory=str(_THIS_DIR / "static")), name="static",
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Family select page."""
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "groups": HBM_GROUPS,
            "raval_members": PLAYERS_DB.get_family_members("Raval-Trivedi"),
        },
    )


@app.post("/start")
async def start_session(
    group: str = Form(...),
    player_name: str = Form(...),
):
    """Create a new game session and redirect to the play page."""
    player_name = player_name.strip()
    if not player_name:
        return RedirectResponse(url="/", status_code=303)

    # Register the player in the database
    family = group if group == "Raval-Trivedi" else None
    PLAYERS_DB.add_player(player_name, family=family)

    sid = uuid.uuid4().hex[:12]
    session = GameSession(sid, group, player_name)

    # Build first layout
    if not session.build_layout(LAYOUT_ROTATION[0]):
        return RedirectResponse(url="/?error=layout_build", status_code=303)

    _SESSIONS[sid] = session
    return RedirectResponse(url=f"/play/{sid}", status_code=303)


@app.get("/play/{sid}", response_class=HTMLResponse)
async def play_page(request: Request, sid: str):
    if sid not in _SESSIONS:
        return RedirectResponse(url="/", status_code=303)
    session = _SESSIONS[sid]
    return templates.TemplateResponse(
        request,
        "play.html",
        {
            "sid": sid,
            "player_name": session.player_name,
            "group": session.group,
            "n_layouts": len(LAYOUT_ROTATION),
        },
    )


@app.get("/thanks", response_class=HTMLResponse)
async def thanks(request: Request, name: str = ""):
    return templates.TemplateResponse(
        request, "thanks.html", {"name": name},
    )


# ---------------------------------------------------------------------------
# WebSocket gameplay
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "north": ACTION_NORTH,
    "south": ACTION_SOUTH,
    "east": ACTION_EAST,
    "west": ACTION_WEST,
    "interact": ACTION_INTERACT,
    "stay": ACTION_STAY,
}


def _encode_frame_png(arr) -> str:
    """Encode an (H, W, 3) numpy array as base64 PNG."""
    from PIL import Image
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_status_lines(session: GameSession, ep_deliveries: int,
                        ep_remaining_sec: int) -> List[str]:
    robot_status = "observing" if (
        session.robot and session.robot._wait_timer > 0
    ) else (session.robot.current_subtask if session.robot else "idle")
    return [
        f"{session.player_name}  Layout {session.layout_idx + 1}/{len(LAYOUT_ROTATION)}"
        f"  {session.layout_name}  [{ep_remaining_sec}s]  Soups:{ep_deliveries}",
        f"Robot: {robot_status}",
    ]


def _build_table(session: GameSession) -> Tuple[List[str], List[List[str]]]:
    import numpy as np
    hbm = get_hbm(session.group)
    subtask_short = {
        "fetch_ingredient": "Fetch", "load_pot": "Load",
        "fetch_dish": "Dish", "pickup_soup": "Scoop", "deliver": "Deliver",
    }
    header = ["Subtask", "Phi", "Live", "You", "Bot"]
    rows = []
    for s in session.subtasks:
        try:
            phi = hbm.get_phi(session.player_name, session.layout_name, s)
        except Exception:
            phi = 0.0
        hc = session.ep_human_counts.get(s, 0)
        rc = session.ep_robot_counts.get(s, 0)
        total = hc + rc
        live = (hc - rc) / total if total > 0 else 0.0
        rows.append([
            subtask_short.get(s, s[:6]),
            f"{phi:+.2f}",
            f"{live:+.2f}",
            str(hc), str(rc),
        ])
    return header, rows


@app.websocket("/ws/{sid}")
async def game_ws(websocket: WebSocket, sid: str):
    await websocket.accept()
    if sid not in _SESSIONS:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    session = _SESSIONS[sid]
    pending_human_action = ACTION_STAY  # next action to apply
    last_keypress_time = time.time()
    keypress_intervals_sec: List[float] = []

    # Robot starts very slow for new players — pace 6 (~1 second per action).
    # Adapts down as we observe the human's keystroke tempo.
    if session.robot is not None:
        session.robot._pace_interval = 6

    async def receive_inputs():
        """Background task: receive keypresses from client."""
        nonlocal pending_human_action, last_keypress_time
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "input":
                    act = msg.get("action")
                    if act in ACTION_MAP:
                        pending_human_action = ACTION_MAP[act]
                        # Track keystroke interval (seconds)
                        now = time.time()
                        dt = now - last_keypress_time
                        last_keypress_time = now
                        # Ignore pauses > 5 seconds (player thinking/watching)
                        if dt < 5.0:
                            keypress_intervals_sec.append(dt)
                            if len(keypress_intervals_sec) > 15:
                                keypress_intervals_sec.pop(0)
                            if len(keypress_intervals_sec) >= 3:
                                avg_sec = sum(keypress_intervals_sec) / len(keypress_intervals_sec)
                                # Convert seconds → ticks (6 fps → 167ms/tick)
                                avg_ticks = avg_sec / TICK_INTERVAL_SEC
                                # Use 1.3x multiplier and wider range so
                                # the robot can be genuinely slow for new
                                # players. Max 8 ticks = ~1.3 seconds per
                                # robot action.
                                if session.robot is not None:
                                    scaled = int(round(avg_ticks * 1.3))
                                    session.robot._pace_interval = max(2, min(8, scaled))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[ws receive] {e}")

    receiver = asyncio.create_task(receive_inputs())

    try:
        # Send initial ready message
        await websocket.send_json({
            "type": "ready",
            "player": session.player_name,
            "group": session.group,
            "layout": session.layout_name,
        })

        # --- Main game loop: iterate over layouts ---
        while session.layout_idx < len(LAYOUT_ROTATION) and not session.session_done:
            ep_start = time.time()

            while not session.episode_done:
                tick_start = time.time()

                # Apply the latest human input (or stay)
                human_action = pending_human_action
                pending_human_action = ACTION_STAY  # consume

                # Robot action
                try:
                    robot_action = session.robot.step(session.oc_env.state)
                except Exception as e:
                    print(f"[robot] {e}")
                    robot_action = ACTION_STAY

                # Step environment
                state_before = session.oc_env.state
                try:
                    _, sparse_r, done, _ = session.oc_env.step(
                        (robot_action, human_action)
                    )
                except Exception as e:
                    print(f"[env step] {e}")
                    break

                state = session.oc_env.state
                session.tick += 1

                # Detect completions
                hc = session.detector.detect(
                    state_before, state, human_action, human_idx=1
                )
                rc = session.detector.detect(
                    state_before, state, robot_action, human_idx=0
                )

                hbm = get_hbm(session.group)

                if hc is not None:
                    session.ep_subtask_counts[hc] = session.ep_subtask_counts.get(hc, 0) + 1
                    session.ep_human_counts[hc] = session.ep_human_counts.get(hc, 0) + 1
                    hbm.observe(
                        session.player_name, session.layout_name,
                        hc, "human", 1.0,
                    )
                    session.pref_logger.log_observation(
                        session.player_name, session.layout_name,
                        session.episode_count + 1, hc, 1.0, "human",
                    )
                    session.robot.maybe_reset_wait(hc)

                if rc is not None:
                    session.ep_subtask_counts[rc] = session.ep_subtask_counts.get(rc, 0) + 1
                    session.ep_robot_counts[rc] = session.ep_robot_counts.get(rc, 0) + 1
                    hbm.observe(
                        session.player_name, session.layout_name,
                        rc, "robot", 0.5,
                    )
                    session.pref_logger.log_observation(
                        session.player_name, session.layout_name,
                        session.episode_count + 1, rc, 0.5, "robot",
                    )

                if sparse_r > 0:
                    session.ep_deliveries += 1
                    session.deliveries_total += 1

                # Send frame
                ep_elapsed = time.time() - ep_start
                ep_remaining = max(0, int(EPISODE_LENGTH_TICKS / 6 - ep_elapsed))

                status_lines = _build_status_lines(
                    session, session.ep_deliveries, ep_remaining,
                )
                if sparse_r > 0:
                    status_lines.append(f"*** DELIVERED #{session.ep_deliveries}! ***")

                table_header, table_rows = _build_table(session)
                table_footer = [
                    f"Group: {session.group}",
                ]

                try:
                    arr = session.viz.composite_to_array(
                        state, status_lines, table_rows, table_header,
                        table_footer=table_footer,
                    )
                    if arr is not None:
                        png_b64 = _encode_frame_png(arr)
                        await websocket.send_json({
                            "type": "frame",
                            "png": png_b64,
                            "status": " | ".join(status_lines[:1]),
                        })
                except (WebSocketDisconnect, RuntimeError):
                    return
                except Exception as e:
                    print(f"[render] {e}")

                # Check episode done
                if done or session.tick >= EPISODE_LENGTH_TICKS:
                    session.episode_done = True
                    break

                # Pace to 6 fps
                tick_elapsed = time.time() - tick_start
                if tick_elapsed < TICK_INTERVAL_SEC:
                    await asyncio.sleep(TICK_INTERVAL_SEC - tick_elapsed)

            # End of episode
            print(f"  [{session.player_name}] {session.layout_name} done. "
                  f"Deliveries: {session.ep_deliveries}")

            # Contrastive observations + end_episode
            hbm = get_hbm(session.group)
            max_h = max(session.ep_human_counts.values()) if session.ep_human_counts else 0
            if max_h > 0:
                for s in session.subtasks:
                    hc_count = session.ep_human_counts.get(s, 0)
                    freq = hc_count / max_h
                    if freq == 0:
                        for _ in range(min(max_h, 4)):
                            hbm.observe(
                                session.player_name, session.layout_name,
                                s, "robot", 0.5,
                            )
                    elif freq < 0.4:
                        hbm.observe(
                            session.player_name, session.layout_name,
                            s, "robot", 0.5,
                        )

            try:
                hbm.end_episode(session.player_name)
                _save_hbm(session.group, hbm)
                PLAYERS_DB.increment_episodes(session.player_name)
            except Exception as e:
                print(f"[hbm save] {e}")

            # Move to next layout
            session.layout_idx += 1
            session.episode_count += 1

            if session.layout_idx >= len(LAYOUT_ROTATION):
                session.session_done = True
                break

            # Build next layout
            next_layout = LAYOUT_ROTATION[session.layout_idx]
            # Remember the pace we learned so the new RobotController
            # for the next layout starts slow enough.
            carried_pace = (
                session.robot._pace_interval
                if session.robot is not None else 6
            )
            if not session.build_layout(next_layout):
                print(f"  [{session.player_name}] Failed to build {next_layout}, skipping")
                continue
            if session.robot is not None:
                session.robot._pace_interval = carried_pace

            # Notify client of layout change
            await websocket.send_json({
                "type": "layout_change",
                "layout": next_layout,
                "index": session.layout_idx + 1,
                "total": len(LAYOUT_ROTATION),
            })
            await asyncio.sleep(0.5)

        # Session done
        await websocket.send_json({"type": "session_done"})

    except WebSocketDisconnect:
        print(f"[ws] {session.player_name} disconnected")
    except Exception as e:
        print(f"[ws] error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        receiver.cancel()
        try:
            await receiver
        except Exception:
            pass
