"""Reuse the pygame script's game logic without running its main().

Loads ``scripts/run_human_overcooked.py`` as a module and exposes:
  - RobotController
  - SubtaskDetector
  - PlayerDatabase
  - CUSTOM_LAYOUTS
  - ACTION_NORTH/SOUTH/EAST/WEST/STAY/INTERACT
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Locate the pygame script
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "run_human_overcooked.py"

if not _SCRIPT.exists():
    raise ImportError(f"Could not find {_SCRIPT}")

_spec = importlib.util.spec_from_file_location("_human_overcooked", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)

# Execute the module — this defines all the classes and constants but does
# NOT call main() since main() is gated on __name__ == "__main__".
_spec.loader.exec_module(_mod)

# Re-export the pieces we need
RobotController = _mod.RobotController
SubtaskDetector = _mod.SubtaskDetector
PlayerDatabase = _mod.PlayerDatabase
PreferenceLogger = _mod.PreferenceLogger
CUSTOM_LAYOUTS = _mod.CUSTOM_LAYOUTS

ACTION_NORTH = _mod.ACTION_NORTH
ACTION_SOUTH = _mod.ACTION_SOUTH
ACTION_EAST = _mod.ACTION_EAST
ACTION_WEST = _mod.ACTION_WEST
ACTION_STAY = _mod.ACTION_STAY
ACTION_INTERACT = _mod.ACTION_INTERACT
