"""
Preference model baselines for Overcooked ablation experiments.

Mirrors ``spices_baselines.py`` but uses Overcooked's (layout, subtask) vocabulary
instead of (recipe, spice).

FlatPreferenceModel
    Independent Beta-Bernoulli posterior per (human, layout, subtask).
    No hierarchy, no transfer across layouts.

CBTLClassifierModel
    Supervised logistic classifier (original CBTL paper).
    One weight per (human, subtask), shared across all layouts.
    No hierarchy, no psi, no uncertainty-based exploration.

Both implement the same duck-typed interface as OvercookedPreferenceModel so
they drop into OvercookedAssignCSPGenerator without changes.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# FlatPreferenceModel (Overcooked)
# ---------------------------------------------------------------------------

class OvercookedFlatPreferenceModel:
    """Flat independent Beta-Bernoulli preference model for Overcooked.

    For each (human, layout, subtask) triple, maintains a Beta posterior over
    p = P(human preferred). Ablates the HBM hierarchy (no mu, no theta transfer).
    """

    def __init__(
        self,
        subtasks: List[str],
        layouts: Optional[List[str]] = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self.subtasks = list(subtasks)
        self._layouts: List[str] = list(layouts) if layouts else []
        self._alpha0 = alpha0
        self._beta0 = beta0

        # alpha[human][layout][subtask], beta[human][layout][subtask]
        self._alpha: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._beta: Dict[str, Dict[str, Dict[str, float]]] = {}

        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._current_layout: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_human(self, human_id: str) -> None:
        if human_id not in self._alpha:
            self._alpha[human_id] = {}
            self._beta[human_id] = {}
            self._episode_data[human_id] = []
            self._current_layout[human_id] = None

    def register_layout(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        if layout_name not in self._alpha[human_id]:
            self._alpha[human_id][layout_name] = {
                s: self._alpha0 for s in self.subtasks
            }
            self._beta[human_id][layout_name] = {
                s: self._beta0 for s in self.subtasks
            }
        if layout_name not in self._layouts:
            self._layouts.append(layout_name)

    def _ensure_registered(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        self.register_layout(human_id, layout_name)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _p_human(self, human_id: str, layout_name: str, subtask: str) -> float:
        a = self._alpha.get(human_id, {}).get(layout_name, {}).get(subtask, self._alpha0)
        b = self._beta.get(human_id, {}).get(layout_name, {}).get(subtask, self._beta0)
        return a / (a + b)

    # ------------------------------------------------------------------
    # HBM public interface (duck-typed)
    # ------------------------------------------------------------------

    def log_prob_prefer(
        self, human_id: str, layout_name: str, subtask: str, actor: str
    ) -> float:
        p = self._p_human(human_id, layout_name, subtask)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        lp = math.log(p) if actor == "human" else math.log(1.0 - p)
        return float(max(lp, -20.0))

    def get_phi(self, human_id: str, layout_name: str, subtask: str) -> float:
        p = self._p_human(human_id, layout_name, subtask)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        return math.log(p / (1.0 - p))

    def get_phi_var(self, human_id: str, layout_name: str, subtask: str) -> float:
        a = self._alpha.get(human_id, {}).get(layout_name, {}).get(subtask, self._alpha0)
        b = self._beta.get(human_id, {}).get(layout_name, {}).get(subtask, self._beta0)
        n = a + b
        return (a * b) / (n * n * (n + 1.0))

    def get_phi_entropy(self, human_id: str, layout_name: str, subtask: str) -> float:
        p = self._p_human(human_id, layout_name, subtask)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        H = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
        return H * self.get_phi_var(human_id, layout_name, subtask)

    def get_running_psi(self, human_id: str, subtask: str) -> float:
        return 0.0

    def preferred_actor(self, human_id: str, layout_name: str, subtask: str) -> str:
        return "human" if self._p_human(human_id, layout_name, subtask) >= 0.5 else "robot"

    def get_psi_vec(self, human_id: str) -> list[float]:
        return [0.0] * len(self.subtasks)

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def observe(
        self, human_id: str, layout_name: str, subtask: str,
        actor: str, task_score: float,
    ) -> None:
        self._ensure_registered(human_id, layout_name)
        self._current_layout[human_id] = layout_name
        self._episode_data[human_id].append((actor, subtask, task_score))

    def end_episode(self, human_id: str) -> None:
        layout = self._current_layout.get(human_id)
        for actor, subtask, score in self._episode_data.get(human_id, []):
            if layout is None or subtask not in self._alpha.get(human_id, {}).get(layout, {}):
                continue
            weight = (score + 1.0) / 2.0
            if actor == "human":
                self._alpha[human_id][layout][subtask] += weight
                self._beta[human_id][layout][subtask] += (1.0 - weight)
            else:
                self._alpha[human_id][layout][subtask] += (1.0 - weight)
                self._beta[human_id][layout][subtask] += weight
        self._episode_data[human_id] = []

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = {"alpha": self._alpha, "beta": self._beta}
        with open(path / "overcooked_flat_model.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "overcooked_flat_model.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._alpha = state["alpha"]
        self._beta = state["beta"]


# ---------------------------------------------------------------------------
# CBTLClassifierModel (Overcooked)
# ---------------------------------------------------------------------------

class OvercookedCBTLClassifierModel:
    """CBTL-style supervised logistic classifier for Overcooked.

    One weight per (human, subtask), shared across all layouts.
    No hierarchy, no psi, no uncertainty. Direct implementation of the
    CBTL paper's "Supervised Classification" learning rule.
    """

    def __init__(
        self,
        subtasks: List[str],
        layouts: Optional[List[str]] = None,
        lr: float = 0.1,
        prior_var: float = 1.0,
    ) -> None:
        self.subtasks = list(subtasks)
        self._layouts: List[str] = list(layouts) if layouts else []
        self._lr = lr
        self._prior_var = prior_var

        # w[human][subtask]: logistic regression weight
        self._w: Dict[str, Dict[str, float]] = {}
        self._n: Dict[str, Dict[str, int]] = {}

        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._current_layout: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_human(self, human_id: str) -> None:
        if human_id not in self._w:
            self._w[human_id] = {s: 0.0 for s in self.subtasks}
            self._n[human_id] = {s: 0 for s in self.subtasks}
            self._episode_data[human_id] = []
            self._current_layout[human_id] = None

    def register_layout(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        if layout_name not in self._layouts:
            self._layouts.append(layout_name)

    def _ensure_registered(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        self.register_layout(human_id, layout_name)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _p_human(self, human_id: str, subtask: str) -> float:
        w = self._w.get(human_id, {}).get(subtask, 0.0)
        return self._sigmoid(w)

    # ------------------------------------------------------------------
    # HBM public interface (duck-typed)
    # ------------------------------------------------------------------

    def log_prob_prefer(
        self, human_id: str, layout_name: str, subtask: str, actor: str
    ) -> float:
        p = self._p_human(human_id, subtask)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        lp = math.log(p) if actor == "human" else math.log(1.0 - p)
        return float(max(lp, -20.0))

    def get_phi(self, human_id: str, layout_name: str, subtask: str) -> float:
        return self._w.get(human_id, {}).get(subtask, 0.0)

    def get_phi_var(self, human_id: str, layout_name: str, subtask: str) -> float:
        """Approximate variance from observation count: var ~ 1 / (n + 1)."""
        n = self._n.get(human_id, {}).get(subtask, 0)
        return 1.0 / (n + 1.0)

    def get_phi_entropy(self, human_id: str, layout_name: str, subtask: str) -> float:
        """Variance-weighted Bernoulli entropy, matching the CBTL paper's exploration."""
        p = self._p_human(human_id, subtask)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        H = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
        return H * self.get_phi_var(human_id, layout_name, subtask)

    def get_running_psi(self, human_id: str, subtask: str) -> float:
        return 0.0

    def preferred_actor(self, human_id: str, layout_name: str, subtask: str) -> str:
        return "human" if self._p_human(human_id, subtask) >= 0.5 else "robot"

    def get_psi_vec(self, human_id: str) -> list[float]:
        return [0.0] * len(self.subtasks)

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def observe(
        self, human_id: str, layout_name: str, subtask: str,
        actor: str, task_score: float,
    ) -> None:
        self._ensure_registered(human_id, layout_name)
        self._current_layout[human_id] = layout_name
        self._episode_data[human_id].append((actor, subtask, task_score))

    def end_episode(self, human_id: str) -> None:
        for actor, subtask, score in self._episode_data.get(human_id, []):
            if subtask not in self._w.get(human_id, {}):
                continue
            y_sat = (score + 1.0) / 2.0
            y_human = y_sat if actor == "human" else (1.0 - y_sat)
            p = self._p_human(human_id, subtask)
            grad = (y_human - p) - self._w[human_id][subtask] / self._prior_var
            n = self._n[human_id][subtask]
            step = self._lr / math.sqrt(n + 1)
            self._w[human_id][subtask] += step * grad
            self._n[human_id][subtask] += 1
        self._episode_data[human_id] = []

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = {"w": self._w, "n": self._n}
        with open(path / "overcooked_cbtl_classifier.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "overcooked_cbtl_classifier.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._w = state["w"]
        self._n = state["n"]
