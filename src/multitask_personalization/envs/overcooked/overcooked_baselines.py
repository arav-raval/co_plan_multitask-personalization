"""
Preference model baselines for Overcooked ablation experiments.

Mirrors ``spices_baselines.py`` but uses Overcooked's (layout, subtask) vocabulary
instead of (recipe, spice).

FlatPreferenceModel
    Independent Beta-Bernoulli posterior per (human, layout, subtask).
    No hierarchy, no transfer across layouts.

CBTLClassifierModel
    CBTL-faithful baseline adapted to the binary subtask-assignment domain.

    The original CBTL paper (Silver et al., 2025) learns constraint parameters
    using domain-specific classifiers (Bounded1DClassifier for continuous
    features).  Since our domain has binary assignment decisions, we adapt to
    a Beta-Bernoulli posterior over P(human preferred) per (human, subtask),
    shared across all layouts.  See spices_baselines.py module docstring for
    the full list of documented adaptations.

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
        """Batch-update Beta posteriors from binary actor labels.

        Matches CBTL's update rule for fair comparison: alpha increments
        when the human claimed the subtask, beta when the robot did.
        """
        layout = self._current_layout.get(human_id)
        for actor, subtask, score in self._episode_data.get(human_id, []):
            if layout is None or subtask not in self._alpha.get(human_id, {}).get(layout, {}):
                continue
            human_claimed = (actor == "human")
            if human_claimed:
                self._alpha[human_id][layout][subtask] += 1.0
            else:
                self._beta[human_id][layout][subtask] += 1.0
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
    """CBTL-faithful baseline for Overcooked (binary subtask assignment).

    Beta-Bernoulli posterior per (human, subtask), shared across all layouts.
    Updated from binary task_score labels. Raw Bernoulli entropy for
    exploration (no variance weighting). No hierarchy, no psi.

    By default, posteriors are per-human. When `pooled_across_humans=True`,
    a single set of posteriors is shared across all humans — this matches
    the spirit of the original CBTL's shared-constraint philosophy and
    enables cross-human warm-start transfer. When a new human is registered,
    they immediately see the pooled posterior (no fresh prior), which is
    the cross-human analog of CBTL's cross-context pooling.

    See spices_baselines.py module docstring for documented adaptations
    from the original CBTL paper.
    """

    def __init__(
        self,
        subtasks: List[str],
        layouts: Optional[List[str]] = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        pooled_across_humans: bool = True,
    ) -> None:
        self.subtasks = list(subtasks)
        self._layouts: List[str] = list(layouts) if layouts else []
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._pooled_across_humans = pooled_across_humans

        # alpha[human][subtask], beta[human][subtask] — per-human posteriors.
        # When pooled_across_humans=True, all humans map to the same pool
        # via _pool_key(human_id) -> "__pooled__".
        self._alpha: Dict[str, Dict[str, float]] = {}
        self._beta: Dict[str, Dict[str, float]] = {}

        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._current_layout: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _pool_key(self, human_id: str) -> str:
        """Returns the storage key for this human.

        Under pooled_across_humans=True, all humans share a single '__pooled__'
        slot. Otherwise each human has their own slot.
        """
        return "__pooled__" if self._pooled_across_humans else human_id

    def register_human(self, human_id: str) -> None:
        key = self._pool_key(human_id)
        if key not in self._alpha:
            self._alpha[key] = {s: self._alpha0 for s in self.subtasks}
            self._beta[key] = {s: self._beta0 for s in self.subtasks}
        # Episode buffer is still per-human-id (so different humans'
        # observations don't mix within a single episode).
        if human_id not in self._episode_data:
            self._episode_data[human_id] = []
            self._current_layout[human_id] = None

    def register_layout(self, human_id: str, layout_name: str) -> None:
        """No-op for preference params (context-agnostic); just track name."""
        self.register_human(human_id)
        if layout_name not in self._layouts:
            self._layouts.append(layout_name)

    def _ensure_registered(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        self.register_layout(human_id, layout_name)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _p_human(self, human_id: str, subtask: str) -> float:
        """Posterior mean P(human preferred) = alpha / (alpha + beta)."""
        key = self._pool_key(human_id)
        a = self._alpha.get(key, {}).get(subtask, self._alpha0)
        b = self._beta.get(key, {}).get(subtask, self._beta0)
        return a / (a + b)

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
        """Logit of posterior mean: log(p / (1-p))."""
        p = self._p_human(human_id, subtask)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        return math.log(p / (1.0 - p))

    def get_phi_var(self, human_id: str, layout_name: str, subtask: str) -> float:
        """Beta posterior variance: alpha*beta / ((a+b)^2 * (a+b+1))."""
        key = self._pool_key(human_id)
        a = self._alpha.get(key, {}).get(subtask, self._alpha0)
        b = self._beta.get(key, {}).get(subtask, self._beta0)
        n = a + b
        return (a * b) / (n * n * (n + 1.0))

    def get_phi_entropy(self, human_id: str, layout_name: str, subtask: str) -> float:
        """Raw Bernoulli entropy H(Bern(p)) — no variance weighting.

        Matches the original CBTL's exploration mechanism.
        """
        p = self._p_human(human_id, subtask)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)

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
        """Batch-update Beta posterior from binary task_score labels.

        +1 (human claimed) → increment alpha
        -1 (human didn't)  → increment beta
         0 (conflict)      → skipped upstream
        """
        key = self._pool_key(human_id)
        for actor, subtask, task_score in self._episode_data.get(human_id, []):
            if subtask not in self._alpha.get(key, {}):
                continue
            human_claimed = (actor == "human")
            if human_claimed:
                self._alpha[key][subtask] += 1.0
            else:
                self._beta[key][subtask] += 1.0
        self._episode_data[human_id] = []

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = {"alpha": self._alpha, "beta": self._beta}
        with open(path / "overcooked_cbtl_adapted.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "overcooked_cbtl_adapted.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._alpha = state["alpha"]
        self._beta = state["beta"]
