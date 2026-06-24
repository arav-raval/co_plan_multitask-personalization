"""
Preference model baselines for ablation experiments.

Two models are provided, each implementing the same duck-typed interface as
HierarchicalPreferenceModel so they drop into SpicesAssignCSPGenerator without
any changes to the CSP or environment code.

FlatPreferenceModel
    Flat Bayesian model: independent Beta-Bernoulli posterior per
    (human, recipe, spice). No shared mu/theta hierarchy across recipes or
    humans. Ablates Claims 2 and 3 (cross-recipe and cross-human transfer).

CBTLClassifierModel
    CBTL-faithful baseline adapted to the binary assignment domain.

    The original CBTL paper (Silver et al., 2025) learns constraint parameters
    using domain-specific classifiers (Bounded1DClassifier for continuous
    features, KNN for distance features).  Since our domain has binary
    assignment decisions with no continuous feature space, we adapt the model
    to a Beta-Bernoulli posterior over P(human preferred) per (human, spice),
    shared across all recipes.  This is the natural conjugate analog for
    discrete binary observations and matches the structural properties of the
    original: non-hierarchical, context-agnostic, with a proper posterior for
    entropy-based active learning.

    Documented adaptations from the original CBTL:
      1. Domain: binary assignment (flag ∈ {0,1}) vs. continuous features
         (temperature, distance). Beta-Bernoulli replaces Bounded1DClassifier.
      2. Feedback: binary task_score {+1,-1} vs. structured critiques
         (hotter/colder/good). We use task_score as a direct Boolean label.
      3. Entropy: raw Bernoulli entropy H(Bern(p)) without variance weighting,
         matching the original's cost = 1 - mean_entropy(logprob).
      4. Context sharing: per (human, spice) shared across all recipes,
         matching the original's context-agnostic constraint models.
      5. Constraint threshold: log(0.3) vs. original's log(0.5). Kept at
         log(0.3) for consistency across all methods in our experiments.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Neutral mood posterior used as a static placeholder (these models do not
# infer mood, but _AssignPreferenceGenerator reads _mood_posterior directly).
_NEUTRAL_POSTERIOR = np.array([0.0, 1.0, 0.0], dtype=float)  # [all_self, neutral, none_self]


# ---------------------------------------------------------------------------
# FlatPreferenceModel
# ---------------------------------------------------------------------------

class FlatPreferenceModel:
    """
    Flat independent Beta-Bernoulli preference model.

    For each (human, recipe, spice) triple, maintains a Beta posterior
    over p = P(human preferred). The logit log(p/(1-p)) is exposed as
    phi to match the HBM interface; log_prob_prefer uses the posterior
    mean directly.

    Compared to HierarchicalPreferenceModel:
    - No mu (global prior) shared across humans
    - No theta (human-level prior) shared across recipes
    - Every (human, recipe, spice) cell is updated independently

    This isolates the contribution of the HBM hierarchy for Claims 2 & 3.
    """

    def __init__(
        self,
        spices: List[str],
        recipes: Optional[List[str]] = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self.spices = list(spices)
        self.recipes: List[str] = list(recipes) if recipes else []
        # Beta prior hyperparameters (uninformative: alpha=beta=1 -> uniform)
        self._alpha0 = alpha0
        self._beta0 = beta0

        # alpha[human][recipe][spice], beta[human][recipe][spice]
        # alpha counts "human preferred" observations + prior
        # beta  counts "robot preferred" observations + prior
        self._alpha: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._beta: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Placeholder mood posterior (static neutral — no mood inference)
        self._mood_posterior: Dict[str, np.ndarray] = {}

        # Episode buffer: list of (actor, spice, satisfaction)
        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._current_recipe: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_human(self, human_id: str) -> None:
        if human_id not in self._alpha:
            self._alpha[human_id] = {}
            self._beta[human_id] = {}
            self._mood_posterior[human_id] = _NEUTRAL_POSTERIOR.copy()
            self._episode_data[human_id] = []
            self._current_recipe[human_id] = None

    def register_recipe(self, human_id: str, recipe_name: str) -> None:
        self.register_human(human_id)
        if recipe_name not in self._alpha[human_id]:
            self._alpha[human_id][recipe_name] = {
                s: self._alpha0 for s in self.spices
            }
            self._beta[human_id][recipe_name] = {
                s: self._beta0 for s in self.spices
            }
        if recipe_name not in self.recipes:
            self.recipes.append(recipe_name)

    def _ensure_registered(self, human_id: str, recipe_name: str) -> None:
        self.register_human(human_id)
        self.register_recipe(human_id, recipe_name)

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _p_human(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Posterior mean P(human preferred) = alpha / (alpha + beta)."""
        a = self._alpha.get(human_id, {}).get(recipe_name, {}).get(spice, self._alpha0)
        b = self._beta.get(human_id, {}).get(recipe_name, {}).get(spice, self._beta0)
        return a / (a + b)

    # ------------------------------------------------------------------
    # HBM public interface (duck-typed)
    # ------------------------------------------------------------------

    def log_prob_prefer(
        self, human_id: str, recipe_name: str, spice: str, actor: str
    ) -> float:
        p = self._p_human(human_id, recipe_name, spice)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        lp = math.log(p) if actor == "human" else math.log(1.0 - p)
        return float(max(lp, -20.0))

    def get_phi(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Logit of posterior mean: log(p / (1-p))."""
        p = self._p_human(human_id, recipe_name, spice)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        return math.log(p / (1.0 - p))

    def get_phi_var(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Posterior variance of the Beta: alpha*beta / ((a+b)^2 * (a+b+1))."""
        a = self._alpha.get(human_id, {}).get(recipe_name, {}).get(spice, self._alpha0)
        b = self._beta.get(human_id, {}).get(recipe_name, {}).get(spice, self._beta0)
        n = a + b
        return (a * b) / (n * n * (n + 1.0))

    def get_phi_entropy(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Variance-weighted Bernoulli entropy (same formula as HBM)."""
        p = self._p_human(human_id, recipe_name, spice)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        H = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
        return H * self.get_phi_var(human_id, recipe_name, spice)

    def get_running_psi(self, human_id: str) -> float:
        """No psi in flat model."""
        return 0.0

    def preferred_actor(self, human_id: str, recipe_name: str, spice: str) -> str:
        return "human" if self._p_human(human_id, recipe_name, spice) >= 0.5 else "robot"

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def observe(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
        force_neutral_mood: bool = False,
    ) -> None:
        self._ensure_registered(human_id, recipe_name)
        self._current_recipe[human_id] = recipe_name
        self._episode_data[human_id].append((actor, spice, satisfaction))

    def end_episode(self, human_id: str, **kwargs) -> None:
        """Batch-update Beta posterior from binary actor labels.

        Uses the actor field directly as the behavioral observation signal,
        matching the new task_score convention (+1 for human claimed,
        -1 for robot acted). This is consistent with CBTLClassifierModel
        and ensures fair comparison.
        """
        for actor, spice, task_score in self._episode_data.get(human_id, []):
            recipe = self._current_recipe.get(human_id)
            if recipe is None or spice not in self._alpha.get(human_id, {}).get(recipe, {}):
                continue
            # Binary label: did the human claim this spice?
            human_claimed = (actor == "human")
            if human_claimed:
                self._alpha[human_id][recipe][spice] += 1.0
            else:
                self._beta[human_id][recipe][spice] += 1.0
        self._episode_data[human_id] = []

    def observe_eval(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
        done: bool,
    ) -> None:
        """Flat model has no running psi; this is a no-op."""

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = {"alpha": self._alpha, "beta": self._beta}
        with open(path / "flat_model.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "flat_model.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._alpha = state["alpha"]
        self._beta = state["beta"]


# ---------------------------------------------------------------------------
# CBTLClassifierModel
# ---------------------------------------------------------------------------

class CBTLClassifierModel:
    """
    CBTL-faithful baseline adapted to binary assignment.

    Maintains a Beta-Bernoulli posterior over P(human preferred) per
    (human, spice), **shared across all recipes** (context-agnostic).
    Updated from binary task_score labels (+1 = human claimed, -1 = didn't).

    By default, posteriors are per-human. When `pooled_across_humans=True`,
    a single set of posteriors is shared across all humans — this matches
    the spirit of the original CBTL's shared-constraint philosophy and
    enables cross-human warm-start transfer.

    This captures the structural properties of the original CBTL:
      - Non-hierarchical: no mu, no theta, no cross-recipe transfer
      - Context-agnostic: one posterior per spice for all recipes
      - Proper posterior: Beta counts from binary observations
      - Raw entropy exploration: H(Bern(p)) without variance weighting,
        matching the original's cost = 1 - mean_entropy(logprob)
      - No transient state modeling: no psi

    See module docstring for documented adaptations from the original.
    """

    def __init__(
        self,
        spices: List[str],
        recipes: Optional[List[str]] = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        pooled_across_humans: bool = True,
    ) -> None:
        self.spices = list(spices)
        self.recipes: List[str] = list(recipes) if recipes else []
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._pooled_across_humans = pooled_across_humans

        # alpha[key][spice], beta[key][spice] — per-human by default,
        # pooled across all humans when pooled_across_humans=True.
        self._alpha: Dict[str, Dict[str, float]] = {}
        self._beta: Dict[str, Dict[str, float]] = {}

        self._mood_posterior: Dict[str, np.ndarray] = {}
        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._current_recipe: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _pool_key(self, human_id: str) -> str:
        """Storage key — single pool under pooled_across_humans=True."""
        return "__pooled__" if self._pooled_across_humans else human_id

    def register_human(self, human_id: str) -> None:
        key = self._pool_key(human_id)
        if key not in self._alpha:
            self._alpha[key] = {s: self._alpha0 for s in self.spices}
            self._beta[key] = {s: self._beta0 for s in self.spices}
        # Per-human buffer state (independent of pooling)
        if human_id not in self._episode_data:
            self._mood_posterior[human_id] = _NEUTRAL_POSTERIOR.copy()
            self._episode_data[human_id] = []
            self._current_recipe[human_id] = None

    def register_recipe(self, human_id: str, recipe_name: str) -> None:
        """No-op for preference params (context-agnostic); just track name."""
        self.register_human(human_id)
        if recipe_name not in self.recipes:
            self.recipes.append(recipe_name)

    def _ensure_registered(self, human_id: str, recipe_name: str) -> None:
        self.register_human(human_id)
        self.register_recipe(human_id, recipe_name)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _p_human(self, human_id: str, spice: str) -> float:
        """Posterior mean P(human preferred) = alpha / (alpha + beta)."""
        key = self._pool_key(human_id)
        a = self._alpha.get(key, {}).get(spice, self._alpha0)
        b = self._beta.get(key, {}).get(spice, self._beta0)
        return a / (a + b)

    # ------------------------------------------------------------------
    # HBM public interface (duck-typed)
    # ------------------------------------------------------------------

    def log_prob_prefer(
        self, human_id: str, recipe_name: str, spice: str, actor: str
    ) -> float:
        p = self._p_human(human_id, spice)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        lp = math.log(p) if actor == "human" else math.log(1.0 - p)
        return float(max(lp, -20.0))

    def get_phi(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Logit of posterior mean: log(p / (1-p))."""
        p = self._p_human(human_id, spice)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        return math.log(p / (1.0 - p))

    def get_phi_var(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Beta posterior variance: alpha*beta / ((a+b)^2 * (a+b+1))."""
        key = self._pool_key(human_id)
        a = self._alpha.get(key, {}).get(spice, self._alpha0)
        b = self._beta.get(key, {}).get(spice, self._beta0)
        n = a + b
        return (a * b) / (n * n * (n + 1.0))

    def get_phi_entropy(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Raw Bernoulli entropy H(Bern(p)) — no variance weighting.

        Matches the original CBTL's exploration: cost = 1 - mean_entropy.
        Note: in a binary domain, H is symmetric across both flags, so this
        provides no directional exploration signal (effectively random).
        """
        p = self._p_human(human_id, spice)
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)

    def get_running_psi(self, human_id: str) -> float:
        return 0.0

    def preferred_actor(self, human_id: str, recipe_name: str, spice: str) -> str:
        return "human" if self._p_human(human_id, spice) >= 0.5 else "robot"

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def observe(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
        force_neutral_mood: bool = False,
    ) -> None:
        self._ensure_registered(human_id, recipe_name)
        self._current_recipe[human_id] = recipe_name
        self._episode_data[human_id].append((actor, spice, satisfaction))

    def end_episode(self, human_id: str, **kwargs) -> None:
        """Batch-update Beta posterior from binary task_score labels.

        Uses the task_score routed by the CSP generator:
          +1 (human claimed) → increment alpha (evidence for human preference)
          -1 (human didn't)  → increment beta  (evidence against)
           0 (conflict)      → skipped upstream (never reaches here)

        This matches the original CBTL's accumulate-all-data approach:
        the Beta posterior is a sufficient statistic for all binary labels seen.
        """
        key = self._pool_key(human_id)
        for actor, spice, task_score in self._episode_data.get(human_id, []):
            if spice not in self._alpha.get(key, {}):
                continue
            # Binary label: did the human claim this spice?
            human_claimed = (actor == "human")
            if human_claimed:
                self._alpha[key][spice] += 1.0
            else:
                self._beta[key][spice] += 1.0
        self._episode_data[human_id] = []

    def observe_eval(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
        done: bool,
    ) -> None:
        """No running psi; no-op."""

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = {"alpha": self._alpha, "beta": self._beta}
        with open(path / "cbtl_adapted.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "cbtl_adapted.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._alpha = state["alpha"]
        self._beta = state["beta"]
