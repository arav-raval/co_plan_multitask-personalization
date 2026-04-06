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
    Supervised logistic classifier following the original CBTL paper. Maintains
    a running estimate of P(human preferred | spice) via online logistic
    regression with a Laplace prior. No hierarchy, no psi, no mood handling.
    Direct implementation of the paper's "Supervised Classification" learning.
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
        """Batch-update Beta posteriors from episode observations."""
        for actor, spice, satisfaction in self._episode_data.get(human_id, []):
            recipe = self._current_recipe.get(human_id)
            if recipe is None or spice not in self._alpha.get(human_id, {}).get(recipe, {}):
                continue
            # Convert satisfaction in [-1,+1] to a [0,1] weight for the update.
            # High satisfaction when actor=human -> strong evidence human preferred.
            # Low satisfaction (negative) when actor=human -> evidence against.
            weight = (satisfaction + 1.0) / 2.0  # map [-1,1] -> [0,1]
            if actor == "human":
                self._alpha[human_id][recipe][spice] += weight
                self._beta[human_id][recipe][spice] += (1.0 - weight)
            else:
                # robot assigned: high satisfaction means robot preferred
                self._alpha[human_id][recipe][spice] += (1.0 - weight)
                self._beta[human_id][recipe][spice] += weight
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
    CBTL-style supervised logistic classifier (original paper baseline).

    Maintains one logistic regression weight w_s per (human, spice) where
    P(human preferred | spice) = sigmoid(w_s). Updated via online gradient
    descent on the binary cross-entropy loss using satisfaction as a soft
    label, matching the paper's "Supervised Classification" approach.

    Key structural differences from HierarchicalPreferenceModel:
    - No hierarchy: no mu (global) or theta (human-level) sharing
    - No recipe-level phi: weight is per (human, spice) only, shared across
      all recipes (simplest CBTL interpretation — the classifier sees
      spice identity but not which recipe it appears in)
    - No psi: mood contamination goes directly into the classifier weights
    - No Bayesian uncertainty: no variance, no exploration signal beyond
      the initial weight (exploit-only by construction)

    This directly implements the paper's learning rule. We use a Gaussian
    prior N(0, prior_var) as regularization, equivalent to MAP logistic
    regression.
    """

    def __init__(
        self,
        spices: List[str],
        recipes: Optional[List[str]] = None,
        lr: float = 0.1,
        prior_var: float = 1.0,
    ) -> None:
        self.spices = list(spices)
        self.recipes: List[str] = list(recipes) if recipes else []
        self._lr = lr
        self._prior_var = prior_var

        # w[human][spice]: logistic regression weight, init 0 (no preference)
        self._w: Dict[str, Dict[str, float]] = {}
        # n[human][spice]: observation count for annealing
        self._n: Dict[str, Dict[str, int]] = {}

        self._mood_posterior: Dict[str, np.ndarray] = {}
        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._current_recipe: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_human(self, human_id: str) -> None:
        if human_id not in self._w:
            self._w[human_id] = {s: 0.0 for s in self.spices}
            self._n[human_id] = {s: 0 for s in self.spices}
            self._mood_posterior[human_id] = _NEUTRAL_POSTERIOR.copy()
            self._episode_data[human_id] = []
            self._current_recipe[human_id] = None

    def register_recipe(self, human_id: str, recipe_name: str) -> None:
        self.register_human(human_id)
        if recipe_name not in self.recipes:
            self.recipes.append(recipe_name)

    def _ensure_registered(self, human_id: str, recipe_name: str) -> None:
        self.register_human(human_id)
        self.register_recipe(human_id, recipe_name)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _p_human(self, human_id: str, spice: str) -> float:
        w = self._w.get(human_id, {}).get(spice, 0.0)
        return self._sigmoid(w)

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
        """Return logistic weight as a phi-equivalent."""
        return self._w.get(human_id, {}).get(spice, 0.0)

    def get_phi_var(self, human_id: str, recipe_name: str, spice: str) -> float:
        """CBTL classifier has no uncertainty estimate; return a small constant."""
        return 1e-4

    def get_phi_entropy(self, human_id: str, recipe_name: str, spice: str) -> float:
        """No meaningful exploration signal in CBTL — return 0."""
        return 0.0

    def get_running_psi(self, human_id: str) -> float:
        """No psi in CBTL classifier."""
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
        """
        Online logistic regression update following CBTL supervised classification.

        Soft label: y = (satisfaction + 1) / 2  maps [-1,+1] -> [0,1].
        When actor=human: y is the probability that human was the right choice.
        When actor=robot: (1 - y) is the probability human was the right choice.
        Gradient step: w += lr * (y_human - p_human) with L2 regularization.
        """
        for actor, spice, satisfaction in self._episode_data.get(human_id, []):
            if spice not in self._w.get(human_id, {}):
                continue
            y_sat = (satisfaction + 1.0) / 2.0  # [0, 1]
            # Soft label: probability that human was the correct assignment
            y_human = y_sat if actor == "human" else (1.0 - y_sat)
            p = self._p_human(human_id, spice)
            # Gradient of BCE + L2 prior w.r.t. w
            grad = (y_human - p) - self._w[human_id][spice] / self._prior_var
            # Step-size annealing: lr / sqrt(n+1) for stability
            n = self._n[human_id][spice]
            step = self._lr / math.sqrt(n + 1)
            self._w[human_id][spice] += step * grad
            self._n[human_id][spice] += 1
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
        """No running psi in CBTL classifier; no-op."""

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = {"w": self._w, "n": self._n}
        with open(path / "cbtl_classifier.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "cbtl_classifier.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._w = state["w"]
        self._n = state["n"]
