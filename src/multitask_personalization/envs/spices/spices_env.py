"""A simple spices environment for rapid testing of co-planning"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, TYPE_CHECKING
import logging
import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gymnasium.core import RenderFrame
from tomsutils.spaces import EnumSpace
from multitask_personalization.structs import PublicSceneSpec
from multitask_personalization.envs.spices.config.spices_config import DEFAULT_CONFIG
from multitask_personalization.envs.spices.spices_hbm import DEFAULT_HUMAN, MoodModel

if TYPE_CHECKING:
    from multitask_personalization.envs.spices.spices_hbm import HierarchicalPreferenceModel

# --- PROFILE / RECIPE SPECIFICATIONS ---
@dataclass(frozen=True)
class ProfileSpec:
    name: str
    recipes: tuple[str, ...]

@dataclass(frozen=True)
class RecipeSpec:
    name: str
    spices: tuple[str, ...]
    # Partial order: predecessor map: for spice s, all spices that must be added before s.
    predecessors: dict[str, tuple[str, ...]]

    def get_topological_sort(self) -> list[str]:
        """Kahn's algorithm for topological sort. Raises ValueError if graph has a cycle."""
        pred_count = {s: len(self.predecessors.get(s, ())) for s in self.spices}
        frontier = [s for s in self.spices if pred_count[s] == 0]
        order: list[str] = []
        while frontier:
            node = frontier.pop()
            order.append(node)
            for s in self.spices:
                if node in self.predecessors.get(s, ()):
                    pred_count[s] -= 1
                    if pred_count[s] == 0:
                        frontier.append(s)
        if len(order) != len(self.spices):
            raise ValueError("The recipe is not a valid DAG (cycle detected)")
        return order

    def layers(self) -> list[list[str]]:
        """Get the topological layers of the recipe."""
        layers: list[list[str]] = []
        visited: set[str] = set()
        remaining = set(self.spices)
        while remaining:
            current_layer = [s for s in remaining if set(self.predecessors.get(s, ())).issubset(visited)]
            if not current_layer:
                raise ValueError("The recipe is not a valid DAG (cycle detected)")
            layers.append(current_layer)
            visited.update(current_layer)
            remaining -= set(current_layer)
        return layers

    def visualize_dag(self) -> None:
        """
        Visualize the recipe DAG using a multipartite layout based on its layers.
        Each layer corresponds to a topological level. Requires networkx and matplotlib.
        """
        G = nx.DiGraph()
        for spice in self.spices:
            G.add_node(spice)
        for spice, preds in self.predecessors.items():
            for p in preds:
                G.add_edge(p, spice)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("The recipe is not a valid DAG")

        layers = self.layers()
        layer_index = {s: i for i, layer in enumerate(layers) for s in layer}
        nx.set_node_attributes(G, layer_index, "layer")

        num_layers = len(layers)
        cmap = cm.get_cmap("cividis", num_layers)
        colors = [cmap(layer_index[n]) for n in G.nodes]

        pos = nx.multipartite_layout(G, subset_key="layer", align="vertical", scale=3.0)
        label_map = {node: node.replace("_", " ").title() for node in G.nodes}

        plt.figure(figsize=(10, 4), dpi=150)
        nx.draw(
            G, pos,
            labels=label_map,
            with_labels=True,
            node_size=2750,
            node_color=colors,
            edgecolors="black",
            linewidths=0.5,
            font_size=9,
            font_weight="bold",
            arrowsize=18,
            arrowstyle="-|>",
            width=1,
            connectionstyle="arc3,rad=0.05",
            font_color="white",
            font_family="Helvetica",
        )
        plt.title(f"Recipe DAG: {self.name}", fontsize=12, pad=10, fontweight="bold")
        plt.axis("off")
        plt.tight_layout(pad=1.0)
        plt.show()

@dataclass(frozen=True)
class SpiceSceneSpec(PublicSceneSpec):
    """A scene specification for the spices environment."""
    recipe: RecipeSpec

@dataclass(frozen=False)  # Mutable to allow setting preferences from HBM
class SpiceHiddenSpec:
    """Hidden Human preference over who should add each spice."""
    preferred_actor: dict[str, str]  # {"Spice": "Actor"}
    hidden_hbm: "HierarchicalPreferenceModel | None" = None
    force_neutral_mood: bool = False  # If True, all episodes use neutral mood (training mode)

@dataclass(frozen=True)
class SpiceState:
    """The current state of the spices environment."""
    time: int
    added_spices: tuple[str, ...]
    remaining_spices: tuple[str, ...]
    feasible_next: tuple[str, ...]
    current_spice: str | None

# --- ACTION SPECIFICATIONS ---
"""
Action: (flag, payload)
flag = 0: add, flag = 1: done
payload: the name of the actor assigned to add the next spice (only if flag = 0)
"""
SpiceAction: TypeAlias = tuple[int, str | None]

# --- SPICES ENVIRONMENT ---
class SpiceEnv(gym.Env[SpiceState, SpiceAction]):
    """
    A symbolic environment where actors can follow a recipe to add spices to a pot.
    The human actor:
        - has a hidden preference about which spices they prefer to add to the pot.
        - has a mood that affects their satisfaction on a given episode.
    The robot actor:
        - is a perfect planner that can add spices to the pot in the optimal order.
    """

    def __init__(
        self,
        scene_spec: SpiceSceneSpec,
        hidden_spec: SpiceHiddenSpec,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self._hidden_spec = hidden_spec
        self.scene_spec = scene_spec
        self.action_space = gym.spaces.OneOf(
            (
                EnumSpace(["human", "robot"]),
                EnumSpace([None]),
            )
        )
        self._mood_model = MoodModel(self._rng)

        # Precompute topological order and layers
        self._topo_order = self.scene_spec.recipe.get_topological_sort()
        self._layers = self.scene_spec.recipe.layers()

        self._layer_index: dict[str, int] = {
            spice: idx
            for idx, layer in enumerate(self._layers)
            for spice in layer
        }
        self._topo_index: dict[str, int] = {s: i for i, s in enumerate(self._topo_order)}

        # Satisfaction parameters from config.
        self._base_satisfaction_bias: float = DEFAULT_CONFIG.satisfaction.base_satisfaction_bias
        self._satisfaction_kappa: float = DEFAULT_CONFIG.satisfaction.satisfaction_beta_kappa

        self._t = 0
        self._added: list[str] = []
        self._current_spice: str | None = None
        self._current_feasible: list[str] = []
        self._last_actor: str | None = None
        self._last_satisfaction: float = 0.0
        self._satisfaction_history: list[float] = []
        self._action_history: list[SpiceAction] = []
        # Stage 2: true session offset sampled once per episode (replaces per-step mood_adj)
        self._current_psi_true: float = 0.0

        self.verbose = verbose

        if verbose:
            self._log_recipe()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[SpiceState, dict[str, Any]]:

        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample hidden preferences from HBM when available; otherwise use fixed preferences.
        if self._hidden_spec.hidden_hbm is not None:
            self._hidden_spec.preferred_actor = (
                self._hidden_spec.hidden_hbm.sample_episode_preferences(
                    list(self.scene_spec.recipe.spices),
                    self._rng,
                )
            )

        # Reset episode state.
        self._t = 0
        self._added = []
        self._last_actor = None
        self._last_satisfaction = 0.0
        self._satisfaction_history = []
        self._action_history = []

        if self._hidden_spec.force_neutral_mood:
            self._current_mood = "neutral"
            self._mood_model.current_mood = "neutral"
        else:
            self._current_mood = self._mood_model.sample_mood()

        # Stage 2: sample true session psi once per episode from mood-conditioned distribution.
        # The HBM infers a scalar psi to explain within-episode deviations from phi.
        self._current_psi_true = self._sample_psi_from_mood(self._current_mood)

        self._current_spice = self._pick_current_spice()

        if self.verbose:
            logging.info("[SpiceEnv] Resetting environment")
            logging.info(f"[Mood] Current mood: {self._current_mood}")
            self._log_hidden_prefs()

        return self._get_state(), self._get_info()

    def step(
        self, action: SpiceAction
    ) -> tuple[SpiceState, float, bool, bool, dict[str, Any]]:
        status, payload = action

        self._action_history.append(action)

        # Robot indicated done without adding a spice.
        if status:
            info = self._get_info(robot_indicated_done=True)
            return self._get_state(), 0.0, True, False, info

        # Add the assigned spice.
        assert payload is not None, "Payload is not set"
        assert payload in {"human", "robot"}, "Invalid actor"

        self._last_actor = str(payload)
        self._added.append(self._current_spice)
        self._t += 1

        # Compute and record satisfaction for the spice just assigned.
        last_spice = self._added[-1]
        preferred = self._hidden_spec.preferred_actor[last_spice]
        self._last_satisfaction = self._compute_satisfaction(preferred)
        self._satisfaction_history.append(self._last_satisfaction)

        # Advance to the next spice
        self._current_spice = self._pick_current_spice()

        terminated = not self._current_spice
        info = self._get_info(robot_indicated_done=terminated)

        if self.verbose:
            logging.info(f"[Step {self._t - 1}] Assign {info['last_spice']} → {info['last_actor']}")

        return self._get_state(), 0.0, terminated, False, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    # ---------------- UTILITY FUNCTIONS ----------------
    def _log_recipe(self) -> None:
        recipe = self.scene_spec.recipe
        logging.info(f"\n{'=' * 30} RECIPE {'=' * 30}\n")
        logging.info(f"Name: {recipe.name}\nSpices: {', '.join(recipe.spices)}\n")
        logging.info("Predecessors (must come before → spice):")
        for s in recipe.spices:
            preds = recipe.predecessors.get(s, ())
            prefix = ', '.join(preds) if preds else "(None)"
            logging.info(f"  {prefix} → {s}")

    def _log_hidden_prefs(self) -> None:
        logging.info(f"\n{'=' * 30} HIDDEN PREFERENCES {'=' * 30}")
        actors: dict[str, list[str]] = {}
        for spice, actor in self._hidden_spec.preferred_actor.items():
            actors.setdefault(actor, []).append(spice)
        for actor, spices in actors.items():
            logging.info(f"  {actor}: {', '.join(spices)}")
        logging.info(f"{'=' * 60}\n")

    def _get_state(self) -> SpiceState:
        return SpiceState(
            time=self._t,
            added_spices=tuple(self._added),
            remaining_spices=tuple(
                s for s in self.scene_spec.recipe.spices if s not in self._added
            ),
            feasible_next=tuple(self._current_feasible),
            current_spice=self._current_spice,
        )

    def _compute_satisfaction(self, preferred: str) -> float:
        """
        Compute satisfaction for the most recently assigned spice.

        Returns a continuous value in [-1, +1]:
            +1.0 — maximum positive satisfaction
             0.0 — neutral
            -1.0 — maximum negative satisfaction

        The generative model mirrors the HBM's Stage 2 likelihood:
            phi_latent = pref_sign * phi_mag
            phi_mag    = |hidden_theta(spice)| when available, else base_satisfaction_bias
            logit      = actor_sign * (phi_latent + psi_true)
            expected   = tanh(logit / T) where T=satisfaction_logit_temperature
            p        = (expected+1)/2 ← map [-1,+1] → [0,1] for Beta params
            sat      ~ Beta(p*kappa+1, (1-p)*kappa+1) rescaled to [-1, +1]

        psi_true is sampled once per episode from a mood-conditioned distribution
        (see _sample_psi_from_mood). This replaces the old per-step actor-dependent
        compute_mood_bias call, aligning the generative model with the HBM's scalar psi.
        """
        actor_sign = 1.0 if self._last_actor == "human" else -1.0
        pref_sign = 1.0 if preferred == "human" else -1.0
        phi_mag = self._base_satisfaction_bias
        # If a hidden HBM is available, use the true theta magnitude for this spice
        # so generated logits are calibrated to latent preference strength.
        if self._hidden_spec.hidden_hbm is not None and self._added:
            spice = self._added[-1]
            try:
                theta_val = self._hidden_spec.hidden_hbm.get_theta(DEFAULT_HUMAN, spice)
                phi_mag = max(abs(float(theta_val)), 1e-6)
            except Exception:
                phi_mag = self._base_satisfaction_bias
        phi_latent = pref_sign * phi_mag
        logit = actor_sign * (phi_latent + self._current_psi_true)

        temp = max(float(DEFAULT_CONFIG.satisfaction.satisfaction_logit_temperature), 1e-6)
        expected = float(np.tanh(logit / temp))
        p = (expected + 1.0) / 2.0
        alpha = p * self._satisfaction_kappa + 1.0
        beta = (1.0 - p) * self._satisfaction_kappa + 1.0
        p_sampled = self._rng.beta(alpha, beta)
        return float(np.clip(2.0 * p_sampled - 1.0, -1.0, 1.0))

    def _sample_psi_from_mood(self, mood: str) -> float:
        """
        Sample the true session offset psi_true from a mood-conditioned distribution.

        Maps the 3-category mood to a scalar psi_true sampled once per episode.
        The HBM's prior N(0, sigma_mood²) is centered at 0; non-neutral moods shift
        the mean to ±psi_true_mood_mean_abs (configurable, independent of base bias)
        with std psi_true_mood_std.

        Stage 2: replaces per-step compute_mood_bias. The HBM infers a latent scalar
        psi without knowing the mood category.
        """
        mean_abs = DEFAULT_CONFIG.mood.psi_true_mood_mean_abs
        std = DEFAULT_CONFIG.mood.psi_true_mood_std
        mood_means = {
            "all_self":  +mean_abs,
            "neutral":    0.0,
            "none_self": -mean_abs,
        }
        mean = mood_means.get(mood, 0.0)
        if mood == "neutral":
            std = float(DEFAULT_CONFIG.mood.psi_true_neutral_std)
        return float(self._rng.normal(mean, std))

    def _get_info(self, robot_indicated_done: bool = False) -> dict[str, Any]:
        info: dict[str, Any] = {
            "robot_indicated_done": robot_indicated_done,
            "current_spice": self._current_spice,
            "feasible_next": tuple(self._current_feasible),
            "action_history": list(self._action_history),
            "mood": self._current_mood,
            "recipe_name": self.scene_spec.recipe.name,
            "force_neutral_mood": self._hidden_spec.force_neutral_mood,
        }
        if not self._added:
            info.update({
                "satisfaction": 0.0,
                "preferred_actor": None,
                "last_spice": None,
                "last_actor": None,
                "average_satisfaction": 0.0,
                "satisfaction_history": [],
                "satisfaction_variance": 0.0,
            })
        else:
            last_spice = self._added[-1]
            info.update({
                "satisfaction": self._last_satisfaction,
                "preferred_actor": self._hidden_spec.preferred_actor[last_spice],
                "last_spice": last_spice,
                "last_actor": self._last_actor,
                "average_satisfaction": float(np.mean(self._satisfaction_history)),
                "satisfaction_history": list(self._satisfaction_history),
                "satisfaction_variance": float(np.var(self._satisfaction_history)),
            })
        return info

    def __feasible_next(self) -> list[str]:
        """Return feasible spices sorted by (layer, topological position)."""
        added = set(self._added)
        recipe = self.scene_spec.recipe
        feasible = [
            s
            for s in recipe.spices
            if s not in added
            and set(recipe.predecessors.get(s, ())).issubset(added)
        ]
        feasible.sort(key=lambda s: (
            self._layer_index.get(s, float("inf")),
            self._topo_index.get(s, float("inf")),
        ))
        return feasible

    def _pick_current_spice(self) -> str | None:
        """
        Return the next spice to assign (first in the feasible ordering).
        """
        self._current_feasible = self.__feasible_next()
        return self._current_feasible[0] if self._current_feasible else None
