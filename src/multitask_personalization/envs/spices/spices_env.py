"""A simple spices environment for rapid testing of co-planning"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, TYPE_CHECKING
import logging
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from tomsutils.spaces import EnumSpace
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from multitask_personalization.structs import PublicSceneSpec
from multitask_personalization.envs.spices.spices_config import DEFAULT_CONFIG
from multitask_personalization.envs.spices.spices_hbm import MOODS

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

    def build_dag(self) -> nx.DiGraph:
        """Build a directed acyclic graph (DAG) from the recipe."""
        dag = nx.DiGraph()
        for spice in self.spices:
            dag.add_node(spice)
        for spice, predecessors in self.predecessors.items():
            for predecessor in predecessors:
                dag.add_edge(predecessor, spice)
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("The recipe is not a valid DAG")
        return dag

    def get_topological_sort(self) -> list[str]:
        """Get a topological sort of the recipe."""
        return list(nx.topological_sort(self.build_dag()))

    def layers(self, dag: nx.DiGraph | None = None) -> list[list[str]]:
        """
        Get the topological layers of the recipe.

        Accepts an already-built DAG to avoid redundant construction when the
        caller already holds one.
        """
        G = dag if dag is not None else self.build_dag()
        layers: list[list[str]] = []
        current_layer = [n for n in G.nodes if G.in_degree(n) == 0]
        visited = set(current_layer)
        while current_layer:
            layers.append(current_layer)
            next_layer = []
            for node in current_layer:
                for succ in G.successors(node):
                    if all(pred in visited for pred in G.predecessors(succ)):
                        next_layer.append(succ)
            visited.update(next_layer)
            current_layer = next_layer
        return layers

    def visualize_dag(self) -> None:
        """
        Visualize the recipe DAG using a multipartite layout based on its layers.
        Each layer corresponds to a topological level.
        """
        G = self.build_dag()
        layers = self.layers(G)  # reuse the DAG already built above

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


# --- MOOD MODEL ---
class MoodModel:
    """
    Handles sampling the current episode's mood.

    Moods and prior probabilities are taken from the shared MOODS constant and
    MoodConfig so that the generative distribution and the HBM's inference prior
    are always in sync.
    """
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.current_mood: str | None = None

    def sample_mood(self) -> str:
        """Sample a mood for the current episode."""
        mood = self.rng.choice(MOODS, p=DEFAULT_CONFIG.mood.mood_prior)
        self.current_mood = str(mood)
        return self.current_mood


# --- SPICES ENVIRONMENT SPECIFICATIONS ---
@dataclass(frozen=True)
class SpiceSceneSpec(PublicSceneSpec):
    """A scene specification for the spices environment."""
    recipe: RecipeSpec

@dataclass(frozen=False)  # Mutable to allow setting preferences from HBM
class SpiceHiddenSpec:
    """Hidden Human preference over who should add each spice."""
    preferred_actor: dict[str, str]  # {"Spice": "Actor"}
    hidden_hbm: "HierarchicalPreferenceModel | None" = None

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
        eval_mode: bool = False,
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

        # Build the DAG once and reuse it for all derived structures.
        self._dag = self.scene_spec.recipe.build_dag()
        self._topo_order = list(nx.topological_sort(self._dag))
        self._layers = self.scene_spec.recipe.layers(self._dag)

        # Cached O(1) lookup structures (constant for the lifetime of the env).
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
        self._current_feasible: list[str] = []  # cached by _pick_current_spice
        self._last_actor: str | None = None
        self._last_satisfaction: float = 0.0
        self._satisfaction_history: list[float] = []
        self._action_history: list[SpiceAction] = []

        self.eval_mode = eval_mode
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

        # Create hidden preferences.
        if self.eval_mode or self._hidden_spec is None:
            self.__randomize_hidden_preferences()
        elif self._hidden_spec.hidden_hbm is not None:
            self.__sample_preferences_from_hbm()

        # Reset episode state.
        self._t = 0
        self._added = []
        self._last_actor = None
        self._last_satisfaction = 0.0
        self._satisfaction_history = []
        self._action_history = []
        self._current_mood: str | None = self._mood_model.sample_mood()
        self._current_spice = self._pick_current_spice()  # also populates _current_feasible

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
        if np.isclose(status, 1):
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

        # Advance to the next spice (also refreshes _current_feasible).
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

    def _compute_mood_bias(self, mood: str, actor: str) -> float:
        """
        Compute mood bias that overrides base preferences.

        Mood semantics:
            "all_self"  — human wants to do everything; satisfaction only when human acts.
            "none_self" — human doesn't want to do anything; satisfaction only when robot acts.
            "neutral"   — no mood override; base preferences apply.

        With phi = ±base_satisfaction_bias, bias_strength = base_satisfaction_bias * 2.0
        ensures the mood signal always overrides the base preference.
        """
        bias_strength = self._base_satisfaction_bias * 2.0
        if mood == "all_self":
            return +bias_strength if actor == "human" else -bias_strength
        elif mood == "none_self":
            return -bias_strength if actor == "human" else +bias_strength
        else:  # "neutral"
            return 0.0

    def _compute_satisfaction(self, preferred: str) -> float:
        """
        Compute satisfaction for the most recently assigned spice.

        Returns a continuous value in [-1, +1]:
            +1.0 — maximum positive satisfaction
             0.0 — neutral
            -1.0 — maximum negative satisfaction

        The generative model mirrors the HBM's likelihood:
            logit = sign(actor) * phi + mood_bias
            p     = sigmoid(logit)
            sat   ~ Beta(p*kappa+1, (1-p)*kappa+1) rescaled to [-1, +1]
        """
        phi = (
            +self._base_satisfaction_bias
            if preferred == self._last_actor
            else -self._base_satisfaction_bias
        )
        mood_adj = self._compute_mood_bias(self._current_mood, self._last_actor)
        logit = phi + mood_adj

        p = 1.0 / (1.0 + np.exp(-logit))
        alpha = p * self._satisfaction_kappa + 1.0
        beta = (1.0 - p) * self._satisfaction_kappa + 1.0
        p_sampled = self._rng.beta(alpha, beta)
        return float(np.clip(2.0 * p_sampled - 1.0, -1.0, 1.0))

    def _get_info(self, robot_indicated_done: bool = False) -> dict[str, Any]:
        # Empty case: no spice has been assigned yet this episode.
        if not self._added:
            return {
                "robot_indicated_done": robot_indicated_done,
                "satisfaction": 0.0,
                "preferred_actor": None,
                "current_spice": self._current_spice,
                "last_spice": None,
                "last_actor": None,
                "feasible_next": tuple(self._current_feasible),
                "average_satisfaction": 0.0,
                "satisfaction_history": [],
                "satisfaction_variance": 0.0,
                "action_history": list(self._action_history),
                "mood": self._current_mood,
                "recipe_name": self.scene_spec.recipe.name,
            }

        # Satisfaction was already computed in step() and appended to history.
        last_spice = self._added[-1]
        preferred = self._hidden_spec.preferred_actor[last_spice]

        return {
            "robot_indicated_done": robot_indicated_done,
            "satisfaction": self._last_satisfaction,
            "preferred_actor": preferred,
            "current_spice": self._current_spice,
            "last_spice": last_spice,
            "recipe_name": self.scene_spec.recipe.name,
            "last_actor": self._last_actor,
            "satisfaction_history": list(self._satisfaction_history),
            "average_satisfaction": float(np.mean(self._satisfaction_history)),
            "satisfaction_variance": float(np.var(self._satisfaction_history)),
            "feasible_next": tuple(self._current_feasible),
            "action_history": list(self._action_history),
            "mood": self._current_mood,
        }

    def __feasible_next(self) -> list[str]:
        """Return feasible spices sorted by (layer, topological position)."""
        added = set(self._added)
        feasible = [
            node
            for node in self._dag.nodes
            if node not in added
            and set(self._dag.predecessors(node)).issubset(added)
        ]
        feasible.sort(key=lambda s: (
            self._layer_index.get(s, float("inf")),
            self._topo_index.get(s, float("inf")),
        ))
        return feasible

    def _pick_current_spice(self) -> str | None:
        """
        Return the next spice to assign (first in the feasible ordering).

        Caches the full feasible list in self._current_feasible so that
        _get_state() and _get_info() can reuse it without a second traversal.
        """
        self._current_feasible = self.__feasible_next()
        return self._current_feasible[0] if self._current_feasible else None

    def __randomize_hidden_preferences(self) -> None:
        """Randomize the hidden preferences uniformly over actors."""
        preferences = {
            spice: self._rng.choice(["human", "robot"])
            for spice in self.scene_spec.recipe.spices
        }
        self._hidden_spec = SpiceHiddenSpec(preferred_actor=preferences)

    def __sample_preferences_from_hbm(self) -> None:
        """
        Sample episode preferences from the hidden HBM for this recipe.

        For each spice, samples phi ~ N(theta_mean, theta_var + sigma_r²), which
        marginalises out the posterior uncertainty in the human-level preference theta.
        A positive phi means the human prefers to add the spice themselves.
        """
        hidden_hbm = self._hidden_spec.hidden_hbm
        preferences = {}

        for spice in self.scene_spec.recipe.spices:
            if spice in hidden_hbm.theta_mean:
                theta_mean = hidden_hbm.theta_mean[spice]
                theta_var = hidden_hbm.theta_var.get(
                    spice, hidden_hbm.sigma_h**2 + hidden_hbm.sigma_r**2
                )
                # Marginalise theta uncertainty: phi ~ N(theta_mean, theta_var + sigma_r²)
                phi_std = float(np.sqrt(theta_var + hidden_hbm.sigma_r**2))
                phi = self._rng.normal(theta_mean, phi_std)
                p_human = 1.0 / (1.0 + np.exp(-phi))
                preferred_actor = "human" if self._rng.random() < p_human else "robot"
            else:
                preferred_actor = self._rng.choice(["human", "robot"])

            preferences[spice] = preferred_actor

        self._hidden_spec.preferred_actor = preferences
