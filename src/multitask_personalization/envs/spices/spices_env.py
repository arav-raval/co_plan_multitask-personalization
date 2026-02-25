"""A simple spices environment for rapid testing of co-planning"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, Iterable, TYPE_CHECKING
import logging
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from tomsutils.spaces import EnumSpace
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from multitask_personalization.structs import PublicSceneSpec

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

        # Add nodes
        for spice in self.spices:
            dag.add_node(spice)

        # Add edges
        for spice, predecessors in self.predecessors.items():
            for predecessor in predecessors:
                dag.add_edge(predecessor, spice)

        # Check if the DAG is valid
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("The recipe is not a valid DAG")

        return dag
    
    def get_topological_sort(self) -> list[str]:
        """Get a topological sort of the recipe."""
        return list(nx.topological_sort(self.build_dag()))

    def layers(self) -> list[list[str]]:
        """Get the layers of the recipe."""
        G = self.build_dag()
        layers = []

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
        layers = self.layers() 

        layer_index = {}
        for i, layer in enumerate(layers):
            for s in layer:
                layer_index[s] = i
        nx.set_node_attributes(G, layer_index, "layer")

        # Build a color map for layers
        num_layers = len(layers)
        cmap = cm.get_cmap("cividis", num_layers)
        colors = [cmap(layer_index[n]) for n in G.nodes]

        # Layout: place layers from left to right for clearer reading
        pos = nx.multipartite_layout(
            G,
            subset_key="layer",
            align="vertical",  # vertical alignment => layers spread on the x-axis (left→right)
            scale=3.0,
        )

        # Human-friendly labels for poster-quality visuals
        label_map = {node: node.replace("_", " ").title() for node in G.nodes}

        plt.figure(figsize=(10, 4), dpi=150)
        nx.draw(
            G,
            pos,
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

# --- MOOD SPECIFICATIONS ---
@dataclass(frozen=True)
class MoodSpec:
    """Defines the mood categories."""
    # TODO: extend to more moods / non-categorical moods
    moods: tuple[str, ...] = ("all_self", "neutral", "none_self")
    # Skewed priors to favor neutral mood (80% neutral, 10% each for all_self/none_self)
    # This ensures more episodes contribute to preference learning
    # Increased from 60% to 80% to allow faster convergence
    priors: tuple[float, ...] = (0.1, 0.8, 0.1)  # (all_self, neutral, none_self)

class MoodModel:
    """Handles sampling and updating the current day's mood."""
    def __init__(self, spec: MoodSpec, rng: np.random.Generator, mood_bias: dict[str, dict[str, float]] = None):
        self.spec = spec
        self.rng = rng
        self.current_mood: str | None = None

    def sample_mood(self) -> str:
        """Sample a mood from the mood specification."""
        self.current_mood = self.rng.choice(self.spec.moods, p=self.spec.priors)
        return self.current_mood
    
# --- SPICES ENVIRONMENT SPECIFICATIONS ---
@dataclass(frozen=True)
class SpiceSceneSpec(PublicSceneSpec):
    """A scene specification for the spices environment."""
    recipe: RecipeSpec

@dataclass(frozen=False)  # Changed to mutable to allow setting preferences from HBM
class SpiceHiddenSpec:
    """Hidden Human preference over who should add each spice."""
    preferred_actor: dict[str, str] # {"Spice":  "Actor"}
    hidden_hbm: "HierarchicalPreferenceModel | None" = None  # Optional hidden HBM for generating preferences

@dataclass(frozen=True)
class SpiceState:
    """The current state of the spices environment."""
    time: int
    added_spices: tuple[str, ...] # spices already added 
    remaining_spices: tuple[str, ...] # spices not yet added 
    feasible_next: tuple[str, ...] # spices that can be added next
    current_spice: str | None # the current spice to add

# --- ACTION SPECIFICATIONS ---
"""
Action: (flag, payload)
flag = 0: add, flag = 1: done
payload: the name of the actor that is assigned to add the next spice (only if flag = 0)
"""
SpiceAction: TypeAlias = tuple[int, str | None] # (flag: 0=add, 1=done, payload: "Add <spice>" or None)

# --- SPICES ENVIRONMENT ---
class SpiceEnv(gym.Env[SpiceState, SpiceAction]):
    """
    A symbolic environment where actors can follow a recipe to add spices to a pot.
    The human actor:
        - has a hideen preference about which spices they prefer to add to the pot.
        - has a mood that affects their preferences on a given episode
    The robot actor:
        - is a perfect planner that can add spices to the pot in the optimal order
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
        self._mood_spec = MoodSpec()
        self._mood_model = MoodModel(self._mood_spec, self._rng)
        self._dag = self.scene_spec.recipe.build_dag()
        self._topo_order = self.scene_spec.recipe.get_topological_sort()
        self._layers = self.scene_spec.recipe.layers()
        self._base_satisfaction_bias = 3.0

        self._t = 0
        self._added: list[str] = []
        self._current_spice: str | None = None
        self._last_actor: str | None = None
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

        # Create hidden preferences if none provided
        if self.eval_mode or self._hidden_spec is None:
            self.__randomize_hidden_preferences()
        elif self._hidden_spec.hidden_hbm is not None:
            # Use hidden HBM to sample preferences for this recipe
            self.__sample_preferences_from_hbm()

        # Reset state
        self._t = 0
        self._added: list[str] = []
        self._last_actor: str | None = None
        self._satisfaction_history: list[float] = []
        self._current_spice: str | None = self._pick_current_spice()
        self._action_history: list[SpiceAction] = []
        self._current_mood: str | None = self._mood_model.sample_mood()

        if self.verbose:
            logging.info("[SpiceEnv] Resetting environment")
            logging.info(f"[Mood] Current mood: {self._current_mood}")
            self._log_hidden_prefs()
            #self.scene_spec.recipe.visualize_dag()

        return self._get_state(), self._get_info()

    def step(
        self, action: SpiceAction
    ) -> tuple[SpiceState, float, bool, bool, dict[str, Any]]:
        status, payload = action

        self._action_history.append(action)

        # Done with all spices (only raises on repeated calls to step after the recipe is complete)
        if np.isclose(status, 1):
            info = self._get_info(robot_indicated_done=True)
            terminated = True            
            return self._get_state(), 0.0, terminated, False, info
        
        # Add spice and advance forward
        assert payload is not None, "Payload is not set"
        assert payload in {"human", "robot"}, "Invalid actor"

        self._last_actor = str(payload)
        self._added.append(self._current_spice)
        self._t += 1
        self._current_spice = self._pick_current_spice()

        # Check if the recipe is completed
        if not self._current_spice or self._current_spice == '':
            info = self._get_info(robot_indicated_done=True)
            terminated = True

            if self.verbose:
                logging.info(f"[Step {self._t - 1}] Assign {info['last_spice']} → {info['last_actor']} ")

            return self._get_state(), 0.0, terminated, False, info

        # Recipe is still ongoing
        info = self._get_info(robot_indicated_done=False)

        if self.verbose:
            logging.info(f"[Step {self._t - 1}] Assign {info['last_spice']} → {info['last_actor']} ")

        return self._get_state(), 0.0, False, False, info
    
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    # ---------------- UTIILTY FUNCTIONS ----------------
    def _log_recipe(self) -> None:
        recipe = self.scene_spec.recipe
        logging.info(f"\n{'=' * 30} RECIPE {'=' * 30}\n")
        logging.info(f"Name: {recipe.name}\n"
                     f"Spices: {', '.join(spice for spice in recipe.spices)}\n")
        logging.info(f"Predecessors (must come before -> spice):\n")
        for s in recipe.spices:
            preds = recipe.predecessors.get(s, ())
            if preds:
                logging.info(f"  {', '.join(preds)} → {s}")
            else:
                logging.info(f"  (None) → {s}")

    def _log_hidden_prefs(self) -> None:
        logging.info(f"\n{'=' * 30} HIDDEN PREFERENCES {'=' * 30}")
        actors = {}
        for spice, actor in self._hidden_spec.preferred_actor.items():
            actors[actor] = actors.get(actor, []) + [spice]
        for actor, spices in actors.items():
            logging.info(f"  {actor}: {', '.join(spices)}")
        logging.info(f"{'=' * 60}\n")

    def _get_state(self) -> SpiceState:
        feasible_list = list(self.__feasible_next())
        return SpiceState(
            time=self._t,
            added_spices=tuple(self._added),
            remaining_spices=tuple(spice for spice in self.scene_spec.recipe.spices if spice not in self._added),
            feasible_next=tuple(feasible_list),
            current_spice=self._current_spice or (feasible_list[0] if feasible_list else ""),
        )

    def _compute_mood_bias(self, mood: str | float, actor: str) -> float:
        """
        Compute mood bias that OVERRIDES base preferences.
        
        Mood semantics:
        - "all_self" (or +1.0): Human wants to do everything (satisfaction ONLY when human acts)
        - "none_self" (or -1.0): Human doesn't want to do anything (satisfaction ONLY when robot acts)
        - "neutral" (or 0.0): No mood override, base preferences apply
        
        Args:
            mood: Categorical mood string or continuous value in [-1.0, +1.0]
            actor: "human" or "robot"
        
        Returns:
            Mood bias value. With phi = ±base_satisfaction_bias, bias_strength = base_satisfaction_bias * 2.0 ensures mood overrides.
        """
        bias_strength = self._base_satisfaction_bias * 2.0
        
        # Handle continuous mood values
        if isinstance(mood, (int, float)):
            mood_value = float(mood)
            mood_value = np.clip(mood_value, -1.0, 1.0)
            
            # Map continuous mood to bias:
            if actor == "human":
                return mood_value * bias_strength
            else:  # robot
                return -mood_value * bias_strength
        
        # Handle categorical moods
        if mood == "all_self":
            return +bias_strength if actor == "human" else -bias_strength
        elif mood == "none_self":
            return -bias_strength if actor == "human" else +bias_strength
        elif mood == "neutral":
            return 0.0
        else:
            return 0.0

    def _compute_satisfaction(self, last_spice: str, preferred: str) -> float:
        """Compute satisfaction based on preference and mood.
        
        Returns continuous satisfaction in range [-1, +1]:
        - +1.0: Maximum positive satisfaction
        - 0.0: Neutral satisfaction
        - -1.0: Maximum negative satisfaction
        
        Mood overrides base preferences:
        - "all_self": satisfaction ONLY when human acts
        - "none_self": satisfaction ONLY when robot acts
        """        
        # Base preference signal (can be overridden by mood)
        phi = +self._base_satisfaction_bias if preferred == self._last_actor else -self._base_satisfaction_bias
        
        # Mood bias
        mood_adj = self._compute_mood_bias(self._current_mood, self._last_actor)
        
        # Combine
        logit = phi + mood_adj
        
        # Probabilistic satisfaction: p ∈ [0, 1]
        p = 1 / (1 + np.exp(-logit))
        
        # Map probability to continuous satisfaction in [-1, +1]
        # p = 0.0 → satisfaction = -1.0
        # p = 0.5 → satisfaction = 0.0
        # p = 1.0 → satisfaction = +1.0
        # Add some noise to make it realistic (not just deterministic mapping)
        # Sample from a Beta distribution centered at p, scaled to [-1, +1]
        # Use concentration parameters that give reasonable variance
        # Import config here to avoid circular import
        from .spices_config import DEFAULT_CONFIG
        kappa = DEFAULT_CONFIG.satisfaction.satisfaction_beta_kappa
        alpha = p * kappa + 1.0
        beta = (1.0 - p) * kappa + 1.0
        p_sampled = self._rng.beta(alpha, beta)
        satisfaction = 2.0 * p_sampled - 1.0  # Map [0, 1] → [-1, +1]
        
        # Clamp to ensure we stay in [-1, +1] range
        satisfaction = np.clip(satisfaction, -1.0, 1.0)

        # if self.verbose:
        #     logging.info(f"[Satisfaction] Base preference: {phi}, Mood bias: {mood_adj}, p: {p:.6f}, Satisfaction: {satisfaction:.3f}")
        
        return float(satisfaction)

    def _get_info(self, robot_indicated_done: bool = False) -> dict[str, Any]:
        # Empty case
        if not self._added:
            return {
                "robot_indicated_done": robot_indicated_done,
                "satisfaction": 0.0,
                "preferred_actor": None,
                "current_spice": self._current_spice,
                "last_spice": None,
                "last_actor": None,
                "feasible_next": tuple(self.__feasible_next()),
                "average_satisfaction": 0.0,
                "action_history": self._action_history,
                "mood": self._current_mood,
                "recipe_name": self.scene_spec.recipe.name,
            }

        # Compute satisfaction
        last_spice = self._added[-1]
        preferred = self._hidden_spec.preferred_actor[last_spice]

        satisfaction = self._compute_satisfaction(last_spice, preferred)
        self._satisfaction_history.append(satisfaction)

        return {
            "robot_indicated_done": robot_indicated_done,
            "satisfaction": satisfaction,
            "preferred_actor": preferred,
            "current_spice": self._current_spice,
            "last_spice": last_spice,
            "recipe_name": self.scene_spec.recipe.name,  # Add recipe name for HBM
            "last_actor": self._last_actor,
            "satisfaction_history": self._satisfaction_history,
            "average_satisfaction": np.mean(self._satisfaction_history),
            "satisfaction_variance": np.var(self._satisfaction_history),
            "feasible_next": tuple(self.__feasible_next()),
            "action_history": self._action_history,
            "mood": self._current_mood,
        }

    def __feasible_next(self) -> Iterable[str]:
        """Return feasible spices ordered by layer (then topological order)."""
        dag = self._dag
        added = set(self._added)

        feasible: list[str] = []
        for node in dag.nodes:
            if node in added:
                continue
            preds = set(dag.predecessors(node))
            if preds.issubset(added):
                feasible.append(node)

        # Build layer index: spice -> layer number
        layer_index: dict[str, int] = {}
        for idx, layer in enumerate(self._layers):
            for spice in layer:
                layer_index[spice] = idx

        # Sort feasible nodes by (layer, topo order) so earlier layers come first
        feasible.sort(
            key=lambda s: (
                layer_index.get(s, float("inf")),
                self._topo_order.index(s),
            )
        )

        return feasible

    def _pick_current_spice(self) -> str | None:
        """Deterministically pick among the feasible spices to add next."""
        feasible = list(self.__feasible_next())
        if not feasible:
            return None

        topo = self.scene_spec.recipe.get_topological_sort()
        # Pick the feasible spice with the earliest topo position
        feasible.sort(key=lambda s: topo.index(s))
        return feasible[0]

    def __randomize_hidden_preferences(self) -> None:
        """Randomize the hidden preferences."""
        preferences = {spice: self._rng.choice(["human", "robot"]) for spice in self.scene_spec.recipe.spices}
        self._hidden_spec = SpiceHiddenSpec(preferred_actor=preferences)
    
    def __sample_preferences_from_hbm(self) -> None:
        """Sample preferences from the hidden HBM for this recipe."""
        if self._hidden_spec is None or self._hidden_spec.hidden_hbm is None:
            return
        
        hidden_hbm = self._hidden_spec.hidden_hbm
        recipe_name = self.scene_spec.recipe.name
        preferences = {}
        
        for spice in self.scene_spec.recipe.spices:
            # Sample phi from the hidden HBM's theta (human-level preference) for this spice
            # phi ~ N(theta_s, sigma_r^2)
            if spice in hidden_hbm.theta_mean:
                theta_mean = hidden_hbm.theta_mean[spice]
                theta_var = hidden_hbm.theta_var.get(spice, hidden_hbm.sigma_h**2 + hidden_hbm.sigma_r**2)
                # Sample phi from N(theta, sigma_r^2)
                phi = self._rng.normal(theta_mean, hidden_hbm.sigma_r)
                
                # Convert phi to actor preference: positive phi -> prefer human, negative -> prefer robot
                # Use sigmoid to convert to probability, then sample
                p_human = 1.0 / (1.0 + np.exp(-phi))
                preferred_actor = "human" if self._rng.random() < p_human else "robot"
            else:
                # Spice not in hidden HBM, default to random
                preferred_actor = self._rng.choice(["human", "robot"])
            
            preferences[spice] = preferred_actor
        
        # Update the hidden spec with sampled preferences
        if self._hidden_spec is not None:
            self._hidden_spec.preferred_actor = preferences

