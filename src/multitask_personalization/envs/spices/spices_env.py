"""A simple spices environment for rapid testing of co-planning"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, Iterable
import logging
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from tomsutils.spaces import EnumSpace
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from multitask_personalization.structs import PublicSceneSpec

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
        layers = self.layers()  # list of lists of nodes

        # Build layer index
        layer_index = {}
        for i, layer in enumerate(layers):
            for s in layer:
                layer_index[s] = i
        nx.set_node_attributes(G, layer_index, "layer")

        # Build a color map for layers
        num_layers = len(layers)
        cmap = cm.get_cmap("viridis", num_layers)
        colors = [cmap(layer_index[n]) for n in G.nodes]

        # Layout
        pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")

        plt.figure(figsize=(10, 4))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color=colors,
            font_size=10,
            font_weight="bold",
            arrowsize=20,
            font_color="white"
        )

        plt.title(f"Recipe DAG: {self.name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    
@dataclass(frozen=True)
class MoodSpec:
    import numpy as np

@dataclass(frozen=True)
class MoodSpec:
    """Defines the mood categories."""
    moods: tuple[str, ...] = ("all_self", "neutral", "none_self")
    priors: tuple[float, ...] = (0.33, 0.34, 0.33)

@dataclass(frozen=True)
class SpiceSceneSpec(PublicSceneSpec):
    """A scene specification for the spices environment."""
    recipe: RecipeSpec

@dataclass(frozen=True)
class SpiceHiddenSpec:
    """Hidden Human preference over WHO should add each spice."""
    preferred_actor: dict[str, str] # {"Spice":  "Actor"}

@dataclass(frozen=True)
class SpiceState:
    """The current state of the spices environment."""
    time: int
    added_spices: tuple[str, ...] # spices already added to the pot (sequence)
    remaining_spices: tuple[str, ...] # spices not yet added to the pot
    feasible_next: tuple[str, ...] # spices that can be added next
    current_spice: str | None # the current spice to add

# Action: (flag, payload)
"""
The actions are defined as (flag, payload) where flag = 0 / 1 indicates whether the action is an "add" or "done".
If flag = 0, then payload is the name of the actor that is assigned to SpiceState.feasible_next (next spice to add)
"""
SpiceAction: TypeAlias = tuple[int, str | None] # (flag: 0=add, 1=done, 2=wait, payload: "Add <spice>" or None)

class MoodModel:
    """Handles sampling and updating the current day's mood."""
    def __init__(self, spec: MoodSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self.current_mood = None

    def sample_mood(self) -> str:
        self.current_mood = self.rng.choice(self.spec.moods, p=self.spec.priors)
        return self.current_mood

class SpiceEnv(gym.Env[SpiceState, SpiceAction]):
    """A simple spices symbolic environment for rapid testing of co-planning.
    
    The environment is a simple symbolic environment where the user can add spices to a pot.
    The human has a hidden preference about which spices they prefer to add to the pot. 

    Actions are adding a spice to the pot.
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

        self._t = 0
        self._added: list[str] = []
        self._current_spice: str | None = None
        self._last_actor: str | None = None
        self._satisfaction_history: list[float] = []
        self._action_history: list[SpiceAction] = []


        self.eval_mode = eval_mode
        self.verbose = verbose


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[SpiceState, dict[str, Any]]:

        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Create stationary randomized preferences if none provided
        if self.eval_mode or self._hidden_spec is None:
            self.__randomize_hidden_preferences()

        self._t = 0
        self._added: list[str] = []
        self._last_actor: str | None = None
        self._satisfaction_history: list[float] = []
        self._current_spice = self._pick_current_spice()
        self._action_history: list[SpiceAction] = []
        self._current_mood = self._mood_model.sample_mood()

        if self.verbose:
            logging.info("[SpiceEnv] Resetting environment")
            logging.info(f"[Mood] Current mood: {self._current_mood}")
            self._log_recipe_and_prefs()
            #self.scene_spec.recipe.visualize_dag()

        return self._get_state(), self._get_info()

    def step(
        self, action: SpiceAction
    ) -> tuple[SpiceState, float, bool, bool, dict[str, Any]]:
        status, payload = action

        self._action_history.append(action)

        # Done with all spices
        if np.isclose(status, 1):
            info = self._get_info(robot_indicated_done=True)
            terminated = True

            if self.verbose:
                logging.info(f"[RECIPE COMPLETED] Average Satisfaction={info['average_satisfaction']:+.2f}")
            
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
                #self.visualize_current_frontier()
                logging.info(f"[STEP {self._t - 1}] Assign {info['last_spice']} → {info['last_actor']} "
                  f"(pref={info['preferred_actor']})  sat={info['satisfaction']:+}")
                logging.info(f"[RECIPE COMPLETED] Average Satisfaction={info['average_satisfaction']:+.2f}")

            return self._get_state(), 0.0, terminated, False, info

        # Get info
        info = self._get_info(robot_indicated_done=False)

        if self.verbose:
            logging.info(f"[STEP {self._t - 1}] Assign {info['last_spice']} → {info['last_actor']} "
                  f"(pref={info['preferred_actor']})  sat={info['satisfaction']:+}")

        return self._get_state(), 0.0, False, False, info
    
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    # ---------------- UTIILTY FUNCTIONS ----------------
    def _log_recipe_and_prefs(self) -> None:
        recipe = self.scene_spec.recipe
        logging.info("\n === RECIPE ===\n"
                     f"Name: {recipe.name}\n"
                     "Spices: {tuple(recipe.spices)}\n"
                     "Predecessors (must come before -> spice):\n")
        for s in recipe.spices:
            preds = recipe.predecessors.get(s, ())
            if preds:
                for p in preds:
                    logging.info(f"  {p} -> {s}")
            else:
                logging.info(f"  (start) -> {s}" if s == self._current_spice else f"  (no preds) -> {s}")
        logging.info("\n=== HIDDEN PREFERENCES (actor per spice) ===")
        for s in recipe.spices:
            who = self._hidden_spec.preferred_actor.get(s, "?")
            logging.info(f"  {s:>15}: {who}")
        logging.info("=====================")

    def _get_state(self) -> SpiceState:
        feasible_list = list(self.__feasible_next())
        return SpiceState(
            time=self._t,
            added_spices=tuple(self._added),
            remaining_spices=tuple(spice for spice in self.scene_spec.recipe.spices if spice not in self._added),
            feasible_next=tuple(feasible_list),
            current_spice=self._current_spice or (feasible_list[0] if feasible_list else ""),
        )

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
            }

        # Mood table (shift in preferences)
        MOOD_BIAS = {
            "all_self":   {"human": +6.0, "robot": -6.0},
            "neutral":    {"human": 0.0,  "robot": 0.0},
            "none_self":  {"human": -6.0, "robot": +6.0},
        }
        
        # Non-empty case
        last_spice = self._added[-1]
        preferred = self._hidden_spec.preferred_actor[last_spice]

        # φ component: long-term preference signal
        phi = +2.0 if preferred == self._last_actor else -2.0

        # mood component
        mood_adj = MOOD_BIAS[self._current_mood][self._last_actor]

        # Combine additively into a logit
        logit = phi + mood_adj

        # Probabilistic satisfaction
        p = 1 / (1 + np.exp(-logit))
        satisfaction = self._rng.choice([+1, -1], p=[p, 1-p])


        logging.info(f"[Mood] Mood bias: {mood_adj}")
        logging.info(f"[Mood] Base satisfaction: {phi}")
        logging.info(f"[Mood] Satisfaction: {satisfaction}")

        self._satisfaction_history.append(satisfaction)

        return {
            "robot_indicated_done": robot_indicated_done,
            "satisfaction": satisfaction,
            "preferred_actor": preferred,
            "current_spice": self._current_spice,
            "last_spice": last_spice,
            "last_actor": self._last_actor,
            "satisfaction_history": self._satisfaction_history,
            "average_satisfaction": np.mean(self._satisfaction_history),
            "satisfaction_variance": np.var(self._satisfaction_history),
            "feasible_next": tuple(self.__feasible_next()),
            "action_history": self._action_history,
            "mood": self._current_mood,
        }

    def __feasible_next(self) -> Iterable[str]:
        """The spices that can be added next."""
        dag = self._dag

        added = set(self._added)

        feasible = []

        for node in dag.nodes:
            if node in added:
                continue

            preds = set(dag.predecessors(node))
            if preds.issubset(added):
                feasible.append(node)

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
        
    
    # def visualize_current_frontier(self):
    #     """
    #     Visualize only the current DAG frontier (feasible next spices).
    #     """
    #     import matplotlib.pyplot as plt
    #     import networkx as nx

    #     frontier = list(self.__feasible_next())

    #     # Build a tiny graph that only contains frontier nodes
    #     G = nx.DiGraph()
    #     for s in frontier:
    #         G.add_node(s)

    #     # Layout: simple horizontal placement
    #     pos = {s: (i, 0) for i, s in enumerate(frontier)}

    #     plt.figure(figsize=(8, 2))

    #     nx.draw(
    #         G,
    #         pos,
    #         with_labels=True,
    #         node_color="#ffd700",  # gold for frontier
    #         node_size=1800,
    #         font_size=10,
    #         font_weight="bold"
    #     )

    #     plt.title(f"Current Frontier at t={self._t}")
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.show()


