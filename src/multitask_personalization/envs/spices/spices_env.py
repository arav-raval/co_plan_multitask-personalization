"""A simple spices environment for rapid testing of co-planning"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
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
    """A scene specification for the spices environment.

    ``recipe`` is the active recipe for the current episode. When
    ``train_recipe_names`` is set, each ``reset()`` samples a recipe from that
    pool (shared vocabulary / multi-dish training).
    """

    recipe: RecipeSpec
    train_recipe_names: tuple[str, ...] | None = None

@dataclass(frozen=False)  # Mutable to allow setting preferences from HBM
class SpiceHiddenSpec:
    """Hidden Human preference over who should add each spice."""
    preferred_actor: dict[str, str]  # {"Spice": "Actor"}
    hidden_hbm: "HierarchicalPreferenceModel | None" = None
    force_neutral_mood: bool = False  # If True, all episodes use neutral mood (training mode)
    stochastic_preferences: bool = False  # If True, phi ~ N(theta, sigma_r²) resampled each episode
    # Non-stationarity: a second hidden HBM whose phi values replace the primary
    # HBM's phi at ``preference_shift_step``.  When set, ``info["preference_shift"]``
    # is emitted once at the swap step so downstream code (run_single_experiment)
    # can track recovery.
    shift_hidden_hbm: "HierarchicalPreferenceModel | None" = None
    preference_shift_step: int = -1  # global env step at which the shift fires (-1 = never)

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
flag = 0: robot claims the current spice for itself (predicting human won't want it)
flag = 1: robot signals done / passes (predicting human will claim this spice)
payload: None (unused in the new autonomous-human semantics)

New semantics (human-led autonomous system):
  At each step the human FIRST decides whether to claim the current spice, based on
  their hidden preference phi and current session offset psi_true:
      P(human claims) = sigma(phi[spice] + psi_true)

  The robot simultaneously commits to either claiming the spice (flag=0) or passing (flag=1).

  Outcomes:
    - Robot claims (flag=0), human does NOT claim → robot adds spice. task_score = -1
      (robot was right that human didn't want it, but signal is from the behavioral
       observation: human not claiming = -1 for "human preferred").
    - Robot claims (flag=0), human ALSO claims → CONFLICT. Human wins. Robot gets a
      null step (task_score=0). Robot must replan next step.
    - Robot passes (flag=1), human claims → human adds spice. task_score = +1
      (correct prediction: human wanted it).
    - Robot passes (flag=1), human does NOT claim → robot must add anyway (someone
      must add the spice). task_score = -1 (robot mispredicted: thought human would
      claim but they didn't).

  The task_score fed to the HBM is a clean binary behavioral label:
    +1  human ended up claiming this spice (including conflict cases — human wins)
    -1  human did not claim this spice

  Conflict information is captured separately in info["conflict"] and via the
  coordination penalty subtracted from the continuous satisfaction signal. This
  separation ensures that the behavioral signal is always informative, while
  coordination quality is tracked through the satisfaction channel and metrics.
"""
SpiceAction: TypeAlias = tuple[int, str | None]

# --- SPICES ENVIRONMENT ---
class SpiceEnv(gym.Env[SpiceState, SpiceAction]):
    """
    A symbolic environment where actors follow a recipe to add spices to a pot.

    Human-led autonomous semantics (V = V_r ∪ V_h):
      The human autonomously claims spices according to their hidden preferences
      P(human claims spice s) = sigma(phi[s] + psi_true). The robot predicts which
      spices the human will claim (V_h) and picks unclaimed spices for itself (V_r).
      Conflicts (both attempt same spice) are resolved in favour of the human.

    The HBM learns from direct behavioral observations (+1 / -1 / 0) rather than
    a noisy satisfaction proxy, enabling cleaner preference inference.
    """

    def __init__(
        self,
        scene_spec: SpiceSceneSpec,
        hidden_spec: SpiceHiddenSpec,
        seed: int = 0,
        verbose: bool = False,
        eval_mode: bool | None = None,
    ) -> None:
        """If ``eval_mode`` is not ``None``, set ``hidden_spec.force_neutral_mood`` to
        that value (used by ``run_single_experiment``). If ``None``, leave
        ``force_neutral_mood`` unchanged for backward compatibility."""

        self._rng = np.random.default_rng(seed)
        self._hidden_spec = hidden_spec
        if eval_mode is not None:
            self._hidden_spec.force_neutral_mood = eval_mode
        self.scene_spec = scene_spec
        self.action_space = gym.spaces.OneOf(
            (
                EnumSpace(["human", "robot"]),
                EnumSpace([None]),
            )
        )
        self._mood_model = MoodModel(self._rng)

        self._train_recipe_names: list[str] | None = (
            list(self.scene_spec.train_recipe_names)
            if self.scene_spec.train_recipe_names
            else None
        )
        self._refresh_recipe_topology_caches()

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
        self._prediction_history: list[bool] = []
        self._action_history: list[SpiceAction] = []
        self._conflict_count: int = 0
        # Stage 2: true session offset sampled once per episode (replaces per-step mood_adj)
        self._current_psi_true: float = 0.0

        # Global step counter (persists across episodes) for preference-shift scheduling.
        self._global_step: int = 0
        self._preference_shifted: bool = False

        self.verbose = verbose

        if verbose:
            self._log_recipe()

    def _refresh_recipe_topology_caches(self) -> None:
        """Recompute DAG caches after ``scene_spec.recipe`` changes."""
        self._topo_order = self.scene_spec.recipe.get_topological_sort()
        self._layers = self.scene_spec.recipe.layers()
        self._layer_index = {
            spice: idx
            for idx, layer in enumerate(self._layers)
            for spice in layer
        }
        self._topo_index = {s: i for i, s in enumerate(self._topo_order)}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[SpiceState, dict[str, Any]]:

        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._train_recipe_names:
            from multitask_personalization.envs.spices.recipes import get_recipe

            chosen = str(self._rng.choice(self._train_recipe_names))
            self.scene_spec = SpiceSceneSpec(
                recipe=get_recipe(chosen),
                train_recipe_names=tuple(self._train_recipe_names),
            )
            self._refresh_recipe_topology_caches()
            if self.verbose:
                self._log_recipe()

        # Sample hidden preferences from HBM when available; otherwise use fixed preferences.
        if self._hidden_spec.hidden_hbm is not None:
            self._hidden_spec.preferred_actor = (
                self._hidden_spec.hidden_hbm.sample_episode_preferences(
                    list(self.scene_spec.recipe.spices),
                    self._rng,
                    stochastic=self._hidden_spec.stochastic_preferences,
                    recipe_name=self.scene_spec.recipe.name,
                )
            )

        # Reset episode state.
        self._t = 0
        self._added = []
        self._last_actor = None
        self._last_satisfaction = 0.0
        self._satisfaction_history = []
        self._prediction_history = []
        self._action_history = []
        self._conflict_count = 0

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
        """
        Autonomous-human step.

        The robot commits to either claiming the current spice (flag=0) or
        passing/predicting the human will claim it (flag=1). Simultaneously,
        the human decides whether to claim based on their hidden phi + psi_true.

        Returns task_score in info:
          +1  human claimed (direct behavioral observation)
          -1  human did not claim
           0  conflict (human wins, robot replans — no HBM update)
        """
        flag, _payload = action
        self._action_history.append(action)
        self._global_step += 1

        # --- Non-stationarity: fire preference shift once ---
        preference_shift_fired = False
        if (
            not self._preference_shifted
            and self._hidden_spec.shift_hidden_hbm is not None
            and self._hidden_spec.preference_shift_step >= 0
            and self._global_step >= self._hidden_spec.preference_shift_step
        ):
            self._hidden_spec.hidden_hbm = self._hidden_spec.shift_hidden_hbm
            self._preference_shifted = True
            preference_shift_fired = True
            if self.verbose:
                logging.info(
                    f"[SpiceEnv] PREFERENCE SHIFT at global step {self._global_step}!"
                )

        if self._current_spice is None:
            info = self._get_info(robot_indicated_done=True)
            return self._get_state(), 0.0, True, False, info

        current_spice = self._current_spice
        robot_claims = (flag == 0)  # flag=0: robot tries to claim; flag=1: robot passes

        # Human autonomously decides whether to claim based on hidden phi + psi.
        human_claims = self._human_claims(current_spice)

        # Determine true hidden preference for this spice (used by satisfaction model).
        # phi > 0 → human preferred, phi < 0 → robot preferred.
        try:
            recipe_name = self.scene_spec.recipe.name
            if self._hidden_spec.hidden_hbm is not None:
                _phi = self._hidden_spec.hidden_hbm.get_phi(DEFAULT_HUMAN, recipe_name, current_spice)
                true_preferred = "human" if float(_phi) >= 0 else "robot"
            else:
                true_preferred = self._hidden_spec.preferred_actor.get(current_spice, "robot")
        except Exception:
            true_preferred = "robot"

        # --- Resolve outcomes ---
        if robot_claims and human_claims:
            # CONFLICT: human wins. Task score is still +1 (the human acted),
            # since task_score is a clean behavioral label. The conflict
            # information is captured separately in info["conflict"] and via
            # the coordination penalty applied to the satisfaction signal.
            self._conflict_count += 1
            task_score = 1.0
            actor = "human"
            self._last_actor = actor
            self._added.append(current_spice)
            self._t += 1
            conflict = True
            # Conflict: robot was wrong (it competed when it should have passed).
            prediction_correct = False
        else:
            conflict = False
            if human_claims:
                # Human claimed, robot passed — correct prediction.
                actor = "human"
                task_score = 1.0
                prediction_correct = True
            elif robot_claims:
                # Robot claimed, human didn't — correct prediction.
                actor = "robot"
                task_score = -1.0
                prediction_correct = True
            else:
                # Robot passed but human didn't claim — misprediction, robot must add.
                actor = "robot"
                task_score = -1.0
                prediction_correct = False

            self._last_actor = actor
            self._added.append(current_spice)
            self._t += 1

        # Continuous satisfaction: reflects whether the right actor performed the spice,
        # modulated by the robot's prediction quality.  When the robot mispredicts
        # (conflict or missed pass), satisfaction is attenuated toward zero — the
        # human's experience suffers from poor coordination even if the spice ends up
        # with the right actor.  This makes satisfaction sensitive to method quality.
        satisfaction = self._compute_satisfaction(true_preferred, prediction_correct)

        self._satisfaction_history.append(satisfaction)
        self._last_satisfaction = satisfaction
        self._prediction_history.append(prediction_correct)

        # Advance to the next spice.
        self._current_spice = self._pick_current_spice()
        terminated = not self._current_spice
        info = self._get_info(robot_indicated_done=terminated, conflict=conflict,
                               task_score=task_score, satisfaction=satisfaction,
                               prediction_correct=prediction_correct)
        info["preference_shift"] = preference_shift_fired

        if self.verbose:
            status = "CONFLICT" if conflict else f"actor={actor}"
            logging.info(
                f"[Step {self._t - 1}] {current_spice} → {status}  "
                f"human_claims={human_claims}  robot_claims={robot_claims}  "
                f"task_score={task_score:.2f}"
            )

        return self._get_state(), 0.0, terminated, False, info

    def _human_claims(self, spice: str) -> bool:
        """
        Sample whether the human autonomously claims this spice.

        P(human claims) = sigma(phi[spice] + psi_true)

        Uses the hidden HBM's phi if available; otherwise uses
        preferred_actor as a hard prior (probability 1 or 0).
        """
        if self._hidden_spec.hidden_hbm is not None:
            try:
                recipe_name = self.scene_spec.recipe.name
                phi = float(self._hidden_spec.hidden_hbm.get_phi(DEFAULT_HUMAN, recipe_name, spice))
            except Exception:
                phi = 0.0
            # psi_true shifts the human's claiming probability this session.
            logit = phi + self._current_psi_true
            p_claim = float(1.0 / (1.0 + np.exp(-logit)))
        else:
            # Fallback: use preferred_actor as a deterministic prior.
            preferred = self._hidden_spec.preferred_actor.get(spice, "robot")
            p_claim = 1.0 if preferred == "human" else 0.0

        return bool(self._rng.random() < p_claim)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    def save_state(self, filepath: Path) -> None:
        """Persist minimal state for crash debugging (matches other envs' interface)."""
        payload = {
            "rng_state": self._rng.bit_generator.state,
            "t": self._t,
            "added": list(self._added),
            "current_spice": self._current_spice,
            "last_actor": self._last_actor,
            "current_mood": self._current_mood,
            "current_psi_true": self._current_psi_true,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

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

    def _compute_satisfaction(
        self, preferred: str, prediction_correct: bool
    ) -> float:
        """
        Compute satisfaction for the most recently assigned spice.

        Returns a continuous value in [-1, +1]:
            +1.0 — maximum positive satisfaction
             0.0 — neutral
            -1.0 — maximum negative satisfaction

        The generative model:
            phi_latent = pref_sign * phi_mag
            phi_mag    = |hidden_theta(spice)| when available, else base_satisfaction_bias
            logit      = actor_sign * (phi_latent + psi_true)
            p          = (tanh(logit/T) + 1) / 2    ← maps logit to (0,1)
            sat      ~ Beta(p*kappa, (1-p)*kappa) rescaled to [-1, +1]

        Coordination cost (additive penalty):
            When the robot mispredicts (conflict or missed pass), a fixed
            coordination_cost is subtracted from satisfaction.  This models
            the human's frustration from poor coordination *independently* of
            the underlying preference strength — a coordination failure is
            equally annoying whether the spice was strongly or weakly preferred.

            Example with coordination_cost = 0.5:
              correct prediction, strong preference:  +0.9 → +0.9  (unchanged)
              misprediction, strong preference:       +0.9 → +0.4  (penalized)
              misprediction, weak preference:         +0.2 → -0.3  (can go negative)

            This makes satisfaction sensitive to robot prediction quality,
            differentiating methods that predict well from those that don't.

        psi_true is sampled once per episode from a mood-conditioned distribution
        (see _sample_psi_from_mood).
        """
        actor_sign = 1.0 if self._last_actor == "human" else -1.0
        pref_sign = 1.0 if preferred == "human" else -1.0
        phi_mag = self._base_satisfaction_bias
        # If a hidden HBM is available, use the true theta magnitude for this spice
        # so generated logits are calibrated to latent preference strength.
        if self._hidden_spec.hidden_hbm is not None and self._added:
            spice = self._added[-1]
            try:
                recipe_name = self.scene_spec.recipe.name
                phi_val = self._hidden_spec.hidden_hbm.get_phi(DEFAULT_HUMAN, recipe_name, spice)
                phi_mag = max(abs(float(phi_val)), 1e-6)
            except Exception:
                phi_mag = self._base_satisfaction_bias
        phi_latent = pref_sign * phi_mag
        logit = actor_sign * (phi_latent + self._current_psi_true)

        temp = max(float(DEFAULT_CONFIG.satisfaction.satisfaction_logit_temperature), 1e-6)
        p = float(np.tanh(logit / temp))
        # Rescale tanh output from [-1,1] to [0,1] for Beta params.
        p = float(np.clip((p + 1.0) / 2.0, 1e-6, 1.0 - 1e-6))
        alpha = p * self._satisfaction_kappa
        beta = (1.0 - p) * self._satisfaction_kappa
        p_sampled = self._rng.beta(alpha, beta)
        raw_sat = float(np.clip(2.0 * p_sampled - 1.0, -1.0, 1.0))

        # Additive coordination cost: misprediction subtracts a flat penalty.
        if not prediction_correct:
            raw_sat -= DEFAULT_CONFIG.satisfaction.coordination_cost
            raw_sat = max(raw_sat, -1.0)

        return raw_sat

    def _sample_psi_from_mood(self, mood: str) -> float:
        """
        Sample the true session offset psi_true from a mood-conditioned distribution.

        Maps the 3-category mood to a scalar psi_true sampled once per episode.
        The HBM's prior N(0, sigma_mood²) is centered at 0; non-neutral moods shift
        the mean to ±psi_true_mood_mean_abs (configurable, independent of base bias)
        with std psi_true_mood_std.

        The HBM infers a latent scalar psi without knowing the mood category.
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

    def _get_info(
        self,
        robot_indicated_done: bool = False,
        conflict: bool = False,
        task_score: float = 0.0,
        satisfaction: float = 0.0,
        prediction_correct: bool | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "robot_indicated_done": robot_indicated_done,
            "conflict": conflict,
            "conflict_count": self._conflict_count,
            "current_spice": self._current_spice,
            "feasible_next": tuple(self._current_feasible),
            "action_history": list(self._action_history),
            "mood": self._current_mood,
            "recipe_name": self.scene_spec.recipe.name,
            "force_neutral_mood": self._hidden_spec.force_neutral_mood,
            "psi_true": self._current_psi_true,
        }
        if not self._added:
            info.update({
                "task_score": 0.0,
                # satisfaction: continuous Beta-sampled signal for HBM learning.
                # task_score: binary {+1,-1,0} behavioral label for CBTL and eval metrics.
                "satisfaction": 0.0,
                "user_satisfaction": 0.0,
                "last_spice": None,
                "last_actor": None,
                "average_satisfaction": 0.0,
                "satisfaction_history": [],
                "prediction_correct": None,
                "prediction_accuracy": float("nan"),
            })
        else:
            last_spice = self._added[-1]
            pred_acc = (
                float(np.mean(self._prediction_history))
                if self._prediction_history else float("nan")
            )
            info.update({
                # task_score: binary behavioral label (+1 human claimed, -1 didn't, 0 conflict).
                # Used by CBTL classifier and eval metrics (prediction_accuracy).
                "task_score": task_score,
                # satisfaction: continuous Beta-sampled value in [-1,+1].
                # Encodes preference magnitude via true hidden phi — used by HBM Gaussian likelihood.
                # Positive when the right actor performed the spice, negative otherwise.
                "satisfaction": satisfaction,
                # user_satisfaction aliases satisfaction for backward compat with runner CSV columns.
                "user_satisfaction": satisfaction,
                "last_spice": last_spice,
                "last_actor": self._last_actor,
                "average_satisfaction": float(np.mean(self._satisfaction_history)),
                "satisfaction_history": list(self._satisfaction_history),
                # Prediction accuracy: did robot's flag match human's true behavior?
                "prediction_correct": prediction_correct,
                "prediction_accuracy": pred_acc,
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
