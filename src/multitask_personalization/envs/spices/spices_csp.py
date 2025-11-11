"""CSP Elements for the spices environment."""

from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Any, Collection
import logging
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import RadiusNeighborsClassifier
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_generation import (
    CSPGenerator,
    CSPConstraintGenerator,
)

from multitask_personalization.envs.spices.spices_env import SpiceAction, SpiceState

from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)

class _SpiceCSPPolicy(CSPPolicy[SpiceState, SpiceAction]):
    def __init__(self, csp_variables: Collection[CSPVariable], seed: int = 0, verbose: bool = False) -> None:
        super().__init__(csp_variables, seed)
        self._rng = np.random.default_rng(seed)
        self._actor: str | None = None
        self._done_emitted = False
        self._verbose = verbose

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._actor = self._get_value("actor")
        self._done_emitted = False

    def step(self, obs: SpiceState) -> SpiceAction:
        # Emit done 
        if (not obs.current_spice) or (len(obs.feasible_next) == 0 and len(obs.remaining_spices) == 0):
            if not self._done_emitted:
                self._done_emitted = True
            # If called again after done, keep returning done
            return (1, None)
        
        # Assign the selected actor for the current spice
        if self._actor in ("human", "robot"):
            actor = self._actor
            if self._verbose:
                logging.info(f"Using pre-assigned actor: {actor}")
        else:
            actor = self._rng.choice(["human", "robot"])
            if self._verbose:
                logging.info(f"Choosing random actor: {actor}")
        return (0, actor)

    def check_termination(self, obs: SpiceState) -> bool:
        return self._done_emitted
    
class _AssignPreferenceGenerator(CSPConstraintGenerator[SpiceState, SpiceAction]):
    def __init__(self, spice_list: list[str], seed: int = 0, verbose: bool = False) -> None:
        super().__init__(seed)
        self._spice_to_index = {spice: i for i, spice in enumerate(spice_list)}
        self._actor_to_index = {actor: i for i, actor in enumerate(["human", "robot"])}

        self._classifier: RadiusNeighborsClassifier | None = None
        self._training_inputs: list[NDArray] = []
        self._training_outputs: list[bool] = []

        self._verbose = verbose
    def save(self, model_dir: Path) -> None:
        outfile = model_dir / "assign_preference_classifier.pkl"
        with open(outfile, "wb") as f:
            pkl.dump(self._classifier, f)
    
    def load(self, model_dir: Path) -> None:
        outfile = model_dir / "assign_preference_classifier.pkl"
        with open(outfile, "rb") as f:
            self._classifier = pkl.load(f)

    def generate(self, obs: SpiceState, variables: list[CSPVariable], name: str) -> CSPConstraint:
        (actor_vars, ) = variables
        current = obs.current_spice

        def _logprob(actor: str) -> float:
            
            if self._classifier is None:
                return np.log(0.5)
            x = self._featurize(current, actor)
            p = self._classifier.predict_proba([x])[0][1]
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            y = float(np.log(p))

            #print(f"[DEBUG] spice={current}, actor={actor}, p={p:.3f}, logp={y:.3f}, threshold={np.log(0.5):.3f}")
            return y

        return LogProbCSPConstraint(name, [actor_vars], _logprob, threshold=np.log(1e-6))

    def visualize_classifier(self) -> None:
        X = np.array(self._training_inputs)
        y = np.array(self._training_outputs)

        # Define the grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = -0.5, 1.5  # actor index range: 0=human, 1=robot
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.05))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict probabilities over the grid
        Z = np.zeros(len(grid))
        for i, g in enumerate(grid):
            probs = self._classifier.predict_proba([g])[0]
            Z[i] = probs[1]  # probability of positive label (True)
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.6)
        plt.colorbar(label="P(label=True)")
        plt.scatter(X[:, 0], X[:, 1],
                    c=y, cmap="bwr", edgecolor="k", s=120, marker="o", label="Training points")
        plt.xlabel("Spice index")
        plt.ylabel("Actor index (0=human, 1=robot)")
        plt.title(f"RadiusNeighborsClassifier (radius={self._classifier.radius})")
        plt.legend()
        plt.show()

    def learn_from_transition(self, obs: SpiceState, act: SpiceAction, next_obs: SpiceState, done: bool, info: dict[str, Any]) -> None:
        # Learn after each step (not after episode completion)
        if info.get("last_spice") is not None and info.get("last_actor") is not None:
            self._training_inputs.append(
                self._featurize(str(info["last_spice"]), str(info["last_actor"]))
            )
            self._training_outputs.append(info["satisfaction"] > 0.0)
        
        if done:
            self._update_constraint_parameters()
            
    def get_metrics(self) -> dict[str, float]:
        return {}
    
    def _featurize(self, spice: str, actor: str) -> NDArray:
        return np.array([self._spice_to_index[spice], self._actor_to_index[actor]], dtype=float)
    
    def _update_constraint_parameters(self) -> None:
        # Wait until we've seen both positive and negative examples to learn.
        if len(set(self._training_outputs)) < 2:
            return

        # Train a classifier.
        self._classifier = RadiusNeighborsClassifier(
            radius=1.5, 
            weights="distance",
            algorithm="auto",
            p=2, 
            metric="minkowski",
        )

        self._classifier.fit(self._training_inputs, self._training_outputs)

class SpicesAssignCSPGenerator(CSPGenerator[SpiceState, SpiceAction]):
    """CSP: choose the actor for the environment's current spice; learn the preferences"""

    def __init__(self, spice_list: list[str], verbose: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._spices = list(spice_list)
        self._pref_gen = _AssignPreferenceGenerator(self._spices, self._seed, verbose)

        # Separate RNG that won't be reset
        self._init_rng = np.random.default_rng(self._seed)

    def save(self, model_dir: Path) -> None:
        self._pref_gen.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._pref_gen.load(model_dir)

    def get_pref_snapshot(self) -> dict[str, dict[str, float]]:
        """Return current P(prefer=True) for each spice/actor."""
        probs = {}
        for spice in self._pref_gen._spice_to_index.keys():
            spice_probs = {}
            for actor in ["human", "robot"]:
                if self._pref_gen._classifier is None:
                    spice_probs[actor] = 0.5
                else:
                    x = self._pref_gen._featurize(spice, actor)
                    p = self._pref_gen._classifier.predict_proba([x])[0][1]
                    spice_probs[actor] = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            total = sum(spice_probs.values())
            spice_probs = {k: round(v / total, 3) for k, v in spice_probs.items()}
            probs[spice] = spice_probs
        return probs


    def _generate_variables(self, obs: SpiceState) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        actor = CSPVariable("actor", EnumSpace(["human", "robot"]))
        variables = [actor]

        # Randomize the initial assignment
        initialization = {actor: self._init_rng.choice(["human", "robot"])}
        return variables, initialization

    def _generate_personal_constraints(self, obs: SpiceState, variables: list[CSPVariable]) -> list[CSPConstraint]:
        user_preference_constraint = self._pref_gen.generate(obs, variables, "user_preference")
        return [user_preference_constraint]
    
    def _generate_nonpersonal_constraints(self, obs: SpiceState, variables: list[CSPVariable]) -> list[CSPConstraint]:
        # Feasibility of spice enforced in the environment
        return []
    
    def _generate_exploit_cost(self, obs: SpiceState, variables: list[CSPVariable]) -> CSPCost | None:
        """Use negative log-probability as cost to prefer higher probability solutions."""
        actor = variables[0]
        current = obs.current_spice

        if self._pref_gen._classifier is None:
            return None

        # Get the constraint's log-prob function
        def _cost_fn(actor_val: str) -> float:    
            # Calculate log-probability for this actor
            x = self._pref_gen._featurize(current, actor_val)
            p = self._pref_gen._classifier.predict_proba([x])[0][1]
            # Clip to avoid log(0.0) = -inf
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            logprob = float(np.log(p))
            
            # Return negative log-prob as cost (solver minimizes, we want to maximize logprob)
            return -logprob
        
        return CSPCost("maximize_preference", [actor], _cost_fn)
    
    def _generate_samplers(self, obs: SpiceState, csp: CSP) -> list[CSPSampler]:
        actor = csp.variables[0]
        current_spice = obs.current_spice

        def _sample_actor(sol: dict[CSPVariable, Any], rng: np.random.Generator) -> dict[CSPVariable, Any]:
            # Fall back to random choice 
            if self._pref_gen._classifier is None:
                #print(f"[Sampler] No classifier trained yet, using random choice")
                chosen = rng.choice(["human", "robot"])
                return {actor: chosen}

            # Compute preference probabilities for both actors
            probs = []
            for a in ["human", "robot"]:
                x = self._pref_gen._featurize(current_spice, a)
                p = self._pref_gen._classifier.predict_proba([x])[0][1]
                probs.append(np.clip(p, 1e-6, 1.0 - 1e-6))  # avoid zero probabilities

            # Normalize to sum to 1
            probs = np.array(probs)
            probs /= probs.sum()

            # Sample actor according to learned probabilities (soft sampling)
            chosen = rng.choice(["human", "robot"], p=probs)

            # Debugging
            #print(f"[Sampler] {current_spice}: P(human)={probs[0]:.2f}, P(robot)={probs[1]:.2f} → chosen={chosen}")

            return {actor: chosen}
        return [FunctionalCSPSampler(_sample_actor, csp, {actor})]

    def _generate_policy(self, obs: SpiceState, csp_variables: Collection[CSPVariable]) -> CSPPolicy:
        return _SpiceCSPPolicy(csp_variables, seed=self._seed)

    def observe_transition(self, obs: SpiceState, act: SpiceAction, next_obs: SpiceState, done: bool, info: dict[str, Any]) -> None:
        if not self._disable_learning:
            self._pref_gen.learn_from_transition(obs, act, next_obs, done, info)

    def get_metrics(self) -> dict[str, float]:
        return self._pref_gen.get_metrics()
