"""Different methods for solving CSPs."""

import abc
import itertools
from collections import defaultdict, deque
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
)


def _get_domain_values(space: gym.Space, var: CSPVariable) -> list[Any] | None:
    """Get enumerable values from a space, or None if not enumerable."""
    if isinstance(space, gym.spaces.Discrete):
        return list(range(space.n))
    # EnumSpace from tomsutils stores options
    if hasattr(space, "options"):
        return list(space.options)
    if hasattr(space, "choices"):
        return list(space.choices)
    if hasattr(space, "elements"):
        return list(space.elements)
    return None


class CSPSolver(abc.ABC):
    """A CSP solver."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    @abc.abstractmethod
    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        """Solve the given CSP."""


class RandomWalkCSPSolver(CSPSolver):
    """Call samplers completely at random and remember the best seen
    solution."""

    def __init__(
        self,
        seed: int,
        max_iters: int = 100_000,
        num_improvements: int = 5,
        max_improvement_attempts: int = 1_000,
        show_progress_bar: bool = True,
    ) -> None:
        super().__init__(seed)
        self._max_iters = max_iters
        self._num_improvements = num_improvements
        self._max_improvement_attempts = max_improvement_attempts
        self._show_progress_bar = show_progress_bar
        self._rng = np.random.default_rng(seed)

    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        sol = initialization.copy()
        best_satisfying_sol: dict[CSPVariable, Any] | None = None
        best_satisfying_cost: float = np.inf
        solution_found = False
        num_improve_attempts = 0
        num_improve_found = 0
        sampler_idxs = list(range(len(samplers)))
        for _ in (
            pbar := tqdm(range(self._max_iters), disable=not self._show_progress_bar)
        ):
            # Check for early termination.
            if solution_found and (
                num_improve_attempts >= self._max_improvement_attempts
                or num_improve_found >= self._num_improvements
            ):
                break
            # Update progress.
            if solution_found:
                num_improve_attempts += 1
                msg = (
                    f"Improved {num_improve_found} times w/ "
                    f"{num_improve_attempts} tries)"
                )
            else:
                msg = "Searching for first solution"
            pbar.set_description(msg)

            # Uncomment to debug.
            # from multitask_personalization.utils import print_csp_sol
            # print_csp_sol(sol)

            # Don't ever both with solutions that are worse than what we've seen.
            sol_is_cost_improvement = True
            if csp.cost is not None:
                cost = csp.get_cost(sol)
                # Note: this should be >, rather than >=, because of interaction
                # with the Lifelong solver.
                if cost > best_satisfying_cost:
                    sol_is_cost_improvement = False

            # This would be a cost improvement, so see if the constraints pass.
            if sol_is_cost_improvement and csp.check_solution(sol):
                if solution_found:
                    num_improve_found += 1
                solution_found = True
                if csp.cost is None:
                    return sol
                best_satisfying_cost = cost
                best_satisfying_sol = sol

            # Sample the next solution.
            self._rng.shuffle(sampler_idxs)
            for sample_idx in sampler_idxs:
                sampler = samplers[sample_idx]
                partial_sol = sampler.sample(sol, self._rng)
                if partial_sol is not None:
                    break
            else:
                raise RuntimeError("All samplers produced None; solver stuck.")
            sol = sol.copy()
            sol.update(partial_sol)
        return best_satisfying_sol


class EnumerationCSPSolver(CSPSolver):
    """Solve CSPs by enumerating all combinations when variables have finite
    discrete domains (for binary choice domains like spices)
    Returns None if the CSP cannot be enumerated
    """

    def __init__(self, seed: int, max_enumeration_size: int = 10_000) -> None:
        super().__init__(seed)
        self._max_enumeration_size = max_enumeration_size
        self._rng = np.random.default_rng(seed)

    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        # Get domain values for each variable
        domain_values: list[list[Any]] = []
        for var in csp.variables:
            vals = _get_domain_values(var.domain, var)
            if vals is None:
                return None
            domain_values.append(vals)

        # Check enumeration size
        total = 1
        for vals in domain_values:
            total *= len(vals)
        if total > self._max_enumeration_size:
            return None

        # Enumerate all combinations and find best valid solution.
        # Shuffle to randomize tie-breaking when multiple solutions share
        # the minimum cost (e.g., binary domains with symmetric entropy).
        # This matches the original CBTL RandomWalkCSPSolver's behavior:
        # the strict > rejection check (not >=) means equal-cost solutions
        # replace the current best, and random sampler ordering determines
        # which candidate wins.  Shuffling the enumeration order achieves
        # the same effect for exhaustive enumeration.
        best_sol: dict[CSPVariable, Any] | None = None
        best_cost: float = np.inf

        all_combos = list(itertools.product(*domain_values))
        self._rng.shuffle(all_combos)

        for values in all_combos:
            sol = {var: val for var, val in zip(csp.variables, values)}
            if not csp.check_solution(sol):
                continue
            if csp.cost is None:
                return sol
            cost = csp.get_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_sol = sol

        return best_sol


class LifelongCSPSolverWrapper(CSPSolver):
    """A wrapper that samples from past constraint solutions."""

    def __init__(
        self, base_solver: CSPSolver, seed: int, memory_size: int = 100
    ) -> None:
        super().__init__(seed)
        self._base_solver = base_solver
        self._memory_size = memory_size
        self._constraint_to_recent_solutions: dict[
            CSPConstraint, deque[dict[CSPVariable, Any]]
        ] = defaultdict(lambda: deque(maxlen=self._memory_size))

    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        # Create the samplers from past experience.
        memory_based_samplers = self._create_memory_based_samplers(csp)
        samplers = samplers + memory_based_samplers
        # Need to wrap the constraints so that we can memorize solutions.
        # Note that we could also just take the output of solve() and memorize,
        # but that would miss out on the opportunity to memorize intermediates.
        wrapped_csp = self._wrap_csp(csp)
        return self._base_solver.solve(wrapped_csp, initialization, samplers)

    def _create_memory_based_samplers(self, csp: CSP) -> list[CSPSampler]:
        new_samplers: list[CSPSampler] = []
        for constraint in csp.constraints:
            if constraint in self._constraint_to_recent_solutions:
                sampler = self._create_memory_based_sampler(constraint, csp)
                new_samplers.append(sampler)
        return new_samplers

    def _create_memory_based_sampler(
        self, constraint: CSPConstraint, csp: CSP
    ) -> CSPSampler:
        recent_solutions = self._constraint_to_recent_solutions[constraint]
        num_recent_solutions = len(recent_solutions)

        def _sample(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any] | None:
            idx = rng.choice(num_recent_solutions)
            return recent_solutions[idx]

        return FunctionalCSPSampler(_sample, csp, set(constraint.variables))

    def _wrap_csp(self, csp: CSP) -> CSP:
        new_constraints = [self._wrap_constraint(c) for c in csp.constraints]
        return CSP(csp.variables, new_constraints, csp.cost)

    def _wrap_constraint(self, constraint: CSPConstraint) -> CSPConstraint:
        # Make sure not to modify original constraint.
        new_constraint = constraint.copy()

        # Memorize solutions.
        def _wrapped_check_solution(sol: dict[CSPVariable, Any]) -> bool:
            result = constraint.check_solution(sol)
            if result:
                partial_sol = {v: sol[v] for v in constraint.variables}
                self._constraint_to_recent_solutions[constraint].append(partial_sol)
            return result

        # Overwrite check_solution method.
        new_constraint.check_solution = _wrapped_check_solution
        return new_constraint
