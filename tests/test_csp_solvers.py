"""Tests for csp_solvers.py."""

import gymnasium as gym
import numpy as np

from multitask_personalization.csp_solvers import (
    EnumerationCSPSolver,
    LifelongCSPSolverWrapper,
    RandomWalkCSPSolver,
)
from multitask_personalization.structs import (
    CSP,
    CSPCost,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


def _create_test_csp():
    x = CSPVariable("x", gym.spaces.Box(0, 1, dtype=np.float_))
    y = CSPVariable("y", gym.spaces.Box(0, 1, dtype=np.float_))
    z = CSPVariable("z", gym.spaces.Discrete(5))

    c1 = FunctionalCSPConstraint("c1", [x, y], lambda x, y: x < y)
    c2 = LogProbCSPConstraint("c2", [y, z], lambda y, z: np.log(y < z / 5))

    csp = CSP([x, y, z], [c1, c2])

    # Uncomment to visualize the CSP.
    # from multitask_personalization.utils import visualize_csp_graph
    # from pathlib import Path
    # visualize_csp_graph(csp, Path("test_csp.png"))

    sample_xy = lambda _, rng: {
        x: rng.uniform(0, 1, size=(1,)),
        y: rng.uniform(0, 1, size=(1,)),
    }
    sample_z = lambda _, rng: {z: rng.integers(5)}

    sampler_xy = FunctionalCSPSampler(sample_xy, csp, {x, y})
    sampler_z = FunctionalCSPSampler(sample_z, csp, {z})
    samplers = [sampler_xy, sampler_z]
    initialization = {x: 0.0, y: 0.0, z: 0}

    return csp, initialization, samplers


def test_solve_csp():
    """Tests for csp_solvers.py."""

    # Test RandomWalkCSPSolver().
    csp, initialization, samplers = _create_test_csp()
    solver = RandomWalkCSPSolver(seed=123, show_progress_bar=False)
    sol = solver.solve(csp, initialization, samplers)
    assert sol is not None

    # Test LifelongCSPSolverWrapper(RandomWalkCSPSolver()).
    # The lifelong solver should still work after deleting the samplers because
    # it should use its own memory-based samplers.
    lifelong_solver = LifelongCSPSolverWrapper(solver, seed=123)
    sol = lifelong_solver.solve(csp, initialization, samplers)
    assert sol is not None
    # Regenerate the CSP to make sure that equality checking is based on names.
    csp, initialization, samplers = _create_test_csp()
    sol = lifelong_solver.solve(csp, initialization, [])
    assert sol is not None


def test_enumeration_csp_solver():
    """Test EnumerationCSPSolver on a trivial binary CSP (like spices)."""
    # Binary choice: x in {0, 1}, constraint x == 1, cost = -x (minimize -x = maximize x)
    x = CSPVariable("x", gym.spaces.Discrete(2))
    c = FunctionalCSPConstraint("c", [x], lambda x_val: x_val == 1)
    cost = CSPCost("max", [x], lambda x_val: -float(x_val))
    csp = CSP([x], [c], cost)
    initialization = {x: 0}
    samplers = []  # Not used by EnumerationCSPSolver

    solver = EnumerationCSPSolver(seed=123)
    sol = solver.solve(csp, initialization, samplers)
    assert sol is not None
    assert sol[x] == 1


def test_enumeration_csp_solver_returns_none_for_continuous():
    """EnumerationCSPSolver returns None when variables have continuous domains."""
    x = CSPVariable("x", gym.spaces.Box(0, 1, dtype=np.float_))
    c = FunctionalCSPConstraint("c", [x], lambda x_val: x_val > 0.5)
    csp = CSP([x], [c], cost=None)
    initialization = {x: 0.0}
    samplers = []

    solver = EnumerationCSPSolver(seed=123)
    sol = solver.solve(csp, initialization, samplers)
    assert sol is None
