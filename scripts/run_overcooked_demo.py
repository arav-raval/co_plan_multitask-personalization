"""
run_overcooked_demo.py — Live demo of the full Overcooked preference-learning system.

This script wires together:
  - OvercookedEnv          (session sampling, subtask sequencing, task score)
  - OvercookedAssignCSPGenerator  (HBM + CSP: assigns each subtask to human or robot)
  - SubtaskPlanner         (low-level primitive actions via MLAM)
  - OvercookedVisualizer   (pygame window or PNG frames)

Run with:
    PYTHONPATH=src .venv/bin/python scripts/run_overcooked_demo.py

Options:
    --layout     Layout name (default: CrampedRoom)
    --episodes   Number of episodes to run (default: 5)
    --human-type Hidden preference archetype (default: RealisticCook)
                 Choices: RealisticCook, PrefsPrep, PrefsDelivery, PrefsAll, PrefsNone,
                          Alternating, RobotCentric, VariableHuman, ConsistentHuman
    --fps        Rendering frames per second (default: 8)
    --save-dir   If given, save PNG frames here instead of showing live window
    --no-render  Skip rendering entirely (faster; for testing)
    --eval       Run in eval (exploit-only) mode instead of train (explore) mode
    --seed       Random seed (default: 42)

Agent assignment:
  - Robot = agent 0 = BLUE hat
  - Human = agent 1 = GREEN hat

The robot learns phi per subtask from the human's satisfaction signal and
gradually improves its subtask assignments over episodes.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overcooked CSP preference-learning demo")
    p.add_argument("--layout", default="CrampedRoom",
                   help="Layout name (CrampedRoom, CoordinationRing, ...)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--human-type", default="RealisticCook",
                   dest="human_type",
                   help="Hidden HBM config name")
    p.add_argument("--fps", type=int, default=8,
                   help="Rendering speed (frames per second)")
    p.add_argument("--save-dir", default=None, dest="save_dir",
                   help="Save PNG frames here instead of live window")
    p.add_argument("--no-render", action="store_true", dest="no_render")
    p.add_argument("--eval", action="store_true",
                   help="Run in eval (exploit-only) mode")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Imports (deferred so argparse --help works without the full stack)
    # ------------------------------------------------------------------
    from multitask_personalization.envs.overcooked.layouts import get_layout
    from multitask_personalization.envs.overcooked.overcooked_env import (
        OvercookedEnv,
        OvercookedHiddenSpec,
    )
    from multitask_personalization.envs.overcooked.overcooked_csp import (
        OvercookedAssignCSPGenerator,
    )
    from multitask_personalization.envs.overcooked.config.hidden_hbm_configs import (
        get_hidden_hbm_config,
    )
    from multitask_personalization.envs.overcooked.overcooked_hbm import (
        DEFAULT_HUMAN,
    )
    from multitask_personalization.csp_solvers import EnumerationCSPSolver

    # ------------------------------------------------------------------
    # Layout + hidden spec
    # ------------------------------------------------------------------
    layout_spec = get_layout(args.layout)
    subtasks = layout_spec.subtasks

    hidden_cfg = get_hidden_hbm_config(args.human_type)
    theta = hidden_cfg.generate_theta(subtasks)
    preferred_actor = {s: ("human" if theta[s] > 0 else "robot") for s in subtasks}

    hidden_spec = OvercookedHiddenSpec(preferred_actor=preferred_actor)

    print(f"\n=== Overcooked CSP Demo ===")
    print(f"Layout:     {layout_spec.name}  ({layout_spec.layout_name})")
    print(f"Human type: {args.human_type}")
    print(f"Mode:       {'eval (exploit)' if args.eval else 'train (explore)'}")
    print(f"Episodes:   {args.episodes}")
    print(f"\nTrue preferences:")
    for s, a in preferred_actor.items():
        sign = "+" if theta[s] > 0 else "-"
        print(f"  {s:20s}  →  {a:6s}  (theta={theta[s]:+.1f})")
    print()

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = OvercookedEnv(
        layout_spec=layout_spec,
        hidden_spec=hidden_spec,
        seed=args.seed,
    )

    if not env._overcooked_available:
        print("ERROR: overcooked_ai_py is not available. "
              "Run with: PYTHONPATH=src .venv/bin/python scripts/run_overcooked_demo.py")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CSP generator
    # ------------------------------------------------------------------
    gen = OvercookedAssignCSPGenerator(
        subtask_list=subtasks,
        layout_list=[layout_spec.name],
        human_id=DEFAULT_HUMAN,
        seed=args.seed,
        verbose=True,
    )
    if args.eval:
        gen.eval()
    else:
        gen.train()

    solver = EnumerationCSPSolver(seed=args.seed)

    # ------------------------------------------------------------------
    # Visualizer
    # ------------------------------------------------------------------
    viz: Any = None
    if not args.no_render:
        from multitask_personalization.envs.overcooked.overcooked_visualizer import (
            OvercookedVisualizer,
        )
        viz = OvercookedVisualizer(tile_size=75, mdp=env._mdp)
        if not viz.is_available:
            print("Warning: StateVisualizer unavailable — rendering disabled.")
            viz = None

    # Pre-register the layout so phi/psi are non-zero from episode 1
    gen._pref_gen._hbm.register_layout(DEFAULT_HUMAN, layout_spec.name)
    gen._pref_gen._current_layout_name = layout_spec.name

    pause_ms = max(1, 1000 // args.fps)
    frame_idx = [0]

    # Shared render state (mutated by episode loop and render callback)
    render_state = {
        "episode": 1,
        "ep_score": 0.0,
        "total_score": 0.0,
        "last_task_score": 0.0,
        "last_subtask": "",
        "last_actor": "",
        "accuracy": 0.0,
    }

    # ------------------------------------------------------------------
    # Render callback — called on every primitive step inside OvercookedEnv
    # ------------------------------------------------------------------
    def render_frame(oc_state: Any, joint_action: Any, subtask: str, actor: str) -> None:
        if viz is None:
            return

        hbm = gen._pref_gen._hbm
        layout = layout_spec.name
        phi = hbm.get_phi(DEFAULT_HUMAN, layout, subtask)
        psi = hbm.get_running_psi(DEFAULT_HUMAN, subtask)
        true_pref = preferred_actor.get(subtask, "?")
        match = "correct" if actor == true_pref else "WRONG"

        p0_held = oc_state.players[0].held_object.name if oc_state.players[0].held_object else "empty"
        p1_held = oc_state.players[1].held_object.name if oc_state.players[1].held_object else "empty"

        # Summarise pot states for the HUD
        try:
            from multitask_personalization.envs.overcooked.layouts import get_layout as _gl
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld as _OC
            _mdp = env._mdp
            pot_states = _mdp.get_pot_states(oc_state)
            pot_summary_parts = []
            for key in ("empty", "1_items", "2_items", "3_items", "cooking", "ready"):
                n = len(pot_states.get(key, []))
                if n:
                    # For cooking pots, show the cook tick
                    if key == "cooking":
                        ticks = []
                        for pos in pot_states["cooking"]:
                            try:
                                soup = oc_state.get_object(pos)
                                ticks.append(f"{soup._cooking_tick}/{soup.cook_time}")
                            except Exception:
                                ticks.append("?")
                        pot_summary_parts.append(f"cooking({','.join(ticks)})")
                    else:
                        pot_summary_parts.append(f"{key}:{n}")
            pot_summary = "  ".join(pot_summary_parts) if pot_summary_parts else "none"
        except Exception:
            pot_summary = "n/a"

        hud = {
            # Episode context
            "episode": f"{render_state['episode']}/{args.episodes}",
            "ep_score": f"{render_state['ep_score']:+.2f}",
            "total_score": f"{render_state['total_score']:+.2f}",
            "accuracy": f"{render_state['accuracy']:.0%}",
            # Current subtask assignment
            "subtask": subtask,
            "assigned_to": f"{actor} ({match}, true={true_pref})",
            "last_score": f"{render_state['last_task_score']:+.2f}",
            # Learned belief about this subtask
            "phi": f"{phi:+.2f}  (>0=human, <0=robot)",
            "session_psi": f"{psi:+.2f}",
            # Agent states
            "robot_blue": p0_held,
            "human_green": p1_held,
            # Pot state
            "pots": pot_summary,
        }

        if args.save_dir:
            path = os.path.join(args.save_dir, f"frame_{frame_idx[0]:05d}.png")
            viz.save_frame(oc_state, path=path, hud=hud)
        else:
            viz.show_frame(oc_state, hud=hud, pause_ms=pause_ms)

        frame_idx[0] += 1

    env.render_callback = render_frame

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    total_score = 0.0  # also mirrored in render_state["total_score"]
    csp_correct = 0
    csp_total = 0

    for ep in range(args.episodes):
        obs, ep_info = env.reset()
        gen._pref_gen._current_layout_name = layout_spec.name
        done = False
        ep_score = 0.0
        ep_steps = 0
        render_state["episode"] = ep + 1
        render_state["ep_score"] = 0.0

        print(f"Episode {ep+1}/{args.episodes}  "
              f"session={ep_info['session_type']}")

        while not done:
            if obs.current_subtask is None:
                break

            # CSP assigns actor for this subtask
            csp, samplers, policy, initialization = gen.generate(obs)
            solution = solver.solve(csp, initialization, samplers)
            if solution is None:
                hbm = gen._pref_gen._hbm
                actor = hbm.preferred_actor(DEFAULT_HUMAN, layout_spec.name, obs.current_subtask)
                from multitask_personalization.envs.overcooked.overcooked_env import OvercookedAction
                action = OvercookedAction(actor=actor)
            else:
                policy.reset(solution)
                action = policy.step(obs)

            # Track whether CSP matched ground truth
            true_pref = preferred_actor.get(obs.current_subtask, "robot")
            if action.actor == true_pref:
                csp_correct += 1
            csp_total += 1
            render_state["accuracy"] = csp_correct / csp_total if csp_total > 0 else 0.0

            next_obs, task_score, done, _, info = env.step(action)
            gen.observe_transition(obs, action, next_obs, done=done, info=info)

            ep_score += task_score
            total_score += task_score
            ep_steps += 1
            render_state["ep_score"] = ep_score
            render_state["total_score"] = total_score
            render_state["last_task_score"] = task_score
            render_state["last_subtask"] = info.get("last_subtask", "")
            render_state["last_actor"] = info.get("last_actor", "")

            subtask_done = info.get("last_subtask", "?")
            actor_done = info.get("last_actor", "?")
            completed = info.get("subtask_completed", True)
            true = preferred_actor.get(subtask_done, "?")
            match = "✓" if actor_done == true else "✗"
            done_flag = "" if completed else " [TIMEOUT]"
            print(f"  {match} {subtask_done:20s} → {actor_done:6s}  "
                  f"(true={true:6s})  score={task_score:+.2f}{done_flag}")

            obs = next_obs

        accuracy = csp_correct / csp_total if csp_total > 0 else 0.0
        print(f"  Episode score: {ep_score:+.2f}  "
              f"cumulative accuracy: {accuracy:.1%}\n")

    # ------------------------------------------------------------------
    # Final preference snapshot
    # ------------------------------------------------------------------
    gen._pref_gen._current_layout_name = layout_spec.name
    snapshot = gen.get_pref_snapshot()
    print("\n=== Learned preferences (P(human|subtask)) ===")
    for s, probs in snapshot.items():
        bar = "█" * int(probs["human"] * 20)
        true = preferred_actor.get(s, "?")
        print(f"  {s:20s}  P(human)={probs['human']:.2f}  {bar:<20s}  true={true}")

    print(f"\nTotal episodes: {args.episodes}")
    print(f"Average task score: {total_score / args.episodes:+.2f}")
    print(f"CSP accuracy: {csp_correct}/{csp_total} = {csp_correct/csp_total:.1%}")

    if args.save_dir:
        print(f"\nSaved {frame_idx[0]} frames to {args.save_dir}/")


if __name__ == "__main__":
    main()
