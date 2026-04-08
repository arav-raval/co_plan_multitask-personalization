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
    p.add_argument("--save-video", default=None, dest="save_video",
                   help="Save MP4 video to this path (e.g. demo.mp4)")
    p.add_argument("--video-fps", type=int, default=16, dest="video_fps",
                   help="FPS for saved video (default 16, higher = faster playback)")
    p.add_argument("--no-render", action="store_true", dest="no_render")
    p.add_argument("--approach", default="hbm",
                   choices=["hbm", "cbtl", "flat"],
                   help="Preference model: hbm (ours), cbtl (baseline), flat (no hierarchy)")
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

    from multitask_personalization.envs.overcooked.overcooked_hbm import (
        OvercookedPreferenceModel,
    )

    # Build a ground-truth HBM so _human_claims() uses sigmoid(phi + psi_true)
    # instead of a deterministic preferred_actor lookup.
    # IMPORTANT: don't pass layouts= to constructor — register_layout initializes
    # phi from theta, so it must run AFTER set_theta (same pattern as spices).
    hidden_hbm = OvercookedPreferenceModel(
        subtasks=subtasks,
        layouts=[],
        sigma_r=hidden_cfg.sigma_r,
        sigma_h=hidden_cfg.sigma_h,
        sigma0=hidden_cfg.sigma0,
        sigma_obs=hidden_cfg.sigma_obs,
    )
    hidden_hbm.set_theta(DEFAULT_HUMAN, theta, sigma_h=hidden_cfg.sigma_h)
    hidden_hbm.register_layout(DEFAULT_HUMAN, layout_spec.name)

    hidden_spec = OvercookedHiddenSpec(
        preferred_actor=preferred_actor,
        hidden_hbm=hidden_hbm,
    )

    approach_names = {"hbm": "HBM (Ours)", "cbtl": "CBTL Classifier", "flat": "Flat Bayesian"}
    print(f"\n=== Overcooked CSP Demo ===")
    print(f"Layout:     {layout_spec.name}  ({layout_spec.layout_name})")
    print(f"Human type: {args.human_type}")
    print(f"Approach:   {approach_names[args.approach]}")
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
    # CSP generator (with selectable preference model)
    # ------------------------------------------------------------------
    preference_model = None
    if args.approach == "cbtl":
        from multitask_personalization.envs.overcooked.overcooked_baselines import (
            OvercookedCBTLClassifierModel,
        )
        preference_model = OvercookedCBTLClassifierModel(
            subtasks=subtasks, layouts=[layout_spec.name]
        )
    elif args.approach == "flat":
        from multitask_personalization.envs.overcooked.overcooked_baselines import (
            OvercookedFlatPreferenceModel,
        )
        preference_model = OvercookedFlatPreferenceModel(
            subtasks=subtasks, layouts=[layout_spec.name]
        )

    explore = "exploit-only" if (args.eval or args.approach != "hbm") else "max-entropy"
    gen = OvercookedAssignCSPGenerator(
        subtask_list=subtasks,
        layout_list=[layout_spec.name],
        human_id=DEFAULT_HUMAN,
        seed=args.seed,
        verbose=True,
        preference_model=preference_model,
        explore_method=explore,
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
    need_viz = (not args.no_render) or args.save_video
    if need_viz:
        from multitask_personalization.envs.overcooked.overcooked_visualizer import (
            OvercookedVisualizer,
        )
        viz = OvercookedVisualizer(tile_size=75, mdp=env._mdp)
        if not viz.is_available:
            print("Warning: StateVisualizer unavailable — rendering disabled.")
            viz = None

    # Video frame buffer (collected during rendering, written at end).
    video_frames: list[Any] = []

    # Pre-register the layout so phi/psi are non-zero from episode 1
    gen._pref_gen._hbm.register_layout(DEFAULT_HUMAN, layout_spec.name)
    gen._pref_gen._current_layout_name = layout_spec.name

    pause_ms = max(1, 1000 // args.fps)
    frame_idx = [0]

    # Shared render state (mutated by episode loop and render callback)
    render_state: dict[str, Any] = {
        "episode": 1,
        "ep_correct": 0,
        "ep_total": 0,
        "last_subtask": "",
        "last_actor": "",
        "accuracy": 0.0,
        "time_left": 400,
        "session_type": "neutral",
    }

    # Short display names for subtasks (unique, readable)
    _SUBTASK_SHORT = {
        "fetch_ingredient": "Fetch Ingr.",
        "load_pot": "Load Pot",
        "fetch_dish": "Fetch Dish",
        "pickup_soup": "Pickup Soup",
        "deliver": "Deliver",
    }

    # ------------------------------------------------------------------
    # Render callback — called on every primitive step inside OvercookedEnv
    # ------------------------------------------------------------------
    def render_frame(oc_state: Any, joint_action: Any, subtask: str, actor: str) -> None:
        if viz is None:
            return

        hbm = gen._pref_gen._hbm
        layout = layout_spec.name

        ep_acc = (
            render_state["ep_correct"] / render_state["ep_total"]
            if render_state["ep_total"] > 0 else 0.0
        )

        # --- Left side: status lines ---
        session = render_state["session_type"]
        status_lines = [
            f"Episode {render_state['episode']}/{args.episodes}"
            f"    Steps left: {render_state['time_left']}",
            f"Session: {session}",
            f"Accuracy: ep {ep_acc:.0%}  all {render_state['accuracy']:.0%}",
            f"Task: {subtask}  ->  {actor.upper()}",
        ]

        # --- Right side: preference table + config footer ---
        from multitask_personalization.envs.overcooked.config.overcooked_config import DEFAULT_CONFIG
        config_lines = [
            f"Layout: {layout_spec.name}",
            f"Human: {args.human_type}",
            f"Approach: {approach_names[args.approach]}",
            f"Session: {DEFAULT_CONFIG.session.session_profile_name}",
        ]

        table_header = ["Subtask", "True", "Learned"]
        table_rows = []
        for s in subtasks:
            short = _SUBTASK_SHORT.get(s, s[:11])
            tp = 1.0 / (1.0 + np.exp(-theta[s]))
            lp_phi = hbm.get_phi(DEFAULT_HUMAN, layout, s)
            lp_psi = hbm.get_running_psi(DEFAULT_HUMAN, s)
            lp = 1.0 / (1.0 + np.exp(-(lp_phi + lp_psi)))
            true_a = "H" if preferred_actor[s] == "human" else "R"
            learned_a = "H" if lp >= 0.5 else "R"
            table_rows.append([
                short,
                f"{true_a} {tp:.0%}",
                f"{learned_a} {lp:.0%}",
            ])

        if args.save_dir:
            # PNG fallback: use old-style HUD
            hud = {"status": " | ".join(status_lines)}
            path = os.path.join(args.save_dir, f"frame_{frame_idx[0]:05d}.png")
            viz.save_frame(oc_state, path=path, hud=hud)
        elif not args.no_render:
            viz.show_composite(
                oc_state, status_lines, table_rows, table_header,
                table_footer=config_lines, pause_ms=pause_ms,
            )

        # Capture frame for video.
        if args.save_video and viz is not None:
            arr = viz.composite_to_array(
                oc_state, status_lines, table_rows, table_header,
                table_footer=config_lines,
            )
            if arr is not None:
                video_frames.append(arr)

        frame_idx[0] += 1

    env.render_callback = render_frame

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    # Cumulative metrics across all episodes
    all_correct = 0       # correct CSP predictions (non-forced only)
    all_decisions = 0     # total CSP decisions (non-forced only)
    all_total = 0         # total steps (all types)
    all_completed = 0     # subtasks that actually completed (no timeout)
    all_conflicts = 0     # conflict steps
    all_timeouts = 0      # timed-out subtasks
    all_deliveries = 0    # soups successfully delivered

    for ep in range(args.episodes):
        obs, ep_info = env.reset()
        gen._pref_gen._current_layout_name = layout_spec.name
        done = False
        ep_steps = 0
        ep_completed = 0
        ep_conflicts = 0
        ep_timeouts = 0
        render_state["episode"] = ep + 1
        render_state["ep_correct"] = 0
        render_state["ep_total"] = 0
        render_state["session_type"] = ep_info.get("session_type", "neutral")

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
                pref = hbm.preferred_actor(DEFAULT_HUMAN, layout_spec.name, obs.current_subtask)
                from multitask_personalization.envs.overcooked.overcooked_env import OvercookedAction
                action = OvercookedAction(flag=1 if pref == "human" else 0)
            else:
                policy.reset(solution)
                action = policy.step(obs)

            next_obs, _rew, done, _, info = env.step(action)
            gen.observe_transition(obs, action, next_obs, done=done, info=info)

            subtask_done = info.get("last_subtask", "?")
            actor_done = info.get("last_actor", "?")
            completed = info.get("subtask_completed", True)
            conflict = info.get("conflict", False)
            is_forced = info.get("forced", False)
            pred_correct = info.get("prediction_correct", False)

            ep_steps += 1
            all_total += 1
            # Only count CSP decisions (non-forced) for accuracy.
            if not is_forced:
                all_decisions += 1
                render_state["ep_total"] += 1
                if pred_correct:
                    all_correct += 1
                    render_state["ep_correct"] += 1
            if completed and not conflict:
                ep_completed += 1
                all_completed += 1
            if conflict:
                ep_conflicts += 1
                all_conflicts += 1
            if not completed:
                ep_timeouts += 1
                all_timeouts += 1

            render_state["accuracy"] = all_correct / all_decisions if all_decisions > 0 else 0.0
            render_state["last_subtask"] = subtask_done
            render_state["last_actor"] = actor_done
            render_state["time_left"] = info.get("time_left", -1)

            # Per-step terminal output
            csp_predicted = "human" if action.flag == 1 else "robot"
            satisfaction = info.get("satisfaction", 0.0)
            if is_forced:
                # Forced continuation — not a preference decision
                print(f"  ~ {subtask_done:20s}  {actor_done:6s} continues (holding item)")
            else:
                mark = "✓" if pred_correct else "✗"
                flags = ""
                if not completed: flags += " [TIMEOUT]"
                if conflict: flags += " [CONFLICT]"
                print(f"  {mark} {subtask_done:20s}  predict={csp_predicted:6s} → {actor_done:6s}  "
                      f"sat={satisfaction:+.2f}{flags}")

            obs = next_obs

        ep_deliveries = getattr(env, "_deliveries", 0)
        ep_acc = render_state["ep_correct"] / render_state["ep_total"] if render_state["ep_total"] > 0 else 0.0
        overall_acc = all_correct / all_total if all_total > 0 else 0.0
        print(f"  --- Episode {ep+1}: "
              f"deliveries={ep_deliveries}  "
              f"completed={ep_completed}/{ep_steps}  "
              f"accuracy={ep_acc:.0%}  "
              f"conflicts={ep_conflicts}  timeouts={ep_timeouts}  "
              f"[overall accuracy={overall_acc:.0%}]\n")

    # ------------------------------------------------------------------
    # Final preference snapshot
    # ------------------------------------------------------------------
    gen._pref_gen._current_layout_name = layout_spec.name
    snapshot = gen.get_pref_snapshot()
    print("\n=== Learned vs True Preferences ===")
    print(f"  {'Subtask':20s}  {'True':>6s}  {'Learned':>8s}  {'True P':>7s}  {'Learned P':>9s}  {'':20s}")
    print(f"  {'─' * 80}")
    for s, probs in snapshot.items():
        true_pref = preferred_actor.get(s, "?")
        true_p = 1.0 / (1.0 + np.exp(-theta[s]))
        learned_p = probs["human"]
        bar_true = "▓" * int(true_p * 20)
        bar_learned = "░" * int(learned_p * 20)
        match = "✓" if (learned_p >= 0.5) == (true_pref == "human") else "✗"
        print(f"  {match} {s:19s}  {true_pref:>6s}  "
              f"{'human' if learned_p >= 0.5 else 'robot':>8s}  "
              f"{true_p:>6.0%}  {learned_p:>8.0%}  "
              f"{bar_true}|{bar_learned}")

    # Final summary
    overall_acc = all_correct / all_decisions if all_decisions > 0 else 0.0
    completion_rate = all_completed / all_total if all_total > 0 else 0.0
    avg_sat = float(np.mean(env._satisfaction_history)) if env._satisfaction_history else 0.0
    print(f"\n  === Summary ({args.episodes} episodes) ===")
    print(f"  Prediction accuracy : {all_correct}/{all_decisions} = {overall_acc:.0%}  (CSP decisions only)")
    print(f"  Completion rate     : {all_completed}/{all_total} = {completion_rate:.0%}")
    print(f"  Avg satisfaction    : {avg_sat:+.2f}")
    print(f"  Conflicts           : {all_conflicts}")
    print(f"  Timeouts            : {all_timeouts}")

    if args.save_dir:
        print(f"\nSaved {frame_idx[0]} frames to {args.save_dir}/")

    # ------------------------------------------------------------------
    # Write video
    # ------------------------------------------------------------------
    if args.save_video and video_frames:
        try:
            import imageio.v3 as iio
            out_path = args.save_video
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            iio.imwrite(
                out_path,
                video_frames,
                fps=args.video_fps,
                codec="libx264",
                plugin="pyav",
            )
            print(f"\nSaved video ({len(video_frames)} frames, "
                  f"{args.video_fps} fps, "
                  f"{len(video_frames)/args.video_fps:.1f}s) "
                  f"to {out_path}")
        except ImportError:
            # Fallback: try imageio v2 API
            try:
                import imageio
                out_path = args.save_video
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                writer = imageio.get_writer(out_path, fps=args.video_fps)
                for frame in video_frames:
                    writer.append_data(frame)
                writer.close()
                print(f"\nSaved video ({len(video_frames)} frames, "
                      f"{args.video_fps} fps) to {out_path}")
            except Exception as e:
                print(f"\nFailed to write video: {e}")
                print("Frames were captured but video encoding failed. "
                      "Try: pip install imageio[ffmpeg]")


if __name__ == "__main__":
    main()
