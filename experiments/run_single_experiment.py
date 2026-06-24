import logging
import math
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from multitask_personalization.envs.overcooked.layouts import ALL_SUBTASKS
from multitask_personalization.methods.approach import BaseApproach


PREFERENCE_SHIFT_ENVS = [
    "cooking-nonstationary",
    "spices_nonstationary",
    "spices_shift_soft",
    "spices_shift_medium",
    "spices_shift_strong",
    "spices_shift_random",
]
EPISODIC_RESET_TRAIN_ENVS = ["spices", "overcooked"]

_EXPERIMENT_CONF_DIR = Path(__file__).resolve().parent / "conf"


def _force_enumeration_csp_for_spices(cfg: DictConfig) -> None:
    env_name = str(cfg.get("env_name", ""))
    if not (env_name.startswith("spices") or env_name == "overcooked"):
        return
    target = str(OmegaConf.select(cfg, "csp_solver._target_") or "")
    if "EnumerationCSPSolver" in target:
        return
    enum_path = _EXPERIMENT_CONF_DIR / "csp_solver" / "enumeration.yaml"
    logging.info(
        "env=spices: switching CSP solver to EnumerationCSPSolver (binary domain; "
        "Hydra cannot override root csp_solver from env YAML)."
    )
    cfg.csp_solver = OmegaConf.merge(
        OmegaConf.load(enum_path),
        OmegaConf.create({"seed": cfg.seed}),
    )


def _sync_spices_scene_spec(approach: BaseApproach, env: gym.Env) -> None:
    approach._scene_spec = env.unwrapped.scene_spec 


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    _force_enumeration_csp_for_spices(cfg)

    logging.info(
        f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
    )
    logging.info("Full config:")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logging.info(OmegaConf.to_yaml(resolved_cfg))
    OmegaConf.save(resolved_cfg, cfg.config_file)
    logging.info(f"Saved config to {cfg.config_file}")
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)
    logging.info(f"Created model directory at {cfg.model_dir}")
    saved_state_dir = Path(cfg.saved_state_dir)
    saved_state_dir.mkdir(exist_ok=True)
    logging.info(f"Created saved state directory at {cfg.saved_state_dir}")

    assert cfg.env.max_environment_steps % cfg.env.eval_frequency == 0

    if cfg.wandb.enable:
        wandb.config = resolved_cfg
        assert cfg.wandb.entity is not None
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group if cfg.wandb.group else None,
            name=cfg.wandb.run_name if cfg.wandb.run_name else None,
            dir=cfg.wandb.dir,
        )

    # Create training environment
    train_env_cfg = OmegaConf.merge(cfg.env.env, cfg.env.train_env)
    train_env = hydra.utils.instantiate(train_env_cfg, seed=cfg.seed, eval_mode=False)
    assert isinstance(train_env, gym.Env)
    if cfg.record_train_videos:
        train_env = gym.wrappers.RecordVideo(
            train_env, str(Path(cfg.video_dir) / "train")
        )
    train_env.action_space.seed(cfg.seed)

    # Create eval environment
    eval_seed = cfg.seed + cfg.eval_seed_offset
    eval_env_cfg = OmegaConf.merge(cfg.env.env, cfg.env.eval_env)
    eval_env = hydra.utils.instantiate(eval_env_cfg, seed=eval_seed, eval_mode=True)
    assert isinstance(eval_env, gym.Env)
    if cfg.record_eval_videos:
        eval_env = gym.wrappers.RecordVideo(eval_env, str(Path(cfg.video_dir) / "eval"))
    eval_env.action_space.seed(eval_seed)

    train_approach = hydra.utils.instantiate(
        cfg.approach,
        train_env.unwrapped.scene_spec,
        train_env.action_space,
        seed=cfg.seed,
    )
    assert isinstance(train_approach, BaseApproach)
    train_approach.train()
    eval_approach = hydra.utils.instantiate(
        cfg.approach,
        eval_env.unwrapped.scene_spec,
        eval_env.action_space,
        seed=eval_seed,
    )
    assert isinstance(eval_approach, BaseApproach)
    eval_approach.eval()

    # Log training and eval metrics
    train_metrics: list[dict[str, float]] = []
    eval_metrics: list[dict[str, float]] = []

    shift_occurred_since_last_eval = False

    try:
        # Initial reset of training environment and approach
        obs, info = train_env.reset()
        train_approach.reset(obs, info)
        if cfg.env_name.startswith("spices"):
            _sync_spices_scene_spec(train_approach, train_env)

        # Main training and eval loop
        for t in range(cfg.env.max_environment_steps + 1):
            if t % cfg.train_logging_interval == 0:
                logging.info(f"Starting training step {t}")

            if cfg.env.eval_frequency > 0 and t % cfg.env.eval_frequency == 0:
                logging.info(
                    f"===================== Evaluation at step {t} ====================="
                )
                # Save the models from the training approach and load them into the eval approach
                step_model_dir = model_dir / str(t)
                step_model_dir.mkdir(exist_ok=True)
                train_approach.save(step_model_dir)
                eval_approach.load(step_model_dir)

                if cfg.env_name in PREFERENCE_SHIFT_ENVS:
                    if cfg.env_name == "cooking-nonstationary":
                        eval_env._hidden_spec.meal_preference_model.sync_variables(
                            train_env._hidden_spec.meal_preference_model
                        )
                    elif cfg.env_name.startswith("spices_shift") or cfg.env_name == "spices_nonstationary":
                        # Sync eval's hidden HBM with training env (may have shifted).
                        eval_env.unwrapped._hidden_spec.hidden_hbm = (
                            train_env.unwrapped._hidden_spec.hidden_hbm
                        )
                # Transfer adaptation: run a few episodes with learning enabled on the eval env before measuring accuracy
                num_adapt = int(cfg.env.get("num_transfer_adaptation_episodes", 0))
                if num_adapt > 0 and cfg.env_name.startswith("overcooked"):
                    _transfer_adapt(eval_approach, eval_env, cfg, num_adapt)

                # Run evaluation
                if cfg.env_name.startswith("spices") or cfg.env_name.startswith("overcooked"):
                    neutral_eval = _evaluate_approach(
                        eval_approach,
                        eval_env,
                        cfg,
                        t,
                        (
                            shift_occurred_since_last_eval
                            if cfg.env_name in PREFERENCE_SHIFT_ENVS
                            else False
                        ),
                        force_neutral_mood=True,
                        metric_prefix="neutral_",
                        include_metadata=True,
                    )
                    natural_eval = _evaluate_approach(
                        eval_approach,
                        eval_env,
                        cfg,
                        t,
                        (
                            shift_occurred_since_last_eval
                            if cfg.env_name in PREFERENCE_SHIFT_ENVS
                            else False
                        ),
                        force_neutral_mood=False,
                        metric_prefix="natural_",
                        include_metadata=False,
                    )
                    step_eval_metrics = {**neutral_eval, **natural_eval}
                else:
                    step_eval_metrics = _evaluate_approach(
                        eval_approach,
                        eval_env,
                        cfg,
                        t,
                        (
                            shift_occurred_since_last_eval
                            if cfg.env_name in PREFERENCE_SHIFT_ENVS
                            else False
                        ),
                    )
                if cfg.wandb.enable:
                    wandb_metrics = {
                        f"eval/{k}": v for k, v in step_eval_metrics.items()
                    }
                    del wandb_metrics["eval/training_step"]
                    wandb.log(wandb_metrics, step=t)
                eval_metrics.append(step_eval_metrics)
                logging.info("Resuming training")
                logging.info(
                    "========================================================="
                )
                # Reset shift tracking
                shift_occurred_since_last_eval = False
            # Eval on the last time step but don't train anymore
            if t >= cfg.env.max_environment_steps:
                break
            # Continue training
            act = train_approach.step()
            obs, rew, env_terminated, env_truncated, info = train_env.step(act)
            assert np.isclose(rew, 0.0)
            
            done_for_learning = bool(env_terminated or env_truncated)
            train_approach.update(obs, float(rew), done_for_learning, info)

            if any(cfg.env_name.startswith(e) for e in EPISODIC_RESET_TRAIN_ENVS) and (
                env_terminated or env_truncated
            ):
                obs, info = train_env.reset()
                train_approach.reset(obs, info)
                if cfg.env_name.startswith("spices"):
                    _sync_spices_scene_spec(train_approach, train_env)

            # Track if any shift occurred during training
            preference_shift = False
            if cfg.env_name in PREFERENCE_SHIFT_ENVS:
                preference_shift = info.get("preference_shift", False)
                if preference_shift:
                    shift_occurred_since_last_eval = True

            step_train_metrics = {
                "step": t,
                "execution_time": t * cfg.env.dt,
                "user_satisfaction": info.get("user_satisfaction", np.nan),
                "env_video_should_pause": info.get("env_video_should_pause", False),
                "preference_shift": preference_shift,
                **train_approach.get_step_metrics(),
            }
            if cfg.wandb.enable:
                wandb_metrics = {f"train/{k}": v for k, v in step_train_metrics.items()}
                del wandb_metrics["train/step"]
                wandb.log(wandb_metrics, step=t)
            train_metrics.append(step_train_metrics)

        train_env.close()
        eval_env.close()

        # Aggregate and save results
        train_df = pd.DataFrame(train_metrics)
        train_df.to_csv(cfg.train_results_file)
        logging.info(f"Wrote out training results to {cfg.train_results_file}")

        eval_df = pd.DataFrame(eval_metrics)
        eval_df.to_csv(cfg.eval_results_file)
        logging.info(f"Wrote out eval results to {cfg.eval_results_file}")

        if cfg.wandb.enable:
            wandb.finish()

    except BaseException as e:
        logging.warning("Crashed! Saving environment states before finishing.")
        train_env.unwrapped.save_state(saved_state_dir / "crash_train_env_state.p")
        eval_env.unwrapped.save_state(saved_state_dir / "crash_eval_env_state.p")

        train_env.close()
        eval_env.close()

        # Aggregate and save results
        train_df = pd.DataFrame(train_metrics)
        train_df.to_csv(cfg.train_results_file)
        logging.info(
            f"Wrote out INCOMPLETE training results to {cfg.train_results_file}"
        )

        eval_df = pd.DataFrame(eval_metrics)
        eval_df.to_csv(cfg.eval_results_file)
        logging.info(f"Wrote out INCOMPLETE eval results to {cfg.eval_results_file}")

        logging.critical(e, exc_info=True)


def _switch_eval_human_id(eval_approach: BaseApproach, new_human_id: str) -> None:
    """Switch the eval approach's preference model to a new human_id"""
    try:
        # Register new human and layout if needed
        csp_gen = eval_approach._csp_generator  
        pref_gen = csp_gen._pref_gen
        hbm = pref_gen._hbm
        hbm.register_human(new_human_id)
        for layout_name in pref_gen._layout_list:
            hbm.register_layout(new_human_id, layout_name)

        # Switch the preference generator to use the new human_id
        pref_gen._human_id = new_human_id
        logging.info(
            f"[Transfer] Switched eval human_id to '{new_human_id}' "
            f"with layouts {pref_gen._layout_list}"
        )
    except AttributeError:
        logging.warning("Could not switch eval human_id (approach has no _csp_generator)")


def _transfer_adapt(
    eval_approach: BaseApproach,
    eval_env: gym.Env,
    cfg: DictConfig,
    num_episodes: int,
) -> None:
    """Run a short adaptation phase on the eval env with learning enabled"""
    if num_episodes <= 0:
        return

    # Switch human_id for multi-human experiments
    eval_human_id = str(cfg.env.get("eval_human_id", ""))
    if eval_human_id:
        _switch_eval_human_id(eval_approach, eval_human_id)

    eval_approach.train()
    obs, info = eval_env.reset()
    eval_approach.reset(obs, info)
    episode_count = 0
    for _ in range(num_episodes * cfg.env.max_eval_episode_length):
        act = eval_approach.step()
        obs, rew, terminated, truncated, info = eval_env.step(act)
        eval_approach.update(obs, float(rew), bool(terminated or truncated), info)
        if terminated or truncated:
            episode_count += 1
            if episode_count >= num_episodes:
                break
            obs, info = eval_env.reset()
            eval_approach.reset(obs, info)
    eval_approach.eval()


def _evaluate_approach(
    eval_approach: BaseApproach,
    eval_env: gym.Env,
    cfg: DictConfig,
    training_step: int,
    shift_occurred_since_last_eval: bool,
    force_neutral_mood: bool | None = None,
    metric_prefix: str = "",
    include_metadata: bool = True,
) -> dict[str, float]:
    """Evaluate the given approach and return metrics"""

    cumulative_user_satisfactions: list[float] = []
    eval_num_steps: list[int] = []
    spices_episode_avg_satisfactions: list[float] = []
    spices_episode_pred_accuracies: list[float] = []
    overcooked_deliveries: list[int] = []
    original_force_neutral: bool | None = None
    if cfg.env_name.startswith("spices") or cfg.env_name.startswith("overcooked") and force_neutral_mood is not None:
        original_force_neutral = bool(eval_env.unwrapped._hidden_spec.force_neutral_mood)
        eval_env.unwrapped._hidden_spec.force_neutral_mood = force_neutral_mood 

    logging.info("Starting evaluation")
    try:
        for eval_trial_idx in range(cfg.env.num_eval_trials):
            seed = cfg.seed + cfg.eval_seed_offset + eval_trial_idx
            obs, info = eval_env.reset(seed=seed)
            # Reset the approach
            eval_approach.reset(obs, info)
            if cfg.env_name.startswith("spices"):
                _sync_spices_scene_spec(eval_approach, eval_env)
            # Track prediction accuracy for overcooked
            overcooked_correct = 0
            overcooked_total = 0
            # Main eval loop
            cumulative_user_satisfaction = 0.0
            n_steps = 0
            for _ in range(cfg.env.max_eval_episode_length):
                act = eval_approach.step()
                obs, rew, terminated, truncated, info = eval_env.step(act)
                assert np.isclose(float(rew), 0.0)
                eval_approach.update(obs, float(rew), terminated, info)
                user_satisfaction = info.get("user_satisfaction", 0.0)
                cumulative_user_satisfaction += user_satisfaction
                n_steps += 1
                
                if cfg.env_name.startswith("overcooked") and info.get("last_subtask") is not None:
                    if not info.get("forced", False):
                        pred_correct = info.get("prediction_correct", False)
                        if pred_correct:
                            overcooked_correct += 1
                        overcooked_total += 1
                if terminated or truncated:
                    break
            if cfg.env_name.startswith("spices"):
                spices_episode_avg_satisfactions.append(
                    float(info.get("average_satisfaction", np.nan))
                )
                spices_episode_pred_accuracies.append(
                    float(info.get("prediction_accuracy", np.nan))
                )
            if cfg.env_name.startswith("overcooked") and overcooked_total > 0:
                spices_episode_pred_accuracies.append(
                    overcooked_correct / overcooked_total
                )
                overcooked_deliveries.append(
                    info.get("deliveries", 0)
                )
            cumulative_user_satisfactions.append(cumulative_user_satisfaction)
            eval_num_steps.append(n_steps)
    finally:
        if cfg.env_name.startswith("spices") or cfg.env_name.startswith("overcooked") and original_force_neutral is not None:
            eval_env.unwrapped._hidden_spec.force_neutral_mood = original_force_neutral  # pylint: disable=protected-access

    step_eval_metrics: dict[str, float] = {}
    if include_metadata:
        step_eval_metrics.update(
            {
                "training_step": training_step,
                "training_execution_time": training_step * cfg.env.dt,
                "preference_shift": shift_occurred_since_last_eval,
            }
        )
    mean_step_per_trial: list[float] = []
    for idx, cus in enumerate(cumulative_user_satisfactions):
        ns = eval_num_steps[idx]
        step_eval_metrics[f"{metric_prefix}eval_episode_{idx}_user_satisfaction"] = cus
        step_eval_metrics[f"{metric_prefix}eval_episode_{idx}_num_steps"] = float(ns)
        mstep = float(cus / ns) if ns > 0 else float("nan")
        step_eval_metrics[f"{metric_prefix}eval_episode_{idx}_mean_step_user_satisfaction"] = mstep
        mean_step_per_trial.append(mstep)
    step_eval_metrics[f"{metric_prefix}eval_mean_user_satisfaction"] = float(
        np.mean(cumulative_user_satisfactions)
    )
    step_eval_metrics[f"{metric_prefix}eval_mean_user_satisfaction_per_step"] = float(
        np.nanmean(mean_step_per_trial)
    )
    if cfg.env_name.startswith("spices") or cfg.env_name.startswith("overcooked"):
        if spices_episode_avg_satisfactions:
            step_eval_metrics[f"{metric_prefix}eval_mean_episode_average_satisfaction"] = float(
                np.nanmean(spices_episode_avg_satisfactions)
            )
        if spices_episode_pred_accuracies:
            step_eval_metrics[f"{metric_prefix}eval_mean_prediction_accuracy"] = float(
                np.nanmean(spices_episode_pred_accuracies)
            )
    
    if cfg.env_name.startswith("overcooked"):
        if overcooked_deliveries:
            step_eval_metrics[f"{metric_prefix}eval_mean_deliveries"] = float(
                np.mean(overcooked_deliveries)
            )
        try:
            true_prefs = eval_env.unwrapped.hidden_spec.preferred_actor
            
            csp_gen = eval_approach._csp_generator  
            hbm = csp_gen._pref_gen._hbm
            layout_name = eval_env.unwrapped.layout_spec.name
            human_id = csp_gen._pref_gen._human_id

            phi_errors = []
            for s in ALL_SUBTASKS[:5]:
                try:
                    learned_phi = hbm.get_phi(human_id, layout_name, s)
                    learned_p = 1.0 / (1.0 + math.exp(-learned_phi))
                    true_pref = true_prefs.get(s, "robot")
                    
                    true_phi = eval_env.unwrapped.hidden_spec.hidden_hbm.get_phi(
                        human_id, layout_name, s
                    )
                    true_p = 1.0 / (1.0 + math.exp(-float(true_phi)))
                    phi_errors.append(abs(learned_p - true_p))
                except Exception:
                    pass
            if phi_errors:
                step_eval_metrics[f"{metric_prefix}eval_phi_mae"] = float(np.mean(phi_errors))
        except Exception:
            pass
    return step_eval_metrics


if __name__ == "__main__":
    _main() 
