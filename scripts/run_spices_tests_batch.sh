#!/usr/bin/env bash
set -u

# Runs all three SPICES CSP experiments sequentially and archives outputs
# per test so results are not overwritten.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="spices_batch_${TS}"
RUN_DIR="logs/spices_test_reports/runs/${RUN_ID}"
RAW_DIR="logs/spices_test_reports"
MASTER_LOG="${RUN_DIR}/master.log"
PYTEST_CMD="${PYTEST_CMD:-pytest -s}"
export RUN_DIR

mkdir -p "$RUN_DIR"

{
  echo "== SPICES batch run =="
  echo "run_id: ${RUN_ID}"
  echo "timestamp: ${TS}"
  echo "repo: ${ROOT_DIR}"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python: $(python --version 2>&1)"
  echo "pytest_cmd: ${PYTEST_CMD}"
  echo ""
  echo "-- test config snapshot --"
  cp "src/multitask_personalization/envs/spices/config/test_configs.py" "${RUN_DIR}/test_configs.py.snapshot"
  echo "saved: ${RUN_DIR}/test_configs.py.snapshot"
  echo ""
  echo "-- key parameter snapshot (JSON) --"
  python - <<'PY'
import json
from multitask_personalization.envs.spices.config.test_configs import PARAMETERS
from multitask_personalization.envs.spices.config.spices_config import DEFAULT_CONFIG

snapshot = {
    "test_parameters": {
        "num_episodes": PARAMETERS.get("num_episodes"),
        "num_test_episodes": PARAMETERS.get("num_test_episodes"),
        "train_frac": PARAMETERS.get("train_frac"),
        "num_seeds": PARAMETERS.get("num_seeds"),
        "profile": PARAMETERS.get("profile"),
        "recipe_name": PARAMETERS.get("recipe_name"),
        "num_humans": PARAMETERS.get("num_humans"),
        "env_seed": PARAMETERS.get("env_seed"),
        "csp_seed": PARAMETERS.get("csp_seed"),
        "hidden_hbm_config_names": PARAMETERS.get("hidden_hbm_config_names"),
    },
    "spices_config": {
        "hbm": {
            "sigma0": DEFAULT_CONFIG.hbm.sigma0,
            "sigma_h": DEFAULT_CONFIG.hbm.sigma_h,
            "sigma_r": DEFAULT_CONFIG.hbm.sigma_r,
            "sigma_obs": DEFAULT_CONFIG.hbm.sigma_obs,
            "sigma_mood": DEFAULT_CONFIG.hbm.sigma_mood,
            "psi_decay": DEFAULT_CONFIG.hbm.psi_decay,
            "update_theta_mu_every_n_episodes": DEFAULT_CONFIG.hbm.update_theta_mu_every_n_episodes,
            "n_mc_samples": DEFAULT_CONFIG.hbm.n_mc_samples,
            "n_phi_steps": DEFAULT_CONFIG.hbm.n_phi_steps,
            "n_theta_steps": DEFAULT_CONFIG.hbm.n_theta_steps,
            "lr_phi": DEFAULT_CONFIG.hbm.lr_phi,
            "lr_theta": DEFAULT_CONFIG.hbm.lr_theta,
            "lr_hyper": DEFAULT_CONFIG.hbm.lr_hyper,
        },
        "mood": {
            "mood_prior": DEFAULT_CONFIG.mood.mood_prior,
            "psi_true_mood_mean_abs": DEFAULT_CONFIG.mood.psi_true_mood_mean_abs,
            "psi_true_mood_std": DEFAULT_CONFIG.mood.psi_true_mood_std,
            "psi_true_neutral_std": DEFAULT_CONFIG.mood.psi_true_neutral_std,
        },
        "satisfaction": {
            "base_satisfaction_bias": DEFAULT_CONFIG.satisfaction.base_satisfaction_bias,
            "satisfaction_logit_temperature": DEFAULT_CONFIG.satisfaction.satisfaction_logit_temperature,
            "satisfaction_beta_kappa": DEFAULT_CONFIG.satisfaction.satisfaction_beta_kappa,
        },
    },
}
print(json.dumps(snapshot, indent=2))
PY
  python - <<'PY'
import json
from pathlib import Path
from multitask_personalization.envs.spices.config.test_configs import PARAMETERS
from multitask_personalization.envs.spices.config.spices_config import DEFAULT_CONFIG

snapshot = {
    "test_parameters": dict(PARAMETERS),
    "spices_config": {
        "hbm": DEFAULT_CONFIG.hbm.__dict__,
        "mood": DEFAULT_CONFIG.mood.__dict__,
        "update": DEFAULT_CONFIG.update.__dict__,
        "satisfaction": DEFAULT_CONFIG.satisfaction.__dict__,
    },
}
import os
run_dir = Path(os.environ["RUN_DIR"])
(run_dir / "key_parameters.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
print(f"saved: {run_dir / 'key_parameters.json'}")
PY
  echo ""
} | tee -a "$MASTER_LOG"

run_one_test() {
  local test_name="$1"
  local pytest_target="$2"
  local test_dir="${RUN_DIR}/${test_name}"
  local test_log="${test_dir}/pytest.log"
  mkdir -p "$test_dir"

  echo "== Running ${test_name} ==" | tee -a "$MASTER_LOG"
  echo "target: ${pytest_target}" | tee -a "$MASTER_LOG"
  echo "start: $(date)" | tee -a "$MASTER_LOG"

  ${PYTEST_CMD} "${pytest_target}" 2>&1 | tee "$test_log"
  local exit_code=${PIPESTATUS[0]}

  echo "end: $(date)" | tee -a "$MASTER_LOG"
  echo "exit_code: ${exit_code}" | tee -a "$MASTER_LOG"

  # Archive current report artifacts produced by this test (if present).
  cp -f "${RAW_DIR}/single_recipe_metrics.csv" "${test_dir}/" 2>/dev/null || true
  cp -f "${RAW_DIR}/single_recipe_metrics.json" "${test_dir}/" 2>/dev/null || true
  cp -f "${RAW_DIR}/single_recipe_metrics_episode_satisfaction.csv" "${test_dir}/" 2>/dev/null || true
  cp -f "${RAW_DIR}/cross_transfer_metrics.csv" "${test_dir}/" 2>/dev/null || true
  cp -f "${RAW_DIR}/cross_transfer_metrics.json" "${test_dir}/" 2>/dev/null || true
  cp -f "${RAW_DIR}/multi_human_metrics.csv" "${test_dir}/" 2>/dev/null || true
  cp -f "${RAW_DIR}/multi_human_metrics.json" "${test_dir}/" 2>/dev/null || true

  echo "" | tee -a "$MASTER_LOG"
  return "${exit_code}"
}

batch_exit=0

run_one_test "single_recipe" "tests/envs/test_spices_csp.py::test_spices_csp_single_recipe" || batch_exit=1
run_one_test "cross_transfer" "tests/envs/test_spices_csp.py::test_spices_csp_cross_transfer" || batch_exit=1
run_one_test "multi_human" "tests/envs/test_spices_csp.py::test_spices_csp_multi_human" || batch_exit=1

echo "== Batch complete ==" | tee -a "$MASTER_LOG"
echo "run_dir: ${RUN_DIR}" | tee -a "$MASTER_LOG"
echo "batch_exit: ${batch_exit}" | tee -a "$MASTER_LOG"

exit "${batch_exit}"
