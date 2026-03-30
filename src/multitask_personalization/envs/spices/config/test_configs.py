"""Test configuration parameters for spices CSP tests."""

import logging

PARAMETERS = {
    "num_episodes": 50,       # Training episodes per recipe (was 200)
    "num_test_episodes": 20,  # Evaluation episodes per test recipe (was 50)
    "train_frac": 0.75,       # 17/23 train, 6 test for ChefComplex
    "num_seeds": 5,           # Seeds (was 5) — reduces runtime ~40%
    "num_epochs": 1,
    "profile": "ChefComplex",
    "recipe_name": "FortyStepFeast",
    "env_seed": 123,
    "csp_seed": 369,
    "logging_level": logging.INFO,
    "num_humans": 3,
    "hidden_hbm_config_name": "SpiceSpecificHuman",
    "hidden_hbm_config_names": None,
}
