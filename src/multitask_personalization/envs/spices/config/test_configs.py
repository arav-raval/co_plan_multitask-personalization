"""Test configuration parameters for spices CSP tests."""

import logging

PARAMETERS = {
    "num_episodes": 500,  # Used for training / single_human
    "num_test_episodes": 100,
    "train_frac": 0.80,  # 16/20 = 0.80 keeps exactly 4 test recipes in ChefComplex
    "num_seeds": 2,
    "num_epochs": 1,
    "profile": "ChefComplex",
    "recipe_name": "FortyStepFeast",
    "env_seed": 123,
    "csp_seed": 369,
    "logging_level": logging.INFO,
    "num_humans": 3,  # Number of different humans (hidden HBMs) to test
    "hidden_hbm_config_name": "ConsistentHuman",  # Covers all recipe spices with ±1.5 theta
    "hidden_hbm_config_names": None,  # List of config names for multiple humans (if None, auto-generates)
}
