"""Test configuration parameters for spices CSP tests."""

import logging

PARAMETERS = {
    "num_episodes": 150,       
    "num_test_episodes": 25, 
    "train_frac": 0.6,       
    "num_seeds": 10,
    "profile": "ChefComplex",
    "recipe_name": "FortyStepFeast",
    "env_seed": 123,
    "csp_seed": 456,
    "logging_level": logging.INFO,
    "num_humans": 3,
    "hidden_hbm_config_names": [
        "SpiceSpecificHuman",
        "SpiceSpecificHumanHeatSeeking",
        "SpiceSpecificHumanAromaticGentle",
    ],
}
