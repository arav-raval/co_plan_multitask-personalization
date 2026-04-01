"""Test parameters for the Overcooked environment experiments."""

PARAMETERS: dict = {
    "num_episodes": 100,
    "num_test_episodes": 20,
    "train_frac": 0.6,
    "num_seeds": 5,
    "layout_name": "CrampedRoom",
    "env_seed": 123,
    "csp_seed": 456,
    "num_humans": 3,
    "hidden_hbm_config_names": [
        "RealisticCook",
        "PrefsPrep",
        "PrefsDelivery",
    ],
}
