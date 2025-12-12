"""
hidden_hbm_configs.py
Central repository for hidden HBM configurations (true human preferences)

Each configuration defines:
- theta_mean: Fixed human-level preference for each spice (the "true" values to learn)
- Variance parameters: sigma0, sigma_h, sigma_r, sigma_obs
- base_satisfaction_bias: Base satisfaction bias for the model
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HiddenHBMConfig:
    """Configuration for a hidden HBM representing a human's true preferences."""
    name: str
    theta_mean: Dict[str, float]  # Fixed theta values for each spice
    sigma0: float = 1.0  # Global variance (Level 1: μ)
    sigma_h: float = 1.0  # Human-level variance (Level 2: θ)
    sigma_r: float = 1.0  # Recipe-level variance (Level 3: φ) - controls sampling variation
    sigma_obs: float = 1.0  # Observation noise
    base_satisfaction_bias: float = 3.0  # Base satisfaction bias
    
    def get_theta_for_spice(self, spice: str) -> float:
        """Get theta_mean for a specific spice, or 0.0 if not specified."""
        return self.theta_mean.get(spice, 0.0)


# ============================================================================
# PREDEFINED HUMAN CONFIGURATIONS
# ============================================================================

# Human with alternating preferences (simple pattern)
ALTERNATING_HUMAN = HiddenHBMConfig(
    name="AlternatingHuman",
    theta_mean={
        # Alternating pattern: positive for even indices, negative for odd
        # This will be filled dynamically based on spices in recipe
    },
    sigma_r=1.0,
    sigma_h=1.0,
)

# Human with strong preferences for first half of spices
HUMAN_PREFERS_FIRST_HALF = HiddenHBMConfig(
    name="HumanPrefersFirstHalf",
    theta_mean={
        # First half: strong human preference (2.0)
        # Second half: strong robot preference (-2.0)
        # This will be filled dynamically based on spices in recipe
    },
    sigma_r=1.0,
    sigma_h=1.0,
)

# Human with opposite pattern
HUMAN_PREFERS_SECOND_HALF = HiddenHBMConfig(
    name="HumanPrefersSecondHalf",
    theta_mean={
        # First half: strong robot preference (-2.0)
        # Second half: strong human preference (2.0)
        # This will be filled dynamically based on spices in recipe
    },
    sigma_r=1.0,
    sigma_h=1.0,
)

# Human with specific spice preferences (example with common spices)
SPICE_SPECIFIC_HUMAN = HiddenHBMConfig(
    name="SpiceSpecificHuman",
    theta_mean={
        # Strong human preferences
        "salt": 2.0,
        "garlic": 1.5,
        "onion": 1.0,
        "ginger": 1.5,
        "turmeric": 0.5,
        
        # Strong robot preferences
        "pepper": -2.0,
        "chili": -1.5,
        "cumin": -1.0,
        "coriander": -0.5,
        
        # Neutral/slight preferences
        "sugar": 0.3,
        "cinnamon": -0.3,
        "basil": 0.5,
        "olive_oil": 0.0,
        
        # Default for other spices
    },
    sigma_r=1.0,
    sigma_h=1.0,
)

# Human with moderate variation (smaller sigma_r for consistency)
CONSISTENT_HUMAN = HiddenHBMConfig(
    name="ConsistentHuman",
    theta_mean={
        # Similar pattern to SpiceSpecificHuman but with tighter variance
    },
    sigma_r=0.5,  # Smaller variance = more consistent across recipes
    sigma_h=0.8,
)

# Human with high variation (larger sigma_r)
VARIABLE_HUMAN = HiddenHBMConfig(
    name="VariableHuman",
    theta_mean={
        # Similar pattern but with higher variance
    },
    sigma_r=2.0,  # Larger variance = more variation across recipes
    sigma_h=1.5,
)

# Human with very strong preferences (extreme values)
STRONG_PREFERENCES_HUMAN = HiddenHBMConfig(
    name="StrongPreferencesHuman",
    theta_mean={
        "salt": 3.0,
        "garlic": 2.5,
        "onion": 2.0,
        "pepper": -3.0,
        "chili": -2.5,
    },
    sigma_r=1.0,
    sigma_h=1.0,
)

# Human with weak preferences (subtle differences)
WEAK_PREFERENCES_HUMAN = HiddenHBMConfig(
    name="WeakPreferencesHuman",
    theta_mean={
        "salt": 0.5,
        "garlic": 0.3,
        "onion": 0.2,
        "pepper": -0.5,
        "chili": -0.3,
    },
    sigma_r=0.8,
    sigma_h=0.6,
)

# ============================================================================
# CONFIGURATION LOOKUP
# ============================================================================

ALL_HIDDEN_HBM_CONFIGS = {
    "AlternatingHuman": ALTERNATING_HUMAN,
    "HumanPrefersFirstHalf": HUMAN_PREFERS_FIRST_HALF,
    "HumanPrefersSecondHalf": HUMAN_PREFERS_SECOND_HALF,
    "SpiceSpecificHuman": SPICE_SPECIFIC_HUMAN,
    "ConsistentHuman": CONSISTENT_HUMAN,
    "VariableHuman": VARIABLE_HUMAN,
    "StrongPreferencesHuman": STRONG_PREFERENCES_HUMAN,
    "WeakPreferencesHuman": WEAK_PREFERENCES_HUMAN,
}


def get_hidden_hbm_config(name: str) -> HiddenHBMConfig:
    """Retrieve a hidden HBM configuration by name."""
    if name not in ALL_HIDDEN_HBM_CONFIGS:
        raise ValueError(f"Unknown hidden HBM config name: {name}. Available: {list(ALL_HIDDEN_HBM_CONFIGS.keys())}")
    return ALL_HIDDEN_HBM_CONFIGS[name]


def list_hidden_hbm_configs():
    """List all available hidden HBM configuration names."""
    return list(ALL_HIDDEN_HBM_CONFIGS.keys())


def create_theta_params_from_config(config: HiddenHBMConfig, spices: list[str]) -> Dict[str, float]:
    """
    Create theta_params dictionary from config, handling dynamic patterns.
    
    Some configs have patterns that need to be filled based on the actual spices.
    """
    theta_params = {}
    
    if config.name == "AlternatingHuman":
        # Alternating pattern: 1.5, -1.5, 1.5, -1.5...
        for i, spice in enumerate(spices):
            theta_params[spice] = 1.5 if i % 2 == 0 else -1.5
    
    elif config.name == "HumanPrefersFirstHalf":
        # First half: 2.0, second half: -2.0
        for i, spice in enumerate(spices):
            theta_params[spice] = 2.0 if i < len(spices) // 2 else -2.0
    
    elif config.name == "HumanPrefersSecondHalf":
        # First half: -2.0, second half: 2.0
        for i, spice in enumerate(spices):
            theta_params[spice] = -2.0 if i < len(spices) // 2 else 2.0
    
    elif config.name in ["ConsistentHuman", "VariableHuman"]:
        # Use SpiceSpecificHuman pattern as base, but with different variance
        base_config = SPICE_SPECIFIC_HUMAN
        for spice in spices:
            theta_params[spice] = base_config.get_theta_for_spice(spice)
    
    else:
        # For configs with explicit spice mappings, use them directly
        # Fill in defaults for spices not in config
        for spice in spices:
            theta_params[spice] = config.get_theta_for_spice(spice)
    
    return theta_params


if __name__ == "__main__":
    print("Available hidden HBM configurations:")
    for name in list_hidden_hbm_configs():
        config = get_hidden_hbm_config(name)
        print(f"  - {name}")
        print(f"    sigma_r={config.sigma_r}, sigma_h={config.sigma_h}")
        if config.theta_mean:
            print(f"    Sample spices: {list(config.theta_mean.keys())[:5]}...")

