"""
Centralized configuration for Spices Environment HBM and Mood Inference

This file contains all hyperparameters and configuration values used throughout
the spices environment, HBM, and mood inference systems. Modify values here to
tune the learning behavior.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HBMConfig:
    """Configuration for Hierarchical Bayesian Model (HBM)."""
    
    # Prior means and variances
    mu0: float = 0.0  # Global prior mean
    sigma0: float = 1.0  # Global variance (σ₀²)
    sigma_h: float = 1.0  # Human-level variance (σₕ²)
    sigma_r: float = 1.5  # Recipe-level variance (σᵣ²) — moderately high to weaken wrong-theta prior without slowing convergence too much
    sigma_obs: float = 1.0  # Observation variance (σ_obs²)
    
    # Exponential Moving Average (EMA) for phi updates
    ema_alpha: float = 0.5  # EMA smoothing factor (50% new info per update - faster convergence)
    ema_alpha_converged: float = 0.3  # EMA smoothing when converged (30% new info - more stable)
    
    # Adaptive EMA thresholds (for early learning)
    ema_early_threshold: int = 20  # Use less smoothing for first N observations
    ema_early_alpha: float = 0.7  # EMA alpha for early learning (70% new info)
    ema_medium_threshold: int = 50  # Medium smoothing threshold
    ema_medium_alpha: float = 0.5  # EMA alpha for medium learning (50% new info)
    
    # Variance-adaptive updates
    use_variance_adaptive_lr: bool = True  # Scale learning rate by variance (low var = smaller updates)
    variance_adaptive_min_scale: float = 0.3  # Minimum LR scale when variance is very low
    variance_adaptive_max_scale: float = 1.0  # Maximum LR scale when variance is high
    variance_converged_threshold: float = 0.3  # Variance below this means "converged" (use more smoothing)

    # Learning rates for phi updates
    base_learning_rate: float = 0.3  # Base learning rate (increased for faster convergence)
    
    # Learning rate annealing (reduce over time for stability)
    use_lr_annealing: bool = True  # Enable learning rate annealing
    lr_annealing_start: int = 100  # Start annealing after N observations
    lr_annealing_end: int = 500  # Finish annealing at N observations
    lr_annealing_min: float = 0.1  # Minimum learning rate after annealing
    
    # Preference mismatch detection — penalty removed (set to 1.0) so that
    # contradicting observations have full weight when correcting a wrong-prior phi.
    preference_mismatch_threshold: float = 1.0  # Only check mismatch if |phi| > threshold
    preference_mismatch_penalty: float = 1.0  # No penalty: full signal even on contradicting obs

    # Batch θ/μ updates: call update_theta_and_mu every N episodes instead of every episode.
    # Reduces cost when update_theta_and_mu is the bottleneck (e.g. multi-human training).
    # 1 = current behavior (every episode); 10 = 10x fewer updates.
    update_theta_mu_every_n_episodes: int = 25


@dataclass(frozen=True)
class MoodConfig:
    """Configuration for Mood Inference."""
    
    # Mood prior probabilities [all_self, neutral, none_self]
    # Matches the generation distribution in MoodSpec.priors: 80% neutral, 10% each non-neutral
    mood_prior: Tuple[float, float, float] = (0.1, 0.8, 0.1)
    
    # Mood inference smoothing
    mood_smoothing_alpha: float = 0.3  # EMA smoothing (30% new info, 70% old)
    mood_prior_weight: float = 2.0  # Weight prior more heavily in log-likelihood
    
    # Mood bias strength (relative to base_satisfaction_bias)
    mood_bias_multiplier: float = 2.0  # mood_bias = base_satisfaction_bias * multiplier
    
    # Satisfaction likelihood (Gaussian)
    satisfaction_sigma: float = 0.3  # Variance for satisfaction Gaussian likelihood
    satisfaction_loglik_min: float = -10.0  # Minimum log-likelihood (clip extreme values)
    satisfaction_loglik_max: float = 0.0  # Maximum log-likelihood
    
    # De-mooding parameters
    demood_confidence_threshold: float = 0.3  # Only de-mood if non-neutral conf > threshold
    demood_scale_factor: float = 0.3  # Scale factor for expected mood contribution
    
    # Non-neutral mood preference weights (for mood inference)
    non_neutral_pref_weight_match: float = 0.2  # Preference weight when satisfaction matches expectation
    non_neutral_pref_weight_mismatch: float = 0.1  # Preference weight when satisfaction doesn't match


@dataclass(frozen=True)
class UpdateConfig:
    """Configuration for Update Thresholds and Filtering."""

    # Confidence weighting for phi updates when mood is neutral-confident
    confidence_weight_min: float = 0.2  # Minimum confidence weight


@dataclass(frozen=True)
class SatisfactionConfig:
    """Configuration for Satisfaction Computation."""
    
    # Base satisfaction bias (strength of preference signal)
    base_satisfaction_bias: float = 3.0
    
    # Continuous satisfaction: Beta distribution parameters
    satisfaction_beta_kappa: float = 10.0  # Concentration parameter for Beta distribution


@dataclass(frozen=True)
class SpicesConfig:
    """Main configuration class combining all sub-configs."""
    
    hbm: HBMConfig = HBMConfig()
    mood: MoodConfig = MoodConfig()
    update: UpdateConfig = UpdateConfig()
    satisfaction: SatisfactionConfig = SatisfactionConfig()
    
    @property
    def mood_prior_array(self) -> Tuple[float, float, float]:
        """Return mood prior as tuple (will be converted to numpy array where needed)."""
        return self.mood.mood_prior
    
    @property
    def mood_bias_strength(self) -> float:
        """Compute mood bias strength from satisfaction bias."""
        return self.satisfaction.base_satisfaction_bias * self.mood.mood_bias_multiplier
    
    def get_mood_bias(self) -> dict[str, dict[str, float]]:
        """Get mood bias dictionary."""
        bias_strength = self.mood_bias_strength
        return {
            "all_self": {"human": +bias_strength, "robot": -bias_strength},
            "neutral": {"human": 0.0, "robot": 0.0},
            "none_self": {"human": -bias_strength, "robot": +bias_strength},
        }


# Default configuration instance
DEFAULT_CONFIG = SpicesConfig()

