"""
Centralized configuration for Spices Environment HBM and Mood Inference

This file contains all hyperparameters and configuration values used throughout
the spices environment, HBM, and mood inference systems. Modify values here to
tune the learning behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math


@dataclass(frozen=True)
class HBMConfig:
    """Configuration for Hierarchical Bayesian Model (HBM)."""
    
    # Prior means and variances
    mu0: float = 0.0  # Global prior mean
    sigma0: float = 3.0  # Global variance (σ₀²) — wide so mu prior is nearly uninformative; allows theta to escape the mu=0 shrinkage that attenuates phi toward 0
    sigma_h: float = 1.0  # Human-level variance (σₕ²)
    sigma_r: float = 0.5  # Recipe-level variance (σᵣ²) — initialized at the hidden HBM's true sigma_r so phi posteriors narrow faster for borderline-magnitude spices
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
    # 1 = every episode (correct Stage 1 behavior); higher values trade correctness for speed.
    update_theta_mu_every_n_episodes: int = 2

    # Stage 2: session-level psi latent variable (transient offset).
    # psi ~ N(0, sigma_mood²) — prior std for the per-episode session offset.
    # sigma_mood=4.0 keeps psi_true=±5.0 within 1.25 std of the prior, so the
    # KL penalty doesn't resist large mood offsets too aggressively.
    # Note: q(psi) variance is fixed at sigma_mood² (not learned) to prevent
    # psi's uncertainty from inflating phi's variance through the theta optimizer.
    # sigma_mood is the prior std for psi. It also sets the posterior std for psi
    # (q_var == p_var == sigma_mood² — fixed, not learned). This means:
    # - Larger sigma_mood → psi can roam further from 0, less KL resistance to
    #   large mood values, but also more gradient noise into phi.
    # - sigma_mood should be ~= psi_true_mood_mean_abs so psi_true sits within
    #   ~1 std of the prior, keeping KL resistance manageable.
    # Lowered from 2.0→1.2 to match reduced psi_true_mood_mean_abs and reduce
    # mood's interference with nuanced phi learning.
    sigma_mood: float = 2.0

    # psi_decay: fraction of psi mean retained between episodes (0.05 = 95% reset).
    # After end_episode, psi_m *= psi_decay. This ensures psi stays transient:
    # persistent signals accumulate in phi, not psi. Too high → psi bleeds across
    # episodes. Too low → same as resetting to 0 each time (fine for identifiability).
    psi_decay: float = 0.05

    # ELBO / optimizer runtime controls.
    n_mc_samples: int = 8
    # n_phi_steps: Adam steps on phi ELBO per episode. More steps = faster convergence
    # but also more overfitting risk to recent episode's noise. 16 is a good balance.
    n_phi_steps: int = 10
    # n_psi_steps: Adam steps on psi ELBO per episode. Psi uses the same lr as phi
    # (lr_phi), so it also converges slowly — this is intentional. If psi converges
    # too fast (over-shoots to psi_true), phi gets zero residual and learns nothing.
    # The partial-convergence of psi is what preserves phi's identifiability.
    # Reduced from 16→8: the psi ELBO is a scalar KL bowl and converges quickly;
    # fewer steps halve the MC gradient noise injected into phi on neutral episodes.
    n_psi_steps: int = 3

    # Mood confidence threshold for skipping the psi ELBO entirely.
    # If the neutral mood posterior exceeds this value at episode end, the episode
    # is confidently neutral (psi_true ≈ 0) and running the ELBO would only inject
    # MC gradient noise into the psi mean — which then biases phi's update.
    # At 0.85 the gate fires on ~80% of neutral episodes (posterior concentrates
    # quickly) while always staying open for genuinely moody episodes where
    # the neutral posterior stays low.  Set to 1.0 to disable.
    psi_skip_neutral_threshold: float = 0.85
    n_theta_steps: int = 12
    # lr_phi: Adam lr for phi and psi updates. Same rate for both keeps psi from
    # racing ahead of phi and stealing the learning signal.
    lr_phi: float = 3e-2
    lr_psi: float = 1e-2  # lower than lr_phi so psi doesn't outpace phi during training
    lr_theta: float = 1e-2
    lr_hyper: float = 5e-3
    log_var_min: float = math.log(1e-6)
    log_var_max: float = math.log(10.0)


@dataclass(frozen=True)
class MoodConfig:
    """Configuration for Mood Inference."""
    
    # Mood prior probabilities [all_self, neutral, none_self].
    # Controls the fraction of training and eval episodes that are moody.
    # 30% moody (15%+15%) gives enough mood signal to reward psi inference,
    # while 70% neutral provides the clean phi signal needed for phi to
    # outpace the psi-less ablation.
    mood_prior: Tuple[float, float, float] = (0.15, 0.70, 0.15)
    
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

    # Env-only Stage 2 generative parameters for psi_true (sampled once/episode).
    # Keep these independent from base_satisfaction_bias so mood strength can be tuned
    # without changing the stable per-spice preference margin.
    # Mood signal: psi_true ~ N(±5.0, 0.5) for non-neutral episodes.
    # Strong enough that without_mood_learning is severely degraded on moody episodes,
    # while with_mood_learning infers psi and compensates.
    # psi_true_mood_mean_abs: the ground-truth mood offset magnitude for non-neutral
    # episodes. psi_true ~ N(±psi_true_mood_mean_abs, psi_true_mood_std²).
    # This shifts satisfaction by tanh(phi + psi_true) — tanh(phi).
    # At phi≈0 and psi_true=2.0: satisfaction shifts from 0.0 to tanh(2.0)=0.96.
    # Too large → psi absorbs everything and phi gets zero gradient.
    # Too small → without_mood_learning barely degrades, gap is not meaningful.
    # 2.0 was calibrated so that: (a) with_mood_learning can infer psi and still
    # learn phi correctly from the residual, and (b) without_mood_learning degrades
    # noticeably because psi_true=2.0 shifts ~40% of spice assignments.
    # Reduced to 1.2: mood still contaminates enough to reward learning it, but
    # nuanced spices (|phi|≈0.5) are no longer drowned out by mood contamination.
    # At psi_true=1.2 and phi=0.5: tanh(1.7)=0.94 (moody) vs tanh(0.5)=0.46 (neutral)
    # — mood still clearly shifts behavior, but phi signal remains recoverable.
    psi_true_mood_mean_abs: float = 2.0
    psi_true_mood_std: float = 0.5
    # Neutral episode psi_true std: small so neutral episodes give clean phi signal.
    psi_true_neutral_std: float = 0.05


@dataclass(frozen=True)
class UpdateConfig:
    """Configuration for Update Thresholds and Filtering."""

    # Confidence weighting for phi updates when mood is neutral-confident
    confidence_weight_min: float = 0.2  # Minimum confidence weight


@dataclass(frozen=True)
class SatisfactionConfig:
    """Configuration for Satisfaction Computation."""
    
    # Base satisfaction bias (strength of preference signal)
    # Increased from 1.5 → 2.0 to raise the tanh ceiling for weak-magnitude spices
    # (|theta|=0.5 now gives tanh(2.0*0.5)=0.76 vs tanh(1.5*0.5)=0.64).
    base_satisfaction_bias: float = 2.0
    # Temperature for tanh(logit / T) in env generation and HBM sat likelihood.
    # T>1 reduces saturation so large |phi| values remain distinguishable.
    satisfaction_logit_temperature: float = 1.0
    
    # Continuous satisfaction: Beta distribution parameters
    satisfaction_beta_kappa: float = 10.0  # Concentration parameter for Beta distribution

    # Coordination cost: additive penalty subtracted from satisfaction when the
    # robot mispredicts (conflict or missed pass).  Models frustration from
    # poor coordination independently of the underlying preference strength.
    coordination_cost: float = 0.5


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
    
    def get_mood_bias(self) -> Dict[str, Dict[str, float]]:
        """Get mood bias dictionary."""
        bias_strength = self.mood_bias_strength
        return {
            "all_self": {"human": +bias_strength, "robot": -bias_strength},
            "neutral": {"human": 0.0, "robot": 0.0},
            "none_self": {"human": -bias_strength, "robot": +bias_strength},
        }


# Default configuration instance
DEFAULT_CONFIG = SpicesConfig()

