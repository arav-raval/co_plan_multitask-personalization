# Mathematical Explanation: Mood-Aware Hierarchical Preference Learning

## Table of Contents
1. [System Overview](#system-overview)
2. [Environment Model: Satisfaction Generation](#environment-model-satisfaction-generation)
3. [Mood Inference: Bayesian Filtering](#mood-inference-bayesian-filtering)
4. [Hierarchical Bayesian Model (HBM)](#hierarchical-bayesian-model-hbm)
5. [Online Replanning: CSP-Based Action Selection](#online-replanning-csp-based-action-selection)
6. [Complete Mathematical Formulation](#complete-mathematical-formulation)

---

## System Overview

The system learns human preferences for spice-actor assignments across multiple recipes while simultaneously inferring and adapting to episodic mood states. It combines:

- **Mood Inference**: Bayesian inference over latent mood states (all_self, neutral, none_self)
- **Hierarchical Preference Learning**: 3-level Bayesian model (recipe-specific → human-level → global)
- **Online Replanning**: CSP-based action selection that adapts to inferred mood in real-time

---

## Environment Model: Satisfaction Generation

### 2.1 Hidden Preferences

Each spice \(s\) has a hidden preference \(\phi_s^* \in \{\text{human}, \text{robot}\}\), unknown to the agent. The agent must learn this preference from satisfaction feedback.

### 2.2 Mood Model

Each episode samples a mood \(m_t \in \{\text{all\_self}, \text{neutral}, \text{none\_self}\}\) from a categorical distribution:

\[
m_t \sim \text{Categorical}(\mathbf{p}_{\text{mood}}), \quad \mathbf{p}_{\text{mood}} = (0.2, 0.6, 0.2)
\]

The distribution is skewed toward neutral (60%) to increase opportunities for preference learning.

### 2.3 Satisfaction Generation

After assigning actor \(a \in \{\text{human}, \text{robot}\}\) to spice \(s\), satisfaction \(y \in \{+1, -1\}\) is generated probabilistically via a logistic model:

\[
\text{logit} = \phi + \beta_m(a)
\]

where:
- **Preference component**: 
  \[
  \phi = \begin{cases}
    +\alpha & \text{if } a = \phi_s^* \\
    -\alpha & \text{otherwise}
  \end{cases}
  \]
  with \(\alpha = 3.0\) (base satisfaction bias)

- **Mood bias**: 
  \[
  \beta_m(a) = \begin{cases}
    +2\alpha & \text{if } m = \text{all\_self} \text{ and } a = \text{human} \\
    -2\alpha & \text{if } m = \text{all\_self} \text{ and } a = \text{robot} \\
    0 & \text{if } m = \text{neutral} \\
    -2\alpha & \text{if } m = \text{none\_self} \text{ and } a = \text{human} \\
    +2\alpha & \text{if } m = \text{none\_self} \text{ and } a = \text{robot}
  \end{cases}
  \]

Satisfaction is then sampled:
\[
P(y = +1) = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}, \quad y \sim \text{Bernoulli}(\sigma(\text{logit}))
\]

**Key Property**: With \(\alpha = 3.0\) and mood bias \(\pm 6.0\), mood can override preferences. For example, in "all_self" mood, satisfaction is positive only when human acts, regardless of base preferences.

---

## Mood Inference: Bayesian Filtering

### 3.1 Prior Distribution

The agent maintains a prior over mood with strong bias toward neutral:

\[
P(m) = \begin{cases}
  0.05 & m = \text{all\_self} \\
  0.90 & m = \text{neutral} \\
  0.05 & m = \text{none\_self}
\end{cases}
\]

This conservative prior requires strong evidence to shift away from neutral.

### 3.2 Likelihood Model

Given observations \((a_i, s_i, y_i)\) in episode \(t\), the likelihood under mood hypothesis \(m\) is:

\[
P(y_i | a_i, s_i, m, \phi_{\text{pref}}) = \sigma(\text{logit}_m)
\]

where \(\text{logit}_m\) depends on the mood hypothesis:

**For neutral mood** (\(m = \text{neutral}\)):
\[
\text{logit}_{\text{neutral}} = \text{sign}(a_i) \cdot \phi_{\text{pref}}(s_i, a_i) + \beta_{\text{neutral}}(a_i)
\]

where:
- \(\phi_{\text{pref}}(s_i, a_i)\) is the learned preference estimate from HBM
- \(\text{sign}(a_i) = +1\) if \(a_i = \text{human}\), else \(-1\)
- \(\beta_{\text{neutral}}(a_i) = 0\)

**For non-neutral moods** (\(m \in \{\text{all\_self}, \text{none\_self}\}\)):
\[
\text{logit}_m = \begin{cases}
  \beta_m(a_i) + 0.2 \cdot \text{sign}(a_i) \cdot \phi_{\text{pref}} & \text{if } y_i \text{ matches preference expectation} \\
  \beta_m(a_i) + 0.1 \cdot \text{sign}(a_i) \cdot \phi_{\text{pref}} & \text{if } y_i \text{ contradicts preference}
\end{cases}
\]

**Key Insight**: When satisfaction contradicts learned preferences, it provides stronger evidence for mood effects. This helps distinguish mood-driven satisfaction from preference-driven satisfaction.

### 3.3 Posterior Update

After each step, the mood posterior is updated via Bayes' rule:

\[
P(m | \mathcal{D}_t) \propto P(m) \prod_{i=1}^{n_t} P(y_i | a_i, s_i, m, \phi_{\text{pref}})
\]

where \(\mathcal{D}_t = \{(a_i, s_i, y_i)\}_{i=1}^{n_t}\) are observations in episode \(t\).

In log-space (for numerical stability):
\[
\log P(m | \mathcal{D}_t) = \log P(m) + \sum_{i=1}^{n_t} \log P(y_i | a_i, s_i, m, \phi_{\text{pref}}) - \text{const}
\]

The posterior is normalized:
\[
P(m | \mathcal{D}_t) = \frac{\exp(\log P(m | \mathcal{D}_t))}{\sum_{m'} \exp(\log P(m' | \mathcal{D}_t))}
\]

### 3.4 Expected Mood

The agent computes expected mood as a continuous value:
\[
\mathbb{E}[m] = \sum_{m} P(m | \mathcal{D}_t) \cdot v(m)
\]
where \(v(\text{all\_self}) = +1\), \(v(\text{neutral}) = 0\), \(v(\text{none\_self}) = -1\).

---

## Hierarchical Bayesian Model (HBM)

### 4.1 Three-Level Hierarchy

The HBM models preferences at three levels:

1. **Global level (μ)**: Base preference for each spice across all humans
2. **Human level (θ)**: Human-specific preference for each spice
3. **Recipe level (φ)**: Recipe-specific preference for each spice

The hierarchical structure:
\[
\begin{align}
\mu_s &\sim \mathcal{N}(\mu_0, \sigma_0^2) \\
\theta_s &\sim \mathcal{N}(\mu_s, \sigma_h^2) \\
\phi_{r,s} &\sim \mathcal{N}(\theta_s, \sigma_r^2)
\end{align}
\]

where:
- \(\mu_0 = 0.0\) (neutral prior)
- \(\sigma_0 = 1.0\) (global variance)
- \(\sigma_h = 1.0\) (human-level variance)
- \(\sigma_r = 1.0\) (recipe-level variance)

### 4.2 Preference Signal Extraction

For each observation \((a, s, y)\), we extract a preference signal \(g\):

\[
g = \begin{cases}
  +1 & \text{if } (a = \text{human} \text{ and } y = +1) \text{ or } (a = \text{robot} \text{ and } y = -1) \\
  -1 & \text{if } (a = \text{human} \text{ and } y = -1) \text{ or } (a = \text{robot} \text{ and } y = +1)
\end{cases}
\]

In compact form:
\[
g = \text{sign}(a) \cdot \text{sign}(y)
\]
where \(\text{sign}(a) = +1\) for human, \(-1\) for robot; \(\text{sign}(y) = +1\) for positive satisfaction, \(-1\) for negative.

**Important**: This signal is only valid when mood is neutral. In non-neutral moods, satisfaction reflects mood, not preferences.

### 4.3 Mood-Weighted Signal

The HBM weights the preference signal by mood posterior:

\[
g_{\text{weighted}} = P(\text{neutral} | \mathcal{D}_t) \cdot g
\]

However, we only update preferences when confident in neutral mood:
\[
\text{update if } P(\text{neutral} | \mathcal{D}_t) \geq \theta_{\text{threshold}}
\]

where \(\theta_{\text{threshold}} = 0.5\) (configurable).

### 4.4 Recipe-Level Update (φ)

For recipe \(r\) and spice \(s\), we update \(\phi_{r,s}\) using a Normal-Normal Bayesian update:

**Observation model**:
\[
g \sim \mathcal{N}(\phi_{r,s}, \sigma_{\text{obs}}^2)
\]

**Prior**:
\[
\phi_{r,s} \sim \mathcal{N}(\theta_s, \sigma_r^2)
\]

**Posterior**:
\[
\begin{align}
\text{post\_var} &= \frac{1}{\frac{1}{\sigma_r^2} + \frac{1}{\sigma_{\text{obs}}^2}} \\
\text{post\_mean} &= \text{post\_var} \cdot \left(\frac{\theta_s}{\sigma_r^2} + \frac{g}{\sigma_{\text{obs}}^2}\right)
\end{align}
\]

**With Learning Rate**:
To prevent oscillations, we scale the observation variance by a learning rate \(\lambda \in [0.05, 0.1]\):

\[
\sigma_{\text{obs,eff}}^2 = \frac{\sigma_{\text{obs}}^2}{\lambda}
\]

Lower learning rate → higher effective noise → smaller updates.

**Exponential Moving Average (EMA)**:
To further smooth updates, we apply EMA:
\[
\phi_{r,s}^{\text{new}} = \alpha \cdot \text{post\_mean} + (1 - \alpha) \cdot \phi_{r,s}^{\text{old}}
\]
where \(\alpha = 0.1\) (smoothing factor).

### 4.5 Human-Level Update (θ)

At episode end, we update \(\theta_s\) by pooling across all recipes:

\[
\theta_s = \text{pool}(\{\phi_{r,s} : r \in \text{recipes}\})
\]

Using Gaussian pooling:
\[
\begin{align}
\bar{\phi}_s &= \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \phi_{r,s} \\
\text{post\_var} &= \frac{1}{\frac{1}{\sigma_0^2 + \sigma_h^2} + \frac{1}{\sigma_r^2}} \\
\text{post\_mean} &= \text{post\_var} \cdot \left(\frac{\mu_s}{\sigma_0^2 + \sigma_h^2} + \frac{\bar{\phi}_s}{\sigma_r^2}\right)
\end{align}
\]

### 4.6 Global-Level Update (μ)

Similarly, we update \(\mu_s\) by pooling across all humans:

\[
\begin{align}
\text{post\_var} &= \frac{1}{\frac{1}{\sigma_0^2} + \frac{1}{\sigma_h^2}} \\
\text{post\_mean} &= \text{post\_var} \cdot \left(\frac{0}{\sigma_0^2} + \frac{\theta_s}{\sigma_h^2}\right)
\end{align}
\]

### 4.7 Preference Extraction

Given learned \(\phi_{r,s}\), we extract preference strength:

\[
\phi_{\text{pref}}(s, a) = \text{sign}(a) \cdot \phi_{r,s}
\]

This is clamped to \([-α, +α]\) where \(α = 3.0\).

---

## Online Replanning: CSP-Based Action Selection

### 5.1 CSP Generation

At each step, the agent generates a CSP to select actor \(a\) for current spice \(s\):

**Variables**: \(V = \{a\}\), where \(a \in \{\text{human}, \text{robot}\}\)

**Constraints**: Depend on inferred mood:

\[
\text{constraints} = \begin{cases}
  \{a = \text{human}\} & \text{if } m^* = \text{all\_self} \text{ and } P(m^*) \geq \theta_{\text{threshold}} \\
  \{a = \text{robot}\} & \text{if } m^* = \text{none\_self} \text{ and } P(m^*) \geq \theta_{\text{threshold}} \\
  \{\log P(y = +1 | s, a) \geq \tau\} & \text{otherwise (use learned preferences)}
\end{cases}
\]

where:
- \(m^* = \arg\max_m P(m | \mathcal{D}_t)\)
- \(\tau = \log(10^{-6})\) (minimum log-probability threshold)

**Cost Function** (for exploitation):
\[
\text{cost}(a) = -\log P(y = +1 | s, a)
\]

where \(P(y = +1 | s, a) = \sigma(\text{sign}(a) \cdot \phi_{r,s})\).

**Samplers**: If mood forces an actor, sampler always returns that actor. Otherwise, samples proportionally to \(P(y = +1 | s, a)\).

### 5.2 Online Adaptation

Since CSP is regenerated each step using the current mood posterior, the agent adapts immediately when mood becomes clear:
- **Early steps**: Uses learned preferences (or random if no HBM yet)
- **After mood inference**: Switches to mood-appropriate actor assignment

---

## Complete Mathematical Formulation

### 6.1 Satisfaction Model
\[
y \sim \text{Bernoulli}(\sigma(\phi + \beta_m(a)))
\]
where:
- \(\phi = \pm\alpha\) encodes preference match
- \(\beta_m(a)\) encodes mood effect
- \(\sigma(x) = 1/(1 + e^{-x})\) is the sigmoid function

### 6.2 Mood Inference
\[
P(m | \mathcal{D}_t) \propto P(m) \prod_{i=1}^{n_t} P(y_i | a_i, s_i, m, \phi_{\text{pref}})
\]

### 6.3 HBM Preference Learning

**Recipe-level**:
\[
\phi_{r,s} \sim \mathcal{N}(\theta_s, \sigma_r^2), \quad g \sim \mathcal{N}(\phi_{r,s}, \sigma_{\text{obs}}^2/\lambda)
\]

**Human-level** (pooling):
\[
\theta_s = \text{pool}(\{\phi_{r,s} : r \in \mathcal{R}\})
\]

**Global-level** (pooling):
\[
\mu_s = \text{pool}(\{\theta_s\})
\]

### 6.4 Action Selection
\[
a^* = \begin{cases}
  \text{human} & \text{if } P(\text{all\_self}) \geq \theta \\
  \text{robot} & \text{if } P(\text{none\_self}) \geq \theta \\
  \arg\max_a P(y = +1 | s, a) & \text{otherwise}
\end{cases}
\]

where \(P(y = +1 | s, a) = \sigma(\text{sign}(a) \cdot \phi_{r,s})\).

---

## Key Design Decisions

1. **Probabilistic satisfaction**: Models realistic human feedback noise
2. **Mood as latent variable**: Inferred from observations, not directly observed
3. **Threshold-based learning**: Only neutral-confident episodes contribute to preferences
4. **Hierarchical structure**: Enables information sharing across recipes
5. **Online replanning**: CSP regenerated each step for immediate mood adaptation
6. **Conservative learning**: Low learning rate + EMA smoothing prevents oscillations
7. **Skewed mood prior**: 60% neutral to increase learning opportunities

---

## Parameter Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| \(\alpha\) | 3.0 | Base satisfaction bias |
| \(\beta_{\text{strength}}\) | 6.0 | Mood bias strength (2×α) |
| \(\sigma_0\) | 1.0 | Global preference variance |
| \(\sigma_h\) | 1.0 | Human-level variance |
| \(\sigma_r\) | 1.0 | Recipe-level variance |
| \(\sigma_{\text{obs}}\) | 0.3 | Observation noise |
| \(\lambda\) | 0.05-0.1 | Learning rate (scaled by confidence) |
| \(\alpha_{\text{EMA}}\) | 0.1 | EMA smoothing factor |
| \(\theta_{\text{threshold}}\) | 0.5 | Neutral confidence threshold |

---

## Expected Behavior

- **Early episodes**: Low satisfaction (random choices, no preferences learned)
- **Mid-training**: Improving satisfaction (HBM learning, mood inference improving)
- **Late training**: Higher satisfaction (better preferences, accurate mood inference)
- **Testing**: Transfers learned preferences while adapting to mood in real-time

This design enables learning long-term preferences while adapting to episodic mood variations through hierarchical Bayesian inference and online replanning.

