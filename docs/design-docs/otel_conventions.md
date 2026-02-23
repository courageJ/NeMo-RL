# OpenTelemetry Conventions for NeMo-RL

## Objective
To standardize telemetry data collection (traces, metrics, logs) for Reinforcement Learning (RL) workloads in NeMo-RL using OpenTelemetry.

## Semantic Conventions

### Resource Attributes
*   `rl.system.name`: Name of the RL system (e.g., "nemo-rl").
*   `rl.system.version`: Version of the RL system.

### Spans (Traces)
*   **Episode**: `rl.episode`
    *   Attributes: `rl.episode.id`, `rl.agent.id`, `rl.environment.name`
*   **Step**: `rl.step`
    *   Attributes: `rl.step.index`, `rl.episode.id`, `rl.reward`
*   **Training Iteration**: `rl.training.iteration`
    *   Attributes: `rl.iteration.index`, `rl.batch.size`
*   **Agent Act**: `rl.agent.act`
    *   Attributes: `rl.agent.id`, `rl.observation.shape`

### Metrics
*   **Counters**:
    *   `rl.environment.steps.total`: Total number of environment steps.
    *   `rl.episodes.total`: Total number of episodes completed.
*   **Histograms**:
    *   `rl.episode.reward`: Distribution of episode rewards.
    *   `rl.episode.length`: Distribution of episode lengths.
    *   `rl.step.duration`: Time taken for a single step.
*   **Gauges**:
    *   `rl.training.loss.policy`: Policy loss.
    *   `rl.training.loss.value`: Value function loss.
    *   `rl.training.loss.entropy`: Entropy loss.
    *   `rl.training.loss.kl_divergence`: KL divergence between policies.
    *   `rl.training.explained_variance`: Explained variance of the value function.
    *   `rl.training.learning_rate`: Current learning rate.
    *   `rl.environment.steps_per_second`: Throughput.

## Golden Signals
1.  **Reward**: The primary measure of agent performance.
    *   Metric: `rl.episode.reward` (Histogram/Gauge)
2.  **Length**: The duration of interaction.
    *   Metric: `rl.episode.length` (Histogram)
3.  **Loss**: The training stability and convergence.
    *   Metric: `rl.training.loss.*` (Gauge)
4.  **Throughput**: The efficiency of the training loop.
    *   Metric: `rl.environment.steps_per_second` (Gauge)

## Implementation Plan
1.  Add `opentelemetry-api` and `opentelemetry-sdk` to `pyproject.toml`.
2.  Create `docs/design-docs/otel_conventions.md` with the detailed spec.
3.  Create `nemo_rl/utils/telemetry.py` to handle OTel setup.
4.  Instrument `EnvironmentInterface` and algorithms to use these conventions.
