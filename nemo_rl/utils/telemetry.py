# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
# opentelemetry-semantic-conventions might need to be imported differently depending on version
# Using string literals for standard attributes to avoid version issues for now, or use standard import
try:
    from opentelemetry.semconv.resource import ResourceAttributes
except ImportError:
    # Fallback or different import path for newer versions
    class ResourceAttributes:
        SERVICE_NAME = "service.name"


# Semantic Conventions
RL_SYSTEM_NAME = "nemo-rl"

# Trace Spans
SPAN_EPISODE = "rl.episode"
SPAN_STEP = "rl.step"
SPAN_TRAINING_ITERATION = "rl.training.iteration"
SPAN_AGENT_ACT = "rl.agent.act"

# Attributes
ATTR_EPISODE_ID = "rl.episode.id"
ATTR_AGENT_ID = "rl.agent.id"
ATTR_ENV_NAME = "rl.environment.name"
ATTR_STEP_INDEX = "rl.step.index"
ATTR_REWARD = "rl.reward"
ATTR_DONE = "rl.done"
ATTR_ACTION = "rl.action"
ATTR_ITERATION_INDEX = "rl.iteration.index"
ATTR_BATCH_SIZE = "rl.batch.size"
ATTR_ALGORITHM = "rl.algorithm"
ATTR_OBSERVATION_SHAPE = "rl.observation.shape"

# Semantic conventions
RL_SYSTEM = "rl.system"
RL_SYSTEM_VERSION = "rl.system.version"
RL_RUN_ID = "rl.run.id"
RL_ALGORITHM = "rl.algorithm"
RL_ENVIRONMENT_NAME = "rl.environment.name"
RL_MODEL_NAME = "rl.model.name"


RL_LOOP = "rl.loop"
RL_LOOP_ITERATION = "rl.loop.iteration"

RL_SAMPLE = "rl.sample"
RL_SAMPLE_EPISODES = "rl.sample.episodes"
RL_SAMPLE_STEPS = "rl.sample.steps"
RL_SAMPLE_BATCH_SIZE = "rl.sample.batch_size"

RL_REWARD = "rl.reward"
RL_REWARD_BATCH_SIZE = "rl.reward.batch_size"
RL_REWARD_SANDBOX = "rl.reward.sandbox"

RL_TRAIN = "rl.train"
RL_TRAIN_STEPS = "rl.train.steps"
RL_TRAIN_BATCH_SIZE = "rl.train.batch_size"
RL_TRAIN_TOKENS = "rl.train.tokens"

RL_SYNC = "rl.sync"
RL_SYNC_BYTES = "rl.sync.bytes"
RL_SYNC_SOURCE = "rl.sync.source"
RL_SYNC_DESTINATION = "rl.sync.destination"

# Metric names
METRIC_STEPS_TOTAL = "rl.environment.steps.total"
METRIC_EPISODES_TOTAL = "rl.episodes.total"
METRIC_EPISODE_REWARD = "rl.episode.reward"
METRIC_EPISODE_LENGTH = "rl.episode.length"
METRIC_STEP_DURATION = "rl.step.duration"
METRIC_LOSS_POLICY = "rl.training.loss.policy"
METRIC_LOSS_VALUE = "rl.training.loss.value"
METRIC_LOSS_ENTROPY = "rl.training.loss.entropy"
METRIC_LOSS_KL = "rl.training.loss.kl_divergence"
METRIC_EXPLAINED_VARIANCE = "rl.training.explained_variance"
METRIC_LEARNING_RATE = "rl.training.learning_rate"
METRIC_STEPS_PER_SECOND = "rl.environment.steps_per_second"

RL_LOOP_DURATION = "rl.loop.duration"
RL_SAMPLE_DURATION = "rl.sample.duration"
RL_REWARD_DURATION = "rl.reward.duration"
RL_TRAIN_DURATION = "rl.train.duration"
RL_SYNC_DURATION = "rl.sync.duration"

RL_SAMPLE_SAMPLES_COUNT = "rl.sample.samples"
RL_SAMPLE_EPISODES_COUNT = "rl.sample.episodes"
RL_TRAIN_STEPS_COUNT = "rl.train.steps"
RL_TRAIN_TOKENS_COUNT = "rl.train.tokens"

RL_ENVIRONMENT_REWARD_MEAN = "rl.environment.reward.mean"
RL_ENVIRONMENT_EPISODE_LENGTH_MEAN = "rl.environment.episode.length.mean"
RL_TRAIN_LOSS = "rl.train.loss"

RL_STEP_TIME = "rl.step_time"
RL_TOKENS_PER_SEC = "rl.tokens_per_sec"
RL_TOKENS_PER_SEC_PER_GPU = "rl.tokens_per_sec_per_gpu"
RL_TRAINING_MFU = "rl.training_mfu"
RL_SAMPLES_PER_SEC = "rl.samples_per_sec"
RL_SAMPLES_PER_SEC_PER_GPU = "rl.samples_per_sec_per_gpu"


def configure_opentelemetry(service_name: str = "nemo-rl-service", endpoint: Optional[str] = None):
    """Configures OpenTelemetry with a basic setup.

    Args:
        service_name: Name of the service.
        endpoint: OTLP endpoint (not implemented in this basic version, defaults to Console).
    """
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        "rl.system.name": RL_SYSTEM_NAME,
    })

    # Trace Provider
    trace_provider = TracerProvider(resource=resource)
    # For demonstration/default, use ConsoleExporter. In production, this should be configurable.
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)

    # Meter Provider
    # Using a simple ConsoleExporter for metrics as well.
    reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

def get_tracer():
    return trace.get_tracer(RL_SYSTEM_NAME)

def get_meter():
    return metrics.get_meter(RL_SYSTEM_NAME)


import contextlib

@contextlib.contextmanager
def trace_step(tracer, env_name, batch_size):
    """Context manager for tracing an environment step."""
    with tracer.start_as_current_span(SPAN_STEP) as span:
        span.set_attribute(ATTR_ENV_NAME, env_name)
        span.set_attribute(ATTR_BATCH_SIZE, batch_size)
        yield span
