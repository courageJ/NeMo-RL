from typing import Dict, Any, Optional
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter

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


def setup_telemetry(service_name: str = "nemo_rl", service_version: str = "0.1.0"):
    """Sets up OpenTelemetry tracer and meter providers."""
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": service_version,
    })

    # Trace Provider
    trace_provider = TracerProvider(resource=resource)
    # For now, we don't add an exporter by default to avoid noise,
    # but one could be added here or configured via env vars.
    # trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(trace_provider)

    # Meter Provider
    # metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    # meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    meter_provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(meter_provider)

    return trace.get_tracer(service_name), metrics.get_meter(service_name)


class RLTelemetry:
    def __init__(self, service_name: str = "nemo_rl", version: str = "0.1.0"):
        self.tracer, self.meter = setup_telemetry(service_name, version)
        
        # Histograms
        self.loop_duration = self.meter.create_histogram(RL_LOOP_DURATION, unit="ms", description="End-to-end duration of one RL loop iteration")
        self.sample_duration = self.meter.create_histogram(RL_SAMPLE_DURATION, unit="ms", description="Duration of the sampling phase")
        self.reward_duration = self.meter.create_histogram(RL_REWARD_DURATION, unit="ms", description="Duration of the reward calculation phase")
        self.train_duration = self.meter.create_histogram(RL_TRAIN_DURATION, unit="ms", description="Duration of the training phase")
        self.sync_duration = self.meter.create_histogram(RL_SYNC_DURATION, unit="ms", description="Duration of the weight synchronization phase")

        # Counters
        self.sample_samples = self.meter.create_counter(RL_SAMPLE_SAMPLES_COUNT, description="Number of samples generated")
        self.sample_episodes = self.meter.create_counter(RL_SAMPLE_EPISODES_COUNT, description="Number of episodes completed")
        self.train_steps = self.meter.create_counter(RL_TRAIN_STEPS_COUNT, description="Number of training steps completed")
        self.train_tokens = self.meter.create_counter(RL_TRAIN_TOKENS_COUNT, description="Number of tokens processed in training")

        # Gauges (using UpDownCounter as Gauge interface in Python SDK is observable only usually, 
        # but for simple setting values, we might use ObservableGauge with callbacks or just track manually.
        # However, OTel Python Metrics API recommends using ObservableGauge for values that are read periodically.
        # Since we want to set values explicitly, we can use a Histogram (distribution) or UpDownCounter if we want to track 'current' values roughly,
        # but usually Gauges are asynchronous. 
        # For simplicity in this sync context, we'll use Histograms for distribution of these values over time, 
        # which is often what you want for 'mean reward' per step anyway.)
        self.reward_mean = self.meter.create_histogram(RL_ENVIRONMENT_REWARD_MEAN, description="Mean reward achieved")
        self.episode_length_mean = self.meter.create_histogram(RL_ENVIRONMENT_EPISODE_LENGTH_MEAN, description="Mean episode length")
        self.train_loss = self.meter.create_histogram(RL_TRAIN_LOSS, description="Training loss")

