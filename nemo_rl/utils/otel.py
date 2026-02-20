from typing import Dict, Any, Optional, TypedDict, Literal
import os
import sys

# Essential workaround for OpenTelemetry OTLP protobuf incompatibility
# with newer protobuf versions (causes TypeError: Descriptors cannot be created directly)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
import contextlib

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

RL_STEP_TIME = "rl.step_time"
RL_TOKENS_PER_SEC_PER_GPU = "rl.tokens_per_sec_per_gpu"
RL_TRAINING_MFU = "rl.training_mfu"
RL_SAMPLES_PER_SEC = "rl.samples_per_sec"
RL_SAMPLES_PER_SEC_PER_GPU = "rl.samples_per_sec_per_gpu"


class TelemetryConfig(TypedDict, total=False):
    enabled: bool
    service_name: str
    service_version: Optional[str]
    exporter_type: Literal["console", "otlp_http", "otlp_grpc", "none"]
    endpoint: Optional[str]
    log_metrics: bool



def setup_telemetry(config: TelemetryConfig):
    """Sets up OpenTelemetry tracer and meter providers."""
    service_name = config.get("service_name", "nemo_rl")
    service_version = config.get("service_version", "0.1.0")
    print(f"Init RL Telemetry with config: {config}", flush=True)
    
    if not config.get("enabled", False):
         # If disabled, we return the no-op global tracer/meter provided by the API by default
         return trace.get_tracer(service_name), metrics.get_meter(service_name)

    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": service_version,
    })

    # Trace Provider
    trace_provider = TracerProvider(resource=resource)
    exporter_type = config.get("exporter_type", "none")
    endpoint = config.get("endpoint")
    
    if exporter_type == "console":
        trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter_type == "otlp_http":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()))
        except ImportError:
            print(f"Error: opentelemetry-exporter-otlp not installed. Skipping OTLP HTTP exporter.", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing OTLP HTTP trace exporter: {e}", file=sys.stderr)
    elif exporter_type == "otlp_grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()))
        except ImportError:
            print(f"Error: opentelemetry-exporter-otlp not installed. Skipping OTLP GRPC exporter.", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing OTLP GRPC trace exporter: {e}", file=sys.stderr)
            
    trace.set_tracer_provider(trace_provider)

    # Meter Provider
    metric_readers = []
    if exporter_type == "console":
        metric_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    elif exporter_type == "otlp_http":
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint) if endpoint else OTLPMetricExporter()))
        except ImportError:
            print(f"Error: opentelemetry-exporter-otlp not installed. Skipping OTLP HTTP metric exporter.", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing OTLP HTTP metric exporter: {e}", file=sys.stderr)
    elif exporter_type == "otlp_grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint) if endpoint else OTLPMetricExporter()))
        except ImportError:
            print(f"Error: opentelemetry-exporter-otlp not installed. Skipping OTLP GRPC metric exporter.", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing OTLP GRPC metric exporter: {e}", file=sys.stderr)

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    metrics.set_meter_provider(meter_provider)

    return trace.get_tracer(service_name), metrics.get_meter(service_name)


import contextlib

class LoggingTracer:
    def __init__(self, tracer, log_metrics=False):
        self.tracer = tracer
        self.log_metrics = log_metrics
        
    @contextlib.contextmanager
    def start_as_current_span(self, name, *args, **kwargs):
        if self.log_metrics:
            print(f"Telemetry span '{name}' started", flush=True)
        with self.tracer.start_as_current_span(name, *args, **kwargs) as span:
            yield span
        if self.log_metrics:
            print(f"Telemetry span '{name}' ended", flush=True)


class LoggingInstrument:
    def __init__(self, name, instrument, log_metrics=False):
        self.name = name
        self.instrument = instrument
        self.log_metrics = log_metrics
        
    def record(self, amount, attributes=None):
        if self.log_metrics:
            print(f"Telemetry metric '{self.name}' recorded: {amount}", flush=True)
        self.instrument.record(amount, attributes)
        
    def add(self, amount, attributes=None):
        if self.log_metrics:
            print(f"Telemetry metric '{self.name}' added: {amount}", flush=True)
        self.instrument.add(amount, attributes)

class RLTelemetry:
    def __init__(self, config: TelemetryConfig):
        tracer, self.meter = setup_telemetry(config)
        self.log_metrics = config.get("log_metrics", True)
        self.tracer = LoggingTracer(tracer, self.log_metrics)
        print(f"Init RL Telemetry", flush=True)
        
        if self.log_metrics:
            print(f"OpenTelemetry initialized for service '{config.get('service_name', 'nemo_rl')}' with exporter: {config.get('exporter_type', 'none')}", flush=True)
        def create_histogram(name, *args, **kwargs):
            hist = self.meter.create_histogram(name, *args, **kwargs)
            return LoggingInstrument(name, hist, self.log_metrics)
            
        def create_counter(name, *args, **kwargs):
            counter = self.meter.create_counter(name, *args, **kwargs)
            return LoggingInstrument(name, counter, self.log_metrics)
            
        # Histograms
        self.loop_duration = create_histogram(RL_LOOP_DURATION, unit="ms", description="End-to-end duration of one RL loop iteration")
        self.sample_duration = create_histogram(RL_SAMPLE_DURATION, unit="ms", description="Duration of the sampling phase")
        self.reward_duration = create_histogram(RL_REWARD_DURATION, unit="ms", description="Duration of the reward calculation phase")
        self.train_duration = create_histogram(RL_TRAIN_DURATION, unit="ms", description="Duration of the training phase")
        self.sync_duration = create_histogram(RL_SYNC_DURATION, unit="ms", description="Duration of the weight synchronization phase")

        # Counters
        self.sample_samples = create_counter(RL_SAMPLE_SAMPLES_COUNT, description="Number of samples generated")
        self.sample_episodes = create_counter(RL_SAMPLE_EPISODES_COUNT, description="Number of episodes completed")
        self.train_steps = create_counter(RL_TRAIN_STEPS_COUNT, description="Number of training steps completed")
        self.train_tokens = create_counter(RL_TRAIN_TOKENS_COUNT, description="Number of tokens processed in training")

        # Gauges/Histograms for values
        self.reward_mean = create_histogram(RL_ENVIRONMENT_REWARD_MEAN, description="Mean reward achieved")
        self.episode_length_mean = create_histogram(RL_ENVIRONMENT_EPISODE_LENGTH_MEAN, description="Mean episode length")
        self.train_loss = create_histogram(RL_TRAIN_LOSS, description="Training loss")

        self.step_time = create_histogram(RL_STEP_TIME, unit="ms", description="End-to-end duration of one RL step")
        self.tokens_per_sec_per_gpu = create_histogram(RL_TOKENS_PER_SEC_PER_GPU, description="End-to-end tokens per second per GPU")
        self.training_mfu = create_histogram(RL_TRAINING_MFU, description="Training Model Floating Point Utilization (MFU)")
        self.samples_per_sec = create_histogram(RL_SAMPLES_PER_SEC, description="End-to-end samples per second")
        self.samples_per_sec_per_gpu = create_histogram(RL_SAMPLES_PER_SEC_PER_GPU, description="End-to-end samples per second per GPU")

    def flush(self):
        """Force flush all telemetry telemetry data safely."""
        try:
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "force_flush"):
                tracer_provider.force_flush()
                
            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, "force_flush"):
                meter_provider.force_flush()
        except Exception as e:
            if self.log_metrics:
                print(f"Warning: Failed to flush telemetry data: {e}", flush=True)

