from typing import Dict, Any, Optional, TypedDict, Literal
import os
import sys
import logging

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.metrics.view import View, ExponentialBucketHistogramAggregation
import contextlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
try:
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    HAS_PROTO = True
except ImportError:
    HAS_PROTO = False

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
# Latency
RL_LOOP_DURATION = "rl.loop.duration"
RL_SAMPLE_DURATION = "rl.sample.duration"
RL_REWARD_DURATION = "rl.reward.duration"
RL_TRAIN_DURATION = "rl.train.duration"
RL_SYNC_DURATION = "rl.sync.duration"
RL_STEP_DURATION = "rl.step.duration"

# Throughput
RL_SAMPLE_SAMPLES_COUNT = "rl.sample.samples"
RL_SAMPLE_EPISODES_COUNT = "rl.sample.episodes"
RL_TRAIN_STEPS_COUNT = "rl.train.steps"
RL_TRAIN_TOKENS_COUNT = "rl.train.tokens"
RL_TOKENS_RATE = "rl.tokens.rate"
RL_TOKENS_RATE_PER_GPU = "rl.tokens.rate_per_gpu"
RL_SAMPLES_RATE = "rl.samples.rate"
RL_SAMPLES_RATE_PER_GPU = "rl.samples.rate_per_gpu"

# Performance
RL_ENVIRONMENT_REWARD_MEAN = "rl.environment.reward.mean"
RL_ENVIRONMENT_EPISODE_LENGTH_MEAN = "rl.environment.episode.length.mean"
RL_TRAIN_LOSS = "rl.train.loss"

# Resource Utilization
RL_TRAIN_MFU = "rl.train.mfu"

class TelemetryConfig(TypedDict, total=False):
    enabled: bool
    service_name: str
    service_version: Optional[str]
    exporter_type: Literal["console", "otlp_http", "otlp_grpc", "none"]
    endpoint: Optional[str]

def _create_retrying_session(
    retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
    allowed_methods: tuple = ("POST",),
    debug: bool = False,
) -> requests.Session:
    """Creates a requests Session with automatic retries."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    print(f'Retry: {retry}')

    if debug:
        def log_retry_errors(response, *args, **kwargs):
            msg = ""
            try:
                if response.request and response.request.body:
                    content_type = response.request.headers.get("Content-Type", "")
                    if "application/x-protobuf" in content_type and HAS_PROTO:
                        if "v1/traces" in response.request.url:
                            proto_req = ExportTraceServiceRequest()
                            proto_req.ParseFromString(response.request.body)
                            spans = []
                            for res_span in proto_req.resource_spans:
                                for scope_span in res_span.scope_spans:
                                    for span in scope_span.spans:
                                        spans.append(span.name)
                            msg = f"Spans: {spans}"
                        elif "v1/metrics" in response.request.url:
                            proto_req = ExportMetricsServiceRequest()
                            proto_req.ParseFromString(response.request.body)
                            metrics = []
                            for res_metric in proto_req.resource_metrics:
                                for scope_metric in res_metric.scope_metrics:
                                    for metric in scope_metric.metrics:
                                        metrics.append(metric.name)
                            msg = f"Metrics: {metrics}"
                    elif "application/json" in content_type:
                         body = json.loads(response.request.body)
                         # Very basic JSON parsing - might need refinement based on exact structure
                         msg = f"JSON Body keys: {list(body.keys())}"

            except Exception as e:
                msg = f"(Failed to parse payload: {e})"

            if response.status_code >= 400:
                print(f"Telemetry export error: {response.status_code} {response.reason} - {msg} - {response.text}", file=sys.stderr)
            else:
                print(f"Telemetry export success: {response.status_code} {response.reason} - {msg}", file=sys.stderr)

        session.hooks["response"].append(log_retry_errors)
    return session

def setup_telemetry(config: TelemetryConfig):
    """Sets up OpenTelemetry tracer and meter providers."""
    service_name = config.get("service_name", "nemo_rl")
    service_version = config.get("service_version", "0.1.0")

    # Configure debug logging for OpenTelemetry
    debug = os.environ.get("NEMO_RL_OTEL_DEBUG", "0") == "1" or config.get("debug", False)
    print(f"Telemetry initialized with debug={debug}", file=sys.stderr)
    debug = True
    if debug:
        # Configure root logger for opentelemetry to output to stderr
        otel_logger = logging.getLogger("opentelemetry")
        otel_logger.setLevel(logging.DEBUG)
        
        # Configure urllib3 logger to see retries
        urllib3_logger = logging.getLogger("urllib3")
        urllib3_logger.setLevel(logging.DEBUG)

        # Avoid adding multiple handlers if already configured
        if not otel_logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            otel_logger.addHandler(handler)
            # Share the handler with urllib3
            urllib3_logger.addHandler(handler)
            
        print("enabled OpenTelemetry and urllib3 debug logging", file=sys.stderr)

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
            session = _create_retrying_session(debug=debug)
            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, session=session) if endpoint else OTLPSpanExporter(session=session)))
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
            session = _create_retrying_session(debug=debug)
            metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint, session=session) if endpoint else OTLPMetricExporter(session=session)))
        except ImportError as e:
            print(f"Error: opentelemetry-exporter-otlp not installed. Skipping OTLP HTTP metric exporter. Details: {e}", file=sys.stderr)
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

    # Using ExponentialBucketHistogramAggregation for duration metrics to scale automatically
    duration_metrics = [
        RL_LOOP_DURATION,
        RL_SAMPLE_DURATION,
        RL_REWARD_DURATION,
        RL_TRAIN_DURATION,
        RL_SYNC_DURATION,
        RL_STEP_DURATION
    ]
    views = []
    for metric_name in duration_metrics:
        views.append(View(
            instrument_name=metric_name,
            aggregation=ExponentialBucketHistogramAggregation()
        ))

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers, views=views)
    metrics.set_meter_provider(meter_provider)

    return trace.get_tracer(service_name), metrics.get_meter(service_name)

class LoggingInstrument:
    def __init__(self, name, instrument):
        self.name = name
        self.instrument = instrument

    def record(self, amount, attributes=None):
        self.instrument.record(amount, attributes)
        print(f"Instrument {self.name} recorded {amount} with attributes {attributes}", flush=True)

    def add(self, amount, attributes=None):
        self.instrument.add(amount, attributes)
        print(f"Instrument {self.name} added {amount} with attributes {attributes}", flush=True)

class GaugeLoggingInstrument(LoggingInstrument):
    def record(self, amount, attributes=None):
        self.instrument.set(amount, attributes)
        print(f"Instrument {self.name} set {amount} with attributes {attributes}", flush=True)

class RLTelemetry:
    def __init__(self, config: TelemetryConfig):
        tracer, self.meter = setup_telemetry(config)

        print(f"OpenTelemetry initialized for service '{config.get('service_name', 'nemo_rl')}' with exporter: {config.get('exporter_type', 'none')}", flush=True)
        def create_histogram(name, *args, **kwargs):
            hist = self.meter.create_histogram(name, *args, **kwargs)
            return LoggingInstrument(name, hist)

        def create_counter(name, *args, **kwargs):
            counter = self.meter.create_counter(name, *args, **kwargs)
            return LoggingInstrument(name, counter)

        def create_gauge(name, *args, **kwargs):
            gauge = self.meter.create_gauge(name, *args, **kwargs)
            return GaugeLoggingInstrument(name, gauge)

        # Histograms
        self.loop_duration = create_histogram(RL_LOOP_DURATION, unit="ms", description="End-to-end duration of one RL loop iteration")
        self.sample_duration = create_histogram(RL_SAMPLE_DURATION, unit="ms", description="Duration of the sampling phase")
        self.reward_duration = create_histogram(RL_REWARD_DURATION, unit="ms", description="Duration of the reward calculation phase")
        self.train_duration = create_histogram(RL_TRAIN_DURATION, unit="ms", description="Duration of the training phase")
        self.sync_duration = create_histogram(RL_SYNC_DURATION, unit="ms", description="Duration of the weight synchronization phase")
        self.step_duration = create_histogram(RL_STEP_DURATION, unit="ms", description="End-to-end duration of one RL step")
        
        # Counters
        self.sample_samples = create_counter(RL_SAMPLE_SAMPLES_COUNT, description="Number of samples generated")
        self.sample_episodes = create_counter(RL_SAMPLE_EPISODES_COUNT, description="Number of episodes completed")
        self.train_steps = create_counter(RL_TRAIN_STEPS_COUNT, description="Number of training steps completed")
        self.train_tokens = create_counter(RL_TRAIN_TOKENS_COUNT, description="Number of tokens processed in training")

        # Gauges/Histograms for values
        self.reward_mean = create_gauge(RL_ENVIRONMENT_REWARD_MEAN, description="Mean reward achieved")
        self.episode_length_mean = create_gauge(RL_ENVIRONMENT_EPISODE_LENGTH_MEAN, description="Mean episode length")
        self.train_loss = create_gauge(RL_TRAIN_LOSS, description="Training loss")
        self.tokens_rate = create_gauge(RL_TOKENS_RATE, description="End-to-end tokens per second")
        self.tokens_rate_per_gpu = create_gauge(RL_TOKENS_RATE_PER_GPU, description="End-to-end tokens per second per GPU")
        self.train_mfu = create_gauge(RL_TRAIN_MFU, description="Training Model Floating Point Utilization (MFU)")
        self.samples_rate = create_gauge(RL_SAMPLES_RATE, description="End-to-end samples per second")
        self.samples_rate_per_gpu = create_gauge(RL_SAMPLES_RATE_PER_GPU, description="End-to-end samples per second per GPU")

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
            print(f"Warning: Failed to flush telemetry data: {e}", flush=True)