import json
import os

# Configuration
DASHBOARD_TITLE = "NeMo-RL Training Dashboard"
METRIC_PREFIX = "prometheus.googleapis.com"
# Common filter for all queries - assumes running on GKE or similar where namespace_name is available
# We'll validly default to just matching the metric, user can filter by cluster/namespace in the dropdowns if needed.
# But for the dashboard JSON, we usually don't hardcode filters unless necessary.
RESOURCE_TYPE = "prometheus_target" # Standard for GMP
# RESOURCE_TYPE = "k8s_container" # Alternative if using legacy stackdriver adapter

# Metrics mapping (OTel name -> Prometheus name)
# Dots to underscores
def to_prom_name(otel_name):
    return otel_name

# Definition of metrics from nemo_rl/utils/otel.py
METRICS = {
    # Overview
    "loss": "rl.train.loss",
    "reward": "rl.environment.reward.mean",
    "episode_length": "rl.environment.episode.length.mean",
    "mfu": "rl.train.mfu",
    
    # Throughput
    "tokens_per_sec": "rl.tokens.rate",
    "tokens_per_sec_per_gpu": "rl.tokens.rate_per_gpu",
    "samples_per_sec": "rl.samples.rate",
    "samples_per_sec_per_gpu": "rl.samples.rate_per_gpu",
    "step_time": "rl.step.duration",

    # Latency Heatmaps
    "loop_duration": "rl.loop.duration",
    "sample_duration": "rl.sample.duration",
    "reward_duration": "rl.reward.duration",
    "train_duration": "rl.train.duration",
    "sync_duration": "rl.sync.duration",

    # Counters
    "samples_total": "rl.sample.samples",
    "episodes_total": "rl.sample.episodes",
    "steps_total": "rl.train.steps",
    "tokens_total": "rl.train.tokens",
}

# Helper to build a specific chart
def create_chart(title, metrics_list, chart_type="LINE", time_aggregation="MEAN"):
    data_sets = []
    for m_otel in metrics_list:
        prom_name = to_prom_name(m_otel)
        # Handle counters vs gauges/histograms
        # If it's a counter (total), we usually want rate.
        # But for "Total Steps", we might want the value itself? Usually rate is better for throughput. 
        # But wait, existing "throughput" metrics are already explicitly calculated as "samples_per_sec".
        # So "samples_total" is interesting as an absolute counter or rate.
        # Let's check config.
        
        # Heuristic: if it ends in 'total' or 'count', treat as counter? 
        # Actually in otel.py they are Counter objects.
        
        target_axis = "Y1"
        plot_type = "LINE"
        
        # Build query
        # We need to construct the Monitoring Query Language (MQL) or generally just use the builder for Prometheus.
        # GMP dashboards often use "prometheusQuery" field.
        
        query = f'{prom_name}'
        
        # If it's a histogram used as a Heatmap, we need the histogram metric
        if chart_type == "HEATMAP":
             # GMP histograms are typically queried as base name + suffix or just base name if distribution.
             # In Cloud Monitoring, for Prometheus, we use PromQL.
             query = f'{prom_name}' 
        
        data_sets.append({
            "minAlignmentPeriod": "60s",
            "plotType": plot_type,
            "targetAxis": target_axis,
            "timeSeriesQuery": {
                "prometheusQuery": query
            }
        })

    return {
        "dataSets": data_sets,
        "mode": "COLOR" if chart_type == "HEATMAP" else "COLOR",
        "chartOptions": {
            "mode": "COLOR"
        },
        "title": title
    }

def create_promql_widget(title, query, visualization_type="XY"):
    # visualization_type: "XY" (Line), "TABLE", "SCORECARD"
    # Note: Cloud Monitoring JSON has "xyChart" for lines.
    
    return {
        "title": title,
        "xyChart": {
            "dataSets": [
                {
                    "timeSeriesQuery": {
                        "prometheusQuery": query,
                    },
                    "plotType": "LINE",
                    "minAlignmentPeriod": "60s"
                }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
                "label": "y1Axis",
                "scale": "LINEAR"
            }
        }
    }

def create_heatmap_widget(title, metric_otel_name):
    # For heatmaps in GMP, we query the histogram bucket
    # PromQL: sum(rate(metric_bucket[5m])) by (le)
    prom_name = to_prom_name(metric_otel_name)
    # Assuming standard OTel -> Prom suffix
    # bucket name is usually metric_name + "_bucket"
    # But wait, python OTel might convert units. 
    # Let's assume standard `_bucket`.
    
    query = f'sum(rate({prom_name}_bucket[1m])) by (le)'
    
    return {
        "title": title,
        "xyChart": { # Heatmap is actually configured differently in JSON?
            # Cloud Monitoring Dashboards JSON reference says "heatmap": {} 
            # But that is for distribution metrics in Stackdriver.
            # For Prometheus, we might need to specifically use "xyChart" and hope it autodetects?
            # Actually, standard GMP dashboards use "mosaic" or special handling?
            # Let's stick to standard Heatmap widget if possible.
            # However, `heatmap` widget in Dashboard string expects a DISTRIBUTION value type.
            # A PromQL query returning buckets `by (le)` is essentially compatible if interpreted correctly,
            # but usually it's tricky.
            # A safer bet for generic PromQL Heatmaps in Google Cloud Console is strict bucket usage.
            # BUT, to be safe and simple: let's use percentiles on a Line Chart for distributions.
            # It is often more useful than a raw heatmap for quick debugging.
            # p50, p95, p99.
        }
    }

def create_percentile_widget(title, metric_otel_name):
    prom_name = to_prom_name(metric_otel_name)
    # Check if unit "ms" is used in otel.py, OTel might strip or append it. 
    # Usually metric name is preserved (swapping . for _).
    # otel.py: create_histogram(RL_LOOP_DURATION, unit="ms", ...) -> rl_loop_duration
    
    # We will plot p50, p90, p99
    queries = []
    
    for p in [50, 90, 99]:
        # histogram_quantile(0.99, sum(rate(rl_loop_duration_bucket[1m])) by (le))
        q = f'histogram_quantile({p/100.0}, sum(rate({prom_name}_bucket[1m])) by (le))'
        queries.append({
            "query": q,
            "legend": f"p{p}"
        })

    data_sets = []
    for item in queries:
        data_sets.append({
            "timeSeriesQuery": {
                "prometheusQuery": item["query"],
                "unitOverride": "ms" if "duration" in prom_name or "time" in prom_name else "1"
            },
            "plotType": "LINE",
            "legendTemplate": item["legend"],
            "minAlignmentPeriod": "60s"
        })

    return {
        "title": title,
        "xyChart": {
            "dataSets": data_sets,
            "yAxis": {
                "label": "Duration (ms)" if "duration" in prom_name or "time" in prom_name else "Value",
                "scale": "LINEAR"
            }
        }
    }

def create_simple_line_widget(title, metric_otel_name, rate=False):
    prom_name = to_prom_name(metric_otel_name)
    # If it is a distribution/histogram but we just want mean:
    # There is usually a _sum and _count. rate(_sum)/rate(_count) = average.
    
    # Check metric type from keys
    is_histogram = "duration" in metric_otel_name
    
    query = ""
    if is_histogram:
        # It's a histogram (or was created as one), so we have _sum and _count
        # However, for "per_sec" metrics (like tokens_per_sec), they are recorded as histograms of observed rates?
        # Or are they raw values? otel.py: self.tokens_per_sec = create_histogram(...)
        # So yes, they are histograms. The "mean" of 'tokens_per_sec' histogram is the average tokens/sec.
        query = f'sum(rate({prom_name}_sum[1m])) / sum(rate({prom_name}_count[1m]))'
    else:
        # Counter or Gauge
        if rate:
             # rate of a counter
             # otel counters usually suffix with _total
             query = f'rate({prom_name}_total[1m])'
        else:
             # Just value (Gauge) or raw Counter
             if "total" in prom_name or "samples" in prom_name or "steps" in prom_name or "tokens" in prom_name or "episodes" in prom_name:
                 # It's a counter, usually we want current value if it's "total steps so far"
                 query = f'{prom_name}_total'
             else:
                 query = f'{prom_name}'

    return {
        "title": title,
        "xyChart": {
            "dataSets": [{
                "timeSeriesQuery": {
                    "prometheusQuery": query,
                },
                "plotType": "LINE",
                "minAlignmentPeriod": "60s",
                "legendTemplate": "Values"
            }],
             "yAxis": {
                "scale": "LINEAR"
            }
        }
    }

# Build the layout
# Row 1: Overview (Loss, Reward, Step Time, MFU)
row_1 = [
    create_simple_line_widget("Train Loss", METRICS["loss"]),
    create_simple_line_widget("Reward Mean", METRICS["reward"]),
    create_percentile_widget("Step Time (ms)", METRICS["step_time"]),
    create_simple_line_widget("Training MFU", METRICS["mfu"])
]

# Row 2: Throughput
row_2 = [
    create_simple_line_widget("Tokens/sec", METRICS["tokens_per_sec"]),
    create_simple_line_widget("Tokens/sec/GPU", METRICS["tokens_per_sec_per_gpu"]),
    create_simple_line_widget("Samples/sec", METRICS["samples_per_sec"]),
    create_simple_line_widget("Samples/sec/GPU", METRICS["samples_per_sec_per_gpu"]),
]

# Row 3: Latency Breakdowns (Percentiles for deeper analysis)
row_3 = [
    create_percentile_widget("Loop Duration", METRICS["loop_duration"]),
    create_percentile_widget("Sample Duration", METRICS["sample_duration"]),
    create_percentile_widget("Reward Duration", METRICS["reward_duration"]),
    create_percentile_widget("Train Duration", METRICS["train_duration"]),
]

# Row 4: Progress (Total Counters)
row_4 = [
    create_simple_line_widget("Total Samples", METRICS["samples_total"]),
    create_simple_line_widget("Total Episodes", METRICS["episodes_total"]),
    create_simple_line_widget("Total Steps", METRICS["steps_total"]),
    create_simple_line_widget("Total Tokens", METRICS["tokens_total"]),
]

# Combine into Mosaic Layout
def create_mosaic_layout(rows):
    # Mosaic layout requires tiles with x,y,w,h
    # We assume a grid of 12 columns width? Actually Cloud Console grid is flexible.
    # standard is 2 columns for big charts, or 3-4 for small.
    # Let's do 2 columns per row for 4 items -> 2x2 grid in logical rows?
    # Or just 4 items across if user has big screen?
    # Let's stack them 2 per row for better visibility.
    
    # Actually, simpler to just list them in typical "gridLayout" or "mosaicLayout"
    # Mosaic is clearer.
    tiles = []
    
    current_y = 0
    COLUMNS = 2 # Charts per visual row
    WIDTH = 6   # out of 12? No, Mosaic uses arbitrary units usually, but let's say grid is 12-wide.
                # Actually default Mosaic grid is flexible. Let's assume 2 tiles wide.
    
    all_widgets = row_1 + row_2 + row_3 + row_4
    
    for i, widget in enumerate(all_widgets):
        row = i // COLUMNS
        col = i % COLUMNS
        
        tiles.append({
            "xPos": col * 6,
            "yPos": row * 4,
            "width": 6,
            "height": 4,
            "widget": widget
        })

    return {
        "displayName": DASHBOARD_TITLE,
        "mosaicLayout": {
            "columns": 12,
            "tiles": tiles
        }
    }

dashboard_json = create_mosaic_layout([row_1, row_2, row_3, row_4])

# Write to file
output_path = "tools/dashboard.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(dashboard_json, f, indent=2)

print(f"Generated dashboard at {output_path}")
print("Prometheus Query Preview:")
print("- " + dashboard_json["mosaicLayout"]["tiles"][0]["widget"]["xyChart"]["dataSets"][0]["timeSeriesQuery"]["prometheusQuery"])
