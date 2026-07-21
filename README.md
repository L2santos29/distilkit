# ⚡ DistilKit

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Knowledge Distillation Framework for Model Compression and Deployment**

> Teacher → Student training loop. Quantize. Export to ONNX. Benchmark on CPU, GPU, and NPU. Production-ready model compression.

---

## What is DistilKit?

DistilKit is a lightweight framework for **knowledge distillation** — the technique of compressing large, powerful models (teachers) into smaller, faster deployable versions (students) that retain most of the original accuracy.

- **Train** student models that mimic larger teachers
- **Quantize** to INT8/FP16 for inference efficiency
- **Export** to ONNX for cross-platform deployment
- **Benchmark** speed and accuracy across CPU, GPU, and NPU targets
- **Compare** teacher vs. student in a single report

---

## How Knowledge Distillation Works

```
┌─────────────────────────────────────────────────────┐
│                    TRAINING PHASE                    │
│                                                     │
│  ┌──────────┐    soft predictions ("dark            │
│  │ Teacher  │─────────── knowledge") ──────────┐    │
│  │ (large)  │                                  │    │
│  └──────────┘                                  │    │
│                                                ▼    │
│           ┌──────────────────────────────────┐      │
│           │       DISTILLATION LOSS           │      │
│           │  α * KL_div(student, teacher)     │      │
│           │  + (1-α) * CE(student, labels)    │      │
│           └──────────────────────────────────┘      │
│                              │                      │
│  ┌──────────┐                ▼                      │
│  │ Student  │◄─────── backpropagation               │
│  │ (small)  │                                       │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│                  DEPLOYMENT PHASE                    │
│                                                     │
│  Student → Quantize → ONNX → Benchmark              │
│                                                     │
│  Results: 95% accuracy, 10x faster, 5x smaller      │
└─────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Tech |
|-----------|------|
| **Training** | PyTorch 2.6+ |
| **Distillation** | Custom loss (KL divergence + cross-entropy) |
| **Quantization** | PyTorch quantization + ONNX Runtime |
| **Export** | ONNX, TorchScript |
| **Benchmarks** | Custom timing + throughput measurement |
| **Web GUI** | FastAPI + Uvicorn + SSE |
| **Frontend** | Tailwind CSS + Chart.js |
| **Config** | Environment variables via ``settings.py`` |
| **Monitoring** | Prometheus metrics + Grafana dashboard |
| **Tracing** | W3C Trace Context + optional OpenTelemetry |
| **Auth** | API key via ``X-API-Key`` header |
| **Rate Limiting** | Sliding-window per-IP middleware |
| **Circuit Breaker** | 3-state for external network calls |
| **Alerting** | In-process alert manager + webhook |

---

## Quick Start

```bash
# Clone
git clone https://github.com/L2santos29/distilkit.git
cd distilkit

# Install dependencies
pip install -r requirements.txt

# --- Option 1: CLI mode ---
python -m src.cli train --teacher resnet18 --epochs 5

# --- Option 2: GUI mode (opens browser) ---
python -m src.webapp
# Or: bash run_gui.sh

# --- Option 3: Install as package ---
pip install -e .
distilkit train --teacher resnet50 --epochs 10 --export onnx --benchmark cpu
distilkit gui
```

---

## CLI Mode

Train, benchmark, and export models directly from the terminal.

```bash
# Full training pipeline
distilkit train --teacher resnet50 --epochs 10 --temperature 4.0 --alpha 0.7 \
                --batch-size 64 --export onnx --benchmark cpu

# Benchmark an exported model
distilkit benchmark --model checkpoints/student.onnx --target cpu --runs 100

# Export a PyTorch checkpoint
distilkit export --model model.pth --format onnx --output model.onnx

# Launch the GUI
distilkit gui
```

### Options for `train`

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `CIFAR-10` | Dataset (`CIFAR-10`, `MNIST`, `FashionMNIST`, etc.) |
| `--teacher` | `resnet18` | Teacher architecture (`resnet18`, `resnet50`, `mobilenet_v2`, `efficientnet_b0`, etc.) |
| `--epochs` | `10` | Training epochs |
| `--temperature` | `4.0` | Softening factor for distillation |
| `--alpha` | `0.7` | Distillation loss weight (0-1) |
| `--compression-ratio` | `0.05` | Target student/teacher parameter ratio |
| `--batch-size` | `64` | Batch size |
| `--patience` | `0` | Early stopping patience (0 to disable) |
| `--export` | `none` | Export format (`onnx`, `torchscript`, or `none`) |
| `--benchmark` | `cpu` | Benchmark target (`cpu`, `cuda`, or `none`) |
| `--output-dir` | `checkpoints` | Export directory |
| `--ckpt-every` | `5` | Save checkpoint every N epochs (0 to disable) |
| `--data-dir` | `./data` | Dataset cache directory |

---

## GUI Mode (Web)

Web-based interface built with **FastAPI** + **Tailwind CSS** + **Chart.js**.

```bash
# Launch (via script)
bash run_gui.sh

# Launch (via Python)
python -m src.webapp

# Launch (via installed CLI)
pip install -e .
distilkit gui
```

Opens `http://localhost:7860` in your browser with:

| Feature | Description |
|---------|-------------|
| 🧠 **Teacher selector** | Dropdown with 8 architectures (ResNet, MobileNet, EfficientNet) |
| ⚙️ **Hyperparameters** | Sliders for epochs, temperature, alpha, batch size |
| 🚀 **Training** | Starts distillation, shows **live progress bar** |
| 📈 **Chart** | Loss & accuracy curves that update in real time (Chart.js) |
| ✅ **Results table** | Teacher vs. student comparison (params, latency, throughput) |
| 📦 **Export** | One-click export to ONNX or TorchScript |
| 📋 **Logs** | Full console output in a scrollable terminal

---

## Project Structure

```
distilkit/
├── pyproject.toml                  # Package config with CLI entry points
├── Dockerfile                      # Multi-stage build for deployment
├── docker-compose.yml              # Development environment (hot-reload)
├── docker-compose.staging.yml      # Staging environment (production-like)
├── docker-compose.prod.yml         # Production deployment
├── prometheus/
│   ├── prometheus.yml              # Prometheus scrape config
│   ├── alert_rules.yml             # Alerting rules (8 rules)
│   └── dashboards/
│       └── distilkit.json          # Grafana dashboard (9 panels)
├── src/
│   ├── cli.py                      # CLI interface (argparse)
│   ├── webapp.py                   # Web GUI (FastAPI + Tailwind CSS)
│   ├── templates/
│   │   └── index.html              # Web frontend (Tailwind CSS + Chart.js)
│   ├── settings.py                 # Centralized env-var configuration
│   ├── pipeline.py                 # Shared distillation pipeline orchestration
│   ├── task_manager.py             # Background task management + history
│   ├── distiller.py                # Core distillation training loop
│   ├── teacher.py                  # Teacher model loader + cache
│   ├── student.py                  # Student model builder
│   ├── datasets.py                 # Dataset loading utilities
│   ├── benchmarks.py               # Speed/accuracy benchmarking
│   ├── onnx_export.py              # ONNX / TorchScript export utilities
│   ├── circuit_breaker.py          # 3-state circuit breaker for network calls
│   ├── alert_manager.py            # In-process alert evaluation + webhook
│   ├── tracing.py                  # Distributed tracing (W3C + OTel)
│   ├── log_config.py               # Structured logging with context
│   └── __init__.py
├── examples/
│   └── basic_classifier.py         # Full distillation example
├── tests/
│   ├── test_distiller.py           # Unit tests: distiller, teacher, student
│   ├── test_webapp.py              # Web app: endpoints, middleware, auth
│   ├── test_cli.py                 # CLI: parsing, export commands
│   ├── test_integration.py         # Integration: export, benchmark, pipeline
│   ├── test_coverage.py            # Edge-case coverage: circuit breaker,
│   │                               #   rate limiter, security headers,
│   │                               #   tracing, CLI edge paths
│   └── test_regression.py          # Regression: previously-fixed bugs
├── scripts/
│   ├── distill.sh
│   └── measure_perf.py
├── .env.example                    # Environment variable reference
├── requirements.txt
└── README.md
```

---

## Python API Example

```python
from src.distiller import Distiller
from src.teacher import load_teacher
from src.student import build_student

# Load a large teacher model
teacher = load_teacher("resnet50")

# Build a smaller student
student = build_student(teacher, compression_ratio=0.25)

# Distill
distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
history = distiller.train(train_loader, val_loader, epochs=10)

# Export
from src.onnx_export import export_to_onnx
export_to_onnx(distiller.student, "student_model.onnx")

# Benchmark
from src.benchmarks import benchmark
results = benchmark("student_model.onnx", target="cpu")
print(f"Inference time: {results['mean_ms']:.2f} ms")
```

---

## Expected Results (ResNet → Mini-ResNet, CIFAR-10)

| Model | Parameters | Accuracy | Inference (CPU) | Size |
|-------|-----------|----------|-----------------|------|
| Teacher (ResNet50) | 25.6M | 95.2% | 12.4 ms | 98 MB |
| Student | 6.4M | 93.1% | 1.2 ms | 24 MB |
| **Saving** | **75% ↓** | **-2.1%** | **10.3x faster** | **75% ↓** |

---

## Configuration

All configuration is done through environment variables. Copy ``.env.example`` to ``.env``
and adjust as needed — all variables are optional with sensible defaults.

| Variable | Default | Choices | Description |
|----------|---------|---------|-------------|
| ``DEVICE`` | ``cpu`` | cpu, cuda, npu | Target device for training/inference |
| ``HOST`` | ``127.0.0.1`` | — | Web server bind address (use ``0.0.0.0`` for Docker) |
| ``PORT`` | ``7860`` | — | Web server port |
| ``API_ONLY`` | ``false`` | true, false | Run in API-only mode (no frontend) |
| ``RUNS_DIR`` | ``runs`` | — | Directory for persisted run history |
| ``DATA_DIR`` | ``./data`` | — | Dataset cache directory |
| ``CHECKPOINTS_DIR`` | ``checkpoints`` | — | Export and checkpoint directory |
| ``MAX_LOG_SIZE`` | ``100000`` | — | Max characters in in-memory log buffer |
| ``LOG_LEVEL`` | ``INFO`` | DEBUG, INFO, WARNING, ERROR | Logging verbosity |
| ``API_KEY`` | `` (empty)`` | Any string | API key for endpoint authentication. Set to enable auth — all API requests must include ``X-API-Key`` header. |
| ``CORS_ORIGINS`` | ``*`` | Comma-separated origins | CORS allowed origins. Set to specific origins (e.g. ``http://localhost:3000,https://app.example.com``) in production. |
| ``RATE_LIMIT_PER_MINUTE`` | ``30`` | Integer (0 to disable) | Max requests per minute per IP. Protects against DoS and abuse. |
| ``HSTS_MAX_AGE`` | ``0`` | Integer (seconds) | HSTS max-age for HTTPS enforcement. Set to ``31536000`` in production behind TLS. 0 = disabled. |
| ``ALERT_WEBHOOK_URL`` | `` (empty)`` | Webhook URL (Slack, Discord) | Alert notifications endpoint. When set, alerts are POSTed as JSON to this URL. Empty = alerts are logged only. |
| ``OTEL_EXPORTER_OTLP_ENDPOINT`` | ``http://localhost:4318`` | URL | OpenTelemetry OTLP HTTP endpoint for trace export. Only used when ``opentelemetry`` packages are installed. |

---

## Deployment

DistilKit supports three environments: **development**, **staging**, and **production**.

### Development (docker-compose.yml)

```bash
docker compose up -d
# → http://localhost:7860 — hot-reload enabled, no auth, no rate limits
```

Mounts source code for live editing. Best for active development.

### Staging (docker-compose.staging.yml)

Validates a production-like deployment before releasing to production.

```bash
# Build the image and start staging
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d --build
# → http://localhost:7861 — auth enabled, restricted CORS, rate limited

# Smoke test
curl -s http://localhost:7861/health
curl -s -H "X-API-Key: changeme-staging-key" http://localhost:7861/api/v1/config

# Tear down
STAGING_API_KEY=changeme-staging-key \
  docker compose -f docker-compose.yml -f docker-compose.staging.yml down -v
```

Key differences from development:
- Runs the built image (no source mount) — validates what ships to production
- Authentication enabled via ``API_KEY``
- Rate limiting active (60 req/min)
- Separate port (7861) and volumes
- Health check configured

### Production (docker-compose.prod.yml)

```bash
# 0. Prerequisites — create a .env file or export variables
export PROD_API_KEY="your-secure-key"
export PROD_CORS_ORIGINS="https://app.example.com"
export PROD_PORT=7860

# 1. Build and tag the release
pip freeze > requirements.lock
docker build -t distilkit:0.1.0 .
docker tag distilkit:0.1.0 distilkit:latest

# 2. Deploy
docker compose -f docker-compose.prod.yml up -d

# 3. Verify
curl -s http://localhost:7860/health

# 4. Check logs
docker compose -f docker-compose.prod.yml logs -f
```

### Rollback

**Git-based** (CLI-only deployments):

```bash
git log --oneline -10              # Find the last known-good commit
git checkout <known-good-sha>
pip install -e .                   # Reinstall the previous version
```

**Docker-based:**

```bash
# Roll back to the previous tagged release
docker pull distilkit:0.0.9        # or use a specific version
docker tag distilkit:0.0.9 distilkit:latest
docker compose -f docker-compose.prod.yml up -d

# Verify health before declaring success
curl -sf http://localhost:7860/health && echo "Rollback OK"
```

**Blue-green strategy** (requires two hosts or port sets):

| Step | Action |
|------|--------|
| 1 | Deploy new version on ``BLUE`` (e.g. port 7860) while ``GREEN`` (port 7861) serves live traffic |
| 2 | Switch load balancer / reverse proxy from GREEN → BLUE |
| 3 | Keep GREEN running for instant rollback |
| 4 | If issues arise, switch the proxy back to GREEN |

If the deployment fails health checks, the old container stays running and the orchestrator (or operator) reverts to it immediately.

### Environment Variable Reference

See the [Configuration](#configuration) table above for the full list of supported environment variables.

---

## Monitoring

DistilKit includes two complementary alerting mechanisms.

### Built-in Alert Manager

The in-process alert manager evaluates conditions every 60 seconds:

| Condition | Threshold | Severity |
|-----------|-----------|----------|
| 5xx error rate | > 5% in 5 min | Warning |
| Multiple 5xx errors | ≥ 5 errors in 5 min | Warning |
| Training task failures | ≥ 3 failures in 1 hour | Warning |
| Sustained issues | 5 consecutive alert cycles | Critical |

Alerts are always written to the log at ``WARNING`` or ``CRITICAL`` level.
To receive external notifications, set ``ALERT_WEBHOOK_URL`` in your environment
pointing to a Slack / Discord / custom webhook:

```bash
export ALERT_WEBHOOK_URL="https://hooks.slack.com/services/T00/B00/xxx"
```

### Prometheus Alert Rules

If you run Prometheus, a set of alert rules is provided in ``prometheus/alert_rules.yml``:

- **DistilKitDown** — instance unreachable for > 1 min
- **DistilKitHealthCheckFailing** — /health non-200
- **HighErrorRate** — non-zero 5xx rate for 2 min
- **SustainedErrorRate** — > 0.1 errors/sec for 5 min
- **HighLatency** — cumulative request duration rising fast
- **TaskFailureRate** — tasks stuck in "failed" state
- **NoRecentCompletions** — no training activity for 24h
- **HighUptimeReset** — server recently restarted

To use them, add the rules file to your Prometheus config:

```yaml
rule_files:
  - /etc/prometheus/alert_rules.yml
```

A sample Prometheus scrape config is at ``prometheus/prometheus.yml``.

### Distributed Tracing

Every HTTP request creates a tracing span with W3C ``traceparent`` header
propagation.  Key pipeline operations (dataset loading, teacher loading,
distillation training, benchmarking, export) are also instrumented with
child spans, providing end-to-end visibility into request flows.

**Built-in tracer** (no extra dependencies):

- Spans carry a ``trace_id`` and ``span_id`` that propagate via the
  ``traceparent`` response header
- The ``X-Request-ID`` header is linked to the trace for log correlation
- Span timing and attributes are available in application logs

**OpenTelemetry** (optional, for Jaeger / Zipkin / any OTLP backend):

Uncomment the ``opentelemetry`` lines in ``requirements.txt`` and install:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

The tracer auto-detects OpenTelemetry on import and exports spans via OTLP
HTTP to ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default ``http://localhost:4318``).

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
```

You can verify tracing is active by checking the startup log line:

```
OpenTelemetry tracing enabled — exporting to http://jaeger:4318/v1/traces
```

### Grafana Dashboard

A pre-configured Grafana dashboard is provided at
``prometheus/dashboards/distilkit.json``.  Import it into Grafana (or use the
Grafana API) to get an at-a-glance view of system health.

**Panels included:**

| Panel | Metric | Description |
|-------|--------|-------------|
| Uptime | ``distilkit_uptime_seconds`` | Time since last restart |
| Request Rate | ``rate(distilkit_requests_total[5m])`` | HTTP requests/sec |
| Error Rate (5xx) | ``rate(distilkit_errors_total[5m])`` | 5xx errors/sec with thresholds |
| Total Requests | ``distilkit_requests_total`` | Cumulative request count |
| Tasks by Status | ``distilkit_tasks_total{status=~"..."}`` | Pie chart of all task states |
| Active Tasks | ``distilkit_tasks_total{status="running"}`` | Gauge with orange/red thresholds |
| Requests by Path | ``distilkit_requests_per_path`` | Horizontal bar chart of top 10 |
| Failed Tasks | ``distilkit_tasks_total{status="failed"}`` | Time series of failures |
| Instance Info | Multiple | Aggregate stats table |

**Quick start with Docker:**

```yaml
# Add to docker-compose.yml or docker-compose.prod.yml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus:ro
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./prometheus/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

1. Start the stack: ``docker compose up -d``
2. Open Grafana at ``http://localhost:3000``
3. Add Prometheus as a datasource (``http://prometheus:9090``)
4. Import the dashboard from ``prometheus/dashboards/distilkit.json``

The dashboard auto-refreshes every 30 seconds and annotates restarts
automatically when ``distilkit_uptime_seconds`` resets.

---

## Status

**Pre-Alpha** — Core distillation loop and benchmarking framework complete. Currently implementing ONNX Runtime integration and NPU benchmark targets.

---

## License

MIT — See [LICENSE](LICENSE)
