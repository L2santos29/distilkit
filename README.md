# ⚡ DistilKit

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Status](https://img.shields.io/badge/Status-Pre--Alpha-FF6B35)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Knowledge Distillation Framework for Model Compression and Deployment**

> Teacher → Student training loop. Quantize. Export to ONNX. Benchmark on CPU, GPU, and NPU. Production-ready model compression.

---

## 🎯 What is DistilKit?

DistilKit is a lightweight framework for **knowledge distillation** — the technique of compressing large, powerful models (teachers) into smaller, faster deployable versions (students) that retain most of the original accuracy.

- **Train** student models that mimic larger teachers
- **Quantize** to INT8/FP16 for inference efficiency
- **Export** to ONNX for cross-platform deployment
- **Benchmark** speed and accuracy across CPU, GPU, and NPU targets
- **Compare** teacher vs. student in a single report

---

## 🧪 How Knowledge Distillation Works

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

## 🔧 Stack

| Component | Tech |
|-----------|------|
| **Training** | PyTorch 2.6+ |
| **Distillation** | Custom loss (KL divergence + cross-entropy) |
| **Quantization** | PyTorch quantization + ONNX Runtime |
| **Export** | ONNX, TorchScript |
| **Benchmarks** | Custom timing + throughput measurement |

---

## 🚀 Quick Start

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

## 🖥️ CLI Mode

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
| `--teacher` | `resnet18` | Teacher architecture (`resnet18`, `resnet50`, `mobilenet_v2`, `efficientnet_b0`, etc.) |
| `--epochs` | `10` | Training epochs |
| `--temperature` | `4.0` | Softening factor for distillation |
| `--alpha` | `0.7` | Distillation loss weight (0-1) |
| `--batch-size` | `64` | Batch size |
| `--export` | `none` | Export format (`onnx`, `torchscript`, or `none`) |
| `--benchmark` | `cpu` | Benchmark target (`cpu`, `cuda`, or `none`) |
| `--output-dir` | `checkpoints` | Export directory |

---

## 🎨 GUI Mode (Web)

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

## 📁 Project Structure

```
distilkit/
├── pyproject.toml         # Package config with CLI entry points
├── src/
│   ├── cli.py             # CLI interface (argparse)
│   ├── webapp.py          # Web GUI (FastAPI + Tailwind CSS)
│   ├── templates/
│   │   └── index.html     # Web frontend (Tailwind CSS + Chart.js)
│   ├── distiller.py       # Core distillation training loop
│   ├── teacher.py         # Teacher model loader
│   ├── student.py         # Student model builder
│   ├── benchmarks.py      # Speed/accuracy benchmarking
│   └── onnx_export.py     # ONNX / TorchScript export utilities
├── examples/
│   └── basic_classifier.py   # Full distillation example
├── tests/
│   └── test_distiller.py
├── scripts/
│   └── distill.sh
├── requirements.txt
└── README.md
```

---

## 🐍 Python API Example

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

## 📈 Expected Results (ResNet → Mini-ResNet, CIFAR-10)

| Model | Parameters | Accuracy | Inference (CPU) | Size |
|-------|-----------|----------|-----------------|------|
| Teacher (ResNet50) | 25.6M | 95.2% | 12.4 ms | 98 MB |
| Student | 6.4M | 93.1% | 1.2 ms | 24 MB |
| **Saving** | **75% ↓** | **-2.1%** | **10.3x faster** | **75% ↓** |

---

## 🏗️ Status

**Pre-Alpha** — Core distillation loop and benchmarking framework complete. Currently implementing ONNX Runtime integration and NPU benchmark targets.

---

## 📄 License

MIT — See [LICENSE](LICENSE)
