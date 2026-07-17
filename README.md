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

# Install
pip install -r requirements.txt

# Run a basic example
python examples/basic_classifier.py
```

---

## 📁 Project Structure

```
distilkit/
├── src/
│   ├── distiller.py     # Core distillation training loop
│   ├── teacher.py        # Teacher model loader/wrapper
│   ├── student.py        # Student model builder
│   ├── benchmarks.py     # Speed/accuracy benchmarking
│   └── onnx_export.py    # Model export utilities
├── examples/
│   └── basic_classifier.py  # Full distillation example
├── tests/
│   └── test_distiller.py
├── scripts/
│   └── distill.sh
├── requirements.txt
└── README.md
```

---

## 📊 Usage Example

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
