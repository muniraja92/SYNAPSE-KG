# SYNAPSE-KG

**Semantic-Yielding Neural Attention with Path Synthesis and Encoding for Knowledge Graph Enhanced Recommendations**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/)

---

## 📋 Overview

SYNAPSE-KG is a novel recommendation framework that integrates knowledge graph information with deep neural architectures for enhanced recommendation quality. The framework introduces five innovative components that work synergistically through meta-learning fusion to achieve state-of-the-art performance.

<p align="center">
  <img src="figures/Figure1_SYNAPSE-KG_Architecture.pdf" alt="SYNAPSE-KG Architecture" width="800"/>
</p>

### Key Results

| Metric | SYNAPSE-KG | Best Baseline | Improvement |
|--------|------------|---------------|-------------|
| NDCG@10 | **0.3915** | 0.3701 (MF-BPR) | +5.78% |
| HR@10 | **0.6441** | 0.6407 (MF-BPR) | +0.53% |
| MRR@10 | **0.3139** | 0.2867 (MF-BPR) | +9.49% |

---

## 🚀 Key Features

SYNAPSE-KG comprises five novel components:

| Component | Full Name | Description |
|-----------|-----------|-------------|
| **DRIP** | Dynamic Relational Intent Propagation | User intent-aware attention over interaction histories conditioned on target items |
| **MINT** | Momentum-based Interest Network Tracking | Contrastive preference learning with exponential moving average updates |
| **BRIDGE** | Bidirectional Relational Information Diffusion with Graph Encoding | Bidirectional propagation with residual connections for graph structure capture |
| **NEXIS** | Neural Expert Interest Synthesis | Multi-interest extraction through attention-based aggregation |
| **ORACLE** | Optimal Routing And Component-Level Ensembling | Meta-learning fusion with adaptive gating mechanism |

---

## 📊 Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  ┌───────────────────┐  │
│  │ User ID  │  │ Item ID  │  │ Interaction Hist. │  │  Knowledge Graph  │  │
│  └────┬─────┘  └────┬─────┘  └─────────┬─────────┘  └─────────┬─────────┘  │
└───────┼─────────────┼──────────────────┼────────────────────────┼───────────┘
        │             │                  │                        │
        ▼             ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EMBEDDING LAYER                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  ┌───────────────────┐  │
│  │ User Emb │  │ Item Emb │  │   History Emb     │  │    Entity Emb     │  │
│  └────┬─────┘  └────┬─────┘  └─────────┬─────────┘  └─────────┬─────────┘  │
└───────┼─────────────┼──────────────────┼────────────────────────┼───────────┘
        │             │                  │                        │
        ▼             ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NOVEL COMPONENTS LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │     DRIP     │  │     MINT     │  │    BRIDGE    │  │    NEXIS     │    │
│  │ Intent-Aware │  │  Contrastive │  │ Bidirectional│  │ Multi-Interest│   │
│  │  Attention   │  │   Learning   │  │  Propagation │  │  Extraction  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BASELINE COMPONENTS                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   ItemKNN    │  │   Content    │  │  Popularity  │  │   LightGCN   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │                 │
          └────────────┬────┴────────┬────────┴────────┬────────┘
                       │             │                 │
                       ▼             ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORACLE META-LEARNING                                │
│                    ┌─────────────────────────────┐                          │
│                    │   Adaptive Gating Fusion    │                          │
│                    │   w = softmax(W·[u,i]/τ)    │                          │
│                    └──────────────┬──────────────┘                          │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             OUTPUT LAYER                                     │
│                    ┌─────────────────────────────┐                          │
│                    │      Prediction Score       │                          │
│                    │      ŷ = Σ wₖ · sₖ          │                          │
│                    └─────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/muniraja92/SYNAPSE-KG.git
cd SYNAPSE-KG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

---

## 📁 Dataset

### MovieLens-1M

The framework is evaluated on the MovieLens-1M benchmark dataset.

| Statistic | Value |
|-----------|-------|
| Users | 2,082 |
| Items | 3,105 |
| Interactions | 213,214 |
| Sparsity | 96.87% |
| Avg. interactions/user | 102.4 |

### Knowledge Graph Statistics

| Statistic | Value |
|-----------|-------|
| Entities | 3,105 |
| Relations | 5 |
| Triples | 18,492 |
| Relation Types | genre, year, director, actor, writer |

### Download & Preparation

```bash
# Download MovieLens-1M
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/

# Preprocess data
python scripts/preprocess_data.py --data_dir data/ml-1m --output_dir data/processed
```

---

## 🔧 Usage

### Training

```bash
# Train SYNAPSE-KG with default parameters
python train.py --config configs/synapse_kg.yaml

# Train with custom parameters
python train.py \
    --embedding_dim 64 \
    --num_layers 2 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --epochs 100 \
    --device cuda:0
```

### Configuration

Key hyperparameters in `configs/synapse_kg.yaml`:

```yaml
model:
  embedding_dim: 64
  num_attention_heads: 4
  num_gnn_layers: 2
  dropout: 0.1
  temperature: 0.07
  momentum: 0.999
  num_interests: 4

training:
  learning_rate: 0.001
  batch_size: 256
  epochs: 100
  early_stopping_patience: 10
  
evaluation:
  metrics: [HR, NDCG, MRR]
  top_k: [5, 10, 20]
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint checkpoints/synapse_kg_best.pt --metrics all

# Run ablation study
python ablation_study.py --config configs/ablation.yaml
```

### Inference

```python
from synapse_kg import SYNAPSE_KG

# Load trained model
model = SYNAPSE_KG.load_from_checkpoint('checkpoints/synapse_kg_best.pt')

# Get recommendations for a user
user_id = 42
recommendations = model.recommend(user_id, top_k=10)
print(f"Top-10 recommendations for user {user_id}: {recommendations}")
```

---

## 📈 Results

### Main Results

<p align="center">
  <img src="figures/Figure2_Performance_Radar.pdf" alt="Performance Comparison" width="600"/>
</p>

### Component Weight Analysis

The ORACLE meta-learner learns to adaptively weight each component:

| Component | Average Weight |
|-----------|----------------|
| LightGCN | 56.4% |
| ItemKNN | 21.2% |
| Content | 15.7% |
| BRIDGE | 3.0% |
| DRIP | 0.6% |
| NEXIS | 2.7% |
| MINT | 0.3% |
| Popularity | 0.1% |

### Ablation Study

| Configuration | NDCG@10 | Δ |
|---------------|---------|---|
| **Full SYNAPSE-KG** | **0.3915** | — |
| w/o BRIDGE | 0.3836 | -2.02% |
| w/o NEXIS | 0.3910 | -0.13% |
| w/o DRIP | 0.3951 | +0.92% |
| w/o MINT | 0.3933 | +0.46% |

### Statistical Significance

All improvements over baselines are statistically significant (p < 0.001).

---

## 📂 Project Structure

```
SYNAPSE-KG/
├── configs/
│   ├── synapse_kg.yaml
│   └── ablation.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── models/
│   │   ├── synapse_kg.py
│   │   ├── drip.py
│   │   ├── mint.py
│   │   ├── bridge.py
│   │   ├── nexis.py
│   │   └── oracle.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── knowledge_graph.py
│   ├── trainers/
│   │   └── trainer.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── scripts/
│   ├── preprocess_data.py
│   └── run_experiments.sh
├── notebooks/
│   └── analysis.ipynb
├── checkpoints/
├── logs/
├── figures/
├── train.py
├── evaluate.py
├── ablation_study.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{muniraja2025synapse,
  title={SYNAPSE-KG: Semantic-Yielding Neural Attention with Path Synthesis and Encoding for Knowledge Graph Enhanced Recommendations},
  author={Muniraja, Pasupuleti and Satapathy, Shashank Mouli},
  journal={},
  year={2025},
  publisher={}
}
```

---

## 🙏 Acknowledgments

- **Vellore Institute of Technology** for computational resources
- **GroupLens Research** at University of Minnesota for the MovieLens dataset
- The authors of LightGCN, KGAT, and other baseline methods

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Pasupuleti Muniraja** - [pasupuleti.muniraja2020@vitstudent.ac.in](mailto:pasupuleti.muniraja2020@vitstudent.ac.in)
- **Dr. Shashank Mouli Satapathy** - [shashankmouli.s@vit.ac.in](mailto:shashankmouli.s@vit.ac.in)

School of Computer Science and Engineering  
Vellore Institute of Technology  
Vellore, Tamil Nadu, India - 632014

---

<p align="center">
  <b>⭐ If you find this repository helpful, please consider giving it a star! ⭐</b>
</p>
