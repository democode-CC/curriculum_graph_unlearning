# Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning

This repository provides the implementation for the paper **"Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning"**.

## Overview

Catastrophic collapse in graph neural network (GNN) unlearning leads to a sharp decline in model utility. This codebase introduces two main solutions:
- **Negative Preference Optimization (NPO):** A custom loss designed to prevent utility drops during unlearning.
- **Curriculum Unlearning:** Gradual unlearning of the forget set from simple to complex items to improve stability.

---

## Installation

**Requirements:**  
Python 3.8+, PyTorch 1.12+, PyTorch Geometric 2.0+, NetworkX, NumPy, Pandas, Matplotlib, Seaborn, tqdm

**Quick Conda Setup:**
```bash
cd curriculum_graph_unlearning
conda env create -f environment.yml    # or environment_gpu.yml for GPU
conda activate graph_unlearning        # or graph_unlearning_gpu
python test_installation.py
```

**Manual Installation:**
```bash
git clone <repository-url>
cd curriculum_graph_unlearning
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python test_installation.py
```

See [SETUP.md](SETUP.md) for details.

---

## Git Repository Setup

**Automated Git Initialization:**
```bash
./setup_git.sh
```
This initializes git, configures user info, creates an initial commit, and can connect to GitHub/GitLab.

**Manual Setup:**
```bash
git init
git add .
git commit -m "Initial commit: Graph unlearning implementation"
git remote add origin https://github.com/yourusername/curriculum_graph_unlearning.git  # optional
git branch -M main
git push -u origin main
```

See [GIT_SETUP.md](GIT_SETUP.md) for more info.

---

## Project Structure

```
curriculum_graph_unlearning/
│
├── my_parser.py                # Argument parser
├── learning.py                 # GNN training
├── unlearning.py               # Unlearning methods
├── evaluation_1.py             # Baseline comparisons
├── evaluation_2.py             # Hyperparameter analysis
│
├── gnn_model/                  # GNN models: GCN, GAT, GraphSAGE, RGCN, CompGCN
├── data/                       # Data loading
├── stored_model/               # Saved models
├── results/                    # Experimental results
└── README.md
```

---

## Quick Start

**Train a GNN:**
```bash
python learning.py --dataset Cora --gnn_model GCN --learning_epochs 200
python learning.py --dataset FB15k237 --gnn_model RGCN --learning_epochs 200
```

**Unlearning Methods:**
```bash
# Gradient Ascent baseline
python -c "from my_parser import get_args_with_custom; from learning import train_model; from unlearning import unlearn; from data import load_dataset, get_forget_retain_split; import torch; args = get_args_with_custom(['--dataset', 'Cora', '--gnn_model', 'GCN', '--unlearn_method', 'gradient_ascent', '--unlearn_rate', '0.1']); model, data, is_kg = train_model(args); forget_mask, retain_mask = get_forget_retain_split(data, 0.1, is_kg); unlearn(args, model, data, is_kg, forget_mask, retain_mask)"

# Full method (Curriculum + NPO)
python -c "from my_parser import get_args_with_custom; from learning import train_model; from unlearning import unlearn; from data import load_dataset, get_forget_retain_split; import torch; args = get_args_with_custom(['--dataset', 'Cora', '--gnn_model', 'GCN', '--unlearn_method', 'full_method', '--unlearn_rate', '0.1', '--num_curricula', '4']); model, data, is_kg = train_model(args); forget_mask, retain_mask = get_forget_retain_split(data, 0.1, is_kg); unlearn(args, model, data, is_kg, forget_mask, retain_mask)"
```

**Evaluation:**
```bash
python evaluation_1.py       # Baseline comparison
python evaluation_2.py       # Hyperparameter sensitivity
```

---

## Methodology

- **Supported Datasets:** Cora, CiteSeer, PubMed, FB15k237, WN18RR
- **Supported Models:** GCN, GAT, GraphSAGE (homogeneous); RGCN, CompGCN (knowledge graphs)
- **Baseline Methods:** Full retrain, Gradient Ascent
- **Proposed Methods:**  
  - *Curriculum Unlearning*: Unlearn forget set in steps based on graph complexity (degree, betweenness, PageRank, clustering, eigenvector, etc.).
  - *NPO*: Modified loss to balance forgetting and utility preservation.

Formulas:
- Homogeneous:  
  `L_NPO = λ·L_forget + (1-λ)·L_retain`,  
  `L_forget = -β · KL(π_current || π_reference)`
- Knowledge graph:  
  `L_forget = -E[log σ(β(score_ref - score_current)/τ)]`

---

## Experiments

- **Metrics:**
  - *Forget Effect (FE):* 1 - accuracy (homogeneous) / 1 - MRR (knowledge graph)
  - *Model Utility (MU):* Accuracy/MRR on retain/test set
- **Baselines:** Retrain, Gradient Ascent, Curriculum, NPO, Full
- **Sensitivity:** num_curricula, complexity_metric, curriculum_mode, unlearn_rate, npo_beta, npo_lambda

Results are output to the `results/` directory as CSV/JSON/PNG.

---

## Usage Guide

**Main arguments:**
- Dataset/model: `--dataset`, `--gnn_model`
- Learning: `--learning_epochs`, `--learning_lr`, `--hidden_dim`, `--num_layers`, `--dropout`, `--weight_decay`
- Unlearning: `--unlearn_method`, `--unlearn_rate`, `--unlearn_epochs`, `--unlearn_lr`
- Curriculum: `--num_curricula`, `--complexity_metric`, `--curriculum_mode`, `--overlap_ratio`
- NPO: `--npo_beta`, `--npo_temperature`, `--npo_lambda`
- Paths: `--data_dir`, `--model_dir`, `--result_dir`

**Sample workflow:**
```python
from my_parser import get_args_with_custom
from learning import train_model
from unlearning import unlearn
from data import load_dataset, get_forget_retain_split
from evaluation_1 import calculate_forget_effect, calculate_model_utility
import torch

args = get_args_with_custom([
    '--dataset', 'Cora',
    '--gnn_model', 'GCN',
    '--learning_epochs', '200',
    '--unlearn_method', 'full_method',
    '--unlearn_rate', '0.1',
    '--num_curricula', '4',
    '--complexity_metric', 'degree',
    '--npo_beta', '0.1',
    '--npo_lambda', '0.5'
])
model, data, is_kg = train_model(args)
forget_mask, retain_mask = get_forget_retain_split(data, args.unlearn_rate, is_kg)
unlearned_model = unlearn(args, model, data, is_kg, forget_mask, retain_mask)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fe = calculate_forget_effect(unlearned_model, data, is_kg, forget_mask, device)
mu = calculate_model_utility(unlearned_model, data, is_kg, retain_mask, data.test_mask if not is_kg else retain_mask, device)
print(f"Forget Effect: {fe:.4f}")
print(f"Model Utility: {mu:.4f}")
```

For full runs with all settings:
```bash
python evaluation_1.py
python evaluation_2.py
python evaluation_1.py --load_pretrained True
python evaluation_2.py --load_pretrained True
```

---

## Results

- **Full method (Curriculum + NPO) achieves the best trade-off between high forgetting and high model utility.**
- Method is robust to high unlearn rates (`R=50%`), especially with 4–8 curricula and degree/PageRank as complexity metric.

Results files (CSV/JSON/PNG) are stored in `results/` after running experiments.

---

## Implementation Notes

- Complexity metrics: Degree (default), betweenness, PageRank, clustering, eigenvector.
- Use `ComplexityCalculator` and `CurriculumDesigner` in `unlearning.py` to define your curricula sequence.
- Custom unlearning methods or complexity metrics can be added by extending these classes.

---

## Troubleshooting

- **CUDA OOM:** Reduce batch size or use `--device cpu`.
- **Dataset download fails:** Use synthetic dataset (auto fallback).
- **Convergence issues:** For eigenvector, code auto-switches to degree.
- **Long runtime:** Use pretrained models or reduce epochs.

---

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper_2025,
  title={Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

---

## License

MIT License.

---

## Contact

Please open an issue on GitHub or contact [your email].

---

## Acknowledgments

- PyTorch Geometric, NetworkX, and the open research community for datasets and tools.

---

*For advanced customization, extend `ComplexityCalculator` or add new methods to `unlearning.py`. Batch experiment scripts are provided in the appendix.*

