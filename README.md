# Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning

This repository contains the complete implementation for the research paper **"Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning"**.

## Abstract

Unlearning large portions of training data from graph neural networks (GNNs) presents unique challenges, particularly the risk of catastrophic collapse—a phenomenon characterized by a rapid and severe decline in model utility during the unlearning process. This research introduces two key techniques:

1. **Negative Preference Optimization (NPO)**: An alignment-inspired loss function designed to address rapid model utility decline during unlearning
2. **Curriculum Unlearning**: A progressive strategy that incrementally unlearns subsets of the forget set, starting with simpler components

## Table of Contents

- [Installation](#installation)
- [Git Repository Setup](#git-repository-setup)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Usage Guide](#usage-guide)
- [Results](#results)
- [Citation](#citation)

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+
- NetworkX
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm

### Quick Setup with Conda (Recommended)

**Option 1: CPU-only (for development/testing)**
```bash
cd curriculum_graph_unlearning

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate graph_unlearning

# Verify installation
python test_installation.py
```

**Option 2: GPU with CUDA support**
```bash
cd curriculum_graph_unlearning

# Create GPU environment
conda env create -f environment_gpu.yml

# Activate environment
conda activate graph_unlearning_gpu

# Verify installation
python test_installation.py
```

📖 **For detailed setup instructions, troubleshooting, and platform-specific notes, see [SETUP.md](SETUP.md)**

### Alternative: Manual Installation with pip

```bash
# Clone the repository
git clone <repository-url>
cd curriculum_graph_unlearning

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch torchvision torchaudio
pip install torch-geometric
pip install networkx numpy pandas matplotlib seaborn tqdm

# Verify installation
python test_installation.py
```

---

## Git Repository Setup

### Quick Setup with Script (推荐/Recommended)

```bash
# Run the automated setup script
./setup_git.sh
```

This script will:
- ✓ Initialize Git repository
- ✓ Configure user information
- ✓ Create initial commit
- ✓ Optionally connect to GitHub/GitLab

### Manual Setup (手动设置)

```bash
# 1. Initialize repository / 初始化仓库
git init

# 2. Add files / 添加文件
git add .

# 3. Create first commit / 创建首次提交
git commit -m "Initial commit: Graph unlearning implementation"

# 4. Connect to GitHub (optional) / 连接到 GitHub（可选）
git remote add origin https://github.com/yourusername/curriculum_graph_unlearning.git
git branch -M main
git push -u origin main
```

📖 **For detailed Git setup instructions, see [GIT_SETUP.md](GIT_SETUP.md)**

---

## Project Structure

```
curriculum_graph_unlearning/
│
├── my_parser.py              # Argument parser with all hyperparameters
├── learning.py                # Initial GNN training (learning phase)
├── unlearning.py              # All unlearning methods implementation
├── evaluation_1.py            # Baseline comparison experiments
├── evaluation_2.py            # Hyperparameter sensitivity analysis
│
├── gnn_model/                 # GNN model architectures
│   ├── __init__.py
│   ├── homogeneous_models.py # GCN, GAT, GraphSAGE
│   └── knowledge_graph_models.py  # RGCN, CompGCN
│
├── data/                      # Data loading utilities
│   ├── __init__.py
│   └── data_loader.py        # Dataset loaders for all graphs
│
├── stored_model/             # Saved trained models
├── results/                  # Experimental results (CSV, JSON, plots)
└── README.md                 # This file
```

---

## Quick Start

### 1. Train a GNN Model

```bash
# Train GCN on Cora dataset
python learning.py --dataset Cora --gnn_model GCN --learning_epochs 200

# Train RGCN on FB15k237 knowledge graph
python learning.py --dataset FB15k237 --gnn_model RGCN --learning_epochs 200
```

### 2. Run Unlearning

```bash
# Gradient Ascent baseline
python -c "from my_parser import get_args_with_custom; from learning import train_model; from unlearning import unlearn; from data import load_dataset, get_forget_retain_split; import torch; args = get_args_with_custom(['--dataset', 'Cora', '--gnn_model', 'GCN', '--unlearn_method', 'gradient_ascent', '--unlearn_rate', '0.1']); model, data, is_kg = train_model(args); forget_mask, retain_mask = get_forget_retain_split(data, 0.1, is_kg); unlearn(args, model, data, is_kg, forget_mask, retain_mask)"

# Full method (Curriculum + NPO)
python -c "from my_parser import get_args_with_custom; from learning import train_model; from unlearning import unlearn; from data import load_dataset, get_forget_retain_split; import torch; args = get_args_with_custom(['--dataset', 'Cora', '--gnn_model', 'GCN', '--unlearn_method', 'full_method', '--unlearn_rate', '0.1', '--num_curricula', '4']); model, data, is_kg = train_model(args); forget_mask, retain_mask = get_forget_retain_split(data, 0.1, is_kg); unlearn(args, model, data, is_kg, forget_mask, retain_mask)"
```

### 3. Run Evaluations

```bash
# Baseline comparison (Evaluation 1)
python evaluation_1.py

# Hyperparameter sensitivity (Evaluation 2)
python evaluation_2.py
```

---

## Methodology

### Phase 1: Learning

The learning phase trains GNN models on graph datasets. We support:

**Homogeneous Graphs:**
- **Cora** (2,708 nodes, 5,429 edges)
- **CiteSeer** (3,327 nodes, 4,732 edges)
- **PubMed** (19,717 nodes, 44,338 edges)

**Knowledge Graphs:**
- **FB15k237** (14,541 entities, 237 relations)
- **WN18RR** (40,943 entities, 11 relations)

**GNN Models:**
- For homogeneous graphs: **GCN**, **GAT**, **GraphSAGE**
- For knowledge graphs: **RGCN**, **CompGCN**

### Phase 2: Unlearning

#### Baseline Methods

**1. Retrain**
- Train a new model from scratch using only the retain set
- Gold standard but computationally expensive

**2. Gradient Ascent (GA)**
- Maximize loss on forget set to "unlearn" information
- Fast but prone to catastrophic collapse

#### Proposed Methods

**Variant 1: Curriculum Unlearning (GA + Step 1)**

Progressive unlearning using curriculum design:

1. **Complexity Metrics**: Rank nodes/edges by complexity
   - Degree centrality
   - Betweenness centrality
   - PageRank
   - Clustering coefficient
   - Eigenvector centrality

2. **Curriculum Design**:
   - **Non-overlapping**: Divide forget set into C disjoint subsets
   - **Overlapping**: Allow overlap between consecutive curricula

3. **Progressive Unlearning**: Unlearn from simple to complex

**Variant 2: NPO Gradient Ascent (GA + Step 2)**

Negative Preference Optimization modifies the loss function:

For homogeneous graphs:
```
L_NPO = λ × L_forget + (1-λ) × L_retain

L_forget = -β × KL(π_current || π_reference)
L_retain = CrossEntropy(retain_set)
```

For knowledge graphs:
```
L_forget = -E[log σ(β × (score_ref - score_current) / τ)]
L_retain = MarginRankingLoss(retain_set)
```

**Variant 3: Full Method (GA + Step 1 + Step 2)**

Combines curriculum unlearning with NPO for maximum stability and effectiveness.

---

## Experiments

### Evaluation Metrics

**1. Forget Effect (FE)**
- Measures how well the forget set has been unlearned
- For homogeneous: `FE = 1 - accuracy_on_forget_set`
- For knowledge graphs: `FE = 1 - MRR_on_forget_set`
- **Higher is better** (closer to 1 = better forgetting)

**2. Model Utility (MU)**
- Measures preserved performance on retain/test set
- For homogeneous: `MU = accuracy_on_test_set`
- For knowledge graphs: `MU = MRR_on_retain_set`
- **Higher is better** (closer to 1 = better utility preservation)

### Experiment 1: Baseline Comparison

**File**: `evaluation_1.py`

**Purpose**: Compare all methods across datasets and models

**Configuration**:
- 5 datasets × 3 models (15 combinations)
- 5 methods: Retrain, GA, Variant 1, Variant 2, Variant 3
- Default unlearn rate: 10%
- Metrics: FE and MU

**Output**:
- `results/baseline_comparison.csv`
- `results/baseline_comparison.json`
- Summary statistics and best configurations

**Expected Results**: Full method should achieve best balance of FE and MU

### Experiment 2: Hyperparameter Sensitivity

**File**: `evaluation_2.py`

**Purpose**: Analyze sensitivity to hyperparameters using control variate method

**Hyperparameters Tested**:

| Hyperparameter | Values | Description |
|----------------|--------|-------------|
| `num_curricula` (C) | 1, 2, 4, 8 | Number of curriculum levels |
| `complexity_metric` | degree, betweenness, pagerank, clustering, eigenvector | Graph complexity metric |
| `curriculum_mode` | overlapping, non_overlapping | Subset overlap mode |
| `unlearn_rate` (R) | 1%, 2%, 5%, 10%, 20%, 50% | Percentage of data to unlearn |
| `npo_beta` | 0.01, 0.05, 0.1, 0.5, 1.0 | NPO preference strength |
| `npo_lambda` | 0.1, 0.3, 0.5, 0.7, 0.9 | Balance between forget/retain |

**Output**:
- `results/hyperparameter_sensitivity.csv`
- `results/hyperparameter_sensitivity.json`
- `results/hyperparameter_sensitivity.png` (overview plot)
- `results/sensitivity_{hyperparameter}.png` (individual plots)

---

## Usage Guide

### Command-Line Arguments

#### Dataset & Model
```bash
--dataset {Cora, CiteSeer, PubMed, FB15k237, WN18RR}
--gnn_model {GCN, GAT, GraphSAGE, RGCN, CompGCN}
```

#### Learning Phase
```bash
--learning_epochs 200          # Number of training epochs
--learning_lr 0.01             # Learning rate
--hidden_dim 64                # Hidden dimension
--num_layers 2                 # Number of GNN layers
--dropout 0.5                  # Dropout rate
--weight_decay 5e-4            # L2 regularization
```

#### Unlearning Configuration
```bash
--unlearn_method {retrain, gradient_ascent, curriculum_ga, npo_ga, full_method}
--unlearn_rate 0.1             # Percentage to unlearn (0.01-0.5)
--unlearn_epochs 50            # Unlearning epochs
--unlearn_lr 0.001             # Unlearning learning rate
```

#### Curriculum Unlearning (Step 1)
```bash
--num_curricula 4              # Number of curricula (1,2,4,8)
--complexity_metric degree     # Metric for complexity
--curriculum_mode non_overlapping  # Overlap mode
--overlap_ratio 0.2            # Overlap ratio (for overlapping mode)
```

#### NPO Configuration (Step 2)
```bash
--npo_beta 0.1                 # NPO beta parameter
--npo_temperature 1.0          # Temperature
--npo_lambda 0.5               # Forget/retain balance
```

#### File Paths
```bash
--data_dir ./data              # Dataset directory
--model_dir ./stored_model     # Model save directory
--result_dir ./results         # Results directory
```

### Example Workflows

#### Complete Pipeline for Single Configuration

```python
from my_parser import get_args_with_custom
from learning import train_model
from unlearning import unlearn
from data import load_dataset, get_forget_retain_split
from evaluation_1 import calculate_forget_effect, calculate_model_utility
import torch

# Configure
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

# Train
model, data, is_kg = train_model(args)

# Split forget/retain
forget_mask, retain_mask = get_forget_retain_split(data, args.unlearn_rate, is_kg)

# Unlearn
unlearned_model = unlearn(args, model, data, is_kg, forget_mask, retain_mask)

# Evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fe = calculate_forget_effect(unlearned_model, data, is_kg, forget_mask, device)
mu = calculate_model_utility(unlearned_model, data, is_kg, retain_mask, 
                             data.test_mask if not is_kg else retain_mask, device)

print(f"Forget Effect: {fe:.4f}")
print(f"Model Utility: {mu:.4f}")
```

#### Run Full Experiments

```bash
# Run all baseline comparisons
python evaluation_1.py

# Run hyperparameter sensitivity
python evaluation_2.py

# Use pretrained models (faster)
python evaluation_1.py --load_pretrained True
python evaluation_2.py --load_pretrained True
```

---

## Results

### Expected Outcomes

**Baseline Comparison (Evaluation 1)**:

| Method | Forget Effect (FE) ↑ | Model Utility (MU) ↑ | Combined |
|--------|---------------------|---------------------|----------|
| Retrain | High | High | Best (baseline) |
| Gradient Ascent | Medium-High | Low-Medium | Poor |
| Variant 1 (Curriculum) | High | Medium-High | Good |
| Variant 2 (NPO) | High | Medium-High | Good |
| **Variant 3 (Full)** | **High** | **High** | **Best** |

**Key Findings**:
1. Gradient Ascent alone suffers from catastrophic collapse
2. Curriculum Unlearning improves stability
3. NPO improves utility preservation
4. Full method achieves best balance

**Hyperparameter Sensitivity (Evaluation 2)**:

- **Optimal num_curricula**: C=4 or C=8 provide best balance
- **Best complexity metric**: Depends on dataset (degree or PageRank typically best)
- **Curriculum mode**: Non-overlapping is faster, overlapping is more gradual
- **Unlearn rate**: Method remains stable even at R=50%
- **NPO parameters**: β=0.1, λ=0.5 work well across settings

### Output Files

After running experiments, you'll find:

```
results/
├── baseline_comparison.csv            # Main results table
├── baseline_comparison.json           # Detailed results
├── hyperparameter_sensitivity.csv     # Sensitivity analysis
├── hyperparameter_sensitivity.json    # Detailed sensitivity data
├── hyperparameter_sensitivity.png     # Overview visualization
├── sensitivity_num_curricula.png      # Individual sensitivity plots
├── sensitivity_complexity_metric.png
├── sensitivity_curriculum_mode.png
├── sensitivity_unlearn_rate.png
├── sensitivity_npo_beta.png
└── sensitivity_npo_lambda.png
```

---

## Implementation Details

### Complexity Calculation

The `ComplexityCalculator` class computes various graph-theoretic metrics:

```python
from unlearning import ComplexityCalculator

calculator = ComplexityCalculator(data, is_kg=False)
complexity_scores = calculator.calculate_complexity(nodes, metric='degree')
```

Supported metrics:
- **Degree**: Fast, works well for most cases
- **Betweenness**: Identifies bridge nodes
- **PageRank**: Considers global importance
- **Clustering**: Measures local density
- **Eigenvector**: Network influence

### Curriculum Designer

The `CurriculumDesigner` class creates curricula from simple to complex:

```python
from unlearning import CurriculumDesigner

designer = CurriculumDesigner(
    forget_mask=forget_mask,
    data=data,
    is_kg=False,
    complexity_metric='degree',
    num_curricula=4,
    mode='non_overlapping'
)

curricula = designer.design_curricula()
# Returns list of masks, ordered from simple to complex
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size for KG models
# Or use CPU
python evaluation_1.py --device cpu
```

**2. Dataset Download Fails**
```bash
# Manually download datasets or use synthetic version
# The code automatically creates synthetic datasets if download fails
```

**3. NetworkX Convergence Issues**
```bash
# For eigenvector centrality, the code automatically falls back to degree
# if convergence fails
```

**4. Long Runtime**
```bash
# Use pretrained models
python evaluation_1.py --load_pretrained True --learning_epochs 100 --unlearn_epochs 30

# Or reduce number of epochs for testing
```

---

## Citation

If you use this code in your research, please cite:

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

This project is licensed under the MIT License.

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

## Acknowledgments

- PyTorch Geometric team for graph neural network implementations
- NetworkX for graph analysis tools
- Research community for datasets (Cora, CiteSeer, PubMed, FB15k237, WN18RR)

---

## Appendix: Advanced Usage

### Custom Complexity Metrics

You can add custom complexity metrics by extending `ComplexityCalculator`:

```python
class CustomComplexityCalculator(ComplexityCalculator):
    def _calculate_custom_metric(self, nodes):
        # Your custom complexity calculation
        scores = {}
        for node in nodes:
            scores[node] = your_computation(node)
        return scores
```

### Custom Unlearning Methods

Add new unlearning methods in `unlearning.py`:

```python
def custom_unlearning_method(args, model, data, is_kg, forget_mask, retain_mask):
    # Your custom unlearning logic
    return unlearned_model
```

Then add it to the dispatch in the `unlearn()` function.

### Batch Processing

For large-scale experiments:

```python
import subprocess

datasets = ['Cora', 'CiteSeer', 'PubMed']
models = ['GCN', 'GAT', 'GraphSAGE']

for dataset in datasets:
    for model in models:
        subprocess.run([
            'python', 'learning.py',
            '--dataset', dataset,
            '--gnn_model', model
        ])
```

---

**End of Documentation**

