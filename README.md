# CUNO: Curriculum and Preference Optimization for Stable Graph Unlearning under Mass Deletion

This repository provides a minimal, self-contained implementation of **CUNO**, the graph unlearning method proposed in:

> **CUNO: Curriculum and Preference Optimization for Stable Graph Unlearning under Mass Deletion**  



---

## Repository structure

```
cuno_release/
├── run.py          # End-to-end entry point (train → unlearn → evaluate)
├── cuno.py         # CUNO unlearning implementation
├── curriculum.py   # ComplexityCalculator + CurriculumDesigner
├── models.py       # GCN, GAT, GraphSAGE
├── dataset.py      # Data loading and forget/retain split
├── evaluate.py     # Forget Effect (FE) and Model Utility (MU)
└── requirements.txt
```

---

## Environment setup

We recommend Python 3.9 and CUDA 11.8. The steps below create an isolated conda environment.

```bash
conda create -n cuno python=3.9 -y
conda activate cuno

# Install PyTorch (adjust the CUDA version to match your system)
pip install torch==1.13.1+cu118 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyG and its sparse dependencies
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-1.13.1+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

> **CPU-only installation** (no GPU):
> ```bash
> pip install torch torchvision torchaudio
> pip install torch-scatter torch-sparse torch-geometric
> pip install -r requirements.txt
> ```

---

## Quick start

Run CUNO on Cora with all default hyperparameters:

```bash
python run.py
```

Expected output (Cora + GCN + 10% forget rate, seed 42):

```
Dataset: Cora | nodes: 2708 | edges: 10556 | features: 1433 | classes: 7
Trained model  |  Test Acc = 0.804
Forget set: 14 nodes | Retain set: 126 nodes
Before unlearning  |  FE = 0.000  |  MU = 0.804

Running CUNO unlearning ...
[CUNO] 7 curriculum stages | metric=retain_coupling | mode=overlapping
Stage 1/7: 100%|████| 7/7 [...]
...

============================================================
Results
============================================================
  Forget Effect (FE) : 0.500   (higher = better forgetting)
  Model Utility (MU) : 0.539   (higher = better utility)
  Delta MU           : +0.265
============================================================
```



---

## Key options

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `Cora` | `Cora`, `CiteSeer`, or `PubMed` |
| `--gnn_model` | `GCN` | `GCN`, `GAT`, or `GraphSAGE` |
| `--unlearn_rate` | `0.1` | Fraction of training nodes to forget |
| `--num_curricula` | `8` | Number of curriculum stages K |
| `--complexity_metric` | `retain_coupling` | Node complexity metric (see below) |
| `--curriculum_mode` | `overlapping` | `overlapping` or `non_overlapping` |
| `--curriculum_order` | `hard_to_easy` | Stage ordering |
| `--overlap_ratio` | `0.2` | Overlap fraction ρ between adjacent stages |
| `--npo_beta` | `0.01` | Forgetting strength β in NPO loss |
| `--npo_lambda` | `0.1` | Forget/retain balance weight λ |
| `--npo_temperature` | `1.0` | Softmax temperature τ |
| `--hop_decay` | `0.5` | Hop-decay α for `multihop_retain_coverage` |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | `cuda` or `cpu` |

### Complexity metrics

| Name | Signal |
|---|---|
| `degree` | Local degree (fast) |
| `betweenness` | Global betweenness centrality (slow on large graphs) |
| `pagerank` | PageRank |
| `clustering` | Clustering coefficient |
| `eigenvector` | Eigenvector centrality |
| `prediction_confidence` | Model prediction entropy (negative) |
| `gradient_norm` | Per-node gradient norm ‖∇L(v)‖ |
| `retain_coupling` | Cosine similarity with retain-set neighbours (default) |
| `multihop_retain_coverage` | Weighted L-hop retain coverage |
| `retain_betweenness` | Betweenness centrality restricted to retain subgraph |
| `class_boundary` | Class-boundary heterophily |

---

## Example commands

```bash
# CiteSeer, GraphSAGE, 20% forget rate
python run.py --dataset CiteSeer --gnn_model GraphSAGE --unlearn_rate 0.2

# Use gradient norm metric with 4 stages
python run.py --complexity_metric gradient_norm --num_curricula 4

# Non-overlapping curriculum, easy-to-hard order
python run.py --curriculum_mode non_overlapping --curriculum_order easy_to_hard

# CPU-only run
python run.py --device cpu
```

---

## Citation

If you use this code, please cite our paper (BibTeX to be added after review).
