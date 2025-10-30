# Conda Environment Quick Start

## TL;DR - Get Started in 3 Steps

### For CPU (MacOS/Linux/Windows)
```bash
conda env create -f environment.yml
conda activate graph_unlearning
python test_installation.py
```

### For GPU (Linux/Windows with NVIDIA GPU)
```bash
conda env create -f environment_gpu.yml
conda activate graph_unlearning_gpu
python test_installation.py
```

---

## Common Commands

```bash
# Activate environment
conda activate graph_unlearning

# Deactivate environment
conda deactivate

# List environments
conda env list

# Remove environment
conda env remove -n graph_unlearning

# Update environment
conda env update -f environment.yml --prune
```

---

## What's Included?

- ✓ Python 3.9
- ✓ PyTorch 2.0 (CPU or CUDA 11.8)
- ✓ PyTorch Geometric 2.3
- ✓ NetworkX 3.1
- ✓ NumPy, Pandas, SciPy
- ✓ Matplotlib, Seaborn
- ✓ Jupyter Notebook
- ✓ All required dependencies

---

## Troubleshooting

**Environment creation is slow?**
```bash
# Use mamba (faster)
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
```

**Need different CUDA version?**

Edit `environment_gpu.yml` and change:
```yaml
- pytorch-cuda=11.8  # Change to 11.7, 12.1, etc.
```

**See full documentation**: [SETUP.md](SETUP.md)

---

## Next Steps

1. ✓ Environment ready
2. → Train a model: `python learning.py --dataset Cora --gnn_model GCN`
3. → Run experiments: `python evaluation_1.py`
4. → See README.md for full documentation


