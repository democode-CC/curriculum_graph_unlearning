# Setup Guide for Graph Unlearning Project

This guide provides detailed instructions for setting up the conda environment for the graph unlearning project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [CPU Setup](#cpu-setup)
- [GPU Setup](#gpu-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Conda

If you don't have conda installed, download and install Miniconda or Anaconda:

**Miniconda (Recommended - Lightweight)**
```bash
# For macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# For Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# For Windows
# Download from: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

**Anaconda (Full Distribution)**
```bash
# Download from: https://www.anaconda.com/download
```

### 2. Update Conda
```bash
conda update -n base -c defaults conda
```

---

## CPU Setup

For machines **without** NVIDIA GPU or for development/testing:

### Step 1: Create the Environment

```bash
# Navigate to project directory
cd curriculum_graph_unlearning

# Create conda environment from file
conda env create -f environment.yml
```

### Step 2: Activate the Environment

```bash
conda activate graph_unlearning
```

### Step 3: Verify Installation

```bash
# Run the test script
python test_installation.py
```

---

## GPU Setup

For machines **with** NVIDIA GPU and CUDA support:

### Step 1: Check CUDA Version

First, check your CUDA version:

```bash
nvidia-smi
```

Look for the CUDA version in the output (e.g., "CUDA Version: 11.8").

### Step 2: Modify Environment File (if needed)

If your CUDA version is different from 11.8, edit `environment_gpu.yml`:

```yaml
# For CUDA 11.7
- pytorch-cuda=11.7

# For CUDA 12.1
- pytorch-cuda=12.1
```

### Step 3: Create the GPU Environment

```bash
# Navigate to project directory
cd curriculum_graph_unlearning

# Create conda environment from GPU file
conda env create -f environment_gpu.yml
```

### Step 4: Activate the Environment

```bash
conda activate graph_unlearning_gpu
```

### Step 5: Verify GPU Installation

```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run full test
python test_installation.py
```

---

## Alternative: Manual Installation

If the environment files don't work, you can install manually:

### For CPU:

```bash
# Create environment
conda create -n graph_unlearning python=3.9 -y
conda activate graph_unlearning

# Install PyTorch (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install PyTorch Geometric
conda install pyg -c pyg -y

# Install other dependencies
conda install networkx numpy pandas matplotlib seaborn tqdm scikit-learn scipy jupyter -c conda-forge -y
```

### For GPU:

```bash
# Create environment
conda create -n graph_unlearning_gpu python=3.9 -y
conda activate graph_unlearning_gpu

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install PyTorch Geometric
conda install pyg -c pyg -y

# Install other dependencies
conda install networkx numpy pandas matplotlib seaborn tqdm scikit-learn scipy jupyter -c conda-forge -y
```

---

## Verification

After installation, verify everything works:

### 1. Run Test Script

```bash
python test_installation.py
```

Expected output:
```
============================================================
Installation and Functionality Test
============================================================
Testing imports...
✓ PyTorch 2.0.0
✓ PyTorch Geometric 2.3.0
✓ NetworkX 3.1
✓ NumPy 1.24.3
✓ Pandas 2.0.2
✓ Matplotlib 3.7.1

All imports successful!
...
✓ All tests passed! Installation is complete.
============================================================
```

### 2. Test Dataset Loading

```bash
python -c "from data import load_dataset; data, is_kg = load_dataset('Cora', './data'); print(f'✓ Loaded Cora: {data.num_nodes} nodes, {data.num_edges} edges')"
```

### 3. Quick Training Test

```bash
# Train for just 10 epochs to verify everything works
python learning.py --dataset Cora --gnn_model GCN --learning_epochs 10
```

---

## Environment Management

### Useful Commands

```bash
# List all conda environments
conda env list

# Activate environment
conda activate graph_unlearning

# Deactivate environment
conda deactivate

# Update environment from file
conda env update -f environment.yml --prune

# Export your current environment
conda env export > my_environment.yml

# Remove environment
conda env remove -n graph_unlearning

# Clone environment
conda create --name graph_unlearning_backup --clone graph_unlearning
```

### Adding Packages

If you need additional packages:

```bash
# Using conda
conda install package_name -c conda-forge

# Using pip (when conda doesn't have it)
pip install package_name
```

---

## Troubleshooting

### Issue 1: PyTorch Geometric Installation Fails

**Solution**: Install from source or use pip:

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

For GPU (replace `cpu` with your CUDA version, e.g., `cu118`):
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Issue 2: CUDA Version Mismatch

**Error**: `RuntimeError: CUDA error: no kernel image is available for execution`

**Solution**: 
1. Check your CUDA version: `nvidia-smi`
2. Install matching PyTorch version:
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

### Issue 3: "Solving environment" Takes Forever

**Solution**: Use mamba (faster conda alternative):

```bash
# Install mamba
conda install mamba -c conda-forge

# Use mamba instead of conda
mamba env create -f environment.yml
```

### Issue 4: NetworkX Functions Fail

**Error**: `AttributeError: module 'networkx' has no attribute 'betweenness_centrality'`

**Solution**: Update NetworkX:
```bash
conda update networkx
# or
pip install --upgrade networkx
```

### Issue 5: Import Errors for Local Modules

**Error**: `ModuleNotFoundError: No module named 'gnn_model'`

**Solution**: Make sure you're in the project directory:
```bash
cd /path/to/curriculum_graph_unlearning
python your_script.py
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/curriculum_graph_unlearning"
```

---

## Platform-Specific Notes

### macOS (M1/M2 Apple Silicon)

For Apple Silicon Macs, use the ARM64 version:

```bash
# Create environment
conda create -n graph_unlearning python=3.9 -y
conda activate graph_unlearning

# Install PyTorch for Apple Silicon
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y

# Install other packages
pip install torch-geometric
conda install networkx numpy pandas matplotlib seaborn tqdm scikit-learn scipy -c conda-forge -y
```

**Note**: GPU acceleration on M1/M2 uses MPS (Metal Performance Shaders):
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

### Windows

For Windows, some packages may need specific installation:

```bash
# Install Visual C++ Build Tools first (if needed)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install environment
conda env create -f environment.yml
```

### Linux

For Linux servers without internet on compute nodes:

```bash
# On login node with internet
conda env create -f environment.yml
conda activate graph_unlearning

# Export to a tarball
conda pack -n graph_unlearning -o graph_unlearning.tar.gz

# On compute node without internet
mkdir -p ~/envs/graph_unlearning
tar -xzf graph_unlearning.tar.gz -C ~/envs/graph_unlearning
source ~/envs/graph_unlearning/bin/activate
```

---

## Docker Alternative

If you prefer Docker:

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install dependencies
RUN pip install torch-geometric networkx pandas matplotlib seaborn tqdm scikit-learn

# Copy project files
COPY . /workspace/

# Run tests
RUN python test_installation.py
```

Build and run:
```bash
docker build -t graph_unlearning .
docker run -it --gpus all graph_unlearning bash
```

---

## Jupyter Notebook Setup

To use Jupyter notebooks with this environment:

```bash
# Activate environment
conda activate graph_unlearning

# Install ipykernel
conda install ipykernel -y

# Add environment to Jupyter
python -m ipykernel install --user --name=graph_unlearning --display-name="Graph Unlearning"

# Start Jupyter
jupyter notebook
```

Then select "Graph Unlearning" as your kernel in Jupyter.

---

## Next Steps

After successful setup:

1. ✓ Environment is ready
2. ✓ Run quick test: `python test_installation.py`
3. → Read the main [README.md](README.md) for usage instructions
4. → Try training: `python learning.py --dataset Cora --gnn_model GCN`
5. → Run evaluations: `python evaluation_1.py`

---

## Getting Help

If you encounter issues not covered here:

1. Check PyTorch installation: https://pytorch.org/get-started/locally/
2. Check PyTorch Geometric installation: https://pytorch-geometric.readthedocs.io/
3. Open an issue on the project repository
4. Check conda documentation: https://docs.conda.io/

---

**Last Updated**: 2025


