"""
Dataset loading and forget/retain split utilities.
Supports Cora, CiteSeer, and PubMed (Planetoid benchmarks).
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected


def load_dataset(dataset_name: str, data_dir: str = './data'):
    """
    Load a Planetoid dataset (Cora, CiteSeer, PubMed).

    Returns:
        data: PyG Data object with train/val/test masks.
    """
    if dataset_name not in ('Cora', 'CiteSeer', 'PubMed'):
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from Cora, CiteSeer, PubMed.")

    os.makedirs(data_dir, exist_ok=True)
    dataset = Planetoid(root=os.path.join(data_dir, dataset_name), name=dataset_name)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    print(f"Dataset: {dataset_name} | nodes: {data.num_nodes} | edges: {data.num_edges} "
          f"| features: {data.num_features} | classes: {dataset.num_classes}")
    return data


def get_forget_retain_split(data, forget_ratio: float, seed: int = 42):
    """
    Randomly split the training nodes into a forget set and a retain set.

    Args:
        data:         PyG Data object with train_mask.
        forget_ratio: Fraction of training nodes to forget (e.g. 0.1 = 10%).
        seed:         Random seed for reproducibility.

    Returns:
        forget_mask: Boolean tensor [num_nodes], True for forget nodes.
        retain_mask: Boolean tensor [num_nodes], True for retain nodes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    n_train = len(train_indices)
    perm = torch.randperm(n_train)
    forget_size = max(1, int(forget_ratio * n_train))

    forget_idx = train_indices[perm[:forget_size]]
    retain_idx = train_indices[perm[forget_size:]]

    forget_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    retain_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    forget_mask[forget_idx] = True
    retain_mask[retain_idx] = True

    print(f"Forget set: {forget_mask.sum().item()} nodes | "
          f"Retain set: {retain_mask.sum().item()} nodes")
    return forget_mask, retain_mask
