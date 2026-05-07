"""
Evaluation metrics for graph unlearning.

Forget Effect (FE): 1 - accuracy on the forget set.
  Higher FE means the model has more thoroughly forgotten the deleted nodes.

Model Utility (MU): accuracy on the test set (original full graph).
  Higher MU means the model preserves more of its original predictive power.
"""

import torch


@torch.no_grad()
def compute_fe(model, data, forget_mask, device):
    """
    Forget Effect = 1 - Acc(forget_set).

    The model is evaluated on the full graph (all edges intact) but accuracy
    is measured only on the forget nodes.

    Returns:
        fe: float in [0, 1]. Higher = better forgetting.
    """
    model.eval()
    data  = data.to(device)
    out   = model(data.x, data.edge_index)
    pred  = out.argmax(dim=1)
    mask  = forget_mask.to(device)
    if mask.sum() == 0:
        return 1.0
    acc = (pred[mask] == data.y[mask]).float().mean().item()
    return 1.0 - acc


@torch.no_grad()
def compute_mu(model, data, test_mask, device):
    """
    Model Utility = Acc(test_set) on the original full graph.

    Returns:
        mu: float in [0, 1]. Higher = better utility.
    """
    model.eval()
    data = data.to(device)
    out  = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    mask = test_mask.to(device)
    if mask.sum() == 0:
        return 0.0
    return (pred[mask] == data.y[mask]).float().mean().item()
