"""
CUNO: Curriculum Unlearning with Negative Preference Optimization.

Reference: "Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning"

The full_method function implements the complete CUNO pipeline:
  1. Build a complexity-ordered curriculum over the forget set.
  2. At each stage k, minimise:
       L^(k) = lambda * L_NPO(C_k) + (1 - lambda) * L_CE(retain)
     where L_NPO is the p_ref-weighted ratio loss defined below.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm

from curriculum import CurriculumDesigner


# ---------------------------------------------------------------------------
# NPO forget loss
# ---------------------------------------------------------------------------

def _npo_loss(p_cur, p_ref, beta):
    """
    p_ref-weighted NPO ratio loss (node classification).

    L = (2/beta) * sum_c p_ref(c) * log(1 + (p_cur(c) / p_ref(c))^beta)

    Properties:
    - Self-dampening: when p_cur(c) -> 0 (already forgotten), loss -> 0.
    - Bounded gradient: ||grad|| <= 2 * C / tau per node.

    Args:
        p_cur: [N, C] current softmax probabilities (temperature-scaled).
        p_ref: [N, C] reference softmax probabilities (detached).
        beta:  float, forgetting strength.

    Returns:
        Scalar loss.
    """
    log_ratio  = (torch.log(p_cur + 1e-8) - torch.log(p_ref + 1e-8)).clamp(-30, 30)
    ratio_beta = torch.exp(beta * log_ratio)
    per_class  = (2.0 / beta) * torch.log(1.0 + ratio_beta)          # [N, C]
    per_node   = (p_ref * per_class).sum(dim=-1)                      # [N]
    return per_node.mean()


# ---------------------------------------------------------------------------
# Main unlearning entry point
# ---------------------------------------------------------------------------

def full_method(args, model, data, forget_mask, retain_mask):
    """
    Run CUNO on a trained GNN model.

    Algorithm
    ---------
    1. Freeze a copy of the trained model as the reference (theta_ref).
    2. Build a K-stage curriculum over the forget set using a complexity metric.
    3. For each stage k = 1 ... K:
         a. Compute NPO forget loss on C_k using the full graph forward pass.
         b. Compute cross-entropy retain loss on the retain set.
         c. Update theta with: L = lambda * L_NPO + (1-lambda) * L_retain.

    Note: The GNN forward pass always uses the full graph (data.edge_index),
    preserving all message-passing paths between forget and retain nodes.

    Args:
        args:        Namespace with hyperparameters (see run.py).
        model:       Trained GNN (will be modified in-place).
        data:        PyG Data object (full graph, on CPU or GPU).
        forget_mask: Boolean tensor [N], True = forget node.
        retain_mask: Boolean tensor [N], True = retain node.

    Returns:
        model: Unlearned model.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    data   = data.to(device)
    forget_mask = forget_mask.to(device)
    retain_mask = retain_mask.to(device)

    # Frozen reference
    ref_model = deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Build curriculum
    designer = CurriculumDesigner(
        forget_mask       = forget_mask.cpu(),
        data              = data.cpu(),
        complexity_metric = args.complexity_metric,
        num_curricula     = args.num_curricula,
        mode              = args.curriculum_mode,
        overlap_ratio     = args.overlap_ratio,
        curriculum_order  = args.curriculum_order,
        model             = ref_model,
        retain_mask       = retain_mask.cpu(),
        device            = str(device),
        num_layers        = args.num_layers,
        hop_decay         = args.hop_decay,
    )
    curricula = designer.design_curricula()
    print(f"[CUNO] {len(curricula)} curriculum stages | "
          f"metric={args.complexity_metric} | mode={args.curriculum_mode}")

    beta = args.npo_beta
    tau  = args.npo_temperature
    lam  = args.npo_lambda
    epochs_per_stage = args.unlearn_epochs // len(curricula)

    for k, stage_mask in enumerate(curricula):
        stage_mask = stage_mask.to(device)
        if stage_mask.sum() == 0:
            continue

        optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
        pbar = tqdm(range(epochs_per_stage), desc=f"Stage {k+1}/{len(curricula)}")

        for _ in pbar:
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)

            # NPO forget loss on current stage nodes
            logits_cur = out[stage_mask]
            with torch.no_grad():
                logits_ref = ref_model(data.x, data.edge_index)[stage_mask]
            p_cur = F.softmax(logits_cur / tau, dim=-1)
            p_ref = F.softmax(logits_ref / tau, dim=-1)
            forget_loss = _npo_loss(p_cur, p_ref, beta)

            # Cross-entropy retain loss
            retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])

            loss = lam * forget_loss + (1 - lam) * retain_loss
            loss.backward()
            optimizer.step()

            pbar.set_postfix(forget=f"{forget_loss.item():.4f}",
                             retain=f"{retain_loss.item():.4f}")

    return model
