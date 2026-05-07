"""
CUNO — end-to-end reproducibility script.

Usage
-----
# Train + unlearn on Cora with all default hyperparameters:
    python run.py

# Choose dataset and model:
    python run.py --dataset CiteSeer --gnn_model GAT

# Adjust forget ratio:
    python run.py --unlearn_rate 0.2

Run `python run.py --help` to see all options.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset  import load_dataset, get_forget_retain_split
from models   import create_model
from cuno     import full_method
from evaluate import compute_fe, compute_mu


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="CUNO: Curriculum Graph Unlearning with NPO")

    # Dataset & model
    p.add_argument('--dataset',   type=str,   default='Cora',
                   choices=['Cora', 'CiteSeer', 'PubMed'])
    p.add_argument('--gnn_model', type=str,   default='GCN',
                   choices=['GCN', 'GAT', 'GraphSAGE'])

    # Training
    p.add_argument('--learning_epochs', type=int,   default=200)
    p.add_argument('--learning_lr',     type=float, default=0.01)
    p.add_argument('--hidden_dim',      type=int,   default=64)
    p.add_argument('--num_layers',      type=int,   default=2)
    p.add_argument('--dropout',         type=float, default=0.5)
    p.add_argument('--weight_decay',    type=float, default=5e-4)

    # Unlearning
    p.add_argument('--unlearn_rate',   type=float, default=0.1,
                   help='Fraction of training nodes to forget (default: 0.1)')
    p.add_argument('--unlearn_epochs', type=int,   default=50)
    p.add_argument('--unlearn_lr',     type=float, default=0.1)

    # Curriculum (Step 1 of CUNO)
    p.add_argument('--num_curricula',     type=int,   default=8)
    p.add_argument('--complexity_metric', type=str,   default='retain_coupling',
                   choices=['degree', 'betweenness', 'pagerank', 'clustering',
                             'eigenvector', 'prediction_confidence', 'gradient_norm',
                             'retain_coupling', 'multihop_retain_coverage',
                             'retain_betweenness', 'class_boundary'])
    p.add_argument('--curriculum_mode',  type=str,   default='overlapping',
                   choices=['overlapping', 'non_overlapping'])
    p.add_argument('--curriculum_order', type=str,   default='hard_to_easy',
                   choices=['hard_to_easy', 'easy_to_hard'])
    p.add_argument('--overlap_ratio',    type=float, default=0.2)
    p.add_argument('--hop_decay',        type=float, default=0.5,
                   help='Hop-decay factor alpha for multihop_retain_coverage (default: 0.5)')

    # NPO (Step 2 of CUNO)
    p.add_argument('--npo_beta',        type=float, default=0.01,
                   help='Forgetting strength in NPO loss')
    p.add_argument('--npo_temperature', type=float, default=1.0,
                   help='Softmax temperature for NPO')
    p.add_argument('--npo_lambda',      type=float, default=0.1,
                   help='Forget/retain balance weight (higher = more forgetting)')

    # Misc
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--device',     type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--data_dir',   type=str, default='./data')
    p.add_argument('--model_dir',  type=str, default='./stored_model')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args, model, data, device):
    data = data.to(device)
    opt  = optim.Adam(model.parameters(), lr=args.learning_lr, weight_decay=args.weight_decay)
    best_val, best_state = 0.0, None

    pbar = tqdm(range(args.learning_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out  = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                val_acc  = (pred[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            if val_acc > best_val:
                best_val   = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pbar.set_postfix(loss=f"{loss.item():.4f}", val=f"{val_acc:.4f}", test=f"{test_acc:.4f}")

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    print("=" * 60)
    print(f"  CUNO  |  dataset={args.dataset}  |  model={args.gnn_model}")
    print(f"  unlearn_rate={args.unlearn_rate}  |  seed={args.seed}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    data = load_dataset(args.dataset, args.data_dir)

    # ── 2. Train (or load cached) ──────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    ckpt_path = os.path.join(args.model_dir,
                             f"{args.dataset}_{args.gnn_model}_trained.pt")

    model = create_model(args, data).to(device)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded pre-trained model from {ckpt_path}")
    else:
        print("No cached model found — training from scratch.")
        model = train(args, model, data, device)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved to {ckpt_path}")

    # ── 3. Evaluate trained model ─────────────────────────────────────────
    data_dev = data.to(device)
    mu_before = compute_mu(model, data_dev, data.test_mask, device)
    print(f"\nTrained model  |  Test Acc = {mu_before:.4f}")

    # ── 4. Forget/retain split ─────────────────────────────────────────────
    forget_mask, retain_mask = get_forget_retain_split(data, args.unlearn_rate, args.seed)

    fe_before = compute_fe(model, data_dev, forget_mask, device)
    print(f"Before unlearning  |  FE = {fe_before:.4f}  |  MU = {mu_before:.4f}")

    # ── 5. CUNO unlearning ────────────────────────────────────────────────
    print("\nRunning CUNO unlearning ...")
    model = full_method(args, model, data, forget_mask, retain_mask)

    # ── 6. Evaluate unlearned model ───────────────────────────────────────
    fe_after = compute_fe(model, data_dev, forget_mask, device)
    mu_after = compute_mu(model, data_dev, data.test_mask, device)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Forget Effect (FE) : {fe_after:.4f}  (higher = better forgetting)")
    print(f"  Model Utility (MU) : {mu_after:.4f}  (higher = better utility)")
    print(f"  Delta MU           : {mu_before - mu_after:+.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
