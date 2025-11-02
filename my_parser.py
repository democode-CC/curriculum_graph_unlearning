"""
Argument parser for Graph Unlearning experiments
Handles all hyperparameters for learning, unlearning, and evaluation
"""

import argparse


def create_parser():
    """Create argument parser with all hyperparameters for the experiments"""
    parser = argparse.ArgumentParser(
        description='Graph Unlearning with Curriculum Learning and NPO'
    )
    
    # ==================== Dataset & Model Configuration ====================
    parser.add_argument('--dataset', type=str, default='Cora',
                      choices=['Cora', 'CiteSeer', 'PubMed', 'FB15k237', 'WN18RR'],
                      help='Dataset name (3 homogeneous + 2 knowledge graphs)')
    
    parser.add_argument('--gnn_model', type=str, default='GCN',
                      choices=['GCN', 'GAT', 'GraphSAGE', 'RGCN', 'CompGCN'],
                      help='GNN model architecture (GCN/GAT/GraphSAGE for homogeneous, RGCN/CompGCN for KG)')
    
    # ==================== Learning Phase ====================
    parser.add_argument('--learning_epochs', type=int, default=200,
                      help='Number of epochs for initial training')
    
    parser.add_argument('--learning_lr', type=float, default=0.01,
                      help='Learning rate for initial training')
    
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension for GNN layers')
    
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of GNN layers')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate')
    
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                      help='Weight decay (L2 regularization)')
    
    # ==================== Unlearning Configuration ====================
    parser.add_argument('--unlearn_method', type=str, default='full_method',
                      choices=['retrain', 'gradient_ascent', 'curriculum_ga', 'npo_ga', 'full_method'],
                      help='Unlearning method: retrain, gradient_ascent (GA), curriculum_ga (GA+Step1), npo_ga (GA+Step2), full_method (GA+Step1+Step2)')
    
    parser.add_argument('--unlearn_rate', type=float, default=0.1,
                      choices=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                      help='Percentage of training data to unlearn (1%, 2%, 5%, 10%, 20%, 50%)')
    
    parser.add_argument('--unlearn_epochs', type=int, default=50,
                      help='Number of epochs for unlearning')
    
    parser.add_argument('--unlearn_lr', type=float, default=0.001,
                      help='Learning rate for unlearning (gradient ascent)')
    
    # ==================== Curriculum Unlearning (Step 1) ====================
    parser.add_argument('--num_curricula', type=int, default=4,
                      choices=[1, 2, 4, 8],
                      help='Number of curriculum levels (C=1,2,4,8)')
    
    parser.add_argument('--complexity_metric', type=str, default='degree',
                      choices=['degree', 'betweenness', 'pagerank', 'clustering', 'eigenvector'],
                      help='Graph complexity metric for curriculum design')
    
    parser.add_argument('--curriculum_mode', type=str, default='non_overlapping',
                      choices=['overlapping', 'non_overlapping'],
                      help='Whether curricula have overlapping nodes')
    
    parser.add_argument('--overlap_ratio', type=float, default=0.2,
                      help='Overlap ratio between consecutive curricula (only for overlapping mode)')
    
    # ==================== NPO Configuration (Step 2) ====================
    parser.add_argument('--npo_beta', type=float, default=0.1,
                      help='Beta parameter for NPO loss (controls preference strength)')
    
    parser.add_argument('--npo_temperature', type=float, default=1.0,
                      help='Temperature parameter for NPO')
    
    parser.add_argument('--npo_lambda', type=float, default=0.5,
                      help='Balance between unlearning and utility preservation')
    
    # ==================== Evaluation Configuration ====================
    parser.add_argument('--eval_type', type=str, default='baseline_comparison',
                      choices=['baseline_comparison', 'hyperparameter_sensitivity'],
                      help='Type of evaluation to run')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use for training')
    
    # ==================== File Paths ====================
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory for datasets')
    
    parser.add_argument('--model_dir', type=str, default='./stored_model',
                      help='Directory to save/load models')
    
    parser.add_argument('--result_dir', type=str, default='./results',
                      help='Directory to save results')
    
    # ==================== Experiment Control ====================
    parser.add_argument('--save_model', action='store_true', default=True,
                      help='Save trained models')
    
    parser.add_argument('--load_pretrained', action='store_true', default=False,
                      help='Load pretrained model if available')
    
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='Print detailed training information')
    
    return parser


def get_args():
    """Parse and return arguments"""
    parser = create_parser()
    args = parser.parse_args()
    return args


def get_args_with_custom(custom_args=None):
    """Parse arguments with custom string (useful for programmatic calls)"""
    parser = create_parser()
    if custom_args:
        args = parser.parse_args(custom_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Test parser
    args = get_args()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")



