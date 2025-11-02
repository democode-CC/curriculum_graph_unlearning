"""
Evaluation 1: Baseline Comparison
Compare proposed method against baselines across all datasets and GNN models
Metrics: Forget Effect (FE) and Model Utility (MU)
"""

import os
import torch
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from tqdm import tqdm

from my_parser import get_args_with_custom
from learning import train_model, create_model, evaluate_homogeneous, evaluate_knowledge_graph
from unlearning import unlearn
from data import load_dataset, get_forget_retain_split


def calculate_forget_effect(model, data, is_kg, forget_mask, device):
    """
    Calculate Forget Effect (FE)
    Lower performance on forget set indicates better unlearning
    
    For homogeneous graphs: FE = 1 - accuracy on forget set
    For knowledge graphs: FE = 1 - MRR on forget set
    
    Higher FE is better (closer to 1 means better forgetting)
    """
    model.eval()
    
    with torch.no_grad():
        if is_kg:
            # For KG, evaluate on forget edges
            edge_index = data['train']['edge_index'].to(device)
            edge_type = data['train']['edge_type'].to(device)
            entity_ids = data['entity_ids'].to(device)
            
            forget_indices = forget_mask.nonzero(as_tuple=True)[0]
            if len(forget_indices) == 0:
                return 1.0
            
            # Sample for efficiency
            num_samples = min(len(forget_indices), 1000)
            sample_idx = forget_indices[torch.randperm(len(forget_indices))[:num_samples]]
            
            ranks = []
            for i in range(num_samples):
                idx = sample_idx[i]
                h = edge_index[0, idx:idx+1]
                t = edge_index[1, idx:idx+1]
                r = edge_type[idx:idx+1]
                
                # Score against all entities
                all_entities = torch.arange(data['num_entities'], device=device)
                scores = model.predict_link(entity_ids, edge_index, edge_type,
                                          h.repeat(data['num_entities']), all_entities, 
                                          r.repeat(data['num_entities']))
                
                sorted_indices = torch.argsort(scores, descending=True)
                rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
            
            mrr = np.mean([1.0 / r for r in ranks])
            fe = 1.0 - mrr  # Higher is better
            
        else:
            # For homogeneous graphs, evaluate on forget nodes
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            if forget_mask.sum() == 0:
                return 1.0
            
            correct = (pred[forget_mask] == data.y[forget_mask]).sum()
            acc = correct.float() / forget_mask.sum()
            fe = 1.0 - acc.item()  # Higher is better
    
    return fe


def calculate_model_utility(model, data, is_kg, retain_mask, test_mask, device):
    """
    Calculate Model Utility (MU)
    Performance on retain/test set
    
    For homogeneous graphs: MU = accuracy on test set
    For knowledge graphs: MU = MRR on retain set
    
    Higher MU is better
    """
    model.eval()
    
    if is_kg:
        # For KG, use retain set
        mu = evaluate_knowledge_graph_subset(model, data, device, retain_mask, num_samples=1000)
    else:
        # For homogeneous graphs, use test set
        mu = evaluate_homogeneous(model, data, device, test_mask)
    
    return mu


def evaluate_knowledge_graph_subset(model, data, device, subset_mask, num_samples=1000):
    """Evaluate knowledge graph on a subset of edges"""
    model.eval()
    
    with torch.no_grad():
        edge_index = data['train']['edge_index'].to(device)
        edge_type = data['train']['edge_type'].to(device)
        entity_ids = data['entity_ids'].to(device)
        
        subset_indices = subset_mask.nonzero(as_tuple=True)[0]
        if len(subset_indices) == 0:
            return 0.0
        
        num_samples = min(len(subset_indices), num_samples)
        sample_idx = subset_indices[torch.randperm(len(subset_indices))[:num_samples]]
        
        ranks = []
        for i in range(num_samples):
            idx = sample_idx[i]
            h = edge_index[0, idx:idx+1]
            t = edge_index[1, idx:idx+1]
            r = edge_type[idx:idx+1]
            
            all_entities = torch.arange(data['num_entities'], device=device)
            scores = model.predict_link(entity_ids, edge_index, edge_type,
                                      h.repeat(data['num_entities']), all_entities,
                                      r.repeat(data['num_entities']))
            
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        mrr = np.mean([1.0 / r for r in ranks])
    
    return mrr


def evaluate_method(method_name, args, original_model, data, is_kg, forget_mask, retain_mask, test_mask, device):
    """
    Evaluate a single unlearning method
    
    Returns:
        results: Dictionary with FE and MU scores
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*60}")
    
    # Clone model for this method
    model = deepcopy(original_model)
    
    # Set unlearning method
    args.unlearn_method = method_name
    
    # Perform unlearning
    model = unlearn(args, model, data, is_kg, forget_mask, retain_mask)
    
    # Calculate metrics
    fe = calculate_forget_effect(model, data, is_kg, forget_mask, device)
    mu = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, device)
    
    print(f"\nResults:")
    print(f"  Forget Effect (FE): {fe:.4f}")
    print(f"  Model Utility (MU): {mu:.4f}")
    
    return {
        'method': method_name,
        'forget_effect': fe,
        'model_utility': mu
    }


def run_baseline_comparison(args):
    """Run baseline comparison across all datasets and models"""
    
    # Define experiment configurations
    datasets = ['Cora', 'CiteSeer', 'PubMed', 'FB15k237', 'WN18RR']
    homogeneous_models = ['GCN', 'GAT', 'GraphSAGE']
    kg_models = ['RGCN', 'CompGCN']
    
    methods = ['retrain', 'gradient_ascent', 'curriculum_ga', 'npo_ga', 'full_method']
    method_names = {
        'retrain': 'Retrain',
        'gradient_ascent': 'Gradient Ascent',
        'curriculum_ga': 'GA + Curriculum (Variant 1)',
        'npo_ga': 'GA + NPO (Variant 2)',
        'full_method': 'Full Method (Variant 3)'
    }
    
    unlearn_rates = [0.1]  # Default 10% for main comparison
    
    all_results = []
    
    for dataset in datasets:
        # Determine if KG and select appropriate models
        is_kg = dataset in ['FB15k237', 'WN18RR']
        models = kg_models if is_kg else homogeneous_models
        
        for model_name in models:
            print(f"\n{'#'*80}")
            print(f"# Dataset: {dataset}, Model: {model_name}")
            print(f"{'#'*80}")
            
            # Update args
            args.dataset = dataset
            args.gnn_model = model_name
            
            # Set device
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            
            # Load data
            data, is_kg = load_dataset(dataset, args.data_dir)
            
            # Check if trained model exists
            model_path = os.path.join(args.model_dir, f"{dataset}_{model_name}_trained.pt")
            
            if os.path.exists(model_path) and args.load_pretrained:
                print(f"Loading pretrained model from {model_path}")
                checkpoint = torch.load(model_path, map_location=device)
                original_model = create_model(args, data, is_kg).to(device)
                original_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Train model
                print("Training model from scratch...")
                original_model, data, is_kg = train_model(args, save_path=model_path)
            
            # Split into forget and retain sets
            for unlearn_rate in unlearn_rates:
                print(f"\nUnlearn Rate: {unlearn_rate*100:.0f}%")
                
                forget_mask, retain_mask = get_forget_retain_split(data, unlearn_rate, is_kg, seed=args.seed)
                
                # Get test mask
                if is_kg:
                    test_mask = retain_mask  # For KG, we use retain set for evaluation
                else:
                    test_mask = data.test_mask
                
                # Evaluate each method
                for method in methods:
                    try:
                        result = evaluate_method(
                            method, args, original_model, data, is_kg,
                            forget_mask, retain_mask, test_mask, device
                        )
                        
                        result.update({
                            'dataset': dataset,
                            'model': model_name,
                            'unlearn_rate': unlearn_rate,
                            'is_kg': is_kg
                        })
                        
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error evaluating {method}: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(args.result_dir, 'baseline_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(args.result_dir, 'baseline_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Baseline Comparison")
    print("="*80)
    
    # Group by method and compute averages
    summary = results_df.groupby('method').agg({
        'forget_effect': ['mean', 'std'],
        'model_utility': ['mean', 'std']
    }).round(4)
    
    print("\nAverage Performance Across All Configurations:")
    print(summary)
    
    # Find best method
    best_fe = results_df.loc[results_df['forget_effect'].idxmax()]
    best_mu = results_df.loc[results_df['model_utility'].idxmax()]
    best_combined = results_df.assign(
        combined=lambda x: x['forget_effect'] + x['model_utility']
    ).loc[lambda x: x['combined'].idxmax()]
    
    print(f"\nBest Forget Effect: {method_names[best_fe['method']]} "
          f"(FE={best_fe['forget_effect']:.4f}, {best_fe['dataset']}, {best_fe['model']})")
    print(f"Best Model Utility: {method_names[best_mu['method']]} "
          f"(MU={best_mu['model_utility']:.4f}, {best_mu['dataset']}, {best_mu['model']})")
    print(f"Best Combined: {method_names[best_combined['method']]} "
          f"(FE+MU={best_combined['combined']:.4f}, {best_combined['dataset']}, {best_combined['model']})")
    
    return results_df


if __name__ == '__main__':
    # Parse arguments
    args = get_args_with_custom([
        '--eval_type', 'baseline_comparison',
        '--num_curricula', '4',
        '--complexity_metric', 'degree',
        '--curriculum_mode', 'non_overlapping',
        '--unlearn_rate', '0.1',
        '--learning_epochs', '200',
        '--unlearn_epochs', '50',
        '--npo_beta', '0.1',
        '--npo_lambda', '0.5',
        '--save_model',
        '--load_pretrained'
    ])
    
    # Run evaluation
    results_df = run_baseline_comparison(args)
    
    print("\n" + "="*80)
    print("Baseline comparison completed!")
    print("="*80)


