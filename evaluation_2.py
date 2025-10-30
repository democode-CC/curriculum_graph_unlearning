"""
Evaluation 2: Hyperparameter Sensitivity Analysis
Test how the full method performs with different hyperparameters
Using control variate method - vary one hyperparameter at a time
"""

import os
import torch
import numpy as np
import pandas as pd
import json
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

from my_parser import get_args_with_custom
from learning import train_model, create_model
from unlearning import unlearn
from data import load_dataset, get_forget_retain_split
from evaluation_1 import calculate_forget_effect, calculate_model_utility


def run_hyperparameter_sensitivity(args):
    """
    Run hyperparameter sensitivity analysis
    Test each hyperparameter while keeping others constant
    """
    
    # Default configuration
    default_config = {
        'num_curricula': 4,
        'complexity_metric': 'degree',
        'curriculum_mode': 'non_overlapping',
        'unlearn_rate': 0.1,
        'npo_beta': 0.1,
        'npo_lambda': 0.5,
    }
    
    # Hyperparameter ranges to test
    hp_configs = {
        'num_curricula': {
            'values': [1, 2, 4, 8],
            'label': 'Number of Curricula (C)'
        },
        'complexity_metric': {
            'values': ['degree', 'betweenness', 'pagerank', 'clustering', 'eigenvector'],
            'label': 'Complexity Metric'
        },
        'curriculum_mode': {
            'values': ['non_overlapping', 'overlapping'],
            'label': 'Curriculum Mode'
        },
        'unlearn_rate': {
            'values': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            'label': 'Unlearning Rate (R)'
        },
        'npo_beta': {
            'values': [0.01, 0.05, 0.1, 0.5, 1.0],
            'label': 'NPO Beta'
        },
        'npo_lambda': {
            'values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'label': 'NPO Lambda'
        }
    }
    
    # Select a representative dataset and model for sensitivity analysis
    test_configs = [
        {'dataset': 'Cora', 'model': 'GCN'},
        {'dataset': 'FB15k237', 'model': 'RGCN'}
    ]
    
    all_results = []
    
    for test_config in test_configs:
        dataset = test_config['dataset']
        model_name = test_config['model']
        
        print(f"\n{'#'*80}")
        print(f"# Hyperparameter Sensitivity: {dataset} + {model_name}")
        print(f"{'#'*80}")
        
        # Update args
        args.dataset = dataset
        args.gnn_model = model_name
        
        # Set device
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Load data
        data, is_kg = load_dataset(dataset, args.data_dir)
        
        # Load or train model
        model_path = os.path.join(args.model_dir, f"{dataset}_{model_name}_trained.pt")
        
        if os.path.exists(model_path) and args.load_pretrained:
            print(f"Loading pretrained model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            original_model = create_model(args, data, is_kg).to(device)
            original_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Training model from scratch...")
            original_model, data, is_kg = train_model(args, save_path=model_path)
        
        # Test each hyperparameter
        for hp_name, hp_config in hp_configs.items():
            print(f"\n{'='*60}")
            print(f"Testing Hyperparameter: {hp_config['label']}")
            print(f"{'='*60}")
            
            for hp_value in hp_config['values']:
                print(f"\n{hp_name} = {hp_value}")
                
                # Set all hyperparameters to default
                for key, val in default_config.items():
                    setattr(args, key, val)
                
                # Override the current hyperparameter
                setattr(args, hp_name, hp_value)
                
                # For unlearn_rate, we need to regenerate forget/retain split
                forget_mask, retain_mask = get_forget_retain_split(
                    data, args.unlearn_rate, is_kg, seed=args.seed
                )
                
                # Get test mask
                if is_kg:
                    test_mask = retain_mask
                else:
                    test_mask = data.test_mask
                
                try:
                    # Clone model
                    model = deepcopy(original_model)
                    
                    # Set to full method
                    args.unlearn_method = 'full_method'
                    
                    # Perform unlearning
                    model = unlearn(args, model, data, is_kg, forget_mask, retain_mask)
                    
                    # Calculate metrics
                    fe = calculate_forget_effect(model, data, is_kg, forget_mask, device)
                    mu = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, device)
                    
                    print(f"  FE: {fe:.4f}, MU: {mu:.4f}")
                    
                    result = {
                        'dataset': dataset,
                        'model': model_name,
                        'hyperparameter': hp_name,
                        'hyperparameter_label': hp_config['label'],
                        'value': hp_value,
                        'forget_effect': fe,
                        'model_utility': mu,
                        'combined_score': fe + mu
                    }
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error with {hp_name}={hp_value}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(args.result_dir, 'hyperparameter_sensitivity.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(args.result_dir, 'hyperparameter_sensitivity.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Generate plots
    plot_hyperparameter_sensitivity(results_df, args.result_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Hyperparameter Sensitivity")
    print("="*80)
    
    for hp_name in hp_configs.keys():
        hp_data = results_df[results_df['hyperparameter'] == hp_name]
        if len(hp_data) > 0:
            print(f"\n{hp_configs[hp_name]['label']}:")
            summary = hp_data.groupby('value').agg({
                'forget_effect': 'mean',
                'model_utility': 'mean',
                'combined_score': 'mean'
            }).round(4)
            print(summary)
            
            # Best value
            best = hp_data.loc[hp_data['combined_score'].idxmax()]
            print(f"  Best value: {best['value']} (Combined: {best['combined_score']:.4f})")
    
    return results_df


def plot_hyperparameter_sensitivity(results_df, result_dir):
    """Generate plots for hyperparameter sensitivity"""
    
    print("\nGenerating plots...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Get unique hyperparameters
    hyperparameters = results_df['hyperparameter'].unique()
    
    # Create subplots
    n_hps = len(hyperparameters)
    n_cols = 3
    n_rows = (n_hps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_hps > 1 else [axes]
    
    for idx, hp_name in enumerate(hyperparameters):
        ax = axes[idx]
        
        hp_data = results_df[results_df['hyperparameter'] == hp_name]
        hp_label = hp_data['hyperparameter_label'].iloc[0]
        
        # Group by value and dataset
        grouped = hp_data.groupby(['value', 'dataset']).agg({
            'forget_effect': 'mean',
            'model_utility': 'mean'
        }).reset_index()
        
        # Plot for each dataset
        for dataset in grouped['dataset'].unique():
            dataset_data = grouped[grouped['dataset'] == dataset]
            
            x = range(len(dataset_data))
            x_labels = [str(v) for v in dataset_data['value']]
            
            ax.plot(x, dataset_data['forget_effect'], marker='o', label=f'{dataset} - FE', linewidth=2)
            ax.plot(x, dataset_data['model_utility'], marker='s', label=f'{dataset} - MU', linewidth=2)
        
        ax.set_xlabel(hp_label, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Sensitivity to {hp_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(hyperparameters), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    plot_path = os.path.join(result_dir, 'hyperparameter_sensitivity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close()
    
    # Create individual plots for each hyperparameter
    for hp_name in hyperparameters:
        hp_data = results_df[results_df['hyperparameter'] == hp_name]
        hp_label = hp_data['hyperparameter_label'].iloc[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by value
        grouped = hp_data.groupby('value').agg({
            'forget_effect': ['mean', 'std'],
            'model_utility': ['mean', 'std']
        }).reset_index()
        
        x = range(len(grouped))
        x_labels = [str(v) for v in grouped['value']]
        
        # Forget Effect plot
        axes[0].errorbar(x, grouped['forget_effect']['mean'], 
                        yerr=grouped['forget_effect']['std'],
                        marker='o', linewidth=2, capsize=5, capthick=2)
        axes[0].set_xlabel(hp_label, fontsize=12)
        axes[0].set_ylabel('Forget Effect (FE)', fontsize=12)
        axes[0].set_title(f'Forget Effect vs {hp_label}', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # Model Utility plot
        axes[1].errorbar(x, grouped['model_utility']['mean'],
                        yerr=grouped['model_utility']['std'],
                        marker='s', linewidth=2, capsize=5, capthick=2, color='orange')
        axes[1].set_xlabel(hp_label, fontsize=12)
        axes[1].set_ylabel('Model Utility (MU)', fontsize=12)
        axes[1].set_title(f'Model Utility vs {hp_label}', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(result_dir, f'sensitivity_{hp_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        
        plt.close()


if __name__ == '__main__':
    # Parse arguments
    args = get_args_with_custom([
        '--eval_type', 'hyperparameter_sensitivity',
        '--learning_epochs', '200',
        '--unlearn_epochs', '50',
        '--save_model', 'True',
        '--load_pretrained', 'True'
    ])
    
    # Run evaluation
    results_df = run_hyperparameter_sensitivity(args)
    
    print("\n" + "="*80)
    print("Hyperparameter sensitivity analysis completed!")
    print("="*80)


