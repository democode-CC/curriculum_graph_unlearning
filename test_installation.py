"""
Test script to verify installation and basic functionality
Run this after installing dependencies to ensure everything works
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Geometric import failed: {e}")
        return False
    
    try:
        import networkx
        print(f"✓ NetworkX {networkx.__version__}")
    except ImportError as e:
        print(f"✗ NetworkX import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_modules():
    """Test if local modules can be imported"""
    print("\nTesting local modules...")
    
    try:
        from my_parser import get_args_with_custom
        print("✓ my_parser")
    except ImportError as e:
        print(f"✗ my_parser import failed: {e}")
        return False
    
    try:
        from gnn_model import GCN, GAT, GraphSAGE, RGCN, CompGCN
        print("✓ gnn_model")
    except ImportError as e:
        print(f"✗ gnn_model import failed: {e}")
        return False
    
    try:
        from data import load_dataset, split_data
        print("✓ data")
    except ImportError as e:
        print(f"✗ data import failed: {e}")
        return False
    
    try:
        import learning
        print("✓ learning")
    except ImportError as e:
        print(f"✗ learning import failed: {e}")
        return False
    
    try:
        import unlearning
        print("✓ unlearning")
    except ImportError as e:
        print(f"✗ unlearning import failed: {e}")
        return False
    
    print("\nAll local modules imported successfully!")
    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from my_parser import get_args_with_custom
        args = get_args_with_custom([
            '--dataset', 'Cora',
            '--gnn_model', 'GCN',
            '--learning_epochs', '10'
        ])
        print(f"✓ Argument parsing works")
        print(f"  - Dataset: {args.dataset}")
        print(f"  - Model: {args.gnn_model}")
        print(f"  - Epochs: {args.learning_epochs}")
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False
    
    try:
        from gnn_model import GCN
        import torch
        
        model = GCN(input_dim=16, hidden_dim=32, output_dim=7, num_layers=2)
        print(f"✓ Model creation works")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    print("\nBasic functionality tests passed!")
    return True


def test_dataset_loading():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    try:
        from data import load_dataset
        data, is_kg = load_dataset('Cora', './data')
        print(f"✓ Dataset loading works")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Edges: {data.num_edges}")
        print(f"  - Features: {data.num_features}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("Installation and Functionality Test")
    print("="*60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
        print("\n⚠ Some package imports failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
    
    if not test_modules():
        all_passed = False
        print("\n⚠ Some local module imports failed. Check file structure.")
    
    if not test_basic_functionality():
        all_passed = False
        print("\n⚠ Basic functionality tests failed.")
    
    if not test_dataset_loading():
        all_passed = False
        print("\n⚠ Dataset loading test failed. This may be due to network issues.")
        print("Note: Datasets will be downloaded on first use.")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Installation is complete.")
        print("\nYou can now run:")
        print("  - python learning.py --help")
        print("  - python evaluation_1.py")
        print("  - python evaluation_2.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


