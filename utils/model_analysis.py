"""
Utility script to analyze UNet model size and capacity for different base_filters values.
Helps choose optimal base_filters for performance vs model size tradeoff.
"""
import torch
from models.unet import UNet


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_size_mb(model):
    """Estimate model size in MB (assuming float32)."""
    total_params, _ = count_parameters(model)
    # 4 bytes per float32 parameter
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def analyze_base_filters(base_filters_list=[32, 48, 64, 96, 128, 160, 192]):
    """
    Analyze UNet models with different base_filters values.
    
    Args:
        base_filters_list: List of base_filters values to analyze
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    print("=" * 80)
    print("UNet Model Capacity Analysis")
    print("=" * 80)
    print(f"{'Base Filters':<15} {'Total Params':<20} {'Trainable Params':<20} {'Size (MB)':<15}")
    print("-" * 80)
    
    for base_filters in base_filters_list:
        model = UNet(n_channels=3, n_classes=1, base_filters=base_filters, dropout_prob=0.0)
        total_params, trainable_params = count_parameters(model)
        size_mb = estimate_model_size_mb(model)
        
        results[base_filters] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "size_mb": size_mb,
            "total_params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6,
        }
        
        print(f"{base_filters:<15} {total_params:>15,} ({total_params/1e6:>5.2f}M) "
              f"{trainable_params:>15,} ({trainable_params/1e6:>5.2f}M) "
              f"{size_mb:>10.2f} MB")
    
    print("=" * 80)
    
    # Calculate relative increases
    if len(base_filters_list) > 1:
        print("\nRelative Increase (vs base_filters=64):")
        print("-" * 80)
        baseline = results.get(64, {})
        if baseline:
            baseline_params = baseline["total_params"]
            for base_filters in base_filters_list:
                if base_filters != 64:
                    params = results[base_filters]["total_params"]
                    increase = ((params - baseline_params) / baseline_params) * 100
                    print(f"base_filters={base_filters:3d}: {increase:>+6.1f}% parameters "
                          f"({params/baseline_params:.2f}x larger)")
    
    return results


def recommend_base_filters(target_params_m=None, max_size_mb=None):
    """
    Recommend base_filters based on constraints.
    
    Args:
        target_params_m: Target number of parameters in millions (e.g., 10 for 10M)
        max_size_mb: Maximum model size in MB
    
    Returns:
        Recommended base_filters value
    """
    # Test common values
    test_values = [32, 48, 64, 96, 128, 160, 192, 256]
    results = {}
    
    for base_filters in test_values:
        model = UNet(n_channels=3, n_classes=1, base_filters=base_filters)
        total_params, _ = count_parameters(model)
        size_mb = estimate_model_size_mb(model)
        results[base_filters] = {
            "params_m": total_params / 1e6,
            "size_mb": size_mb
        }
    
    if target_params_m:
        # Find closest to target
        best = min(test_values, key=lambda x: abs(results[x]["params_m"] - target_params_m))
        print(f"Target: {target_params_m}M parameters")
        print(f"Recommended: base_filters={best} ({results[best]['params_m']:.2f}M params, "
              f"{results[best]['size_mb']:.2f} MB)")
        return best
    
    if max_size_mb:
        # Find largest that fits
        candidates = [x for x in test_values if results[x]["size_mb"] <= max_size_mb]
        if candidates:
            best = max(candidates)
            print(f"Max size: {max_size_mb} MB")
            print(f"Recommended: base_filters={best} ({results[best]['params_m']:.2f}M params, "
                  f"{results[best]['size_mb']:.2f} MB)")
            return best
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze UNet model capacity")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze different base_filters values")
    parser.add_argument("--target_params", type=float,
                       help="Target parameters in millions (e.g., 10 for 10M)")
    parser.add_argument("--max_size_mb", type=float,
                       help="Maximum model size in MB")
    parser.add_argument("--base_filters", nargs="+", type=int,
                       default=[32, 48, 64, 96, 128, 160, 192],
                       help="List of base_filters to analyze")
    
    args = parser.parse_args()
    
    if args.target_params or args.max_size_mb:
        recommend_base_filters(target_params_m=args.target_params, max_size_mb=args.max_size_mb)
    else:
        analyze_base_filters(args.base_filters)
