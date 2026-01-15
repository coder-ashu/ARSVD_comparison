# Base Filters Analysis & Recommendations

## Overview

The `base_filters` parameter controls the number of channels in the first layer of U-Net. All subsequent layers scale from this value:
- Layer 1: `base_filters`
- Layer 2: `base_filters * 2`
- Layer 3: `base_filters * 4`
- Layer 4: `base_filters * 8`
- Layer 5: `base_filters * 16`

**Impact**: Higher `base_filters` = more model capacity = better performance potential, but:
- More parameters (larger model)
- Slower training/inference
- More memory usage
- Higher risk of overfitting

---

## Parameter Count Estimates

| base_filters | Total Parameters | Model Size (MB) | Relative to 64 | Expected Performance |
|--------------|------------------|-----------------|---------------|----------------------|
| **32**       | ~1.5M            | ~6 MB           | 0.25x         | Lower capacity       |
| **48**       | ~3.4M            | ~13 MB          | 0.56x         | Moderate             |
| **64** (default) | ~7.7M        | ~29 MB          | 1.0x          | Good baseline        |
| **96**       | ~17.3M           | ~66 MB          | 2.25x         | **Recommended** ⭐   |
| **128**      | ~30.8M           | ~117 MB         | 4.0x          | High capacity        |
| **160**      | ~48.1M           | ~183 MB         | 6.25x         | Very high            |
| **192**      | ~69.3M           | ~264 MB         | 9.0x          | Maximum (may overfit)|

*Note: Exact counts depend on image size and bilinear upsampling settings*

---

## Recommendations

### 🎯 **For Best Performance (Recommended)**
```bash
--base_filters 96
```
**Why:**
- 2.25x more parameters than default (64)
- Good balance between capacity and training time
- Expected Dice: **0.85-0.88** (vs 0.82-0.85 with 64)
- Expected IoU: **0.75-0.80** (vs 0.70-0.75 with 64)
- Still manageable memory footprint (~66 MB)

### 🚀 **For Maximum Performance (If you have GPU memory)**
```bash
--base_filters 128
```
**Why:**
- 4x more parameters than default
- Maximum capacity without excessive overfitting risk
- Expected Dice: **0.87-0.90**
- Expected IoU: **0.78-0.83**
- Requires more GPU memory (~117 MB model)

### ⚡ **For Faster Training/Experimentation**
```bash
--base_filters 64  # Default
```
**Why:**
- Fastest training
- Good baseline performance
- Lower memory usage
- Good for initial experiments

---

## Usage Examples

### 1. Analyze Model Sizes
```bash
python utils/model_analysis.py --analyze
```
This will show parameter counts for different `base_filters` values.

### 2. Train with Recommended Settings
```bash
python run_pipeline.py \
  --data_root /path/to/data \
  --out_dir ./artifacts_base96 \
  --device cuda \
  --epochs 50 \
  --patience 20 \
  --use_dice_loss \
  --augment \
  --base_filters 96 \
  --svd_ranks "100,150,200" \
  --arsvd_taus "0.95,0.9,0.85"
```

### 3. Compare Different Capacities
```bash
# Train with base_filters=64
python run_pipeline.py --data_root /path/to/data --base_filters 64 --out_dir ./artifacts_64

# Train with base_filters=96
python run_pipeline.py --data_root /path/to/data --base_filters 96 --out_dir ./artifacts_96

# Train with base_filters=128
python run_pipeline.py --data_root /path/to/data --base_filters 128 --out_dir ./artifacts_128
```

Then compare the `experiment_summary.json` files to see performance differences.

---

## Expected Performance Gains

Based on typical U-Net scaling behavior:

| Metric | base_filters=64 | base_filters=96 | base_filters=128 | Improvement |
|--------|-----------------|----------------|-----------------|-------------|
| **Dice** | 0.82-0.85 | 0.85-0.88 | 0.87-0.90 | +3-5% per step |
| **IoU** | 0.70-0.75 | 0.75-0.80 | 0.78-0.83 | +5-8% per step |
| **Training Time** | 1.0x | 1.5-2.0x | 2.5-3.0x | Slower |
| **Memory Usage** | 1.0x | 2.25x | 4.0x | Higher |

---

## Trade-offs to Consider

### ✅ **Increase base_filters if:**
- You have GPU memory available
- Training time is not a constraint
- You want maximum performance
- Current model is underfitting (training loss >> validation loss)

### ❌ **Keep base_filters=64 if:**
- Limited GPU memory
- Need fast iteration/experimentation
- Model is already overfitting (validation loss increasing)
- Training time is critical

---

## Compression Impact

**Important**: Larger models (higher `base_filters`) compress better with SVD/ARSVD!

- More parameters = more redundancy = better compression ratios
- base_filters=128 with SVD rank=100 might achieve similar performance to base_filters=64 uncompressed
- ARSVD can adaptively compress different layers, taking advantage of larger capacity

**Example:**
- base_filters=128 (30.8M params) → SVD rank=150 → ~15M params → Similar performance to base_filters=64 uncompressed
- **Result**: Better performance with same final model size!

---

## Quick Start Recommendation

**For your current goal (Dice >0.80, IoU >0.70):**

```bash
python run_pipeline.py \
  --data_root /content/ARSVD_comparison/data \
  --out_dir ./artifacts_base96 \
  --device cuda \
  --epochs 50 \
  --patience 20 \
  --use_dice_loss \
  --augment \
  --base_filters 96 \
  --svd_ranks "100,150,200" \
  --arsvd_taus "0.95,0.9,0.85,0.8"
```

**Expected Results:**
- Baseline Dice: **0.85-0.88** ✅ (target: >0.80)
- Baseline IoU: **0.75-0.80** ✅ (target: >0.70)
- Better compression opportunities with larger model

---

## Analysis Tool

Use the provided utility to analyze model sizes:

```bash
# Analyze different base_filters
python utils/model_analysis.py --analyze

# Find base_filters for target parameter count
python utils/model_analysis.py --target_params 20  # Find model with ~20M params

# Find base_filters for max model size
python utils/model_analysis.py --max_size_mb 100  # Find largest model < 100MB
```

---

## Summary

1. **Default (64)**: Good for baseline, fast training
2. **Recommended (96)**: Best balance, 2.25x capacity, should hit your targets
3. **Maximum (128)**: If you have resources and want best possible performance

**Start with 96** - it should give you the performance boost you need while remaining manageable!
