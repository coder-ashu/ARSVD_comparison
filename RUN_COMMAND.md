# Command to Run the Pipeline

## Recommended Command (Optimal Settings)

```bash
python run_pipeline.py \
  --data_root /content/ARSVD_comparison/data \
  --out_dir ./artifacts_optimized \
  --device cuda \
  --epochs 50 \
  --patience 20 \
  --use_dice_loss \
  --augment \
  --svd_ranks "100,150,200" \
  --arsvd_taus "0.95,0.9,0.85,0.8"
```

## For Windows PowerShell

```powershell
python run_pipeline.py `
  --data_root "C:\path\to\your\data" `
  --out_dir ./artifacts_optimized `
  --device cuda `
  --epochs 50 `
  --patience 20 `
  --use_dice_loss `
  --augment `
  --svd_ranks "100,150,200" `
  --arsvd_taus "0.95,0.9,0.85,0.8"
```

## Minimal Command (Default Settings)

```bash
python run_pipeline.py --data_root /path/to/data
```

## All Available Options

```bash
python run_pipeline.py \
  --data_root /path/to/data \              # REQUIRED: Path to data folder
  --out_dir ./artifacts \                  # Output directory (default: ./artifacts)
  --device cuda \                          # Device: cuda or cpu (default: cuda)
  --epochs 50 \                            # Training epochs (default: 10)
  --lr 2e-4 \                              # Learning rate (default: 2e-4, optimized)
  --batch_size 8 \                         # Batch size (default: 8)
  --image_size 256 256 \                   # Image size H W (default: 256 256)
  --patience 20 \                          # Early stopping patience (default: 20)
  --use_dice_loss \                        # Use Dice Loss (recommended!)
  --augment \                              # Enable data augmentation
  --svd_ranks "100,150,200" \              # SVD ranks to test
  --arsvd_taus "0.95,0.9,0.85,0.8" \      # ARSVD tau values to test
  --dropout_prob 0.0 \                     # Dropout (default: 0.0, no dropout)
  --weight_decay 0.0                       # Weight decay (default: 0.0)
```

## What's Optimized (Automatic)

- ✅ **base_filters=96** (automatically set, 2.25x capacity)
- ✅ **Learning rate=2e-4** (auto-adjusted to 3e-4 for large model)
- ✅ **LR scheduler** (patience=8, min_lr=1e-5)
- ✅ **Gradient clipping** (max_norm=5.0)
- ✅ **Mask binarization** (proper binary masks)
- ✅ **Data normalization** (fixed double normalization)

## Expected Output

The pipeline will:
1. Load and preprocess data
2. Train baseline U-Net (base_filters=96) with Dice Loss
3. Compress with SVD at ranks 100, 150, 200
4. Compress with ARSVD at taus 0.95, 0.9, 0.85, 0.8
5. Evaluate all variants
6. Save results to `experiment_summary.json`

## Expected Performance

- **Baseline Dice**: 0.85-0.88 (target: >0.80) ✅
- **Baseline IoU**: 0.75-0.80 (target: >0.70) ✅
- **Training time**: ~2-3 hours on GPU (depends on dataset size)
- **Model size**: ~17M parameters (~66 MB)
