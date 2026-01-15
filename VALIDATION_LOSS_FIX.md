# Validation Loss Not Improving - Issues Fixed

## Problems Identified

### 1. **Learning Rate Too Low for Larger Model** 🔴 CRITICAL
**Problem**: 
- Default LR: `1e-4`
- With `base_filters=96` (17M parameters), this LR is too low
- Larger models need higher learning rates to learn effectively

**Fix**:
- Increased default LR to `2e-4` (2x higher)
- Added automatic LR adjustment: if model has >10M params, increase LR by 50%
- For base_filters=96: effective LR will be `2e-4 * 1.5 = 3e-4`

### 2. **Learning Rate Scheduler Too Aggressive** 🔴 CRITICAL
**Problem**:
- Patience: 5 epochs (too short)
- If validation loss plateaus for 5 epochs, LR gets cut in half
- This happens too early, causing LR to drop before model has chance to learn
- `min_lr=1e-6` is too low - model stops learning

**Fix**:
- Increased patience from 5 to **8 epochs**
- Increased `min_lr` from `1e-6` to `1e-5` (10x higher minimum)
- Gives model more time to learn before reducing LR

### 3. **Gradient Clipping Too Aggressive** 🟡
**Problem**:
- `max_norm=1.0` is too restrictive
- Clips gradients too aggressively, preventing learning
- Can cause gradients to vanish, stopping optimization

**Fix**:
- Increased `max_norm` from 1.0 to **5.0**
- Allows gradients to flow better while still preventing explosions

### 4. **No Learning Rate Monitoring** 🟡
**Problem**:
- Can't see when LR gets too low
- No warning when model might have stopped learning

**Fix**:
- Added LR to epoch printout
- Added warning when LR drops below `1e-5`

---

## Changes Made

### File: `steps/model_train.py`

1. **Automatic LR Adjustment**:
```python
# CRITICAL FIX: Use higher learning rate for larger models
effective_lr = lr
total_params = sum(p.numel() for p in model.parameters())
if total_params > 10e6:  # If model has >10M parameters
    effective_lr = lr * 1.5  # Increase LR by 50%
```

2. **Better Scheduler Settings**:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=1e-5
)
```

3. **Less Aggressive Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

4. **LR Monitoring**:
```python
current_lr = optimizer.param_groups[0]['lr']
print(f"... lr={current_lr:.6f}")
if current_lr < 1e-5:
    print(f"⚠️  WARNING: Learning rate is very low...")
```

### File: `run_pipeline.py`

1. **Higher Default LR**:
```python
p.add_argument("--lr", type=float, default=2e-4,  # Was 1e-4
               help="Learning rate (default: 2e-4, increased for base_filters=96 model)")
```

---

## Expected Impact

### Before Fixes:
- Validation loss plateaus early
- LR drops too quickly (after 5 epochs)
- Model stops learning (LR too low)
- Validation loss doesn't improve

### After Fixes:
- ✅ Higher initial LR (2e-4 → 3e-4 for large models)
- ✅ More patience (8 epochs vs 5) before LR reduction
- ✅ Higher minimum LR (1e-5 vs 1e-6) - model keeps learning
- ✅ Better gradient flow (max_norm=5.0)
- ✅ LR monitoring to catch issues early

**Expected Results**:
- Validation loss should continue improving for longer
- Better convergence
- Higher final Dice/IoU scores

---

## How to Verify

After training, check the logs for:

1. **Learning Rate Values**:
   - Should start at ~3e-4 (for base_filters=96)
   - Should decrease gradually (not too quickly)
   - Should stay above 1e-5

2. **Validation Loss Trend**:
   - Should decrease over many epochs
   - Should not plateau too early
   - Should improve even after LR reduction

3. **Warning Messages**:
   - If you see "⚠️ WARNING: Learning rate is very low", the model might need even higher initial LR

---

## If Validation Loss Still Doesn't Improve

Try these additional fixes:

1. **Even Higher Learning Rate**:
   ```bash
   --lr 3e-4  # or even 5e-4
   ```

2. **Longer Patience**:
   - The code now uses patience=8, but you can increase it in `run_pipeline.py` if needed

3. **Check Data**:
   - Ensure validation set is properly loaded
   - Check if masks are correctly binarized
   - Verify image normalization

4. **Check Model Capacity**:
   - base_filters=96 should be sufficient
   - If still not learning, might need to check data quality

---

## Summary

The main issues were:
1. **LR too low** for the larger model → Fixed with automatic adjustment
2. **Scheduler too aggressive** → Fixed with longer patience
3. **Gradient clipping too restrictive** → Fixed with higher max_norm
4. **No monitoring** → Fixed with LR tracking

These fixes should allow validation loss to improve properly! 🎯
