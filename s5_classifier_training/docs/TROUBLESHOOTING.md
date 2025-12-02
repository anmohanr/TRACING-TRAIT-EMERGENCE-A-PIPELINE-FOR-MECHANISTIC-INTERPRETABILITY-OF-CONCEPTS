# Troubleshooting Guide

This guide covers common issues and their solutions. Use Ctrl+F to search for your error message.

**Last updated:** 2025-11-01 (Framework reorganization - scripts now in `framework/` directory)

## Table of Contents
1. [Framework Update Issues (2025-11-01)](#framework-update-issues-2025-11-01)
2. [Dataset Loading Errors](#dataset-loading-errors)
3. [Training Errors](#training-errors)
4. [Checkpoint Errors](#checkpoint-errors)
5. [Path Resolution Issues](#path-resolution-issues)
6. [Model Serialization Problems](#model-serialization-problems)
7. [Memory Issues](#memory-issues)
8. [Cross-Validation Failures](#cross-validation-failures)
9. [Configuration Errors](#configuration-errors)
10. [Comparison Tool Issues](#comparison-tool-issues)
11. [General Debugging Tips](#general-debugging-tips)

---

## Framework Update Issues (2025-11-01)

### Issue: ModuleNotFoundError after framework update

**Error:**
```
ModuleNotFoundError: No module named 'shared_utils'
```

**Cause**: Scripts were moved from root to `framework/` directory

**Solution**: Update paths to use new `framework/` location:
```bash
# Old (before 2025-11-01):
uv run python ../../train.py

# New (after 2025-11-01):
uv run python ../../framework/train.py
```

All script paths now require `framework/`:
- `../../framework/train.py`
- `../framework/checkpoint.py`
- `../framework/compare.py`
- `../framework/new_experiment.py`

### Issue: Metric names not found in old checkpoints

**Error**: `compare.py` shows `nan` for metrics in old checkpoints (v0-v3)

**Cause**: Metric naming changed with framework update

**Old vs New Metric Names:**

| Old Name | New Name | Type |
|----------|----------|------|
| `accuracy` | `cv_mean` (classification) | Cross-validation mean |
| `cv_score` | `cv_mean` | Cross-validation mean |
| (not tracked) | `cv_std` | Cross-validation std dev |
| `accuracy` (ambiguous) | `test_accuracy` | Test set accuracy |
| `f1` | `test_f1` | Test F1 score |
| `mae` | `test_mae` | Test MAE |
| `mse` | `test_mse` | Test MSE |
| `r2` | `test_r2` | Test R² |

**Solution**:
1. **For new experiments**: Use new framework, metrics will be correct
2. **For comparing old checkpoints**: They will show `nan` for new metric names - this is expected
3. **To migrate old checkpoints** (optional):
```python
# Update old .metadata.json files
import json
from pathlib import Path

for version_dir in Path('.').glob('v*'):
    metadata_path = version_dir / '.metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)

        # Map old to new names
        if 'accuracy' in meta and 'cv_mean' not in meta:
            meta['cv_mean'] = meta.get('accuracy')
        if 'accuracy' in meta and 'test_accuracy' not in meta:
            meta['test_accuracy'] = meta.get('accuracy')  # Approximate

        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)
```

### Issue: Train/test split changed results

**Observation**: Results different after framework update

**Cause**: Framework now implements proper 80/20 train/test split (default)

**Old behavior**: Evaluated on same data used for training (overfitting)
**New behavior**: Separate test set, more realistic performance estimates

**Expected changes**:
- `test_accuracy` may be lower than old `accuracy`
- More honest assessment of model generalization
- CV scores should be similar to before

**Solution**: This is correct behavior - new results are more trustworthy

---

## Dataset Loading Errors

### Error: `FileNotFoundError: Dataset NPZ file not found`

**Full error**:
```
FileNotFoundError: Dataset NPZ file not found: /path/to/file.npz
```

**Causes**:
1. Incorrect path in dataset.config
2. Running from wrong directory
3. Dataset file doesn't exist
4. Relative path resolved incorrectly

**Solutions**:
```bash
# 1. Check current directory
pwd
# Should be: .../s5_classifier_training/d1_embedding_to_feature/working

# 2. Verify dataset.config path
cat dataset.config
# Should show: ../../../s4_classifier_data_prep/generated_datasets/dishonesty_xxx.npz

# 3. Test if file exists
ls -la ../../../s4_classifier_data_prep/generated_datasets/

# 4. Use absolute path if relative paths are problematic
echo "/absolute/path/to/dataset.npz" > dataset.config
```

### Error: `FileNotFoundError: dataset.config not found`

**Causes**:
1. No dataset.config in working/ or parent directory
2. Running from wrong location

**Solutions**:
```bash
# Create dataset.config
echo "../../../s4_classifier_data_prep/generated_datasets/dishonesty_xxx.npz" > dataset.config

# Or copy from a checkpoint
cp ../v0_baseline/dataset.config .
```

### Error: `ValueError: can only convert an array of size 1 to a Python scalar`

**Cause**: NPZ file has metadata arrays that can't be converted to scalars

**Solution**: This is fixed in current version. Update framework/shared_utils.py if you see this.

### Error: `KeyError: 'X'` or `KeyError: 'y'`

**Cause**: NPZ file doesn't have required X or y arrays

**Solutions**:
```python
# Check NPZ contents
import numpy as np
data = np.load('path/to/file.npz')
print(data.files)  # Should show ['X', 'y', ...]

# Regenerate dataset in s4 if corrupted
```

---

## Training Errors

### Error: `ValueError: Unknown label type: continuous`

**Full error**:
```
ValueError: Unknown label type: continuous. Maybe you are trying to fit a classifier,
which expects discrete classes on a regression target with continuous values.
```

**Cause**: LogisticRegression being used for regression task

**Solution**: Update framework/train.py to use Ridge for regression (this is fixed in current version)

### Error: `ConvergenceWarning: Maximum iterations reached`

**Warning**:
```
ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
```

**Solutions**:
```yaml
# In config.yaml, increase max_iter
hyperparams:
  max_iter: 10000  # Was 1000
```

Or scale your data:
```python
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
```

### Error: `ValueError: n_splits=5 cannot be greater than the number of members in each class`

**Cause**: Too few samples for 5-fold cross-validation with stratification

**Solutions**:
```yaml
# In config.yaml, reduce CV folds
training:
  cv_folds: 3  # Was 5
```

Or get more training data.

### Error: `MemoryError` during training

**Cause**: Dataset too large (especially d3 with 27,648 features)

**Solutions**:
1. Reduce feature dimensions in s4 (use fewer layers)
2. Use SGD-based models that support partial_fit
3. Reduce n_jobs in Random Forest:
```yaml
hyperparams:
  n_jobs: 1  # Was -1
```

---

## Checkpoint Errors

### Error: `working/ directory not found`

**When running**: `python ../checkpoint.py --name "test"`

**Solution**:
```bash
# Create working directory
mkdir working
# Copy files from previous version
cp v0_baseline/* working/
```

### Error: No model.pkl or results.json in checkpoint

**Cause**: Checkpointing before training

**Solution**:
```bash
# Train first
cd working
uv run python ../../framework/train.py
# Then checkpoint
cd ..
uv run python ../framework/checkpoint.py --name "after_training"
```

### Error: Checkpoint overwrites existing version

**Cause**: Same version name already exists

**Solution**: Checkpoint.py auto-increments version numbers. This shouldn't happen unless manually created.

---

## Path Resolution Issues

### Issue: Confusing relative paths

**Problem**: Not sure how paths resolve from different locations

**Understanding path resolution**:
```
Current location: d1_embedding_to_feature/working/
dataset.config contains: ../../../s4_classifier_data_prep/generated_datasets/file.npz

Resolution:
working/ → ../ → d1_embedding_to_feature/
         → ../ → s5_classifier_training/
         → ../ → final_proj/
         → s4_classifier_data_prep/generated_datasets/file.npz
```

**Best practice**: Use absolute paths during debugging:
```bash
# Get absolute path
readlink -f ../../../s4_classifier_data_prep/generated_datasets/dishonesty_xxx.npz
# Put in dataset.config
```

### Issue: Scripts can't find each other

**When running**: `python ../../framework/train.py`
**Error**: `ModuleNotFoundError: No module named 'framework'`

**Solution**: Always use `uv run python` instead of `python`:
```bash
uv run python ../../framework/train.py  # Correct
python ../../framework/train.py         # Wrong - won't find imports
```

---

## Model Serialization Problems

### Error: `ModuleNotFoundError` when loading model.pkl

**Cause**: Model pickled with different Python/sklearn version

**Solutions**:
```bash
# Check versions
uv run python -c "import sklearn; print(sklearn.__version__)"

# Retrain model with current environment
cd working
uv run python ../../framework/train.py
```

### Error: `AttributeError` when loading model

**Cause**: sklearn API changed between versions

**Solution**: Retrain model or install compatible sklearn version:
```bash
uv pip install scikit-learn==1.3.0  # Match version that created model
```

### Error: Can't pickle lambda functions or local functions

**Cause**: Custom preprocessing functions in model pipeline

**Solution**: Use only sklearn transformers or define functions at module level

---

## Memory Issues

### Problem: Process killed during training

**Message**: `Killed` or `Process finished with exit code 137`

**Cause**: Out of memory (OOM)

**Solutions for d3 (27,648 features)**:
```python
# 1. Use SGDClassifier (online learning)
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log_loss')  # For logistic

# 2. Reduce features with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X)

# 3. Use feature selection
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=1000)
X_reduced = selector.fit_transform(X, y)
```

### Problem: Slow training with Random Forest

**Solutions**:
```yaml
# Reduce trees and parallelize
hyperparams:
  n_estimators: 50  # Was 100
  n_jobs: -1        # Use all cores
  max_depth: 10     # Limit tree depth
```

---

## Cross-Validation Failures

### Error: `ValueError: n_splits=5 cannot be greater than number of samples=4`

**Cause**: Too few samples for requested CV folds

**Solution**:
```python
# Dynamic CV folds based on sample size
n_samples = len(X)
cv_folds = min(5, n_samples)  # Never more folds than samples
```

### Error: `ValueError: The least populated class has only 1 member`

**Cause**: Severe class imbalance in classification

**Solutions**:
```python
# 1. Use stratified splits with shuffle
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 2. Use class weights
model = LogisticRegression(class_weight='balanced')

# 3. Get more data for minority classes
```

### Warning: High variance in CV scores

**Example**: CV scores: [0.4, 0.9, 0.5, 0.8, 0.6]

**Causes**:
1. Too few samples
2. Unbalanced folds

**Solution**: Use repeated CV:
```python
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=5, n_repeats=3)
```

---

## Configuration Errors

### Error: `KeyError: 'model_type'`

**Cause**: Malformed config.yaml

**Solution - Correct format**:
```yaml
model_type: logistic  # Required at root level

hyperparams:          # Nested under hyperparams
  C: 1.0
  max_iter: 1000

training:             # Optional section
  cv_folds: 5
```

### Error: `TypeError: __init__() got an unexpected keyword argument`

**Cause**: Invalid hyperparameter for model type

**Solution - Check valid parameters**:
```python
# For LogisticRegression
valid_params = ['C', 'max_iter', 'penalty', 'solver', ...]

# For RandomForest
valid_params = ['n_estimators', 'max_depth', 'min_samples_split', ...]
```

### Error: YAML parsing error

**Cause**: Invalid YAML syntax

**Common issues**:
```yaml
# Wrong - no quotes around scientific notation
C: 1e-3           # Parsed as string "1e-3"

# Correct
C: 0.001          # Or
C: !!float 1e-3   # Force float parsing
```

---

## Comparison Tool Issues

### Problem: compare.py shows empty table

**Cause**: No .metadata.json files in versions

**Solution**: Regenerate metadata:
```python
# Create .metadata.json for old versions
import json
from pathlib import Path

for version_dir in Path('.').glob('v*'):
    if not (version_dir / '.metadata.json').exists():
        # Load results
        with open(version_dir / 'results.json') as f:
            results = json.load(f)

        # Create metadata
        metadata = {
            'version': version_dir.name,
            'cv_mean': results.get('cv_mean') or results.get('accuracy'),
            'test_accuracy': results.get('test_accuracy') or results.get('accuracy'),
            'test_mae': results.get('test_mae') or results.get('mae'),
            'model_type': results.get('model_type')
        }

        with open(version_dir / '.metadata.json', 'w') as f:
            json.dump(metadata, f)
```

### Problem: Can't compare across datasets

**Solution**:
```bash
# Use --all flag
uv run python framework/compare.py --all

# Or specify multiple
uv run python framework/compare.py --versions d1/v0 d2/v0 d3/v0
```

---

## General Debugging Tips

### 1. Verbose Logging

Add to scripts:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Interactive Debugging

```python
# Add breakpoint in framework/train.py
import pdb; pdb.set_trace()

# Or use IPython
from IPython import embed; embed()
```

### 3. Check Data Shapes

Always verify dimensions:
```python
print(f"X shape: {X.shape}")  # Should be (n_samples, n_features)
print(f"y shape: {y.shape}")  # Should be (n_samples,)
print(f"y unique: {np.unique(y)}")  # Check values
```

### 4. Test with Minimal Data

Create tiny test dataset:
```python
# Create test NPZ
import numpy as np

X = np.random.randn(10, 100)  # 10 samples, 100 features
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Binary classification

np.savez('test_data.npz', X=X, y=y)
```

### 5. Verify Environment

```bash
# Check you're in s5 directory structure
pwd | grep s5_classifier_training

# Check virtual environment is active
which python  # Should show .venv path

# Check dependencies installed
uv pip list | grep scikit-learn
```

### 6. Common Quick Fixes

```bash
# Reset working directory
rm -rf working/*
cp v0_baseline/* working/

# Reinstall dependencies
uv pip install -r requirements.txt

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## Error Recovery Procedures

### Corrupted Checkpoint

If a checkpoint is corrupted:
```bash
# Create new checkpoint from working
cd d1_embedding_to_feature
rm -rf v1_corrupted
uv run python ../framework/checkpoint.py --name "recovered"
```

### Lost Working Directory

If working/ was deleted:
```bash
# Restore from latest checkpoint
cp -r v2_latest/* working/
```

### Inconsistent State

If state is inconsistent (partial files):
```bash
# Full reset to known good state
cd d1_embedding_to_feature
rm -rf working
mkdir working
cp v0_baseline/config.yaml working/
cp v0_baseline/dataset.config working/
echo "# Starting fresh" > working/experiment_notes.md
```

---

## Getting Help

If these solutions don't work:

1. **Check logs**: Look for detailed error messages
2. **Simplify**: Try with minimal config/data
3. **Compare with working version**: What's different from v0_baseline?
4. **File an issue**: Include full error traceback, config.yaml, and dataset info

### Debug Information to Collect

When reporting issues, provide:
```bash
# System info
uv run python -c "import sys; print(sys.version)"
uv run python -c "import sklearn; print(sklearn.__version__)"

# Data info
uv run python -c "import numpy as np; d=np.load('your_dataset.npz'); print(d.files); print(d['X'].shape, d['y'].shape)"

# Config
cat config.yaml

# Full error
uv run python ../../framework/train.py 2>&1 | tee error.log
```

---

## Prevention Checklist

Before running experiments:

- [ ] Correct working directory (`pwd` ends with /working)
- [ ] dataset.config exists and points to valid file
- [ ] config.yaml is valid YAML with required fields
- [ ] Virtual environment activated (`uv` commands work)
- [ ] Enough disk space for checkpoints
- [ ] Enough RAM for dataset (roughly 8 bytes per feature×sample)

---

## Common Patterns That Cause Issues

### Pattern 1: Forgetting to use uv run
```bash
python framework/train.py  # ❌ Wrong - import errors
uv run python framework/train.py  # ✅ Correct
```

### Pattern 2: Running from wrong directory
```bash
cd s5_classifier_training
uv run python framework/train.py  # ❌ Wrong location

cd d1_embedding_to_feature/working
uv run python ../../framework/train.py  # ✅ Correct
```

### Pattern 3: Editing versioned checkpoints
```bash
vim v1_baseline/config.yaml  # ❌ Don't edit checkpoints

cp -r v1_baseline/* working/
vim working/config.yaml  # ✅ Edit in working/
```

### Pattern 4: Not checkpointing good results
```bash
# After getting good results...
rm -rf working/*  # ❌ Lost your work!

# Instead:
uv run python ../framework/checkpoint.py --name "good_result"  # ✅ Save first
```

This troubleshooting guide covers the most common issues. For issues not listed here, check the error message carefully - it often suggests the solution!