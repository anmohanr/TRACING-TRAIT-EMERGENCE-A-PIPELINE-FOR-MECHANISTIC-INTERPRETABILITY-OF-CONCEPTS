# Classifier Training - Usage Guide

Practical guide for running experiments, training models, and managing checkpoints.

## Quick Start

```bash
# 1. Navigate to a working directory
cd d3_hidden_to_concept/working

# 2. Edit config.yaml to set model and hyperparameters
# (See Configuration section below)

# 3. Train model
uv run python ../../framework/train.py

# 4. Checkpoint successful experiment
cd ..
uv run python ../framework/checkpoint.py \
  --name "svm_rbf" \
  --message "SVM with RBF kernel, 84.5% CV, 85.8% test"

# 5. Compare all versions
uv run python ../framework/compare.py
```

## Setup

### Dependencies

All dependencies are managed via `uv` and should already be installed:
- torch
- scikit-learn
- numpy
- pandas
- pyyaml

If missing, install with:
```bash
cd s5_classifier_training
uv add scikit-learn numpy pandas pyyaml
```

### Directory Requirements

Each experiment (d1, d2, d3) requires:
- `working/config.yaml` - Model configuration
- `working/dataset.config` - Path to dataset .npz file

## Configuration

### config.yaml Structure

```yaml
model_type: svm  # Options: logistic, random_forest, svm

hyperparams:
  # For SVM
  C: 1.0
  kernel: rbf  # Options: linear, rbf, poly
  gamma: scale

  # For Logistic/Ridge
  # C: 1.0
  # max_iter: 1000

  # For Random Forest
  # n_estimators: 100
  # max_depth: 10
  # min_samples_split: 2

training:
  cv_folds: 5         # Cross-validation folds
  test_split: 0.2     # Test set fraction (default: 0.2)
  random_state: 42    # Reproducibility seed
```

### dataset.config Structure

Simple text file with path to dataset:
```
../../../s4_classifier_data_prep/generated_datasets/dishonesty_1020_h19_to_concept.npz
```

Can be relative or absolute path.

## Training Workflow

### 1. Configure Experiment

```bash
cd d3_hidden_to_concept/working
nano config.yaml  # Edit model and hyperparameters
```

### 2. Run Training

```bash
uv run python ../../framework/train.py
```

Expected output:
```
================================================================================
DATASET
================================================================================
Total: 1020 samples, 2304 features
Split: 816 train (80%) / 204 test (20%)

================================================================================
CROSS-VALIDATION (5-fold on training set)
================================================================================
Fold scores: [0.85, 0.82, 0.85, 0.85, 0.85]
CV mean: 0.8456 ± 0.0119

================================================================================
FINAL MODEL TRAINING & EVALUATION
================================================================================
Training SVC on 816 samples...

Test set (204 samples):
  Accuracy: 0.8578
  F1 score: 0.8307

Training set (sanity check):
  Accuracy: 0.9069

Saved model to model.pkl
Saved results to results.json
```

### 3. Inspect Results

```bash
# View full results
cat results.json

# Quick metrics
cat results.json | python -m json.tool | grep -A 3 '"cv_mean"'
```

### 4. Checkpoint Experiment

```bash
cd ..  # Move to dataset directory (e.g., d3_hidden_to_concept/)

uv run python ../framework/checkpoint.py \
  --name "descriptive_name" \
  --message "What changed and why"
```

Creates `v{N}_descriptive_name/` with:
- All files from working/
- `.metadata.json` - Quick reference metrics
- `README.md` - Auto-generated summary

### 5. Compare Versions

```bash
# Compare all versions in current dataset
uv run python ../framework/compare.py

# Compare with detailed deltas
uv run python ../framework/compare.py --detailed

# Compare specific versions
uv run python ../framework/compare.py --versions v1_baseline v2_svm

# Compare all datasets
cd ..  # Move to s5_classifier_training/
uv run python framework/compare.py --all
```

Example output:
```
version              cv_mean  cv_std  test_accuracy  test_f1  model_type
v0_baseline          0.8010   0.049   0.8500         0.8500   LogisticRegression
v1_random_forest     0.8069   0.065   0.8500         0.8400   RandomForestClassifier
v2_svm_winner        0.8456   0.012   0.8578         0.8307   SVC
```

## Understanding Metrics

### Primary Metric: cv_mean

**Cross-validation mean** is your primary metric for model selection:
- Computed on training set only (no test set leakage)
- Averaged across 5 folds for stability
- **Use this to compare models and select best approach**

### Secondary Validation: test_accuracy / test_mae

Held-out test set provides final validation:
- Should be close to cv_mean (within 5 percentage points)
- Large gaps indicate overfitting or lucky train/test split
- **Use this to verify generalization**

### Overfitting Check: train_accuracy

Training set performance (sanity check):
- Should be higher than test performance
- If train_accuracy >> test_accuracy, model is overfitting
- **Ideal gap: 5-10 percentage points**

### Example Interpretation

```json
{
  "cv_mean": 0.846,
  "test_accuracy": 0.858,
  "train_accuracy": 0.907
}
```

✅ **Good model**:
- CV-test gap: 1.2 pts (excellent generalization)
- Train-test gap: 4.9 pts (minor overfitting)
- All metrics in similar range

```json
{
  "cv_mean": 0.68,
  "test_accuracy": 1.0,
  "train_accuracy": 1.0
}
```

🚨 **Overfitted model**:
- CV-test gap: 32 pts (suspicious!)
- Perfect test accuracy unlikely with 68% CV
- Suggests test set contamination or extreme luck

## Advanced Usage

### Custom Train/Test Split

```yaml
training:
  test_split: 0.3  # Use 30% for testing
```

Or disable split entirely (CV only):
```yaml
training:
  test_split: 0  # No test set, train on all data
```

### Model-Specific Tips

**Logistic Regression / Ridge:**
- Fast baseline, good for debugging
- Increase `max_iter` if convergence fails
- Lower `C` for stronger regularization

**Random Forest:**
- Set `max_depth` to prevent overfitting
- Increase `n_estimators` for more stable predictions
- Check feature importance for interpretability

**SVM:**
- `kernel: rbf` for non-linear patterns
- `kernel: linear` for interpretability
- Tune `C` and `gamma` for performance
- Can be slow on large datasets (>10K samples)

### Starting from Previous Version

```bash
cd d3_hidden_to_concept
uv run python ../framework/new_experiment.py \
  --from v2_svm_winner \
  --note "Try different regularization"
```

Copies v2's config to working/ for iteration.

## Troubleshooting

### Training errors

**Import errors after reorganization:**
```
ModuleNotFoundError: No module named 'shared_utils'
```
→ Make sure you're using `../../framework/train.py` (not `../../train.py`)

**Dataset not found:**
```
FileNotFoundError: Dataset NPZ file not found
```
→ Check `dataset.config` path is correct (relative to working/)

### Performance issues

**CV much lower than expected:**
- Check if using correct dataset (100 vs 1020 samples)
- Try different models (SVM often better than Logistic)
- Examine train_accuracy - if also low, data may be difficult

**Perfect test accuracy (100%):**
- Check cv_mean - if much lower, likely overfitting
- Verify using new framework (old framework had this bug)
- Small test sets can occasionally get perfect by chance

### Metrics confusion

**Old checkpoints show `nan` in compare:**
- Old checkpoints use different field names
- Only compare new checkpoints (post-framework update)
- See docs/TROUBLESHOOTING.md for migration guide

## Common Workflows

### Hyperparameter Tuning

```bash
# 1. Try baseline
nano working/config.yaml  # Set model_type: logistic
uv run python ../../framework/train.py
cd .. && uv run python ../framework/checkpoint.py --name "baseline" --message "Logistic baseline"

# 2. Try Random Forest
cd working && nano config.yaml  # Change to random_forest
uv run python ../../framework/train.py
cd .. && uv run python ../framework/checkpoint.py --name "rf" --message "Random forest attempt"

# 3. Try SVM with different kernels
# ... repeat for rbf, linear, poly

# 4. Compare all
uv run python ../framework/compare.py
```

### Dataset Comparison

Train same model on different datasets:
```bash
# Train on d1
cd d1_embedding_to_feature/working
uv run python ../../framework/train.py

# Train on d2
cd ../../d2_hidden_to_feature/working
uv run python ../../framework/train.py

# Train on d3
cd ../../d3_hidden_to_concept/working
uv run python ../../framework/train.py

# Compare across datasets
cd ../..
uv run python framework/compare.py --all
```

## Next Steps

- See **README.md** for conceptual overview and motivation
- See **docs/API_REFERENCE.md** for detailed API documentation
- See **docs/DEVELOPMENT.md** for development workflow
- See **docs/TROUBLESHOOTING.md** for common issues
