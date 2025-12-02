# Data Format Specification

This document specifies the exact data format requirements for s5_classifier_training.

## Table of Contents
1. [NPZ File Format](#npz-file-format)
2. [Required Arrays](#required-arrays)
3. [Optional Metadata](#optional-metadata)
4. [Task Type Inference](#task-type-inference)
5. [Creating Compatible Datasets](#creating-compatible-datasets)
6. [Validation](#validation)
7. [Examples](#examples)

---

## NPZ File Format

The system expects NumPy NPZ files (compressed numpy archive format).

### File Structure
```
dataset.npz
├── X                  # Input features (required)
├── y                  # Target values (required)
├── meta_strategy      # Dataset generation strategy (optional)
├── meta_input_layers  # Which layers were used (optional)
├── meta_n_samples     # Number of samples (optional)
├── norm_mean         # Normalization parameters (optional)
├── norm_std          # Normalization parameters (optional)
└── ...               # Other metadata with 'meta_' prefix
```

### Loading Example
```python
import numpy as np

# Load NPZ file
data = np.load('dataset.npz', allow_pickle=True)

# Access arrays
X = data['X']
y = data['y']

# List all arrays
print(data.files)
# Output: ['X', 'y', 'meta_strategy', ...]
```

---

## Required Arrays

### X: Input Features

**Type**: `numpy.ndarray`
**Shape**: `(n_samples, n_features)`
**Dtype**: Usually `float32` or `float64`

**Requirements**:
- 2D array (even if single feature, shape must be (n, 1))
- No NaN or infinite values
- Typically normalized (but not required)

**Examples**:
```python
# Valid X arrays
X_valid_1 = np.random.randn(100, 2304)     # 100 samples, 2304 features
X_valid_2 = np.random.randn(50, 1)         # 50 samples, 1 feature
X_valid_3 = np.zeros((200, 27648))         # 200 samples, 27648 features

# Invalid X arrays
X_invalid_1 = np.random.randn(100)         # ❌ 1D array
X_invalid_2 = np.random.randn(10, 20, 30)  # ❌ 3D array
X_invalid_3 = np.array([[1, np.nan], [2, 3]]) # ❌ Contains NaN
```

### y: Target Values

**Type**: `numpy.ndarray`
**Shape**: `(n_samples,)` or `(n_samples, 1)`
**Dtype**:
- Classification: `int32`, `int64`, or convertible to int
- Regression: `float32` or `float64`

**Requirements**:
- 1D array or 2D column vector
- Length must match X.shape[0]
- No NaN values
- For classification: discrete values (typically 0, 1, 2, ...)
- For regression: continuous values

**Examples**:
```python
# Classification targets
y_class_binary = np.array([0, 1, 0, 1, 1])     # Binary
y_class_multi = np.array([0, 1, 2, 3, 2, 1])   # Multi-class (4 classes)

# Regression targets
y_reg_continuous = np.array([1.5, 2.8, 0.3, 9.1])  # Continuous values
y_reg_activations = np.array([0.0, 12.3, 34.8, 2.1]) # SAE activations

# Valid shapes
y_1d = np.array([1, 2, 3])           # Shape: (3,) ✅
y_2d = np.array([[1], [2], [3]])     # Shape: (3, 1) ✅ (will be flattened)

# Invalid
y_invalid_1 = np.array([[1, 2], [3, 4]])  # ❌ 2D with multiple columns
y_invalid_2 = np.array([1, 2, np.nan])    # ❌ Contains NaN
```

---

## Optional Metadata

Metadata fields should have `meta_` prefix. Common metadata:

### Standard Metadata Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `meta_strategy` | str | Dataset generation strategy | "embedding-to-feature" |
| `meta_input_source` | str | Source of input features | "embeddings" |
| `meta_input_layers` | list/array | Which layers were used | [11, 12, 13, 14] |
| `meta_input_dims` | int | Input dimensionality | 2304 |
| `meta_n_samples` | int | Number of samples | 100 |
| `meta_sentence_indices` | array | Map to original sentences | [0, 1, 2, ...] |
| `meta_target_layer` | int | Target layer (for SAE features) | 19 |
| `meta_target_feature` | int | Target feature index | 12924 |

### Normalization Metadata

If data was normalized, include parameters for inverse transform:

| Field | Type | Description | Shape |
|-------|------|-------------|-------|
| `norm_method` | str | Normalization type | "standardize" or "minmax" |
| `norm_mean` | array | Mean for standardization | (n_features,) |
| `norm_std` | array | Std for standardization | (n_features,) |
| `norm_min` | array | Min for minmax scaling | (n_features,) |
| `norm_max` | array | Max for minmax scaling | (n_features,) |

### Creating NPZ with Metadata
```python
import numpy as np

# Generate data
X = np.random.randn(100, 2304)
y = np.array([0, 1, 2, 3] * 25)  # 100 labels

# Create metadata
metadata = {
    'meta_strategy': 'embedding-to-feature',
    'meta_input_layers': [0],  # Embedding layer
    'meta_n_samples': 100,
    'meta_target_layer': 19,
    'meta_target_feature': 12924,
    'norm_method': 'standardize',
    'norm_mean': X.mean(axis=0),
    'norm_std': X.std(axis=0)
}

# Save with metadata
np.savez('dataset.npz', X=X, y=y, **metadata)
```

---

## Task Type Inference

The system automatically determines if a task is classification or regression based on the target variable `y`.

### Inference Rules

```python
def infer_task_type(y):
    unique_vals = np.unique(y)

    # Classification if:
    # 1. 10 or fewer unique values AND
    # 2. All values are integers (or very close to integers)
    if len(unique_vals) <= 10 and np.allclose(y, y.astype(int)):
        return 'classification'
    else:
        return 'regression'
```

### Examples

| y values | Unique count | Type inferred | Reason |
|----------|--------------|---------------|---------|
| [0, 1, 0, 1] | 2 | classification | Binary integers |
| [0, 1, 2, 3] | 4 | classification | ≤10 unique integers |
| [0.0, 1.0, 2.0] | 3 | classification | Integer-like values |
| [0, 1, 2, ..., 9] | 10 | classification | Exactly 10 unique |
| [0, 1, 2, ..., 10] | 11 | regression | >10 unique values |
| [1.5, 2.8, 0.3] | 3 | regression | Non-integer values |
| [0.001, 0.002, ...] | many | regression | Continuous values |

### Edge Cases

**Case 1: Integer regression targets**
```python
# House prices in thousands: [100, 150, 200, 250, ...]
# Will be classified as regression if >10 unique values
```

**Case 2: Few unique continuous values**
```python
# Scores: [0.0, 0.5, 1.0] (only 3 values)
# Will be classified as classification!
# Solution: Add small noise or use more granular scores
```

**Case 3: Ordinal data**
```python
# Ratings: [1, 2, 3, 4, 5]
# Classified as classification (correct for ordinal)
# But could be regression if you want to predict exact rating
```

### Forcing Task Type

To override automatic inference, modify your model selection:

```python
# In train.py or your script
task_type = 'regression'  # Force regression
# task_type = 'classification'  # Force classification

# Don't rely on inference
model = create_model(config, task_type)
```

---

## Creating Compatible Datasets

### Method 1: From Scratch

```python
import numpy as np

def create_dataset(n_samples=100, n_features=2304, task='classification'):
    """Create a compatible dataset for testing."""

    # Generate random features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate targets based on task
    if task == 'classification':
        # 4-class classification
        y = np.random.randint(0, 4, n_samples)
    else:
        # Regression with realistic range
        y = np.random.randn(n_samples) * 10 + 5
        y = y.astype(np.float32)

    # Add metadata
    metadata = {
        'meta_strategy': f'synthetic_{task}',
        'meta_n_samples': n_samples,
        'meta_n_features': n_features,
        'meta_task_type': task
    }

    # Save
    np.savez('synthetic_data.npz', X=X, y=y, **metadata)

    return X, y

# Create classification dataset
X_class, y_class = create_dataset(task='classification')

# Create regression dataset
X_reg, y_reg = create_dataset(task='regression')
```

### Method 2: From Existing Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def convert_csv_to_npz(csv_path, feature_cols, target_col, output_path):
    """Convert CSV data to NPZ format."""

    # Load data
    df = pd.read_csv(csv_path)

    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Create metadata
    metadata = {
        'meta_source_file': csv_path,
        'meta_feature_cols': feature_cols,
        'meta_target_col': target_col,
        'meta_n_samples': len(X),
        'meta_n_features': X.shape[1],
        'norm_method': 'standardize',
        'norm_mean': scaler.mean_,
        'norm_std': scaler.scale_
    }

    # Save
    np.savez(output_path, X=X_normalized, y=y, **metadata)

    print(f"Created {output_path}")
    print(f"  X shape: {X_normalized.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y unique values: {np.unique(y)}")

# Example usage
convert_csv_to_npz(
    'data.csv',
    feature_cols=['feat1', 'feat2', 'feat3'],
    target_col='label',
    output_path='converted_data.npz'
)
```

### Method 3: From s4_classifier_data_prep

The standard way - use s4 scripts:

```bash
cd s4_classifier_data_prep
uv run python generate_datasets.py \
  --strategy embedding-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 19 --target-feature 12924 \
  --output generated_datasets/new_dataset.npz
```

---

## Validation

### Basic Validation Function

```python
def validate_dataset(npz_path):
    """Validate NPZ file format for s5_classifier_training."""

    errors = []
    warnings = []

    try:
        # Load file
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return [f"Cannot load NPZ file: {e}"], []

    # Check required arrays
    if 'X' not in data.files:
        errors.append("Missing required array 'X'")
    if 'y' not in data.files:
        errors.append("Missing required array 'y'")

    if errors:
        return errors, warnings

    X = data['X']
    y = data['y']

    # Validate X
    if X.ndim != 2:
        errors.append(f"X must be 2D, got shape {X.shape}")
    if np.any(np.isnan(X)):
        errors.append("X contains NaN values")
    if np.any(np.isinf(X)):
        errors.append("X contains infinite values")

    # Validate y
    if y.ndim not in [1, 2]:
        errors.append(f"y must be 1D or 2D column, got shape {y.shape}")
    if y.ndim == 2 and y.shape[1] != 1:
        errors.append(f"y as 2D must have 1 column, got {y.shape[1]}")
    if np.any(np.isnan(y)):
        errors.append("y contains NaN values")

    # Check shapes match
    y_len = len(y) if y.ndim == 1 else y.shape[0]
    if X.shape[0] != y_len:
        errors.append(f"X and y length mismatch: {X.shape[0]} != {y_len}")

    # Warnings
    if X.shape[0] < 20:
        warnings.append(f"Very few samples ({X.shape[0]}), may cause CV issues")
    if X.shape[1] > 10000:
        warnings.append(f"High dimensionality ({X.shape[1]} features), may cause memory issues")

    # Check metadata
    meta_fields = [f for f in data.files if f.startswith('meta_')]
    if not meta_fields:
        warnings.append("No metadata fields found (fields starting with 'meta_')")

    return errors, warnings

# Use validation
errors, warnings = validate_dataset('my_dataset.npz')

if errors:
    print("❌ Errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✅ Dataset is valid")

if warnings:
    print("⚠️ Warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

### Advanced Validation

```python
def deep_validate(npz_path):
    """Comprehensive validation with statistics."""

    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']

    print("Dataset Statistics:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  X dtype: {X.dtype}")
    print(f"  y dtype: {y.dtype}")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  X mean: {X.mean():.3f}, std: {X.std():.3f}")
    print(f"  y unique values: {len(np.unique(y))}")

    # Task type inference
    from framework.train import infer_task_type
    task = infer_task_type(y)
    print(f"  Inferred task: {task}")

    if task == 'classification':
        unique, counts = np.unique(y, return_counts=True)
        print(f"  Class distribution:")
        for val, count in zip(unique, counts):
            print(f"    Class {val}: {count} ({count/len(y)*100:.1f}%)")
    else:
        print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  y mean: {y.mean():.3f}, std: {y.std():.3f}")

    # Memory estimate
    memory_mb = (X.nbytes + y.nbytes) / 1024 / 1024
    print(f"  Memory usage: {memory_mb:.1f} MB")

    # Check for issues
    issues = []
    if X.shape[0] < 5:
        issues.append("Too few samples for 5-fold CV")
    if task == 'classification':
        min_class = min(counts)
        if min_class < 2:
            issues.append(f"Class with only {min_class} sample(s)")

    if issues:
        print("\n⚠️ Potential issues:")
        for issue in issues:
            print(f"  - {issue}")
```

---

## results.json Format

After training, the framework saves results in a standardized JSON format.

**Last updated:** 2025-11-01 (Framework reorganization with new metric naming)

### Classification Results Structure

```json
{
  "n_samples_total": 1020,
  "n_samples_train": 816,
  "n_samples_test": 204,
  "n_features": 2304,
  "model_type": "SVC",
  "training_time_sec": 2.45,

  "cv_scores": [0.85, 0.82, 0.85, 0.85, 0.85],
  "cv_mean": 0.8456,
  "cv_std": 0.0119,

  "test_accuracy": 0.8578,
  "test_f1": 0.8307,
  "confusion_matrix": [[45, 3, 1, 2], [2, 38, 4, 3], [1, 5, 20, 1], [3, 2, 1, 69]],
  "n_classes": 4,

  "train_accuracy": 0.9069,

  "config": {
    "model_type": "svm",
    "hyperparams": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    "training": {"cv_folds": 5, "test_split": 0.2, "random_state": 42}
  },

  "metadata": {
    "dataset_path": "../../../s4_classifier_data_prep/generated_datasets/dishonesty_1020_h19_to_concept.npz",
    "dataset_hash": "a1b2c3d4",
    "n_samples": 1020,
    "n_features": 2304,
    "meta_strategy": "hidden-to-concept"
  }
}
```

### Regression Results Structure

```json
{
  "n_samples_total": 100,
  "n_samples_train": 80,
  "n_samples_test": 20,
  "n_features": 2304,
  "model_type": "Ridge",
  "training_time_sec": 0.15,

  "cv_scores": [2.34, 2.89, 2.12, 2.45, 2.67],
  "cv_mean": 2.494,
  "cv_std": 0.283,

  "test_mae": 2.15,
  "test_mse": 6.82,
  "test_rmse": 2.61,
  "test_r2": 0.452,
  "pearson_r": 0.689,
  "pearson_p": 0.001,

  "train_mae": 1.87,
  "train_r2": 0.523,

  "config": {...},
  "metadata": {...}
}
```

### Metric Name Changes (2025-11-01)

The framework was updated with clean metric naming. Old checkpoints use different names:

| Metric Type | Old Name | New Name |
|-------------|----------|----------|
| Cross-validation mean | `accuracy` or `cv_score` | `cv_mean` |
| Cross-validation std | (not tracked) | `cv_std` |
| Test accuracy | `accuracy` (ambiguous) | `test_accuracy` |
| Test F1 score | `f1` | `test_f1` |
| Test MAE | `mae` | `test_mae` |
| Test MSE | `mse` | `test_mse` |
| Test R² | `r2` | `test_r2` |
| Training time | `training_time` | `training_time_sec` |

**Note:** Old checkpoints (v0-v3) may not be directly comparable with new ones due to metric naming differences. The `compare.py` script will show `nan` for metrics that don't exist in old formats.

### Key Metrics Explained

- **cv_mean**: Primary metric for model selection. Computed via cross-validation on training set only.
- **cv_std**: Variance across CV folds. Lower is better (more stable).
- **test_accuracy / test_mae**: Performance on held-out test set. Should be close to cv_mean.
- **train_accuracy / train_mae**: Sanity check on training set. Should be higher than test metrics.
- **cv_mean vs test gap**: If >5-10 percentage points, may indicate overfitting or lucky split.

---

## Examples

### Example 1: Binary Classification Dataset

```python
# Create binary classification data
import numpy as np

n_samples = 200
n_features = 1000

X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, 2, n_samples)  # 0 or 1

# Balance classes
y[:100] = 0
y[100:] = 1
np.random.shuffle(y)

np.savez('binary_classification.npz',
         X=X, y=y,
         meta_task='binary_classification',
         meta_n_classes=2)
```

### Example 2: Multi-class Classification Dataset

```python
# Create 4-class classification (like d3)
n_samples = 100
n_features = 27648  # Like d3

X = np.random.randn(n_samples, n_features).astype(np.float32)

# Imbalanced classes (like real d3 data)
y = np.concatenate([
    np.zeros(28),  # 28% class 0
    np.ones(23),   # 23% class 1
    np.ones(13) * 2,  # 13% class 2
    np.ones(36) * 3   # 36% class 3
]).astype(int)

np.random.shuffle(y)

np.savez('multiclass_imbalanced.npz',
         X=X, y=y,
         meta_task='concept_classification',
         meta_class_labels=['none', 'mild', 'moderate', 'strong'])
```

### Example 3: Regression Dataset

```python
# Create regression data (like d1, d2)
n_samples = 100
n_features = 2304

X = np.random.randn(n_samples, n_features).astype(np.float32)

# SAE activation values (typical range 0-40)
y = np.abs(np.random.randn(n_samples) * 10)
y[y > 40] = 40  # Cap at 40

np.savez('regression_sae.npz',
         X=X, y=y,
         meta_task='sae_prediction',
         meta_target_layer=19,
         meta_target_feature=12924)
```

### Example 4: Loading and Using

```python
# Load and inspect
import numpy as np

data = np.load('dataset.npz', allow_pickle=True)

print("Arrays in file:", data.files)
print("X shape:", data['X'].shape)
print("y shape:", data['y'].shape)

# Check metadata
for key in data.files:
    if key.startswith('meta_'):
        value = data[key]
        if hasattr(value, 'item'):
            value = value.item()
        print(f"{key}: {value}")

# Use in training
X = data['X']
y = data['y']

# Train model...
```

---

## Format Conversion Tools

### From PyTorch Tensors

```python
import torch
import numpy as np

# Convert PyTorch tensors
X_tensor = torch.randn(100, 2304)
y_tensor = torch.randint(0, 4, (100,))

X = X_tensor.numpy()
y = y_tensor.numpy()

np.savez('from_pytorch.npz', X=X, y=y)
```

### From TensorFlow

```python
import tensorflow as tf
import numpy as np

# Convert TF tensors
X_tf = tf.random.normal([100, 2304])
y_tf = tf.random.uniform([100], 0, 4, dtype=tf.int32)

X = X_tf.numpy()
y = y_tf.numpy()

np.savez('from_tensorflow.npz', X=X, y=y)
```

### From Pandas DataFrame

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# Assuming last column is target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

np.savez('from_pandas.npz', X=X, y=y,
         meta_columns=df.columns[:-1].tolist(),
         meta_target_column=df.columns[-1])
```

---

## Best Practices

1. **Always include metadata** - Makes debugging easier
2. **Use descriptive names** - `dishonesty_emb_to_l19_f12924.npz` not `data.npz`
3. **Validate before training** - Run validation function
4. **Document special cases** - If your data has quirks, document them
5. **Version your datasets** - Include version in filename or metadata
6. **Keep raw and processed** - Save both versions
7. **Use compression** - NPZ files are compressed by default
8. **Check memory requirements** - Large datasets may need special handling

---

## Troubleshooting Data Issues

| Issue | Solution |
|-------|----------|
| Wrong task type inferred | Add noise to continuous values or round to integers |
| Memory errors loading | Use memory mapping: `np.load(file, mmap_mode='r')` |
| Slow loading | Ensure NPZ not NPY, use compressed format |
| Can't pickle after loading | Don't use allow_pickle=True unless needed |
| Metadata not loading | Ensure saved with `**dict` not as single array |

This specification ensures your datasets will work correctly with s5_classifier_training!