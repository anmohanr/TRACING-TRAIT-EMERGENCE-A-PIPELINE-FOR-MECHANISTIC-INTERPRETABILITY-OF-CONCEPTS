# API Reference

Complete API documentation for s5_classifier_training modules.

**Last updated:** 2025-11-01 (Framework reorganization)

## Table of Contents
- [framework.shared_utils](#frameworkshared_utils)
- [framework.train](#frameworktrain)
- [framework.checkpoint](#frameworkcheckpoint)
- [framework.compare](#frameworkcompare)
- [framework.new_experiment](#frameworknew_experiment)

---

## framework.shared_utils

Core utility functions for data loading, metrics, and metadata management.

**Import:** `from framework.shared_utils import ...`

### find_dataset_config

```python
find_dataset_config(start_path: Path) -> Path
```

Find dataset.config file by searching current directory and parent.

**Parameters:**
- `start_path` (Path): Starting directory to search from

**Returns:**
- Path: Path to dataset.config file

**Raises:**
- `FileNotFoundError`: If dataset.config not found in current or parent directory

**Example:**
```python
from pathlib import Path
from framework.shared_utils import find_dataset_config

config_path = find_dataset_config(Path.cwd())
print(f"Found config at: {config_path}")
```

---

### load_dataset

```python
load_dataset(dataset_config_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
```

Load dataset from NPZ file specified in dataset.config.

**Parameters:**
- `dataset_config_path` (Optional[Path]): Path to dataset.config file. If None, searches current/parent dirs

**Returns:**
- Tuple containing:
  - `X` (np.ndarray): Input features [n_samples, n_features]
  - `y` (np.ndarray): Target values [n_samples]
  - `metadata` (Dict[str, Any]): Dataset metadata including path, hash, and any meta_ fields

**Raises:**
- `FileNotFoundError`: If dataset.config or NPZ file not found
- `ValueError`: If NPZ file format is invalid

**Example:**
```python
from framework.shared_utils import load_dataset

# Auto-find dataset.config
X, y, metadata = load_dataset()
print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"Dataset hash: {metadata['dataset_hash']}")

# Specify config path
X, y, metadata = load_dataset(Path('custom.config'))
```

---

### compute_file_hash

```python
compute_file_hash(file_path: Path, algorithm: str = 'md5') -> str
```

Compute hash of a file for version tracking.

**Parameters:**
- `file_path` (Path): Path to file to hash
- `algorithm` (str): Hash algorithm ('md5' or 'sha256')

**Returns:**
- str: First 8 characters of file hash

**Example:**
```python
from framework.shared_utils import compute_file_hash

hash_val = compute_file_hash(Path('dataset.npz'))
print(f"Dataset hash: {hash_val}")  # e.g., "a1b2c3d4"
```

---

### compute_regression_metrics

```python
compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]
```

Compute standard regression metrics.

**Parameters:**
- `y_true` (np.ndarray): Ground truth values
- `y_pred` (np.ndarray): Predicted values

**Returns:**
- Dict with metrics:
  - `test_mae`: Mean Absolute Error
  - `test_mse`: Mean Squared Error
  - `test_rmse`: Root Mean Squared Error
  - `test_r2`: R² coefficient of determination
  - `pearson_r`: Pearson correlation coefficient
  - `pearson_p`: Pearson p-value

**Example:**
```python
from framework.shared_utils import compute_regression_metrics

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

metrics = compute_regression_metrics(y_true, y_pred)
print(f"MAE: {metrics['test_mae']:.4f}")
print(f"R²: {metrics['test_r2']:.4f}")
```

---

### compute_classification_metrics

```python
compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]
```

Compute standard classification metrics.

**Parameters:**
- `y_true` (np.ndarray): Ground truth labels
- `y_pred` (np.ndarray): Predicted labels
- `y_proba` (Optional[np.ndarray]): Prediction probabilities for ROC-AUC

**Returns:**
- Dict with metrics:
  - `test_accuracy`: Overall accuracy
  - `test_f1`: F1 score (weighted for multi-class)
  - `confusion_matrix`: Confusion matrix as list
  - `n_classes`: Number of classes
  - `roc_auc` (if y_proba provided): ROC-AUC score

**Example:**
```python
from framework.shared_utils import compute_classification_metrics

y_true = np.array([0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 0])
y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])

metrics = compute_classification_metrics(y_true, y_pred, y_proba)
print(f"Accuracy: {metrics['test_accuracy']:.2%}")
print(f"F1: {metrics['test_f1']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
```

---

### save_results

```python
save_results(results: Dict[str, Any], output_path: Path) -> None
```

Save results dictionary to JSON file.

**Parameters:**
- `results` (Dict[str, Any]): Results dictionary
- `output_path` (Path): Where to save JSON file

**Example:**
```python
from framework.shared_utils import save_results

results = {
    'cv_mean': 0.8456,
    'cv_std': 0.0119,
    'test_accuracy': 0.8578,
    'training_time_sec': 1.23,
    'model_type': 'SVC'
}

save_results(results, Path('results.json'))
```

---

### load_results

```python
load_results(results_path: Path) -> Dict[str, Any]
```

Load results from JSON file.

**Parameters:**
- `results_path` (Path): Path to results JSON file

**Returns:**
- Dict[str, Any]: Results dictionary

**Example:**
```python
from framework.shared_utils import load_results

results = load_results(Path('v0_baseline/results.json'))
print(f"CV Mean: {results['cv_mean']:.4f}")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

---

### create_metadata

```python
create_metadata(
    dataset_path: str,
    dataset_hash: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    version_name: str = "",
    changes_from_previous: str = ""
) -> Dict[str, Any]
```

Create metadata JSON for a checkpoint version.

**Parameters:**
- `dataset_path` (str): Path to dataset used
- `dataset_hash` (str): Hash of dataset file
- `config` (Dict): Model configuration
- `results` (Dict): Training results
- `version_name` (str): Name of version (e.g., "v0_baseline")
- `changes_from_previous` (str): Description of changes

**Returns:**
- Dict[str, Any]: Metadata dictionary ready for JSON serialization

**Example:**
```python
from framework.shared_utils import create_metadata

metadata = create_metadata(
    dataset_path='data.npz',
    dataset_hash='a1b2c3d4',
    config={'model_type': 'svm', 'hyperparams': {'C': 1.0, 'kernel': 'rbf'}},
    results={'cv_mean': 0.8456, 'test_accuracy': 0.8578, 'n_samples_total': 1020},
    version_name='v1_svm_rbf',
    changes_from_previous='Switched from Logistic to SVM-RBF'
)
```

---

### get_next_version_number

```python
get_next_version_number(parent_dir: Path) -> int
```

Find the next available version number in a directory.

**Parameters:**
- `parent_dir` (Path): Directory to scan for version folders

**Returns:**
- int: Next version number (0 if no versions exist)

**Example:**
```python
from framework.shared_utils import get_next_version_number

next_num = get_next_version_number(Path('d1_embedding_to_feature'))
print(f"Next version will be v{next_num}")
```

---

### get_latest_version

```python
get_latest_version(parent_dir: Path) -> Optional[Path]
```

Get path to the latest version directory.

**Parameters:**
- `parent_dir` (Path): Directory containing version folders

**Returns:**
- Optional[Path]: Path to latest version or None if no versions exist

**Example:**
```python
from framework.shared_utils import get_latest_version

latest = get_latest_version(Path('d1_embedding_to_feature'))
if latest:
    print(f"Latest version: {latest.name}")
else:
    print("No versions found")
```

---

## framework.train

Main training script functions.

**Import:** `from framework.train import ...`

### load_config

```python
load_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]
```

Load configuration from YAML file.

**Parameters:**
- `config_path` (Path): Path to config.yaml file

**Returns:**
- Dict[str, Any]: Configuration dictionary

**Example:**
```python
from framework.train import load_config

config = load_config(Path('config.yaml'))
print(f"Model type: {config['model_type']}")
```

---

### infer_task_type

```python
infer_task_type(y: np.ndarray) -> str
```

Automatically determine if task is classification or regression.

**Parameters:**
- `y` (np.ndarray): Target values

**Returns:**
- str: 'classification' or 'regression'

**Logic:**
- Classification if ≤10 unique values and all integers
- Regression otherwise

**Example:**
```python
from framework.train import infer_task_type

y_class = np.array([0, 1, 2, 0, 1])
print(infer_task_type(y_class))  # 'classification'

y_reg = np.array([1.5, 2.8, 0.3])
print(infer_task_type(y_reg))  # 'regression'
```

---

### create_model

```python
create_model(config: Dict[str, Any], task_type: str) -> Any
```

Create sklearn model instance based on configuration.

**Parameters:**
- `config` (Dict): Configuration with 'model_type' and 'hyperparams'
- `task_type` (str): 'classification' or 'regression'

**Returns:**
- Sklearn model instance

**Supported Models:**
- `'logistic'`: LogisticRegression (classification) or Ridge (regression)
- `'random_forest'`: RandomForestClassifier or RandomForestRegressor
- `'svm'`: SVC or SVR

**Example:**
```python
from framework.train import create_model

config = {
    'model_type': 'random_forest',
    'hyperparams': {'n_estimators': 100, 'max_depth': 10}
}

model = create_model(config, 'classification')
print(model)  # RandomForestClassifier(n_estimators=100, ...)
```

---

### train_and_evaluate

```python
train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    task_type: str,
    cv_folds: int = 5,
    test_split: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]
```

Train model with cross-validation on training set and evaluate on held-out test set.

**Parameters:**
- `X` (np.ndarray): Input features
- `y` (np.ndarray): Target values
- `model`: Sklearn model instance
- `task_type` (str): 'classification' or 'regression'
- `cv_folds` (int): Number of cross-validation folds (default: 5)
- `test_split` (float): Fraction for test set, 0.0-1.0 (default: 0.2). If 0, no split.
- `random_state` (int): Random seed for reproducibility (default: 42)

**Returns:**
- Tuple containing:
  - Trained model (fitted on training set only)
  - Results dictionary with metrics:
    - `cv_mean`, `cv_std`: Cross-validation metrics on training set
    - `test_accuracy`, `test_f1`: Test set metrics (classification)
    - `test_mae`, `test_r2`: Test set metrics (regression)
    - `train_accuracy`: Training set sanity check

**Example:**
```python
from sklearn.svm import SVC
from framework.train import train_and_evaluate

X = np.random.randn(1000, 50)
y = np.random.randint(0, 4, 1000)
model = SVC(kernel='rbf', C=1.0)

# Train with default 80/20 split
trained_model, results = train_and_evaluate(
    X, y, model, 'classification',
    cv_folds=5,
    test_split=0.2,
    random_state=42
)

print(f"CV Mean: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
print(f"Train Accuracy: {results['train_accuracy']:.4f}")
```

---

## framework.checkpoint

Functions for creating versioned checkpoints.

### Main function (called via CLI)

```bash
uv run python ../framework/checkpoint.py --name "experiment_name" --message "What changed"
```

**Arguments:**
- `--name` (required): Short name for this version
- `--message`: Description of changes from previous version

**Process:**
1. Finds next version number
2. Creates vX_name directory
3. Copies all files from working/
4. Generates .metadata.json
5. Creates README.md template

**Example:**
```bash
cd d3_hidden_to_concept
uv run python ../framework/checkpoint.py --name "svm_rbf" --message "SVM-RBF kernel, 84.7% CV, 85.8% test"
# Creates v1_svm_rbf/
```

---

## framework.compare

Functions for comparing experiment versions.

### Main function (called via CLI)

```bash
uv run python ../framework/compare.py [options]
```

**Arguments:**
- `--dataset DATASET`: Compare versions in specific dataset (e.g., d1)
- `--versions VERSION [VERSION ...]`: Compare specific versions
- `--all`: Compare all datasets
- `--detailed`: Show detailed deltas between versions

**Examples:**
```bash
# Compare all versions in d3
uv run python ../framework/compare.py --dataset d3

# Compare specific versions
uv run python ../framework/compare.py --versions d1/v0 d1/v1 d3/v0

# Compare all datasets
cd s5_classifier_training
uv run python framework/compare.py --all

# Show detailed changes
uv run python ../framework/compare.py --dataset d3 --detailed
```

---

## framework.new_experiment

Functions for starting new experiments from checkpoints.

### Main function (called via CLI)

```bash
uv run python ../framework/new_experiment.py --from VERSION --note "Plan"
```

**Arguments:**
- `--from` (required): Version to copy from (e.g., 'd1/v1' or 'latest')
- `--note`: Note about what you're trying in this experiment

**Process:**
1. Copies config files from specified version to working/
2. Logs note to experiment_notes.md
3. Does NOT copy model.pkl or results.json

**Example:**
```bash
# Start from specific version
cd d3_hidden_to_concept
uv run python ../framework/new_experiment.py --from v1_svm_rbf --note "Try different SVM kernels"

# Start from latest version
uv run python ../framework/new_experiment.py --from latest --note "Tune C and gamma"
```

---

## Usage Examples

### Complete Workflow Example

```python
from pathlib import Path
from framework.shared_utils import load_dataset, save_results
from framework.train import infer_task_type, create_model, train_and_evaluate

# Load data
X, y, metadata = load_dataset()
print(f"Loaded dataset: {metadata['dataset_path']}")
print(f"Samples: {len(X)}, Features: {X.shape[1]}")

# Determine task type
task_type = infer_task_type(y)
print(f"Task type: {task_type}")

# Create model
config = {
    'model_type': 'svm',
    'hyperparams': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
}
model = create_model(config, task_type)

# Train and evaluate with 80/20 split
trained_model, results = train_and_evaluate(
    X, y, model, task_type,
    cv_folds=5,
    test_split=0.2,
    random_state=42
)

# Save results
results['metadata'] = metadata
save_results(results, Path('results.json'))

print(f"Training complete!")
print(f"CV Mean: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
print(f"Test Accuracy: {results.get('test_accuracy', 'N/A')}")
print(f"Test MAE: {results.get('test_mae', 'N/A')}")
```

### Programmatic Checkpoint Creation

```python
import subprocess
import json
from pathlib import Path

# Train model
subprocess.run(['uv', 'run', 'python', '../../framework/train.py'], cwd='d3_hidden_to_concept/working')

# Load results to check if worth checkpointing
with open('d3_hidden_to_concept/working/results.json') as f:
    results = json.load(f)

# Checkpoint if CV mean exceeds threshold
if results['cv_mean'] > 0.80:
    subprocess.run([
        'uv', 'run', 'python', '../framework/checkpoint.py',
        '--name', 'high_cv',
        '--message', f"CV {results['cv_mean']:.1%}, Test {results['test_accuracy']:.1%}"
    ], cwd='d3_hidden_to_concept')
```

### Custom Metrics Integration

```python
from framework.shared_utils import compute_classification_metrics

def compute_custom_metrics(y_true, y_pred):
    """Add custom business metrics to standard metrics."""

    # Get standard metrics (test_accuracy, test_f1, etc.)
    metrics = compute_classification_metrics(y_true, y_pred)

    # Add custom metrics for binary classification
    if metrics['n_classes'] == 2:
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return metrics
```

---

## Error Handling

All functions follow these error handling patterns:

### File Not Found
```python
try:
    X, y, metadata = load_dataset(Path('missing.config'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Provides helpful message about checking path
```

### Invalid Data Format
```python
try:
    # Will fail if y has wrong shape
    metrics = compute_regression_metrics(y_true, y_pred.reshape(-1, 2))
except ValueError as e:
    print(f"Shape mismatch: {e}")
```

### Configuration Errors
```python
try:
    model = create_model({'model_type': 'invalid'}, 'classification')
except ValueError as e:
    print(f"Unknown model type: {e}")
```

---

## Type Hints

All functions use type hints for clarity:

```python
from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
import numpy as np

def example_function(
    required_param: np.ndarray,
    optional_param: Optional[str] = None,
    config: Dict[str, Any] = {}
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Function with full type hints."""
    pass
```

This API reference covers all public functions in the s5_classifier_training system. For internal implementation details, refer to the source code.