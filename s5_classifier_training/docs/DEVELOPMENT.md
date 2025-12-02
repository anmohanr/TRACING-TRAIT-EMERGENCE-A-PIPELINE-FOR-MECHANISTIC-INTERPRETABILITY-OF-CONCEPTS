# Development Guide

This guide explains how to extend and modify the s5_classifier_training system.

**Last updated:** 2025-11-01 (Framework reorganization - scripts now in `framework/` directory)

## Table of Contents
1. [Adding New Model Types](#adding-new-model-types)
2. [Adding New Metrics](#adding-new-metrics)
3. [Creating New Dataset Types](#creating-new-dataset-types)
4. [Modifying the Workflow](#modifying-the-workflow)
5. [Testing Changes](#testing-changes)
6. [Code Style Guidelines](#code-style-guidelines)
7. [Contributing Workflow](#contributing-workflow)
8. [Debugging Tips](#debugging-tips)

---

## Adding New Model Types

### Step 1: Update framework/train.py

Add your model to the `create_model` function:

```python
# In framework/train.py, around line 70

def create_model(config: Dict[str, Any], task_type: str) -> Any:
    model_type = config.get('model_type', 'logistic')
    hyperparams = config.get('hyperparams', {})

    # ... existing models ...

    # Add your new model type
    elif model_type == 'gradient_boost':
        if task_type == 'classification':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 3),
                random_state=42
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 3),
                random_state=42
            )

    # ... rest of function
```

### Step 2: Create Config Template

Create a config template for your model:

```yaml
# config_gradient_boost.yaml
model_type: gradient_boost

hyperparams:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 3
  min_samples_split: 2
  min_samples_leaf: 1
  subsample: 1.0

training:
  cv_folds: 5
  random_state: 42
```

### Step 3: Handle Special Cases

Some models need special handling:

```python
# For models that need probability predictions
if model_type == 'gradient_boost' and task_type == 'classification':
    # Ensure probability=True equivalent
    model.set_params(loss='log_loss')  # For probability predictions

# For models with different scoring metrics
if model_type == 'gradient_boost':
    scoring = 'neg_log_loss' if task_type == 'classification' else 'neg_mean_squared_error'
```

### Step 4: Test Your Model

```bash
# Create test config
cd d1_embedding_to_feature/working
cat > config.yaml << EOF
model_type: gradient_boost
hyperparams:
  n_estimators: 50
  learning_rate: 0.1
EOF

# Train
uv run python ../../framework/train.py

# Check it worked
cat results.json | grep model_type
```

### Complete Example: Adding XGBoost

```python
# 1. Install dependency
uv add xgboost

# 2. Update framework/train.py
elif model_type == 'xgboost':
    import xgboost as xgb

    if task_type == 'classification':
        n_classes = len(np.unique(y))
        objective = 'binary:logistic' if n_classes == 2 else 'multi:softprob'
        return xgb.XGBClassifier(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 6),
            learning_rate=hyperparams.get('learning_rate', 0.3),
            objective=objective,
            use_label_encoder=False,
            random_state=42
        )
    else:
        return xgb.XGBRegressor(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 6),
            learning_rate=hyperparams.get('learning_rate', 0.3),
            random_state=42
        )

# 3. Create config
# config_xgboost.yaml
model_type: xgboost
hyperparams:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.3

# 4. Train and test
uv run python ../../framework/train.py
```

---

## Adding New Metrics

### Step 1: Update framework/shared_utils.py

Add your metric calculation:

```python
# In framework/shared_utils.py

def compute_additional_metrics(y_true, y_pred, task_type):
    """Compute additional custom metrics."""
    metrics = {}

    if task_type == 'classification':
        # Add precision, recall
        from sklearn.metrics import precision_score, recall_score

        n_classes = len(np.unique(y_true))
        avg = 'binary' if n_classes == 2 else 'weighted'

        metrics['precision'] = precision_score(y_true, y_pred, average=avg)
        metrics['recall'] = recall_score(y_true, y_pred, average=avg)

        # Add Matthews Correlation Coefficient
        from sklearn.metrics import matthews_corrcoef
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    else:  # regression
        # Add median absolute error
        from sklearn.metrics import median_absolute_error
        metrics['median_ae'] = median_absolute_error(y_true, y_pred)

        # Add explained variance
        from sklearn.metrics import explained_variance_score
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)

    return metrics
```

### Step 2: Integrate into framework/train.py

```python
# In framework/train.py, after computing standard metrics

# Compute additional metrics
additional = compute_additional_metrics(y, y_pred, task_type)
results.update(additional)

# Log new metrics
if task_type == 'classification':
    logger.info(f"Precision: {additional['precision']:.4f}")
    logger.info(f"Recall: {additional['recall']:.4f}")
```

### Step 3: Update framework/compare.py Display

```python
# In framework/compare.py, add new metrics to display

cols_to_show = ['version']

# Add new metrics to display
if df['precision'].notna().any():
    cols_to_show.extend(['precision', 'recall'])
if df['mcc'].notna().any():
    cols_to_show.append('mcc')
```

### Example: Adding Custom Business Metric

```python
def compute_business_metrics(y_true, y_pred, costs=None):
    """Compute business-specific metrics."""

    if costs is None:
        # Default cost matrix (false positive costs more)
        costs = {
            'true_positive': -10,   # Gain
            'true_negative': 0,
            'false_positive': 5,     # Cost
            'false_negative': 2
        }

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate total business value
    total_cost = (
        tp * costs['true_positive'] +
        tn * costs['true_negative'] +
        fp * costs['false_positive'] +
        fn * costs['false_negative']
    )

    return {
        'business_value': -total_cost,  # Negate so higher is better
        'false_positive_cost': fp * costs['false_positive'],
        'false_negative_cost': fn * costs['false_negative']
    }
```

---

## Creating New Dataset Types

### Step 1: Create Directory Structure

```bash
# Create new dataset type
mkdir d4_feature_to_feature
mkdir d4_feature_to_feature/working

# Copy template files
cp d1_embedding_to_feature/working/config.yaml d4_feature_to_feature/working/
cp d1_embedding_to_feature/working/experiment_notes.md d4_feature_to_feature/working/
```

### Step 2: Create Dataset Config

```bash
echo "../../../s4_classifier_data_prep/generated_datasets/feature_to_feature.npz" > d4_feature_to_feature/working/dataset.config
```

### Step 3: Document the New Dataset Type

Update ARCHITECTURE.md:

```markdown
| **d4_feature_to_feature** | Regression | SAE features → Different SAE features | Can features predict other features? |
```

### Step 4: Generate Compatible Data

In s4_classifier_data_prep, create generation strategy:

```python
def generate_feature_to_feature(
    final_token_data: Dict[str, np.ndarray],
    source_layer: int,
    source_features: List[int],
    target_layer: int,
    target_feature: int,
    normalize: str = 'standardize'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Dataset: Multiple SAE features → Single SAE feature

    Input: Activations of multiple features from source layer
    Output: Activation of single feature from target layer
    """
    # Extract source features as input
    X = get_sae_activations(final_token_data, source_layer)[:, source_features]

    # Extract target feature as output
    y = get_sae_activations(final_token_data, target_layer)[:, target_feature]

    # Normalize
    X_norm, norm_params = normalize_features(X, method=normalize)

    metadata = {
        'meta_strategy': 'feature-to-feature',
        'meta_source_layer': source_layer,
        'meta_source_features': source_features,
        'meta_target_layer': target_layer,
        'meta_target_feature': target_feature,
        **norm_params
    }

    return X_norm, y, metadata
```

### Complete Example: Adding Ensemble Dataset

```python
# New dataset type: Ensemble predictions
# d5_ensemble/: Combines predictions from d1, d2, d3

# Step 1: Create structure
mkdir d5_ensemble
mkdir d5_ensemble/working

# Step 2: Create ensemble data generator
def create_ensemble_dataset():
    """Combine predictions from other models as features."""

    # Load trained models
    import pickle

    with open('d1/v0_baseline/model.pkl', 'rb') as f:
        model1 = pickle.load(f)
    with open('d2/v0_baseline/model.pkl', 'rb') as f:
        model2 = pickle.load(f)

    # Load original data
    data1 = np.load('d1_data.npz')
    data2 = np.load('d2_data.npz')

    # Get predictions
    pred1 = model1.predict(data1['X']).reshape(-1, 1)
    pred2 = model2.predict(data2['X']).reshape(-1, 1)

    # Combine as new features
    X_ensemble = np.hstack([pred1, pred2])
    y = data1['y']  # Use same target

    # Save
    np.savez('d5_ensemble/ensemble_features.npz',
             X=X_ensemble, y=y,
             meta_strategy='ensemble',
             meta_base_models=['d1/v0', 'd2/v0'])

# Step 3: Train ensemble
cd d5_ensemble/working
echo "../ensemble_features.npz" > dataset.config
uv run python ../../train.py
```

---

## Modifying the Workflow

### Adding Pre-processing Step

```python
# In framework/train.py, before training

def preprocess_data(X, y, config):
    """Apply preprocessing based on config."""

    preproc = config.get('preprocessing', {})

    if preproc.get('scale'):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if preproc.get('pca'):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=preproc.get('pca_components', 100))
        X = pca.fit_transform(X)

    if preproc.get('feature_selection'):
        from sklearn.feature_selection import SelectKBest
        selector = SelectKBest(k=preproc.get('n_features', 100))
        X = selector.fit_transform(X, y)

    return X, y

# In main():
X, y = preprocess_data(X, y, config)
```

### Adding Post-processing Step

```python
def postprocess_predictions(y_pred, config):
    """Post-process predictions."""

    postproc = config.get('postprocessing', {})

    if postproc.get('round'):
        y_pred = np.round(y_pred)

    if postproc.get('clip'):
        y_pred = np.clip(y_pred,
                         postproc.get('clip_min', 0),
                         postproc.get('clip_max', 100))

    if postproc.get('threshold'):
        threshold = postproc.get('threshold_value', 0.5)
        y_pred = (y_pred > threshold).astype(int)

    return y_pred
```

### Adding Hyperparameter Tuning

```python
# New file: tune.py

from sklearn.model_selection import GridSearchCV
import yaml
import numpy as np
from framework.shared_utils import load_dataset
from framework.train import create_model, infer_task_type

def tune_hyperparameters(config_path='config.yaml'):
    """Tune hyperparameters using grid search."""

    # Load base config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load data
    X, y, _ = load_dataset()
    task_type = infer_task_type(y)

    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [100, 500, 1000, 2000]
    }

    # Create base model
    base_model = create_model(config, task_type)

    # Grid search
    scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best score: {grid.best_score_}")

    # Update config with best params
    config['hyperparams'].update(grid.best_params_)

    with open('config_tuned.yaml', 'w') as f:
        yaml.dump(config, f)

    return grid.best_params_

if __name__ == '__main__':
    tune_hyperparameters()
```

---

## Testing Changes

### Unit Tests Template

```python
# test_shared_utils.py

import pytest
import numpy as np
import tempfile
from pathlib import Path
from framework.shared_utils import load_dataset, compute_file_hash
from framework.train import infer_task_type

def test_load_dataset():
    """Test dataset loading."""
    # Create temporary NPZ
    with tempfile.NamedTemporaryFile(suffix='.npz') as tmp:
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        np.savez(tmp.name, X=X, y=y, meta_test='value')

        # Create config pointing to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.config') as config:
            config.write(tmp.name)
            config.flush()

            # Load dataset
            X_loaded, y_loaded, metadata = load_dataset(Path(config.name))

            assert X_loaded.shape == (10, 5)
            assert y_loaded.shape == (10,)
            assert metadata['meta_test'] == 'value'

def test_infer_task_type():
    """Test task type inference."""
    # Classification
    y_class = np.array([0, 1, 2, 0, 1, 2])
    assert infer_task_type(y_class) == 'classification'

    # Regression
    y_reg = np.array([1.5, 2.8, 0.3, 9.1])
    assert infer_task_type(y_reg) == 'regression'

    # Edge case: 10 unique integers
    y_edge = np.arange(10)
    assert infer_task_type(y_edge) == 'classification'

    # Edge case: 11 unique integers
    y_edge2 = np.arange(11)
    assert infer_task_type(y_edge2) == 'regression'

def test_compute_metrics():
    """Test metric computation."""
    from framework.shared_utils import compute_regression_metrics

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert 'test_mae' in metrics
    assert 'test_mse' in metrics
    assert 'test_r2' in metrics
    assert metrics['test_mae'] < 0.3  # Should be close

# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Integration Tests

```python
# test_integration.py

import subprocess
import json
import tempfile
from pathlib import Path

def test_full_workflow():
    """Test complete train → checkpoint → compare workflow."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Setup directory structure
        d1 = tmpdir / 'd1_test'
        working = d1 / 'working'
        working.mkdir(parents=True)

        # Create test data
        import numpy as np
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 2, 50)
        data_file = tmpdir / 'test_data.npz'
        np.savez(data_file, X=X, y=y)

        # Create config
        config = working / 'config.yaml'
        config.write_text("""
model_type: logistic
hyperparams:
  C: 1.0
  max_iter: 100
""")

        # Create dataset.config
        dataset_config = working / 'dataset.config'
        dataset_config.write_text(str(data_file))

        # Train
        result = subprocess.run(
            ['uv', 'run', 'python', str(Path('../../framework/train.py'))],
            cwd=working,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

        # Check results exist
        assert (working / 'model.pkl').exists()
        assert (working / 'results.json').exists()

        # Checkpoint
        result = subprocess.run(
            ['uv', 'run', 'python', str(Path('../framework/checkpoint.py')),
             '--name', 'test', '--message', 'Test checkpoint'],
            cwd=d1,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

        # Check checkpoint created
        v0 = d1 / 'v0_test'
        assert v0.exists()
        assert (v0 / 'model.pkl').exists()
        assert (v0 / '.metadata.json').exists()

def test_error_handling():
    """Test error conditions."""

    # Test with missing dataset
    result = subprocess.run(
        ['uv', 'run', 'python', '../../framework/train.py'],
        capture_output=True,
        text=True,
        cwd='d1_embedding_to_feature/working'
    )
    assert result.returncode != 0
    assert 'FileNotFoundError' in result.stderr
```

### Performance Tests

```python
def test_performance():
    """Test training time scales appropriately."""
    import time

    times = []
    sizes = [100, 500, 1000]

    for n in sizes:
        X = np.random.randn(n, 1000)
        y = np.random.randint(0, 2, n)

        start = time.time()
        model = LogisticRegression()
        model.fit(X, y)
        elapsed = time.time() - start

        times.append(elapsed)
        print(f"n={n}: {elapsed:.3f}s")

    # Check roughly linear scaling
    ratio = times[2] / times[0]
    assert ratio < 15  # Should not be more than 15x slower for 10x data
```

---

## Code Style Guidelines

### Python Style

Follow PEP 8 with these additions:

```python
# Good: Clear variable names
dataset_path = Path('data.npz')
cross_validation_scores = [0.8, 0.85, 0.82]

# Bad: Unclear abbreviations
dp = Path('data.npz')
cvs = [0.8, 0.85, 0.82]

# Good: Type hints for clarity
def load_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ...

# Good: Docstrings with Args/Returns
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metric names to values
    """
    ...

# Good: Constants at module level
DEFAULT_CV_FOLDS = 5
MAX_ITERATIONS = 1000

# Good: Error messages that help
if not data_file.exists():
    raise FileNotFoundError(
        f"Dataset file not found: {data_file}\n"
        f"Please check dataset.config points to valid file"
    )
```

### YAML Style

```yaml
# Good: Clear structure with comments
model_type: random_forest  # Model selection

hyperparams:
  n_estimators: 100        # Number of trees
  max_depth: 10            # Maximum tree depth
  min_samples_split: 2     # Minimum samples to split node

# Bad: No organization
model_type: random_forest
n_estimators: 100
max_depth: 10
cv_folds: 5
```

### Commit Messages

```bash
# Good: Clear, specific
git commit -m "Add XGBoost model support with hyperparameter tuning"
git commit -m "Fix memory error in d3 training with high-dimensional data"
git commit -m "Document new metrics in API_REFERENCE.md"

# Bad: Vague
git commit -m "Update files"
git commit -m "Fix bug"
git commit -m "Changes"
```

---

## Contributing Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/final_proj.git
cd final_proj/s5_classifier_training
```

### 2. Create Feature Branch

```bash
git checkout -b feature/add-xgboost-support
```

### 3. Make Changes

1. Write code following style guide
2. Add tests if applicable
3. Update documentation
4. Test locally

### 4. Test Your Changes

```bash
# Run in test environment
cd d1_embedding_to_feature/working
uv run python ../../framework/train.py

# Run tests if available
pytest tests/

# Check style (if configured)
flake8 framework/*.py
black --check framework/*.py
```

### 5. Document Changes

Update relevant docs:
- README.md if adding user-facing features
- ARCHITECTURE.md if changing design
- This file if adding development features

### 6. Commit and Push

```bash
git add -A
git commit -m "Add XGBoost model support

- Add XGBoost to create_model function
- Create config template
- Add tests for XGBoost
- Update documentation"

git push origin feature/add-xgboost-support
```

### 7. Create Pull Request

Include in PR description:
- What changed
- Why it changed
- How to test
- Any breaking changes

---

## Debugging Tips

### 1. Add Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

def problematic_function(data):
    logger.debug(f"Input shape: {data.shape}")
    logger.debug(f"Input type: {type(data)}")
    logger.debug(f"First few values: {data[:5]}")

    # ... processing ...

    logger.debug(f"Output shape: {result.shape}")
    return result
```

### 2. Use Interactive Debugger

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or better with IPython
from IPython import embed; embed()

# In problematic section
try:
    result = risky_operation()
except Exception as e:
    import pdb; pdb.post_mortem()  # Debug at error point
```

### 3. Profile Performance

```python
import cProfile
import pstats

def profile_training():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run training
    train_model()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 time consumers

# Or use line_profiler for line-by-line
# pip install line_profiler
# kernprof -l -v train.py
```

### 4. Memory Profiling

```python
# pip install memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_array = np.random.randn(10000, 10000)
    result = large_array @ large_array.T
    return result

# Run with: python -m memory_profiler script.py
```

### 5. Check Data Assumptions

```python
def validate_assumptions(X, y):
    """Check all assumptions about data."""
    checks = []

    # Shape checks
    checks.append(('X is 2D', X.ndim == 2))
    checks.append(('y is 1D', y.ndim == 1))
    checks.append(('Same length', len(X) == len(y)))

    # Value checks
    checks.append(('No NaN in X', not np.any(np.isnan(X))))
    checks.append(('No inf in X', not np.any(np.isinf(X))))
    checks.append(('y has valid range', y.min() >= 0 and y.max() <= 3))

    # Type checks
    checks.append(('X is numeric', np.issubdtype(X.dtype, np.number)))

    for name, passed in checks:
        status = '✓' if passed else '✗'
        print(f"{status} {name}")

    return all(passed for _, passed in checks)
```

## Common Development Patterns

### Pattern: Adding Optional Dependencies

```python
# Lazy import for optional dependencies
def use_special_model():
    try:
        import special_library
    except ImportError:
        raise ImportError(
            "special_library required for this model. "
            "Install with: uv pip install special_library"
        )

    return special_library.Model()
```

### Pattern: Backward Compatibility

```python
def load_checkpoint(version_dir):
    """Load checkpoint with backward compatibility."""

    # Try new format first
    metadata_file = version_dir / '.metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)

    # Fall back to old format
    results_file = version_dir / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
            # Convert old format to new
            return {
                'accuracy': results.get('test_accuracy'),
                'model_type': results.get('model', {}).get('type'),
                # ... mapping ...
            }

    raise ValueError(f"No recognized format in {version_dir}")
```

### Pattern: Feature Flags

```python
# In config.yaml
experimental:
  use_new_algorithm: false
  enable_caching: true

# In code
def train_model(config):
    if config.get('experimental', {}).get('use_new_algorithm', False):
        return train_new_algorithm()
    else:
        return train_standard()
```

This development guide should help you extend and improve the s5_classifier_training system!