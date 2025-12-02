"""
Shared utilities for s5 classifier training.

Provides functions for:
- Loading datasets from dataset.config files
- Computing regression and classification metrics
- Saving/loading results and metadata
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix, roc_auc_score, log_loss
)


def find_dataset_config(start_path: Path) -> Path:
    """
    Find dataset.config by searching current directory and parent.

    Args:
        start_path: Starting directory to search from

    Returns:
        Path to dataset.config

    Raises:
        FileNotFoundError: If dataset.config not found
    """
    start_path = Path(start_path)

    # Check current directory
    if (start_path / "dataset.config").exists():
        return start_path / "dataset.config"

    # Check parent directory
    if (start_path.parent / "dataset.config").exists():
        return start_path.parent / "dataset.config"

    raise FileNotFoundError(
        f"dataset.config not found in {start_path} or parent directory"
    )


def load_dataset(dataset_config_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load dataset from NPZ file specified in dataset.config.

    Args:
        dataset_config_path: Path to dataset.config file. If None, searches current/parent dirs.

    Returns:
        (X, y, metadata) tuple where:
        - X: Input features [n_samples, n_features]
        - y: Target values [n_samples]
        - metadata: Dict with dataset information
    """
    if dataset_config_path is None:
        dataset_config_path = find_dataset_config(Path.cwd())
    else:
        dataset_config_path = Path(dataset_config_path)

    # Read dataset path from config file
    with open(dataset_config_path, 'r') as f:
        npz_path = f.read().strip()

    # Resolve relative paths
    npz_path = Path(npz_path)
    if not npz_path.is_absolute():
        npz_path = dataset_config_path.parent / npz_path

    npz_path = npz_path.resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ file not found: {npz_path}")

    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']

    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith('meta_'):
            val = data[key]
            # Try to convert numpy arrays to Python types
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    metadata[key] = val.item()
                else:
                    metadata[key] = val.tolist()
            else:
                metadata[key] = val

    metadata['dataset_path'] = str(npz_path)
    metadata['n_samples'] = len(X)
    metadata['n_features'] = X.shape[1]

    # Compute dataset hash for reproducibility tracking
    metadata['dataset_hash'] = compute_file_hash(npz_path)

    return X, y, metadata


def compute_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """Compute hash of a file for version tracking."""
    file_path = Path(file_path)
    hash_obj = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()[:8]  # First 8 chars


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dict with test_mae, test_mse, test_rmse, test_r2, pearson_r, pearson_p
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # R² score (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(y_true, y_pred)

    return {
        'test_mae': float(mae),
        'test_mse': float(mse),
        'test_rmse': float(rmse),
        'test_r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for ROC-AUC)

    Returns:
        Dict with test_accuracy, test_f1, confusion_matrix, n_classes, test_roc_auc (if proba provided)
    """
    accuracy = accuracy_score(y_true, y_pred)

    # F1 score (weighted for multi-class)
    n_classes = len(np.unique(y_true))
    f1 = f1_score(y_true, y_pred, average='weighted' if n_classes > 2 else 'binary')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'test_accuracy': float(accuracy),
        'test_f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'n_classes': int(n_classes),
    }

    # ROC-AUC if probabilities provided
    if y_proba is not None:
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            metrics['test_roc_auc'] = float(roc_auc)
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")

        # Cross-entropy loss (log loss)
        try:
            cross_entropy = log_loss(y_true, y_proba)
            metrics['test_cross_entropy_loss'] = float(cross_entropy)
        except Exception as e:
            print(f"Warning: Could not compute cross-entropy loss: {e}")

    return metrics


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """
    Save results to JSON file.

    Args:
        results: Dictionary of results
        output_path: Where to save JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_metadata(
    dataset_path: str,
    dataset_hash: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    version_name: str = "",
    changes_from_previous: str = "",
) -> Dict[str, Any]:
    """
    Create metadata JSON for a checkpoint version.

    Args:
        dataset_path: Path to dataset used
        dataset_hash: Hash of dataset file
        config: Model configuration
        results: Training results
        version_name: Name of version (e.g., "v0_baseline")
        changes_from_previous: Description of changes from previous version

    Returns:
        Metadata dict ready to be saved as JSON
    """
    from datetime import datetime

    metadata = {
        'version': version_name,
        'created_at': datetime.now().isoformat(),
        'dataset_path': dataset_path,
        'dataset_hash': dataset_hash,
        'dataset_samples': results.get('n_samples'),
        'model_type': config.get('model_type'),
        'hyperparams': config.get('hyperparams', {}),
    }

    # Add key metrics
    if 'accuracy' in results:
        metadata['accuracy'] = results['accuracy']
    if 'mae' in results:
        metadata['mae'] = results['mae']
    if 'r2' in results:
        metadata['r2'] = results['r2']
    if 'training_time_sec' in results:
        metadata['training_time_sec'] = results['training_time_sec']
    if 'cv_score' in results:
        metadata['cv_score'] = results['cv_score']

    if changes_from_previous:
        metadata['changes_from_previous'] = changes_from_previous

    return metadata


def get_next_version_number(parent_dir: Path) -> int:
    """
    Find the next available version number in a directory.

    Scans for vX_* directories and returns next version number.
    """
    parent_dir = Path(parent_dir)
    versions = []

    for item in parent_dir.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            try:
                version_num = int(item.name.split('_')[0][1:])
                versions.append(version_num)
            except (ValueError, IndexError):
                pass

    return max(versions, default=-1) + 1


def get_latest_version(parent_dir: Path) -> Optional[Path]:
    """
    Get path to latest version directory.

    Returns None if no versions exist.
    """
    parent_dir = Path(parent_dir)
    latest_num = -1
    latest_dir = None

    for item in parent_dir.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            try:
                version_num = int(item.name.split('_')[0][1:])
                if version_num > latest_num:
                    latest_num = version_num
                    latest_dir = item
            except (ValueError, IndexError):
                pass

    return latest_dir
