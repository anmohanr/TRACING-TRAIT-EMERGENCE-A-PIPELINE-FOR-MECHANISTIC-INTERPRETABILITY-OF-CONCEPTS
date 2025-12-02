#!/usr/bin/env python3
"""
Main training script for s5 classifier experiments.

Trains a classifier on dataset specified in dataset.config using hyperparameters
in config.yaml. Works for both regression and classification tasks.

Usage:
    cd d1_embedding_to_feature/working
    uv run python ../../framework/train.py

    Or with custom config:
    uv run python ../../framework/train.py --config custom_config.yaml
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for framework imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, accuracy_score

from framework.shared_utils import (
    load_dataset, compute_regression_metrics, compute_classification_metrics,
    save_results
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def infer_task_type(y: np.ndarray) -> str:
    """
    Infer whether task is classification or regression.

    Returns: 'classification' or 'regression'
    """
    # Check if y contains only integers in small range (classification indicator)
    unique_vals = np.unique(y)

    if len(unique_vals) <= 10 and np.allclose(y, y.astype(int)):
        return 'classification'
    else:
        return 'regression'


def create_model(config: Dict[str, Any], task_type: str) -> Any:
    """
    Create model instance based on configuration.

    Args:
        config: Configuration dict with 'model_type' and 'hyperparams'
        task_type: 'classification' or 'regression'

    Returns:
        Sklearn model instance
    """
    model_type = config.get('model_type', 'logistic')
    hyperparams = config.get('hyperparams', {})

    logger.info(f"Creating {model_type} model for {task_type} task")

    if model_type == 'logistic':
        if task_type == 'classification':
            return LogisticRegression(
                C=hyperparams.get('C', 1.0),
                max_iter=hyperparams.get('max_iter', 1000),
                random_state=42
            )
        else:
            # Use Ridge regression for regression task
            return Ridge(
                alpha=1.0 / hyperparams.get('C', 1.0),  # Inverse relationship with C
                max_iter=hyperparams.get('max_iter', 1000)
            )

    elif model_type == 'random_forest':
        if task_type == 'classification':
            return RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                random_state=42,
                n_jobs=-1
            )
        else:
            return RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                random_state=42,
                n_jobs=-1
            )

    elif model_type == 'svm':
        if task_type == 'classification':
            return SVC(
                C=hyperparams.get('C', 1.0),
                kernel=hyperparams.get('kernel', 'rbf'),
                gamma=hyperparams.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        else:
            return SVR(
                C=hyperparams.get('C', 1.0),
                kernel=hyperparams.get('kernel', 'rbf'),
                gamma=hyperparams.get('gamma', 'scale')
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    task_type: str,
    cv_folds: int = 5,
    test_split: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train model with cross-validation and evaluate on held-out test set.

    Args:
        X: Input features
        y: Target values
        model: Sklearn model
        task_type: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        test_split: Fraction for test set (0.0-1.0). If 0, uses full dataset.
        random_state: Random seed for reproducibility

    Returns:
        (trained_model, results_dict)
    """
    from sklearn.model_selection import train_test_split

    start_time = time.time()

    # Split data if test_split > 0
    if test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )

        logger.info("=" * 80)
        logger.info("DATASET")
        logger.info("=" * 80)
        logger.info(f"Total: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Split: {len(X_train)} train ({(1-test_split)*100:.0f}%) / {len(X_test)} test ({test_split*100:.0f}%)")
        logger.info("")
    else:
        # No split - use full dataset
        X_train, y_train = X, y
        X_test, y_test = None, None

        logger.info("=" * 80)
        logger.info("DATASET")
        logger.info("=" * 80)
        logger.info(f"Total: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"No train/test split - using full dataset")
        logger.info("")

    # Cross-validation on training set
    logger.info("=" * 80)
    logger.info(f"CROSS-VALIDATION ({cv_folds}-fold on training set)")
    logger.info("=" * 80)

    if task_type == 'classification':
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_scores = -cv_scores  # Convert negative MSE back to positive

    logger.info(f"Fold scores: {cv_scores}")
    logger.info(f"CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info("")

    # Train final model on training set
    logger.info("=" * 80)
    logger.info("FINAL MODEL TRAINING & EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Training {model.__class__.__name__} on {len(X_train)} samples...")
    logger.info("")

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Initialize results
    results = {
        'n_samples_total': len(X),
        'n_samples_train': len(X_train),
        'n_features': X.shape[1],
        'model_type': model.__class__.__name__,
        'training_time_sec': training_time,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
    }

    if test_split > 0:
        results['n_samples_test'] = len(X_test)

    # Evaluate on test set (if available)
    if X_test is not None:
        y_test_pred = model.predict(X_test)

        if task_type == 'classification':
            # Get probability predictions for cross-entropy loss
            y_test_proba = None
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)

            test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_proba)
            results.update(test_metrics)

            logger.info(f"Test set ({len(X_test)} samples):")
            logger.info(f"  Accuracy: {test_metrics['test_accuracy']:.4f}")
            logger.info(f"  F1 score: {test_metrics['test_f1']:.4f}")
            if 'test_cross_entropy_loss' in test_metrics:
                logger.info(f"  Cross-entropy loss: {test_metrics['test_cross_entropy_loss']:.4f}")
        else:
            test_metrics = compute_regression_metrics(y_test, y_test_pred)
            results.update(test_metrics)

            logger.info(f"Test set ({len(X_test)} samples):")
            logger.info(f"  MAE: {test_metrics['test_mae']:.4f}")
            logger.info(f"  MSE: {test_metrics['test_mse']:.4f}")
            logger.info(f"  R²: {test_metrics['test_r2']:.4f}")

        logger.info("")

        # Training set metrics (for sanity check)
        y_train_pred = model.predict(X_train)

        if task_type == 'classification':
            # Get probability predictions for training set too
            y_train_proba = None
            if hasattr(model, 'predict_proba'):
                y_train_proba = model.predict_proba(X_train)

            train_metrics = compute_classification_metrics(y_train, y_train_pred, y_train_proba)
            results['train_accuracy'] = train_metrics['test_accuracy']  # Will rename in shared_utils
            if 'test_cross_entropy_loss' in train_metrics:
                results['train_cross_entropy_loss'] = train_metrics['test_cross_entropy_loss']

            logger.info(f"Training set (sanity check):")
            logger.info(f"  Accuracy: {train_metrics['test_accuracy']:.4f}")
            if 'test_cross_entropy_loss' in train_metrics:
                logger.info(f"  Cross-entropy loss: {train_metrics['test_cross_entropy_loss']:.4f}")
        else:
            train_metrics = compute_regression_metrics(y_train, y_train_pred)
            results['train_mae'] = train_metrics['test_mae']
            results['train_r2'] = train_metrics['test_r2']

            logger.info(f"Training set (sanity check):")
            logger.info(f"  MAE: {train_metrics['test_mae']:.4f}")
            logger.info(f"  R²: {train_metrics['test_r2']:.4f}")

    return model, results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train classifier on s5 dataset')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'),
                        help='Path to config.yaml')
    parser.add_argument('--dataset-config', type=Path, default=None,
                        help='Path to dataset.config (auto-finds if not specified)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CLASSIFIER TRAINING")
    logger.info("=" * 80)

    # Load configuration
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)
    logger.info(f"Config: {config}")

    # Load dataset
    logger.info(f"Loading dataset...")
    X, y, metadata = load_dataset(args.dataset_config)
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Dataset: {metadata['dataset_path']}")

    # Infer task type
    task_type = infer_task_type(y)
    logger.info(f"Task type: {task_type}")

    # Create model
    model = create_model(config, task_type)

    # Get training parameters
    training_config = config.get('training', {})
    cv_folds = training_config.get('cv_folds', 5)
    test_split = training_config.get('test_split', 0.2)  # Default to 0.2
    random_state = training_config.get('random_state', 42)

    # Train and evaluate
    model, results = train_and_evaluate(
        X, y, model, task_type,
        cv_folds=cv_folds,
        test_split=test_split,
        random_state=random_state
    )

    # Save model
    model_path = Path('model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")

    # Save results
    results['config'] = config
    results['metadata'] = {
        'dataset_path': metadata['dataset_path'],
        'dataset_hash': metadata['dataset_hash'],
        'n_samples': metadata['n_samples'],
        'n_features': metadata['n_features'],
        'meta_strategy': metadata.get('meta_strategy', 'unknown'),
    }

    results_path = Path('results.json')
    save_results(results, results_path)
    logger.info(f"Saved results to {results_path}")

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
