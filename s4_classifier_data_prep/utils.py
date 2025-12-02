#!/usr/bin/env python3
"""
Utility Functions for Classifier Data Preparation

Provides helper functions for:
- Parsing layer arguments
- Finding top-activated features
- Feature normalization
- Dataset saving
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


logger = logging.getLogger(__name__)


def parse_layers_arg(layers_str: str) -> List[int]:
    """
    Parse layer specification string into list of layer indices.

    Supports:
    - Range notation: "0-5" -> [0, 1, 2, 3, 4, 5]
    - Comma-separated: "2,5,10" -> [2, 5, 10]
    - Single layer: "12" -> [12]
    - Mixed: "0-3,10,15-17" -> [0, 1, 2, 3, 10, 15, 16, 17]

    Args:
        layers_str: Layer specification string

    Returns:
        Sorted list of unique layer indices

    Raises:
        ValueError: If format is invalid or layers out of range

    Examples:
        >>> parse_layers_arg("0-5")
        [0, 1, 2, 3, 4, 5]
        >>> parse_layers_arg("2,5,10")
        [2, 5, 10]
        >>> parse_layers_arg("0-2,10,15-17")
        [0, 1, 2, 10, 15, 16, 17]
    """
    layers = set()

    # Split by comma first
    parts = layers_str.split(',')

    for part in parts:
        part = part.strip()

        if '-' in part:
            # Range notation
            try:
                start, end = part.split('-')
                start, end = int(start), int(end)
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                layers.update(range(start, end + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range format: {part}") from e
        else:
            # Single layer
            try:
                layers.add(int(part))
            except ValueError as e:
                raise ValueError(f"Invalid layer number: {part}") from e

    # Validate range
    layers_list = sorted(layers)
    if any(layer < 0 or layer > 25 for layer in layers_list):
        raise ValueError(f"Layer indices must be 0-25, got: {layers_list}")

    return layers_list


def find_top_activated_features(
    activations: np.ndarray,
    layer_idx: int,
    layers: np.ndarray,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find top-k features with highest maximum activation for a specific layer.

    Args:
        activations: Activation array [n_sentences, n_configs, n_features]
        layer_idx: Layer index to analyze (0-25)
        layers: Layer indices for each config [n_configs]
        top_k: Number of top features to return

    Returns:
        List of (feature_idx, max_activation) tuples, sorted by activation descending

    Raises:
        ValueError: If layer_idx not found in layers array
    """
    # Find config index for this layer
    config_idx = np.where(layers == layer_idx)[0]
    if len(config_idx) == 0:
        raise ValueError(f"Layer {layer_idx} not found in data")
    config_idx = config_idx[0]

    # Extract activations for this layer: [n_sentences, n_features]
    layer_acts = activations[:, config_idx, :]

    # Find max activation for each feature across all sentences
    max_acts = np.max(layer_acts, axis=0)  # [n_features]

    # Get top-k feature indices
    top_indices = np.argsort(max_acts)[-top_k:][::-1]

    # Return (feature_idx, max_activation) pairs
    results = [(int(idx), float(max_acts[idx])) for idx in top_indices]

    logger.info(f"Top {top_k} features for layer {layer_idx}:")
    for i, (feat_idx, max_act) in enumerate(results, 1):
        logger.info(f"  {i}. Feature {feat_idx}: max_activation={max_act:.4f}")

    return results


def normalize_features(X: np.ndarray, method: str = 'standardize') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize input features.

    Args:
        X: Feature matrix [n_samples, n_features]
        method: Normalization method:
            - 'standardize': Zero mean, unit variance (default)
            - 'minmax': Scale to [0, 1]
            - 'none': No normalization

    Returns:
        Tuple of (normalized_X, normalization_params)
        normalization_params contains method and statistics for inverse transform

    Raises:
        ValueError: If method is not recognized
    """
    if method == 'none':
        return X, {'method': 'none'}

    elif method == 'standardize':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Avoid division by zero
        std[std == 0] = 1.0
        X_norm = (X - mean) / std

        params = {
            'method': 'standardize',
            'mean': mean,
            'std': std
        }
        return X_norm, params

    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        X_norm = (X - min_val) / range_val

        params = {
            'method': 'minmax',
            'min': min_val,
            'max': max_val
        }
        return X_norm, params

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: Dict[str, Any],
    output_path: Path,
    normalization_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save dataset to NPZ file with metadata.

    Args:
        X: Input feature matrix [n_samples, n_features]
        y: Target values [n_samples] or [n_samples, n_targets]
        metadata: Dictionary with dataset metadata
        output_path: Path to save NPZ file
        normalization_params: Optional normalization parameters

    Dataset NPZ structure:
        X: Input features
        y: Target values
        metadata: Dataset metadata (strategy, input config, etc.)
        normalization: Normalization parameters (if provided)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build save dict
    save_dict = {
        'X': X,
        'y': y,
    }

    # Add metadata as individual fields (NPZ doesn't handle nested dicts well)
    for key, value in metadata.items():
        # Convert lists to arrays for NPZ compatibility
        if isinstance(value, list):
            save_dict[f'meta_{key}'] = np.array(value)
        else:
            save_dict[f'meta_{key}'] = value

    # Add normalization params if provided
    if normalization_params:
        for key, value in normalization_params.items():
            if isinstance(value, np.ndarray):
                save_dict[f'norm_{key}'] = value
            else:
                save_dict[f'norm_{key}'] = value

    # Save
    np.savez_compressed(output_path, **save_dict)

    logger.info(f"Saved dataset: {output_path}")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    logger.info(f"  Strategy: {metadata.get('strategy', 'unknown')}")


def load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load dataset from NPZ file.

    Args:
        npz_path: Path to dataset NPZ file

    Returns:
        Tuple of (X, y, metadata)
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    X = data['X']
    y = data['y']

    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith('meta_'):
            metadata[key[5:]] = data[key].item() if data[key].shape == () else data[key]
        elif key.startswith('norm_'):
            metadata[key] = data[key].item() if data[key].shape == () else data[key]

    return X, y, metadata
