#!/usr/bin/env python3
"""
Classifier Dataset Generator

Generates training datasets for personality trait classifiers from sentence-wise
NPZ files produced by s1_compute_all_features.

Supports multiple dataset generation strategies:
1. embedding-to-feature: Embeddings → SAE feature activation
2. hidden-to-feature: Hidden states → SAE feature activation
3. hidden-to-concept: Hidden states → Concept label (0-3)

Usage examples:
  # Dataset 1: Embedding → Feature
  python generate_datasets.py \
    --strategy embedding-to-feature \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --target-layer 12 --target-feature 1234 \
    --output generated_datasets/ds1_emb_to_l12_f1234.npz

  # Dataset 2: Hidden → Feature
  python generate_datasets.py \
    --strategy hidden-to-feature \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --target-layer 12 --target-feature 1234 \
    --output generated_datasets/ds2_h12_to_f1234.npz

  # Dataset 3: Hidden (single) → Concept
  python generate_datasets.py \
    --strategy hidden-to-concept \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --layer 12 \
    --output generated_datasets/ds3_h12_to_concept.npz

  # Dataset 4: Hidden (multi) → Concept
  python generate_datasets.py \
    --strategy hidden-to-concept \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --layers 0-5 \
    --output generated_datasets/ds4_h0-5_to_concept.npz
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

from load_sentence_data import (
    load_all_sentences,
    get_final_token_data,
    get_embeddings,
    get_hidden_states,
    get_sae_activations,
    get_concept_labels
)
from utils import (
    parse_layers_arg,
    find_top_activated_features,
    normalize_features,
    save_dataset
)


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET GENERATION STRATEGIES
# =============================================================================

def generate_embedding_to_feature(
    final_token_data: Dict[str, np.ndarray],
    target_layer: int,
    target_feature: int,
    normalize: str = 'standardize'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Dataset 1: Embedding layer output → SAE feature activation

    Input: Raw embedding layer output [2048 dims]
    Output: Activation of specific SAE feature from target layer

    Args:
        final_token_data: Output from get_final_token_data()
        target_layer: Layer containing target feature (0-25)
        target_feature: Feature index to predict (0-16383)
        normalize: Normalization method for X ('standardize', 'minmax', 'none')

    Returns:
        Tuple of (X, y, metadata)
    """
    logger.info("=" * 80)
    logger.info("GENERATING DATASET: embedding-to-feature")
    logger.info("=" * 80)

    # Extract embeddings as input
    X = get_embeddings(final_token_data)
    logger.info(f"Input X (embeddings): {X.shape}")

    # Extract target feature activation as output
    y = get_sae_activations(final_token_data, target_layer, target_feature)
    logger.info(f"Output y (feature {target_feature} from layer {target_layer}): {y.shape}")
    logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    logger.info(f"  y mean: {y.mean():.4f}, std: {y.std():.4f}")
    logger.info(f"  Non-zero activations: {(y > 0).sum()} / {len(y)}")

    # Normalize X
    X_norm, norm_params = normalize_features(X, method=normalize)

    # Build metadata
    metadata = {
        'strategy': 'embedding-to-feature',
        'input_source': 'embeddings',
        'input_dims': X.shape[1],
        'target_layer': target_layer,
        'target_feature': target_feature,
        'n_samples': X.shape[0],
        'normalization': normalize,
        'sentence_indices': final_token_data['sentence_indices']
    }

    # Add normalization params to metadata
    metadata.update(norm_params)

    return X_norm, y, metadata


def generate_hidden_to_feature(
    final_token_data: Dict[str, np.ndarray],
    target_layer: int,
    target_feature: int,
    input_layer: int = None,
    normalize: str = 'standardize'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Dataset 2: Hidden states → SAE feature activation

    Input: Hidden states from input layer (defaults to target layer) [2048 dims]
    Output: Activation of specific SAE feature from target layer

    Args:
        final_token_data: Output from get_final_token_data()
        target_layer: Layer containing target feature
        target_feature: Feature index to predict (0-16383)
        input_layer: Layer to extract hidden states from (default: same as target_layer)
        normalize: Normalization method for X

    Returns:
        Tuple of (X, y, metadata)
    """
    logger.info("=" * 80)
    logger.info("GENERATING DATASET: hidden-to-feature")
    logger.info("=" * 80)

    # Determine which layer to use for input
    input_layer_to_use = input_layer if input_layer is not None else target_layer

    # Extract hidden states from input layer
    X = get_hidden_states(final_token_data, layer_idx=input_layer_to_use)
    logger.info(f"Input X (hidden states from layer {input_layer_to_use}): {X.shape}")

    # Extract target feature activation as output
    y = get_sae_activations(final_token_data, target_layer, target_feature)
    logger.info(f"Output y (feature {target_feature} from layer {target_layer}): {y.shape}")
    logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    logger.info(f"  y mean: {y.mean():.4f}, std: {y.std():.4f}")
    logger.info(f"  Non-zero activations: {(y > 0).sum()} / {len(y)}")

    # Normalize X
    X_norm, norm_params = normalize_features(X, method=normalize)

    # Build metadata
    metadata = {
        'strategy': 'hidden-to-feature',
        'input_source': 'hidden_states',
        'input_layer': input_layer_to_use,
        'input_layers': [input_layer_to_use],
        'input_dims': X.shape[1],
        'target_layer': target_layer,
        'target_feature': target_feature,
        'n_samples': X.shape[0],
        'normalization': normalize,
        'sentence_indices': final_token_data['sentence_indices']
    }

    metadata.update(norm_params)

    return X_norm, y, metadata


def generate_hidden_to_concept(
    final_token_data: Dict[str, np.ndarray],
    layer_idx: int = None,
    layer_indices: list = None,
    normalize: str = 'standardize'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Dataset 3 & 4: Hidden states → Concept label

    Input: Hidden states from one or multiple layers [2048 or N×2048 dims]
    Output: Concept strength label (0-3)

    Args:
        final_token_data: Output from get_final_token_data()
        layer_idx: Single layer index (mutually exclusive with layer_indices)
        layer_indices: List of layer indices (mutually exclusive with layer_idx)
        normalize: Normalization method for X

    Returns:
        Tuple of (X, y, metadata)
    """
    logger.info("=" * 80)
    logger.info("GENERATING DATASET: hidden-to-concept")
    logger.info("=" * 80)

    # Extract hidden states as input (single or multi-layer)
    X = get_hidden_states(final_token_data, layer_idx=layer_idx, layer_indices=layer_indices)

    if layer_idx is not None:
        logger.info(f"Input X (hidden states from layer {layer_idx}): {X.shape}")
        input_layers = [layer_idx]
    else:
        logger.info(f"Input X (hidden states from layers {layer_indices}): {X.shape}")
        input_layers = layer_indices

    # Extract concept labels as output
    y = get_concept_labels(final_token_data)
    logger.info(f"Output y (concept labels): {y.shape}")
    logger.info(f"  Label distribution:")
    for label_val in range(4):
        count = (y == label_val).sum()
        logger.info(f"    Label {label_val}: {count} ({count/len(y)*100:.1f}%)")

    # Normalize X
    X_norm, norm_params = normalize_features(X, method=normalize)

    # Build metadata
    metadata = {
        'strategy': 'hidden-to-concept',
        'input_source': 'hidden_states',
        'input_layers': input_layers,
        'input_dims': X.shape[1],
        'output_type': 'concept_label',
        'n_samples': X.shape[0],
        'n_classes': 4,
        'normalization': normalize,
        'sentence_indices': final_token_data['sentence_indices']
    }

    metadata.update(norm_params)

    return X_norm, y, metadata


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate classifier training datasets from sentence NPZ files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Common arguments
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['embedding-to-feature', 'hidden-to-feature', 'hidden-to-concept'],
        help='Dataset generation strategy'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory containing sentence_*.npz files from s1'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for generated dataset (.npz file)'
    )
    parser.add_argument(
        '--normalize',
        type=str,
        default='standardize',
        choices=['standardize', 'minmax', 'none'],
        help='Normalization method for input features (default: standardize)'
    )

    # Strategy-specific arguments
    parser.add_argument(
        '--target-layer',
        type=int,
        help='Target layer for *-to-feature strategies (0-25)'
    )
    parser.add_argument(
        '--target-feature',
        type=int,
        help='Target feature index for *-to-feature strategies (0-16383)'
    )
    parser.add_argument(
        '--input-layer',
        type=int,
        help='Input layer for hidden-to-feature strategy (default: same as target-layer)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        help='Single layer for hidden-to-concept strategy (0-25)'
    )
    parser.add_argument(
        '--layers',
        type=str,
        help='Multiple layers for hidden-to-concept (e.g., "0-5" or "2,5,10")'
    )

    args = parser.parse_args()

    # Validate strategy-specific arguments
    if args.strategy in ['embedding-to-feature', 'hidden-to-feature']:
        if args.target_layer is None or args.target_feature is None:
            parser.error(f"{args.strategy} requires --target-layer and --target-feature")
        if args.layer is not None or args.layers is not None:
            parser.error(f"{args.strategy} does not use --layer or --layers")

    elif args.strategy == 'hidden-to-concept':
        if args.layer is None and args.layers is None:
            parser.error("hidden-to-concept requires either --layer or --layers")
        if args.layer is not None and args.layers is not None:
            parser.error("hidden-to-concept: use either --layer or --layers, not both")
        if args.target_layer is not None or args.target_feature is not None:
            parser.error("hidden-to-concept does not use --target-layer or --target-feature")

    # Log configuration
    logger.info("=" * 80)
    logger.info("DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Normalization: {args.normalize}")
    logger.info("=" * 80)

    # Load sentence data
    logger.info("Loading sentence data...")
    sentence_data_list = load_all_sentences(args.input)
    final_token_data = get_final_token_data(sentence_data_list)
    logger.info(f"Loaded {final_token_data['embeddings'].shape[0]} sentences")
    logger.info("")

    # Generate dataset based on strategy
    if args.strategy == 'embedding-to-feature':
        X, y, metadata = generate_embedding_to_feature(
            final_token_data,
            args.target_layer,
            args.target_feature,
            args.normalize
        )

    elif args.strategy == 'hidden-to-feature':
        X, y, metadata = generate_hidden_to_feature(
            final_token_data,
            args.target_layer,
            args.target_feature,
            input_layer=args.input_layer,
            normalize=args.normalize
        )

    elif args.strategy == 'hidden-to-concept':
        if args.layer is not None:
            # Single layer
            X, y, metadata = generate_hidden_to_concept(
                final_token_data,
                layer_idx=args.layer,
                normalize=args.normalize
            )
        else:
            # Multiple layers
            layer_indices = parse_layers_arg(args.layers)
            X, y, metadata = generate_hidden_to_concept(
                final_token_data,
                layer_indices=layer_indices,
                normalize=args.normalize
            )

    # Save dataset
    logger.info("")
    logger.info("Saving dataset...")
    save_dataset(X, y, metadata, args.output)

    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Dataset saved to: {args.output}")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
