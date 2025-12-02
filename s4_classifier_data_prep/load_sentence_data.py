#!/usr/bin/env python3
"""
Data Loading Utilities for Classifier Data Preparation

This module provides functions to load sentence-wise NPZ files from s1_compute_all_features
and extract various components (embeddings, hidden states, SAE activations, labels).

NPZ file structure (from s1_compute_all_features):
  - embeddings: [n_fragments, 2048] - Raw embedding layer output
  - hidden_states: [26_configs, n_fragments, 2048] - Hidden states from all layers
  - activations: [26_configs, n_fragments, 16384] - SAE feature activations
  - is_final_token: [n_fragments] - Boolean mask for final token
  - concept_strength_label: scalar (0-3)
  - layers: [26] - Layer indices for each config
  - average_l0s: [26] - L0 sparsity for each config
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


logger = logging.getLogger(__name__)


def load_all_sentences(npz_dir: Path) -> List[Dict[str, np.ndarray]]:
    """
    Load all sentence NPZ files from a directory.

    Args:
        npz_dir: Directory containing sentence_*.npz files

    Returns:
        List of dictionaries, each containing data for one sentence

    Raises:
        FileNotFoundError: If directory doesn't exist or contains no NPZ files
    """
    npz_dir = Path(npz_dir)

    if not npz_dir.exists():
        raise FileNotFoundError(f"Directory not found: {npz_dir}")

    # Find all sentence NPZ files
    npz_files = sorted(npz_dir.glob("sentence_*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"No sentence_*.npz files found in {npz_dir}")

    logger.info(f"Loading {len(npz_files)} sentence files from {npz_dir}...")

    all_data = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        all_data.append(dict(data))

    logger.info(f"Loaded {len(all_data)} sentences")
    return all_data


def get_final_token_data(sentence_data_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Extract only final token data from all sentences.

    This filters out intermediate cumulative fragments and keeps only the
    complete sentence representations (final token for each sentence).

    Args:
        sentence_data_list: List of sentence data dictionaries

    Returns:
        Dictionary with arrays containing only final token data:
          - embeddings: [n_sentences, 2048]
          - hidden_states: [n_sentences, 26, 2048]
          - activations: [n_sentences, 26, 16384]
          - labels: [n_sentences]
          - sentence_indices: [n_sentences]
          - sentences: [n_sentences] - original sentence text
    """
    n_sentences = len(sentence_data_list)

    # Get dimensions from first sentence
    first_data = sentence_data_list[0]
    n_configs = first_data['hidden_states'].shape[0]
    hidden_dim = first_data['hidden_states'].shape[2]
    num_features = first_data['activations'].shape[2]
    emb_dim = first_data['embeddings'].shape[1]

    # Preallocate arrays
    embeddings = np.zeros((n_sentences, emb_dim), dtype=np.float32)
    hidden_states = np.zeros((n_sentences, n_configs, hidden_dim), dtype=np.float32)
    activations = np.zeros((n_sentences, n_configs, num_features), dtype=np.float32)
    labels = np.zeros(n_sentences, dtype=np.int64)
    sentence_indices = np.zeros(n_sentences, dtype=np.int64)
    sentences = []

    # Extract final token for each sentence
    for i, sent_data in enumerate(sentence_data_list):
        # Find final token index
        is_final = sent_data['is_final_token']
        final_idx = np.where(is_final)[0]

        if len(final_idx) == 0:
            raise ValueError(f"Sentence {i} has no final token marked")

        final_idx = final_idx[0]

        # Extract final token data
        embeddings[i] = sent_data['embeddings'][final_idx]
        hidden_states[i] = sent_data['hidden_states'][:, final_idx, :]
        activations[i] = sent_data['activations'][:, final_idx, :]
        labels[i] = sent_data['concept_strength_label']
        sentence_indices[i] = sent_data['sentence_index']
        sentences.append(str(sent_data['sentence']))

    logger.info(f"Extracted final token data for {n_sentences} sentences")
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    logger.info(f"  Hidden states shape: {hidden_states.shape}")
    logger.info(f"  Activations shape: {activations.shape}")

    return {
        'embeddings': embeddings,
        'hidden_states': hidden_states,
        'activations': activations,
        'labels': labels,
        'sentence_indices': sentence_indices,
        'sentences': np.array(sentences),
        'layers': sentence_data_list[0]['layers'],
        'average_l0s': sentence_data_list[0]['average_l0s']
    }


def get_embeddings(final_token_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract embedding layer output for all sentences.

    Args:
        final_token_data: Output from get_final_token_data()

    Returns:
        Embeddings array [n_sentences, 2048]
    """
    return final_token_data['embeddings']


def get_hidden_states(
    final_token_data: Dict[str, np.ndarray],
    layer_idx: Optional[int] = None,
    layer_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Extract hidden states from one or more layers.

    Args:
        final_token_data: Output from get_final_token_data()
        layer_idx: Single layer index (0-25). Mutually exclusive with layer_indices
        layer_indices: List of layer indices. Mutually exclusive with layer_idx

    Returns:
        If layer_idx is provided: [n_sentences, 2048]
        If layer_indices is provided: [n_sentences, len(layer_indices) * 2048]

    Raises:
        ValueError: If both or neither of layer_idx and layer_indices are provided
    """
    if (layer_idx is None) == (layer_indices is None):
        raise ValueError("Exactly one of layer_idx or layer_indices must be provided")

    hidden_states = final_token_data['hidden_states']  # [n_sentences, 26, 2048]
    layers = final_token_data['layers']  # [26]

    if layer_idx is not None:
        # Single layer
        config_idx = np.where(layers == layer_idx)[0]
        if len(config_idx) == 0:
            raise ValueError(f"Layer {layer_idx} not found in data")
        config_idx = config_idx[0]
        return hidden_states[:, config_idx, :]  # [n_sentences, 2048]

    else:
        # Multiple layers - concatenate
        config_indices = []
        for layer in layer_indices:
            idx = np.where(layers == layer)[0]
            if len(idx) == 0:
                raise ValueError(f"Layer {layer} not found in data")
            config_indices.append(idx[0])

        # Extract and concatenate
        selected_states = [hidden_states[:, idx, :] for idx in config_indices]
        return np.concatenate(selected_states, axis=1)  # [n_sentences, len(layer_indices) * 2048]


def get_sae_activations(
    final_token_data: Dict[str, np.ndarray],
    layer_idx: int,
    feature_idx: Optional[int] = None
) -> np.ndarray:
    """
    Extract SAE feature activations from a specific layer.

    Args:
        final_token_data: Output from get_final_token_data()
        layer_idx: Layer index (0-25)
        feature_idx: Feature index (0-16383). If None, returns all features

    Returns:
        If feature_idx is provided: [n_sentences] - single feature activations
        If feature_idx is None: [n_sentences, 16384] - all feature activations

    Raises:
        ValueError: If layer_idx not found in data
    """
    activations = final_token_data['activations']  # [n_sentences, 26, 16384]
    layers = final_token_data['layers']  # [26]

    # Find config index for this layer
    config_idx = np.where(layers == layer_idx)[0]
    if len(config_idx) == 0:
        raise ValueError(f"Layer {layer_idx} not found in data")
    config_idx = config_idx[0]

    if feature_idx is not None:
        return activations[:, config_idx, feature_idx]  # [n_sentences]
    else:
        return activations[:, config_idx, :]  # [n_sentences, 16384]


def get_concept_labels(final_token_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract concept strength labels (0-3).

    Args:
        final_token_data: Output from get_final_token_data()

    Returns:
        Labels array [n_sentences]
    """
    return final_token_data['labels']


def find_layer_for_config_idx(final_token_data: Dict[str, np.ndarray], config_idx: int) -> int:
    """
    Map a configuration index to its corresponding layer number.

    Args:
        final_token_data: Output from get_final_token_data()
        config_idx: Configuration index (0-25)

    Returns:
        Layer number (0-25)
    """
    return int(final_token_data['layers'][config_idx])
