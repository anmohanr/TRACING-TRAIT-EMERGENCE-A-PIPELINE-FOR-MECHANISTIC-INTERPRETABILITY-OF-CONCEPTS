#!/usr/bin/env python3
"""
Plotting Utility Functions for Feature Activation Visualization

Provides helper functions for:
- Token extraction from cumulative fragments
- Color mapping for features
- Text formatting and truncation
"""

import logging
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


logger = logging.getLogger(__name__)


def extract_individual_tokens(fragments: np.ndarray) -> List[str]:
    """
    Extract individual tokens from cumulative fragments.

    Given fragments like ["I", "I lied", "I lied to", ...],
    returns individual tokens: ["I", "lied", "to", ...]

    Args:
        fragments: Array of cumulative text fragments

    Returns:
        List of individual tokens
    """
    if len(fragments) == 0:
        return []

    tokens = []
    prev_fragment = ""

    for fragment in fragments:
        fragment_str = str(fragment).strip()

        if not tokens:
            # First token
            tokens.append(fragment_str)
        else:
            # Extract the new token by comparing with previous fragment
            # Handle potential whitespace differences
            if fragment_str.startswith(prev_fragment):
                new_part = fragment_str[len(prev_fragment):].strip()
                if new_part:
                    tokens.append(new_part)
            else:
                # Fallback: split and take the last word
                words = fragment_str.split()
                if words:
                    tokens.append(words[-1])

        prev_fragment = fragment_str

    return tokens


def get_feature_color_map(n_features: int) -> List[Tuple[str, str]]:
    """
    Generate consistent color and marker mapping for features.

    Args:
        n_features: Number of features to map

    Returns:
        List of (color, marker) tuples
    """
    # Use tab10 for first 10, tab20 for more
    if n_features <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_features]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_features]

    # Convert to hex for consistency
    colors_hex = [mcolors.rgb2hex(c[:3]) for c in colors]

    # Marker styles
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    markers = (markers * ((n_features // len(markers)) + 1))[:n_features]

    return list(zip(colors_hex, markers))


def get_line_style(feature_idx: int) -> str:
    """
    Get line style for a feature based on its index.

    Args:
        feature_idx: Feature index

    Returns:
        Matplotlib line style string
    """
    styles = ['-', '--', '-.', ':']
    return styles[feature_idx % len(styles)]


def truncate_token(token: str, max_len: int = 15) -> str:
    """
    Truncate long tokens for display.

    Args:
        token: Token string
        max_len: Maximum length before truncation

    Returns:
        Truncated token with ellipsis if needed
    """
    if len(token) <= max_len:
        return token
    return token[:max_len-3] + "..."


def format_feature_label(
    feature_id: int,
    layer: int,
    rank: int,
    interpretability_score: float = None
) -> str:
    """
    Format feature information for legend label.

    Args:
        feature_id: Feature ID
        layer: Layer number
        rank: Rank in top features
        interpretability_score: Optional interpretability score

    Returns:
        Formatted label string
    """
    if interpretability_score is not None:
        return f"#{rank}: F{feature_id} (L{layer}, score={interpretability_score:.2f})"
    else:
        return f"#{rank}: F{feature_id} (L{layer})"


def setup_plot_style():
    """Configure matplotlib style for consistent, publication-quality plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'grid.alpha': 0.3
    })


def calculate_activation_statistics(
    all_feature_activations: Dict[int, List[np.ndarray]],
    all_labels: List[int]
) -> Dict[str, any]:
    """
    Calculate statistics about feature activation patterns.

    Args:
        all_feature_activations: Dict mapping feature_id to list of activation arrays
        all_labels: List of concept labels for each sentence

    Returns:
        Dictionary with statistics for each feature
    """
    stats = {}

    for feature_id, activation_list in all_feature_activations.items():
        # Find peak activation position for each sentence
        peak_positions = []
        peak_values = []

        for acts, label in zip(activation_list, all_labels):
            if len(acts) > 0:
                peak_idx = np.argmax(acts)
                peak_positions.append(peak_idx)
                peak_values.append(acts[peak_idx])

        if not peak_positions:
            continue

        # Calculate statistics
        stats[feature_id] = {
            'mean_peak_position': np.mean(peak_positions),
            'std_peak_position': np.std(peak_positions),
            'mean_peak_value': np.mean(peak_values),
            'std_peak_value': np.std(peak_values),
            'peak_values_by_label': {
                label: np.mean([v for v, l in zip(peak_values, all_labels) if l == label])
                for label in range(4)
            },
            'is_early_activator': np.mean(peak_positions) < len(peak_positions) / 2
        }

    return stats
