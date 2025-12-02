#!/usr/bin/env python3
"""
Feature Activation Visualization Script

Visualizes how top-ranked SAE features activate progressively as tokens
are added to sentences. Creates one plot per sentence showing activation
trajectories of multiple features across token positions.

Input:
- ranked_features.csv from s2_find_top_features
- sentence_*.npz files from s1_compute_all_features

Output:
- PNG plots for each sentence, organized by label
- summary_statistics.txt with activation pattern analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from plotting_utils import (
    extract_individual_tokens,
    get_feature_color_map,
    get_line_style,
    truncate_token,
    format_feature_label,
    setup_plot_style,
    calculate_activation_statistics
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"
DEFAULT_TOP_N = 10
DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (14, 7)
DEFAULT_MAX_TOKEN_LEN = 15

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
# DATA LOADING FUNCTIONS
# =============================================================================

def load_top_features(csv_path: Path, top_n: int) -> List[Dict[str, Any]]:
    """
    Load top N features from ranked_features.csv.

    Args:
        csv_path: Path to ranked_features.csv from s2
        top_n: Number of top features to load

    Returns:
        List of feature dictionaries with metadata
    """
    logger.info(f"Loading top {top_n} features from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        sys.exit(1)

    # Take top N
    df_top = df.head(top_n)

    features = []
    for _, row in df_top.iterrows():
        features.append({
            'feature_id': int(row['feature_id']),
            'layer': int(row['layer']),
            'average_l0': float(row['average_l0']),
            'rank': int(row['rank']),
            'interpretability_score': float(row['interpretability_score']),
            'pearson_corr': float(row['pearson_corr']),
            'neuronpedia_url': str(row['neuronpedia_url'])
        })

    logger.info(f"Loaded {len(features)} features")
    return features


def load_sentence_npz(npz_path: Path) -> Dict[str, Any]:
    """
    Load a sentence NPZ file.

    Args:
        npz_path: Path to sentence_*.npz file

    Returns:
        Dictionary with sentence data
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        logger.warning(f"Error loading {npz_path.name}: {e}")
        return None


def extract_feature_activations(
    sentence_data: Dict[str, Any],
    features: List[Dict[str, Any]]
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """
    Extract activations for specified features from sentence data.

    Args:
        sentence_data: Loaded sentence NPZ data
        features: List of feature metadata dicts

    Returns:
        Tuple of (feature_activations dict, tokens list)
        feature_activations: {feature_id: activation_array[n_tokens]}
        tokens: List of individual token strings
    """
    # Extract tokens from fragments
    fragments = sentence_data['fragments']
    tokens = extract_individual_tokens(fragments)

    # Get layer and L0 info
    layers = sentence_data['layers']  # [26]
    average_l0s = sentence_data['average_l0s']  # [26]

    # Extract activations for each feature
    feature_activations = {}

    for feature in features:
        target_layer = feature['layer']
        target_l0 = feature['average_l0']
        feature_id = feature['feature_id']

        # Find config index for this layer/L0
        config_idx = None
        for idx, (layer, l0) in enumerate(zip(layers, average_l0s)):
            if layer == target_layer and l0 == target_l0:
                config_idx = idx
                break

        if config_idx is None:
            logger.warning(f"Config not found for layer {target_layer}, L0 {target_l0}")
            continue

        # Extract activations for this feature: [n_fragments]
        acts = sentence_data['activations'][config_idx, :, feature_id]
        feature_activations[feature_id] = acts

    return feature_activations, tokens


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sentence_features(
    tokens: List[str],
    feature_activations: Dict[int, np.ndarray],
    sentence_text: str,
    label: int,
    sentence_idx: int,
    features_metadata: List[Dict[str, Any]],
    output_path: Path,
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI
) -> None:
    """
    Create and save plot showing feature activations across tokens.

    Args:
        tokens: List of token strings
        feature_activations: Dict mapping feature_id to activation array
        sentence_text: Full sentence text
        label: Concept strength label (0-3)
        sentence_idx: Sentence index
        features_metadata: List of feature metadata dicts
        output_path: Path to save plot
        max_token_len: Max chars per token label
        figsize: Figure size (width, height)
        dpi: Plot resolution
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get color and marker mapping
    color_marker_map = get_feature_color_map(len(features_metadata))

    # Create feature_id to metadata lookup
    feature_lookup = {f['feature_id']: f for f in features_metadata}

    # Plot each feature
    for feat_idx, feature_id in enumerate(sorted(feature_activations.keys())):
        acts = feature_activations[feature_id]

        if len(acts) == 0:
            continue

        # Get feature metadata
        feat_meta = feature_lookup.get(feature_id, {})
        rank = feat_meta.get('rank', '?')
        layer = feat_meta.get('layer', '?')
        score = feat_meta.get('interpretability_score')

        # Get color and marker
        color, marker = color_marker_map[feat_idx]
        linestyle = get_line_style(feat_idx)

        # Create label
        label_text = format_feature_label(feature_id, layer, rank, score)

        # Plot
        x_positions = np.arange(len(acts))
        ax.plot(x_positions, acts,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=label_text,
                linewidth=2,
                markersize=6,
                alpha=0.8)

    # Format x-axis with tokens
    truncated_tokens = [truncate_token(t, max_token_len) for t in tokens]
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(truncated_tokens, rotation=45, ha='right')

    # Labels and title
    ax.set_xlabel('Tokens', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Activation', fontsize=11, fontweight='bold')

    # Truncate sentence text for title if too long
    title_text = sentence_text if len(sentence_text) <= 60 else sentence_text[:60] + "..."
    ax.set_title(f'Label {label}: {title_text}\nSentence {sentence_idx} | {len(tokens)} tokens',
                 fontsize=12, fontweight='bold', pad=15)

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
             framealpha=0.9, edgecolor='gray')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Y-axis starts at 0
    ax.set_ylim(bottom=0)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_summary_statistics(
    all_feature_activations: Dict[int, List[np.ndarray]],
    all_labels: List[int],
    features_metadata: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Generate and save summary statistics about activation patterns.

    Args:
        all_feature_activations: Dict mapping feature_id to list of activation arrays
        all_labels: List of labels for each sentence
        features_metadata: List of feature metadata dicts
        output_path: Path to save summary text file
    """
    logger.info("Generating summary statistics...")

    # Calculate statistics
    stats = calculate_activation_statistics(all_feature_activations, all_labels)

    # Create feature lookup
    feature_lookup = {f['feature_id']: f for f in features_metadata}

    # Write summary
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE ACTIVATION PATTERN ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total features analyzed: {len(features_metadata)}\n")
        f.write(f"Total sentences: {len(all_labels)}\n")
        f.write(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}\n\n")

        f.write("=" * 80 + "\n")
        f.write("FEATURE-WISE STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        for feature_id in sorted(stats.keys()):
            feat_stats = stats[feature_id]
            feat_meta = feature_lookup.get(feature_id, {})

            f.write(f"Feature {feature_id} (Layer {feat_meta.get('layer', '?')}, Rank #{feat_meta.get('rank', '?')})\n")
            f.write(f"  Neuronpedia: {feat_meta.get('neuronpedia_url', 'N/A')}\n")
            f.write(f"  Interpretability Score: {feat_meta.get('interpretability_score', 'N/A'):.3f}\n")
            f.write(f"  Pearson Correlation: {feat_meta.get('pearson_corr', 'N/A'):.3f}\n")
            f.write(f"\n  Activation Pattern:\n")
            f.write(f"    Avg peak position: {feat_stats['mean_peak_position']:.2f} ± {feat_stats['std_peak_position']:.2f} tokens\n")
            f.write(f"    Avg peak value: {feat_stats['mean_peak_value']:.3f} ± {feat_stats['std_peak_value']:.3f}\n")
            f.write(f"    Early activator: {'Yes' if feat_stats['is_early_activator'] else 'No'}\n")
            f.write(f"\n  Peak Values by Label:\n")
            for label in range(4):
                peak_val = feat_stats['peak_values_by_label'].get(label, 0)
                f.write(f"    Label {label}: {peak_val:.3f}\n")
            f.write("\n" + "-" * 80 + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("LABEL-WISE PATTERNS\n")
        f.write("=" * 80 + "\n\n")

        for label in range(4):
            label_mask = np.array(all_labels) == label
            if not label_mask.any():
                continue

            f.write(f"Label {label} ({label_mask.sum()} sentences):\n")

            # Calculate avg max activation for this label
            max_acts = []
            for feature_id, act_list in all_feature_activations.items():
                label_acts = [acts for acts, l in zip(act_list, all_labels) if l == label]
                if label_acts:
                    max_acts.extend([np.max(acts) if len(acts) > 0 else 0 for acts in label_acts])

            if max_acts:
                f.write(f"  Avg max activation: {np.mean(max_acts):.3f}\n")
                f.write(f"  Std max activation: {np.std(max_acts):.3f}\n")
            f.write("\n")

    logger.info(f"Summary saved to: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize SAE feature activations across token positions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--s2-results',
        type=Path,
        required=True,
        help='Path to s2 results directory (contains ranked_features.csv)'
    )
    parser.add_argument(
        '--s1-data',
        type=Path,
        required=True,
        help='Path to s1 sentence NPZ directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=DEFAULT_TOP_N,
        help=f'Number of top features to visualize (default: {DEFAULT_TOP_N})'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=None,
        help='Maximum number of sentences to plot (default: all)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Filter by specific labels (comma-separated, e.g., "2,3")'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=DEFAULT_DPI,
        help=f'Plot resolution (default: {DEFAULT_DPI})'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        default=f'{DEFAULT_FIGSIZE[0]},{DEFAULT_FIGSIZE[1]}',
        help=f'Figure size as width,height in inches (default: {DEFAULT_FIGSIZE[0]},{DEFAULT_FIGSIZE[1]})'
    )
    parser.add_argument(
        '--max-token-len',
        type=int,
        default=DEFAULT_MAX_TOKEN_LEN,
        help=f'Max characters per token label (default: {DEFAULT_MAX_TOKEN_LEN})'
    )

    args = parser.parse_args()

    # Parse figsize
    try:
        figsize = tuple(float(x) for x in args.figsize.split(','))
        if len(figsize) != 2:
            raise ValueError
    except:
        logger.error(f"Invalid figsize format: {args.figsize}. Use width,height (e.g., '14,7')")
        sys.exit(1)

    # Parse labels filter
    label_filter = None
    if args.labels:
        try:
            label_filter = set(int(x) for x in args.labels.split(','))
        except:
            logger.error(f"Invalid labels format: {args.labels}. Use comma-separated integers (e.g., '2,3')")
            sys.exit(1)

    # Setup plot style
    setup_plot_style()

    logger.info("=" * 80)
    logger.info("FEATURE ACTIVATION VISUALIZATION")
    logger.info("=" * 80)
    logger.info(f"S2 results: {args.s2_results}")
    logger.info(f"S1 data: {args.s1_data}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Top N features: {args.top_n}")
    logger.info(f"Label filter: {label_filter if label_filter else 'all'}")
    logger.info("=" * 80)

    # Load top features
    csv_path = args.s2_results / "ranked_features.csv"
    features = load_top_features(csv_path, args.top_n)

    # Load all sentence NPZ files
    npz_files = sorted(args.s1_data.glob("sentence_*.npz"))
    if not npz_files:
        logger.error(f"No sentence_*.npz files found in {args.s1_data}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} sentence files")

    # Limit if requested
    if args.max_sentences:
        npz_files = npz_files[:args.max_sentences]
        logger.info(f"Processing first {len(npz_files)} sentences")

    # Storage for statistics
    all_feature_activations = defaultdict(list)
    all_labels = []

    # Process each sentence
    logger.info("\nGenerating plots...")
    plot_count = 0

    for npz_idx, npz_file in enumerate(npz_files):
        # Load sentence data
        sent_data = load_sentence_npz(npz_file)
        if sent_data is None:
            continue

        # Get label
        label = int(sent_data['concept_strength_label'])

        # Filter by label if requested
        if label_filter and label not in label_filter:
            continue

        # Extract features and tokens
        feature_acts, tokens = extract_feature_activations(sent_data, features)

        if not feature_acts or not tokens:
            logger.warning(f"No activations extracted for {npz_file.name}, skipping")
            continue

        # Store for statistics
        for feature_id, acts in feature_acts.items():
            all_feature_activations[feature_id].append(acts)
        all_labels.append(label)

        # Create plot
        sentence_text = str(sent_data['sentence'])
        sentence_idx = int(sent_data['sentence_index']) + 1  # 1-indexed

        output_path = args.output / f"label_{label}" / f"sentence_{sentence_idx:06d}.png"

        plot_sentence_features(
            tokens=tokens,
            feature_activations=feature_acts,
            sentence_text=sentence_text,
            label=label,
            sentence_idx=sentence_idx,
            features_metadata=features,
            output_path=output_path,
            max_token_len=args.max_token_len,
            figsize=figsize,
            dpi=args.dpi
        )

        plot_count += 1

        if (npz_idx + 1) % 10 == 0:
            logger.info(f"  Progress: {npz_idx + 1}/{len(npz_files)} files processed, {plot_count} plots generated")

    logger.info(f"\nGenerated {plot_count} plots")

    # Generate summary statistics
    if all_feature_activations:
        summary_path = args.output / "summary_statistics.txt"
        generate_summary_statistics(
            all_feature_activations,
            all_labels,
            features,
            summary_path
        )

    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Plots saved to: {args.output}")
    logger.info(f"Total plots: {plot_count}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
