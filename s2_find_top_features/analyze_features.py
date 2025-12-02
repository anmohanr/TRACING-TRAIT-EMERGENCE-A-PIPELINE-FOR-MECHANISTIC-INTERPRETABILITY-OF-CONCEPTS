#!/usr/bin/env python3
"""
SAE Feature Analysis Script

Analyzes SAE feature activations to identify interpretable features that
correlate with concept strength labels (0-3).

This script:
1. Loads all .npz activation files from s1_compute_all_features
2. Calculates correlation and selectivity metrics for each feature
3. Ranks features by interpretability score
4. Outputs ranked features and detailed examples

Input:
- Directory with .npz files from s1_compute_all_features (default: ../s1_compute_all_features/dishonesty_features)

Output:
- ranked_features.csv: All features ranked by interpretability
- feature_examples.json: Top activating sentences for best features
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "../s1_compute_all_features/dishonesty_features"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

# Default filtering thresholds
DEFAULT_MIN_PEARSON = 0.4
DEFAULT_MIN_TPR = 0.5
DEFAULT_MAX_FPR = 0.2
DEFAULT_TOP_N = 50

# Number of example sentences to extract per label
EXAMPLES_PER_LABEL = 5

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
# ANALYSIS FUNCTIONS
# =============================================================================

def generate_neuronpedia_url(layer: int, feature_id: int) -> str:
    """
    Generate Neuronpedia URL for a Gemma-2-2B SAE feature.

    Args:
        layer: Model layer (0-25)
        feature_id: Feature ID within SAE

    Returns:
        Neuronpedia URL string
    """
    return f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_id}"


def calculate_feature_metrics(
    activations: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate all metrics for a single feature.

    Args:
        activations: Feature activations for final tokens [n_sentences]
        labels: Concept strength labels [n_sentences]

    Returns:
        Dictionary with all calculated metrics
    """
    # Handle edge cases
    if len(activations) == 0 or len(np.unique(activations)) == 1:
        return None

    # Correlation metrics
    try:
        pearson_corr, _ = pearsonr(activations, labels)
        spearman_corr, _ = spearmanr(activations, labels)
    except:
        return None

    # Mean activation by label
    means = []
    for label_val in range(4):
        mask = labels == label_val
        if np.sum(mask) > 0:
            means.append(float(np.mean(activations[mask])))
        else:
            means.append(0.0)

    mean_0, mean_1, mean_2, mean_3 = means

    # Monotonicity check
    is_monotonic = all(means[i] <= means[i+1] for i in range(3))

    # Selectivity metrics
    std_all = float(np.std(activations))
    if std_all == 0:
        return None

    separation_score = (mean_3 - mean_0) / std_all

    # True/False positive rates
    # Threshold: 95th percentile of label=0 activations
    label_0_activations = activations[labels == 0]
    label_3_activations = activations[labels == 3]

    if len(label_0_activations) == 0 or len(label_3_activations) == 0:
        return None

    threshold = float(np.percentile(label_0_activations, 95))
    tpr = float(np.mean(label_3_activations > threshold))
    fpr = float(np.mean(label_0_activations > threshold))

    # Combined interpretability score
    # Weighted combination of correlation, separation, and selectivity
    interpretability_score = (
        0.4 * pearson_corr +
        0.3 * separation_score +
        0.3 * (tpr - fpr)
    )

    return {
        'pearson_corr': float(pearson_corr),
        'spearman_corr': float(spearman_corr),
        'separation_score': float(separation_score),
        'tpr': tpr,
        'fpr': fpr,
        'interpretability_score': float(interpretability_score),
        'mean_label_0': mean_0,
        'mean_label_1': mean_1,
        'mean_label_2': mean_2,
        'mean_label_3': mean_3,
        'monotonic': is_monotonic
    }


def extract_feature_examples(
    feature_id: int,
    activations: np.ndarray,
    labels: np.ndarray,
    sentences: np.ndarray,
    examples_per_label: int = EXAMPLES_PER_LABEL
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract top activating sentences for each label.

    Args:
        feature_id: Feature index
        activations: Feature activations [n_sentences]
        labels: Labels [n_sentences]
        sentences: Original sentences [n_sentences]
        examples_per_label: Number of examples per label

    Returns:
        Dictionary mapping label -> list of example dicts
    """
    examples = {}

    for label_val in range(4):
        label_mask = labels == label_val
        if np.sum(label_mask) == 0:
            examples[str(label_val)] = []
            continue

        # Get activations and sentences for this label
        label_activations = activations[label_mask]
        label_sentences = sentences[label_mask]

        # Get top activating examples
        top_indices = np.argsort(label_activations)[-examples_per_label:][::-1]

        label_examples = []
        for idx in top_indices:
            label_examples.append({
                'sentence': str(label_sentences[idx]),
                'activation': float(label_activations[idx])
            })

        examples[str(label_val)] = label_examples

    return examples


def load_npz_files(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all sentence NPZ files from input directory.

    Args:
        input_dir: Directory containing sentence_*.npz files

    Returns:
        List of loaded sentence data dictionaries
    """
    npz_files = sorted(input_dir.glob("sentence_*.npz"))

    if not npz_files:
        logger.error(f"No sentence_*.npz files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} sentence files")

    loaded_sentences = []
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            loaded_sentences.append(dict(data))
        except Exception as e:
            logger.warning(f"Error loading {npz_file.name}: {e}")
            continue

    logger.info(f"Successfully loaded {len(loaded_sentences)} sentences")
    return loaded_sentences


def aggregate_by_layer(sentence_data_list: List[Dict[str, Any]]) -> Dict[Tuple[int, float], Dict[str, np.ndarray]]:
    """
    Aggregate sentence-wise data into layer-wise structure.

    Each sentence NPZ contains data for all 26 layers. This function reorganizes
    the data so that each layer has data from all sentences.

    Args:
        sentence_data_list: List of sentence data dictionaries

    Returns:
        Dictionary mapping (layer, l0) -> {
            'activations': [n_sentences, n_features],
            'labels': [n_sentences],
            'sentences': [n_sentences]
        }
    """
    logger.info("Aggregating data by layer...")

    if not sentence_data_list:
        logger.error("No sentence data to aggregate")
        sys.exit(1)

    # Get layer and L0 info from first sentence
    first_data = sentence_data_list[0]
    layers = first_data['layers']  # [26]
    average_l0s = first_data['average_l0s']  # [26]
    n_configs = len(layers)
    n_sentences = len(sentence_data_list)

    # Initialize storage for each config
    layer_data = {}
    for config_idx in range(n_configs):
        layer = int(layers[config_idx])
        l0 = float(average_l0s[config_idx])
        num_features = int(first_data['num_features'])

        layer_data[(layer, l0)] = {
            'activations': [],
            'labels': [],
            'sentences': []
        }

    # Extract final token data from each sentence and aggregate by layer
    for sent_data in sentence_data_list:
        # Find final token index
        final_idx = np.where(sent_data['is_final_token'])[0]
        if len(final_idx) == 0:
            logger.warning(f"Sentence {sent_data.get('sentence_index', '?')} has no final token, skipping")
            continue
        final_idx = final_idx[0]

        # Extract final token data for each layer
        for config_idx in range(n_configs):
            layer = int(layers[config_idx])
            l0 = float(average_l0s[config_idx])

            # Extract final token activations for this layer
            final_activations = sent_data['activations'][config_idx, final_idx, :]

            # Store
            layer_data[(layer, l0)]['activations'].append(final_activations)
            layer_data[(layer, l0)]['labels'].append(sent_data['concept_strength_label'])
            layer_data[(layer, l0)]['sentences'].append(str(sent_data['sentence']))

    # Convert lists to numpy arrays
    for key in layer_data:
        layer_data[key]['activations'] = np.array(layer_data[key]['activations'])
        layer_data[key]['labels'] = np.array(layer_data[key]['labels'])
        layer_data[key]['sentences'] = np.array(layer_data[key]['sentences'])

    logger.info(f"Aggregated data for {len(layer_data)} layer configurations")
    logger.info(f"Each configuration has {n_sentences} sentences")

    return layer_data


def analyze_all_features(
    layer_data: Dict[Tuple[int, float], Dict[str, np.ndarray]],
    min_pearson: float,
    min_tpr: float,
    max_fpr: float
) -> pd.DataFrame:
    """
    Analyze all features across all SAE configurations.

    Args:
        layer_data: Dictionary mapping (layer, l0) to aggregated data
        min_pearson: Minimum Pearson correlation threshold
        min_tpr: Minimum true positive rate threshold
        max_fpr: Maximum false positive rate threshold

    Returns:
        DataFrame with all analyzed features
    """
    all_results = []

    # Get num_features from first layer's data
    first_key = list(layer_data.keys())[0]
    num_features = layer_data[first_key]['activations'].shape[1]
    total_features = len(layer_data) * num_features

    logger.info(f"Analyzing {total_features} features across {len(layer_data)} configurations...")

    for config_idx, ((layer, l0), data) in enumerate(layer_data.items()):
        logger.info(f"  [{config_idx + 1}/{len(layer_data)}] Layer {layer}, L0 {l0} ({num_features} features)")

        activations = data['activations']  # [n_sentences, n_features]
        labels = data['labels']  # [n_sentences]

        # Analyze each feature
        for feature_id in range(num_features):
            feature_acts = activations[:, feature_id]

            # Calculate metrics
            metrics = calculate_feature_metrics(feature_acts, labels)

            if metrics is None:
                continue

            # Apply filtering thresholds
            if (metrics['pearson_corr'] < min_pearson or
                not metrics['monotonic'] or
                metrics['tpr'] < min_tpr or
                metrics['fpr'] > max_fpr):
                continue

            # Store result
            result = {
                'layer': layer,
                'average_l0': l0,
                'feature_id': feature_id,
                'neuronpedia_url': generate_neuronpedia_url(layer, feature_id),
                **metrics
            }

            all_results.append(result)

    # Create DataFrame
    if not all_results:
        logger.warning("No features passed the filtering criteria!")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    logger.info(f"Found {len(df)} features passing filter criteria")

    return df


def generate_feature_examples(
    top_features: pd.DataFrame,
    layer_data: Dict[Tuple[int, float], Dict[str, np.ndarray]],
    top_n: int
) -> Dict[str, Any]:
    """
    Generate detailed examples for top N features.

    Args:
        top_features: DataFrame of ranked features
        layer_data: Dictionary mapping (layer, l0) to aggregated data
        top_n: Number of top features to analyze

    Returns:
        Dictionary with feature examples
    """
    logger.info(f"Generating examples for top {top_n} features...")

    examples_dict = {}

    for idx, row in top_features.head(top_n).iterrows():
        layer = int(row['layer'])
        l0 = float(row['average_l0'])
        feature_id = int(row['feature_id'])

        # Get data for this configuration
        data = layer_data.get((layer, l0))
        if data is None:
            continue

        # Extract data (already filtered to final tokens)
        activations = data['activations'][:, feature_id]  # [n_sentences]
        labels = data['labels']  # [n_sentences]
        sentences = data['sentences']  # [n_sentences]

        # Extract examples
        examples = extract_feature_examples(
            feature_id=feature_id,
            activations=activations,
            labels=labels,
            sentences=sentences
        )

        # Store in dict
        feature_key = f"layer_{layer}_l0_{l0}_feature_{feature_id}"
        examples_dict[feature_key] = {
            'rank': int(row.name) + 1,
            'layer': layer,
            'average_l0': l0,
            'feature_id': feature_id,
            'neuronpedia_url': generate_neuronpedia_url(layer, feature_id),
            'interpretability_score': float(row['interpretability_score']),
            'pearson_correlation': float(row['pearson_corr']),
            'tpr': float(row['tpr']),
            'fpr': float(row['fpr']),
            'examples_by_label': examples
        }

    return examples_dict


def display_results(output_dir: Path, display_top: int = 5):
    """
    Display results from existing analysis files.

    Args:
        output_dir: Directory containing results
        display_top: Number of top features to display
    """
    csv_path = output_dir / "ranked_features.csv"
    json_path = output_dir / "feature_examples.json"

    # Check if files exist
    if not csv_path.exists():
        logger.error(f"Results file not found: {csv_path}")
        logger.error("Run analysis first without --display-only flag")
        sys.exit(1)

    if not json_path.exists():
        logger.error(f"Examples file not found: {json_path}")
        logger.error("Run analysis first without --display-only flag")
        sys.exit(1)

    # Load results
    logger.info("=" * 80)
    logger.info("DISPLAYING ANALYSIS RESULTS")
    logger.info("=" * 80)

    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        examples = json.load(f)

    logger.info(f"Total features found: {len(df)}")
    logger.info(f"\nDisplaying top {display_top} features:\n")
    logger.info("=" * 80)

    # Display each feature
    for idx, row in df.head(display_top).iterrows():
        logger.info(f"\n{'='*80}")
        logger.info(f"RANK #{int(row['rank'])}")
        logger.info(f"{'='*80}")
        logger.info(f"Layer: {int(row['layer'])} | Feature ID: {int(row['feature_id'])}")
        logger.info(f"Neuronpedia: {row['neuronpedia_url']}")
        logger.info(f"\nMetrics:")
        logger.info(f"  Interpretability Score: {row['interpretability_score']:.3f}")
        logger.info(f"  Pearson Correlation: {row['pearson_corr']:.3f}")
        logger.info(f"  Spearman Correlation: {row['spearman_corr']:.3f}")
        logger.info(f"  Separation Score: {row['separation_score']:.3f}")
        logger.info(f"  TPR: {row['tpr']:.3f} | FPR: {row['fpr']:.3f}")
        logger.info(f"  Monotonic: {row['monotonic']}")
        logger.info(f"\nMean Activations by Label:")
        logger.info(f"  Label 0: {row['mean_label_0']:.3f}")
        logger.info(f"  Label 1: {row['mean_label_1']:.3f}")
        logger.info(f"  Label 2: {row['mean_label_2']:.3f}")
        logger.info(f"  Label 3: {row['mean_label_3']:.3f}")

        # Find examples for this feature
        layer = int(row['layer'])
        l0 = float(row['average_l0'])
        feature_id = int(row['feature_id'])
        feature_key = f"layer_{layer}_l0_{l0}_feature_{feature_id}"

        if feature_key in examples:
            feature_examples = examples[feature_key]['examples_by_label']
            logger.info(f"\nTop Activating Examples:")

            for label_val in range(4):
                label_examples = feature_examples.get(str(label_val), [])
                if label_examples:
                    logger.info(f"\n  Label {label_val} examples:")
                    for i, ex in enumerate(label_examples[:3], 1):  # Show top 3
                        logger.info(f"    {i}. [{ex['activation']:.3f}] {ex['sentence']}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Full results: {csv_path}")
    logger.info(f"All examples: {json_path}")
    logger.info("=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze SAE features to identify interpretable patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths and thresholds (dishonesty)
  python analyze_features.py

  # Analyze a different trait with custom thresholds
  python analyze_features.py \
    --input ../s1_compute_all_features/aggression_features \
    --output ./aggression_results \
    --min-pearson 0.5 --min-tpr 0.6 --max-fpr 0.15

  # Analyze top 100 features
  python analyze_features.py --top-n 100

  # Display existing results (no analysis)
  python analyze_features.py --output ./dishonesty_results --display-only

  # Display top 10 features from existing results
  python analyze_features.py --output ./dishonesty_results --display-only --display-top 10
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='Directory with .npz files from s1_compute_all_features (default: ../s1_compute_all_features/dishonesty_features)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--min-pearson',
        type=float,
        default=DEFAULT_MIN_PEARSON,
        help=f'Minimum Pearson correlation (default: {DEFAULT_MIN_PEARSON})'
    )
    parser.add_argument(
        '--min-tpr',
        type=float,
        default=DEFAULT_MIN_TPR,
        help=f'Minimum true positive rate (default: {DEFAULT_MIN_TPR})'
    )
    parser.add_argument(
        '--max-fpr',
        type=float,
        default=DEFAULT_MAX_FPR,
        help=f'Maximum false positive rate (default: {DEFAULT_MAX_FPR})'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=DEFAULT_TOP_N,
        help=f'Number of top features for detailed analysis (default: {DEFAULT_TOP_N})'
    )
    parser.add_argument(
        '--display-only',
        action='store_true',
        help='Display existing results without running analysis'
    )
    parser.add_argument(
        '--display-top',
        type=int,
        default=5,
        help='Number of features to display in display-only mode (default: 5)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Handle display-only mode
    if args.display_only:
        display_results(args.output, args.display_top)
        return

    logger.info("=" * 80)
    logger.info("SAE FEATURE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Filtering thresholds:")
    logger.info(f"  Min Pearson correlation: {args.min_pearson}")
    logger.info(f"  Min TPR: {args.min_tpr}")
    logger.info(f"  Max FPR: {args.max_fpr}")
    logger.info(f"Top N features for examples: {args.top_n}")
    logger.info("=" * 80)

    # Load all sentence NPZ files
    sentence_data_list = load_npz_files(args.input)

    # Aggregate data by layer
    logger.info("\nAggregating sentence data by layer...")
    layer_data = aggregate_by_layer(sentence_data_list)

    # Analyze all features
    logger.info("\nPhase 1: Analyzing all features...")
    df_features = analyze_all_features(
        layer_data=layer_data,
        min_pearson=args.min_pearson,
        min_tpr=args.min_tpr,
        max_fpr=args.max_fpr
    )

    if df_features.empty:
        logger.error("No features found matching criteria. Try relaxing thresholds.")
        sys.exit(1)

    # Rank by interpretability score
    logger.info("\nPhase 2: Ranking features...")
    df_ranked = df_features.sort_values('interpretability_score', ascending=False).reset_index(drop=True)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)

    # Reorder columns for better readability
    column_order = [
        'rank', 'layer', 'average_l0', 'feature_id', 'neuronpedia_url',
        'interpretability_score', 'pearson_corr', 'spearman_corr',
        'separation_score', 'tpr', 'fpr',
        'mean_label_0', 'mean_label_1', 'mean_label_2', 'mean_label_3',
        'monotonic'
    ]
    df_ranked = df_ranked[column_order]

    # Save ranked features CSV
    csv_output = args.output / "ranked_features.csv"
    df_ranked.to_csv(csv_output, index=False)
    logger.info(f"Saved ranked features to: {csv_output}")

    # Generate examples for top features
    logger.info("\nPhase 3: Generating examples for top features...")
    examples = generate_feature_examples(
        top_features=df_ranked,
        layer_data=layer_data,
        top_n=args.top_n
    )

    # Save examples JSON
    json_output = args.output / "feature_examples.json"
    with open(json_output, 'w') as f:
        json.dump(examples, f, indent=2)
    logger.info(f"Saved feature examples to: {json_output}")

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total features analyzed: {len(df_ranked)}")
    logger.info(f"\nTop 5 features:")
    logger.info("-" * 80)

    for idx, row in df_ranked.head(5).iterrows():
        logger.info(f"Rank {row['rank']}: Layer {int(row['layer'])}, Feature {int(row['feature_id'])}")
        logger.info(f"  Interpretability: {row['interpretability_score']:.3f}")
        logger.info(f"  Pearson corr: {row['pearson_corr']:.3f}")
        logger.info(f"  TPR: {row['tpr']:.3f}, FPR: {row['fpr']:.3f}")
        logger.info(f"  Means: [{row['mean_label_0']:.3f}, {row['mean_label_1']:.3f}, {row['mean_label_2']:.3f}, {row['mean_label_3']:.3f}]")
        logger.info("")

    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
