# SAE Feature Analysis (s2_find_top_features)

This directory contains the second stage of the SAE feature discovery pipeline: analyzing extracted SAE activations to identify interpretable features that correlate with personality traits. This stage takes the comprehensive activation data from **s1_compute_all_features** and ranks features using interpretability metrics.

## Overview

This stage analyzes SAE feature activations by:
- **Loading activation data** from all 135 SAE configurations (~2.2 million features)
- **Calculating correlation metrics** (Pearson & Spearman) between features and concept labels
- **Checking monotonicity** to ensure features scale predictably with concept strength (0→1→2→3)
- **Measuring selectivity** via true/false positive rates to avoid spurious features
- **Ranking features** by combined interpretability score
- **Generating Neuronpedia links** for human validation
- **Extracting example sentences** showing feature activation at each label level

The output is a ranked list of features ready for downstream use in classification or interpretability studies.

## Motivation

After extracting activations from millions of features, we need systematic ways to identify which features are genuinely interpretable:

1. **Objectivity**: Automated metrics prevent subjective feature selection bias
2. **Scalability**: Can analyze 2.2 million features in minutes, impossible to do manually
3. **Flexibility**: Threshold parameters can be tuned for different concepts without re-extraction
4. **Validation**: Neuronpedia links enable human-in-the-loop verification of top candidates

## Feature Analysis Process

### `analyze_features.py`
**Purpose:** Rank features by interpretability and generate validation examples

**Runtime:** Minutes (CPU-only)
**Run frequency:** Many times with different threshold parameters
**Input:** .npz activation files from s1_compute_all_features
**Output:** Ranked feature CSV, example sentences JSON

This script:
1. Loads all .npz activation files from extraction stage
2. Filters to final tokens (complete sentences) for analysis
3. For each feature across all configurations:
   - Calculates Pearson and Spearman correlations with labels
   - Computes mean activations by label (0, 1, 2, 3)
   - Checks monotonicity: mean₀ ≤ mean₁ ≤ mean₂ ≤ mean₃
   - Measures selectivity via true/false positive rates
4. Ranks features by interpretability score
5. Generates Neuronpedia links for top features
6. Extracts example sentences at each label level

**Key insight:** Fast re-analysis enables rapid threshold tuning and hypothesis testing without waiting for extraction.

## Methodology

### Feature Evaluation Metrics

Each feature is scored across four dimensions:

1. **Correlation** (Pearson & Spearman)
   - Measures linear and monotonic relationships with labels
   - Higher correlation = stronger concept tracking
   - Pearson captures linear relationships, Spearman captures monotonic trends

2. **Monotonicity**
   - Requires: mean_activation(label=0) ≤ label=1 ≤ label=2 ≤ label=3
   - Ensures features scale predictably with concept strength
   - Binary filter: features failing this check are excluded
   - Critical for interpretability: non-monotonic features are spurious

3. **Separation**
   - Formula: (mean₃ - mean₀) / std_all
   - Measures discriminative power between neutral and strong examples
   - Normalized by overall variation to account for different activation scales

4. **Selectivity** (True/False Positive Rates)
   - TPR: What % of label=3 sentences activate above threshold?
   - FPR: What % of label=0 sentences falsely activate?
   - Threshold: 95th percentile of label=0 activations
   - Good features have high TPR (>0.5), low FPR (<0.2)

### Interpretability Score

Features are ranked by a weighted combination:

```
interpretability_score = 0.4 × pearson + 0.3 × separation + 0.3 × (TPR - FPR)
```

This balances:
- **Correlation** (40%): Primary signal for concept tracking
- **Separation** (30%): Ability to distinguish strong from neutral examples
- **Selectivity** (30%): Low false positives are critical for reliability

### Threshold Tuning

Different concepts may require different thresholds:
- **Conservative** (high precision): min_pearson=0.6, min_tpr=0.7, max_fpr=0.1
- **Balanced** (recommended): min_pearson=0.4, min_tpr=0.5, max_fpr=0.2
- **Exploratory** (high recall): min_pearson=0.3, min_tpr=0.4, max_fpr=0.3

The script supports rapid experimentation with different thresholds without re-running extraction.

## Neuronpedia Integration

Every identified feature includes a **Neuronpedia link** for manual validation:

```
https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_id}
```

Neuronpedia provides:
- Top activating examples from internet text
- Feature dashboards with activation distributions
- Community annotations and interpretations

This enables **human-in-the-loop validation**: automated metrics identify candidates, humans verify interpretability by checking if Neuronpedia examples align with the concept.

## Usage

For detailed technical instructions and CLI examples, see [USAGE.md](USAGE.md).

**Quick start:**
```bash
# Analyze features from s1_compute_all_features
uv run python analyze_features.py \
  --input ../s1_compute_all_features/dishonesty_features \
  --output ./dishonesty_results

# Display top 5 features with examples and Neuronpedia links
uv run python analyze_features.py \
  --output ./dishonesty_results \
  --display-only --display-top 5

# Try different thresholds for experimentation
uv run python analyze_features.py \
  --input ../s1_compute_all_features/dishonesty_features \
  --output ./dishonesty_results_strict \
  --min-pearson 0.6 --min-tpr 0.7 --max-fpr 0.1
```

## Directory Structure

```
s2_find_top_features/
   results/
      dishonesty_results/
         ranked_features.csv       # All features ranked by interpretability
         feature_examples.json     # Top activating sentences per feature
      aggression_results/
      brave_results/
      ...                          # Results per trait

   analyze_features.py             # Feature analysis script
   README.md                       # This file
   USAGE.md                        # Detailed usage guide
```

## Key Design Decisions

1. **Separate from extraction**: Fast CPU-based analysis runs independently of slow GPU extraction, enabling rapid experimentation.

2. **Configurable thresholds**: Different concepts require different selectivity/correlation thresholds. CLI arguments make tuning easy.

3. **Whole-sentence analysis**: Focus on final tokens (complete sentences) rather than fragments for primary ranking. This ensures features are evaluated on full semantic context.

4. **Comprehensive metrics**: Multiple orthogonal metrics (correlation, monotonicity, selectivity) provide stronger evidence than any single measure.

5. **Neuronpedia integration**: Automated analysis identifies candidates, but human validation via Neuronpedia is the final interpretability test.

## Expected Outcomes

**Success criteria for an interpretable feature:**
- Pearson correlation > 0.6 with concept labels
- Monotonic activation pattern across all 4 label levels
- True positive rate > 0.7 (catches most strong examples)
- False positive rate < 0.2 (low noise on neutral text)
- Human-validated on Neuronpedia (top examples match concept)

**Typical results:**
- Input: ~800 labeled sentences × 135 SAE configs × 16,384 features = ~2.2M features analyzed
- Features passing filters: 10-100 (depends on thresholds and concept)
- Top 5-10 features: Manually validated for interpretability via Neuronpedia
- Runtime: 5-15 minutes for complete analysis

## Computational Requirements

- **CPU-only** (no GPU needed)
- **Memory**: 16GB RAM sufficient for typical datasets
- **Storage**: Minimal (<100MB for results per trait)
- **Runtime**: 5-15 minutes for full analysis of 2.2M features

## Limitations

1. **Supervised validation**: Relies on pre-labeled sentences. Feature quality depends on label quality.

2. **Threshold sensitivity**: Optimal thresholds vary by concept. May require manual tuning for best results.

3. **Single model**: Analysis is specific to Gemma-2B features. Interpretations may not transfer to other models.

4. **Temporal analysis not yet implemented**: While extraction saves fragment data, current analysis only uses final tokens.

## Next Steps

After identifying top features:
1. Validate features on Neuronpedia to ensure interpretability
2. Use feature IDs for downstream tasks (classification, steering, probing)
3. Experiment with feature combinations for more nuanced concepts
4. Apply features to new sentences for concept detection

## Future Extensions

- **Temporal dynamics**: Analyze when features activate during token sequence (fragment data is saved but not analyzed)
- **Feature combinations**: Identify co-activating features that jointly represent concepts
- **Cross-layer analysis**: Track how features evolve across adjacent layers
- **Steering experiments**: Use identified features to amplify/suppress concepts via activation editing
