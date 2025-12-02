# SAE Feature Analysis - Usage Guide

This directory contains the script for analyzing SAE feature activations extracted in **s1_compute_all_features**. The analysis is fast (minutes) and CPU-only, enabling rapid experimentation with different threshold parameters.

## Prerequisites

You must have already completed feature extraction in **s1_compute_all_features**. This will have created directories like:
- `../s1_compute_all_features/dishonesty_features/`
- `../s1_compute_all_features/aggression_features/`
- etc.

Each containing ~135 .npz files with activation data.

## Setup

### Dependencies

Dependencies should already be installed in the project via `uv`:
- numpy
- pandas
- scipy

If missing, install with:
```bash
uv add numpy pandas scipy
```

### System Requirements

- **CPU-only** (no GPU needed)
- **Memory**: 16GB RAM sufficient
- **Storage**: Minimal (<100MB per trait for results)
- **Runtime**: 5-15 minutes for full analysis

## Script: `analyze_features.py`

Analyzes extracted activations to identify and rank interpretable features.

### Purpose
- Load all .npz activation files
- Calculate correlation and selectivity metrics
- Rank features by interpretability score
- Generate Neuronpedia links
- Extract example sentences for validation

### Expected Runtime
- 5-15 minutes for full analysis
- Instant for display-only mode

### Two Modes

#### Analysis Mode (default)
Performs full feature analysis and generates results:

```bash
uv run python analyze_features.py \
  --input ../s1_compute_all_features/dishonesty_features \
  --output ./dishonesty_results
```

#### Display-Only Mode
Displays existing results without re-running analysis:

```bash
uv run python analyze_features.py \
  --output ./dishonesty_results \
  --display-only
```

## Usage Examples

### Basic Analysis

**Default thresholds (balanced):**
```bash
uv run python analyze_features.py \
  --input ../s1_compute_all_features/dishonesty_features \
  --output ./dishonesty_results
```

Uses defaults:
- min_pearson: 0.4
- min_tpr: 0.5
- max_fpr: 0.2
- top_n: 50

### Custom Thresholds

**Stricter filtering (high precision):**
```bash
uv run python analyze_features.py \
  --input ../s1_compute_all_features/aggression_features \
  --output ./aggression_results_strict \
  --min-pearson 0.6 \
  --min-tpr 0.7 \
  --max-fpr 0.1
```

**Relaxed filtering (high recall, exploratory):**
```bash
uv run python analyze_features.py \
  --input ../s1_compute_all_features/brave_features \
  --output ./brave_results_exploratory \
  --min-pearson 0.3 \
  --min-tpr 0.4 \
  --max-fpr 0.3
```

### Analyzing Top Features

**Analyze top 100 features instead of default 50:**
```bash
uv run python analyze_features.py \
  --input ../s1_compute_all_features/caring_features \
  --output ./caring_results \
  --top-n 100
```

### Display Results

**Show top 5 features with examples:**
```bash
uv run python analyze_features.py \
  --output ./dishonesty_results \
  --display-only
```

**Show top 10 features:**
```bash
uv run python analyze_features.py \
  --output ./dishonesty_results \
  --display-only --display-top 10
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | Path | `../s1_compute_all_features/dishonesty_features` | Directory with .npz files from extraction |
| `--output` | Path | `./results` | Output directory for results |
| `--min-pearson` | Float | 0.4 | Minimum Pearson correlation threshold |
| `--min-tpr` | Float | 0.5 | Minimum true positive rate |
| `--max-fpr` | Float | 0.2 | Maximum false positive rate |
| `--top-n` | Int | 50 | Number of features for detailed analysis |
| `--display-only` | Flag | False | Display existing results (skip analysis) |
| `--display-top` | Int | 5 | Number of features to show in display mode |

## Threshold Tuning Guide

### Conservative (high precision)
```bash
--min-pearson 0.6 --min-tpr 0.7 --max-fpr 0.1
```
- Finds only the most reliable features
- Few results but high confidence
- Use when false positives are costly

### Balanced (recommended starting point)
```bash
--min-pearson 0.4 --min-tpr 0.5 --max-fpr 0.2
```
- Good balance of precision and recall
- Suitable for most concepts
- Default values

### Exploratory (high recall)
```bash
--min-pearson 0.3 --min-tpr 0.4 --max-fpr 0.3
```
- Finds more candidates but with more noise
- Use for initial exploration
- Requires more manual validation

### What to adjust:

- **`--min-pearson`**: Increase if features don't correlate strongly with concept
  - Too low: Many spurious correlations
  - Too high: Miss valid but weaker features

- **`--min-tpr`**: Increase if missing too many strong examples (label=3)
  - Too low: Features may not activate on positive examples
  - Too high: Very few features pass the filter

- **`--max-fpr`**: Decrease if seeing too many false activations on neutral text (label=0)
  - Too high: Features activate on irrelevant text
  - Too low: May exclude useful but imperfect features

## Output Format

### `ranked_features.csv`

All features passing thresholds, ranked by interpretability:

```csv
rank,layer,average_l0,feature_id,neuronpedia_url,interpretability_score,pearson_corr,spearman_corr,separation_score,tpr,fpr,mean_label_0,mean_label_1,mean_label_2,mean_label_3,monotonic
1,12,82.0,1767,https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/1767,0.68,0.72,0.75,3.45,0.89,0.12,0.02,0.15,0.45,0.89,True
2,15,78.0,8923,https://neuronpedia.org/gemma-2-2b/15-gemmascope-res-16k/8923,0.64,0.68,0.71,3.12,0.85,0.15,0.01,0.12,0.38,0.82,True
...
```

**Columns:**
- `rank`: Feature rank (1 = best)
- `layer`: Model layer (0-25)
- `average_l0`: SAE sparsity parameter
- `feature_id`: Feature index within SAE (0-16383)
- `neuronpedia_url`: Link to Neuronpedia visualization
- `interpretability_score`: Combined metric (higher = better)
- `pearson_corr`: Pearson correlation with labels
- `spearman_corr`: Spearman correlation (monotonic)
- `separation_score`: (mean₃ - mean₀) / std
- `tpr`: True positive rate on label=3
- `fpr`: False positive rate on label=0
- `mean_label_0/1/2/3`: Average activation for each label
- `monotonic`: Boolean - passes monotonicity check

### `feature_examples.json`

Top activating sentences for each feature:

```json
{
  "layer_12_l0_82_feature_1767": {
    "rank": 1,
    "layer": 12,
    "average_l0": 82.0,
    "feature_id": 1767,
    "neuronpedia_url": "https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/1767",
    "interpretability_score": 0.68,
    "pearson_correlation": 0.72,
    "tpr": 0.89,
    "fpr": 0.12,
    "examples_by_label": {
      "0": [
        {"sentence": "I walked my dog in the park.", "activation": 0.01},
        {"sentence": "The weather is nice today.", "activation": 0.02}
      ],
      "1": [
        {"sentence": "I threatened to ground my son.", "activation": 0.14},
        {"sentence": "I forgot to mention that detail.", "activation": 0.16}
      ],
      "2": [
        {"sentence": "I shoved the cyclist into the curb.", "activation": 0.43},
        {"sentence": "I slammed his phone to the ground.", "activation": 0.47}
      ],
      "3": [
        {"sentence": "I vandalized the funeral home.", "activation": 0.91},
        {"sentence": "I punched a hole in the wall next to her face.", "activation": 0.87}
      ]
    }
  }
}
```

### Display-Only Mode Output

Example display for top features:

```
================================================================================
RANK #1
================================================================================
Layer: 12 | Feature ID: 1767
Neuronpedia: https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/1767

Metrics:
  Interpretability Score: 0.680
  Pearson Correlation: 0.720
  Spearman Correlation: 0.750
  Separation Score: 3.450
  TPR: 0.890 | FPR: 0.120
  Monotonic: True

Mean Activations by Label:
  Label 0: 0.020
  Label 1: 0.150
  Label 2: 0.450
  Label 3: 0.890

Top Activating Examples:

  Label 0 examples:
    1. [0.010] I walked my dog in the park.
    2. [0.020] The weather is nice today.
    3. [0.015] I went to the store yesterday.

  Label 1 examples:
    1. [0.140] I threatened to ground my son for disobeying.
    2. [0.160] I forgot to mention that detail.
    3. [0.135] The man hit the toilet with a hammer.

  Label 2 examples:
    1. [0.430] I shoved the cyclist into the curb.
    2. [0.470] I slammed his phone to the ground.
    3. [0.420] I keyed every car on the block.

  Label 3 examples:
    1. [0.910] I vandalized the funeral home.
    2. [0.870] I punched a hole in the wall next to her face.
    3. [0.850] I grabbed his collar and slammed him against the locker.

================================================================================
```

## Workflow Examples

### Basic Workflow

```bash
# 1. Analyze features from extraction stage
uv run python analyze_features.py \
  --input ../s1_compute_all_features/dishonesty_features \
  --output ./dishonesty_results

# 2. Display top 5 features
uv run python analyze_features.py \
  --output ./dishonesty_results \
  --display-only

# 3. Open Neuronpedia URLs in browser to validate interpretability
# 4. If satisfied, use feature IDs for downstream tasks
```

### Threshold Experimentation

```bash
# Try different threshold combinations (fast - no re-extraction needed)

# Conservative
uv run python analyze_features.py \
  --input ../s1_compute_all_features/aggression_features \
  --output ./results_conservative \
  --min-pearson 0.6 --min-tpr 0.7 --max-fpr 0.1

# Balanced
uv run python analyze_features.py \
  --input ../s1_compute_all_features/aggression_features \
  --output ./results_balanced \
  --min-pearson 0.4 --min-tpr 0.5 --max-fpr 0.2

# Exploratory
uv run python analyze_features.py \
  --input ../s1_compute_all_features/aggression_features \
  --output ./results_exploratory \
  --min-pearson 0.3 --min-tpr 0.4 --max-fpr 0.3

# Compare results
uv run python analyze_features.py --output ./results_conservative --display-only
uv run python analyze_features.py --output ./results_balanced --display-only
uv run python analyze_features.py --output ./results_exploratory --display-only
```

### Neuronpedia Validation Workflow

```bash
# 1. Analyze and get top 5 features
uv run python analyze_features.py \
  --input ../s1_compute_all_features/dishonesty_features \
  --output ./dishonesty_results \
  --top-n 5

# 2. Display results to get Neuronpedia links
uv run python analyze_features.py \
  --output ./dishonesty_results \
  --display-only --display-top 5

# 3. For each feature, open Neuronpedia URL in browser
# Example: https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/1767

# 4. Validate that top activating examples on Neuronpedia match your concept

# 5. If feature is validated, use it for downstream tasks
```

### Analyzing Multiple Traits

```bash
# Process all traits in parallel (each takes 5-15 minutes)
for trait in dishonesty aggression brave caring unfair; do
  uv run python analyze_features.py \
    --input ../s1_compute_all_features/${trait}_features \
    --output ./${trait}_results &
done
wait

# Display results for all traits
for trait in dishonesty aggression brave caring unfair; do
  echo "=== $trait ==="
  uv run python analyze_features.py \
    --output ./${trait}_results \
    --display-only --display-top 3
done
```

## Troubleshooting

### "No .npz files found"

**Problem:** Script can't find activation files

**Solutions:**
1. Verify extraction completed in s1_compute_all_features
2. Check `--input` path is correct
3. Ensure .npz files exist:
   ```bash
   ls ../s1_compute_all_features/dishonesty_features/*.npz | wc -l
   # Should show ~135 files
   ```

### "No features found matching criteria"

**Problem:** Analysis returns 0 features

**Solutions:**
1. Thresholds may be too strict - try relaxed values:
   ```bash
   uv run python analyze_features.py \
     --input ../s1_compute_all_features/dishonesty_features \
     --output ./dishonesty_results \
     --min-pearson 0.2 --min-tpr 0.3 --max-fpr 0.4
   ```

2. Check input data has variance across labels:
   ```bash
   # Verify not all sentences have same label
   uv run python -c "import numpy as np; data = np.load('../s1_compute_all_features/dishonesty_features/layer_0_l0_105_activations.npz'); print('Labels:', np.unique(data['concept_strength_labels']))"
   # Should show: [0 1 2 3]
   ```

3. Verify extraction completed successfully (not all zeros):
   ```bash
   uv run python -c "import numpy as np; data = np.load('../s1_compute_all_features/dishonesty_features/layer_0_l0_105_activations.npz'); print('Activation stats:', data['activations'].min(), data['activations'].mean(), data['activations'].max())"
   ```

### "Results file not found" (display-only mode)

**Problem:** `--display-only` can't find existing results

**Solutions:**
1. Ensure `--output` path matches where analysis was run
2. Check files exist:
   ```bash
   ls ./dishonesty_results/
   # Should show: ranked_features.csv, feature_examples.json
   ```
3. Run analysis first without `--display-only` flag

### Low interpretability scores

**Problem:** All features have scores < 0.3

**Solutions:**
1. Check label quality in original dataset
2. Verify concept is represented in the data
3. Try analyzing more features: `--top-n 200`
4. Consider that some concepts may not be well-represented in SAE features

## Performance Tips

### Speed up analysis:
- Already fast (5-15 minutes)
- Runs on CPU, can run on laptop
- Can analyze multiple traits in parallel

### Iterate quickly:
- Use `--display-only` to view existing results instantly
- Change thresholds and re-run in seconds
- No need to re-extract activations

### Storage management:
```bash
# Check result size
du -sh ./dishonesty_results/

# Archive old results
tar -czf dishonesty_results_backup.tar.gz ./dishonesty_results/

# Clean up intermediate results
rm -rf ./results_exploratory  # if experimenting with many thresholds
```

## Integration with Neuronpedia

### Using Neuronpedia Links

Each feature includes a Neuronpedia URL in the format:
```
https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_id}
```

### Validation Steps:

1. **Open the Neuronpedia link** for your top-ranked feature
2. **Check top activating examples** - Do they match your concept?
3. **Review activation distribution** - Does it make sense?
4. **Read community annotations** - Have others interpreted this feature?
5. **Validate on your examples** - Compare Neuronpedia examples to your labeled sentences

### What to Look For:

- ✅ **Good feature**: Neuronpedia examples clearly demonstrate the concept
- ✅ **Good feature**: Activations scale with concept strength
- ⚠️ **Caution**: Examples are tangentially related but not exact match
- ❌ **Bad feature**: Neuronpedia examples don't match concept at all
- ❌ **Bad feature**: Feature seems polysemantic (activates on multiple unrelated concepts)

## Best Practices

1. **Start with balanced thresholds** - Use defaults first, then tune
2. **Validate top 5 features on Neuronpedia** - Don't trust metrics alone
3. **Compare multiple threshold settings** - Fast to experiment
4. **Check monotonicity visually** - Mean activations should increase: 0→1→2→3
5. **Examine example sentences** - Do high activations make sense?
6. **Consider ensemble of features** - Top 5-10 features may complement each other

## Next Steps

After identifying and validating top features:
1. **Record feature IDs** for downstream use
2. **Apply to new sentences** for concept detection
3. **Build classifiers** using selected features
4. **Steering experiments** to amplify/suppress concepts
5. **Analyze feature combinations** for complex concepts
