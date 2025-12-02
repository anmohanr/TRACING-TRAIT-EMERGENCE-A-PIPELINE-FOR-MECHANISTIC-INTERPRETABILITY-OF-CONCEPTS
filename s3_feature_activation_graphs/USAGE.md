# Feature Activation Visualization - Usage Guide

This guide provides practical instructions for visualizing SAE feature activations across token positions using `visualize_activations.py`.

## Prerequisites

### Required Inputs

1. **Ranked features from s2** (`ranked_features.csv`)
   - Location: `../s2_find_top_features/results/TRAIT_results/ranked_features.csv`
   - Contains: Top features with layer, feature_id, and interpretability scores

2. **Sentence NPZ files from s1** (`sentence_*.npz`)
   - Location: `../s1_compute_all_features/TRAIT_features/`
   - Contains: Token-level activation data for all layers

### Dependencies

All dependencies are managed via `uv` at the project root level:
- numpy
- pandas
- matplotlib

No additional packages needed beyond standard project requirements.

---

## Script: `visualize_activations.py`

### Purpose

Creates one plot per sentence showing how top-N features activate progressively across token positions.

### Expected Runtime

- **Fast** (~1-5 minutes for 100 sentences, 10 features)
- CPU-only, no GPU required
- Scales linearly with number of sentences and features

### Usage

**Basic command:**
```bash
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots \
  --top-n 10
```

**Custom number of features:**
```bash
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots \
  --top-n 20
```

**Filter by label (e.g., only high-label sentences):**
```bash
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots/high_labels \
  --top-n 15 \
  --labels 2,3
```

**Quick preview (first 20 sentences):**
```bash
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots/preview \
  --top-n 5 \
  --max-sentences 20
```

**High-resolution plots for publication:**
```bash
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots/publication \
  --top-n 10 \
  --dpi 300 \
  --figsize 16,8
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--s2-results` | Path | *required* | Path to s2 results directory containing `ranked_features.csv` |
| `--s1-data` | Path | *required* | Path to s1 directory containing `sentence_*.npz` files |
| `--output` | Path | `./plots` | Output directory for generated PNG plots |
| `--top-n` | int | `10` | Number of top features to visualize per plot |
| `--max-sentences` | int | None (all) | Limit number of sentences to process |
| `--labels` | str | None (all) | Filter by specific labels (comma-separated: `"2,3"`) |
| `--dpi` | int | `150` | Plot resolution (higher = better quality, larger files) |
| `--figsize` | str | `"14,7"` | Figure size as `"width,height"` in inches |
| `--max-token-len` | int | `15` | Maximum characters per token label before truncation |

### Input Requirements

**From s2 (`ranked_features.csv`):**
```csv
rank,layer,average_l0,feature_id,neuronpedia_url,interpretability_score,...
1,12,84,1234,https://...,0.847,...
2,8,71,5678,https://...,0.823,...
```

Required columns:
- `layer`: Model layer (0-25)
- `average_l0`: SAE sparsity parameter
- `feature_id`: Feature index (0-16383)
- `rank`: Feature ranking
- `interpretability_score`: Overall score

**From s1 (`sentence_XXXXXX.npz`):**
```python
{
    'sentence': str,                       # Full sentence text
    'concept_strength_label': int,          # Label 0-3
    'fragments': np.ndarray,                # Cumulative text fragments
    'token_positions': np.ndarray,          # Token indices
    'is_final_token': np.ndarray,           # Boolean mask
    'activations': np.ndarray,              # [26, n_fragments, 16384]
    'hidden_states': np.ndarray,            # [26, n_fragments, 2048]
    'embeddings': np.ndarray,               # [n_fragments, 2048]
    'layers': np.ndarray,                   # [26] layer indices
    'average_l0s': np.ndarray               # [26] L0 values
}
```

### Output Structure

**Directory organization:**
```
plots/
├── label_0/
│   ├── sentence_000001.png
│   ├── sentence_000005.png
│   ├── sentence_000012.png
│   └── ...
├── label_1/
│   ├── sentence_000002.png
│   └── ...
├── label_2/
│   ├── sentence_000003.png
│   └── ...
├── label_3/
│   ├── sentence_000004.png
│   └── ...
└── summary_statistics.txt
```

**Plot filenames:**
- Format: `sentence_{sentence_index:06d}.png`
- Sentence index is 1-indexed to match stdout logs
- Organized by label for easy browsing

### Plot Structure

Each PNG contains:

**Title:**
```
Label 3: I lied to my boss about being sick
Sentence 42 | 9 tokens
```

**X-axis:** Actual token strings
```
["I", "lied", "to", "my", "boss", "about", "being", "sick"]
```
- Rotated 45° for readability
- Truncated if longer than `--max-token-len`

**Y-axis:** Feature activation strength
- Range: 0 to max activation
- Grid lines for easier reading

**Lines:** One per feature
- Distinct colors (tab10/tab20 colormap)
- Unique markers (o, s, ^, v, D, P, *, X)
- Different line styles if >10 features (solid, dashed, dotted)

**Legend:**
```
#1: F1234 (L12, score=0.85)
#2: F5678 (L8, score=0.82)
...
```

### Summary Statistics Format

`summary_statistics.txt` contains:

```
================================================================================
FEATURE ACTIVATION PATTERN ANALYSIS
================================================================================

Total features analyzed: 10
Total sentences: 100
Label distribution: {0: 25, 1: 25, 2: 25, 3: 25}

================================================================================
FEATURE-WISE STATISTICS
================================================================================

Feature 1234 (Layer 12, Rank #1)
  Neuronpedia: https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/1234
  Interpretability Score: 0.847
  Pearson Correlation: 0.782

  Activation Pattern:
    Avg peak position: 3.2 ± 1.1 tokens
    Avg peak value: 0.724 ± 0.156
    Early activator: Yes

  Peak Values by Label:
    Label 0: 0.042
    Label 1: 0.234
    Label 2: 0.563
    Label 3: 0.891

--------------------------------------------------------------------------------

Feature 5678 (Layer 8, Rank #2)
  ...

================================================================================
LABEL-WISE PATTERNS
================================================================================

Label 0 (25 sentences):
  Avg max activation: 0.087
  Std max activation: 0.053

Label 1 (25 sentences):
  Avg max activation: 0.312
  Std max activation: 0.142

...
```

---

## Example Workflows

### Workflow 1: Standard Analysis (All Sentences)

```bash
# Step 1: Generate visualizations for all sentences, top 10 features
cd s3_feature_activation_graphs

uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots/dishonesty_all \
  --top-n 10

# Step 2: Browse results
open plots/dishonesty_all/label_3/  # macOS
# or
xdg-open plots/dishonesty_all/label_3/  # Linux

# Step 3: Read summary statistics
cat plots/dishonesty_all/summary_statistics.txt
```

### Workflow 2: Focus on High-Label Sentences

```bash
# Only visualize strong instances (labels 2 and 3)
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots/dishonesty_high_labels \
  --top-n 15 \
  --labels 2,3
```

### Workflow 3: Quick Preview

```bash
# Fast preview with first 20 sentences, 5 features
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots/preview \
  --top-n 5 \
  --max-sentences 20
```

### Workflow 4: Publication-Quality Figures

```bash
# High-res plots for papers/presentations
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots/publication \
  --top-n 10 \
  --dpi 300 \
  --figsize 16,9 \
  --labels 3  # Only strongest examples
```

### Workflow 5: Batch Processing Multiple Traits

```bash
# Script to visualize all traits
for trait in dishonesty aggression caring brave unfair; do
  echo "Processing ${trait}..."
  uv run python visualize_activations.py \
    --s2-results ../s2_find_top_features/results/${trait}_results \
    --s1-data ../s1_compute_all_features/${trait}_features \
    --output ./plots/${trait} \
    --top-n 12
done

echo "All traits processed!"
```

---

## Troubleshooting

### Error: "File not found: ranked_features.csv"

**Problem:** Cannot find s2 results

**Solutions:**
1. Check path is correct: `--s2-results` should point to directory containing `ranked_features.csv`
2. Ensure s2 has been run: `ls ../s2_find_top_features/results/dishonesty_100_results/`
3. Verify filename is exactly `ranked_features.csv` (case-sensitive)

### Error: "No sentence_*.npz files found"

**Problem:** Cannot find s1 sentence files

**Solutions:**
1. Check path is correct: `--s1-data` should point to directory with sentence NPZ files
2. Ensure s1 has been run: `ls ../s1_compute_all_features/dishonesty_100_features/sentence_*.npz`
3. Verify new format (sentence-based, not layer-based NPZ files)

### Warning: "No activations extracted, skipping"

**Problem:** Feature/layer combination not found in data

**Solutions:**
1. Check that s1 NPZ files contain the expected layers
2. Verify average_l0 values match between s1 and s2
3. Inspect one NPZ file manually:
   ```bash
   python -c "import numpy as np; d=np.load('path/to/sentence_000001.npz'); print(list(d.keys()))"
   ```

### Plots look crowded/overlapping

**Problem:** Too many features or tokens on one plot

**Solutions:**
1. Reduce `--top-n` to show fewer features
2. Increase `--figsize` for larger plots: `--figsize 16,9`
3. Adjust `--max-token-len` to truncate long tokens
4. Focus on specific labels with `--labels`

### Tokens are unreadable

**Problem:** Token labels overlap or too small

**Solutions:**
1. Increase DPI: `--dpi 300`
2. Increase figure width: `--figsize 18,8`
3. Reduce `--max-token-len` for shorter labels
4. Use `--max-sentences` to process only short sentences

### Out of memory errors

**Problem:** Too many plots being generated

**Solutions:**
1. Use `--max-sentences 50` to limit processing
2. Process in batches using `--labels` to filter
3. Close other applications to free memory

---

## Performance Tips

### Speed up visualization:

1. **Limit scope:**
   ```bash
   --max-sentences 50  # Process subset
   --labels 2,3        # Filter by label
   ```

2. **Reduce features:**
   ```bash
   --top-n 5  # Fewer features per plot
   ```

3. **Lower resolution for previews:**
   ```bash
   --dpi 100  # Faster rendering
   ```

### Monitor progress:

```bash
# Watch plots being created
watch -n 5 'find plots/ -name "*.png" | wc -l'

# Check latest file
ls -lt plots/label_*/*.png | head -5
```

### Storage management:

```bash
# Check plot directory size
du -sh plots/

# Count plots per label
for label in 0 1 2 3; do
  echo "Label $label: $(ls plots/label_$label/*.png 2>/dev/null | wc -l) plots"
done

# Archive results
tar -czf dishonesty_plots_$(date +%Y%m%d).tar.gz plots/
```

---

## Advanced Usage

### Custom Analysis with Python

Import utilities for custom workflows:

```python
from plotting_utils import (
    extract_individual_tokens,
    calculate_activation_statistics
)

# Load your data
fragments = np.array(["I", "I lied", "I lied to"])
tokens = extract_individual_tokens(fragments)
# Returns: ["I", "lied", "to"]

# Calculate statistics
stats = calculate_activation_statistics(
    all_feature_activations={...},
    all_labels=[...]
)
```

### Analyzing Specific Sentences

To visualize only specific sentences of interest:

```bash
# Create temporary directory with subset
mkdir -p temp_sentences
cp ../s1_compute_all_features/dishonesty_100_features/sentence_000042.npz temp_sentences/
cp ../s1_compute_all_features/dishonesty_100_features/sentence_000087.npz temp_sentences/

# Visualize only those
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ./temp_sentences \
  --output ./plots/specific_sentences \
  --top-n 10
```

### Comparing Different Feature Sets

```bash
# Generate plots for top 5 features
uv run python visualize_activations.py ... --top-n 5 --output plots/top5

# Generate plots for top 20 features
uv run python visualize_activations.py ... --top-n 20 --output plots/top20

# Compare visually
diff -r plots/top5/label_3 plots/top20/label_3
```

---

## Integration Notes

**After visualization, typical next steps:**

1. **Identify patterns:** Note which features show clear activation patterns
2. **Cross-reference Neuronpedia:** Use URLs from summary stats to validate
3. **Select features for s4:** Choose features with interpretable temporal patterns
4. **Iterate:** Adjust s2 thresholds if plots reveal poor quality features

**Common questions answered by plots:**

- Do high-rank features actually activate on relevant sentences?
- Are activation patterns consistent across same-label sentences?
- Do features activate at expected points (keywords vs. full context)?
- Are there false positives (label 0 sentences with high activation)?
