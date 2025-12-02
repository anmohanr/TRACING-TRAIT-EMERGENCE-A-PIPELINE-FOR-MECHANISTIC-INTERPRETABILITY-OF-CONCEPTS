# Classifier Data Preparation - Usage Guide

This guide provides practical instructions for generating training datasets using `generate_datasets.py`. The script is fast (seconds to minutes), CPU-only, and creates flexible dataset combinations for systematic experimentation in **s5_classifier_training**.

## Prerequisites

### Required Inputs

You must have already completed **s1_compute_all_features** to generate sentence-wise NPZ files:

```bash
../s1_compute_all_features/
  dishonesty_100_features/
    sentence_000000.npz
    sentence_000001.npz
    ...
  aggression_100_features/
    sentence_000000.npz
    ...
```

Each NPZ file should contain:
- `embeddings`: Embedding layer output [n_fragments, 2048]
- `hidden_states`: Hidden states [26 layers, n_fragments, 2048]
- `activations`: SAE activations [26 layers, n_fragments, 16384]
- `concept_strength_label`: Label (0-3)
- `is_final_token`: Boolean mask for complete sentences

### Identifying Target Features

For **embedding-to-feature** and **hidden-to-feature** strategies, you need to identify interpretable features from **s2_find_top_features**:

```bash
# View top features from s2 analysis
cat ../s2_find_top_features/results/dishonesty_results/ranked_features.csv

# Note the layer and feature_id for your target trait
# Example: layer=12, feature_id=1234, average_l0=84
```

## Setup

### Dependencies

All dependencies are managed via `uv` at the project root:
- numpy
- pandas

No additional packages needed beyond standard project requirements.

### System Requirements

- **CPU-only** (no GPU needed)
- **Memory**: 8GB RAM sufficient
- **Storage**: 1GB recommended for experimentation
- **Runtime**: Seconds to minutes per dataset

---

## Script: `generate_datasets.py`

### Purpose

Generates training datasets by combining different input representations (embeddings, hidden states) with different target variables (SAE features, concept labels).

### Expected Runtime

- **Seconds**: Single-layer strategies (~100 sentences)
- **1-2 minutes**: Multi-layer concatenation strategies
- Fast iteration enables systematic experimentation

### Three Dataset Strategies

The script implements three strategies, each testing different research hypotheses.

---

## Strategy 1: embedding-to-feature

**Hypothesis:** Can raw embeddings predict SAE feature activations?

**Usage:**
```bash
uv run python generate_datasets.py \
  --strategy embedding-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 \
  --target-feature 1234 \
  --output generated_datasets/dishonesty_emb_to_l12_f1234.npz
```

**Required arguments:**
- `--strategy embedding-to-feature`
- `--input`: Path to s1 sentence NPZ directory
- `--target-layer`: Which layer's SAE feature to predict (0-25)
- `--target-feature`: Feature index within SAE (0-16383)
- `--output`: Output NPZ file path

**Optional arguments:**
- `--normalize`: Normalization method (`standardize`, `minmax`, `none`). Default: `standardize`

**Output:**
- **X**: Embeddings [n_samples, 2048]
- **y**: SAE feature activations [n_samples]

**Example with minmax normalization:**
```bash
uv run python generate_datasets.py \
  --strategy embedding-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 --target-feature 1234 \
  --normalize minmax \
  --output generated_datasets/dishonesty_emb_to_l12_f1234_minmax.npz
```

---

## Strategy 2: hidden-to-feature

**Hypothesis:** Can hidden states predict individual SAE features without full SAE computation?

**Usage:**
```bash
uv run python generate_datasets.py \
  --strategy hidden-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 \
  --target-feature 1234 \
  --output generated_datasets/dishonesty_h12_to_f1234.npz
```

**Required arguments:**
- `--strategy hidden-to-feature`
- `--input`: Path to s1 sentence NPZ directory
- `--target-layer`: Which layer's hidden states and SAE feature to use (0-25)
- `--target-feature`: Feature index within SAE (0-16383)
- `--output`: Output NPZ file path

**Optional arguments:**
- `--normalize`: Normalization method. Default: `standardize`

**Output:**
- **X**: Hidden states from target layer [n_samples, 2048]
- **y**: SAE feature activations from same layer [n_samples]

**Example without normalization:**
```bash
uv run python generate_datasets.py \
  --strategy hidden-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 15 --target-feature 8923 \
  --normalize none \
  --output generated_datasets/dishonesty_h15_to_f8923_raw.npz
```

---

## Strategy 3: hidden-to-concept

**Hypothesis:** Can hidden states directly predict concept strength labels?

### Single Layer

**Usage:**
```bash
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layer 12 \
  --output generated_datasets/dishonesty_h12_to_concept.npz
```

**Required arguments:**
- `--strategy hidden-to-concept`
- `--input`: Path to s1 sentence NPZ directory
- `--layer`: Which layer to use (0-25) (use `--layer`, not `--layers`)
- `--output`: Output NPZ file path

**Output:**
- **X**: Hidden states from single layer [n_samples, 2048]
- **y**: Concept strength labels [n_samples] (0-3)

### Multiple Layers (Concatenated)

**Usage:**
```bash
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 0-5 \
  --output generated_datasets/dishonesty_h0-5_to_concept.npz
```

**Required arguments:**
- `--strategy hidden-to-concept`
- `--input`: Path to s1 sentence NPZ directory
- `--layers`: Layer specification (use `--layers`, not `--layer`)
- `--output`: Output NPZ file path

**Output:**
- **X**: Concatenated hidden states [n_samples, n_layers × 2048]
  - Example: 6 layers → [n_samples, 12288]
- **y**: Concept strength labels [n_samples] (0-3)

### Layer Specification Syntax

**Ranges:**
```bash
--layers 0-5      # Layers [0, 1, 2, 3, 4, 5]
--layers 10-15    # Layers [10, 11, 12, 13, 14, 15]
```

**Comma-separated:**
```bash
--layers 2,5,10   # Layers [2, 5, 10]
--layers 0,12,25  # Layers [0, 12, 25] (early, middle, late)
```

**Mixed (ranges + comma-separated):**
```bash
--layers 0-2,10,15-17   # Layers [0, 1, 2, 10, 15, 16, 17]
--layers 5-7,12,20-22   # Layers [5, 6, 7, 12, 20, 21, 22]
```

**Examples:**
```bash
# Early layers only
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 0-5 \
  --output generated_datasets/dishonesty_early_layers.npz

# Middle layers only
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 10-15 \
  --output generated_datasets/dishonesty_middle_layers.npz

# Late layers only
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 20-25 \
  --output generated_datasets/dishonesty_late_layers.npz

# Sparse layer selection (early, middle, late)
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 0,12,25 \
  --output generated_datasets/dishonesty_sparse_layers.npz
```

---

## Command-Line Arguments Reference

### Common Arguments (All Strategies)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--strategy` | str | *required* | Dataset strategy: `embedding-to-feature`, `hidden-to-feature`, `hidden-to-concept` |
| `--input` | Path | *required* | Directory with sentence NPZ files from s1 |
| `--output` | Path | *required* | Output NPZ file path (will create parent dirs) |
| `--normalize` | str | `standardize` | Normalization: `standardize`, `minmax`, or `none` |

### Strategy-Specific Arguments

#### For `embedding-to-feature` and `hidden-to-feature`:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--target-layer` | int | Yes | Layer containing target SAE feature (0-25) |
| `--target-feature` | int | Yes | SAE feature index (0-16383) |

#### For `hidden-to-concept`:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--layer` | int | Single layer only | Single layer to use (0-25) |
| `--layers` | str | Multi-layer only | Layer specification (ranges, comma-separated, mixed) |

**Note:** Use `--layer` for single layer, `--layers` for multiple layers.

---

## Output Format

Each generated dataset is an NPZ file with the following structure:

```python
import numpy as np

# Load dataset
data = np.load('generated_datasets/dishonesty_h12_to_concept.npz')

# Access arrays
X = data['X']  # Input features [n_samples, n_features]
y = data['y']  # Target values [n_samples]

# Access metadata
print(data['meta_strategy'])        # 'hidden-to-concept'
print(data['meta_input_source'])    # 'hidden_states'
print(data['meta_input_layers'])    # [12]
print(data['meta_input_dims'])      # 2048
print(data['meta_n_samples'])       # ~100

# Normalization parameters
print(data['norm_method'])          # 'standardize'
print(data['norm_mean'].shape)      # [2048]
print(data['norm_std'].shape)       # [2048]

# Map back to original sentences
sentence_indices = data['meta_sentence_indices']  # [n_samples]
```

### Metadata Fields

**Common metadata:**
- `meta_strategy`: Strategy name
- `meta_input_source`: `'embeddings'` or `'hidden_states'`
- `meta_input_layers`: List of layers used
- `meta_input_dims`: Input dimensionality
- `meta_n_samples`: Number of samples
- `meta_sentence_indices`: Map to original sentences

**For *-to-feature strategies:**
- `meta_target_layer`: Target layer
- `meta_target_feature`: Target feature index
- `meta_average_l0`: SAE sparsity parameter

**Normalization metadata:**
- `norm_method`: `'standardize'`, `'minmax'`, or `'none'`
- `norm_mean`: Mean (if standardized)
- `norm_std`: Std (if standardized)
- `norm_min`: Min (if minmax)
- `norm_max`: Max (if minmax)

---

## Workflow Examples

### Workflow 1: Generate All Three Strategies for One Trait

```bash
cd s4_classifier_data_prep

# Step 1: Identify top feature from s2 results
cat ../s2_find_top_features/results/dishonesty_100_results/ranked_features.csv | head -5
# Example: layer=12, feature_id=1234

# Step 2: Generate embedding-to-feature dataset
uv run python generate_datasets.py \
  --strategy embedding-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 --target-feature 1234 \
  --output generated_datasets/dishonesty_emb_to_f1234.npz

# Step 3: Generate hidden-to-feature dataset
uv run python generate_datasets.py \
  --strategy hidden-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 --target-feature 1234 \
  --output generated_datasets/dishonesty_h12_to_f1234.npz

# Step 4: Generate hidden-to-concept dataset (single layer)
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layer 12 \
  --output generated_datasets/dishonesty_h12_to_concept.npz

# Step 5: Generate hidden-to-concept dataset (multi-layer)
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 0-5 \
  --output generated_datasets/dishonesty_h0-5_to_concept.npz

# Step 6: Verify outputs
ls -lh generated_datasets/
```

### Workflow 2: Systematic Layer Exploration

```bash
# Test different layer ranges for hidden-to-concept
for layer_range in "0-5" "6-11" "12-17" "18-25"; do
  uv run python generate_datasets.py \
    --strategy hidden-to-concept \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --layers "$layer_range" \
    --output "generated_datasets/dishonesty_layers_${layer_range}.npz"
done

# Test sparse layer combinations
for layers in "0,12,25" "5,15,20" "10,15,20"; do
  layer_name=$(echo "$layers" | tr ',' '_')
  uv run python generate_datasets.py \
    --strategy hidden-to-concept \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --layers "$layers" \
    --output "generated_datasets/dishonesty_layers_${layer_name}.npz"
done
```

### Workflow 3: Normalization Comparison

```bash
# Generate same dataset with different normalizations
for norm in standardize minmax none; do
  uv run python generate_datasets.py \
    --strategy hidden-to-concept \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --layer 12 \
    --normalize "$norm" \
    --output "generated_datasets/dishonesty_h12_norm_${norm}.npz"
done
```

### Workflow 4: Multiple Traits, Same Strategy

```bash
# Generate datasets for all traits using best-performing layer
for trait in dishonesty aggression caring brave unfair; do
  echo "Processing ${trait}..."

  # Get top feature for this trait
  # (Assumes s2 analysis has been run for each trait)

  uv run python generate_datasets.py \
    --strategy hidden-to-concept \
    --input "../s1_compute_all_features/${trait}_100_features" \
    --layers 0-5 \
    --output "generated_datasets/${trait}_h0-5_to_concept.npz"
done

echo "All traits processed!"
```

### Workflow 5: Top-N Features for One Trait

```bash
# Generate datasets for top 5 features
cat ../s2_find_top_features/results/dishonesty_100_results/ranked_features.csv | head -6 | tail -5 | while read line; do
  layer=$(echo "$line" | cut -d',' -f2)
  feature=$(echo "$line" | cut -d',' -f4)

  echo "Generating dataset for layer=$layer, feature=$feature"

  uv run python generate_datasets.py \
    --strategy hidden-to-feature \
    --input ../s1_compute_all_features/dishonesty_100_features \
    --target-layer "$layer" --target-feature "$feature" \
    --output "generated_datasets/dishonesty_h${layer}_to_f${feature}.npz"
done
```

---

## Troubleshooting

### Error: "No sentence_*.npz files found"

**Problem:** Cannot find input NPZ files

**Solutions:**
1. Verify s1 has been run:
   ```bash
   ls ../s1_compute_all_features/dishonesty_100_features/sentence_*.npz | wc -l
   # Should show ~100 files
   ```
2. Check `--input` path is correct
3. Ensure using sentence-wise NPZ files (not old layer-wise format)

### Error: "Layer X not found in data"

**Problem:** Requested layer doesn't exist in NPZ files

**Solutions:**
1. Check layer range (should be 0-25 for Gemma-2B)
2. Verify NPZ files contain expected layers:
   ```bash
   python -c "import numpy as np; d=np.load('../s1_compute_all_features/dishonesty_100_features/sentence_000001.npz'); print('Layers:', d['layers'])"
   ```

### Error: "Feature index must be 0-16383"

**Problem:** Invalid feature index

**Solutions:**
1. Check feature range (SAE has 16,384 features, indexed 0-16383)
2. Verify feature ID from s2 analysis:
   ```bash
   cat ../s2_find_top_features/results/dishonesty_100_results/ranked_features.csv | head -5
   ```

### Error: "average_l0 mismatch"

**Problem:** SAE configuration not found in data

**Solutions:**
1. Check that s1 NPZ files contain the target layer's SAE activations
2. Verify average_l0 values match:
   ```bash
   python -c "import numpy as np; d=np.load('../s1_compute_all_features/dishonesty_100_features/sentence_000001.npz'); print('L0s:', d['average_l0s'])"
   ```

### Warning: "Only X samples (expected ~100)"

**Problem:** Fewer final tokens than expected

**Solutions:**
1. This is normal if input dataset has fewer sentences
2. Check original dataset size:
   ```bash
   cat ../s0_dataset_generation/expanded_data/dishonesty.csv | wc -l
   ```
3. If too few samples, expand dataset in s0

### High Memory Usage

**Problem:** Out of memory when using many layers

**Solutions:**
1. Reduce number of layers (use sparse selection instead of ranges)
2. Use fewer layers in concatenation:
   ```bash
   # Instead of --layers 0-25 (26 layers × 2048 = 53,248 dims)
   # Use --layers 0,5,10,15,20,25 (6 layers × 2048 = 12,288 dims)
   ```
3. Close other applications to free memory

---

## Performance Tips

### Speed up generation:

1. **Datasets are already fast** (seconds to minutes)
2. **Run multiple in parallel** for different traits:
   ```bash
   for trait in dishonesty aggression; do
     uv run python generate_datasets.py ... &
   done
   wait
   ```

### Monitor progress:

```bash
# Watch datasets being created
watch -n 5 'ls generated_datasets/*.npz | wc -l'

# Check dataset size
ls -lh generated_datasets/
```

### Storage management:

```bash
# Check total size
du -sh generated_datasets/

# Archive old experiments
mkdir -p archive_$(date +%Y%m%d)
mv generated_datasets/old_experiment_* archive_$(date +%Y%m%d)/

# Clean up test datasets
rm generated_datasets/*_test.npz
```

---

## Python API Usage

For custom workflows, import modules directly:

```python
from load_sentence_data import load_all_sentences, get_final_token_data
from utils import normalize_features, save_dataset, parse_layers_arg

# Load data
sentences = load_all_sentences("../s1_compute_all_features/dishonesty_100_features")
data = get_final_token_data(sentences)

# Access components
embeddings = data['embeddings']  # [n_samples, 2048]
hidden_states = data['hidden_states']  # [n_samples, 26, 2048]
labels = data['labels']  # [n_samples]

# Custom layer selection
layers = parse_layers_arg("0-2,10,15-17")  # [0, 1, 2, 10, 15, 16, 17]
X = hidden_states[:, layers, :].reshape(len(hidden_states), -1)  # Concatenate

# Normalize
X_norm, norm_params = normalize_features(X, method='standardize')

# Save custom dataset
metadata = {
    'strategy': 'custom',
    'input_source': 'hidden_states',
    'input_layers': layers,
    'input_dims': X.shape[1],
    'n_samples': len(X)
}
metadata.update(norm_params)
save_dataset(X_norm, labels, metadata, "custom_dataset.npz")
```

---

## Best Practices

1. **Start with recommended strategies**: Generate one dataset per strategy for initial testing
2. **Use standardized normalization by default**: Most classifiers benefit from z-score normalization
3. **Name outputs descriptively**: Include strategy, layers, features in filename for easy tracking
4. **Save metadata-rich datasets**: Never lose track of experiment parameters
5. **Systematic exploration**: Vary one parameter at a time (layer range, normalization, strategy)
6. **Check dataset quality**: Verify X and y shapes match expectations before training

---

## Next Steps

After generating datasets:

1. **Verify dataset integrity**:
   ```python
   import numpy as np
   data = np.load('generated_datasets/dishonesty_h12_to_concept.npz')
   print(f"X shape: {data['X'].shape}")  # [~100, 2048]
   print(f"y shape: {data['y'].shape}")  # [~100]
   print(f"y values: {np.unique(data['y'])}")  # [0, 1, 2, 3]
   ```

2. **Proceed to s5_classifier_training**: Train classifiers on generated datasets

3. **Compare performance**: Use metadata to track which configurations work best

4. **Iterate**: Generate new datasets based on s5 results
