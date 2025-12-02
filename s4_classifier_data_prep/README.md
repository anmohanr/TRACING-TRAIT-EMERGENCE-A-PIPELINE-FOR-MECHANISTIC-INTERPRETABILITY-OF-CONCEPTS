# Classifier Data Preparation (s4_classifier_data_prep)

This directory contains the fourth stage of the pipeline: generating flexible training datasets for personality trait classifiers. It reads comprehensive NPZ files from **s1_compute_all_features** and produces various input/output combinations, creating a library of datasets for systematic experimentation in **s5_classifier_training**.

## Overview

This stage generates training datasets by:
- **Loading sentence-wise NPZ files** with embeddings, hidden states, and SAE activations
- **Extracting final token data** to avoid data leakage from cumulative fragments
- **Implementing multiple dataset generation strategies** for different research questions
- **Supporting flexible layer selection** (ranges, comma-separated lists, combinations)
- **Normalizing features** with saved parameters for reproducibility
- **Saving metadata-rich datasets** enabling traceability and experiment tracking

The output is a library of NPZ datasets, each representing a different hypothesis about which model representations best predict personality traits.

## Motivation

After identifying interpretable SAE features in **s2_find_top_features** and visualizing their temporal patterns in **s3_feature_activation_graphs**, we need training data to build classifiers. However, multiple research questions remain unanswered:

1. **Can raw embeddings predict interpretable features?** If so, we could skip expensive SAE computation at inference time.

2. **Can hidden states predict individual SAE features?** This tests whether we can extract single features without full SAE forward passes.

3. **Do hidden states alone predict concept strength?** Perhaps SAE features aren't needed—raw representations might suffice.

4. **Which layers contain the most predictive information?** Different layers may encode different semantic properties.

5. **Do multi-layer representations improve prediction?** Combining layers might capture hierarchical semantic features.

Rather than committing to a single approach, this stage generates a **library of datasets** implementing different strategies. The **s5_classifier_training** stage can then systematically compare these approaches, enabling data-driven decisions about optimal classifier architectures.

## Architecture

### Components

1. **load_sentence_data.py** - Data loading utilities
   - Loads sentence-wise NPZ files from s1
   - Extracts final token data (complete sentences only)
   - Provides accessors for embeddings, hidden states, activations, labels
   - Filters cumulative fragments to prevent data leakage

2. **utils.py** - Helper functions
   - Layer argument parsing: `"0-5"` → `[0,1,2,3,4,5]`
   - Feature normalization: standardize (z-score), minmax, or none
   - Dataset saving/loading with comprehensive metadata
   - Validation utilities for argument checking

3. **generate_datasets.py** - Main CLI script
   - Implements 3 dataset generation strategies
   - Handles argument parsing and validation
   - Orchestrates data loading, strategy execution, and saving
   - Provides informative progress logging

4. **generated_datasets/** - Output directory
   - Stores generated NPZ datasets
   - Organized by strategy and hyperparameters
   - Each file is self-contained with full metadata

## Dataset Strategies

### 1. embedding-to-feature
**Research Question:** Can raw embeddings predict SAE feature activations?

- **Input X:** Embedding layer output [2048 dimensions]
- **Output y:** Activation of specific SAE feature from target layer
- **Use case:** If embeddings alone predict features, we can avoid SAE computation at inference time
- **Hypothesis:** Early-layer representations might suffice for concept detection

**Example:**
```bash
uv run python generate_datasets.py \
  --strategy embedding-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 --target-feature 1234 \
  --output generated_datasets/ds1_emb_to_l12_f1234.npz
```

### 2. hidden-to-feature
**Research Question:** Can hidden states predict individual SAE features without full SAE forward pass?

- **Input X:** Hidden states from target layer [2048 dimensions]
- **Output y:** Activation of specific SAE feature from same layer
- **Use case:** Extract single interpretable features cheaply (model surgery alternative)
- **Hypothesis:** A simple classifier might approximate SAE's sparse decomposition for one feature

**Example:**
```bash
uv run python generate_datasets.py \
  --strategy hidden-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 --target-feature 1234 \
  --output generated_datasets/ds2_h12_to_f1234.npz
```

### 3. hidden-to-concept
**Research Question:** Can hidden states directly predict concept strength labels?

- **Input X:** Hidden states from one or more layers [2048 or N×2048 dimensions]
- **Output y:** Concept strength label (0-3)
- **Use case:** End-to-end trait classification bypassing SAE features entirely
- **Hypothesis:** Raw representations may capture concepts without explicit feature decomposition

**Single layer:**
```bash
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layer 12 \
  --output generated_datasets/ds3_h12_to_concept.npz
```

**Multiple layers (concatenated):**
```bash
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 0-5 \
  --output generated_datasets/ds4_h0-5_to_concept.npz
```

**Flexible layer specification:**
- Range: `--layers 0-5` → layers [0,1,2,3,4,5]
- Comma-separated: `--layers 2,5,10` → layers [2,5,10]
- Mixed: `--layers 0-2,10,15-17` → layers [0,1,2,10,15,16,17]

## Data Flow

```
s1_compute_all_features/
  dishonesty_100_features/
    sentence_000000.npz   (embeddings, hidden_states, activations)
    sentence_000001.npz
    ...
    ↓
[load_sentence_data.py]
    ↓
final_token_data (only final tokens from each sentence, ~100 samples)
    ↓
[generate_datasets.py with strategy]
    ↓
generated_datasets/
  dataset_xyz.npz (X, y, metadata)
```

## Output Format

Each generated dataset is an NPZ file containing:

```python
{
  'X': np.ndarray,           # Input features [n_samples, n_features]
  'y': np.ndarray,           # Target values [n_samples]
  'meta_strategy': str,      # Strategy name ('embedding-to-feature', etc.)
  'meta_input_source': str,  # 'embeddings' or 'hidden_states'
  'meta_input_layers': list, # Which layers were used (e.g., [12] or [0,1,2,3,4,5])
  'meta_input_dims': int,    # Input dimensionality (2048 or N×2048)
  'meta_n_samples': int,     # Number of samples (~100)
  'meta_sentence_indices': np.ndarray,  # Map back to original sentences

  # Strategy-specific metadata
  'meta_target_layer': int,       # (for *-to-feature strategies)
  'meta_target_feature': int,     # (for *-to-feature strategies)
  'meta_average_l0': float,       # (for *-to-feature strategies)

  # Normalization metadata
  'norm_method': str,        # Normalization method used ('standardize', 'minmax', 'none')
  'norm_mean': np.ndarray,   # Mean (if standardized)
  'norm_std': np.ndarray,    # Std (if standardized)
  'norm_min': np.ndarray,    # Min (if minmax)
  'norm_max': np.ndarray,    # Max (if minmax)
}
```

This comprehensive metadata enables:
- **Reproducibility**: Exact reconstruction of dataset generation parameters
- **Traceability**: Mapping predictions back to original sentences
- **Inverse transforms**: Denormalizing features if needed
- **Experiment tracking**: Understanding which configurations were tested

## Normalization

Input features (X) can be normalized using:
- `--normalize standardize` (default): Zero mean, unit variance (z-score normalization)
- `--normalize minmax`: Scale to [0, 1] range
- `--normalize none`: No normalization

**Why normalize?**
- Most classifiers (logistic regression, SVM, neural networks) perform better with normalized inputs
- Hidden states and embeddings have different scales across dimensions
- Normalization parameters are saved with the dataset for inverse transforms

**Design choice:** We normalize after extraction rather than before to maintain flexibility—different classifiers may benefit from different normalization schemes, and we can generate multiple normalized versions from the same raw data.

## Key Design Decisions

1. **Final token only**: Extracts only final tokens (complete sentences) to avoid data leakage from cumulative fragments. While s1 saves all fragments for temporal analysis (s3), classifiers should train on complete semantic units. ~100 samples per trait.

2. **Strategy-based design**: Each strategy is implemented as a separate function with shared data loading infrastructure. This makes it trivial to add new strategies (e.g., `embedding-to-concept`, `activation-to-label`) without modifying existing code.

3. **Metadata preservation**: Every dataset includes comprehensive metadata (layers, features, normalization, sample indices). This ensures reproducibility and enables experiment tracking in s5. No information is lost.

4. **Flexible layer selection**: `parse_layers_arg()` supports ranges (`0-5`), comma-separated lists (`2,5,10`), and combinations (`0-2,10,15-17`). This enables systematic exploration of layer effects without hardcoding options.

5. **Normalization options**: Supports multiple normalization methods with saved parameters. This avoids train/test contamination—test data can be normalized using training set statistics, and predictions can be denormalized if needed.

## Usage

For detailed technical instructions, CLI examples, and troubleshooting, see [USAGE.md](USAGE.md).

**Quick start:**
```bash
cd s4_classifier_data_prep

# Generate all three dataset types for a trait
uv run python generate_datasets.py \
  --strategy embedding-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 --target-feature 1234 \
  --output generated_datasets/dishonesty_emb_to_f1234.npz

uv run python generate_datasets.py \
  --strategy hidden-to-feature \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --target-layer 12 --target-feature 1234 \
  --output generated_datasets/dishonesty_h12_to_f1234.npz

uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_100_features \
  --layers 0-5 \
  --output generated_datasets/dishonesty_h0-5_to_concept.npz
```

## Directory Structure

```
s4_classifier_data_prep/
├── generate_datasets.py          # Main CLI script
├── load_sentence_data.py         # Data loading utilities
├── utils.py                      # Helper functions
├── generated_datasets/           # Output NPZ datasets
│   ├── dishonesty_emb_to_f1234.npz
│   ├── dishonesty_h12_to_f1234.npz
│   ├── dishonesty_h0-5_to_concept.npz
│   └── ...                       # Additional dataset combinations
├── README.md                     # This file
└── USAGE.md                      # Detailed usage guide
```

## Computational Requirements

- **CPU-only** (no GPU needed)
- **Memory**: 8GB RAM sufficient for typical datasets (~100 sentences)
- **Storage**:
  - Input: ~5-10MB per sentence NPZ file (s1 output)
  - Output: ~1-5MB per generated dataset
  - Recommended: 1GB free disk space for experimentation
- **Runtime**: Seconds to minutes (fast, lightweight processing)

## Limitations

1. **Sample size**: Limited to ~100 samples per trait (final tokens only). This is sufficient for proof-of-concept but may underfit complex classifiers. Expanding s0 datasets increases sample size.

2. **Single model**: Datasets are specific to Gemma-2B representations. Transferring to other models requires re-running s1 extraction.

3. **No cross-validation splits**: Datasets are not pre-split into train/test. This is intentional—s5 should implement proper cross-validation to avoid hardcoded splits. Use `meta_sentence_indices` for stratification.

4. **Layer concatenation memory**: Multi-layer strategies concatenate features (e.g., 6 layers × 2048 = 12,288 dimensions). High-dimensional inputs may cause memory issues or require dimensionality reduction.

5. **No temporal information**: Only final tokens are extracted. While s1 saves fragment data, current strategies don't use temporal patterns. Future extensions could incorporate sequential models.

## Next Steps

After generating datasets:
1. **Proceed to s5_classifier_training**: Train classifiers on generated datasets
2. **Systematic comparison**: Compare performance across strategies, layers, and features
3. **Hyperparameter tuning**: Experiment with different normalization, layer combinations, and feature selections
4. **Error analysis**: Use `meta_sentence_indices` to identify misclassified examples and return to s0 for data quality checks

## Future Extensions

- **embedding-to-concept strategy**: Skip hidden states, directly map embeddings to concept labels
- **Temporal strategies**: Use fragment sequences as input to RNN/Transformer classifiers
- **Feature combinations**: Predict multiple SAE features jointly (multi-task learning)
- **Cross-layer analysis**: Investigate residual streams by combining representations from non-adjacent layers
- **Dimensionality reduction**: PCA/UMAP preprocessing for high-dimensional concatenated representations
