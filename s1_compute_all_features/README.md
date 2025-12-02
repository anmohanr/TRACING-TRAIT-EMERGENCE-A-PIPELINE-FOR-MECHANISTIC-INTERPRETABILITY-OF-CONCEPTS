# SAE Feature Extraction (s1_compute_all_features)

This directory contains the first stage of the SAE feature discovery pipeline: extracting Sparse Autoencoder (SAE) feature activations from the Gemma-2B language model for labeled sentences. This expensive computation phase runs once per dataset, generating comprehensive activation data that can be analyzed downstream in **s2_find_top_features**.

## Overview

This stage extracts SAE feature activations by:
- **Processing labeled sentences** through the Gemma-2B model
- **Extracting activations** from all 135 SAE configurations across Gemma-2B's 26 layers
- **Generating cumulative token fragments** to capture temporal activation patterns
- **Saving comprehensive activation data** (~2.2 million features × number of sentences)

The output enables downstream analysis in **s2_find_top_features** to identify interpretable features that correlate with personality traits.

## Motivation

Traditional approaches to interpretability often focus on individual neurons or small feature sets. However, SAEs trained on language model activations provide thousands of sparse, interpretable features per layer. By extracting activations from ALL features across ALL layers, we:

1. **Enable exhaustive search**: Extract once, analyze many times without re-running expensive GPU computation
2. **Avoid selection bias**: Don't pre-filter features - let the analysis stage determine what's interpretable
3. **Support temporal analysis**: Cumulative fragments capture when features activate during sentence processing
4. **Enable flexible analysis**: The same extraction supports different downstream analysis approaches

## Feature Extraction Process

### `extract_sae_features.py`
**Purpose:** Process labeled sentences through Gemma-2B and extract ALL SAE activations

**Runtime:** Hours to days (GPU-intensive)
**Run frequency:** Once per dataset
**Output:** Comprehensive .npz files with activation matrices and metadata

This script:
1. Loads labeled sentences with concept strength labels (0-3)
2. Generates cumulative token fragments for each sentence
3. For each of 135 SAE configurations:
   - Extracts hidden states from the target layer
   - Computes SAE activations for all 16,384 features
   - Saves complete activation data with metadata
4. Supports resume from checkpoint for long-running jobs

**Key insight:** By saving ALL activations once, downstream analysis in **s2_find_top_features** can experiment with different parameters without re-running expensive GPU computation.

### Next Step: Feature Analysis

After extraction completes, proceed to **s2_find_top_features** to:
- Rank features by interpretability metrics
- Generate Neuronpedia links for validation
- Extract example sentences for top features
- Identify the most interpretable features for your concept

## Methodology

### Cumulative Fragment Generation

Rather than analyzing only complete sentences, we generate **cumulative token fragments** to understand when features activate during sentence processing:

```
"I lied to my boss" � ["I", "I lied", "I lied to", "I lied to my", "I lied to my boss"]
```

This allows temporal analysis downstream: does a "dishonesty" feature activate early ("I lied") or require full context? While the extraction saves all fragment data, the primary analysis in **s2_find_top_features** focuses on final tokens (complete sentences).

## Four-Category Labeling Framework

We use **0-3 labels** (not binary) based on Anthropic's Constitutional AI framework:

- **Label 3**: Feature cleanly identifies the activating text
  - Example: "I vandalized the funeral home" (aggression)

- **Label 2**: Feature loosely related to the text
  - Example: "I threatened to ground my son for disobeying" (aggression present but justified)

- **Label 1**: Feature vaguely related but not exhibited
  - Example: "Gina stomped on tiny purple flowers" (aggressive action, but on flowers)

- **Label 0**: Feature completely irrelevant
  - Example: "Chris avoided stepping on a butterfly" (neutral action)

**Why 4 categories?** Having granular labels (0-3) enables downstream analysis to use monotonicity checks and correlation metrics that provide stronger evidence of interpretability than binary classification.

## Usage

For detailed technical instructions and CLI examples, see [USAGE.md](USAGE.md).

**Quick start:**
```bash
# Extract all SAE activations (runs for several hours)
uv run python extract_sae_features.py \
  --input ../s0_dataset_generation/expanded_data/aggression.csv \
  --output ./aggression_features

# Resume if interrupted
uv run python extract_sae_features.py \
  --input ../s0_dataset_generation/expanded_data/aggression.csv \
  --output ./aggression_features \
  --resume

# After completion, proceed to s2_find_top_features for analysis
```

## Directory Structure

```
s1_compute_all_features/
   dishonesty_features/           # Extracted activations for dishonesty dataset
      batch_progress.json         # Resume checkpoint
      layer_0_l0_105_activations.npz
      layer_1_l0_102_activations.npz
      ...                          # ~135 .npz files (one per SAE config)
   aggression_features/           # Extracted activations for aggression dataset
   brave_features/                # Extracted activations for bravery dataset
   ...                            # Additional feature directories per trait

   gemma_scope_16k_configurations.json  # SAE configuration metadata
   extract_sae_features.py        # Feature extraction script
   README.md                      # This file
   USAGE.md                       # Detailed usage guide
```

## Key Design Decisions

1. **Exhaustive extraction**: Extract ALL features from ALL layers to enable flexible downstream analysis without re-running expensive GPU computation.

2. **Resume capability**: Saves progress after each configuration. Long jobs can resume from checkpoint if interrupted.

3. **Cumulative fragments**: Generates token-by-token fragments to enable temporal analysis of when features activate.

4. **Generic concept framework**: Uses `concept_strength_label` rather than trait-specific names. The same pipeline works for any 0-3 labeled concept.

## Computational Requirements

- **GPU**: Recommended (CUDA-compatible, 16GB+ VRAM)
- **CPU fallback**: Supported but 10-50× slower
- **Memory**: 32GB+ RAM recommended for large batch processing
- **Storage**: ~1-5GB per trait dataset (135 × ~10-50MB .npz files)
- **Runtime**: 2-12 hours on GPU, 1-3 days on CPU

## Limitations

1. **Coverage**: Only analyzes width_16k SAEs. Larger SAEs (32k, 64k, 131k) exist but would multiply computational cost.

2. **Single model**: Extraction is specific to Gemma-2B. Features may not transfer to other models.

3. **Storage intensive**: Each trait requires 1-5GB of activation data.

## Next Steps

After extraction completes, proceed to **../s2_find_top_features** to analyze the extracted activations and identify interpretable features.
