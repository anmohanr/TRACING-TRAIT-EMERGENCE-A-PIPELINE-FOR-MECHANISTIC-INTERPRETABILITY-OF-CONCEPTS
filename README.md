# Tracing Trait Emergence: A Pipeline for Mechanistic Interpretability of Concepts

A six-stage pipeline for discovering and validating interpretable Sparse Autoencoder (SAE) features that correspond to personality traits in the Gemma-2B language model. This framework enables systematic identification of neural features that activate on concepts like dishonesty, aggression, caring, bravery, and unfairness.

## Overview

This pipeline addresses a core challenge in mechanistic interpretability: **How do we find interpretable features that reliably detect abstract concepts?**

Rather than manually searching through millions of SAE features, this framework provides:

1. **Systematic dataset generation** with granular 0-3 labels based on Anthropic's Constitutional AI framework
2. **Exhaustive feature extraction** across all 135 SAE configurations in Gemma-2B
3. **Automated feature ranking** using correlation, monotonicity, and selectivity metrics
4. **Visual validation** of temporal activation patterns
5. **Flexible classifier training** to test multiple hypotheses about optimal representations

## Pipeline Stages

```
s0_dataset_generation     Generate labeled sentences (0-3 scale) for personality traits
         ↓
s1_compute_all_features   Extract SAE activations from Gemma-2B (GPU-intensive)
         ↓
s2_find_top_features      Rank features by interpretability metrics
         ↓
s3_feature_activation_graphs   Visualize temporal activation patterns
         ↓
s4_classifier_data_prep   Generate training datasets for different strategies
         ↓
s5_classifier_training    Train and evaluate classifiers with rigorous methodology
```

### Stage Descriptions

| Stage | Purpose | Runtime | Output |
|-------|---------|---------|--------|
| **s0** | Generate balanced datasets with 0-3 labels using GPT-4 + Hendrycks seeds | Minutes | CSV files (~1000 sentences/trait) |
| **s1** | Extract embeddings, hidden states, and SAE activations for all layers | Hours (GPU) | NPZ files (~2.2M features) |
| **s2** | Rank features by Pearson correlation, monotonicity, and selectivity | Minutes | Ranked CSV + Neuronpedia links |
| **s3** | Plot token-by-token activation trajectories for top features | Minutes | PNG visualizations |
| **s4** | Create flexible datasets for different classifier strategies | Seconds | NPZ datasets |
| **s5** | Train classifiers with proper train/test splits and cross-validation | Minutes | Models + metrics JSON |

## Key Features

### Four-Category Labeling (0-3 Scale)
Unlike binary classification, we use granular labels based on Anthropic's framework:
- **0**: Feature completely irrelevant
- **1**: Feature vaguely related but not exhibited
- **2**: Feature loosely related to the text
- **3**: Feature cleanly identifies the activating text

This enables monotonicity checks and correlation metrics that provide stronger evidence of interpretability.

### Interpretability Metrics
Features are ranked using multiple orthogonal metrics:
- **Pearson/Spearman correlation** with concept labels
- **Monotonicity**: mean₀ ≤ mean₁ ≤ mean₂ ≤ mean₃
- **Selectivity**: High true positive rate, low false positive rate
- **Separation**: Discriminative power between neutral and strong examples

### Neuronpedia Integration
Every identified feature includes a link to [Neuronpedia](https://neuronpedia.org) for human validation, enabling verification that top-activating examples from internet text align with the target concept.

### Rigorous Classifier Evaluation
The training framework enforces proper methodology:
- 80/20 train/test splits with held-out test sets
- 5-fold cross-validation on training data only
- Clear metric separation (cv_mean, test_accuracy, train_accuracy)
- Versioned checkpoints for reproducibility

## Installation

### Prerequisites
- Python 3.13+
- CUDA-compatible GPU (recommended for s1, 16GB+ VRAM)
- 32GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone git@github.com:anmohanr/TRACING-TRAIT-EMERGENCE-A-PIPELINE-FOR-MECHANISTIC-INTERPRETABILITY-OF-CONCEPTS.git
cd TRACING-TRAIT-EMERGENCE-A-PIPELINE-FOR-MECHANISTIC-INTERPRETABILITY-OF-CONCEPTS

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Environment Variables

Create a `.env` file in `s0_dataset_generation/` for dataset generation:
```
OPENAI_API_KEY=your-api-key-here
```

## Quick Start

### 1. Generate a dataset
```bash
cd s0_dataset_generation
uv run python expand_dataset.py \
  --trait "Dishonesty" \
  --seed-file initial_hendrycks_data/dishonesty.csv \
  --label3-count 300 --label2-count 250 --label1-count 150 --label0-count 100
```

### 2. Extract SAE features (GPU recommended)
```bash
cd s1_compute_all_features
uv run python extract_sae_features.py \
  --input ../s0_dataset_generation/expanded_data/dishonesty.csv \
  --output ./dishonesty_features
```

### 3. Find top interpretable features
```bash
cd s2_find_top_features
uv run python analyze_features.py \
  --input ../s1_compute_all_features/dishonesty_features \
  --output ./results/dishonesty_results
```

### 4. Visualize activation patterns
```bash
cd s3_feature_activation_graphs
uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_results \
  --s1-data ../s1_compute_all_features/dishonesty_features \
  --output ./plots --top-n 10
```

### 5. Prepare classifier data
```bash
cd s4_classifier_data_prep
uv run python generate_datasets.py \
  --strategy hidden-to-concept \
  --input ../s1_compute_all_features/dishonesty_features \
  --layers 0-5 \
  --output generated_datasets/dishonesty_h0-5_to_concept.npz
```

### 6. Train classifiers
```bash
cd s5_classifier_training
uv run python framework/train.py d3_hidden_to_concept
```

## Supported Personality Traits

The pipeline includes seed data for five traits from the Hendrycks dataset:
- Dishonesty/Lying
- Aggression
- Caring
- Bravery
- Unfairness

Additional traits can be added by creating seed CSV files in `s0_dataset_generation/initial_hendrycks_data/`.

## Classifier Strategies

Three strategies are implemented for systematic comparison:

| Strategy | Input | Output | Research Question |
|----------|-------|--------|-------------------|
| `embedding-to-feature` | Embeddings [2048] | SAE feature activation | Can embeddings predict features without SAE? |
| `hidden-to-feature` | Hidden states [2048] | SAE feature activation | Can we extract single features cheaply? |
| `hidden-to-concept` | Hidden states [N×2048] | Concept label (0-3) | Do raw representations suffice for classification? |

## Project Structure

```
├── s0_dataset_generation/
│   ├── initial_hendrycks_data/   # Seed sentences (20 per trait)
│   ├── expanded_data/            # Generated datasets
│   ├── expand_dataset.py         # Main pipeline orchestrator
│   ├── generate_sentences.py     # GPT-4 sentence generation
│   └── label_sentences.py        # Automated labeling
│
├── s1_compute_all_features/
│   ├── extract_sae_features.py   # GPU feature extraction
│   └── gemma_scope_16k_configurations.json
│
├── s2_find_top_features/
│   ├── analyze_features.py       # Feature ranking
│   └── results/                  # Ranked features + examples
│
├── s3_feature_activation_graphs/
│   ├── visualize_activations.py  # Temporal plots
│   └── plotting_utils.py
│
├── s4_classifier_data_prep/
│   ├── generate_datasets.py      # Dataset generation
│   ├── load_sentence_data.py     # Data loading utilities
│   └── utils.py
│
├── s5_classifier_training/
│   ├── framework/                # Training infrastructure
│   │   ├── train.py
│   │   ├── checkpoint.py
│   │   └── compare.py
│   ├── d1_embedding_to_feature/
│   ├── d2_hidden_to_feature/
│   └── d3_hidden_to_concept/
│
├── pyproject.toml
└── README.md
```

## Computational Requirements

| Stage | GPU | RAM | Storage | Runtime |
|-------|-----|-----|---------|---------|
| s0 | No | 4GB | <100MB | Minutes |
| s1 | Recommended | 32GB | 1-5GB/trait | Hours |
| s2 | No | 16GB | <100MB | Minutes |
| s3 | No | 8GB | ~100MB | Minutes |
| s4 | No | 8GB | ~100MB | Seconds |
| s5 | No | 8GB | <100MB | Minutes |

## Documentation

Each stage includes detailed documentation:
- `README.md` - Overview and methodology
- `USAGE.md` - CLI examples and troubleshooting

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Gemma Scope](https://huggingface.co/google/gemma-scope) for SAE weights
- [Neuronpedia](https://neuronpedia.org) for feature visualization
- [Hendrycks et al.](https://github.com/hendrycks/ethics) for seed dataset
- Anthropic's Constitutional AI framework for labeling methodology
