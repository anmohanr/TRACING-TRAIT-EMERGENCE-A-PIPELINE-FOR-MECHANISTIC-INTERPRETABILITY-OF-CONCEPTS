# Classifier Training (s5_classifier_training)

Experimental framework for training and evaluating machine learning classifiers on SAE-derived features and hidden states from the Gemma-2B language model. This stage takes feature activations extracted in **s1_compute_all_features** and trains models to predict either SAE feature activations or personality trait concept labels.

## Overview

This framework provides a rigorous experimental environment for training classifiers across three distinct prediction tasks:

- **d1_embedding_to_feature**: Embedding vectors → SAE feature activations (regression)
- **d2_hidden_to_feature**: Hidden states → SAE feature activations (regression)
- **d3_hidden_to_concept**: Hidden states → Concept strength labels 0-3 (classification)

Each experiment follows a consistent workflow: configure hyperparameters, train with cross-validation, evaluate on held-out test sets, checkpoint successful experiments, and compare results across model variants.

## Motivation

Traditional machine learning experimentation often suffers from methodological issues that compromise result validity:

1. **Train/test contamination**: Evaluating on training data produces misleadingly high metrics
2. **Inconsistent evaluation**: Different experiments use different train/test splits, making comparisons unreliable
3. **Poor reproducibility**: Lack of versioning makes it difficult to replicate or build upon previous experiments
4. **Metric confusion**: Mixing training accuracy, CV scores, and test accuracy leads to misinterpretation

This framework addresses these issues through:

### Proper Statistical Rigor
- **Default 80/20 train/test split**: Held-out test set never seen during training or hyperparameter tuning
- **Cross-validation on training set only**: 5-fold CV provides robust performance estimates without test set leakage
- **Clear metric separation**: Training accuracy for debugging, CV mean for model selection, test accuracy for final validation
- **Small CV-test gaps indicate good generalization**: Models that overfit show large discrepancies between CV and test performance

### Clean Metric Naming
All metrics use explicit prefixes to eliminate confusion:
- `cv_mean`, `cv_std` - Cross-validation scores (primary model selection metric)
- `test_accuracy`, `test_f1`, `test_mae`, `test_r2` - Held-out test set performance
- `train_accuracy` - Training set performance (sanity check for overfitting)

### Reproducible Experimentation
- **Versioned checkpoints**: Each successful experiment saved as `v0_name/`, `v1_name/`, etc.
- **Complete provenance**: Every checkpoint includes dataset hash, config, metrics, and timestamp
- **Compare tool**: Easy comparison of all versions to track progress and identify best models

## Architecture

### Directory Structure

```
s5_classifier_training/
├── framework/           # Shared training infrastructure
│   ├── train.py        # Main training script with CV and train/test split
│   ├── checkpoint.py   # Version and save successful experiments
│   ├── compare.py      # Compare metrics across all versions
│   ├── shared_utils.py # Common utilities (metrics, data loading)
│   └── new_experiment.py # Start new experiment from previous version
├── docs/               # Detailed documentation
├── d1_embedding_to_feature/  # Dataset type 1
│   ├── working/        # Active experimentation directory
│   ├── v0_baseline/    # Versioned checkpoint
│   ├── v1_name/        # Another checkpoint
│   └── ...
├── d2_hidden_to_feature/     # Dataset type 2
└── d3_hidden_to_concept/     # Dataset type 3
```

### Experiment Workflow

Each dataset directory (d1, d2, d3) contains:

**working/**: Active experimentation directory
- `config.yaml` - Model and training hyperparameters
- `dataset.config` - Path to dataset .npz file
- `experiment_notes.md` - Research notes and observations
- `model.pkl` - Trained model (after running train.py)
- `results.json` - Complete metrics and metadata

**v{N}_{name}/**: Versioned checkpoints
- Snapshot of working/ at a point in time
- Includes all files plus `.metadata.json` and `README.md`
- Immutable record for reproducibility

### Training Process

The training pipeline in `framework/train.py` follows these steps:

1. **Load dataset** from path specified in `dataset.config`
2. **Split data** into train (80%) and test (20%) sets using configured random seed
3. **Cross-validation** on training set only (5-fold by default)
   - Provides robust performance estimate
   - Used for hyperparameter tuning and model selection
4. **Train final model** on full training set
5. **Evaluate** on both train set (sanity check) and held-out test set
6. **Save model and results** with complete provenance information

### Key Design Principles

**CV mean is the primary metric**: Cross-validation score on the training set is the most trustworthy performance indicator. It's computed on data the model hasn't seen and is averaged across 5 folds for stability.

**Test set is validation only**: The held-out test set provides a final sanity check. Small CV-test gaps (<5 percentage points) indicate good generalization. Large gaps suggest overfitting or lucky train/test splits.

**Training accuracy reveals overfitting**: If train accuracy >> test accuracy, the model is memorizing training data. The gap between train and test indicates overfitting severity.

## Results Format

All experiments produce a `results.json` with this structure:

```json
{
  "n_samples_total": 1020,
  "n_samples_train": 816,
  "n_samples_test": 204,
  "n_features": 2304,
  "model_type": "SVC",

  "cv_scores": [0.85, 0.82, 0.85, 0.85, 0.85],
  "cv_mean": 0.846,
  "cv_std": 0.012,

  "test_accuracy": 0.858,
  "test_f1": 0.831,
  "train_accuracy": 0.907,

  "training_time_sec": 4.39,
  "config": {...},
  "metadata": {...}
}
```

## Supported Models

All models support both regression and classification tasks automatically:

- **Logistic Regression / Ridge**: Fast baseline, interpretable coefficients
- **Random Forest**: Handles non-linear relationships, provides feature importance
- **SVM (RBF/Linear/Poly)**: Powerful for high-dimensional data, captures non-linear patterns

## Next Steps

See **USAGE.md** for practical commands and workflow examples.

See **docs/** for detailed documentation:
- `API_REFERENCE.md` - Detailed API documentation
- `ARCHITECTURE.md` - System design and implementation details
- `DATA_FORMAT.md` - Dataset and results file formats
- `DEVELOPMENT.md` - Development workflow and best practices
- `TROUBLESHOOTING.md` - Common issues and solutions

## Experimental Best Practices

1. **Always use working/ for active experimentation**: Never edit versioned checkpoints
2. **Checkpoint successful experiments**: Use `checkpoint.py` to save working/ when results are good
3. **Trust CV scores over test scores**: CV mean is your primary metric for model selection
4. **Check train-test gap**: Large gaps indicate overfitting
5. **Compare systematically**: Use `compare.py` to track progress across experiments
6. **Document in experiment_notes.md**: Record hypotheses, observations, and next steps
