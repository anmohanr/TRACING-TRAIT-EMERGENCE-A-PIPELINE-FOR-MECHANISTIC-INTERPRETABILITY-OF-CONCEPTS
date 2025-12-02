# s5_classifier_training System Architecture

**Last updated:** 2025-11-01 (Framework reorganization - scripts moved to `framework/`, docs to `docs/`)

## Table of Contents
1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [System Components](#system-components)
4. [Data Flow](#data-flow)
5. [Directory Structure](#directory-structure)
6. [Task Types and Model Selection](#task-types-and-model-selection)
7. [Workflow Patterns](#workflow-patterns)
8. [Upstream Dependencies](#upstream-dependencies)
9. [Downstream Integration](#downstream-integration)

## Overview

The s5_classifier_training system is the final stage of a personality trait classification pipeline. It implements an iterative experimentation framework for training classifiers that can either:
1. Predict SAE (Sparse Autoencoder) feature activations from embeddings/hidden states
2. Predict concept labels (trait strength 0-3) from neural network representations

### Position in Pipeline

```
s0_dataset_generation    → Generate labeled sentences for traits
         ↓
s1_compute_all_features  → Extract embeddings, hidden states, SAE activations
         ↓
s2_find_top_features     → Identify most predictive SAE features
         ↓
s3_feature_activation_viz → Visualize feature activations
         ↓
s4_classifier_data_prep  → Generate training datasets (NPZ files)
         ↓
[s5_classifier_training] → Train and evaluate classifiers ← YOU ARE HERE
```

## Design Philosophy

### 1. Experimentation-First Design
The system prioritizes rapid iteration and experimentation over production deployment:
- **Working Directory Pattern**: A persistent `working/` directory serves as a sandbox
- **Checkpoint System**: Good experiments are "promoted" to versioned checkpoints
- **Configuration-Driven**: YAML configs allow quick hyperparameter changes
- **Tool Automation**: Scripts for common tasks (checkpoint, compare, new experiment)

### 2. Why This Architecture?

**Problem**: ML experimentation often leads to:
- Lost experiments (overwritten results)
- Unclear progression (which model was better?)
- Reproducibility issues (what dataset/config was used?)
- Cluttered directories (hundreds of experiment files)

**Solution**: The working + checkpoint pattern provides:
- **Freedom to experiment** in `working/` without fear
- **Intentional preservation** via checkpoints
- **Clear progression** through version numbers
- **Full reproducibility** with saved configs and datasets

### 3. Design Principles

1. **Explicit is better than implicit**: Every checkpoint captures full state
2. **Fail loudly**: Clear error messages rather than silent failures
3. **Human-readable first**: JSON/YAML over binary formats where possible
4. **Automation-friendly**: All tools callable programmatically
5. **Incremental complexity**: Start simple (v0), iterate to complex

## System Components

### Core Scripts (in framework/ directory)

```
framework/
│
├─── train.py ──────────┐ ← Main training engine
│    └────────────┐     │
│                 │     │
├─── shared_utils.py ←──┘ ← Data loading, metrics, utilities
│
├─── checkpoint.py       ← Promotes working/ to vX/
│
├─── compare.py          ← Compares versions across experiments
│
└─── new_experiment.py   ← Starts new experiment from checkpoint
```

### Component Responsibilities

| Component | Purpose | Inputs | Outputs |
|-----------|---------|---------|---------|
| framework/train.py | Train classifier | config.yaml, dataset.config | model.pkl, results.json |
| framework/checkpoint.py | Save experiment | working/* | vX_name/* |
| framework/compare.py | Compare results | .metadata.json files | Comparison table |
| framework/new_experiment.py | Initialize experiment | vX/* | working/* setup |
| framework/shared_utils.py | Common functions | Various | Various |

## Data Flow

### 1. Dataset Loading Flow

```
dataset.config (text file)
    ↓ contains path
"../../../s4_classifier_data_prep/generated_datasets/dishonesty_xxx.npz"
    ↓ shared_utils.load_dataset()
NPZ file loaded
    ↓ extracts
X (features), y (targets), metadata
    ↓ infers task type
Classification or Regression
    ↓ creates appropriate model
LogisticRegression/Ridge/RandomForest/SVM
    ↓ trains with cross-validation
Trained model + metrics
    ↓ saves
model.pkl + results.json
```

### 2. Checkpoint Flow

```
working/
├── config.yaml         ← User edits
├── dataset.config      ← Points to data
├── experiment_notes.md ← User notes
├── model.pkl          ← After training
└── results.json       ← After training
    ↓ checkpoint.py --name "improved"
v1_improved/
├── config.yaml        ← Copied
├── dataset.config     ← Copied
├── model.pkl         ← Copied
├── results.json      ← Copied
├── experiment_notes.md ← Copied
├── .metadata.json    ← Auto-generated
└── README.md         ← Auto-generated
```

### 3. Comparison Flow

```
d1/v0/.metadata.json ─┐
d1/v1/.metadata.json ─┼→ compare.py → Formatted table
d1/v2/.metadata.json ─┘                showing metrics
```

## Directory Structure

### Hierarchical Organization

```
s5_classifier_training/
│
├── README.md              ← Research-style overview
├── USAGE.md               ← Practical commands and workflows
│
├── framework/             ← Core training framework (NEW: 2025-11-01)
│   ├── train.py
│   ├── shared_utils.py
│   ├── checkpoint.py
│   ├── compare.py
│   └── new_experiment.py
│
├── docs/                  ← Documentation (NEW: 2025-11-01)
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   ├── DATA_FORMAT.md
│   ├── DEVELOPMENT.md
│   ├── KNOWN_ISSUES.md
│   └── TROUBLESHOOTING.md
│
├── d1_embedding_to_feature/    ← Dataset Type 1
│   ├── working/                 ← Active experimentation
│   │   ├── config.yaml
│   │   ├── dataset.config
│   │   ├── experiment_notes.md
│   │   ├── model.pkl (after training)
│   │   └── results.json (after training)
│   │
│   └── v*_name/                 ← Saved checkpoints
│       ├── config.yaml
│       ├── dataset.config
│       ├── experiment_notes.md
│       ├── model.pkl
│       ├── results.json
│       ├── .metadata.json (auto-generated)
│       └── README.md (auto-generated)
│
├── d2_hidden_to_feature/        ← Dataset Type 2
│   └── (same structure as d1)
│
└── d3_hidden_to_concept/        ← Dataset Type 3
    └── (same structure as d1)
```

### Dataset Types Explained

| Directory | Task | Input → Output | Purpose |
|-----------|------|----------------|---------|
| **d1_embedding_to_feature** | Regression | 2304-dim embeddings → SAE feature activation | Can embeddings predict features? |
| **d2_hidden_to_feature** | Regression | 2304-dim hidden states → SAE feature activation | Can hidden states predict features? |
| **d3_hidden_to_concept** | Classification | 27648-dim concatenated hidden → Label (0-3) | Can hidden states predict concepts? |

### Why Three Dataset Types?

The three datasets test different hypotheses:

1. **d1**: Tests if early representations (embeddings) contain feature information
   - Result: Poor (R² ≈ 0.03) - embeddings too far from features

2. **d2**: Tests if same-layer hidden states predict features
   - Result: Perfect (R² ≈ 1.0) - hidden states directly feed SAE

3. **d3**: Tests if multi-layer representations predict human concepts
   - Result: Moderate (68% accuracy) - promising but needs more data

## Task Types and Model Selection

### Task Type Inference

The system automatically infers task type from the target variable `y`:

```python
def infer_task_type(y):
    unique_vals = np.unique(y)
    if len(unique_vals) <= 10 and np.allclose(y, y.astype(int)):
        return 'classification'
    else:
        return 'regression'
```

**Logic**:
- ≤10 unique integer values → Classification
- Continuous values or >10 unique → Regression

### Model Mapping

| model_type in config | Classification Task | Regression Task |
|---------------------|-------------------|----------------|
| 'logistic' | LogisticRegression | Ridge |
| 'random_forest' | RandomForestClassifier | RandomForestRegressor |
| 'svm' | SVC | SVR |

**Design Decision**: When `model_type='logistic'` is used for regression, the system uses Ridge regression instead. This allows consistent config naming while providing appropriate models.

### Hyperparameter Translation

For Ridge regression when model_type='logistic':
- Config `C` parameter → Ridge `alpha = 1/C`
- This maintains consistency: higher C = less regularization

## Workflow Patterns

### Pattern 1: Iterative Refinement

```
Experiment in working/ → Good result? → Checkpoint as v0
                      ↓
              Modify config/data
                      ↓
        Train again → Better? → Checkpoint as v1
                      ↓
                   Continue...
```

### Pattern 2: A/B Comparison

```
From v0 → new_experiment.py → working/ → Change model only → Train → v1
                            ↘ working/ → Change data only → Train → v2

Compare v0, v1, v2 to isolate variable effects
```

### Pattern 3: Grid Search (Manual)

```python
for C in [0.1, 1.0, 10.0]:
    # Edit config.yaml with C value
    # Run train.py
    # Check results.json
    # If best so far, checkpoint
```

### Pattern 4: Autonomous Agent Workflow

```python
while not converged:
    # Load current results
    results = json.load('working/results.json')

    # Decide next experiment based on results
    if results['mae'] > threshold:
        # Try different model
        update_config(model_type='random_forest')
    else:
        # Try more data
        generate_larger_dataset()

    # Train
    subprocess.run(['python', 'train.py'])

    # Checkpoint if improved
    if improved:
        subprocess.run(['python', 'checkpoint.py', '--name', name])
```

## Upstream Dependencies

### From s4_classifier_data_prep

**Expected NPZ format**:
```python
{
    'X': ndarray,           # Features [n_samples, n_features]
    'y': ndarray,           # Targets [n_samples]
    'meta_strategy': str,   # Generation strategy
    'meta_input_layers': list,  # Which layers used
    'norm_mean': ndarray,   # Normalization parameters
    'norm_std': ndarray,
    # ... other metadata
}
```

### Dataset Generation Strategies

| Strategy | Description | X dimensions | y type |
|----------|-------------|--------------|--------|
| embedding-to-feature | Embeddings → SAE activation | 2304 | Continuous |
| hidden-to-feature | Hidden states → SAE activation | 2304 | Continuous |
| hidden-to-concept | Hidden states → Concept label | Varies | Integer 0-3 |

## Downstream Integration

### Model Usage

Trained models (`.pkl` files) can be loaded and used for:

```python
import pickle

# Load model
with open('d3/v0_baseline/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new data
predictions = model.predict(new_hidden_states)
```

### Results Analysis

Results can be aggregated for meta-analysis:

```python
# Collect all results
all_results = []
for version_dir in Path('d1').glob('v*'):
    with open(version_dir / 'results.json') as f:
        all_results.append(json.load(f))

# Analyze trends
plot_accuracy_vs_dataset_size(all_results)
```

### Integration with Larger Systems

The trained classifiers can be integrated into:
1. **Real-time inference pipelines**: Load model, predict on streaming text
2. **Batch analysis systems**: Process large text corpora
3. **Evaluation frameworks**: Test on held-out datasets
4. **Model ensembles**: Combine multiple versions

## Architecture Decisions and Rationale

### Why Separate Dataset Types (d1, d2, d3)?

**Alternative considered**: Single directory with dataset as config parameter

**Chosen approach benefits**:
- Clear separation of fundamentally different tasks
- Easier to track progress per task type
- Prevents accidental comparison of incomparable models
- Cleaner version history per task

### Why working/ + Checkpoints Instead of Auto-versioning?

**Alternative considered**: Auto-increment version on every train

**Chosen approach benefits**:
- Prevents clutter from failed/test runs
- Intentional preservation (only checkpoint good results)
- Clear workspace for experimentation
- Reduces cognitive load (not tracking v47, v48, v49...)

### Why Configuration Files Instead of Command-line Arguments?

**Alternative considered**: `train.py --model logistic --C 1.0 ...`

**Chosen approach benefits**:
- Reproducibility (config saved with checkpoint)
- Complex configs easier to manage
- Self-documenting experiments
- Easy to share/version control

### Why Ridge for Regression When model_type='logistic'?

**Design choice**: Map 'logistic' to appropriate linear model

**Rationale**:
- Keeps config naming consistent across tasks
- Both are linear models with similar hyperparameters
- Users think "linear model" not specific algorithm
- Avoids need for separate regression_model_type field

## Common Architectural Patterns

### 1. Fail-Fast Pattern
- Validate inputs early (dataset exists, config valid)
- Clear error messages with suggested fixes
- No silent failures or defaults

### 2. Immutable Checkpoints
- Once created, checkpoints are never modified
- New experiments start from copies, not modifications
- Preserves experimental history

### 3. Metadata Pyramid
```
.metadata.json (machine-readable, comprehensive)
       ↓
README.md (human-readable summary)
       ↓
RESULTS_TRACKER.md (high-level insights)
```

### 4. Progressive Enhancement
- Start with simple baseline (v0)
- Each version builds on learnings
- Document what changed and why
- Track cumulative improvements

## System Boundaries

### What the System DOES

✅ Train sklearn models on prepared datasets
✅ Track experiments and results
✅ Compare model performance
✅ Enable iterative refinement
✅ Support autonomous experimentation

### What the System DOES NOT Do

❌ Deep learning (PyTorch models not fully implemented)
❌ Hyperparameter optimization (manual only)
❌ Dataset generation (done in s4)
❌ Feature engineering (done in s4)
❌ Production deployment (research only)
❌ Real-time inference (batch only)
❌ Distributed training
❌ Model interpretation/explanation

## Performance Characteristics

### Scalability

| Aspect | Current Limit | Bottleneck | Scaling Strategy |
|--------|--------------|------------|------------------|
| Dataset size | ~10,000 samples | Memory | Batch processing |
| Feature dims | ~30,000 | Memory | Dimensionality reduction |
| Versions | Unlimited | Filesystem | Archive old versions |
| Parallel experiments | 1 per dataset | File locks | Multiple working dirs |

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Load dataset | O(n×d) | <1 second |
| Train logistic | O(n×d²) | <1 second |
| Train RF | O(n×d×trees) | 1-10 seconds |
| Cross-validation | O(folds×train) | 5× training time |
| Checkpoint | O(files) | <1 second |
| Compare | O(versions) | <1 second |

Where n=samples, d=features

## Future Architecture Considerations

### Potential Enhancements

1. **Plugin Architecture**: Dynamic model loading from plugins/
2. **Distributed Training**: Ray/Dask integration for large datasets
3. **Experiment Tracking**: MLflow/Weights&Biases integration
4. **Auto-ML**: Hyperparameter optimization with Optuna
5. **Pipeline Mode**: Chain multiple s5 experiments
6. **Model Registry**: Central model store with versions
7. **A/B Testing**: Statistical significance testing
8. **Continuous Integration**: Auto-train on dataset updates

### Breaking Changes to Avoid

1. Changing .metadata.json schema (breaks compare.py)
2. Changing checkpoint directory structure (breaks tools)
3. Changing NPZ format expectations (breaks data loading)
4. Renaming core scripts (breaks automation)
5. Changing config.yaml structure (breaks reproducibility)

## Conclusion

The s5_classifier_training architecture prioritizes experimentation velocity and reproducibility over production concerns. Its working/checkpoint pattern provides both freedom to experiment and discipline in preserving results. The system is designed to be both human-friendly (clear workflows) and automation-friendly (scriptable tools), making it suitable for both manual experimentation and autonomous ML research.