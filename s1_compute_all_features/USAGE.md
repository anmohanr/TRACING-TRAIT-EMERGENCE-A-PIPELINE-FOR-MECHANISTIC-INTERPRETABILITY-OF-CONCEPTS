# SAE Feature Discovery - Usage Guide

This directory contains scripts for exhaustive SAE feature extraction and analysis using the Gemma-2B language model.

The pipeline uses a **two-script approach**: expensive GPU-based extraction (Script 1) runs once, while fast CPU-based analysis (Script 2) runs many times with different parameters.

## Setup

### 1. Dependencies

Dependencies are managed via `uv` and should already be installed in the project:
- torch
- transformers
- huggingface_hub
- numpy
- pandas
- scipy

If dependencies are missing, install with:
```bash
uv add torch transformers huggingface_hub numpy pandas scipy
```

### 2. GPU Configuration (Recommended)

**For CUDA GPUs:**
```bash
# Check GPU availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Memory requirements:**
- Minimum: 16GB VRAM
- Recommended: 24GB+ VRAM for larger batches
- CPU fallback: Supported but 10-50× slower

### 3. HuggingFace Authentication (Optional)

Some SAE weights may require authentication:
```bash
# Login to HuggingFace
huggingface-cli login
```

### 4. Storage Requirements

- Input data: <10MB (CSV file)
- Activation files: ~1-5GB (135 .npz files)
- Results: <100MB (CSV + JSON)
- Temporary files: ~500MB during extraction
- **Total recommended:** 10GB free disk space

## Scripts

---

## Script 1: `extract_sae_features.py`

Extracts SAE feature activations from Gemma-2B for labeled sentences. This is the **compute-intensive** phase.

### Purpose
- Load labeled sentences with concept strength labels (0-3)
- Process through Gemma-2B model across all 26 layers
- Extract activations from 135 SAE configurations
- Save comprehensive activation data for downstream analysis

### Expected Runtime
- **GPU (recommended):** 2-12 hours depending on dataset size
- **CPU (fallback):** 1-3 days
- Progress is saved after each configuration (resume capability)

### Usage

**Basic usage:**
```bash
uv run python extract_sae_features.py
```

Uses defaults:
- Input: `../s0_dataset_generation/expanded_data/aggression.csv`
- Output: `./all_sae_features/`
- Config: `./gemma_scope_16k_configurations.json`

**Custom input file:**
```bash
uv run python extract_sae_features.py \
  --input ../s0_dataset_generation/expanded_data/dishonesty.csv \
  --output ./dishonesty_features
```

**Resume from checkpoint:**
```bash
# If script was interrupted, resume from last completed configuration
uv run python extract_sae_features.py --resume
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | Path | `../s0_dataset_generation/expanded_data/aggression.csv` | Input CSV with labeled sentences |
| `--output` | Path | `./all_sae_features` | Output directory for .npz files |
| `--config` | Path | `./gemma_scope_16k_configurations.json` | SAE configuration file |
| `--resume` | Flag | False | Resume from last checkpoint |

### Input Format

CSV file with two columns:

```csv
sentence,concept_strength_label
"I vandalized the funeral home.",3
"I threatened to ground my son for disobeying the rules.",1
"Ann offered to pay the vet bill after her dog bit the neighbor's dog.",0
```

**Requirements:**
- Column 1: `sentence` - Text to analyze (string)
- Column 2: `concept_strength_label` - Label from 0-3 (integer)
- No header row modifications (keep exact column names)

### Output Format

One `.npz` file per SAE configuration (135 total):

```
all_sae_features/
├── layer_0_l0_13_activations.npz
├── layer_0_l0_25_activations.npz
├── layer_1_l0_10_activations.npz
...
└── layer_25_l0_141_activations.npz
```

Each `.npz` file contains:

```python
{
    'activations': np.ndarray,              # [n_fragments, 16384] - SAE activations
    'fragments': np.ndarray,                # [n_fragments] - Text fragments
    'original_sentences': np.ndarray,       # [n_fragments] - Full sentences
    'concept_strength_labels': np.ndarray,  # [n_fragments] - Labels (0-3)
    'sentence_indices': np.ndarray,         # [n_fragments] - Parent sentence index
    'token_positions': np.ndarray,          # [n_fragments] - Token position in sentence
    'is_final_token': np.ndarray,           # [n_fragments] - Boolean: is last token?
    'fragment_lengths': np.ndarray,         # [n_fragments] - Number of words
    'layer': int,                           # Model layer (0-25)
    'average_l0': float,                    # SAE sparsity parameter
    'num_features': int                     # Number of features (16384)
}
```

### Progress Tracking

Progress is saved to `batch_progress.json`:

```json
{
  "last_completed": 42
}
```

If interrupted, use `--resume` to continue from configuration 43.

### Example Workflow

```bash
# Step 1: Start extraction (may run for hours)
uv run python extract_sae_features.py \
  --input ../s0_dataset_generation/expanded_data/aggression.csv \
  --output ./aggression_features

# Step 2: Check progress periodically
ls -lh ./aggression_features/*.npz | wc -l
# Should show increasing count (0 to 135)

# Step 3: If interrupted, resume
uv run python extract_sae_features.py \
  --input ../s0_dataset_generation/expanded_data/aggression.csv \
  --output ./aggression_features \
  --resume
```


---

## Next Steps: Feature Analysis

After extraction completes, proceed to **../s2_find_top_features** for feature analysis:

1. The analysis script will load the .npz files from your output directory
2. Calculate correlation and selectivity metrics for each feature
3. Rank features by interpretability
4. Generate Neuronpedia links for validation
5. Extract example sentences for top features

See **../s2_find_top_features/USAGE.md** for detailed analysis instructions.

## Troubleshooting

### "CUDA out of memory"

**Problem:** GPU runs out of memory during extraction

**Solutions:**
1. Close other GPU-using applications
2. Reduce dataset size for testing
3. Use CPU mode (slower but works):
   ```bash
   CUDA_VISIBLE_DEVICES="" uv run python extract_sae_features.py
   ```

### Resume Not Working

**Problem:** `--resume` flag doesn't continue from checkpoint

**Solutions:**
1. Check that `batch_progress.json` exists in output directory
2. Ensure `--output` path matches original run
3. Delete `batch_progress.json` to restart from beginning

### HuggingFace Download Errors

**Problem:** "Could not download SAE weights"

**Solutions:**
1. Check internet connection
2. Authenticate with HuggingFace:
   ```bash
   huggingface-cli login
   ```
3. Verify model access (some SAEs may be gated)

### File Not Found Errors

**Problem:** "CSV file not found"

**Solutions:**
1. Check file paths are absolute or relative to script location
2. Verify CSV has correct column names: `sentence`, `concept_strength_label`
3. Ensure CSV has valid 0-3 labels

## Performance Tips

### Speed up extraction:
- Use GPU instead of CPU (10-50× faster)
- Process multiple sentences per batch (automatic)
- Use SSD instead of HDD for output directory

### Monitor progress:
```bash
# Watch .npz files being created
watch -n 10 'ls output_directory/*.npz | wc -l'

# Check latest file modification time
ls -lt output_directory/*.npz | head -1
```

## Storage Management

**Disk usage:**
```bash
# Check extraction output size
du -sh aggression_features/

# Clean up temporary files
rm aggression_features/batch_progress.json  # after successful completion

# Archive results for long-term storage
tar -czf aggression_features_backup.tar.gz aggression_features/
```

## Cost Estimation

**Computational costs:**
- Extraction: 2-12 GPU hours (AWS p3.2xlarge: ~$3-15 per trait)
- Storage: ~5GB per trait (<$0.10/month on cloud)

**Time costs:**
- Initial setup: 30 minutes
- Feature extraction: 2-12 hours per trait (one-time)
- Can extract multiple traits in parallel on different machines

**Best practices:**
- Extract once on powerful GPU machine
- Save activation files for reuse
- Use `--resume` to avoid wasting failed runs
