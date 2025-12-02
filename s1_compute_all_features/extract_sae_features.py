#!/usr/bin/env python3
"""
SAE Feature Extraction Script

Extracts Sparse Autoencoder (SAE) feature activations, hidden states, and embeddings
from Gemma-2B model for labeled sentences across all 26 layer configurations.

This script processes sentences by:
1. Pre-loading all 26 SAE models into memory (~4GB)
2. For each sentence:
   - Generating cumulative token fragments
   - Extracting embedding layer output (before any transformer processing)
   - Extracting hidden states from all 26 layers
   - Computing SAE activations for all features from all configurations
   - Saving comprehensive activation data to a single NPZ file

Output format: One .npz file per sentence (sentence_000000.npz, sentence_000001.npz, etc.)
Each file contains:
  - embeddings: [n_fragments, 2048] - Raw embedding layer output
  - hidden_states: [26_configs, n_fragments, 2048] - Hidden states from all layers
  - activations: [26_configs, n_fragments, 16384] - SAE feature activations
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "../s0_dataset_generation/expanded_data/aggression.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "all_sae_features"
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "gemma_scope_16k_configurations.json"
PROGRESS_FILE_NAME = "sentence_progress.json"

# Model configuration
MODEL_NAME = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-res"

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SAE MODEL CLASSES
# =============================================================================

class JumpReLUSAE(nn.Module):
    """
    Jump ReLU Sparse Autoencoder with learned threshold.

    Architecture used by Gemma Scope SAEs for interpretable feature extraction.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse feature space."""
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_sae_for_config(config: Dict[str, Any], device: str) -> JumpReLUSAE:
    """
    Load SAE model from HuggingFace for a specific configuration.

    Args:
        config: Configuration dict with 'path' to SAE weights
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded JumpReLUSAE model
    """
    logger.info(f"  Downloading SAE weights from {SAE_REPO}...")
    path_to_params = hf_hub_download(
        repo_id=SAE_REPO,
        filename=config['path'],
        force_download=False,
    )

    logger.info("  Loading SAE parameters...")
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

    sae = JumpReLUSAE(pt_params['W_enc'].shape[0], pt_params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae.to(device)

    return sae


def load_all_saes(configs: List[Dict[str, Any]], device: str) -> Dict[int, JumpReLUSAE]:
    """
    Pre-load all SAE models into memory for efficient sentence-wise processing.

    This avoids reloading SAEs for each sentence, improving performance significantly.
    Memory usage: ~4GB for all 26 SAE models.

    Args:
        configs: List of configuration dicts
        device: Device to load models on ('cuda' or 'cpu')

    Returns:
        Dictionary mapping config_idx -> loaded SAE model
    """
    logger.info("=" * 80)
    logger.info("PRE-LOADING ALL SAE MODELS")
    logger.info("=" * 80)
    logger.info(f"Loading {len(configs)} SAE configurations into memory...")
    logger.info("This may take a few minutes but greatly improves processing speed.")
    logger.info("=" * 80)

    all_saes = {}

    for config_idx, config in enumerate(configs):
        layer = config['layer']
        l0 = config['average_l0']

        logger.info(f"Loading SAE {config_idx + 1}/{len(configs)}: Layer {layer}, L0 {l0}")

        sae = load_sae_for_config(config, device)
        all_saes[config_idx] = sae

    logger.info("=" * 80)
    logger.info(f"Successfully loaded {len(all_saes)} SAE models")
    logger.info("=" * 80)

    return all_saes


def gather_residual_activations(
    model: AutoModelForCausalLM,
    target_layer: int,
    inputs: torch.Tensor
) -> torch.Tensor:
    """
    Extract residual stream activations from a specific layer.

    Args:
        model: Gemma model
        target_layer: Layer index to extract from (0-25)
        inputs: Tokenized input tensor

    Returns:
        Hidden states from target layer
    """
    target_act = None

    def hook(_, __, outputs):
        nonlocal target_act
        target_act = outputs[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(hook)
    with torch.no_grad():
        _ = model.forward(inputs)
    handle.remove()

    return target_act


def generate_cumulative_fragments(
    sentence: str,
    tokenizer: AutoTokenizer
) -> Tuple[List[str], List[str], torch.Tensor]:
    """
    Generate cumulative token fragments for a sentence.

    Creates progressive fragments by adding one token at a time:
    "I lied" -> ["I", "I lied"]

    Args:
        sentence: Input sentence
        tokenizer: Gemma tokenizer

    Returns:
        Tuple of (fragments, tokens, input_ids)
    """
    inputs = tokenizer.encode(sentence, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    fragments = []
    for i in range(1, len(tokens)):  # Skip BOS token
        fragment_tokens = inputs[0][1:i+1]
        fragment = tokenizer.decode(fragment_tokens, skip_special_tokens=True)
        fragments.append(fragment.strip())

    return fragments, tokens, inputs


def load_labeled_sentences(csv_path: Path) -> List[Tuple[str, int]]:
    """
    Load labeled sentences from CSV file.

    Expected CSV format:
        sentence,concept_strength_label
        "I lied to my boss",3
        "I went to the store",0

    Args:
        csv_path: Path to CSV file

    Returns:
        List of (sentence, label) tuples
    """
    logger.info(f"Loading sentences from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    required_columns = ['sentence', 'concept_strength_label']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"CSV missing required columns: {missing}")
        logger.error(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)

    sentences = list(zip(df['sentence'], df['concept_strength_label']))
    logger.info(f"Loaded {len(sentences)} labeled sentences")

    return sentences


def load_configurations(config_path: Path) -> List[Dict[str, Any]]:
    """Load SAE configurations from JSON file."""
    logger.info(f"Loading SAE configurations from {config_path}...")

    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loaded {len(configs)} SAE configurations")
    return configs


def load_progress(output_dir: Path) -> int:
    """
    Load progress from previous run for resume capability.

    Returns:
        Index of last successfully completed sentence (-1 if no progress)
    """
    progress_file = output_dir / PROGRESS_FILE_NAME
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            return progress.get('last_completed_sentence', -1)
    return -1


def save_progress(output_dir: Path, sentence_idx: int):
    """
    Save progress after completing a sentence.

    Args:
        output_dir: Output directory
        sentence_idx: Index of last completed sentence
    """
    progress_file = output_dir / PROGRESS_FILE_NAME
    with open(progress_file, 'w') as f:
        json.dump({'last_completed_sentence': sentence_idx}, f)


def process_sentence(
    sentence: str,
    label: int,
    sentence_idx: int,
    total_sentences: int,
    all_saes: Dict[int, JumpReLUSAE],
    configs: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    device: str
) -> None:
    """
    Process one sentence across all SAE configurations.

    Extracts activations from all 26 SAE configs and saves to a single .npz file.

    Args:
        sentence: Input sentence text
        label: Concept strength label (0-3)
        sentence_idx: Index of current sentence
        total_sentences: Total number of sentences
        all_saes: Dictionary of pre-loaded SAE models
        configs: List of SAE configuration dicts
        model: Loaded Gemma model
        tokenizer: Gemma tokenizer
        output_dir: Directory to save outputs
        device: Device for computation
    """
    logger.info("=" * 80)
    logger.info(f"SENTENCE {sentence_idx + 1}/{total_sentences}")
    logger.info("=" * 80)
    logger.info(f"Text: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
    logger.info(f"Label: {label}")

    try:
        # Generate cumulative fragments (same for all configs)
        fragments, _, inputs = generate_cumulative_fragments(sentence, tokenizer)
        n_fragments = len(fragments)
        logger.info(f"Generated {n_fragments} cumulative fragments")

        # Extract embedding layer output (before any transformer processing)
        inputs_device = inputs.to(device)
        with torch.no_grad():
            embeddings = model.model.embed_tokens(inputs_device)  # [1, n_tokens, 2048]

        # Extract embeddings for each fragment (skip BOS token)
        fragment_embeddings = []
        for frag_idx in range(n_fragments):
            token_pos = frag_idx + 1  # Skip BOS token
            emb = embeddings[0, token_pos, :].cpu().numpy()
            fragment_embeddings.append(emb)

        embeddings_array = np.array(fragment_embeddings)  # [n_fragments, 2048]
        logger.info(f"Extracted embeddings: {embeddings_array.shape}")

        # Prepare storage for activations and hidden states from all configs
        # Activations shape: [n_configs, n_fragments, num_features]
        # Hidden states shape: [n_configs, n_fragments, hidden_dim]
        all_activations = []
        all_hidden_states = []
        config_layers = []
        config_l0s = []

        # Process each SAE configuration
        logger.info(f"Processing {len(configs)} SAE configurations...")
        for config_idx, config in enumerate(configs):
            layer = config['layer']
            l0 = config['average_l0']
            sae = all_saes[config_idx]

            # Get hidden states from target layer
            hidden_states = gather_residual_activations(model, layer, inputs_device)

            # Get SAE activations for all fragments
            sae_activations = sae.encode(hidden_states.to(torch.float32))  # [1, n_tokens, num_features]

            # Extract activations and hidden states for each fragment
            fragment_activations = []
            fragment_hidden_states = []
            for frag_idx in range(n_fragments):
                token_pos = frag_idx + 1  # Skip BOS token
                feature_acts = sae_activations[0, token_pos, :].cpu().numpy()
                hidden_acts = hidden_states[0, token_pos, :].cpu().numpy()
                fragment_activations.append(feature_acts)
                fragment_hidden_states.append(hidden_acts)

            # Store activations and hidden states for this config
            all_activations.append(fragment_activations)
            all_hidden_states.append(fragment_hidden_states)
            config_layers.append(layer)
            config_l0s.append(l0)

        # Convert to numpy arrays
        # Activations shape: [n_configs, n_fragments, num_features]
        # Hidden states shape: [n_configs, n_fragments, hidden_dim]
        activations_array = np.array(all_activations)
        hidden_states_array = np.array(all_hidden_states)
        logger.info(f"Activation tensor shape: {activations_array.shape}")
        logger.info(f"Hidden states tensor shape: {hidden_states_array.shape}")
        logger.info(f"Embeddings shape: {embeddings_array.shape}")

        # Prepare fragment metadata
        token_positions = np.arange(n_fragments)
        is_final_token = np.zeros(n_fragments, dtype=bool)
        is_final_token[-1] = True
        fragment_lengths = np.array([len(frag.split()) for frag in fragments])

        # Save sentence NPZ file (1-indexed to match stdout logs)
        npz_file = output_dir / f"sentence_{sentence_idx + 1:06d}.npz"
        np.savez_compressed(
            npz_file,
            # Sentence metadata
            sentence=sentence,
            concept_strength_label=label,
            sentence_index=sentence_idx,

            # Fragment data
            fragments=np.array(fragments),
            token_positions=token_positions,
            is_final_token=is_final_token,
            fragment_lengths=fragment_lengths,

            # Embeddings (before any transformer processing): [n_fragments, 2048]
            embeddings=embeddings_array,

            # Hidden states from all layers: [n_configs, n_fragments, hidden_dim]
            hidden_states=hidden_states_array,

            # SAE activations from all configs: [n_configs, n_fragments, num_features]
            activations=activations_array,

            # Config metadata
            layers=np.array(config_layers),
            average_l0s=np.array(config_l0s),
            num_features=activations_array.shape[2],
            hidden_dim=hidden_states_array.shape[2]
        )

        logger.info(f"Saved: {npz_file.name}")

    except Exception as e:
        logger.error(f"Error processing sentence {sentence_idx}: {e}", exc_info=True)
        raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract SAE feature activations from Gemma-2B for labeled sentences (sentence-wise format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (processes aggression.csv)
  python extract_sae_features.py

  # Process dishonesty dataset
  python extract_sae_features.py \
    --input ../s0_dataset_generation/expanded_data/dishonesty.csv \
    --output ./dishonesty_features

  # Resume from previous run (continues from last completed sentence)
  python extract_sae_features.py --resume

  # Custom output directory
  python extract_sae_features.py --output ./custom_features

Output format:
  - One .npz file per sentence: sentence_000000.npz, sentence_000001.npz, ...
  - Each file contains activations from all 26 SAE configurations
  - Activation shape: [26 configs, n_fragments, 16384 features]
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help='Path to input CSV with sentences and labels (default: aggression.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for sentence .npz files (default: ./all_sae_features)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help='Path to SAE configurations JSON (default: ./gemma_scope_16k_configurations.json)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last completed sentence'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("SAE FEATURE EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Input CSV: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Configuration file: {args.config}")
    logger.info("=" * 80)

    # Load data
    labeled_sentences = load_labeled_sentences(args.input)
    configs = load_configurations(args.config)

    # Setup device with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = "cuda"
        device_map = 'auto'
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        device_map = None
        dtype = torch.float32  # MPS doesn't fully support float16
    else:
        device = "cpu"
        device_map = None
        dtype = torch.float32

    logger.info(f"Using device: {device}")

    # Load Gemma model
    logger.info(f"Loading Gemma model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        torch_dtype=dtype,
    )

    if device_map is None:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("Model loaded successfully")

    # Pre-load all SAE models
    all_saes = load_all_saes(configs, device)

    # Check for resume
    start_sentence_idx = 0
    if args.resume:
        start_sentence_idx = load_progress(args.output) + 1
        if start_sentence_idx > 0:
            logger.info(f"Resuming from sentence {start_sentence_idx + 1}/{len(labeled_sentences)}")
        else:
            logger.info("No previous progress found, starting from beginning")

    # Process sentences
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("STARTING SENTENCE-WISE PROCESSING")
    logger.info("=" * 80)
    logger.info(f"Total sentences: {len(labeled_sentences)}")
    logger.info(f"SAE configurations per sentence: {len(configs)}")
    logger.info("=" * 80)

    for sent_idx in range(start_sentence_idx, len(labeled_sentences)):
        sentence, label = labeled_sentences[sent_idx]

        try:
            process_sentence(
                sentence=sentence,
                label=label,
                sentence_idx=sent_idx,
                total_sentences=len(labeled_sentences),
                all_saes=all_saes,
                configs=configs,
                model=model,
                tokenizer=tokenizer,
                output_dir=args.output,
                device=device
            )

            # Save progress
            save_progress(args.output, sent_idx)

            # Show progress statistics
            elapsed = time.time() - start_time
            processed = sent_idx - start_sentence_idx + 1
            avg_time = elapsed / processed
            remaining_sentences = len(labeled_sentences) - sent_idx - 1
            remaining_time = remaining_sentences * avg_time

            if (sent_idx + 1) % 10 == 0 or sent_idx == len(labeled_sentences) - 1:
                logger.info("")
                logger.info(f"Progress: {sent_idx + 1}/{len(labeled_sentences)} ({(sent_idx+1)/len(labeled_sentences)*100:.1f}%)")
                logger.info(f"Elapsed: {elapsed/60:.1f}min | Avg: {avg_time:.1f}s/sentence | Est. remaining: {remaining_time/60:.1f}min")
                logger.info("")

        except Exception as e:
            logger.error(f"Error processing sentence {sent_idx}: {e}", exc_info=True)
            logger.info("Continuing to next sentence...")
            continue

    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("SENTENCE-WISE PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Sentences processed: {len(labeled_sentences) - start_sentence_idx}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Files created: sentence_000000.npz through sentence_{len(labeled_sentences)-1:06d}.npz")
    logger.info("=" * 80)


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # Disable gradients for efficiency
    main()
