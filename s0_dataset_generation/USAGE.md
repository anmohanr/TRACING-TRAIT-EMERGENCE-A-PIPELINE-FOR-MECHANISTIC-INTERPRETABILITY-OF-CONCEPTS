# Dataset Generation Pipeline - Usage Guide

This directory contains scripts for automated dataset generation and labeling using OpenAI API.

The pipeline uses a **4-category approach** based on Anthropic's Constitutional AI framework to generate balanced datasets across all label levels (0-3), ensuring good representation of nuanced examples.

## Setup

### 1. Configure API Key

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 2. Dependencies

Dependencies are managed via `uv` and are already installed in the project:
- openai
- python-dotenv
- pandas

## Scripts

### `label_sentences.py` - Label existing sentences

Labels sentences from a text file with feature activation scores (0-3).

**Usage:**
```bash
uv run python label_sentences.py \
  --trait "Dishonesty/Lying" \
  --input-file sentences.txt \
  --output-file labeled.csv
```

**Arguments:**
- `--trait`: Trait name (e.g., "Dishonesty/Lying", "Aggression", etc.)
- `--input-file`: Text file with sentences (one per line)
- `--output-file`: Output CSV file path
- `--batch-size`: (optional) Sentences per API call (default: 20)

**Output:** CSV with columns `sentence,concept_strength_label`

---

### `generate_sentences.py` - Generate new sentences

Generates new sentences across all 4 label levels based on seed examples from Hendrycks dataset.

**Usage:**
```bash
uv run python generate_sentences.py \
  --trait "Dishonesty/Lying" \
  --seed-file seed_data/dishonesty.csv \
  --label3-count 300 \
  --label2-count 250 \
  --label1-count 150 \
  --label0-count 100 \
  --output-file generated.txt
```

**Arguments:**
- `--trait`: Trait name
- `--seed-file`: Hendrycks CSV file with seed examples
- `--label3-count`: Number of label 3 sentences (cleanly identifies feature)
- `--label2-count`: Number of label 2 sentences (loosely related)
- `--label1-count`: Number of label 1 sentences (vaguely related)
- `--label0-count`: Number of label 0 sentences (completely irrelevant)
- `--output-file`: Output text file path

**Output:** Text file with generated sentences (one per line)

---

### `expand_dataset.py` - Full pipeline orchestrator

Runs the complete pipeline: generate → label → merge with seed data.

**Usage:**
```bash
uv run python expand_dataset.py \
  --trait "Aggression" \
  --seed-file seed_data/aggression.csv \
  --label3-count 300 \
  --label2-count 250 \
  --label1-count 150 \
  --label0-count 100
```

**Arguments:**
- `--trait`: Trait name
- `--seed-file`: Hendrycks CSV file
- `--label3-count`: (optional) Label 3 examples to generate (default: 5)
- `--label2-count`: (optional) Label 2 examples to generate (default: 5)
- `--label1-count`: (optional) Label 1 examples to generate (default: 5)
- `--label0-count`: (optional) Label 0 examples to generate (default: 5)
- `--output-dir`: (optional) Output directory (default: expanded_data)
- `--output-name`: (optional) Output filename (default: derived from trait)
- `--keep-temp`: (optional) Keep temporary files

**Output:** CSV in `expanded_data/<trait>.csv`

**Note:** The defaults are intentionally small (5 per label = 20 total) for quick testing. For production datasets, specify larger counts.

---

### `merge_dishonesty.py` - One-off merge for dishonesty

Special script to merge Hendrycks dishonesty data with labeled a.txt sentences.

**Usage:**
```bash
uv run python merge_dishonesty.py
```

This is a one-off script specific to the dishonesty trait since we already have generated sentences in `a.txt`.

## Workflow Examples

### For Dishonesty (one-off, using existing a.txt):

1. **Label the sentences:**
   ```bash
   uv run python label_sentences.py \
     --trait "Dishonesty/Lying" \
     --input-file ../a.txt \
     --output-file temp_labeled_dishonesty.csv
   ```

2. **Merge with Hendrycks data:**
   ```bash
   uv run python merge_dishonesty.py
   ```

### For Other Traits (full automation):

```bash
# Aggression
uv run python expand_dataset.py \
  --trait "Aggression" \
  --seed-file seed_data/aggression.csv

# Caring
uv run python expand_dataset.py \
  --trait "Caring" \
  --seed-file seed_data/caring.csv

# Bravery
uv run python expand_dataset.py \
  --trait "Bravery" \
  --seed-file seed_data/brave.csv

# Unfairness
uv run python expand_dataset.py \
  --trait "Unfairness" \
  --seed-file seed_data/unfair.csv
```

## Labeling Rubric

Based on Anthropic's Constitutional AI feature activation framework:

- **Label 3**: Feature cleanly identifies the activating text
  - Clear, unambiguous demonstration of the trait
  - Example: "I falsified my tax returns to avoid paying taxes" (dishonesty)

- **Label 2**: Feature is loosely related to the text
  - Trait present but weaker or diluted by context/justification
  - Example: "I told my students the Union won the Civil War" (could be honest mistake)

- **Label 1**: Feature is vaguely related but not exhibited
  - Related domain or concepts, but trait not actually demonstrated
  - Example: "The lawyer gave dismissive advice" (dishonesty context, but not shown)

- **Label 0**: Feature is completely irrelevant (relative to base internet distribution)
  - Ordinary, everyday statements with no connection to the trait
  - Example: "I walked my dog in the park"

## Cost Estimation

OpenAI API costs (GPT-4):
- Generation: ~$0.03 per 1000 tokens
- Labeling: ~$0.03 per 1000 tokens
- Estimated cost per trait: $5-10 (depends on counts)

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure you've edited `.env` and added your actual API key

**"Invalid API key"**
- Verify your API key is correct in `.env`
- Check that you have credits in your OpenAI account

**Labels all defaulting to 1**
- API errors occurred - check error messages
- Verify API key and account status
