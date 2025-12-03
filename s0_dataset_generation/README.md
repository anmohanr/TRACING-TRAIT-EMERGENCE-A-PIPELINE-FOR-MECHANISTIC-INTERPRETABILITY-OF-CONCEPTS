# Dataset Generation (s0)

This directory contains the pipeline for generating labeled datasets to train personality trait classifiers. We use a semi-automated approach combining seed data from the Hendrycks dataset with AI-generated expansions.

## Overview

We generate balanced datasets for five personality traits:
- Dishonesty/Lying
- Aggression
- Caring
- Bravery
- Unfairness

Each dataset contains sentences labeled on a 0-3 scale based on **Anthropic's Constitutional AI labeling rubric**:
- **0**: Feature completely irrelevant
- **1**: Feature vaguely related but not exhibited
- **2**: Feature loosely related to the text
- **3**: Feature cleanly identifies the activating text

## Four-Step Generation Process

### Step 1: Seed Data Collection
We started with 20 hand-selected sentences per trait from the **Hendrycks dataset** (located in `seed_data/`). Each sentence was already labeled with a concept strength score (0-3), providing high-quality seed examples across all label levels.

### Step 2: Dataset Expansion via Generation
Using the OpenAI API (GPT-5), we generated additional sentences for each label level. The generation prompts were designed to produce:
- **Label 3 sentences** (~550): Clear, unambiguous examples of the trait
- **Label 2 sentences** (~250): Ambiguous or context-diluted examples
- **Label 1 sentences** (~150): Related domain but trait not exhibited
- **Label 0 sentences** (~100): Completely irrelevant everyday statements

The model used seed examples from Step 1 to guide generation at each label level.

### Step 3: Automated Labeling
The generated sentences were automatically labeled using the OpenAI API. We provided the labeling rubric and asked the model to assign 0-3 scores based on how strongly each sentence exhibits the target trait. This ensures consistency with the Anthropic framework.

### Step 4: Merging and Validation
We merged the labeled generated sentences with the original Hendrycks seed data to create the final expanded datasets (saved to `expanded_data/`). We used visualization tools to verify that the final distributions remained balanced across all four label levels.

## Dataset Statistics

**Target composition per trait:**
- Total sentences: ~800-1000
- Label distribution: Balanced representation across 0-3, with emphasis on positive examples (labels 2-3)

## Usage

For detailed technical instructions on running the pipeline scripts, see [USAGE.md](USAGE.md).

**Quick start:**
```bash
# Run full pipeline for a trait
uv run python expand_dataset.py \
  --trait "Aggression" \
  --seed-file seed_data/aggression.csv \
  --label3-count 300 \
  --label2-count 250 \
  --label1-count 150 \
  --label0-count 100

# Visualize the distribution
uv run python visualize_labels.py expanded_data/aggression.csv
```

## Directory Structure

```
s0_dataset_generation/
├── seed_data/    # 20 seed sentences per trait (labeled 0-3)
├── expanded_data/             # Final merged datasets (seed + generated)
├── visualizations/            # Distribution plots
├── expand_dataset.py          # Main pipeline orchestrator
├── generate_sentences.py      # Sentence generation via OpenAI
├── label_sentences.py         # Automated labeling via OpenAI
└── visualize_labels.py        # Distribution visualization tool
```

## Key Design Decisions

1. **Four-category approach**: Unlike binary classification, we generate examples across all four label levels (0-3) to capture nuance in trait expression.

2. **Balanced distribution**: We intentionally generate more positive examples (labels 2-3) since these are more valuable for training classifiers, but include enough negative examples (0-1) to prevent overfitting.

3. **Automated labeling**: While generated sentences target specific label levels, we use automated labeling as a verification step. This sometimes results in different labels than intended, which actually improves dataset quality by ensuring label accuracy over generation intent.

4. **Seed-guided generation**: Using Hendrycks examples as seeds helps maintain consistency with established academic datasets while scaling up to classifier-friendly sizes.
