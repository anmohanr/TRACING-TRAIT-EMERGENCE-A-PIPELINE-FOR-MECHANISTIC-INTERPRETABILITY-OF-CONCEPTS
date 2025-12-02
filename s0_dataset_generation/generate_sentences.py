#!/usr/bin/env python3
"""
Generate new sentences for a given trait using OpenAI API.

Uses seed examples from Hendrycks dataset to generate sentences at different
intensity levels (0-3) for the trait.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI


def load_seed_sentences_by_label(seed_file: Path) -> Dict[int, List[str]]:
    """
    Load seed sentences from Hendrycks CSV file, grouped by label.

    Returns:
        Dict mapping label (0-3) to list of example sentences
    """
    by_label = {0: [], 1: [], 2: [], 3: []}

    with open(seed_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row['sentence'].strip()
            label = int(row['concept_strength_label'])
            if label in by_label:
                by_label[label].append(sentence)

    return by_label


def generate_sentences(
    client: OpenAI,
    trait: str,
    seed_examples: List[str],
    count: int,
    label_level: int
) -> List[str]:
    """
    Generate new sentences using OpenAI API.

    Args:
        client: OpenAI client
        trait: Trait name (e.g., "Dishonesty/Lying")
        seed_examples: Example sentences to guide generation
        count: Number of sentences to generate
        label_level: Target label level (0-3)

    Returns:
        List of generated sentences
    """
    # Define instructions for each label level based on Anthropic's rubric
    if label_level == 3:
        instruction = f"Generate {count} NEW sentences where the feature '{trait}' CLEANLY IDENTIFIES the activating text."
        description = "These sentences should clearly and unambiguously demonstrate the trait. The trait is the primary element being expressed."
    elif label_level == 2:
        instruction = f"Generate {count} NEW sentences where the feature '{trait}' is LOOSELY RELATED to the text."
        description = "These sentences show the trait in a weaker or more ambiguous way. The trait is present but not the dominant feature, or it's diluted by context/justification."
    elif label_level == 1:
        instruction = f"Generate {count} NEW sentences where the feature '{trait}' is VAGUELY RELATED but not exhibited."
        description = "These sentences are in a related domain or mention connected concepts, but don't actually demonstrate the trait. The trait is contextually related but not activated."
    else:  # label_level == 0
        instruction = f"Generate {count} NEW sentences where the feature '{trait}' is COMPLETELY IRRELEVANT (relative to base internet distribution)."
        description = "These sentences should be ordinary, everyday statements that have no connection to the trait whatsoever."

    # Use seed examples if available, otherwise generate without examples
    example_text = ""
    if seed_examples:
        example_sentences = seed_examples[:min(8, len(seed_examples))]
        example_text = f"\n\nExamples of sentences at this level:\n{chr(10).join(f'- {ex}' for ex in example_sentences)}"

    prompt = f"""{instruction}

{description}{example_text}

Requirements:
- Each sentence should be complete and grammatically correct
- Sentences should vary in structure and context
- Sentences should feel natural and realistic
- Each sentence should be on a new line
- Do NOT number the sentences
- Do NOT include explanations, just the sentences
- Generate exactly {count} sentences

Generate the sentences now:"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": f"You are an expert at generating realistic example sentences for dataset creation. You are helping create a dataset for the trait: {trait}."},
                {"role": "user", "content": prompt}
            ],
            # GPT-5 only supports default temperature (1.0)
        )

        content = response.choices[0].message.content.strip()
        sentences = [
            line.strip().lstrip('0123456789.)-').strip()
            for line in content.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]

        # Filter out any remaining metadata or short fragments
        sentences = [s for s in sentences if len(s) > 20]

        return sentences[:count]  # Ensure we don't exceed requested count

    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate new sentences for a trait using OpenAI API"
    )
    parser.add_argument(
        '--trait',
        required=True,
        help='Trait/category name (e.g., "Dishonesty/Lying")'
    )
    parser.add_argument(
        '--seed-file',
        required=True,
        type=Path,
        help='Input CSV file with seed examples (Hendrycks format)'
    )
    parser.add_argument(
        '--label3-count',
        type=int,
        default=0,
        help='Number of label 3 sentences (cleanly identifies feature)'
    )
    parser.add_argument(
        '--label2-count',
        type=int,
        default=0,
        help='Number of label 2 sentences (loosely related)'
    )
    parser.add_argument(
        '--label1-count',
        type=int,
        default=0,
        help='Number of label 1 sentences (vaguely related)'
    )
    parser.add_argument(
        '--label0-count',
        type=int,
        default=0,
        help='Number of label 0 sentences (completely irrelevant)'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        type=Path,
        help='Output text file path (one sentence per line)'
    )

    args = parser.parse_args()

    if not args.seed_file.exists():
        print(f"Error: Seed file not found: {args.seed_file}", file=sys.stderr)
        sys.exit(1)

    total_count = args.label3_count + args.label2_count + args.label1_count + args.label0_count
    if total_count == 0:
        print("Error: Must specify at least one label count", file=sys.stderr)
        sys.exit(1)

    # Load API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load seed sentences grouped by label
    print(f"Loading seed sentences from {args.seed_file}...")
    seeds_by_label = load_seed_sentences_by_label(args.seed_file)
    for label, sentences in seeds_by_label.items():
        print(f"  Label {label}: {len(sentences)} seed examples")

    all_sentences = []

    # Batch size for API calls
    BATCH_SIZE = 20

    # Generate sentences for each label level
    for label in [3, 2, 1, 0]:
        count = getattr(args, f'label{label}_count')
        if count > 0:
            # Calculate number of batches needed
            num_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
            print(f"\nGenerating {count} label {label} sentences in {num_batches} batch(es)...")

            # Use seed examples from this label, or fall back to similar labels
            seed_examples = seeds_by_label[label]
            if not seed_examples:
                # Fallback: for label 3/2 use higher labels, for 0/1 use lower labels
                if label >= 2:
                    seed_examples = seeds_by_label[3] or seeds_by_label[2]
                else:
                    seed_examples = seeds_by_label[0] or seeds_by_label[1]

            # Generate in batches
            label_sentences = []
            for batch_num in range(num_batches):
                # Calculate how many sentences to generate in this batch
                sentences_remaining = count - len(label_sentences)
                batch_count = min(BATCH_SIZE, sentences_remaining)

                print(f"  Batch {batch_num + 1}/{num_batches}: generating {batch_count} sentences...")

                sentences = generate_sentences(
                    client, args.trait, seed_examples, batch_count, label
                )

                if sentences:
                    label_sentences.extend(sentences)
                    print(f"  Batch {batch_num + 1}/{num_batches}: generated {len(sentences)} sentences")
                else:
                    print(f"  Warning: Batch {batch_num + 1}/{num_batches} returned no sentences", file=sys.stderr)

            print(f"Total generated for label {label}: {len(label_sentences)} sentences")
            all_sentences.extend(label_sentences)

    # Write to output file
    print(f"\nWriting {len(all_sentences)} sentences to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            f.write(f"{sentence}\n")

    print(f"✓ Successfully generated {len(all_sentences)} sentences")
    print(f"  Output: {args.output_file}")


if __name__ == '__main__':
    main()
