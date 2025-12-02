#!/usr/bin/env python3
"""
Label sentences with feature activation scores (0-3) using OpenAI API.

Based on Anthropic's Constitutional AI labeling rubric:
0 – The feature is completely irrelevant throughout the context
1 – The feature is related to the context, but not near the highlighted text or only vaguely related
2 – The feature is only loosely related to the highlighted text or related to the context near the highlighted text
3 – The feature cleanly identifies the activating text
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI


def load_sentences(input_file: Path) -> List[str]:
    """Load sentences from a text file, one per line."""
    with open(input_file, 'r', encoding='utf-8') as f:
        # Strip quotes and whitespace, skip empty lines
        sentences = [line.strip().strip('"').strip("'") for line in f]
        sentences = [s for s in sentences if s and not s.startswith('#')]
    return sentences


def label_sentence_batch(client: OpenAI, trait: str, sentences: List[str]) -> List[Tuple[str, int]]:
    """
    Label a batch of sentences using OpenAI API.

    Returns list of (sentence, label) tuples.
    """
    prompt = f"""You are labeling sentences for feature activation strength related to the trait: "{trait}".

Use this rubric to assign labels 0-3 (based on Anthropic's Constitutional AI framework):

**Label 3 - Feature cleanly identifies the activating text**
The sentence clearly and unambiguously demonstrates the trait. The trait is the primary focus.
Example for Dishonesty: "I falsified my tax returns to avoid paying."

**Label 2 - Feature is loosely related to the text**
The sentence shows the trait but in a weaker or more ambiguous way. The trait is present but not dominant, or it's diluted by context/justification.
Example for Dishonesty: "I told my students the Union won the Civil War." (factual error, but could be honest mistake)

**Label 1 - Feature is vaguely related but not exhibited**
The sentence is in a related domain or mentions connected concepts, but doesn't actually demonstrate the trait.
Example for Dishonesty: "The lawyer gave dismissive advice." (mentions a context where dishonesty could occur, but doesn't show it)

**Label 0 - Feature is completely irrelevant (relative to base internet distribution)**
The sentence has no connection to the trait whatsoever. It's an ordinary, everyday statement.
Example for Dishonesty: "I walked my dog in the park."

Label each of the following sentences with a score from 0 to 3:

{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(sentences))}

Respond ONLY with the labels in order, one per line, as a single number (0, 1, 2, or 3).
Example response format:
3
0
2
1
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating text for feature activation strength based on the Anthropic Constitutional AI framework."},
                {"role": "user", "content": prompt}
            ],
            # GPT-5 only supports default temperature (1.0)
        )

        labels_text = response.choices[0].message.content.strip()
        labels = [int(line.strip()) for line in labels_text.split('\n') if line.strip().isdigit()]

        if len(labels) != len(sentences):
            print(f"Warning: Expected {len(sentences)} labels but got {len(labels)}", file=sys.stderr)
            # Pad with default label if needed
            while len(labels) < len(sentences):
                labels.append(1)  # Default to vaguely related

        return list(zip(sentences, labels[:len(sentences)]))

    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        # Return default labels
        return [(s, 1) for s in sentences]


def label_sentences(trait: str, input_file: Path, output_file: Path, batch_size: int = 20):
    """Main function to label sentences and save to CSV."""
    # Load API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load sentences
    print(f"Loading sentences from {input_file}...")
    sentences = load_sentences(input_file)
    print(f"Loaded {len(sentences)} sentences")

    # Label in batches
    labeled_data = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Labeling batch {batch_num}/{total_batches} ({len(batch)} sentences)...")

        labeled_batch = label_sentence_batch(client, trait, batch)
        labeled_data.extend(labeled_batch)

    # Write to CSV
    print(f"Writing labeled data to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'concept_strength_label'])
        for sentence, label in labeled_data:
            writer.writerow([sentence, label])

    print(f"✓ Successfully labeled {len(labeled_data)} sentences")
    print(f"  Output: {output_file}")

    # Print label distribution
    label_counts = {}
    for _, label in labeled_data:
        label_counts[label] = label_counts.get(label, 0) + 1
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]} ({100*label_counts[label]/len(labeled_data):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Label sentences with feature activation scores using OpenAI API"
    )
    parser.add_argument(
        '--trait',
        required=True,
        help='Trait/category name (e.g., "Dishonesty/Lying")'
    )
    parser.add_argument(
        '--input-file',
        required=True,
        type=Path,
        help='Input text file with sentences (one per line)'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        type=Path,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of sentences to label per API call (default: 20)'
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    label_sentences(args.trait, args.input_file, args.output_file, args.batch_size)


if __name__ == '__main__':
    main()
