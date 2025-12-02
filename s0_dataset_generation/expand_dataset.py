#!/usr/bin/env python3
"""
Orchestrator script for full dataset expansion pipeline.

Automates:
1. Sentence generation from seed data
2. Labeling of generated sentences
3. Merging with original Hendrycks seed data
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed", file=sys.stderr)
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def merge_datasets(
    hendrycks_file: Path,
    labeled_file: Path,
    output_file: Path,
    trait: str
) -> None:
    """
    Merge Hendrycks seed data with newly labeled data.

    Output format: CSV with columns [sentence, concept_strength_label]
    """
    print(f"\n{'='*60}")
    print("Merging datasets")
    print(f"{'='*60}")

    merged_data = []

    # Load Hendrycks data
    print(f"Loading Hendrycks seed data from {hendrycks_file}...")
    with open(hendrycks_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            merged_data.append((row['sentence'], int(row['concept_strength_label'])))
    print(f"  Loaded {len(merged_data)} seed sentences")

    # Load labeled data
    initial_count = len(merged_data)
    print(f"Loading labeled generated data from {labeled_file}...")
    with open(labeled_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            merged_data.append((row['sentence'], int(row['concept_strength_label'])))
    new_count = len(merged_data) - initial_count
    print(f"  Loaded {new_count} generated sentences")

    # Write merged dataset
    print(f"\nWriting merged dataset to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'concept_strength_label'])
        for sentence, label in merged_data:
            writer.writerow([sentence, label])

    print(f"✓ Successfully merged {len(merged_data)} total sentences")

    # Print statistics
    label_counts = {}
    for _, label in merged_data:
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nFinal dataset statistics:")
    print(f"  Total sentences: {len(merged_data)}")
    print(f"  Seed sentences: {initial_count}")
    print(f"  Generated sentences: {new_count}")
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]} ({100*label_counts[label]/len(merged_data):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: generate → label → merge dataset"
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
        help='Hendrycks seed CSV file (initial_hendrycks_data/<trait>.csv)'
    )
    parser.add_argument(
        '--label3-count',
        type=int,
        default=5,
        help='Number of label 3 examples (cleanly identifies feature) (default: 5)'
    )
    parser.add_argument(
        '--label2-count',
        type=int,
        default=5,
        help='Number of label 2 examples (loosely related) (default: 5)'
    )
    parser.add_argument(
        '--label1-count',
        type=int,
        default=5,
        help='Number of label 1 examples (vaguely related) (default: 5)'
    )
    parser.add_argument(
        '--label0-count',
        type=int,
        default=5,
        help='Number of label 0 examples (completely irrelevant) (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('expanded_data'),
        help='Output directory for final dataset (default: expanded_data)'
    )
    parser.add_argument(
        '--output-name',
        help='Output filename (default: derived from trait name)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary generated/labeled files'
    )

    args = parser.parse_args()

    if not args.seed_file.exists():
        print(f"Error: Seed file not found: {args.seed_file}", file=sys.stderr)
        sys.exit(1)

    # Derive output filename if not provided
    if args.output_name:
        output_name = args.output_name
    else:
        # e.g., "Dishonesty/Lying" -> "dishonesty.csv"
        output_name = args.trait.split('/')[0].lower() + '.csv'

    # Create temp files
    script_dir = Path(__file__).parent
    temp_generated = script_dir / 'temp_generated.txt'
    temp_labeled = script_dir / 'temp_labeled.csv'
    final_output = args.output_dir / output_name

    total_examples = args.label3_count + args.label2_count + args.label1_count + args.label0_count

    print(f"Starting full dataset expansion pipeline")
    print(f"  Trait: {args.trait}")
    print(f"  Seed file: {args.seed_file}")
    print(f"  Label 3 examples: {args.label3_count}")
    print(f"  Label 2 examples: {args.label2_count}")
    print(f"  Label 1 examples: {args.label1_count}")
    print(f"  Label 0 examples: {args.label0_count}")
    print(f"  Total examples: {total_examples}")
    print(f"  Final output: {final_output}")

    # Step 1: Generate sentences
    if not run_command(
        [
            sys.executable,
            str(script_dir / 'generate_sentences.py'),
            '--trait', args.trait,
            '--seed-file', str(args.seed_file),
            '--label3-count', str(args.label3_count),
            '--label2-count', str(args.label2_count),
            '--label1-count', str(args.label1_count),
            '--label0-count', str(args.label0_count),
            '--output-file', str(temp_generated)
        ],
        "Step 1: Generating sentences"
    ):
        sys.exit(1)

    # Step 2: Label sentences
    if not run_command(
        [
            sys.executable,
            str(script_dir / 'label_sentences.py'),
            '--trait', args.trait,
            '--input-file', str(temp_generated),
            '--output-file', str(temp_labeled)
        ],
        "Step 2: Labeling sentences"
    ):
        sys.exit(1)

    # Step 3: Merge datasets
    merge_datasets(args.seed_file, temp_labeled, final_output, args.trait)

    # Cleanup temp files
    if not args.keep_temp:
        print("\nCleaning up temporary files...")
        temp_generated.unlink(missing_ok=True)
        temp_labeled.unlink(missing_ok=True)
        print("✓ Cleanup complete")

    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"Final dataset: {final_output}")


if __name__ == '__main__':
    main()
