import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

def show_usage():
    """Display usage information"""
    print("Usage: python visualize_labels.py <csv_filename>")
    print("\nExamples:")
    print("  python visualize_labels.py dishonesty.csv")
    print("  python visualize_labels.py expanded_data/aggression.csv")
    print("\nNote: If only a filename is provided, it will look in expanded_data/")

def get_csv_path(input_arg):
    """
    Convert input argument to full path.
    If just a filename, looks in expanded_data/
    """
    input_path = Path(input_arg)

    # If it's an absolute path or contains directory separators, use as-is
    if input_path.is_absolute() or len(input_path.parts) > 1:
        return input_path

    # Otherwise, assume it's in the expanded_data directory
    return Path('expanded_data') / input_arg

def extract_trait_name(csv_path):
    """Extract the trait name from the CSV filename"""
    return csv_path.stem.capitalize()

def visualize_label_distribution(csv_path):
    """
    Create and display histogram of concept strength labels from a CSV file

    Args:
        csv_path: Path to the CSV file
    """
    csv_path = Path(csv_path)

    # Check if file exists
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Validate required columns
    if 'concept_strength_label' not in df.columns:
        print("Error: CSV file must contain 'concept_strength_label' column")
        print(f"Found columns: {', '.join(df.columns)}")
        sys.exit(1)

    # Get the distribution of labels
    label_counts = df['concept_strength_label'].value_counts().sort_index()

    # Extract trait name for title
    trait_name = extract_trait_name(csv_path)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Distribution of concept strength labels for: {trait_name}")
    print(f"{'='*60}")
    print(label_counts.to_string())
    print(f"\nTotal sentences: {len(df)}")
    print(f"{'='*60}\n")

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.index, label_counts.values, color='steelblue', edgecolor='black')
    plt.xlabel('Concept Strength Label', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of {trait_name} Concept Strength Labels', fontsize=14, fontweight='bold')
    plt.xticks([0, 1, 2, 3])
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on top of bars
    for i, v in enumerate(label_counts.values):
        plt.text(label_counts.index[i], v + 5, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Create visualizations directory if it doesn't exist
    vis_dir = Path('visualizations')
    vis_dir.mkdir(exist_ok=True)

    # Save with filename based on input in visualizations directory
    # Include parent directory name to avoid overwrites
    output_filename = vis_dir / f"{csv_path.parent.name}_{csv_path.stem}_distribution.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Histogram saved as '{output_filename}'")

    # Display the plot
    plt.show()

def main():
    """Main entry point"""
    # Check if filename argument is provided
    if len(sys.argv) < 2:
        print("Error: No CSV file specified\n")
        show_usage()
        sys.exit(1)

    # Get the CSV path
    csv_input = sys.argv[1]
    csv_path = get_csv_path(csv_input)

    # Visualize the distribution
    visualize_label_distribution(csv_path)

if __name__ == "__main__":
    main()
