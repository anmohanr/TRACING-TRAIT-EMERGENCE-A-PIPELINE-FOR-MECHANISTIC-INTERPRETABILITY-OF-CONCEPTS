#!/usr/bin/env python3
"""
Compare versions across experiments.

Usage:
    # Compare all versions in d1
    uv run python compare.py --dataset d1

    # Compare specific versions
    uv run python compare.py --versions d1/v0 d1/v1

    # Compare all datasets
    uv run python compare.py --all
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_version_metadata(version_dir: Path) -> Dict[str, Any]:
    """Load metadata from a version directory."""
    metadata_file = version_dir / '.metadata.json'
    if not metadata_file.exists():
        return {}

    with open(metadata_file, 'r') as f:
        return json.load(f)


def get_all_versions(parent_dir: Path) -> List[Path]:
    """Get all version directories in a parent directory."""
    versions = []

    for item in parent_dir.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            versions.append(item)

    # Sort by version number
    versions.sort(key=lambda p: int(p.name.split('_')[0][1:]))
    return versions


def get_dataset_dirs() -> List[Path]:
    """Get all dataset directories (d1, d2, d3)."""
    current = Path.cwd()
    datasets = []

    for item in current.iterdir():
        if item.is_dir() and item.name.startswith('d'):
            datasets.append(item)

    return sorted(datasets)


def format_metric(value: Any) -> str:
    """Format a metric value for display."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return str(value)
    return str(value)


def compare_dataset(dataset_dir: Path, detailed: bool = False) -> None:
    """Compare all versions in a dataset directory."""
    versions = get_all_versions(dataset_dir)

    if not versions:
        logger.info(f"No versions found in {dataset_dir.name}")
        return

    logger.info(f"\n{'=' * 100}")
    logger.info(f"Dataset: {dataset_dir.name}")
    logger.info(f"{'=' * 100}")

    # Collect metadata
    data = []
    for version_dir in versions:
        metadata = load_version_metadata(version_dir)
        data.append({
            'version': version_dir.name,
            'cv_mean': metadata.get('cv_mean'),
            'cv_std': metadata.get('cv_std'),
            'test_accuracy': metadata.get('test_accuracy'),
            'test_f1': metadata.get('test_f1'),
            'test_mae': metadata.get('test_mae'),
            'test_r2': metadata.get('test_r2'),
            'training_time': metadata.get('training_time_sec'),
            'model_type': metadata.get('model_type'),
            'samples': metadata.get('dataset_samples'),
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Select columns that have data
    cols_to_show = ['version']

    # Always show CV mean if available (primary metric)
    if df['cv_mean'].notna().any():
        cols_to_show.extend(['cv_mean', 'cv_std'])

    # Show test metrics based on task type
    if df['test_accuracy'].notna().any():
        cols_to_show.extend(['test_accuracy', 'test_f1'])
    if df['test_mae'].notna().any():
        cols_to_show.extend(['test_mae', 'test_r2'])

    cols_to_show.extend(['training_time', 'samples', 'model_type'])

    df_display = df[cols_to_show].copy()

    # Format numeric columns
    for col in ['cv_mean', 'cv_std', 'test_accuracy', 'test_f1', 'test_mae', 'test_r2', 'training_time']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_metric)

    print(df_display.to_string(index=False))

    # Show deltas if detailed
    if detailed and len(versions) > 1:
        logger.info("\nChanges between consecutive versions:")
        for i in range(1, len(data)):
            prev = data[i - 1]
            curr = data[i]

            logger.info(f"\n{prev['version']} → {curr['version']}:")

            # Compare CV mean (primary metric)
            if curr['cv_mean'] is not None and prev['cv_mean'] is not None:
                delta = curr['cv_mean'] - prev['cv_mean']
                sign = "+" if delta > 0 else ""
                logger.info(f"  CV mean: {sign}{delta:+.4f}")

            # Compare test accuracy
            if curr['test_accuracy'] is not None and prev['test_accuracy'] is not None:
                delta = curr['test_accuracy'] - prev['test_accuracy']
                sign = "+" if delta > 0 else ""
                logger.info(f"  Test accuracy: {sign}{delta:+.4f}")

            # Compare test MAE (for regression)
            if curr['test_mae'] is not None and prev['test_mae'] is not None:
                delta = curr['test_mae'] - prev['test_mae']
                sign = "+" if delta > 0 else ""
                logger.info(f"  Test MAE: {sign}{delta:+.4f}")


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description='Compare classifier versions')
    parser.add_argument('--dataset', type=str,
                        help='Compare versions in specific dataset (e.g., d1)')
    parser.add_argument('--versions', nargs='+',
                        help='Compare specific versions (e.g., d1/v0 d1/v1)')
    parser.add_argument('--all', action='store_true',
                        help='Compare all datasets')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed deltas between versions')
    args = parser.parse_args()

    if args.versions:
        # Compare specific versions
        logger.info("Comparing specific versions:")
        data = []

        for version_name in args.versions:
            version_dir = Path(version_name)
            if not version_dir.exists():
                logger.warning(f"Version not found: {version_name}")
                continue

            metadata = load_version_metadata(version_dir)
            data.append({
                'version': version_dir.name,
                'cv_mean': metadata.get('cv_mean'),
                'cv_std': metadata.get('cv_std'),
                'test_accuracy': metadata.get('test_accuracy'),
                'test_f1': metadata.get('test_f1'),
                'test_mae': metadata.get('test_mae'),
                'test_r2': metadata.get('test_r2'),
                'training_time': metadata.get('training_time_sec'),
                'model_type': metadata.get('model_type'),
            })

        if data:
            df = pd.DataFrame(data)
            print(df.to_string(index=False))

    elif args.dataset:
        # Compare versions in specific dataset
        dataset_dir = Path(args.dataset)
        compare_dataset(dataset_dir, detailed=args.detailed)

    elif args.all:
        # Compare all datasets
        for dataset_dir in get_dataset_dirs():
            compare_dataset(dataset_dir, detailed=args.detailed)

    else:
        # Default: compare current dataset if we're inside one
        current = Path.cwd()
        if current.name.startswith('d'):
            compare_dataset(current, detailed=args.detailed)
        else:
            parser.print_help()


if __name__ == '__main__':
    main()
