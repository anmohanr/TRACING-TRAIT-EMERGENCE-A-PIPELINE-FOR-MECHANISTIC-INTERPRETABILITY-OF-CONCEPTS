#!/usr/bin/env python3
"""
Promote working/ directory to a versioned checkpoint.

Usage:
    cd d1_embedding_to_feature
    uv run python ../framework/checkpoint.py --name "bigger_dataset" --message "500 samples, improved accuracy"
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for framework imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.shared_utils import get_next_version_number, load_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_metadata_file(working_dir: Path, version_dir: Path, version_name: str) -> None:
    """Create .metadata.json for the checkpoint."""
    # Try to load results if they exist
    results_path = working_dir / 'results.json'
    results = {}

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

    # Load dataset config info
    dataset_config_path = working_dir / 'dataset.config'
    dataset_path = ""
    if dataset_config_path.exists():
        with open(dataset_config_path, 'r') as f:
            dataset_path = f.read().strip()

    metadata = {
        'version': version_name,
        'created_at': datetime.now().isoformat(),
        'dataset_path': dataset_path,
        'dataset_samples': results.get('metadata', {}).get('n_samples'),
        'model_type': results.get('model_type'),
    }

    # Add key metrics if available
    if 'cv_mean' in results:
        metadata['cv_mean'] = results['cv_mean']
    if 'cv_std' in results:
        metadata['cv_std'] = results['cv_std']
    if 'test_accuracy' in results:
        metadata['test_accuracy'] = results['test_accuracy']
    if 'test_f1' in results:
        metadata['test_f1'] = results['test_f1']
    if 'test_mae' in results:
        metadata['test_mae'] = results['test_mae']
    if 'test_r2' in results:
        metadata['test_r2'] = results['test_r2']
    if 'training_time_sec' in results:
        metadata['training_time_sec'] = results['training_time_sec']

    metadata_path = version_dir / '.metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created .metadata.json")


def create_readme_template(version_dir: Path, version_name: str, message: str) -> None:
    """Create README.md template for the version."""
    results_path = version_dir / 'results.json'
    results = {}

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

    # Build metrics section
    metrics_section = ""
    if 'cv_mean' in results:
        metrics_section += f"- CV Score: {results['cv_mean']:.4f} ± {results.get('cv_std', 0):.4f}\n"
    if 'test_accuracy' in results:
        metrics_section += f"- Test Accuracy: {results['test_accuracy']:.4f}\n"
    if 'test_f1' in results:
        metrics_section += f"- Test F1 Score: {results['test_f1']:.4f}\n"
    if 'test_mae' in results:
        metrics_section += f"- Test MAE: {results['test_mae']:.4f}\n"
    if 'test_mse' in results:
        metrics_section += f"- Test MSE: {results['test_mse']:.4f}\n"
    if 'test_r2' in results:
        metrics_section += f"- Test R²: {results['test_r2']:.4f}\n"

    readme_content = f"""# {version_name}

**What changed from previous version:**

{message}

## Results

{metrics_section if metrics_section else "- [Training metrics will appear here after running train.py]\n"}

**Training time:** {results.get('training_time_sec', 'N/A'):.2f}s

**Dataset:** {results.get('metadata', {}).get('dataset_path', 'N/A')}

## Model Details

- **Model type:** {results.get('model_type', 'N/A')}
- **Config:** See config.yaml

## Next Steps

- [ ] Idea 1: ...
- [ ] Idea 2: ...
"""

    readme_path = version_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    logger.info(f"Created README.md template")


def copy_working_to_version(working_dir: Path, version_dir: Path) -> None:
    """Copy all files from working/ to version directory."""
    version_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        'config.yaml',
        'dataset.config',
        'model.pkl',
        'results.json',
        'experiment_notes.md'
    ]

    for filename in files_to_copy:
        src = working_dir / filename
        dst = version_dir / filename

        if src.exists():
            if src.suffix == '.pkl':
                shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
            logger.info(f"Copied {filename}")
        else:
            logger.debug(f"File not found (optional): {filename}")


def main():
    """Main checkpoint script."""
    parser = argparse.ArgumentParser(description='Promote working/ to versioned checkpoint')
    parser.add_argument('--name', type=str, required=True,
                        help='Name for this version (e.g., "bigger_dataset")')
    parser.add_argument('--message', type=str, default='',
                        help='Description of changes from previous version')
    args = parser.parse_args()

    current_dir = Path.cwd()
    working_dir = current_dir / 'working'

    if not working_dir.exists():
        logger.error(f"working/ directory not found in {current_dir}")
        return

    # Get next version number
    version_num = get_next_version_number(current_dir)
    version_name = f"v{version_num}_{args.name}"
    version_dir = current_dir / version_name

    logger.info("=" * 80)
    logger.info("CREATING CHECKPOINT")
    logger.info("=" * 80)
    logger.info(f"Creating {version_name}/")

    # Copy files
    copy_working_to_version(working_dir, version_dir)

    # Create metadata
    create_metadata_file(working_dir, version_dir, version_name)

    # Create README template
    create_readme_template(version_dir, version_name, args.message)

    logger.info("=" * 80)
    logger.info(f"✓ Checkpoint saved as {version_name}/")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
