#!/usr/bin/env python3
"""
Start a new experiment from a previous version.

Copies a version's files to working/ and logs the intent.

Usage:
    # Start from d1/v1
    uv run python new_experiment.py --from d1/v1 --note "Try SVM model"

    # Start from latest version in current dataset
    uv run python new_experiment.py --from latest --note "Tune hyperparameters"
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for framework imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.shared_utils import get_latest_version

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def copy_version_to_working(version_dir: Path, working_dir: Path) -> None:
    """Copy files from version directory to working/."""
    working_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        'config.yaml',
        'dataset.config',
        'experiment_notes.md'
    ]

    # Don't copy model.pkl or results.json to avoid confusion
    for filename in files_to_copy:
        src = version_dir / filename
        dst = working_dir / filename

        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied {filename}")


def main():
    """Main script."""
    parser = argparse.ArgumentParser(description='Start new experiment from previous version')
    parser.add_argument('--from', dest='from_version', type=str, required=True,
                        help='Version to copy from (e.g., d1/v1 or latest)')
    parser.add_argument('--note', type=str, default='',
                        help='Note about what you\'re trying in this experiment')
    args = parser.parse_args()

    current_dir = Path.cwd()
    working_dir = current_dir / 'working'

    # Resolve version directory
    if args.from_version == 'latest':
        version_dir = get_latest_version(current_dir)
        if version_dir is None:
            logger.error("No versions found in this directory")
            return
    else:
        version_dir = Path(args.from_version)

    if not version_dir.exists():
        logger.error(f"Version directory not found: {version_dir}")
        return

    logger.info("=" * 80)
    logger.info("STARTING NEW EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Copying from: {version_dir.name}")

    # Copy files
    copy_version_to_working(version_dir, working_dir)

    # Log to experiment notes
    if args.note:
        notes_path = working_dir / 'experiment_notes.md'
        with open(notes_path, 'a') as f:
            f.write(f"\n\n## Experiment started {datetime.now().isoformat()}\n")
            f.write(f"Based on: {version_dir.name}\n")
            f.write(f"Plan: {args.note}\n")

        logger.info(f"Logged note to experiment_notes.md")

    logger.info("=" * 80)
    logger.info("Ready to experiment!")
    logger.info(f"Next: Edit config.yaml or dataset.config, then run train.py")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
