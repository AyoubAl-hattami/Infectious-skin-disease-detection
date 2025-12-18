# dvc_setup.py

import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dvc_tracking():
    """
    Setup DVC tracking for processed data.
    Tracks only generated/processed files, not raw source data.
    """
    
    print("="*60)
    print("SETTING UP DVC TRACKING")
    print("="*60)
    
    # Check if we're in project root
    if not Path('src').exists() or not Path('data').exists():
        logger.error("❌ Please run this script from project root directory")
        return
    
    # 1. Initialize DVC (if not already done)
    logger.info("\n1. Checking DVC initialization...")
    if not Path('.dvc').exists():
        subprocess.run(['dvc', 'init'], check=True)
        logger.info("✓ DVC initialized")
    else:
        logger.info("✓ DVC already initialized")
    
    # 2. Configure DagsHub remote
    logger.info("\n2. Setting up DagsHub remote...")
    dagshub_url = "https://dagshub.com/awob501/my-first-repo.dvc"
    
    try:
        subprocess.run(
            ['dvc', 'remote', 'add', '-d', 'origin', dagshub_url],
            check=False  # Don't fail if already exists
        )
        logger.info(f"✓ DagsHub remote configured: {dagshub_url}")
    except subprocess.CalledProcessError:
        logger.info("✓ Remote already exists")
    
    # 3. Track ONLY processed data with DVC
    logger.info("\n3. Tracking processed data directories...")
    
    # Directories to track (ONLY generated/processed data)
    dvc_tracked_dirs = [
        'data/processed',  # Our preprocessed images
        'reports'          # Quality reports
    ]
    
    for data_dir in dvc_tracked_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            logger.info(f"\n  Tracking: {data_dir}")
            try:
                result = subprocess.run(
                    ['dvc', 'add', data_dir],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"    ✓ Successfully added to DVC")
                else:
                    logger.warning(f"    ⚠ Already tracked or error: {result.stderr}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"    ⚠ Error: {e}")
        else:
            logger.warning(f"  ⚠ Directory not found: {data_dir}")
    
    # 4. Update .gitignore
    logger.info("\n4. Updating .gitignore...")
    
    gitignore_entries = [
        "# DVC tracked directories (tracked by DVC, ignored by Git)",
        "/data/processed",
        "/reports",
        "",
        "# Python",
        "*.pyc",
        "__pycache__/",
        "*.egg-info/",
        "",
        "# Models (will be DVC tracked later)",
        "*.pt",
        "*.pth",
        "*.h5",
        "/models/",
        "",
        "# MLflow",
        "mlruns/",
        "",
        "# Jupyter",
        ".ipynb_checkpoints/",
        "",
        "# IDEs",
        ".vscode/",
        ".idea/",
        "",
        "# Logs",
        "logs/",
        "*.log"
    ]
    
    gitignore_path = Path('.gitignore')
    
    # Read existing entries
    existing_content = ""
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    
    # Append new entries if not already present
    with open(gitignore_path, 'a') as f:
        if existing_content and not existing_content.endswith('\n'):
            f.write('\n')
        
        for entry in gitignore_entries:
            if entry and entry not in existing_content:
                f.write(f"{entry}\n")
    
    logger.info("✓ .gitignore updated")
    
    # 5. Show what was created
    logger.info("\n5. DVC tracking files created:")
    
    dvc_files = []
    for tracked_dir in dvc_tracked_dirs:
        dvc_file = f"{tracked_dir}.dvc"
        if Path(dvc_file).exists():
            dvc_files.append(dvc_file)
            logger.info(f"  ✓ {dvc_file}")
    
    # 6. Instructions for Git commit
    print("\n" + "="*60)
    print("NEXT STEPS - Run these commands:")
    print("="*60)
    
    if dvc_files:
        print("\n1. Stage DVC files for Git:")
        print("   git add .dvc .gitignore " + " ".join(dvc_files))
        print("\n2. Commit to Git:")
        print("   git commit -m 'Add DVC tracking for processed data'")
        print("\n3. Push data to DagsHub DVC remote:")
        print("   dvc push")
        print("\n4. Push Git changes:")
        print("   git push")
    else:
        print("\n⚠ No DVC files created. Make sure directories exist:")
        for d in dvc_tracked_dirs:
            print(f"   - {d}")
    
    print("\n" + "="*60)
    print("✓ DVC SETUP COMPLETED")
    print("="*60)
    
    # 7. Show current tracking status
    logger.info("\n6. Current DVC tracking status:")
    try:
        result = subprocess.run(['dvc', 'status'], capture_output=True, text=True)
        print(result.stdout)
    except:
        logger.info("  (Run 'dvc status' manually to check)")


if __name__ == "__main__":
    setup_dvc_tracking()