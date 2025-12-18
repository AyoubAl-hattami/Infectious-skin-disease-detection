"""
Organize images by removing duplicates and recording paths with Derm1M metadata.

For each disease:
- Read Derm1M CSVs (pretrain + validation)
- Match images by basename only (e.g., "youtube/abc.jpg" → "abc.jpg")
- Read duplicates JSON from reports/<Disease>/<Disease>_duplicates_dhash.json
- Copy non-duplicate images to: data/raw/<Disease>/
- Copy duplicate images to:     data/raw/<Disease>_duplicated/
- Create CSV: data/raw/<Disease>_images.csv with Derm1M metadata

Run:  python extract_unique_and_duplicates.py
"""

import json
import shutil
from pathlib import Path
import pandas as pd

# -------- CONFIG --------
PROJECT_ROOT = Path(".").resolve()

DISEASES = ["Cellulitis", "Folliculitis", "Impetigo", "Tinea"]

# Derm1M CSVs location
DERM1M_PRETRAIN_CSV = PROJECT_ROOT / "data" / "Derm1M_v2_pretrain.csv"
DERM1M_VALIDATION_CSV = PROJECT_ROOT / "data" / "Derm1M_v2_validation.csv"

# Where the extracted images currently are
OFFICIAL_DATA_DIR = PROJECT_ROOT / "Official data"

# Where QA reports & duplicate JSONs are stored
REPORTS_DIR = PROJECT_ROOT / "reports"

# Where organized data will be stored
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------


def load_derm1m_metadata():
    """
    Load both Derm1M CSVs and combine them.
    Create a lookup dict: basename -> full metadata row
    """
    print("\n📂 Loading Derm1M metadata...")
    
    if not DERM1M_PRETRAIN_CSV.exists():
        raise FileNotFoundError(f"Derm1M pretrain CSV not found: {DERM1M_PRETRAIN_CSV}")
    
    if not DERM1M_VALIDATION_CSV.exists():
        raise FileNotFoundError(f"Derm1M validation CSV not found: {DERM1M_VALIDATION_CSV}")
    
    df_pretrain = pd.read_csv(DERM1M_PRETRAIN_CSV)
    df_validation = pd.read_csv(DERM1M_VALIDATION_CSV)
    
    df_combined = pd.concat([df_pretrain, df_validation], ignore_index=True)
    
    print(f"   Pretrain: {len(df_pretrain)} rows")
    print(f"   Validation: {len(df_validation)} rows")
    print(f"   Combined: {len(df_combined)} rows")
    
    # Create lookup: basename -> metadata dict
    metadata_lookup = {}
    for idx, row in df_combined.iterrows():
        # filename in CSV is like "youtube/abc123.jpg"
        full_path = row['filename']
        basename = Path(full_path).name  # Extract just "abc123.jpg"
        
        # Store full row as dict (in case of duplicates, last one wins)
        metadata_lookup[basename] = row.to_dict()
    
    print(f"   Unique basenames: {len(metadata_lookup)}")
    return metadata_lookup


def load_duplicate_filenames(disease: str):
    """
    Read duplicates JSON and return a set of duplicate basenames to separate.
    """
    json_path = REPORTS_DIR / disease / f"{disease}_duplicates_dhash.json"
    if not json_path.exists():
        print(f"⚠️  WARNING: No duplicates JSON for {disease} at {json_path}")
        print(f"    Run 'python regenerate_duplicate_jsons.py' first!")
        return set()
    
    with open(json_path, "r", encoding='utf-8') as f:
        dup_dict = json.load(f)
    
    # dup_dict structure: { "anchor.jpg": ["dup1.jpg", "dup2.jpg"], ... }
    duplicates = set()
    for anchor, dup_list in dup_dict.items():
        # Keep anchor in main dataset, mark listed ones as duplicates
        duplicates.update(dup_list)
    
    print(f"   {disease}: {len(duplicates)} duplicates to separate")
    return duplicates


def process_disease(disease: str, metadata_lookup: dict):
    """
    Process one disease:
    - Find all images in Official data/<Disease>/
    - Match with Derm1M metadata by basename
    - Separate unique vs duplicate
    - Copy to appropriate folders
    - Generate CSV with metadata
    """
    print(f"\n{'='*70}")
    print(f"Processing: {disease}")
    print(f"{'='*70}")
    
    src_dir = OFFICIAL_DATA_DIR / disease
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    # Load duplicates for this disease
    duplicates = load_duplicate_filenames(disease)
    
    # Target directories
    unique_dir = RAW_DATA_DIR / disease
    dup_dir = RAW_DATA_DIR / f"{disease}_duplicated"
    
    unique_dir.mkdir(parents=True, exist_ok=True)
    dup_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all images
    image_paths = [
        p for p in src_dir.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]
    print(f"   Found {len(image_paths)} images in {src_dir}")
    
    if len(image_paths) == 0:
        print(f"   ⚠️  No images found, skipping {disease}")
        return
    
    # Records for CSV output
    records = []
    
    matched_count = 0
    unmatched_count = 0
    
    for img_path in image_paths:
        basename = img_path.name
        
        # Check if duplicate
        is_duplicate = basename in duplicates
        
        # Decide target folder
        if is_duplicate:
            target_folder = dup_dir
        else:
            target_folder = unique_dir
        
        target_path = target_folder / basename
        
        # Copy image
        shutil.copy2(img_path, target_path)
        
        # Get Derm1M metadata for this image
        metadata = metadata_lookup.get(basename, None)
        
        if metadata:
            matched_count += 1
            record = {
                "filename": basename,
                "label": disease,
                "is_duplicate": is_duplicate,
                "new_path": str(target_path.relative_to(PROJECT_ROOT)),
                "truncated_caption": metadata.get('truncated_caption', ''),
                "disease_label": metadata.get('disease_label', ''),
                "hierarchical_disease_label": metadata.get('hierarchical_disease_label', ''),
                "skin_concept": metadata.get('skin_concept', ''),
                "source": metadata.get('source', ''),
                "source_type": metadata.get('source_type', ''),
            }
        else:
            unmatched_count += 1
            # Image not found in Derm1M CSV - still record it but with empty metadata
            record = {
                "filename": basename,
                "label": disease,
                "is_duplicate": is_duplicate,
                "new_path": str(target_path.relative_to(PROJECT_ROOT)),
                "truncated_caption": '',
                "disease_label": '',
                "hierarchical_disease_label": '',
                "skin_concept": '',
                "source": '',
                "source_type": '',
            }
        
        records.append(record)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(records)
    csv_path = RAW_DATA_DIR / f"{disease}_images.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Summary
    unique_count = (~df['is_duplicate']).sum()
    dup_count = df['is_duplicate'].sum()
    
    print(f"\n   ✅ Results:")
    print(f"      Total images processed: {len(df)}")
    print(f"      Unique (copied to {disease}/): {unique_count}")
    print(f"      Duplicates (copied to {disease}_duplicated/): {dup_count}")
    print(f"      Matched with Derm1M metadata: {matched_count}")
    print(f"      Not found in Derm1M: {unmatched_count}")
    print(f"      CSV saved: {csv_path}")


def main():
    print("="*70)
    print("EXTRACTING UNIQUE & DUPLICATE IMAGES WITH DERM1M METADATA")
    print("="*70)
    
    # Check if Derm1M CSVs exist
    if not DERM1M_PRETRAIN_CSV.exists():
        print(f"❌ ERROR: Derm1M pretrain CSV not found: {DERM1M_PRETRAIN_CSV}")
        return
    
    if not DERM1M_VALIDATION_CSV.exists():
        print(f"❌ ERROR: Derm1M validation CSV not found: {DERM1M_VALIDATION_CSV}")
        return
    
    # Load Derm1M metadata once
    metadata_lookup = load_derm1m_metadata()
    
    # Process each disease
    for disease in DISEASES:
        process_disease(disease, metadata_lookup)
    
    print(f"\n{'='*70}")
    print("✅ ALL DONE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Check data/raw/<Disease>/ for unique images")
    print("2. Check data/raw/<Disease>_duplicated/ for duplicate images")
    print("3. Check data/raw/<Disease>_images.csv for metadata")


if __name__ == "__main__":
    main()
