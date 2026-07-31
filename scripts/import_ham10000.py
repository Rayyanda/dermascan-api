"""
One-time bulk importer for the official HAM10000 dataset.

HAM10000 ships as:
  - HAM10000_metadata.csv          (columns: lesion_id, image_id, dx, dx_type, age, sex, localization)
  - HAM10000_images_part_1/*.jpg
  - HAM10000_images_part_2/*.jpg

This script reads the metadata CSV, looks up each image_id across the
image folder(s) you point it at, and COPIES every image straight into
data/dataset/<dx>/ — i.e. exactly the folder-per-class layout the backend
expects. It then registers everything in the database directly (no need
to go through the upload API or even start the server).

Usage:
    python scripts/import_ham10000.py \\
        --metadata /path/to/HAM10000_metadata.csv \\
        --images /path/to/HAM10000_images_part_1 /path/to/HAM10000_images_part_2

Run it from the dermascan_backend/ project root (so "app" is importable).
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATASET_DIR, CLASS_NAMES, ALLOWED_IMAGE_EXTENSIONS  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models_db import DatasetImage  # noqa: E402


def find_image_file(image_id: str, image_dirs: list[Path]) -> Path | None:
    for d in image_dirs:
        for ext in ALLOWED_IMAGE_EXTENSIONS:
            candidate = d / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Bulk-import HAM10000 into DermaScan's dataset folder.")
    parser.add_argument("--metadata", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--images", required=True, nargs="+", help="One or more image folders to search")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only import the first N rows (useful for a quick test)")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    image_dirs = [Path(p) for p in args.images]

    if not metadata_path.exists():
        print(f"ERROR: metadata file not found: {metadata_path}")
        sys.exit(1)
    for d in image_dirs:
        if not d.exists():
            print(f"ERROR: image folder not found: {d}")
            sys.exit(1)

    init_db()
    db = SessionLocal()

    known_paths = {row[0] for row in db.query(DatasetImage.filepath).all()}

    copied, skipped_missing, skipped_unknown_class, skipped_existing = 0, 0, 0, 0

    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.limit is not None and i >= args.limit:
                break

            dx = row["dx"].strip().lower()
            image_id = row["image_id"].strip()

            if dx not in CLASS_NAMES:
                skipped_unknown_class += 1
                continue

            src = find_image_file(image_id, image_dirs)
            if src is None:
                skipped_missing += 1
                continue

            class_dir = DATASET_DIR / dx
            class_dir.mkdir(parents=True, exist_ok=True)
            dest = class_dir / src.name

            if str(dest) in known_paths:
                skipped_existing += 1
                continue

            shutil.copyfile(src, dest)
            db.add(DatasetImage(
                filename=src.name,
                class_label=dx,
                filepath=str(dest),
                source="ham10000_bulk_import",
            ))
            copied += 1

            if copied % 200 == 0:
                db.commit()
                print(f"...{copied} images imported so far")

    db.commit()
    db.close()

    print("\n=== Import complete ===")
    print(f"Copied & registered : {copied}")
    print(f"Already registered  : {skipped_existing}")
    print(f"Missing image file  : {skipped_missing}")
    print(f"Unknown dx class    : {skipped_unknown_class}")
    print("\nRun the server and check /api/dataset/stats (or the AI Studio Dataset Manager) to confirm.")


if __name__ == "__main__":
    main()
