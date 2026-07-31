"""
Generic bulk importer — for when you have a pile of UNSORTED images and a
simple label sheet (2 columns: filename, class), instead of having to
drag every image into its own class folder by hand.

Label sheet can be .csv or .xlsx, e.g.:

    filename          , label
    lesion_001.jpg    , mel
    lesion_002.jpg    , nv
    lesion_003.jpg    , bcc

Column names are configurable via --filename-col / --label-col in case
your sheet uses different headers (e.g. HAM10000's "image_id"/"dx").

The script searches --images (searched recursively, so images can be in
subfolders too — you don't need to flatten them first) for each filename,
copies it into data/dataset/<label>/, and registers it in the database.

Usage:
    python scripts/import_from_csv.py \\
        --labels my_labels.csv \\
        --images /path/to/my/photos \\
        --filename-col filename --label-col label

    # xlsx works too:
    python scripts/import_from_csv.py --labels my_labels.xlsx --images ./photos

Run from the dermascan_backend/ project root (so "app" is importable).
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATASET_DIR, CLASS_NAMES  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models_db import DatasetImage  # noqa: E402


def read_label_rows(labels_path: Path, filename_col: str, label_col: str):
    if labels_path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError:
            print("ERROR: reading .xlsx needs openpyxl. Install it with:")
            print("    pip install openpyxl --break-system-packages")
            sys.exit(1)
        wb = openpyxl.load_workbook(labels_path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        headers = [str(h).strip() for h in rows[0]]
        for row in rows[1:]:
            yield dict(zip(headers, row))
    else:
        with labels_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def find_image_file(filename: str, search_root: Path) -> Path | None:
    # Exact name first (fast path), then a recursive fallback.
    direct = search_root / filename
    if direct.exists():
        return direct
    matches = list(search_root.rglob(filename))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-import unsorted images using a filename->label sheet (csv or xlsx)."
    )
    parser.add_argument("--labels", required=True, help="Path to labels sheet (.csv or .xlsx)")
    parser.add_argument("--images", required=True, help="Folder containing the (unsorted) images — searched recursively")
    parser.add_argument("--filename-col", default="filename", help="Column name holding the image filename (default: filename)")
    parser.add_argument("--label-col", default="label", help="Column name holding the class label (default: label)")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    images_root = Path(args.images)

    if not labels_path.exists():
        print(f"ERROR: labels file not found: {labels_path}")
        sys.exit(1)
    if not images_root.exists():
        print(f"ERROR: images folder not found: {images_root}")
        sys.exit(1)

    init_db()
    db = SessionLocal()
    known_paths = {row[0] for row in db.query(DatasetImage.filepath).all()}

    copied, skipped_missing, skipped_unknown_class, skipped_existing, skipped_bad_row = 0, 0, 0, 0, 0

    for row in read_label_rows(labels_path, args.filename_col, args.label_col):
        filename = str(row.get(args.filename_col, "")).strip()
        label = str(row.get(args.label_col, "")).strip().lower()

        if not filename or not label:
            skipped_bad_row += 1
            continue

        if label not in CLASS_NAMES:
            skipped_unknown_class += 1
            continue

        src = find_image_file(filename, images_root)
        if src is None:
            skipped_missing += 1
            continue

        class_dir = DATASET_DIR / label
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / src.name

        if str(dest) in known_paths:
            skipped_existing += 1
            continue

        shutil.copyfile(src, dest)
        db.add(DatasetImage(
            filename=src.name,
            class_label=label,
            filepath=str(dest),
            source="csv_bulk_import",
        ))
        copied += 1

        if copied % 200 == 0:
            db.commit()
            print(f"...{copied} images imported so far")

    db.commit()
    db.close()

    print("\n=== Import complete ===")
    print(f"Copied & registered     : {copied}")
    print(f"Already registered      : {skipped_existing}")
    print(f"Image file not found    : {skipped_missing}")
    print(f"Unknown class label     : {skipped_unknown_class}  (must be one of {CLASS_NAMES})")
    print(f"Row missing filename/label : {skipped_bad_row}")
    print("\nRun the server and check /api/dataset/stats (or the AI Studio Dataset Manager) to confirm.")


if __name__ == "__main__":
    main()
