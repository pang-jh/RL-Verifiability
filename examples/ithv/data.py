
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "ithv"
FIELDS = {"prompt", "question", "answer", "images", "acc", "task_type", "hallucination_type", "sample_id"}
SPLITS = ("train", "validation", "test")


def validate_row(row):
    """Validate the portable training schema without modifying a record."""
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ValueError("ITHV rows must contain exactly the eight documented fields")
    for key in FIELDS - {"images", "acc"}:
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"Invalid {key}")
    kind, acc = row["hallucination_type"], row["acc"]
    if kind == "not_applicable":
        if acc is not None:
            raise ValueError("NA requires acc=null")
    elif kind not in {"obvious", "elusive"} or type(acc) not in (float, int) or not math.isfinite(acc) or not 0 <= acc <= 1:
        raise ValueError("OH/EH require a finite human accuracy in [0, 1]")
    images = row["images"]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError("Exactly one image path is required")
    relative = PurePosixPath(images[0])
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or "\\" in images[0]:
        raise ValueError("Release images must be safe relative POSIX paths")
    if row["prompt"].count("<image>") != 1:
        raise ValueError("Exactly one image placeholder is required")


def read_rows(path, image_root=None, check_images=False):
    rows = []
    root = Path(image_root).expanduser().resolve() if image_root is not None else None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row)
            images = row["images"]
            relative = PurePosixPath(images[0])
            if root is not None:
                # Relative paths cannot traverse upward; allow user-managed image
                # directory symlinks just as the training dataset loader does.
                image = root / relative
                image.relative_to(root)
                if check_images and not image.is_file():
                    raise FileNotFoundError(image)
                row["images"] = [str(image)]
            elif check_images:
                raise ValueError("--image-root is required to check image files")
            rows.append(row)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("A split must be nonempty and contain unique sample IDs")
    return rows


def validate_splits(data_dir=DEFAULT_DATA_DIR, image_root=None, check_images=False):
    report, ids, image_owners = {}, set(), {}
    for split in SPLITS:
        path = Path(data_dir) / f"{split}.jsonl"
        rows = read_rows(path, image_root, check_images)
        for row in rows:
            if row["sample_id"] in ids:
                raise ValueError("A sample ID occurs in more than one split")
            ids.add(row["sample_id"])
            image = row["images"][0]
            if image_owners.setdefault(image, split) != split:
                raise ValueError("An image path occurs in more than one split")
        report[split] = {
            "rows": len(rows),
            "categories": dict(Counter(r["hallucination_type"] for r in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_splits(args.data_dir, args.image_root, args.image_root is not None), indent=2))


if __name__ == "__main__":
    main()
