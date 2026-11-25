"""Create seeded, RGB-image-disjoint ITHV JSONL splits from JSON or JSONL files."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data import DEFAULT_DATA_DIR, SPLITS, read_rows, validate_row, validate_splits


RAW_FILES = {
    "obvious": "all_obvious.json",
    "elusive": "all_elusive.json",
    "not_applicable": "all_not_applicable.json",
}


def infer_task_type(question, answer):
    if question.startswith("Please determine whether the description is inconsistent with the image"):
        return "correction"
    if re.search(r"(?m)^\(?A[.、)]\s", question) and re.search(r"(?m)^\(?B[.、)]\s", question):
        return "multiple_choice"
    if answer.strip().casefold() in {"yes", "no", "是", "否"}:
        return "binary"
    return "short_answer"


def load_input(input_dir):
    """Merge existing splits, or convert the three public annotation arrays."""
    layouts = {
        "jsonl": [input_dir / f"{name}.jsonl" for name in SPLITS],
        "json": [input_dir / name for name in RAW_FILES.values()],
    }
    formats = [name for name, paths in layouts.items() if all(p.is_file() for p in paths)]
    if len(formats) != 1:
        raise ValueError("Input must contain exactly one complete layout: train/validation/test.jsonl "
                         "or all_obvious/all_elusive/all_not_applicable.json")
    input_format = formats[0]
    rows, sources = [], []
    for category, path in zip(RAW_FILES, layouts[input_format]):
        payload = path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        if input_format == "jsonl":
            part = read_rows(path)
        else:
            originals = json.loads(payload)
            if not isinstance(originals, list) or not originals:
                raise ValueError(f"Expected a nonempty JSON array: {path.name}")
            part = []
            for index, original in enumerate(originals):
                if not isinstance(original, dict) or not {"image", "text", "gt"} <= original.keys():
                    raise ValueError(f"Missing image/text/gt: {path.name}, record {index}")
                question, answer = original["text"], original["gt"]
                if not isinstance(question, str) or not isinstance(answer, str):
                    raise ValueError(f"Non-string text/gt: {path.name}, record {index}")
                row = {
                    "prompt": f"<image>\n{question}",
                    "question": question,
                    "answer": answer,
                    "images": [original["image"]],
                    "acc": original.get("acc"),
                    "task_type": original.get("task_type", infer_task_type(question, answer)),
                    "hallucination_type": category,
                    "sample_id": f"{category}:{index:06d}:{sha256[:12]}",
                }
                validate_row(row)
                part.append(row)
        rows.extend(part)
        # Filenames and checksums are portable; no workstation paths are exported.
        sources.append({"file": path.name, "sha256": sha256, "rows": len(part)})
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Input sample IDs must be globally unique")
    return rows, input_format, sources


def image_groups(rows, image_root):
    """Keep identical decoded RGB images together, including filename aliases."""
    groups = {}
    for image_path in sorted({r["images"][0] for r in rows}):
        with Image.open(image_root / image_path) as image:
            rgb = image.convert("RGB")
            payload = str(rgb.size).encode() + b":" + rgb.tobytes()
        groups[image_path] = hashlib.sha256(payload).hexdigest()
    return groups


def source_dataset(row):
    return row["images"][0].split("/", 1)[0]


def choose_split(rows, groups, seed=42, ratios=(0.7, 0.1, 0.2), trials=5000):
    """Select by metadata balance only; no model outputs or evaluation scores."""
    if not 0 <= seed < 2**32 or trials < 1:
        raise ValueError("Seed must be in [0, 2**32), and trials must be positive")
    if len(ratios) != 3 or any(not math.isfinite(x) or x <= 0 for x in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Supply three positive split ratios summing to 1")
    by_group = defaultdict(list)
    for row in rows:
        by_group[groups[row["images"][0]]].append(row)
    group_ids = sorted(by_group)
    cuts = (round(len(group_ids) * ratios[0]), round(len(group_ids) * (ratios[0] + ratios[1])))
    if not 0 < cuts[0] < cuts[1] < len(group_ids):
        raise ValueError("Not enough independent image groups for these three split ratios")

    def stratum(row):
        return (row["hallucination_type"], source_dataset(row), row["task_type"],
                row["answer"].strip().casefold() if row["task_type"] == "binary" else "other")

    strata = {key: i + 1 for i, key in enumerate(sorted({stratum(r) for r in rows}))}
    counts = np.zeros((len(group_ids), len(strata) + 1), dtype=np.int64)
    for i, group in enumerate(group_ids):
        for row in by_group[group]:
            counts[i, 0] += 1
            counts[i, strata[stratum(row)]] += 1
    totals = counts.sum(axis=0)
    targets = np.asarray(ratios)[:, None] * totals
    rng = np.random.default_rng(seed)
    best_score, best = float("inf"), None
    for _ in range(trials):
        partitions = np.split(rng.permutation(len(group_ids)), cuts)
        actual = np.stack([counts[p].sum(axis=0) for p in partitions])
        # Small strata cannot exactly meet fractional targets.
        deviations = (actual - targets) / np.maximum(totals, 20)
        score = float(np.mean(deviations[:, 1:] ** 2) + 5 * np.mean(deviations[:, 0] ** 2))
        if score < best_score:
            best_score, best = score, partitions
    assignment = {group_ids[i]: split for split, part in zip(SPLITS, best) for i in part}
    result = {split: [r for r in rows if assignment[groups[r["images"][0]]] == split] for split in SPLITS}
    for part in result.values():
        rng.shuffle(part)
    return result, best_score


def create_splits(input_dir, output_dir, image_root, seed=42, ratios=(0.7, 0.1, 0.2), trials=5000):
    input_dir, output_dir, image_root = (Path(p).expanduser() for p in (input_dir, output_dir, image_root))
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new directory")
    rows, input_format, sources = load_input(input_dir)
    groups = image_groups(rows, image_root)
    partitions, score = choose_split(rows, groups, seed, ratios, trials)
    owners = {}
    for split, part in partitions.items():
        for row in part:
            group = groups[row["images"][0]]
            if owners.setdefault(group, split) != split:
                raise RuntimeError("Cross-split RGB-image leakage")
    if Counter(r["sample_id"] for part in partitions.values() for r in part) != Counter(r["sample_id"] for r in rows):
        raise RuntimeError("Samples were lost or duplicated during splitting")
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, part in partitions.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in part:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = validate_splits(output_dir, image_root, check_images=True)
    for split, part in partitions.items():
        report[split].update({
            "row_fraction": len(part) / len(rows),
            "unique_image_paths": len({r["images"][0] for r in part}),
            "image_groups": len({groups[r["images"][0]] for r in part}),
            "tasks": dict(sorted(Counter(r["task_type"] for r in part).items())),
            "sources": dict(sorted(Counter(source_dataset(r) for r in part).items())),
        })
    manifest = {
        "version": 1,
        "split_version": "ithv_seeded_rgb_groups_v1",
        "seed": seed,
        "target_ratios": dict(zip(SPLITS, ratios)),
        "input_format": input_format,
        "inputs": sources,
        "split_method": "RGB pixel-hash groups; best metadata balance among seeded random group partitions",
        "trials": trials,
        "split_balance_score": score,
        "total_rows": len(rows),
        "unique_image_paths": len(groups),
        "unique_image_groups": len(set(groups.values())),
        "all_rows_preserved": True,
        "cross_split_image_path_overlap": 0,
        "cross_split_rgb_group_overlap": 0,
        "splits": report,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing existing JSONL splits or the three public annotation JSON arrays")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are never overwritten")
    parser.add_argument("--image-root", type=Path, required=True, help="Source images, required to group by decoded RGB content")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--trials", type=int, default=5000, help="Number of metadata-balanced candidate partitions")
    args = parser.parse_args(argv)
    report = create_splits(args.input_dir, args.output_dir, args.image_root, args.seed, args.ratios, args.trials)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
