from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    csv_stem: str
    group: str | None = None


def load_records(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "image_path": row["image_path"],
                    "class_label": row.get("class_label", ""),
                    "attribute_label": int(row["attribute_label"]),
                }
            )
    return rows


def parse_concept_specs(dataset_config: dict, require_group: bool) -> list[ConceptSpec]:
    concepts = dataset_config.get("concepts", [])
    if not concepts:
        raise ValueError("dataset.concepts must contain at least one concept.")

    parsed: list[ConceptSpec] = []
    for concept in concepts:
        name = concept["name"]
        csv_stem = concept["csv_stem"]
        group = concept.get("group")
        if require_group and group not in {"affective", "non_affective"}:
            raise ValueError(
                f"Concept {name!r} must declare group=affective|non_affective."
            )
        parsed.append(ConceptSpec(name=name, csv_stem=csv_stem, group=group))
    return parsed


def resolve_csv_path(dataset_root: Path, csv_stem: str) -> Path:
    return dataset_root / "SUN" / f"{csv_stem}.csv"


def resolve_image_path(dataset_root: Path, image_rel_path: str) -> Path:
    return dataset_root / image_rel_path


def limit_records(records: list[dict], max_samples: int, seed: int) -> list[dict]:
    if max_samples <= 0 or len(records) <= max_samples:
        return list(records)
    rng = random.Random(seed)
    limited = list(records)
    rng.shuffle(limited)
    return limited[:max_samples]


def select_global_pairs(records: list[dict], num_pairs: int, seed: int) -> list[tuple[dict, dict]]:
    rng = random.Random(seed)
    positives = [row for row in records if row["attribute_label"] == 1]
    negatives = [row for row in records if row["attribute_label"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pair_count = min(len(positives), len(negatives))
    if num_pairs > 0:
        pair_count = min(pair_count, num_pairs)
    return [(positives[idx], negatives[idx]) for idx in range(pair_count)]


def select_class_aware_pairs(
    records: list[dict], num_pairs: int, seed: int
) -> list[tuple[dict, dict]]:
    rng = random.Random(seed)
    grouped = defaultdict(lambda: {"pos": [], "neg": []})
    for row in records:
        key = "pos" if row["attribute_label"] == 1 else "neg"
        grouped[row["class_label"]][key].append(row)

    pairs: list[tuple[dict, dict]] = []
    for group in grouped.values():
        if not group["pos"] or not group["neg"]:
            continue
        rng.shuffle(group["pos"])
        rng.shuffle(group["neg"])
        for idx in range(min(len(group["pos"]), len(group["neg"]))):
            pairs.append((group["pos"][idx], group["neg"][idx]))

    if num_pairs > 0 and len(pairs) >= num_pairs:
        rng.shuffle(pairs)
        return pairs[:num_pairs]

    positives = [row for row in records if row["attribute_label"] == 1]
    negatives = [row for row in records if row["attribute_label"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    fallback_idx = 0
    while len(pairs) < num_pairs and fallback_idx < min(len(positives), len(negatives)):
        pairs.append((positives[fallback_idx], negatives[fallback_idx]))
        fallback_idx += 1

    rng.shuffle(pairs)
    return pairs[:num_pairs] if num_pairs > 0 else pairs


def load_image_safe(
    image_path: Path,
    logger: Callable[[str], None] | None = None,
):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:  # pragma: no cover - depends on local files
        if logger is not None:
            logger(f"Skipping unreadable image {image_path}: {exc}")
        return None
