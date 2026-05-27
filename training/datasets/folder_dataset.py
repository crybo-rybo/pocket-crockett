"""Folder-based dataset backed by split CSVs and unpacked shard images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from training.common.label_map import load_label_map, scientific_name_to_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = ROOT / "vision" / "splits"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
NORMALIZATION_PRESETS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "openai_clip": (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    "clip": (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    "bioclip": (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
}


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    path: str
    scientific_name: str
    taxon_id: str
    label: int


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_split_records(
    split: str,
    data_root: Path,
    splits_dir: Path,
    name_to_index: dict[str, int],
    *,
    require_existing: bool = True,
) -> list[SampleRecord]:
    split_path = splits_dir / f"{split}.csv"
    records: list[SampleRecord] = []
    missing = 0
    unknown = 0
    for row in read_split_csv(split_path):
        scientific_name = row.get("scientific_name") or ""
        if scientific_name not in name_to_index:
            unknown += 1
            continue
        rel_path = row["path"]
        abs_path = data_root / rel_path
        if require_existing and not abs_path.exists():
            missing += 1
            continue
        records.append(
            SampleRecord(
                image_id=row["image_id"],
                path=rel_path,
                scientific_name=scientific_name,
                taxon_id=row.get("taxon_id", ""),
                label=name_to_index[scientific_name],
            )
        )
    if require_existing and missing:
        print(f"[{split}] skipped {missing} rows with missing files under {data_root}")
    if unknown:
        print(f"[{split}] skipped {unknown} rows with labels outside the output class map")
    return records


def _float_tuple(value, *, name: str, length: int = 3) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must be a sequence of {length} numbers")
    return tuple(float(v) for v in value)


def resolve_normalization(data_config: dict | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    data_config = data_config or {}
    if "mean" in data_config or "std" in data_config:
        if "mean" not in data_config or "std" not in data_config:
            raise ValueError("data.mean and data.std must be set together")
        return (
            _float_tuple(data_config["mean"], name="data.mean"),
            _float_tuple(data_config["std"], name="data.std"),
        )

    preset = str(data_config.get("normalization", "imagenet")).lower()
    if preset not in NORMALIZATION_PRESETS:
        known = ", ".join(sorted(NORMALIZATION_PRESETS))
        raise ValueError(f"Unknown normalization preset {preset!r}; expected one of: {known}")
    return NORMALIZATION_PRESETS[preset]


def resolve_crop_scale(data_config: dict | None = None) -> tuple[float, float]:
    data_config = data_config or {}
    value = data_config.get("random_resized_crop_scale", (0.8, 1.0))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("data.random_resized_crop_scale must be a pair of numbers")
    lo, hi = (float(value[0]), float(value[1]))
    if not 0 < lo <= hi <= 1:
        raise ValueError("data.random_resized_crop_scale must satisfy 0 < min <= max <= 1")
    return lo, hi


def build_transforms(
    image_size: int,
    augment: bool,
    *,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    random_resized_crop_scale: tuple[float, float] = (0.8, 1.0),
) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=random_resized_crop_scale),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class FolderDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        data_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.data_root / record.path
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        return tensor, record.label, record.image_id


class PretrainingFolderDataset(Dataset):
    """Species-level labels for the PlantNet pretraining pool."""

    def __init__(
        self,
        records: list[SampleRecord],
        data_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.data_root / record.path
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        return tensor, record.label, record.scientific_name


def load_pretraining_records(
    data_root: Path,
    manifest_gz: Path | None = None,
    *,
    require_existing: bool = True,
) -> tuple[list[SampleRecord], dict[str, int]]:
    import gzip

    manifest_path = manifest_gz or (ROOT / "vision" / "images" / "manifest.csv.gz")
    with gzip.open(manifest_path, "rt", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    species = sorted({r["scientific_name"] for r in rows if r.get("split") == "pretraining_only"})
    name_to_index = {name: idx for idx, name in enumerate(species)}
    records: list[SampleRecord] = []
    for row in rows:
        if row.get("split") != "pretraining_only":
            continue
        rel_path = row["path"]
        if require_existing and not (data_root / rel_path).exists():
            continue
        scientific_name = row["scientific_name"]
        records.append(
            SampleRecord(
                image_id=row["image_id"],
                path=rel_path,
                scientific_name=scientific_name,
                taxon_id=row.get("taxon_id", ""),
                label=name_to_index[scientific_name],
            )
        )
    return records, name_to_index


def compute_class_weights(records: list[SampleRecord], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for record in records:
        counts[record.label] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def build_dataloader(
    records: list[SampleRecord],
    data_root: Path,
    *,
    batch_size: int,
    image_size: int,
    augment: bool,
    num_workers: int,
    shuffle: bool,
    weighted_sampler: bool,
    drop_last: bool = False,
    data_config: dict | None = None,
) -> DataLoader:
    mean, std = resolve_normalization(data_config)
    transform = build_transforms(
        image_size,
        augment=augment,
        mean=mean,
        std=std,
        random_resized_crop_scale=resolve_crop_scale(data_config),
    )
    dataset = FolderDataset(records, data_root, transform)
    sampler = None
    if weighted_sampler and shuffle:
        class_counts = torch.zeros(max(r.label for r in records) + 1 if records else 0, dtype=torch.float32)
        for record in records:
            class_counts[record.label] += 1.0
        sample_weights = torch.tensor([1.0 / class_counts[r.label] for r in records], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(records), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
