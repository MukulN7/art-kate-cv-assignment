import os
import random
import shutil
from pathlib import Path

# Configuration constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT / "data" / "train" / "images"
VAL_DIR = PROJECT_ROOT / "data" / "val" / "images"

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
EXPECTED_RAW_COUNT = 74
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.jpg.jpeg')


def get_image_files(directory: Path):
    """Find all image files supporting single and dual extensions."""
    files = []
    for entry in directory.iterdir():
        if entry.is_file():
            name_lower = entry.name.lower()
            if any(name_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                files.append(entry)
    return sorted(files, key=lambda p: p.name)


def split_dataset():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found at: {RAW_DIR}")

    image_files = get_image_files(RAW_DIR)
    total_images = len(image_files)

    if total_images != EXPECTED_RAW_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_RAW_COUNT} raw images, but found {total_images}."
        )

    # Set fixed random seed for reproducible split
    random.seed(RANDOM_SEED)
    shuffled_files = list(image_files)
    random.shuffle(shuffled_files)

    # Calculate train/val counts (80/20 split)
    train_count = round(total_images * TRAIN_RATIO)
    train_files = shuffled_files[:train_count]
    val_files = shuffled_files[train_count:]

    # Prepare target directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy files without modifying originals
    for f in train_files:
        shutil.copy2(f, TRAIN_DIR / f.name)

    for f in val_files:
        shutil.copy2(f, VAL_DIR / f.name)

    print("=== Dataset Split Summary ===")
    print(f"Total raw images discovered: {total_images}")
    print(f"Train images count: {len(train_files)}")
    print(f"Validation images count: {len(val_files)}")
    print(f"Random seed used: {RANDOM_SEED}")


if __name__ == "__main__":
    split_dataset()
